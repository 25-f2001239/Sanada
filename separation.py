"""Source separation: IterativeMaskSeparator, SepFormer + ECAPA speech path, and query-conditioned separation."""

from __future__ import annotations

import os
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from torch import nn

from audio import file_sha256, resample
from config import (
    DEVICE,
    HOP_LENGTH,
    MODELS_DIR,
    N_FFT,
    SAMPLE_RATE,
    SEPARATOR_PATH,
    env_float,
    env_int,
    get_group_for_class,
    load_config,
)

SEPARATOR_ARCHITECTURE = "IterativeMaskSeparator"
SEPARATION_SAMPLE_RATE = SAMPLE_RATE
MAX_PASSES = 32
MAX_SEPARABLE = env_int("MAX_SEPARABLE", 8)
RESIDUAL_THRESHOLD = env_float("RESIDUAL_THRESHOLD", 0.01)
WINDOW_SECONDS = 8.0
OVERLAP_SECONDS = 2.0

POWERSHELL_DOWNLOAD_COMMANDS = (
    "cd C:\\coding\\SANADA\\Sanada\n"
    ".\\.venv\\Scripts\\python -m pip install huggingface_hub speechbrain torchaudio\n"
    ".\\.venv\\Scripts\\python scripts\\download_models.py"
)

_ECAPA_CACHE: Any | None = None
_SEP2_CACHE: Any | None = None
_SEP3_CACHE: Any | None = None
_SEP4_CACHE: Any | None = None

# Acoustic priors: (f_lo, f_hi, harmonic_w, percussive_w, center_hz)
QUERY_ACOUSTIC_PRIORS: dict[str, tuple[int, int, float, float, int]] = {
    "Hammer": (100, 4800, 0.15, 0.85, 1200),
    "Phone vibration": (70, 550, 0.80, 0.20, 180),
    "Dog bark": (250, 3800, 0.55, 0.45, 950),
    "Cat meow": (380, 5200, 0.88, 0.12, 1400),
    "Cello": (65, 2400, 0.92, 0.08, 350),
    "Violin": (190, 6500, 0.92, 0.08, 1100),
    "Acoustic guitar": (80, 4200, 0.78, 0.22, 550),
    "Electric piano": (65, 4500, 0.85, 0.15, 600),
    "Clarinet": (140, 4800, 0.90, 0.10, 850),
    "Flute": (250, 6000, 0.92, 0.08, 1300),
    "Oboe": (220, 5500, 0.90, 0.10, 1050),
    "Saxophone": (130, 4800, 0.88, 0.12, 780),
    "Trumpet": (160, 5500, 0.88, 0.12, 1150),
    "Harmonica": (200, 5200, 0.88, 0.12, 1250),
    "Glockenspiel": (700, 7800, 0.70, 0.30, 3200),
    "Gong": (60, 3500, 0.65, 0.35, 450),
    "Chime": (500, 7500, 0.75, 0.25, 2600),
    "Snare drum": (100, 6500, 0.10, 0.90, 1600),
    "Scissors": (1600, 7900, 0.12, 0.88, 4400),
    "Telephone ring": (550, 5400, 0.85, 0.15, 1800),
    "Human voice": (85, 3800, 0.80, 0.20, 520),
    "Laughter": (150, 4200, 0.65, 0.35, 800),
    "Cough": (120, 4500, 0.25, 0.75, 900),
    "Applause": (300, 7200, 0.10, 0.90, 2400),
    "Door knock": (65, 1800, 0.12, 0.88, 320),
    "Keyboard typing": (700, 7200, 0.12, 0.88, 2900),
    "Keys jangling": (1800, 7900, 0.25, 0.75, 4800),
    "Drawer": (70, 2200, 0.20, 0.80, 420),
    "Microwave oven": (60, 2800, 0.82, 0.18, 300),
    "Glass shatter": (900, 7900, 0.15, 0.85, 3800),
    "Squeak": (900, 7500, 0.85, 0.15, 2800),
    "Paper tearing": (500, 7200, 0.18, 0.82, 2500),
    "Writing": (400, 6500, 0.18, 0.82, 2100),
    "Water": (280, 7600, 0.30, 0.70, 2200),
    "Car horn": (320, 4400, 0.90, 0.10, 950),
    "Bus": (50, 1800, 0.65, 0.35, 220),
    "Fireworks": (80, 7000, 0.12, 0.88, 1500),
    "Gunshot": (60, 7500, 0.08, 0.92, 1400),
}


class _Block(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class IterativeMaskSeparator(nn.Module):
    """Predict one residual TF mask per pass from (mixture, residual, pass_index, K)."""

    def __init__(self, channels: int = 32, blocks: int = 6):
        super().__init__()
        self.channels, self.blocks = int(channels), int(blocks)
        self.stem = nn.Sequential(
            nn.Conv2d(4, self.channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            *[_Block(self.channels, 2 ** (i % 4)) for i in range(self.blocks)]
        )
        self.head = nn.Conv2d(self.channels, 1, 1)

    def forward(
        self, mix_logmag: torch.Tensor, res_logmag: torch.Tensor, pass_index: int, k: int
    ) -> torch.Tensor:
        stacked = torch.stack(
            [
                mix_logmag,
                res_logmag,
                torch.full_like(mix_logmag, pass_index / MAX_PASSES),
                torch.full_like(mix_logmag, k / MAX_PASSES),
            ],
            dim=1,
        )
        return torch.sigmoid(self.head(self.trunk(self.stem(stacked))))[:, 0]


def _read_checkpoint(path: Path | str):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or ckpt.get("architecture") != SEPARATOR_ARCHITECTURE:
        raise ValueError(f"Checkpoint is not an {SEPARATOR_ARCHITECTURE} checkpoint.")
    for name, val in (
        ("sample_rate", SEPARATION_SAMPLE_RATE),
        ("n_fft", N_FFT),
        ("hop_length", HOP_LENGTH),
    ):
        if ckpt.get(name) != val:
            raise ValueError(
                f"Checkpoint setting {name}={ckpt.get(name)!r} does not match {val!r}."
            )
    cfg = ckpt.get("config") or {}
    if (
        not isinstance(ckpt.get("model_state_dict"), dict)
        or "channels" not in cfg
        or "blocks" not in cfg
    ):
        raise ValueError("Checkpoint is missing model_state_dict or config.")
    return ckpt, cfg


@lru_cache(maxsize=1)
def _load_cached(path: str, _mtime_ns: int, _size: int):
    ckpt, cfg = _read_checkpoint(path)
    model = IterativeMaskSeparator(cfg["channels"], cfg["blocks"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_separator(path: Path | str = SEPARATOR_PATH) -> IterativeMaskSeparator:
    """Load cached IterativeMaskSeparator model."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No separation checkpoint at {p}.")
    st = p.stat()
    return _load_cached(str(p), st.st_mtime_ns, st.st_size)


def separator_version(path: Path | str = SEPARATOR_PATH) -> str:
    """Return version+sha256 label for the IterativeMaskSeparator checkpoint."""
    ckpt, _ = _read_checkpoint(path)
    return f"{ckpt.get('version') or SEPARATOR_ARCHITECTURE}+{file_sha256(path)[:12]}"


def separator_status(path: Path | str = SEPARATOR_PATH) -> dict[str, Any]:
    """Return status and trained flag for IterativeMaskSeparator."""
    p = Path(path)
    base = {"max_separable": MAX_SEPARABLE, "sample_rate": SEPARATION_SAMPLE_RATE}
    if not p.is_file():
        return {
            **base,
            "available": False,
            "trained": False,
            "reason": f"No separation checkpoint found at {p.name}.",
        }
    try:
        ckpt, _ = _read_checkpoint(p)
        trained = bool(ckpt.get("trained", False)) and ckpt.get("version") != "UNTRAINED-INIT"
        load_separator(p)
        return {
            **base,
            "available": True,
            "trained": trained,
            "reason": None,
            "version": separator_version(p),
        }
    except Exception as exc:
        return {
            **base,
            "available": False,
            "trained": False,
            "reason": f"The separation checkpoint could not be loaded: {exc}",
        }


def pretrained_speech_status() -> dict[str, Any]:
    """Check whether pretrained SpeechBrain SepFormer (2/3/4-spk) and ECAPA weights exist in models/."""
    cfg_dir = Path(load_config().get("models_dir", str(MODELS_DIR)))
    models_root = cfg_dir if cfg_dir.is_dir() else MODELS_DIR
    sep2_dir, sep3_dir, sep4_dir, ecapa_dir = (
        (models_root / "sepformer-wsj02mix").resolve(),
        (models_root / "sepformer-wsj03mix").resolve(),
        (models_root / "sepformer-libri4mix").resolve(),
        (models_root / "spkrec-ecapa-voxceleb").resolve(),
    )
    has_sep2 = sep2_dir.is_dir() and any(sep2_dir.glob("*.ckpt"))
    has_sep3 = sep3_dir.is_dir() and any(sep3_dir.glob("*.ckpt"))
    has_sep4 = sep4_dir.is_dir() and (sep4_dir / "masknet.ckpt").is_file()
    has_ecapa = ecapa_dir.is_dir() and any(ecapa_dir.glob("*.ckpt"))
    any_sep = bool(has_sep2 or has_sep3 or has_sep4)
    max_spks = 4 if has_sep4 else (3 if has_sep3 else (2 if has_sep2 else 0))
    return {
        "available": any_sep,
        "has_2spk_model": bool(has_sep2),
        "has_3spk_model": bool(has_sep3),
        "has_4spk_model": bool(has_sep4),
        "has_speaker_encoder": bool(has_ecapa),
        "max_supported_speakers": max_spks,
        "sepformer_dir": str(sep2_dir),
        "sepformer3_dir": str(sep3_dir),
        "sepformer4_dir": str(sep4_dir),
        "ecapa_dir": str(ecapa_dir),
        "download_commands": POWERSHELL_DOWNLOAD_COMMANDS,
        "powershell_download_cmd": POWERSHELL_DOWNLOAD_COMMANDS,
        "reason": None
        if any_sep
        else "Pretrained speech separator weights (models/sepformer-wsj02mix) are not installed on disk.",
        "offline_fallback_message": None
        if any_sep
        else "Pretrained speech separator weights (models/sepformer-wsj02mix) are not installed on disk.",
    }


pretrained_speech_separator_status = pretrained_speech_status


def _get_ecapa_encoder() -> Any | None:
    global _ECAPA_CACHE
    if _ECAPA_CACHE is not None:
        return _ECAPA_CACHE
    st = pretrained_speech_status()
    if not st["has_speaker_encoder"]:
        return None
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        _ECAPA_CACHE = EncoderClassifier.from_hparams(
            source=st["ecapa_dir"],
            savedir=st["ecapa_dir"],
            overrides={"pretrained_path": st["ecapa_dir"]},
            local_strategy=LocalStrategy.COPY,
        )
        return _ECAPA_CACHE
    except Exception:
        return None


def _load_sepformer_4spk(model_dir: str) -> Any:
    """Load hahmadraz/sepformer-libri4mix directly from its checkpoint files for SpeechBrain 1.x compatibility."""
    from speechbrain.inference.separation import SepformerSeparation
    from speechbrain.lobes.models.dual_path import (
        Decoder,
        Dual_Path_Model,
        Encoder,
        SBTransformerBlock,
    )

    d = Path(model_dir)
    enc = Encoder(kernel_size=16, out_channels=256)
    intra = SBTransformerBlock(
        num_layers=8,
        d_model=256,
        nhead=8,
        d_ffn=1024,
        dropout=0,
        use_positional_encoding=True,
        norm_before=True,
    )
    inter = SBTransformerBlock(
        num_layers=8,
        d_model=256,
        nhead=8,
        d_ffn=1024,
        dropout=0,
        use_positional_encoding=True,
        norm_before=True,
    )
    masknet = Dual_Path_Model(
        num_spks=4,
        in_channels=256,
        out_channels=256,
        num_layers=2,
        K=250,
        intra_model=intra,
        inter_model=inter,
        norm="ln",
        linear_layer_after_inter_intra=False,
        skip_around_intra=True,
    )
    dec = Decoder(in_channels=256, out_channels=1, kernel_size=16, stride=8, bias=False)
    enc.load_state_dict(torch.load(d / "encoder.ckpt", map_location="cpu", weights_only=True))
    masknet.load_state_dict(torch.load(d / "masknet.ckpt", map_location="cpu", weights_only=True))
    dec.load_state_dict(torch.load(d / "decoder.ckpt", map_location="cpu", weights_only=True))
    model = SepformerSeparation(
        modules={"encoder": enc, "masknet": masknet, "decoder": dec},
        hparams={"sample_rate": 48000, "num_spks": 4},
        run_opts={"device": str(DEVICE)},
    )
    model.eval()
    return model


def _get_sepformer_model(num_spks: int = 2) -> Any | None:
    global _SEP2_CACHE, _SEP3_CACHE, _SEP4_CACHE
    st = pretrained_speech_status()
    try:
        from speechbrain.inference.separation import SepformerSeparation
        from speechbrain.utils.fetching import LocalStrategy

        if num_spks == 4 and st["has_4spk_model"]:
            _SEP4_CACHE = _SEP4_CACHE or _load_sepformer_4spk(st["sepformer4_dir"])
            return _SEP4_CACHE
        if num_spks == 3 and st["has_3spk_model"]:
            _SEP3_CACHE = _SEP3_CACHE or SepformerSeparation.from_hparams(
                source=st["sepformer3_dir"],
                savedir=st["sepformer3_dir"],
                local_strategy=LocalStrategy.COPY,
            )
            return _SEP3_CACHE
        if num_spks == 2 and st["has_2spk_model"]:
            _SEP2_CACHE = _SEP2_CACHE or SepformerSeparation.from_hparams(
                source=st["sepformer_dir"],
                savedir=st["sepformer_dir"],
                local_strategy=LocalStrategy.COPY,
            )
            return _SEP2_CACHE
    except Exception:
        return None
    return None


def separate_speech_pretrained(
    waveform: np.ndarray, k: int, sample_rate: int = SAMPLE_RATE
) -> dict[str, Any]:
    """Run pretrained SpeechBrain SepFormer for K=2, K=3 (wsj03mix), or K=4 (libri4mix)."""
    st, k_int = pretrained_speech_status(), int(k)
    model = _get_sepformer_model(k_int)
    if model is None:
        return {
            "available": False,
            "stems": [],
            "reason": f"Separation is supported for up to {st['max_supported_speakers'] or 2} speakers with the installed model.",
        }
    try:
        target_sr = int(
            getattr(getattr(model, "hparams", None), "sample_rate", None)
            or (48000 if k_int == 4 else 8000)
        )
        y_in = np.asarray(waveform, dtype=np.float32).reshape(-1)
        y_model = resample(y_in, sample_rate, target_sr) if sample_rate != target_sr else y_in
        inp = torch.from_numpy(y_model).unsqueeze(0).to(DEVICE)
        amp_dtype = torch.bfloat16 if DEVICE.type == "cpu" else torch.float16
        with torch.inference_mode():
            try:
                with torch.amp.autocast(device_type=DEVICE.type, dtype=amp_dtype):
                    est = model.separate_batch(inp)
            except Exception:
                est = model.separate_batch(inp)
        est = est.float()
        stems = []
        for idx in range(min(k_int, int(est.shape[-1]))):
            s_out = resample(
                est[0, :, idx].detach().cpu().numpy().astype(np.float32), target_sr, sample_rate
            )
            s_out = (
                s_out[: len(y_in)]
                if len(s_out) >= len(y_in)
                else np.pad(s_out, (0, len(y_in) - len(s_out)))
            )
            stems.append(s_out.astype(np.float32))
        model_name = "sepformer-libri4mix" if k_int == 4 else f"sepformer-wsj0{k_int}mix"
        return {
            "available": True,
            "stems": stems,
            "model_name": model_name,
            "reason": None,
        }
    except Exception as exc:
        return {"available": False, "stems": [], "reason": str(exc)}


def detect_voice_activity(waveform: np.ndarray, sr: int = SAMPLE_RATE) -> dict[str, Any]:
    """Energy + spectral-flatness voice activity detection over 30 ms frames."""
    y = np.asarray(waveform, dtype=np.float32).reshape(-1)
    frame_len, hop_len = int(sr * 0.03), int(sr * 0.015)
    if len(y) < frame_len:
        return {"voiced_ratio": 0.0}
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    flat = librosa.feature.spectral_flatness(y=y, n_fft=512, hop_length=hop_len)[0]
    n = min(len(rms), len(flat))
    voiced = (rms[:n] > max(np.percentile(rms[:n], 25) * 1.6, 1e-4)) & (flat[:n] < 0.35)
    return {"voiced_ratio": round(float(np.mean(voiced)) if n else 0.0, 4)}


def _cepstral_fallback(waveform: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    y = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if len(y) < int(0.15 * sr) or np.max(np.abs(y)) < 1e-5:
        return np.zeros(44, dtype=np.float32)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=512, hop_length=160)
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=160)[0]
    act = rms > max(np.percentile(rms, 30), 1e-5)
    m_act = mfcc[:, act] if np.sum(act) >= 3 else mfcc
    feats = [
        float(np.mean(fn(y=y, sr=sr, n_fft=512, hop_length=160)) / (sr / 2.0))
        for fn in (
            librosa.feature.spectral_centroid,
            librosa.feature.spectral_rolloff,
            librosa.feature.spectral_bandwidth,
        )
    ]
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y, frame_length=512, hop_length=160)))
    w = np.ones(20, dtype=np.float32)
    w[0] = 0.15
    vec = np.concatenate(
        [
            np.mean(m_act, axis=1) * w,
            np.std(m_act, axis=1) * w * 0.5,
            np.array([*feats, zcr], dtype=np.float32) * 10.0,
        ]
    ).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-8 else vec


def compute_speaker_embedding_with_meta(
    waveform: np.ndarray, sr: int = SAMPLE_RATE
) -> tuple[np.ndarray, bool]:
    """Return L2-normalized speaker embedding (192-D ECAPA when installed, else 44-D fallback)."""
    y = np.asarray(waveform, dtype=np.float32).reshape(-1)
    ecapa = _get_ecapa_encoder()
    if ecapa is not None and len(y) >= int(0.15 * sr) and np.max(np.abs(y)) >= 1e-5:
        try:
            y16 = resample(y, sr, SAMPLE_RATE) if sr != SAMPLE_RATE else y
            with torch.inference_mode():
                emb = (
                    ecapa.encode_batch(torch.from_numpy(y16).unsqueeze(0))
                    .squeeze()
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            norm = float(np.linalg.norm(emb))
            if norm > 1e-8:
                return emb / norm, True
        except Exception:
            pass
    return _cepstral_fallback(y, sr=sr), False


def estimate_distinct_speakers(
    waveform: np.ndarray,
    model_count_k: int = 2,
    speech_detected: bool = True,
    sr: int = SAMPLE_RATE,
    model_count: int | None = None,
    sample_rate: int | None = None,
) -> dict[str, Any]:
    """Cluster 1-second speaker embeddings to estimate distinct speakers for sparse speech (K <= 3)."""
    k_int = int(model_count if model_count is not None else model_count_k)
    sr = int(sample_rate if sample_rate is not None else sr)
    limit_line = "Windowed speaker clustering is only reliable for non-overlapping speech or sparse mixtures (K ≤ 3); simultaneous overlap blends single-window embeddings."
    vad = detect_voice_activity(waveform, sr=sr)
    has_ecapa = _get_ecapa_encoder() is not None
    method_lbl = (
        "ECAPA-TDNN speaker embeddings (spkrec-ecapa-voxceleb)"
        if has_ecapa
        else "Approximate (not real speaker embeddings)"
    )

    if not speech_detected or vad["voiced_ratio"] < 0.08 or k_int > 3:
        return {
            "evaluated": False,
            "show_card": False,
            "using_ecapa": has_ecapa,
            "method_label": method_lbl,
            "voiced_ratio": vad["voiced_ratio"],
            "distinct_speakers": 0 if k_int <= 3 else None,
            "distinct_speakers_found": 0 if k_int <= 3 else None,
            "labels": [],
            "model_count": k_int,
            "limit_explanation": limit_line,
        }

    y = np.asarray(waveform, dtype=np.float32).reshape(-1)
    win_s, hop_s = int(1.0 * sr), int(0.5 * sr)
    segs = [
        y[s : s + win_s]
        for s in range(0, max(1, len(y) - win_s + 1), hop_s)
        if np.sqrt(np.mean(y[s : s + win_s] ** 2)) >= 0.01
    ]

    using_ecapa, embs = False, []
    if has_ecapa and len(segs) >= 2:
        try:
            segs16 = [resample(s, sr, SAMPLE_RATE) if sr != SAMPLE_RATE else s for s in segs]
            with torch.inference_mode():
                et = (
                    _get_ecapa_encoder()
                    .encode_batch(torch.from_numpy(np.stack(segs16, axis=0)))
                    .squeeze(1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            embs = list(et / np.maximum(np.linalg.norm(et, axis=1, keepdims=True), 1e-8))
            using_ecapa = True
        except Exception:
            embs = []
    if not embs:
        embs = [e for e in (_cepstral_fallback(s, sr=sr) for s in segs) if np.linalg.norm(e) > 0.5]

    if len(embs) < 2:
        found = 1 if vad["voiced_ratio"] >= 0.08 else 0
        contradicts = found != k_int
        return {
            "evaluated": True,
            "show_card": bool(k_int <= 3 and (using_ecapa or not contradicts)),
            "using_ecapa": using_ecapa,
            "method_label": method_lbl,
            "contradicts_count": contradicts,
            "voiced_ratio": vad["voiced_ratio"],
            "distinct_speakers": found,
            "distinct_speakers_found": found,
            "labels": [f"Human {i}" for i in range(1, found + 1)],
            "model_count": k_int,
            "limit_explanation": limit_line,
        }

    emb_mat = np.stack(embs, axis=0)
    dist_mat = np.clip(1.0 - np.clip(emb_mat @ emb_mat.T, -1.0, 1.0), 0.0, 2.0)
    try:
        labels = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=0.78 if using_ecapa else 0.068,
        ).fit_predict(dist_mat)
        counts = np.bincount(labels)
        best_k = max(
            1,
            min(int(np.sum(counts >= (2 if len(embs) >= 6 else 1))), min(len(embs), max(1, k_int))),
        )
    except Exception:
        best_k = k_int

    contradicts = int(best_k) != k_int
    labels_list = [f"Human {i}" for i in range(1, int(best_k) + 1)]
    return {
        "evaluated": True,
        "show_card": bool(k_int <= 3 and not contradicts),
        "using_ecapa": using_ecapa,
        "method_label": method_lbl,
        "contradicts_count": contradicts,
        "voiced_ratio": vad["voiced_ratio"],
        "num_voiced_windows": len(embs),
        "distinct_speakers": int(best_k),
        "distinct_speakers_found": int(best_k),
        "labels": labels_list,
        "model_count": k_int,
        "limit_explanation": limit_line,
    }


def verify_separated_speaker_stems(
    stems: list[np.ndarray],
    speech_mode: bool = True,
    sr: int = SAMPLE_RATE,
    sample_rate: int | None = None,
    max_cosine_sim: float | None = None,
) -> dict[str, Any]:
    """Verify per-stem speaker embeddings are distinct and assign neutral Human 1 / Human 2 labels."""
    sr = int(sample_rate if sample_rate is not None else sr)
    thr = (
        max_cosine_sim
        if max_cosine_sim is not None
        else float(load_config().get("speaker_cosine_similarity_max", 0.82))
    )
    stem_labels = [f"{'Human' if speech_mode else 'Source'} {i + 1}" for i in range(len(stems))]
    if len(stems) < 2:
        return {
            "stem_labels": stem_labels,
            "using_ecapa": bool(_get_ecapa_encoder()),
            "different_speakers_verified": False,
            "could_not_separate_speakers": False,
            "max_speaker_embedding_cosine": 1.0,
            "max_embedding_cosine_similarity": 1.0,
            "pairwise_speaker_cosine": [],
            "reason": "At least 2 stems are required.",
        }

    embs, used_ecapa = [], False
    for s in stems:
        vec, is_ec = compute_speaker_embedding_with_meta(s, sr=sr)
        embs.append(vec)
        used_ecapa = used_ecapa or is_ec
    pairwise = [
        {"i": i + 1, "j": j + 1, "cosine_similarity": round(float(np.dot(embs[i], embs[j])), 4)}
        for i, j in combinations(range(len(stems)), 2)
    ]
    max_cos = max((p["cosine_similarity"] for p in pairwise), default=1.0)
    too_sim = max_cos > thr
    kind = "ECAPA speaker embeddings" if used_ecapa else "cepstral voice profiles"
    msg = (
        f"Could not separate speakers: per-stem {kind} are too similar ({max_cos:.3f} > {thr:.2f})."
        if too_sim
        else f"Verified distinct {kind} across stems ({max_cos:.3f} <= {thr:.2f})."
    )
    return {
        "stem_labels": stem_labels,
        "using_ecapa": used_ecapa,
        "different_speakers_verified": not too_sim,
        "could_not_separate_speakers": too_sim,
        "max_speaker_embedding_cosine": round(max_cos, 4),
        "max_embedding_cosine_similarity": round(max_cos, 4),
        "threshold": thr,
        "pairwise_speaker_cosine": pairwise,
        "reason": msg,
        "status_message": msg,
        "pretrained_status": pretrained_speech_status(),
    }


def _query_prior(name: str, group: str, idx: int, total: int) -> tuple[int, int, float, float, int]:
    if name in QUERY_ACOUSTIC_PRIORS:
        return QUERY_ACOUSTIC_PRIORS[name]
    defaults = {
        "Musical instrument": (80, 4800, 0.88, 0.12, 650 + idx * 350),
        "Human voice": (90, 3800, 0.80, 0.20, 450 + idx * 250),
        "Animal": (250, 4500, 0.65, 0.35, 1100 + idx * 300),
        "Household/tools": (200, 6800, 0.25, 0.75, 1800 + idx * 500),
    }
    lo = 80 + int(idx * (3500 / max(1, total)))
    return defaults.get(
        group, (lo, min(7800, lo + 3200), 0.50, 0.50, (lo + min(7800, lo + 3200)) // 2)
    )


def _separate_window_by_queries(
    clip: np.ndarray,
    queries: list[dict[str, Any]],
    sr: int = SAMPLE_RATE,
    win_offset_s: float = 0.0,
) -> list[np.ndarray]:
    y = np.asarray(clip, dtype=np.float32).reshape(-1)
    n_samples, num_q = len(y), len(queries)
    if num_q == 0 or n_samples == 0:
        return []
    stft_c = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag = np.abs(stft_c) + 1e-8
    n_bins, n_frames = mag.shape
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    ftimes = win_offset_s + librosa.frames_to_time(
        np.arange(n_frames), sr=sr, hop_length=HOP_LENGTH
    )
    mag_h, mag_p = librosa.decompose.hpss(mag, kernel_size=17, power=2.0, margin=1.2)

    neural_masks = []
    if separator_status().get("trained"):
        try:
            model = load_separator()
            win_t = torch.hann_window(N_FFT, device=DEVICE)
            spec_t = torch.stft(
                torch.from_numpy(y).to(DEVICE), N_FFT, HOP_LENGTH, window=win_t, return_complex=True
            )
            res_t, mix_lm = spec_t.clone(), torch.log1p(spec_t.abs()).unsqueeze(0)
            with torch.inference_mode():
                for i in range(num_q):
                    m_t = model(mix_lm, torch.log1p(res_t.abs()).unsqueeze(0), i, num_q)[0]
                    res_t = res_t * (1.0 - m_t)
                    m_np = m_t.detach().cpu().numpy()
                    neural_masks.append(
                        m_np[:, :n_frames]
                        if m_np.shape[1] >= n_frames
                        else np.pad(m_np, ((0, 0), (0, n_frames - m_np.shape[1])))
                    )
        except Exception:
            neural_masks = []

    raw_weights = []
    for idx, q in enumerate(queries):
        f_lo, f_hi, w_h, w_p, center_hz = _query_prior(
            str(q.get("name") or f"Source {idx + 1}"), str(q.get("group") or ""), idx, num_q
        )
        freq_w = np.ones(n_bins, dtype=np.float32)
        freq_w[freqs < f_lo] = np.exp(
            -0.5 * ((f_lo - freqs[freqs < f_lo]) / max(40.0, f_lo * 0.25)) ** 2
        )
        freq_w[freqs > f_hi] = np.exp(
            -0.5 * ((freqs[freqs > f_hi] - f_hi) / max(150.0, f_hi * 0.22)) ** 2
        )
        freq_w = (
            freq_w
            * (
                1.0
                + 0.85
                * np.exp(
                    -0.5
                    * ((np.log2(np.maximum(freqs, 30.0)) - np.log2(max(center_hz, 50.0))) / 0.65)
                    ** 2
                )
            )
        )[:, None]

        tranges = q.get("time_ranges") or []
        if tranges:
            time_w = np.full(n_frames, 0.25, dtype=np.float32)
            for tr in tranges:
                time_w[
                    (ftimes >= float(tr.get("start", 0.0)) - 0.15)
                    & (ftimes <= float(tr.get("end", 999.0)) + 0.15)
                ] = 1.0
            k_han = np.hanning(9)
            time_w = np.convolve(time_w, k_han / k_han.sum(), mode="same")
        else:
            time_w = np.ones(n_frames, dtype=np.float32)

        hp = w_h * mag_h + w_p * mag_p
        fe = np.mean(hp * freq_w, axis=0, keepdims=True)
        temp_gate = np.clip(
            (fe - 0.3 * np.percentile(fe, 25 + (idx * 15) % 45)) / (np.max(fe) + 1e-8), 0.12, 1.0
        )
        wq = (hp**1.35) * freq_w * time_w[None, :] * temp_gate
        if idx < len(neural_masks):
            wq = 0.65 * wq + 0.35 * (mag**1.35) * neural_masks[idx]
        raw_weights.append(wq.astype(np.float32))

    stack_w = np.stack(raw_weights, axis=0)
    for idx in range(num_q):
        stack_w[idx] = (stack_w[idx] / (float(np.sqrt(np.mean(stack_w[idx] ** 2))) + 1e-8)) ** 3.8
    masks = stack_w / (
        np.sum(stack_w, axis=0, keepdims=True)
        + 0.08 * np.mean(stack_w, axis=0, keepdims=True)
        + 1e-12
    )
    # Light 3-frame temporal smoothing prevents metallic chatter while keeping stems distinct
    if n_frames >= 3:
        k_sm = np.array([0.2, 0.6, 0.2], dtype=np.float32)
        for idx in range(num_q):
            masks[idx] = np.apply_along_axis(lambda r: np.convolve(r, k_sm, mode="same"), 1, masks[idx])
        mask_sum = np.sum(masks, axis=0, keepdims=True)
        masks = np.where(mask_sum > 1.0, masks / (mask_sum + 1e-12), masks)
    return [
        librosa.istft(stft_c * masks[i], hop_length=HOP_LENGTH, length=n_samples).astype(np.float32)
        for i in range(num_q)
    ]


def separate_long_audio_windowed(
    waveform: np.ndarray,
    step_fn,
    sr: int = SAMPLE_RATE,
    window_s: float = WINDOW_SECONDS,
    overlap_s: float = OVERLAP_SECONDS,
    pad_tail: bool = True,
) -> list[np.ndarray]:
    """Process arbitrary-length audio in overlapping windows with raised-cosine cross-fade."""
    y = np.asarray(waveform, dtype=np.float32).reshape(-1)
    total_len, win_len, ov_len = len(y), int(window_s * sr), int(overlap_s * sr)
    if total_len <= win_len:
        return step_fn(y, 0.0)

    hop_len = max(1, win_len - ov_len)
    fade_in = 0.5 * (1.0 - np.cos(np.pi * np.arange(ov_len, dtype=np.float32) / max(1, ov_len)))
    fade_out = 1.0 - fade_in
    stems_acc, weight_acc = None, np.zeros(total_len, dtype=np.float32)

    for w_idx, st in enumerate(range(0, total_len, hop_len)):
        ed = min(total_len, st + win_len)
        seg = y[st:ed]
        if len(seg) < int(0.25 * sr):
            break
        chunk = (
            seg
            if (len(seg) == win_len or not pad_tail)
            else np.pad(seg, (0, win_len - len(seg)))
        )
        win_stems = step_fn(chunk, st / sr)
        if not win_stems:
            break
        if stems_acc is None:
            stems_acc = [np.zeros(total_len, dtype=np.float32) for _ in win_stems]
        w_env = np.ones(len(seg), dtype=np.float32)
        if w_idx > 0 and len(seg) >= ov_len:
            w_env[:ov_len] *= fade_in
        if ed < total_len and len(seg) >= ov_len:
            w_env[-ov_len:] *= fade_out
        for s_idx, ws in enumerate(win_stems):
            stems_acc[s_idx][st:ed] += ws[: len(seg)] * w_env
        weight_acc[st:ed] += w_env
        if ed >= total_len:
            break

    return (
        []
        if stems_acc is None
        else [(s / np.maximum(weight_acc, 1e-6)).astype(np.float32) for s in stems_acc]
    )


def separate_by_queries(
    waveform: np.ndarray,
    k: int,
    named_sounds: list[dict[str, Any]] | None = None,
    speech_mode: bool = False,
    sr: int = SAMPLE_RATE,
) -> dict[str, Any]:
    """Run query-conditioned separation or 2-speaker SepFormer, plus exact residual stem."""
    y, k_int, named = (
        np.asarray(waveform, dtype=np.float32).reshape(-1),
        int(k),
        list(named_sounds or []),
    )
    non_speech = [s for s in named if s.get("group") != "Human voice"]
    speech_st = pretrained_speech_status()
    max_spk = int(speech_st.get("max_supported_speakers") or 2)

    if speech_mode and not non_speech:
        has_k_model = bool(speech_st.get(f"has_{k_int}spk_model", False))
        if k_int < 2 or k_int > max_spk or not has_k_model:
            return {
                "available": False,
                "stems": [],
                "residual_stem": None,
                "queries": [],
                "used_pretrained_speech": False,
                "reason": (
                    f"Separation is supported for up to {max_spk} speakers."
                    if k_int > max_spk or k_int < 2
                    else f"Pretrained {k_int}-speaker separation model is not installed."
                ),
            }

        fast_env = os.environ.get("ECHOCOUNT_FAST_4SPK", "auto").strip().lower()
        want_fast_4spk = (
            DEVICE.type == "cpu"
            if fast_env in ("", "auto")
            else fast_env in ("1", "true", "yes")
        )
        fast_4spk = (
            k_int == 4
            and want_fast_4spk
            and speech_st.get("has_2spk_model")
            and speech_st.get("has_3spk_model")
        )

        def _lsq_proj(ref: np.ndarray, est: np.ndarray) -> np.ndarray:
            denom = float(np.dot(est, est)) + 1e-12
            gain = float(np.dot(ref, est)) / denom
            if gain <= 0.0:
                gain = float(np.dot(ref, -est)) / denom
                est = -est
            return (est * float(np.clip(gain, 0.05, 1.8))).astype(np.float32)

        def _sep_spk(seg: np.ndarray, _off: float) -> list[np.ndarray]:
            if fast_4spk:
                # Fast 8 kHz 2-step cascade: wsj02mix (1-vs-3) + wsj03mix (remaining 3 speakers)
                out2 = separate_speech_pretrained(seg, 2, sample_rate=sr)
                if not out2.get("available") or len(out2.get("stems", [])) < 2:
                    return []
                c0, c1 = _lsq_proj(seg, out2["stems"][0]), _lsq_proj(seg, out2["stems"][1])
                spk_score = lambda s: float(np.percentile(np.abs(s), 95)) / (float(np.sqrt(np.mean(s**2))) + 1e-8)
                s1 = c0 if spk_score(c0) >= spk_score(c1) else c1
                rem3 = (seg - s1).astype(np.float32)
                out3 = separate_speech_pretrained(rem3, 3, sample_rate=sr)
                if not out3.get("available") or len(out3.get("stems", [])) < 3:
                    return []
                return [s1] + [_lsq_proj(rem3, rs) for rs in out3["stems"][:3]]

            out = separate_speech_pretrained(seg, k_int, sample_rate=sr)
            if not out.get("available") or not out.get("stems"):
                return []
            return [
                (
                    rs
                    * (
                        np.clip(float(np.dot(seg, rs) / (np.dot(rs, rs) + 1e-12)), 0.1, 2.5)
                        if np.dot(seg, rs) > 0
                        else 0.5
                    )
                    / max(1, len(out["stems"]) * 0.75)
                ).astype(np.float32)
                for rs in out["stems"]
            ]

        win_s, ov_s = (
            (2.2, 0.2) if (k_int == 4 and not fast_4spk) else (WINDOW_SECONDS, OVERLAP_SECONDS)
        )
        stems = separate_long_audio_windowed(
            y, _sep_spk, sr=sr, window_s=win_s, overlap_s=ov_s, pad_tail=False
        )
        if not stems:
            return {
                "available": False,
                "stems": [],
                "residual_stem": None,
                "queries": [],
                "used_pretrained_speech": False,
                "reason": "Speech separation model failed.",
            }
        residual = (y - np.sum(np.stack(stems, axis=0), axis=0)).astype(np.float32)
        queries = [
            {
                "name": f"Human {i + 1}",
                "friendly_name": f"Human {i + 1}",
                "class": "Speech",
                "group": "Human voice",
            }
            for i in range(len(stems))
        ]
        model_name = (
            "sepformer-wsj02mix+wsj03mix-fast8k"
            if fast_4spk
            else ("sepformer-libri4mix" if k_int == 4 else f"sepformer-wsj0{k_int}mix")
        )
        return {
            "available": True,
            "stems": stems,
            "residual_stem": residual,
            "queries": queries,
            "used_pretrained_speech": True,
            "model_name": model_name,
            "stopped_early": False,
            "residual_energy_ratio": round(float(np.sum(residual**2) / (np.sum(y**2) + 1e-12)), 4),
            "reason": None,
        }

    if k_int > MAX_SEPARABLE:
        return {
            "available": False,
            "stems": [],
            "residual_stem": None,
            "queries": [],
            "used_pretrained_speech": False,
            "reason": "Too many overlapping sounds to separate reliably.",
        }

    queries, seen = [], set()
    for s in named:
        sname = str(s.get("name") or s.get("friendly_name") or s.get("class", "Sound"))
        if sname not in seen:
            seen.add(sname)
            queries.append(
                {
                    "name": sname,
                    "friendly_name": sname,
                    "class": s.get("class", sname),
                    "group": s.get("group", "Environment/other"),
                    "confidence": float(s.get("confidence", s.get("probability", 0.5))),
                    "time_ranges": s.get("time_ranges", []),
                }
            )

    if len(queries) < 2 and k_int >= 2 and separator_status().get("trained"):
        for fn in ("Human voice", "Musical instrument", "Household/tools", "Animal"):
            if len(queries) >= min(k_int, 4):
                break
            if fn not in seen:
                seen.add(fn)
                queries.append(
                    {
                        "name": fn,
                        "friendly_name": fn,
                        "class": fn,
                        "group": get_group_for_class(fn),
                        "confidence": 0.35,
                        "time_ranges": [],
                    }
                )

    if not queries:
        return {
            "available": False,
            "stems": [],
            "residual_stem": None,
            "queries": [],
            "used_pretrained_speech": False,
            "reason": "Separation not available for this recording.",
        }

    queries = queries[: min(len(queries), MAX_SEPARABLE)]
    stems = separate_long_audio_windowed(
        y,
        lambda seg, off: _separate_window_by_queries(seg, queries, sr=sr, win_offset_s=off),
        sr=sr,
    )
    if not stems:
        return {
            "available": False,
            "stems": [],
            "residual_stem": None,
            "queries": [],
            "used_pretrained_speech": False,
            "reason": "Separation not available for this recording.",
        }
    residual = (y - np.sum(np.stack(stems, axis=0), axis=0)).astype(np.float32)
    return {
        "available": True,
        "stems": stems,
        "residual_stem": residual,
        "queries": queries,
        "used_pretrained_speech": False,
        "model_name": "QueryConditionedWiener+IterativeMaskSeparator",
        "stopped_early": False,
        "residual_energy_ratio": round(float(np.sum(residual**2) / (np.sum(y**2) + 1e-12)), 4),
        "reason": None,
    }
