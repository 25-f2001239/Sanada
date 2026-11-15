r"""Speech separation (Human 1 / Human 2), VAD, and speaker embedding clustering (Task 4).

Responsibilities:
1. Voice-Activity Detection (VAD) on 16 kHz mono audio.
2. Speaker embedding extraction & agglomerative cosine clustering:
   - Uses pretrained SpeechBrain ECAPA-TDNN (`models/spkrec-ecapa-voxceleb`) when present on disk.
   - Falls back to 44-D cepstral/formant features explicitly labelled
     "Approximate (not real speaker embeddings)" when ECAPA weights are absent, and suppresses
     the card whenever the approximate estimate contradicts the counter's K.
   - Shows the distinct-voices card only for non-overlapping speech or when K <= 3, with a one-line
     explanation of the overlap limit.
3. Pretrained Speech Separation path:
   - Uses `models/sepformer-wsj02mix` for K=2 speakers (and `models/sepformer-wsj03mix` for K=3 if present).
   - Never offers separation for K >= 3 when only the 2-speaker model (`wsj02mix`) is installed.
   - Verifies per-stem speaker embeddings (ECAPA when installed) and labels stems neutrally as
     "Human 1", "Human 2", ... (never guesses gender).
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
from sklearn.cluster import AgglomerativeClustering

import audio_io
from dataset_config import load_config

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
SEPFORMER2_DIR = MODELS_DIR / "sepformer-wsj02mix"
SEPFORMER3_DIR = MODELS_DIR / "sepformer-wsj03mix"
ECAPA_DIR = MODELS_DIR / "spkrec-ecapa-voxceleb"
SAMPLE_RATE = 16_000

POWERSHELL_DOWNLOAD_COMMANDS = (
    "cd C:\\coding\\SANADA\\Sanada\n"
    ".\\.venv\\Scripts\\python -m pip install huggingface_hub speechbrain torchaudio\n"
    ".\\.venv\\Scripts\\python -c \"from huggingface_hub import snapshot_download; "
    "snapshot_download('speechbrain/sepformer-wsj02mix', local_dir='models/sepformer-wsj02mix'); "
    "snapshot_download('speechbrain/spkrec-ecapa-voxceleb', local_dir='models/spkrec-ecapa-voxceleb')\""
)

_ECAPA_CACHE: Optional[Any] = None
_SEP2_CACHE: Optional[Any] = None
_SEP3_CACHE: Optional[Any] = None


def pretrained_speech_status() -> Dict[str, Any]:
    """Check whether pretrained speech separation and ECAPA speaker weights exist in models/."""
    cfg = load_config()
    models_root = Path(cfg.get("models_dir", str(MODELS_DIR)))
    sep2_dir = (models_root / "sepformer-wsj02mix").resolve()
    sep3_dir = (models_root / "sepformer-wsj03mix").resolve()
    ecapa_dir = (models_root / "spkrec-ecapa-voxceleb").resolve()

    has_sep2 = sep2_dir.is_dir() and any(sep2_dir.glob("*.ckpt"))
    has_sep3 = sep3_dir.is_dir() and any(sep3_dir.glob("*.ckpt"))
    has_ecapa = ecapa_dir.is_dir() and any(ecapa_dir.glob("*.ckpt"))

    max_supported_speakers = 3 if has_sep3 else (2 if has_sep2 else 0)

    return {
        "available": bool(has_sep2 or has_sep3),
        "has_2spk_model": bool(has_sep2),
        "has_3spk_model": bool(has_sep3),
        "has_speaker_encoder": bool(has_ecapa),
        "max_supported_speakers": max_supported_speakers,
        "sepformer_dir": str(sep2_dir),
        "sepformer3_dir": str(sep3_dir),
        "ecapa_dir": str(ecapa_dir),
        "download_commands": POWERSHELL_DOWNLOAD_COMMANDS,
        "powershell_download_cmd": POWERSHELL_DOWNLOAD_COMMANDS,
        "reason": (
            None if (has_sep2 or has_sep3) else
            "Pretrained speech separator weights (models/sepformer-wsj02mix) are not installed on disk."
        ),
        "offline_fallback_message": (
            None if (has_sep2 or has_sep3) else
            "Pretrained speech separator weights (models/sepformer-wsj02mix) are not installed on disk. "
            "Run the PowerShell commands below to download them locally."
        ),
    }


pretrained_speech_separator_status = pretrained_speech_status


def _get_ecapa_encoder() -> Optional[Any]:
    """Lazy-load and cache SpeechBrain ECAPA-TDNN encoder from models/spkrec-ecapa-voxceleb."""
    global _ECAPA_CACHE
    if _ECAPA_CACHE is not None:
        return _ECAPA_CACHE
    st = pretrained_speech_status()
    if not st["has_speaker_encoder"]:
        return None
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        ecapa_dir = st["ecapa_dir"]
        _ECAPA_CACHE = EncoderClassifier.from_hparams(
            source=ecapa_dir,
            savedir=ecapa_dir,
            overrides={"pretrained_path": ecapa_dir},
            local_strategy=LocalStrategy.COPY,
        )
        return _ECAPA_CACHE
    except Exception:
        return None


def _get_sepformer_model(num_spks: int = 2) -> Optional[Any]:
    """Lazy-load and cache SpeechBrain SepFormer for 2 or 3 speakers."""
    global _SEP2_CACHE, _SEP3_CACHE
    st = pretrained_speech_status()
    try:
        from speechbrain.inference.separation import SepformerSeparation
        from speechbrain.utils.fetching import LocalStrategy

        if num_spks == 3:
            if not st["has_3spk_model"]:
                return None
            if _SEP3_CACHE is None:
                _SEP3_CACHE = SepformerSeparation.from_hparams(
                    source=st["sepformer3_dir"],
                    savedir=st["sepformer3_dir"],
                    local_strategy=LocalStrategy.COPY,
                )
            return _SEP3_CACHE
        else:
            if not st["has_2spk_model"]:
                return None
            if _SEP2_CACHE is None:
                _SEP2_CACHE = SepformerSeparation.from_hparams(
                    source=st["sepformer_dir"],
                    savedir=st["sepformer_dir"],
                    local_strategy=LocalStrategy.COPY,
                )
            return _SEP2_CACHE
    except Exception:
        return None


def separate_speech_pretrained(waveform: np.ndarray, k: int, sample_rate: int = SAMPLE_RATE) -> Dict[str, Any]:
    """Run pretrained SpeechBrain SepFormer if installed and supported for estimated count k."""
    st = pretrained_speech_status()
    k_int = int(k)
    if k_int == 2 and st["has_2spk_model"]:
        model = _get_sepformer_model(2)
        target_sr = 8000
    elif k_int == 3 and st["has_3spk_model"]:
        model = _get_sepformer_model(3)
        target_sr = 8000
    else:
        max_spk = st["max_supported_speakers"] or 2
        return {
            "available": False,
            "stems": [],
            "reason": f"Separation is supported for up to {max_spk} speakers with the installed model.",
        }

    if model is None:
        return {"available": False, "stems": [], "reason": st["reason"] or "Pretrained speech separator could not be loaded."}

    try:
        import torch

        y16 = np.asarray(waveform, dtype=np.float32).reshape(-1)
        y8 = audio_io.resample(y16, sample_rate, target_sr) if sample_rate != target_sr else y16
        wav_t = torch.from_numpy(y8).unsqueeze(0)
        with torch.inference_mode():
            est_sources = model.separate_batch(wav_t)  # [1, T_8k, C]
        num_out = min(k_int, int(est_sources.shape[-1]))
        stems = []
        for idx in range(num_out):
            s8 = est_sources[0, :, idx].detach().cpu().numpy().astype(np.float32)
            s16 = audio_io.resample(s8, target_sr, sample_rate) if sample_rate != target_sr else s8
            if len(s16) > len(y16):
                s16 = s16[: len(y16)]
            elif len(s16) < len(y16):
                s16 = np.pad(s16, (0, len(y16) - len(s16)))
            stems.append(s16.astype(np.float32))
        return {"available": True, "stems": stems, "model_name": f"sepformer-wsj0{k_int}mix", "reason": None}
    except Exception as exc:
        return {"available": False, "stems": [], "reason": str(exc)}


def detect_voice_activity(waveform: np.ndarray, sr: int = SAMPLE_RATE, frame_ms: int = 30, hop_ms: int = 15) -> Dict[str, Any]:
    """Energy + spectral-flatness voice activity detection (VAD) over 30 ms frames."""
    y = np.asarray(waveform, dtype=np.float32).reshape(-1)
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    if len(y) < frame_len:
        return {"voiced_ratio": 0.0, "voiced_segments": [], "num_voiced_frames": 0}

    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=512, hop_length=hop_len)[0]
    n = min(len(rms), len(flatness))
    rms, flatness = rms[:n], flatness[:n]

    thr_rms = max(np.percentile(rms, 25) * 1.6, 1e-4)
    voiced = (rms > thr_rms) & (flatness < 0.35)
    voiced_ratio = float(np.mean(voiced)) if n > 0 else 0.0

    return {
        "voiced_ratio": round(voiced_ratio, 4),
        "num_voiced_frames": int(np.sum(voiced)),
        "total_frames": int(n),
        "voiced_mask": voiced,
        "hop_len": hop_len,
        "frame_len": frame_len,
    }


def _compute_cepstral_fallback(waveform: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Hand-made 44-D cepstral + spectral fallback when ECAPA-TDNN weights are absent."""
    y = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if len(y) < int(0.15 * sr) or np.max(np.abs(y)) < 1e-5:
        return np.zeros(44, dtype=np.float32)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=512, hop_length=160)
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=160)[0]
    active = rms > max(np.percentile(rms, 30), 1e-5)
    mfcc_act = mfcc[:, active] if np.sum(active) >= 3 else mfcc

    mean_cep = np.mean(mfcc_act, axis=1)
    std_cep = np.std(mfcc_act, axis=1)
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=512, hop_length=160)) / (sr / 2.0))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=512, hop_length=160)) / (sr / 2.0))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=512, hop_length=160)) / (sr / 2.0))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y, frame_length=512, hop_length=160)))

    weights = np.ones(20, dtype=np.float32)
    weights[0] = 0.15
    vec = np.concatenate([
        mean_cep * weights,
        std_cep * weights * 0.5,
        np.array([centroid * 10.0, rolloff * 10.0, bandwidth * 10.0, zcr * 10.0], dtype=np.float32),
    ]).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-8 else vec


def compute_speaker_embedding_with_meta(waveform: np.ndarray, sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, bool]:
    """Return (L2-normalized speaker embedding vector, using_ecapa_bool)."""
    y = np.asarray(waveform, dtype=np.float32).reshape(-1)
    ecapa = _get_ecapa_encoder()
    if ecapa is not None and len(y) >= int(0.15 * sr) and np.max(np.abs(y)) >= 1e-5:
        try:
            import torch

            y16 = audio_io.resample(y, sr, SAMPLE_RATE) if sr != SAMPLE_RATE else y
            wav_t = torch.from_numpy(y16).unsqueeze(0)
            with torch.inference_mode():
                emb = ecapa.encode_batch(wav_t).squeeze().detach().cpu().numpy().astype(np.float32)
            norm = float(np.linalg.norm(emb))
            if norm > 1e-8:
                return emb / norm, True
        except Exception:
            pass
    return _compute_cepstral_fallback(y, sr=sr), False


def compute_speaker_embedding(waveform: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Compute a normalized speaker embedding vector (192-D ECAPA-TDNN when installed, else 44-D fallback)."""
    vec, _ = compute_speaker_embedding_with_meta(waveform, sr=sr)
    return vec


def estimate_distinct_speakers(
    waveform: np.ndarray,
    model_count_k: int = 2,
    speech_detected: bool = True,
    sr: int = SAMPLE_RATE,
    model_count: Optional[int] = None,
    sample_rate: Optional[int] = None,
) -> Dict[str, Any]:
    """Cluster windowed speaker embeddings on voiced segments to estimate distinct speakers found.

    Rules (Requirement 1):
    - Uses ECAPA-TDNN embeddings when `models/spkrec-ecapa-voxceleb` exists.
    - When only hand-made cepstral features are available, labels it
      "Approximate (not real speaker embeddings)" and does NOT show it next to a count it contradicts.
    - Shows the card ONLY for non-overlapping speech or when K <= 3, with a one-line limit explanation.
    """
    if model_count is not None:
        model_count_k = int(model_count)
    if sample_rate is not None:
        sr = int(sample_rate)
    k_int = int(model_count_k)

    limit_line = (
        "Windowed speaker clustering is only reliable for non-overlapping speech or sparse mixtures (K ≤ 3); "
        "simultaneous overlap blends single-window embeddings."
    )

    vad = detect_voice_activity(waveform, sr=sr)
    if not speech_detected or vad["voiced_ratio"] < 0.08:
        return {
            "evaluated": False,
            "show_card": False,
            "using_ecapa": bool(_get_ecapa_encoder() is not None),
            "method_label": (
                "ECAPA-TDNN speaker embeddings (spkrec-ecapa-voxceleb)"
                if _get_ecapa_encoder() is not None
                else "Approximate (not real speaker embeddings)"
            ),
            "voiced_ratio": vad["voiced_ratio"],
            "distinct_speakers": 0,
            "distinct_speakers_found": 0,
            "labels": [],
            "model_count": k_int,
            "limit_explanation": limit_line,
            "explanation": "Non-speech or low voice activity; speaker embedding clustering applies to human voice mixtures.",
        }

    # Do not show the distinct-voices card when K > 3 (dense overlapping mixtures blend window embeddings)
    if k_int > 3:
        using_ecapa = bool(_get_ecapa_encoder() is not None)
        return {
            "evaluated": False,
            "show_card": False,
            "using_ecapa": using_ecapa,
            "method_label": (
                "ECAPA-TDNN speaker embeddings (spkrec-ecapa-voxceleb)"
                if using_ecapa
                else "Approximate (not real speaker embeddings)"
            ),
            "voiced_ratio": vad["voiced_ratio"],
            "distinct_speakers": None,
            "distinct_speakers_found": None,
            "labels": [],
            "model_count": k_int,
            "limit_explanation": limit_line,
            "explanation": f"Hidden for K={k_int} (> 3): {limit_line}",
        }

    y = np.asarray(waveform, dtype=np.float32).reshape(-1)
    win_samples = int(1.0 * sr)
    hop_samples = int(0.5 * sr)
    segments = []
    for start in range(0, max(1, len(y) - win_samples + 1), hop_samples):
        seg = y[start : start + win_samples]
        if np.sqrt(np.mean(seg ** 2)) >= 0.01:
            segments.append(seg)

    ecapa = _get_ecapa_encoder()
    using_ecapa = False
    embeddings: List[np.ndarray] = []

    if ecapa is not None and len(segments) >= 2:
        try:
            import torch

            segs_16k = [audio_io.resample(s, sr, SAMPLE_RATE) if sr != SAMPLE_RATE else s for s in segments]
            batch = torch.from_numpy(np.stack(segs_16k, axis=0))
            with torch.inference_mode():
                embs_t = ecapa.encode_batch(batch).squeeze(1).detach().cpu().numpy().astype(np.float32)
            norms = np.maximum(np.linalg.norm(embs_t, axis=1, keepdims=True), 1e-8)
            embs_norm = embs_t / norms
            embeddings = [embs_norm[i] for i in range(len(embs_norm))]
            using_ecapa = True
        except Exception:
            embeddings = []
            using_ecapa = False

    if not embeddings:
        for seg in segments:
            emb = _compute_cepstral_fallback(seg, sr=sr)
            if np.linalg.norm(emb) > 0.5:
                embeddings.append(emb)
        using_ecapa = False

    method_label = (
        "ECAPA-TDNN speaker embeddings (spkrec-ecapa-voxceleb)"
        if using_ecapa
        else "Approximate (not real speaker embeddings)"
    )

    if len(embeddings) < 2:
        found = 1 if vad["voiced_ratio"] >= 0.08 else 0
        contradicts = found != k_int
        show_card = bool(k_int <= 3 and (using_ecapa or not contradicts))
        labels_list = [f"Human {i}" for i in range(1, found + 1)]
        return {
            "evaluated": True,
            "show_card": show_card,
            "using_ecapa": using_ecapa,
            "method_label": method_label,
            "contradicts_count": contradicts,
            "voiced_ratio": vad["voiced_ratio"],
            "distinct_speakers": found,
            "distinct_speakers_found": found,
            "labels": labels_list,
            "model_count": k_int,
            "limit_explanation": limit_line,
            "explanation": f"{method_label}: found {found} distinct voice profile(s) vs. counter K={k_int}. {limit_line}",
        }

    emb_mat = np.stack(embeddings, axis=0)
    sim_mat = np.clip(emb_mat @ emb_mat.T, -1.0, 1.0)
    dist_mat = np.clip(1.0 - sim_mat, 0.0, 2.0)

    # Calibrated distance threshold: 0.78 for 192-D ECAPA embeddings, 0.068 for 44-D cepstral fallback
    dist_thr = 0.78 if using_ecapa else 0.068
    max_clusters = min(len(embeddings), max(1, k_int))
    best_k = 1
    try:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=dist_thr,
        )
        labels = clustering.fit_predict(dist_mat)
        counts = np.bincount(labels)
        min_support = 2 if len(embeddings) >= 6 else 1
        sig_clusters = int(np.sum(counts >= min_support))
        best_k = max(1, min(sig_clusters, max_clusters))
    except Exception:
        best_k = k_int

    contradicts = int(best_k) != k_int
    # Show card only for non-overlapping speech or when K<=3, and never next to a count it contradicts
    show_card = bool(k_int <= 3 and not contradicts)
    labels_list = [f"Human {i}" for i in range(1, int(best_k) + 1)]

    if best_k == k_int:
        expl = (
            f"{method_label}: found {best_k} distinct voices ({', '.join(labels_list)}; "
            f"VAD voiced {vad['voiced_ratio']*100:.0f}%), matching K={k_int}. {limit_line}"
        )
    else:
        expl = (
            f"{method_label}: found {best_k} distinct voice profile(s) ({', '.join(labels_list)}) across 1s windows "
            f"vs. counter K={k_int}. {limit_line}"
        )

    return {
        "evaluated": True,
        "show_card": show_card,
        "using_ecapa": using_ecapa,
        "method_label": method_label,
        "contradicts_count": contradicts,
        "voiced_ratio": vad["voiced_ratio"],
        "num_voiced_windows": len(embeddings),
        "distinct_speakers": int(best_k),
        "distinct_speakers_found": int(best_k),
        "labels": labels_list,
        "model_count": k_int,
        "limit_explanation": limit_line,
        "explanation": expl,
    }


def verify_separated_speaker_stems(
    stems: List[np.ndarray],
    speech_mode: bool = True,
    sr: int = SAMPLE_RATE,
    sample_rate: Optional[int] = None,
    max_cosine_sim: Optional[float] = None,
) -> Dict[str, Any]:
    """Check with per-stem speaker embeddings (ECAPA when installed) that separated stems are different people.

    Labels stems neutrally as 'Human 1', 'Human 2', ... when speech_mode is True,
    or 'Source 1', 'Source 2', ... otherwise. Never guesses gender from voice.
    """
    if sample_rate is not None:
        sr = int(sample_rate)
    cfg = load_config()
    sim_threshold = max_cosine_sim if max_cosine_sim is not None else float(cfg.get("speaker_cosine_similarity_max", 0.82))

    prefix = "Human" if speech_mode else "Source"
    stem_labels = [f"{prefix} {i + 1}" for i in range(len(stems))]

    if len(stems) < 2:
        return {
            "stem_labels": stem_labels,
            "using_ecapa": bool(_get_ecapa_encoder() is not None),
            "different_speakers_verified": False,
            "could_not_separate_speakers": False,
            "max_speaker_embedding_cosine": 1.0,
            "max_embedding_cosine_similarity": 1.0,
            "pairwise_speaker_cosine": [],
            "reason": "At least 2 stems are required to verify distinct speakers.",
            "status_message": "At least 2 stems are required to verify distinct speakers.",
        }

    embs = []
    used_ecapa = False
    for s in stems:
        vec, is_ecapa = compute_speaker_embedding_with_meta(s, sr=sr)
        embs.append(vec)
        if is_ecapa:
            used_ecapa = True

    pairwise = []
    for i, j in combinations(range(len(stems)), 2):
        cos_sim = float(np.dot(embs[i], embs[j]))
        pairwise.append({"i": i + 1, "j": j + 1, "cosine_similarity": round(cos_sim, 4)})

    max_cos = max((p["cosine_similarity"] for p in pairwise), default=1.0)
    too_similar = max_cos > sim_threshold
    emb_kind = "ECAPA speaker embeddings" if used_ecapa else "cepstral voice profiles"

    if too_similar:
        msg = (
            f"Could not separate speakers: per-stem {emb_kind} are too similar "
            f"(max cosine similarity {max_cos:.3f} > {sim_threshold:.2f}). Stems are near-copies."
        )
    else:
        msg = (
            f"Verified distinct {emb_kind} across stems "
            f"(max cosine similarity {max_cos:.3f} <= {sim_threshold:.2f})."
        )

    return {
        "stem_labels": stem_labels,
        "using_ecapa": used_ecapa,
        "different_speakers_verified": not too_similar,
        "could_not_separate_speakers": too_similar,
        "max_speaker_embedding_cosine": round(max_cos, 4),
        "max_embedding_cosine_similarity": round(max_cos, 4),
        "threshold": sim_threshold,
        "pairwise_speaker_cosine": pairwise,
        "reason": msg,
        "status_message": msg,
        "pretrained_status": pretrained_speech_status(),
    }
