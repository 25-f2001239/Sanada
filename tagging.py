"""Domain 32-class hierarchical tagger and open-vocabulary AST-527 + CLAP tagger."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from torch import nn

from audio import resample, waveform_to_logmel
from config import (
    DEVICE,
    MODELS_DIR,
    SAMPLE_RATE,
    TAGGER_PATH,
    get_friendly_name,
    get_group_for_class,
    load_taxonomy,
)
from counting import conv_trunk

AST_DIR = MODELS_DIR / "ast-audioset"
CLAP_DIR = MODELS_DIR / "clap-htsat-unfused"
CLAP_SAMPLE_RATE = 48_000

_AST_CACHE: tuple[Any, Any, dict[int, str]] | None = None
_CLAP_CACHE: tuple[Any, Any, torch.Tensor, list[str]] | None = None

# Compact AudioSet-527 label -> (friendly_name, coarse_group) mapping
_AUDIOSET_GROUPS: dict[tuple[str, str], tuple[str, ...]] = {
    ("Human voice", "Human voice"): (
        "Speech",
        "Male speech, man speaking",
        "Female speech, woman speaking",
        "Child speech, kid speaking",
        "Conversation",
        "Narration, monologue",
        "Babbling",
        "Groan",
    ),
    ("Whispering", "Human voice"): ("Whispering",),
    ("Laughter", "Human voice"): ("Laughter", "Giggle", "Chuckle, chortle", "Snicker"),
    ("Crying", "Human voice"): ("Crying, sobbing",),
    ("Sigh", "Human voice"): ("Sigh",),
    ("Singing", "Human voice"): ("Singing",),
    ("Applause", "Human voice"): ("Applause",),
    ("Cough", "Human voice"): ("Cough", "Throat clearing"),
    ("Sneeze", "Human voice"): ("Sneeze",),
    ("Screaming", "Human voice"): ("Screaming",),
    ("Dog bark", "Animal"): ("Bark", "Dog", "Bow-wow", "Yip", "Growling", "Whimper (dog)"),
    ("Cat meow", "Animal"): ("Meow", "Cat", "Purr", "Caterwaul"),
    ("Bird song", "Animal"): ("Bird", "Bird vocalization, bird call, bird song", "Chirp, tweet"),
    ("Rooster crow", "Animal"): ("Rooster, cock-a-doodle-doo",),
    ("Cello", "Musical instrument"): ("Cello", "Bowed string instrument"),
    ("Violin", "Musical instrument"): ("Violin, fiddle",),
    ("Acoustic guitar", "Musical instrument"): (
        "Acoustic guitar",
        "Guitar",
        "Plucked string instrument",
    ),
    ("Electric piano", "Musical instrument"): ("Electric piano", "Piano", "Keyboard (musical)"),
    ("Clarinet", "Musical instrument"): ("Clarinet",),
    ("Flute", "Musical instrument"): ("Flute",),
    ("Oboe", "Musical instrument"): ("Oboe",),
    ("Saxophone", "Musical instrument"): ("Saxophone",),
    ("Trumpet", "Musical instrument"): ("Trumpet", "Brass instrument"),
    ("Harmonica", "Musical instrument"): ("Harmonica",),
    ("Glockenspiel", "Musical instrument"): ("Glockenspiel", "Marimba, xylophone"),
    ("Gong", "Musical instrument"): ("Gong",),
    ("Chime", "Musical instrument"): ("Chime", "Wind chime"),
    ("Snare drum", "Musical instrument"): ("Snare drum", "Drum", "Drum kit"),
    ("Hammer", "Household/tools"): ("Hammer", "Chop", "Whack, thwack"),
    ("Phone vibration", "Household/tools"): ("Vibration", "Buzz"),
    ("Scissors", "Household/tools"): ("Scissors",),
    ("Telephone ring", "Household/tools"): (
        "Telephone",
        "Telephone bell ringing",
        "Ringtone",
        "Telephone dialing, DTMF",
    ),
    ("Door knock", "Household/tools"): ("Knock", "Door", "Tap"),
    ("Keyboard typing", "Household/tools"): ("Computer keyboard", "Typing", "Typewriter"),
    ("Drawer", "Household/tools"): ("Drawer open or close", "Cupboard open or close"),
    ("Keys jangling", "Household/tools"): ("Keys jangling",),
    ("Microwave oven", "Household/tools"): ("Microwave oven",),
    ("Glass shatter", "Household/tools"): ("Shatter", "Glass"),
    ("Squeak", "Household/tools"): ("Squeak",),
    ("Paper tearing", "Household/tools"): ("Tearing",),
    ("Writing", "Household/tools"): ("Writing",),
    ("Alarm clock", "Household/tools"): ("Alarm clock",),
    ("Doorbell", "Household/tools"): ("Doorbell",),
    ("Water", "Environment/other"): (
        "Water",
        "Stream",
        "Pour",
        "Trickle, dribble",
        "Water tap, faucet",
        "Splash, splatter",
        "Rain",
    ),
    ("Car horn", "Environment/other"): (
        "Vehicle horn, car horn, honking",
        "Air horn, truck horn",
        "Toot",
    ),
    ("Bus", "Environment/other"): ("Bus",),
    ("Fireworks", "Environment/other"): ("Fireworks", "Firecracker"),
    ("Gunshot", "Environment/other"): ("Gunshot, gunfire",),
    ("Siren", "Environment/other"): ("Siren",),
    ("Thunder", "Environment/other"): ("Thunder",),
}
AUDIOSET_TO_FRIENDLY: dict[str, tuple[str, str]] = {
    lbl: key for key, labels in _AUDIOSET_GROUPS.items() for lbl in labels
}


class TaggerCNN(nn.Module):
    """Legacy 32-class fine-only tagger architecture."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = conv_trunk()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class GroupFineTaggerCNN(nn.Module):
    """Dual-head coarse-group and fine-class tagger architecture."""

    def __init__(self, num_groups: int, num_fine: int):
        super().__init__()
        self.features = conv_trunk()
        self.shared_fc = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 16, 256), nn.ReLU(inplace=True), nn.Dropout(0.4)
        )
        self.group_head = nn.Linear(256, int(num_groups))
        self.fine_head = nn.Linear(256, int(num_fine))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.shared_fc(self.features(x))
        return self.group_head(emb), self.fine_head(emb)


@lru_cache(maxsize=2)
def _load_tagger_cached(path: str, _mtime_ns: int, _size: int):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or not ckpt.get("trained", False):
        raise ValueError("Checkpoint missing or trained=false.")
    arch = ckpt.get("architecture", "")
    if arch == "GroupFineTaggerCNN":
        model = GroupFineTaggerCNN(len(ckpt["groups"]), len(ckpt["vocabulary"])).to(DEVICE)
    elif arch == "TaggerCNN":
        model = TaggerCNN(len(ckpt["vocabulary"])).to(DEVICE)
    else:
        raise ValueError(f"Unsupported tagger architecture: {arch}")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_tagger(path: Path = TAGGER_PATH):
    """Load cached domain tagger checkpoint."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"No tagger checkpoint at {path}.")
    st = Path(path).stat()
    return _load_tagger_cached(str(path), st.st_mtime_ns, st.st_size)


def tagger_status(path: Path = TAGGER_PATH) -> dict[str, Any]:
    """Return status and validation metrics for the domain 32-class tagger."""
    path = Path(path)
    if not path.is_file():
        return {
            "available": False,
            "trained": False,
            "num_classes": 0,
            "groups": [],
            "vocabulary": [],
            "error": f"No checkpoint at {path.name}.",
        }
    try:
        _, ckpt = load_tagger(path)
        return {
            "available": True,
            "trained": True,
            "hierarchical": ckpt.get("architecture") == "GroupFineTaggerCNN",
            "architecture": ckpt.get("architecture"),
            "version": ckpt.get("version", "v1"),
            "num_groups": len(ckpt.get("groups", [])),
            "groups": ckpt.get("groups", []),
            "num_classes": len(ckpt.get("vocabulary", [])),
            "vocabulary": ckpt.get("vocabulary", []),
            "val_group_map": ckpt.get("val_group_macro_map"),
            "val_group_macro_map": ckpt.get("val_group_macro_map"),
            "test_group_macro_map": ckpt.get("test_group_macro_map"),
            "val_macro_map": ckpt.get("val_macro_map"),
            "test_macro_map": ckpt.get("test_macro_map"),
            "val_macro_f1": ckpt.get("val_macro_f1"),
            "test_macro_f1": ckpt.get("test_macro_f1"),
            "epoch": ckpt.get("epoch"),
            "error": None,
        }
    except Exception as exc:
        return {
            "available": False,
            "trained": False,
            "num_classes": 0,
            "groups": [],
            "vocabulary": [],
            "error": str(exc),
        }


def open_tagger_status() -> dict[str, Any]:
    """Return readiness and license metadata for AST-AudioSet, CLAP, and domain tagger."""
    has_ast = (AST_DIR / "model.safetensors").is_file() or (AST_DIR / "pytorch_model.bin").is_file()
    has_clap = (CLAP_DIR / "pytorch_model.bin").is_file() or (
        CLAP_DIR / "model.safetensors"
    ).is_file()
    dom_st = tagger_status()
    target_12 = list(load_taxonomy().get("target_12_sounds", {}).keys())
    return {
        "available": bool(has_ast or dom_st.get("available", False)),
        "ast_audioset_available": has_ast,
        "ast_model_id": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "ast_license": "BSD-3-Clause (commercial use allowed)",
        "ast_num_classes": 527,
        "clap_available": has_clap,
        "clap_model_id": "laion/clap-htsat-unfused",
        "clap_license": "Apache-2.0 / MIT (commercial use allowed)",
        "domain_32_tagger_available": bool(dom_st.get("available", False)),
        "target_12_sounds": target_12,
        "target_12_coverage": "12/12 (100%)",
    }


def tag_audio_hierarchical(
    waveform: np.ndarray,
    path: Path = TAGGER_PATH,
    precomputed_logmel: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Run hierarchical coarse-group and fine-class inference on an 8-second clip."""
    try:
        model, ckpt = load_tagger(path)
    except Exception as exc:
        return {"available": False, "reason": str(exc), "groups": [], "tags": []}

    feat = (
        precomputed_logmel if precomputed_logmel is not None else waveform_to_logmel(waveform)
    ).to(DEVICE)
    if feat.dim() == 3:
        feat = feat.unsqueeze(0)
    tax = load_taxonomy()

    if ckpt.get("architecture") == "GroupFineTaggerCNN":
        with (
            torch.inference_mode(),
            torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")),
        ):
            g_log, f_log = model(feat)
            g_probs = torch.sigmoid(g_log[0].float()).cpu().numpy()
            f_probs = torch.sigmoid(f_log[0].float()).cpu().numpy()

        group_results = sorted(
            [
                {
                    "group": g,
                    "probability": float(p),
                    "threshold": float(t),
                    "above_threshold": bool(p >= t),
                }
                for g, p, t in zip(ckpt["groups"], g_probs, ckpt["group_thresholds"])
            ],
            key=lambda x: -x["probability"],
        )
        active = {g["group"] for g in group_results if g["above_threshold"]}
        allowed = active or ({group_results[0]["group"]} if group_results else set())
        fine_results = sorted(
            [
                {
                    "class": c,
                    "friendly_name": get_friendly_name(c, tax),
                    "group": get_group_for_class(c, tax),
                    "probability": float(p),
                    "threshold": float(t),
                    "above_threshold": bool((p >= t) and (get_group_for_class(c, tax) in active)),
                }
                for c, p, t in zip(ckpt["vocabulary"], f_probs, ckpt["thresholds"])
                if get_group_for_class(c, tax) in allowed
            ],
            key=lambda x: -x["probability"],
        )
        return {
            "available": True,
            "hierarchical": True,
            "groups": group_results,
            "active_groups": [g["group"] for g in group_results if g["above_threshold"]],
            "tags": fine_results,
            "speech_detected": bool(
                group_results
                and group_results[0]["group"] == "Human voice"
                and group_results[0]["above_threshold"]
            ),
            "caveat": "Tags describe the whole mix (coarse groups first; fine classes filtered by active group).",
        }

    with torch.inference_mode():
        f_probs = torch.sigmoid(model(feat)[0].cpu()).numpy()
    grp_max = {g: 0.0 for g in tax["groups"]}
    fine_results = []
    for cls, fp, ft in zip(ckpt["vocabulary"], f_probs, ckpt["thresholds"]):
        g = get_group_for_class(cls, tax)
        grp_max[g] = max(grp_max.get(g, 0.0), float(fp))
        fine_results.append(
            {
                "class": cls,
                "group": g,
                "probability": float(fp),
                "threshold": float(ft),
                "above_threshold": bool(fp >= ft),
            }
        )
    fine_results.sort(key=lambda x: -x["probability"])
    group_results = [
        {"group": g, "probability": p, "threshold": 0.5, "above_threshold": p >= 0.5}
        for g, p in sorted(grp_max.items(), key=lambda x: -x[1])
    ]
    return {
        "available": True,
        "groups": group_results,
        "tags": fine_results,
        "speech_detected": False,
        "caveat": "Tags describe the whole mix.",
    }


def _get_ast():
    global _AST_CACHE
    if _AST_CACHE is not None:
        return _AST_CACHE
    if not ((AST_DIR / "model.safetensors").is_file() or (AST_DIR / "pytorch_model.bin").is_file()):
        return None
    try:
        from transformers import ASTForAudioClassification, AutoFeatureExtractor

        fe = AutoFeatureExtractor.from_pretrained(str(AST_DIR), local_files_only=True)
        model = ASTForAudioClassification.from_pretrained(
            str(AST_DIR), local_files_only=True
        ).eval()
        _AST_CACHE = (fe, model, {int(k): str(v) for k, v in model.config.id2label.items()})
        return _AST_CACHE
    except Exception:
        return None


def _get_clap():
    global _CLAP_CACHE
    if _CLAP_CACHE is not None:
        return _CLAP_CACHE
    if not (
        (CLAP_DIR / "pytorch_model.bin").is_file() or (CLAP_DIR / "model.safetensors").is_file()
    ):
        return None
    try:
        from transformers import ClapModel, ClapProcessor

        proc = ClapProcessor.from_pretrained(str(CLAP_DIR), local_files_only=True)
        model = ClapModel.from_pretrained(str(CLAP_DIR), local_files_only=True).eval()
        target_12 = load_taxonomy().get("target_12_sounds", {})
        prompts = [
            spec.get("clap_prompts", [f"the sound of {spec['friendly_name']}"])[0]
            for spec in target_12.values()
        ]
        keys = list(target_12.keys())
        with torch.inference_mode():
            out = model.get_text_features(**proc(text=prompts, return_tensors="pt", padding=True))
            emb = out if isinstance(out, torch.Tensor) else getattr(out, "pooler_output", None)
            if emb is None:
                emb = out[0]
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        _CLAP_CACHE = (proc, model, emb, keys)
        return _CLAP_CACHE
    except Exception:
        return None


def compute_ast_scores(waveform: np.ndarray, sr: int = SAMPLE_RATE) -> dict[str, float]:
    """Run AST AudioSet-527 inference and return label -> probability."""
    bundle = _get_ast()
    y = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if bundle is None or np.max(np.abs(y)) < 1e-5:
        return {}
    fe, model, id2label = bundle
    try:
        y16 = resample(y, sr, SAMPLE_RATE) if sr != SAMPLE_RATE else y
        with torch.inference_mode():
            probs = (
                torch.sigmoid(
                    model(**fe(y16, sampling_rate=SAMPLE_RATE, return_tensors="pt")).logits[0]
                )
                .cpu()
                .numpy()
            )
        return {id2label[i]: float(probs[i]) for i in range(len(probs))}
    except Exception:
        return {}


def compute_clap_audio_text_scores(waveform: np.ndarray, sr: int = SAMPLE_RATE) -> dict[str, float]:
    """Run zero-shot CLAP audio-text matching for the 12 target everyday sounds."""
    bundle = _get_clap()
    if bundle is None:
        return {}
    proc, model, text_emb, keys = bundle
    y = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if np.max(np.abs(y)) < 1e-5:
        return {k: 0.0 for k in keys}
    try:
        y48 = resample(y, sr, CLAP_SAMPLE_RATE) if sr != CLAP_SAMPLE_RATE else y
        with torch.inference_mode():
            out = model.get_audio_features(
                **proc(
                    audio=[y48], sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt", padding=True
                )
            )
            aud = out if isinstance(out, torch.Tensor) else getattr(out, "pooler_output", None)
            if aud is None:
                aud = out[0]
            aud = aud / aud.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            sims = (aud @ text_emb.T)[0].cpu().numpy()
        return {
            k: round(float(1.0 / (1.0 + np.exp(-18.0 * (float(s) - 0.31)))), 4)
            for k, s in zip(keys, sims)
        }
    except Exception:
        return {}


def _localize_sound_windows(
    waveform: np.ndarray,
    detected: list[dict[str, Any]],
    sr: int = SAMPLE_RATE,
    win_s: float = 1.5,
    hop_s: float = 0.75,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Localize each detected sound into contiguous time_ranges and detect co-occurring overlaps."""
    y = np.asarray(waveform, dtype=np.float32).reshape(-1)
    dur = len(y) / sr
    if dur <= 0 or not detected:
        return detected, [], 0

    target_12 = load_taxonomy().get("target_12_sounds", {})
    friendly_to_spec = {v["friendly_name"]: v for v in target_12.values()}
    win_len, hop_len = max(int(win_s * sr), 1024), max(int(hop_s * sr), 512)
    starts = list(range(0, max(1, len(y) - win_len // 2), hop_len)) or [0]

    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    total_rms = float(np.sqrt(np.mean(y**2)) + 1e-8)
    try:
        dom_bundle = load_tagger()
    except Exception:
        dom_bundle = None

    win_times = [(round(st / sr, 2), round(min(len(y), st + win_len) / sr, 2)) for st in starts]
    sound_probs: dict[str, list[float]] = {s["name"]: [] for s in detected}

    for st in starts:
        ed = min(len(y), st + win_len)
        seg = y[st:ed]
        seg_rms = float(np.sqrt(np.mean(seg**2)))
        if seg_rms < max(total_rms * 0.12, 1e-4):
            for s in detected:
                sound_probs[s["name"]].append(0.0)
            continue

        dom_f, dom_g = {}, {}
        if dom_bundle is not None and len(seg) >= int(0.4 * sr):
            try:
                m, ck = dom_bundle
                mel = librosa.feature.melspectrogram(
                    y=seg, sr=sr, n_fft=1024, hop_length=256, n_mels=64, fmin=20, fmax=sr // 2
                )
                lm = librosa.power_to_db(mel, ref=np.max)
                lm = (lm - lm.mean()) / (lm.std() + 1e-6)
                with torch.inference_mode():
                    gl, fl = m(
                        torch.from_numpy(lm.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    )
                    dom_g = dict(zip(ck["groups"], torch.sigmoid(gl[0]).cpu().numpy().tolist()))
                    dom_f = dict(zip(ck["vocabulary"], torch.sigmoid(fl[0]).cpu().numpy().tolist()))
            except Exception:
                pass

        S_win = S[:, st // 256 : max(st // 256 + 1, ed // 256)]
        win_e = float(np.sum(S_win**2) + 1e-12)
        e_fac = min(1.0, seg_rms / (total_rms * 0.65 + 1e-8))
        for s in detected:
            spec = friendly_to_spec.get(s["name"])
            f_lo, f_hi = spec.get("freq_band_hz", [60, 7500]) if spec else [60, 7500]
            bmask = (freqs >= f_lo) & (freqs <= f_hi)
            bratio = float(np.sum(S_win[bmask, :] ** 2)) / win_e if np.any(bmask) else 0.5
            dscore = max(
                dom_f.get(s.get("class", s["name"]), 0.0), dom_g.get(s.get("group", ""), 0.0) * 0.7
            )
            sound_probs[s["name"]].append(
                round(
                    float(s["confidence"])
                    * (0.45 * e_fac + 0.30 * min(1.0, bratio * 2.2) + 0.25 * max(dscore, e_fac)),
                    4,
                )
            )

    win_active: list[list[str]] = [[] for _ in starts]
    for s in detected:
        thr = max(0.18, float(s["confidence"]) * 0.42)
        ranges, cs, ce, cm = [], None, None, 0.0
        for wi, wp in enumerate(sound_probs[s["name"]]):
            t0, t1 = win_times[wi]
            if wp >= thr:
                win_active[wi].append(s["name"])
                if cs is None:
                    cs, ce, cm = t0, t1, wp
                elif t0 <= ce + 0.25:
                    ce, cm = t1, max(cm, wp)
                else:
                    ranges.append(
                        {"start": round(cs, 2), "end": round(ce, 2), "confidence": round(cm, 3)}
                    )
                    cs, ce, cm = t0, t1, wp
        if cs is not None:
            ranges.append({"start": round(cs, 2), "end": round(ce, 2), "confidence": round(cm, 3)})
        s["time_ranges"] = ranges or [
            {"start": 0.0, "end": round(dur, 2), "confidence": round(float(s["confidence"]), 3)}
        ]

    pair_wins: dict[tuple[str, str], list[tuple[float, float]]] = {}
    max_simul = 0
    for wi, names in enumerate(win_active):
        uniq = sorted(set(names))
        max_simul = max(max_simul, len(uniq))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                pair_wins.setdefault((uniq[i], uniq[j]), []).append(win_times[wi])

    overlaps = []
    for (s1, s2), tws in sorted(pair_wins.items()):
        merged, (cs, ce) = [], tws[0]
        for t0, t1 in tws[1:]:
            if t0 <= ce + 0.25:
                ce = max(ce, t1)
            else:
                merged.append({"start": round(cs, 2), "end": round(ce, 2)})
                cs, ce = t0, t1
        merged.append({"start": round(cs, 2), "end": round(ce, 2)})
        overlaps.append(
            {
                "sounds": [s1, s2],
                "intervals": merged,
                "total_overlap_seconds": round(sum(m["end"] - m["start"] for m in merged), 2),
            }
        )
    return detected, overlaps, max_simul


def tag_open_vocabulary(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    precomputed_logmel: torch.Tensor | None = None,
    use_clap_fallback: bool = True,
    localize_windows: bool = True,
) -> dict[str, Any]:
    """Combine AST-527, CLAP zero-shot, and the 32-class FSD tagger with time localization."""
    tax = load_taxonomy()
    target_12 = tax.get("target_12_sounds", {})
    dom_res = tag_audio_hierarchical(waveform, precomputed_logmel=precomputed_logmel)
    dom_groups = {g["group"]: g for g in dom_res.get("groups", [])}

    dom_fine_all = {}
    try:
        model, ckpt = load_tagger()
        feat = (
            precomputed_logmel if precomputed_logmel is not None else waveform_to_logmel(waveform)
        ).to(DEVICE)
        if feat.dim() == 3:
            feat = feat.unsqueeze(0)
        with torch.inference_mode():
            f_probs = torch.sigmoid(model(feat)[1][0].float()).cpu().numpy()
        dom_fine_all = {
            c: {"prob": float(p), "threshold": float(t)}
            for c, p, t in zip(ckpt["vocabulary"], f_probs, ckpt["thresholds"])
        }
    except Exception:
        pass

    ast_probs = compute_ast_scores(waveform, sr=sr)
    clap_probs = compute_clap_audio_text_scores(waveform, sr=sr) if use_clap_fallback else {}

    ast_grp_boost = {g: 0.0 for g in tax["groups"]}
    for lbl, p in ast_probs.items():
        if lbl in AUDIOSET_TO_FRIENDLY:
            _, grp = AUDIOSET_TO_FRIENDLY[lbl]
            ast_grp_boost[grp] = max(ast_grp_boost[grp], p)

    group_results = []
    for gname in tax["groups"]:
        dg = dom_groups.get(gname, {"probability": 0.0, "threshold": 0.45})
        gp = max(float(dg["probability"]), float(ast_grp_boost.get(gname, 0.0)) * 0.85)
        gt = float(dg.get("threshold", 0.45))
        group_results.append(
            {
                "group": gname,
                "probability": round(gp, 4),
                "threshold": round(gt, 4),
                "above_threshold": bool(gp >= gt),
            }
        )
    group_results.sort(key=lambda x: -x["probability"])
    active_groups = {g["group"] for g in group_results if g["above_threshold"]}

    candidates: dict[str, dict[str, Any]] = {}
    has_ast = bool(ast_probs)
    for fsd_cls, info in dom_fine_all.items():
        friendly, grp = get_friendly_name(fsd_cls, tax), get_group_for_class(fsd_cls, tax)
        p_dom = info["prob"]
        p_ast = max(ast_probs.get(fsd_cls.replace("_", " "), 0.0), ast_probs.get(friendly, 0.0))
        comb = (
            max(p_ast, 0.72 * p_ast + 0.28 * p_dom if p_ast >= 0.04 else 0.45 * p_dom)
            if has_ast
            else p_dom
        )
        thr = 0.25 if has_ast else info["threshold"]
        candidates[friendly] = {
            "name": friendly,
            "class": fsd_cls,
            "group": grp,
            "confidence": round(float(comb), 4),
            "threshold": round(float(thr), 4),
            "above_threshold": bool(comb >= thr and (grp in active_groups or comb >= 0.32)),
            "source_model": "ast_audioset_527+fsd_32_cnn" if has_ast else "fsd_32_cnn",
        }

    for t_key, spec in target_12.items():
        friendly, grp, fsd_cls = spec["friendly_name"], spec["group"], spec["fsd_class"]
        p_ast = max(
            (ast_probs.get(lbl, 0.0) for lbl in spec.get("audioset_labels", [])), default=0.0
        )
        p_clap = float(clap_probs.get(t_key, 0.0))
        p_dom = float(dom_fine_all.get(fsd_cls, {}).get("prob", 0.0))
        if t_key == "speech":
            p_dom = max(p_dom, float(dom_groups.get("Human voice", {}).get("probability", 0.0)))

        if fsd_cls in dom_fine_all:
            thr = 0.26
            best_p = max(p_ast, 0.75 * p_ast + 0.25 * p_clap if has_ast else p_dom)
            if p_dom >= 0.65 and (p_ast >= 0.08 or not has_ast):
                best_p = max(best_p, 0.55 * p_dom + 0.45 * max(p_ast, p_clap))
            src_mod = "ast_audioset_527+clap" if max(p_ast, p_clap) >= p_dom else "fsd_32_cnn"
        elif t_key == "speech":
            thr = 0.35
            best_p = max(p_ast, p_dom * 0.85 if p_ast >= 0.08 or not has_ast else p_ast)
            src_mod = "ast_audioset_527" if p_ast >= p_dom else "fsd_32_cnn"
        elif t_key == "phone vibration":
            thr = 0.35
            best_p = max(
                p_ast,
                p_clap * 0.80
                if p_clap >= 0.65
                else (0.55 * p_ast + 0.45 * p_clap if p_ast >= 0.10 else p_clap * 0.40),
            )
            src_mod = "clap_zero_shot+ast"
        else:
            thr = 0.30
            best_p = max(p_ast, p_clap * 0.85 if p_clap >= 0.55 else 0.70 * p_ast + 0.30 * p_clap)
            src_mod = "ast_audioset_527+clap"

        if best_p >= thr:
            active_groups.add(grp)
            for g in group_results:
                if g["group"] == grp:
                    g["probability"] = round(max(g["probability"], best_p), 4)
                    g["above_threshold"] = True
        if friendly not in candidates or best_p > candidates[friendly]["confidence"]:
            candidates[friendly] = {
                "name": friendly,
                "class": fsd_cls,
                "group": grp,
                "confidence": round(float(best_p), 4),
                "threshold": round(float(thr), 4),
                "above_threshold": bool(best_p >= thr and grp in active_groups),
                "source_model": src_mod,
            }

    for as_lbl, p_ast in sorted(ast_probs.items(), key=lambda x: -x[1])[:15]:
        if p_ast < 0.28:
            break
        if as_lbl not in AUDIOSET_TO_FRIENDLY:
            continue
        friendly, grp = AUDIOSET_TO_FRIENDLY[as_lbl]
        if friendly not in candidates or p_ast > candidates[friendly]["confidence"]:
            active_groups.add(grp)
            candidates[friendly] = {
                "name": friendly,
                "class": as_lbl.replace(", ", "_").replace(" ", "_"),
                "group": grp,
                "confidence": round(float(p_ast), 4),
                "threshold": 0.28,
                "above_threshold": True,
                "source_model": "ast_audioset_527",
            }

    allowed = active_groups or ({group_results[0]["group"]} if group_results else set())
    all_tags = sorted(
        [
            {
                "name": it["name"],
                "friendly_name": it["name"],
                "class": it["class"],
                "group": it["group"],
                "probability": it["confidence"],
                "confidence": it["confidence"],
                "threshold": it["threshold"],
                "above_threshold": bool(it["above_threshold"] and it["group"] in active_groups),
                "source_model": it["source_model"],
            }
            for it in candidates.values()
            if it["group"] in allowed
        ],
        key=lambda x: -x["probability"],
    )
    detected = [dict(t) for t in all_tags if t["above_threshold"]]
    if "Human voice" in active_groups and not any(d["group"] == "Human voice" for d in detected):
        hv_p = next((g["probability"] for g in group_results if g["group"] == "Human voice"), 0.8)
        hv = {
            "name": "Human voice",
            "friendly_name": "Human voice",
            "class": "Speech",
            "group": "Human voice",
            "probability": round(float(hv_p), 4),
            "confidence": round(float(hv_p), 4),
            "threshold": 0.35,
            "above_threshold": True,
            "source_model": "ast_audioset_527"
            if ast_probs.get("Speech", 0.0) >= 0.3
            else "fsd_32_cnn",
        }
        detected.insert(0, hv)
        if not any(t["name"] == "Human voice" for t in all_tags):
            all_tags.insert(0, dict(hv))

    overlaps, max_simul = [], len(detected)
    if localize_windows and detected:
        detected, overlaps, max_simul = _localize_sound_windows(waveform, detected, sr=sr)
        rmap = {d["name"]: d.get("time_ranges", []) for d in detected}
        for t in all_tags:
            if t["name"] in rmap:
                t["time_ranges"] = rmap[t["name"]]

    is_speech = bool(
        (
            group_results
            and group_results[0]["group"] == "Human voice"
            and group_results[0]["above_threshold"]
        )
        or any(d["name"] == "Human voice" and d["confidence"] >= 0.55 for d in detected)
    )
    return {
        "available": True,
        "hierarchical": True,
        "open_vocabulary": True,
        "models_used": {
            "domain_32_cnn": bool(dom_fine_all),
            "ast_audioset_527": bool(ast_probs),
            "clap_zero_shot": bool(clap_probs),
        },
        "groups": group_results,
        "active_groups": [g["group"] for g in group_results if g["above_threshold"]],
        "named_sounds": detected,
        "tags": all_tags,
        "all_probabilities": {k: v["confidence"] for k, v in candidates.items()},
        "overlapping_sounds": overlaps,
        "max_simultaneous_sounds": max_simul,
        "speech_detected": is_speech,
        "caveat": "Detected sounds combine AudioSet-527 (AST), open-vocabulary CLAP audio-text matching, and the 32-class FSD tagger with frame-level time localization.",
    }
