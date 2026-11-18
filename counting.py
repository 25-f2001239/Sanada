"""SourceCountCNN architecture, checkpoint loading, sparse specialist routing, and MC-dropout uncertainty."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from audio import file_sha256, waveform_to_logmel
from config import (
    CLIP_SECONDS,
    DEVICE,
    FEATURE_VERSION,
    HOP_LENGTH,
    MAX_SOURCES,
    MODEL_PATH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
)

DEFAULT_MC_PASSES = 30


def conv_trunk() -> nn.Sequential:
    """Shared 3-block 2D CNN feature extractor used by SourceCountCNN and tagger models."""
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((4, 16)),
    )


class SourceCountCNN(nn.Module):
    """Classify one full-clip log-mel spectrogram into an ordinal count range."""

    def __init__(self, min_sources: int = 1, max_sources: int = MAX_SOURCES):
        super().__init__()
        self.min_sources, self.max_sources = int(min_sources), int(max_sources)
        if self.min_sources < 1 or self.max_sources < self.min_sources:
            raise ValueError("Source-count range must satisfy 1 <= min_sources <= max_sources.")
        self.features = conv_trunk()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, self.max_sources - self.min_sources + 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _read_checkpoint(path: Path | str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or not str(ckpt.get("architecture", "")).startswith(
        "SourceCountCNN"
    ):
        raise ValueError("Checkpoint does not match the full-clip SourceCountCNN.")
    state = ckpt.get("model_state_dict")
    min_s, max_s = int(ckpt.get("min_sources", 0)), int(ckpt.get("max_sources", 0))
    if not isinstance(state, dict) or min_s < 1 or max_s < min_s:
        raise ValueError("Checkpoint is missing its model weights or source-count range.")
    expected = {
        "sample_rate": SAMPLE_RATE,
        "clip_seconds": CLIP_SECONDS,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "feature_version": FEATURE_VERSION,
    }
    for name, value in expected.items():
        if ckpt.get(name) != value:
            raise ValueError(
                f"Checkpoint setting {name}={ckpt.get(name)!r} does not match {value!r}."
            )
    return ckpt, state, min_s, max_s


def checkpoint_status(path: Path | str = MODEL_PATH) -> dict[str, Any]:
    """Report whether a compatible counting checkpoint exists and its validation metadata."""
    path = Path(path)
    if not path.is_file():
        return {"trained": False, "min_sources": 1, "max_sources": MAX_SOURCES, "error": None}
    try:
        ckpt, state, min_s, max_s = _read_checkpoint(path)
        SourceCountCNN(min_s, max_s).load_state_dict(state)
        return {
            "trained": bool(ckpt.get("trained", True)),
            "min_sources": min_s,
            "max_sources": max_s,
            "error": None,
            "epoch": ckpt.get("epoch"),
            "val_mae": ckpt.get("val_mae"),
            "val_macro_mae": ckpt.get("val_macro_mae"),
            "val_exact_accuracy": ckpt.get("val_exact_accuracy"),
            "val_within_one": ckpt.get("val_within_one"),
            "sparse_specialist": bool(ckpt.get("sparse_specialist_state_dict")),
        }
    except Exception as exc:
        return {"trained": False, "min_sources": 1, "max_sources": MAX_SOURCES, "error": str(exc)}


@lru_cache(maxsize=4)
def _load_model_cached(path: str, _mtime_ns: int, _size: int):
    ckpt, state, min_s, max_s = _read_checkpoint(path)
    model = SourceCountCNN(min_s, max_s).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    sparse_model = None
    sp_state = ckpt.get("sparse_specialist_state_dict")
    if isinstance(sp_state, dict):
        sparse_model = SourceCountCNN(
            int(ckpt.get("sparse_min_sources", 2)), int(ckpt.get("sparse_max_sources", 8))
        ).to(DEVICE)
        sparse_model.load_state_dict(sp_state)
        sparse_model.eval()
    return model, sparse_model, ckpt, min_s, max_s


def load_model_with_specialist(path: Path | str = MODEL_PATH):
    """Load cached general and optional sparse specialist SourceCountCNN models."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No compatible trained checkpoint found at {path}.")
    st = path.stat()
    return _load_model_cached(str(path), st.st_mtime_ns, st.st_size)


def _calibrate(val: float, calibrator: dict | None) -> float:
    if calibrator and "x_thresholds" in calibrator and "y_thresholds" in calibrator:
        return float(
            np.interp(
                val, np.array(calibrator["x_thresholds"]), np.array(calibrator["y_thresholds"])
            )
        )
    return val


def predict_source_count(
    waveform: np.ndarray,
    model_path: Path | str = MODEL_PATH,
    precomputed_logmel: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Predict source count for an 8-second clip using two-stage sparse specialist routing."""
    model, sparse_model, ckpt, min_s, max_s = load_model_with_specialist(model_path)
    feat = (
        precomputed_logmel if precomputed_logmel is not None else waveform_to_logmel(waveform)
    ).to(DEVICE)
    if feat.dim() == 3:
        feat = feat.unsqueeze(0)

    calibrator, sp_calibrator = ckpt.get("calibrator"), ckpt.get("sparse_calibrator")
    with (
        torch.inference_mode(),
        torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")),
    ):
        probs = F.softmax(model(feat), dim=1)[0].float()
        vals = torch.arange(min_s, max_s + 1, dtype=probs.dtype, device=DEVICE)
        coarse_raw = float(torch.sum(probs * vals).item())
        coarse_cal = _calibrate(coarse_raw, calibrator)

        used_specialist = sparse_model is not None and coarse_cal <= 8.5
        if used_specialist:
            sp_min, sp_max = (
                int(ckpt.get("sparse_min_sources", 2)),
                int(ckpt.get("sparse_max_sources", 8)),
            )
            sp_probs = F.softmax(sparse_model(feat), dim=1)[0].float()
            sp_vals = torch.arange(sp_min, sp_max + 1, dtype=sp_probs.dtype, device=DEVICE)
            raw_expected = float(torch.sum(sp_probs * sp_vals).item())
            calibrated_value = _calibrate(raw_expected, sp_calibrator)
            estimated_sources = min(max(int(round(calibrated_value)), sp_min), sp_max)
            confidence = float(sp_probs[estimated_sources - sp_min].item() * 100.0)
        else:
            raw_expected, calibrated_value = coarse_raw, coarse_cal
            if ckpt.get("decision_rule") == "argmax":
                class_index = int(torch.argmax(probs).item())
            else:
                class_index = min(max(int(round(calibrated_value)), min_s), max_s) - min_s
            estimated_sources = class_index + min_s
            confidence = float(probs[class_index].item() * 100.0)

    out_of_range = (raw_expected < min_s - 0.5) or (raw_expected > max_s + 0.5)
    return {
        "raw_value": raw_expected,
        "coarse_raw_value": coarse_raw,
        "calibrated_value": calibrated_value,
        "calibrated": (calibrator is not None) or (sp_calibrator is not None),
        "used_sparse_specialist": used_specialist,
        "estimated_sources": estimated_sources,
        "confidence": confidence,
        "min_sources": min_s,
        "max_sources": max_s,
        "out_of_range": out_of_range,
        "range_warning": (
            f"Input predicted count {raw_expected:.1f} is outside the calibrated model range ({min_s}–{max_s} sources)."
            if out_of_range
            else None
        ),
        "sparse_regime": estimated_sources <= 8,
        "total_windows": 1,
        "timeline": [
            {
                "start": 0.0,
                "end": CLIP_SECONDS,
                "count": estimated_sources,
                "confidence": confidence / 100.0,
            }
        ],
    }


def count_with_uncertainty(
    waveform: np.ndarray,
    model_path: Path | str = MODEL_PATH,
    passes: int = DEFAULT_MC_PASSES,
    precomputed_logmel: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Count sources and decompose total uncertainty into MC-dropout and softmax spread."""
    passes = max(2, int(passes))
    model, sparse_model, ckpt, min_s, max_s = load_model_with_specialist(model_path)
    if precomputed_logmel is None:
        precomputed_logmel = waveform_to_logmel(waveform)
    base = predict_source_count(waveform, model_path, precomputed_logmel=precomputed_logmel)
    feat = (
        precomputed_logmel.unsqueeze(0).to(DEVICE)
        if precomputed_logmel.dim() == 3
        else precomputed_logmel.to(DEVICE)
    )

    use_sp = base.get("used_sparse_specialist", False) and sparse_model is not None
    active_model = sparse_model if use_sp else model
    act_min = int(ckpt.get("sparse_min_sources", 2)) if use_sp else min_s
    act_max = int(ckpt.get("sparse_max_sources", 8)) if use_sp else max_s
    calibrator = ckpt.get("sparse_calibrator") if use_sp else ckpt.get("calibrator")
    head = active_model.classifier

    with (
        torch.inference_mode(),
        torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")),
    ):
        hidden = head[2](head[1](head[0](active_model.features(feat)))).repeat(passes, 1)
        hidden = F.dropout(hidden, p=head[3].p, training=True)
        probs = F.softmax(head[4](hidden), dim=1).float()
        values = torch.arange(act_min, act_max + 1, dtype=probs.dtype, device=DEVICE)
        mean_per_pass = (probs * values).sum(dim=1)
        var_per_pass = (probs * values.pow(2)).sum(dim=1) - mean_per_pass.pow(2)

        if calibrator and "x_thresholds" in calibrator and "y_thresholds" in calibrator:
            cal_passes = torch.from_numpy(
                np.interp(
                    mean_per_pass.cpu().numpy(),
                    np.array(calibrator["x_thresholds"]),
                    np.array(calibrator["y_thresholds"]),
                ).astype(np.float32)
            ).to(DEVICE)
            mc_mean, mc_var = float(cal_passes.mean()), float(cal_passes.var(unbiased=True))
        else:
            mc_mean, mc_var = float(mean_per_pass.mean()), float(mean_per_pass.var(unbiased=True))
        within = float(var_per_pass.clamp_min(0).mean())

    return {
        "raw_regression_value": base["raw_value"],
        "coarse_raw_value": base.get("coarse_raw_value", base["raw_value"]),
        "calibrated_value": base.get("calibrated_value", base["raw_value"]),
        "calibrated": base.get("calibrated", False),
        "used_sparse_specialist": use_sp,
        "rounded_count": int(base["estimated_sources"]),
        "mc_mean": mc_mean,
        "mc_std": mc_var**0.5,
        "softmax_std": within**0.5,
        "total_std": (mc_var + within) ** 0.5,
        "passes": passes,
        "method": "mc_dropout",
        "confidence_percent": float(base["confidence"]),
        "min_sources": int(min_s),
        "max_sources": int(max_s),
        "out_of_range": base.get("out_of_range", False),
        "range_warning": base.get("range_warning"),
        "sparse_regime": base.get("sparse_regime", int(base["estimated_sources"]) <= 8),
        "decision_rule": ckpt.get("decision_rule", "expected"),
    }


def checkpoint_fingerprint(path: Path | str = MODEL_PATH) -> str | None:
    """Return SHA-256 hex digest of checkpoint if it exists."""
    p = Path(path)
    return file_sha256(p) if p.is_file() else None
