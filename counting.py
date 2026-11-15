"""Source counting with an honest uncertainty band.

The counting network (model.SourceCountCNN) is not modified. This module only adds
read-only inference helpers on top of it, so it stays the single source of truth.
"""
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.nn import functional as F

from model import DEVICE, MODEL_PATH, _load_model_with_specialist, predict_source_count, waveform_to_logmel

DEFAULT_MC_PASSES = 30


def file_sha256(path, chunk=1 << 20):
    """Hex SHA-256 of a file; ties evaluation results and manifests to exact weights."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def count_with_uncertainty(
    waveform,
    model_path=MODEL_PATH,
    passes=DEFAULT_MC_PASSES,
    precomputed_logmel: Optional[torch.Tensor] = None,
):
    """Count sources and attach a batched Monte-Carlo-dropout uncertainty band.

    All `passes` stochastic forward passes through the classifier head are batched into a
    single matrix multiply after computing the convolutional trunk once. Supports
    torch.inference_mode, AMP on CUDA, and precomputed log-mel caching.
    """
    passes = max(2, int(passes))
    model, sparse_model, checkpoint, min_sources, max_sources = _load_model_with_specialist(model_path)
    if precomputed_logmel is None:
        precomputed_logmel = waveform_to_logmel(waveform)

    base = predict_source_count(waveform, model_path, precomputed_logmel=precomputed_logmel)
    features = precomputed_logmel.unsqueeze(0).to(DEVICE) if precomputed_logmel.dim() == 3 else precomputed_logmel.to(DEVICE)

    use_specialist = base.get("used_sparse_specialist", False) and sparse_model is not None
    active_model = sparse_model if use_specialist else model
    act_min = int(checkpoint.get("sparse_min_sources", 2)) if use_specialist else min_sources
    act_max = int(checkpoint.get("sparse_max_sources", 8)) if use_specialist else max_sources
    calibrator = checkpoint.get("sparse_calibrator") if use_specialist else checkpoint.get("calibrator")

    head = active_model.classifier  # Flatten, Linear, ReLU, Dropout, Linear
    use_amp = DEVICE.type == "cuda"

    with torch.inference_mode(), torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
        hidden = head[2](head[1](head[0](active_model.features(features))))
        hidden = hidden.repeat(passes, 1)
        hidden = F.dropout(hidden, p=head[3].p, training=True)
        probabilities = F.softmax(head[4](hidden), dim=1).float()
        values = torch.arange(act_min, act_max + 1, dtype=probabilities.dtype, device=DEVICE)
        mean_per_pass = (probabilities * values).sum(dim=1)
        var_per_pass = (probabilities * values.pow(2)).sum(dim=1) - mean_per_pass.pow(2)

        if calibrator and "x_thresholds" in calibrator and "y_thresholds" in calibrator:
            x_thresh = np.array(calibrator["x_thresholds"])
            y_thresh = np.array(calibrator["y_thresholds"])
            calibrated_passes = torch.from_numpy(
                np.interp(mean_per_pass.cpu().numpy(), x_thresh, y_thresh).astype(np.float32)
            ).to(DEVICE)
            mc_mean = float(calibrated_passes.mean())
            mc_var = float(calibrated_passes.var(unbiased=True))
        else:
            mc_mean = float(mean_per_pass.mean())
            mc_var = float(mean_per_pass.var(unbiased=True))
        within = float(var_per_pass.clamp_min(0).mean())

    return {
        "raw_regression_value": base["raw_value"],
        "coarse_raw_value": base.get("coarse_raw_value", base["raw_value"]),
        "calibrated_value": base.get("calibrated_value", base["raw_value"]),
        "calibrated": base.get("calibrated", False),
        "used_sparse_specialist": base.get("used_sparse_specialist", False),
        "rounded_count": int(base["estimated_sources"]),
        "mc_mean": mc_mean,
        "mc_std": mc_var ** 0.5,
        "softmax_std": within ** 0.5,
        "total_std": (mc_var + within) ** 0.5,
        "passes": passes,
        "method": "mc_dropout",
        "confidence_percent": float(base["confidence"]),
        "min_sources": int(min_sources),
        "max_sources": int(max_sources),
        "out_of_range": base.get("out_of_range", False),
        "range_warning": base.get("range_warning"),
        "sparse_regime": base.get("sparse_regime", int(base["estimated_sources"]) <= 8),
        "decision_rule": checkpoint.get("decision_rule", "expected"),
    }


def checkpoint_fingerprint(path=MODEL_PATH):
    path = Path(path)
    return file_sha256(path) if path.is_file() else None
