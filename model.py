"""Shared audio preprocessing, CNN architecture, and source-count inference."""
from functools import lru_cache
from pathlib import Path
from typing import Optional
import os

import av
import librosa
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "checkpoints" / "source_count_clip.pt"))).expanduser().resolve()
SAMPLE_RATE = 16_000
CLIP_SECONDS = 8.0
N_MELS = 64
N_FFT = 1_024
HOP_LENGTH = 256
FEATURE_VERSION = 2
try:
    MAX_SOURCES = max(1, int(os.getenv("MAX_SOURCES", "32")))
except ValueError:
    MAX_SOURCES = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)


class SourceCountCNN(nn.Module):
    """Classify one full-clip log-mel spectrogram into an ordinal count range."""

    def __init__(self, min_sources=1, max_sources=MAX_SOURCES):
        super().__init__()
        self.min_sources = int(min_sources)
        self.max_sources = int(max_sources)
        if self.min_sources < 1 or self.max_sources < self.min_sources:
            raise ValueError("Source-count range must satisfy 1 <= min_sources <= max_sources.")
        class_count = self.max_sources - self.min_sources + 1
        self.features = nn.Sequential(
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
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, class_count),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _decode_audio(path):
    """Decode common audio formats and resample to mono 16 kHz with PyAV."""
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise ValueError("The file contains no audio stream.")
            stream = container.streams.audio[0]
            original_sample_rate = int(stream.rate or 0)
            if original_sample_rate <= 0:
                raise ValueError("The audio stream has no valid sample rate.")
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
            chunks = []
            for frame in container.decode(stream):
                chunks.extend(
                    output.to_ndarray()[0].astype(np.float32, copy=False)
                    for output in resampler.resample(frame)
                )
            chunks.extend(
                output.to_ndarray()[0].astype(np.float32, copy=False)
                for output in resampler.resample(None)
            )
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"Could not decode the audio file: {exc}") from exc
    if not chunks:
        raise ValueError("The audio file is empty or could not be decoded.")
    waveform = np.concatenate(chunks).astype(np.float32, copy=False)
    return np.nan_to_num(waveform, copy=False), original_sample_rate


def load_audio(path):
    """Decode, resample, and center-crop or right-pad audio to eight seconds."""
    waveform, original_sample_rate = _decode_audio(path)
    if waveform.size == 0:
        raise ValueError("The audio file is empty.")
    original_duration = len(waveform) / SAMPLE_RATE
    if len(waveform) > CLIP_SAMPLES:
        start = (len(waveform) - CLIP_SAMPLES) // 2
        crop_start = start / SAMPLE_RATE
        waveform = waveform[start:start + CLIP_SAMPLES]
    else:
        crop_start = 0.0
        waveform = np.pad(waveform, (0, CLIP_SAMPLES - len(waveform)))
    metadata = {
        "original_sample_rate": original_sample_rate,
        "original_duration_seconds": original_duration,
        "crop_start_seconds": crop_start,
    }
    return waveform.astype(np.float32, copy=False), metadata


def waveform_to_logmel(waveform):
    """Convert one eight-second waveform to a normalized [1, 64, frames] tensor."""
    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if len(samples) < CLIP_SAMPLES:
        samples = np.pad(samples, (0, CLIP_SAMPLES - len(samples)))
    elif len(samples) > CLIP_SAMPLES:
        start = (len(samples) - CLIP_SAMPLES) // 2
        samples = samples[start:start + CLIP_SAMPLES]
    mel = librosa.feature.melspectrogram(
        y=samples,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=40,
        fmax=SAMPLE_RATE / 2,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    normalized = np.clip(log_mel / 80.0, -1.0, 0.0)
    return torch.from_numpy(normalized.astype(np.float32, copy=False)).unsqueeze(0)


def _read_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    arch = checkpoint.get("architecture", "") if isinstance(checkpoint, dict) else ""
    if not isinstance(checkpoint, dict) or not str(arch).startswith("SourceCountCNN"):
        raise ValueError("Checkpoint does not match the full-clip SourceCountCNN.")
    state = checkpoint.get("model_state_dict")
    min_sources = int(checkpoint.get("min_sources", 0))
    max_sources = int(checkpoint.get("max_sources", 0))
    if not isinstance(state, dict) or min_sources < 1 or max_sources < min_sources:
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
        if checkpoint.get(name) != value:
            raise ValueError(f"Checkpoint setting {name}={checkpoint.get(name)!r} does not match {value!r}.")
    return checkpoint, state, min_sources, max_sources


def checkpoint_status(path=MODEL_PATH):
    """Report whether a compatible full-clip checkpoint exists."""
    path = Path(path)
    if not path.is_file():
        return {"trained": False, "min_sources": 1, "max_sources": MAX_SOURCES, "error": None}
    try:
        checkpoint, state, min_sources, max_sources = _read_checkpoint(path)
        SourceCountCNN(min_sources, max_sources).load_state_dict(state)
        return {
            "trained": bool(checkpoint.get("trained", True)),
            "min_sources": min_sources,
            "max_sources": max_sources,
            "error": None,
            "epoch": checkpoint.get("epoch"),
            "val_mae": checkpoint.get("val_mae"),
            "val_macro_mae": checkpoint.get("val_macro_mae"),
            "val_exact_accuracy": checkpoint.get("val_exact_accuracy"),
            "val_within_one": checkpoint.get("val_within_one"),
            "sparse_specialist": bool(checkpoint.get("sparse_specialist_state_dict")),
        }
    except Exception as exc:
        return {"trained": False, "min_sources": 1, "max_sources": MAX_SOURCES, "error": str(exc)}


@lru_cache(maxsize=4)
def _load_model_cached(path, modified_ns, file_size):
    checkpoint, state, min_sources, max_sources = _read_checkpoint(path)
    model = SourceCountCNN(min_sources, max_sources).to(DEVICE)
    model.load_state_dict(state)
    model.eval()

    sparse_model = None
    sp_state = checkpoint.get("sparse_specialist_state_dict")
    if isinstance(sp_state, dict):
        sp_min = int(checkpoint.get("sparse_min_sources", 2))
        sp_max = int(checkpoint.get("sparse_max_sources", 8))
        sparse_model = SourceCountCNN(sp_min, sp_max).to(DEVICE)
        sparse_model.load_state_dict(sp_state)
        sparse_model.eval()
    return model, sparse_model, checkpoint, min_sources, max_sources


def _load_model(path=MODEL_PATH):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No compatible trained checkpoint found at {path}. Run train.py first.")
    stat = path.stat()
    full_tuple = _load_model_cached(str(path), stat.st_mtime_ns, stat.st_size)
    return full_tuple[0], full_tuple[2], full_tuple[3], full_tuple[4]


def _load_model_with_specialist(path=MODEL_PATH):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No compatible trained checkpoint found at {path}. Run train.py first.")
    stat = path.stat()
    return _load_model_cached(str(path), stat.st_mtime_ns, stat.st_size)


def predict_source_count(waveform, model_path=MODEL_PATH, precomputed_logmel: Optional[torch.Tensor] = None):
    """Predict a count for the complete eight-second analysis clip.

    If precomputed_logmel ([1, 64, T]) is passed, skips recomputing the mel spectrogram.
    Supports two-stage coarse -> sparse specialist (K <= 8) when embedded in the checkpoint.
    """
    model, sparse_model, checkpoint, min_sources, max_sources = _load_model_with_specialist(model_path)
    if precomputed_logmel is not None:
        features = precomputed_logmel.unsqueeze(0).to(DEVICE) if precomputed_logmel.dim() == 3 else precomputed_logmel.to(DEVICE)
    else:
        features = waveform_to_logmel(waveform).unsqueeze(0).to(DEVICE)

    calibrator = checkpoint.get("calibrator")
    sparse_calibrator = checkpoint.get("sparse_calibrator")
    use_amp = DEVICE.type == "cuda"

    with torch.inference_mode(), torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
        probabilities = F.softmax(model(features), dim=1)[0].float()
        values = torch.arange(min_sources, max_sources + 1, dtype=probabilities.dtype, device=DEVICE)
        coarse_raw = float(torch.sum(probabilities * values).item())

        if calibrator and "x_thresholds" in calibrator and "y_thresholds" in calibrator:
            coarse_cal = float(np.interp(coarse_raw, np.array(calibrator["x_thresholds"]), np.array(calibrator["y_thresholds"])))
        else:
            coarse_cal = coarse_raw

        # Two-stage sparse specialist routing: if coarse estimate <= 8.5 and specialist exists, use specialist
        used_specialist = False
        if sparse_model is not None and coarse_cal <= 8.5:
            used_specialist = True
            sp_min = int(checkpoint.get("sparse_min_sources", 2))
            sp_max = int(checkpoint.get("sparse_max_sources", 8))
            sp_probs = F.softmax(sparse_model(features), dim=1)[0].float()
            sp_vals = torch.arange(sp_min, sp_max + 1, dtype=sp_probs.dtype, device=DEVICE)
            raw_expected = float(torch.sum(sp_probs * sp_vals).item())
            if sparse_calibrator and "x_thresholds" in sparse_calibrator and "y_thresholds" in sparse_calibrator:
                calibrated_value = float(np.interp(raw_expected, np.array(sparse_calibrator["x_thresholds"]), np.array(sparse_calibrator["y_thresholds"])))
            else:
                calibrated_value = raw_expected
            effective_value = calibrated_value
            estimate = min(max(int(round(effective_value)), sp_min), sp_max)
            class_index = estimate - sp_min
            confidence = float(sp_probs[class_index].item() * 100.0)
            estimated_sources = estimate
        else:
            raw_expected = coarse_raw
            calibrated_value = coarse_cal
            effective_value = calibrated_value
            if checkpoint.get("decision_rule") == "argmax":
                class_index = int(torch.argmax(probabilities).item())
            else:
                estimate = min(max(int(round(effective_value)), min_sources), max_sources)
                class_index = estimate - min_sources
            estimated_sources = class_index + min_sources
            confidence = float(probabilities[class_index].item() * 100.0)

    out_of_range = (raw_expected < (min_sources - 0.5)) or (raw_expected > (max_sources + 0.5))
    range_warning = None
    if out_of_range:
        range_warning = f"Input predicted count {raw_expected:.1f} is outside the calibrated model range ({min_sources}–{max_sources} sources)."

    return {
        "raw_value": raw_expected,
        "coarse_raw_value": coarse_raw,
        "calibrated_value": calibrated_value,
        "calibrated": (calibrator is not None) or (sparse_calibrator is not None),
        "used_sparse_specialist": used_specialist,
        "estimated_sources": estimated_sources,
        "confidence": confidence,
        "min_sources": min_sources,
        "max_sources": max_sources,
        "out_of_range": out_of_range,
        "range_warning": range_warning,
        "sparse_regime": estimated_sources <= 8,
        "total_windows": 1,
        "timeline": [{
            "start": 0.0,
            "end": CLIP_SECONDS,
            "count": estimated_sources,
            "confidence": confidence / 100.0,
        }],
    }
