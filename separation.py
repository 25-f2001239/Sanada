"""Count-conditioned, iterative (recursive) source separation.

Design
------
One source is peeled off per pass. Pass i receives the log-magnitude STFT of the
original mixture, the log-magnitude STFT of the current residual, and two constant
conditioning planes (i / MAX_PASSES and K / MAX_PASSES, K = estimated source count).
It predicts a [0, 1] time-frequency mask that is applied to the *residual*; the masked
STFT is that pass's stem and is subtracted from the residual. Passes stop after K
stems, or earlier when the residual energy falls below `residual_threshold` times the
mixture energy.

No weights ship with the app. `separator_status()` reports "not available" until a
checkpoint trained for exactly this architecture exists at SEPARATOR_PATH. Nothing here
synthesises audio without running that network.

Checkpoint format (torch.save of a dict, loaded with weights_only=True):
    architecture: "IterativeMaskSeparator"
    sample_rate, n_fft, hop_length: ints (must equal the constants below)
    config: {"channels": int, "blocks": int}
    model_state_dict: state dict
    version: optional string
"""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

from counting import file_sha256

BASE_DIR = Path(__file__).resolve().parent
SEPARATOR_PATH = Path(
    os.getenv("SEPARATOR_PATH", str(BASE_DIR / "checkpoints" / "separator_iterative.pt"))
).expanduser().resolve()
SEPARATOR_ARCHITECTURE = "IterativeMaskSeparator"
SEPARATION_SAMPLE_RATE = 16_000
N_FFT = 1_024
HOP_LENGTH = 256
MAX_PASSES = 32  # normalisation constant for the conditioning planes; do not change after training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _env_int(name, default, low=1):
    try:
        return max(low, int(os.getenv(name, default)))
    except ValueError:
        return default


def _env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


MAX_SEPARABLE = _env_int("MAX_SEPARABLE", 8)
RESIDUAL_THRESHOLD = _env_float("RESIDUAL_THRESHOLD", 0.01)  # residual/mixture energy ratio (-20 dB)


class _Block(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.body(x))


class IterativeMaskSeparator(nn.Module):
    """Predict one residual mask per pass from (mixture, residual, pass, K) features."""

    def __init__(self, channels=32, blocks=6):
        super().__init__()
        self.channels, self.blocks = int(channels), int(blocks)
        self.stem = nn.Sequential(
            nn.Conv2d(4, self.channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[_Block(self.channels, 2 ** (i % 4)) for i in range(self.blocks)])
        self.head = nn.Conv2d(self.channels, 1, 1)

    def forward(self, mix_logmag, res_logmag, pass_index, k):
        shape = mix_logmag.shape
        index_plane = torch.full_like(mix_logmag, pass_index / MAX_PASSES)
        count_plane = torch.full_like(mix_logmag, k / MAX_PASSES)
        stacked = torch.stack([mix_logmag, res_logmag, index_plane, count_plane], dim=1)
        assert stacked.shape[2:] == shape[1:]
        return torch.sigmoid(self.head(self.trunk(self.stem(stacked))))[:, 0]


def _read_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("architecture") != SEPARATOR_ARCHITECTURE:
        raise ValueError(f"Checkpoint is not an {SEPARATOR_ARCHITECTURE} checkpoint.")
    for name, value in (("sample_rate", SEPARATION_SAMPLE_RATE), ("n_fft", N_FFT), ("hop_length", HOP_LENGTH)):
        if checkpoint.get(name) != value:
            raise ValueError(f"Checkpoint setting {name}={checkpoint.get(name)!r} does not match {value!r}.")
    config = checkpoint.get("config") or {}
    if not isinstance(checkpoint.get("model_state_dict"), dict) or "channels" not in config or "blocks" not in config:
        raise ValueError("Checkpoint is missing model_state_dict or config.")
    return checkpoint, config


def separator_status(path=SEPARATOR_PATH):
    """Cheap status for the UI: available flag plus a human-readable reason when not."""
    path = Path(path)
    base = {"max_separable": MAX_SEPARABLE, "sample_rate": SEPARATION_SAMPLE_RATE}
    if not path.is_file():
        return {**base, "available": False, "trained": False, "reason": f"No separation checkpoint found at {path.name}. Train and add one to enable separation."}
    try:
        checkpoint, _ = _read_checkpoint(path)
        is_trained = bool(checkpoint.get("trained", False)) and checkpoint.get("version") != "UNTRAINED-INIT"
        _load_cached(str(path), path.stat().st_mtime_ns, path.stat().st_size)
        return {**base, "available": True, "trained": is_trained, "reason": None, "version": separator_version(path)}
    except Exception as exc:
        return {**base, "available": False, "trained": False, "reason": f"The separation checkpoint could not be loaded: {exc}"}


def separator_version(path=SEPARATOR_PATH):
    checkpoint, _ = _read_checkpoint(path)
    label = str(checkpoint.get("version") or SEPARATOR_ARCHITECTURE)
    return f"{label}+{file_sha256(path)[:12]}"


@lru_cache(maxsize=1)
def _load_cached(path, modified_ns, size):
    checkpoint, config = _read_checkpoint(path)
    model = IterativeMaskSeparator(config["channels"], config["blocks"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_separator(path=SEPARATOR_PATH):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No separation checkpoint at {path}.")
    stat = path.stat()
    return _load_cached(str(path), stat.st_mtime_ns, stat.st_size)


def separate(waveform, k, path=SEPARATOR_PATH, residual_threshold=RESIDUAL_THRESHOLD, progress=None):
    """Peel up to `k` stems off a mono 16 kHz waveform.

    Returns dict(stems=[np.ndarray, ...] (each len(waveform)), residual_ratios=[...], stopped_early=bool,
    residual_energy_ratio=float). residual_ratios[i] is the residual/mixture energy before pass i.
    """
    k = int(k)
    if k < 1:
        raise ValueError("Source count must be at least 1.")
    model = load_separator(path)
    samples = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).to(DEVICE)
    length = samples.numel()
    window = torch.hann_window(N_FFT, device=DEVICE)
    spec = torch.stft(samples, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    mix_energy = float(torch.sum(samples ** 2)) + 1e-12
    residual = spec.clone()
    mix_logmag = torch.log1p(spec.abs()).unsqueeze(0)
    stems, ratios, stopped_early = [], [], False
    with torch.inference_mode():
        for i in range(k):
            res_time = torch.istft(residual, N_FFT, HOP_LENGTH, window=window, length=length)
            ratio = float(torch.sum(res_time ** 2)) / mix_energy
            if i > 0 and ratio < residual_threshold:
                stopped_early = True
                break
            ratios.append(ratio)
            mask = model(mix_logmag, torch.log1p(residual.abs()).unsqueeze(0), i, k)[0]
            source = residual * mask
            residual = residual - source
            stems.append(torch.istft(source, N_FFT, HOP_LENGTH, window=window, length=length).cpu().numpy())
            if progress:
                progress(i + 1, k)
        final = torch.istft(residual, N_FFT, HOP_LENGTH, window=window, length=length)
        final_ratio = float(torch.sum(final ** 2)) / mix_energy
    return {"stems": stems, "residual_ratios": ratios, "stopped_early": stopped_early, "residual_energy_ratio": final_ratio}
