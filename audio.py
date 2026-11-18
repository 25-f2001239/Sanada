"""Audio validation, PyAV decoding, polyphase resampling, WAV encoding, and log-mel features."""

from __future__ import annotations

import hashlib
import io
from math import gcd
from pathlib import Path

import av
import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from config import CLIP_SAMPLES, HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE

ALLOWED_UPLOAD_EXTENSIONS = {".wav", ".mp3", ".flac"}


class AudioValidationError(ValueError):
    """Raised when an uploaded audio file fails format, size, or duration checks."""


def file_sha256(path: Path | str, chunk: int = 1 << 20) -> str:
    """Return hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sniff_container(path: Path | str) -> str | None:
    """Inspect magic bytes to verify WAV, FLAC, or MP3 audio content."""
    with open(path, "rb") as handle:
        head = handle.read(12)
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return ".wav"
    if head[:4] == b"fLaC":
        return ".flac"
    if head[:3] == b"ID3" or (len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return ".mp3"
    return None


def validate_extension(filename: str) -> str:
    """Validate upload filename extension against ALLOWED_UPLOAD_EXTENSIONS."""
    suffix = Path(str(filename).replace("\\", "/")).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        raise AudioValidationError("Unsupported file type. Upload WAV, MP3 or FLAC.")
    return suffix


def decode_mono(
    path: Path | str, max_seconds: float, target_rate: int | None = None
) -> tuple[np.ndarray, int]:
    """Decode audio to mono float32, enforcing max_seconds during streaming decode."""
    if sniff_container(path) is None:
        raise AudioValidationError("The file content is not WAV, MP3 or FLAC audio.")
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise AudioValidationError("The file contains no audio stream.")
            stream = container.streams.audio[0]
            src_rate = int(stream.rate or 0)
            if src_rate <= 0:
                raise AudioValidationError("The audio stream has no valid sample rate.")
            out_rate = int(target_rate or src_rate)
            resampler = (
                av.AudioResampler(format="fltp", layout="mono", rate=out_rate)
                if target_rate
                else av.AudioResampler(format="fltp", layout="mono")
            )
            limit = int(max_seconds * out_rate)
            chunks, total = [], 0
            for frame in container.decode(stream):
                for out in resampler.resample(frame):
                    block = out.to_ndarray()[0].astype(np.float32, copy=False)
                    chunks.append(block)
                    total += len(block)
                if total > limit:
                    raise AudioValidationError(
                        f"The recording is longer than the {max_seconds:g} second limit."
                    )
            for out in resampler.resample(None):
                block = out.to_ndarray()[0].astype(np.float32, copy=False)
                chunks.append(block)
                total += len(block)
    except AudioValidationError:
        raise
    except Exception as exc:
        raise AudioValidationError(f"Could not decode the audio file: {exc}") from exc
    if total == 0:
        raise AudioValidationError("The audio file is empty or could not be decoded.")
    if total > limit:
        raise AudioValidationError(
            f"The recording is longer than the {max_seconds:g} second limit."
        )
    return np.nan_to_num(np.concatenate(chunks).astype(np.float32, copy=False)), (
        src_rate if target_rate else out_rate
    )


def load_audio(path: Path | str) -> tuple[np.ndarray, dict[str, float | int]]:
    """Decode, resample to 16 kHz mono, and center-crop or pad to 8 seconds."""
    waveform, orig_sr = decode_mono(path, max_seconds=600.0, target_rate=SAMPLE_RATE)
    orig_dur = len(waveform) / SAMPLE_RATE
    if len(waveform) > CLIP_SAMPLES:
        start = (len(waveform) - CLIP_SAMPLES) // 2
        crop_start = start / SAMPLE_RATE
        waveform = waveform[start : start + CLIP_SAMPLES]
    else:
        crop_start = 0.0
        waveform = np.pad(waveform, (0, CLIP_SAMPLES - len(waveform)))
    return waveform.astype(np.float32, copy=False), {
        "original_sample_rate": orig_sr,
        "original_duration_seconds": orig_dur,
        "crop_start_seconds": crop_start,
    }


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Polyphase resample of a mono float32 waveform."""
    source_rate, target_rate = int(source_rate), int(target_rate)
    if source_rate == target_rate:
        return np.asarray(samples, dtype=np.float32)
    div = gcd(source_rate, target_rate)
    return resample_poly(
        np.asarray(samples, dtype=np.float64), target_rate // div, source_rate // div
    ).astype(np.float32)


def peak_normalise(samples: np.ndarray, peak: float = 0.89) -> np.ndarray:
    """Scale waveform so absolute peak equals `peak`, leaving silence unchanged."""
    arr = np.asarray(samples, dtype=np.float32)
    top = float(np.max(np.abs(arr))) if arr.size else 0.0
    return arr.copy() if top <= 1e-9 else np.clip(arr * (peak / top), -1.0, 1.0).astype(np.float32)


def wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode mono float samples as 16-bit PCM WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, np.clip(samples, -1.0, 1.0), int(sample_rate), format="WAV", subtype="PCM_16")
    return buf.getvalue()


def waveform_to_logmel(waveform: np.ndarray) -> torch.Tensor:
    """Convert an 8-second 16 kHz waveform to a normalized [1, 64, T] log-mel tensor."""
    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if len(samples) < CLIP_SAMPLES:
        samples = np.pad(samples, (0, CLIP_SAMPLES - len(samples)))
    elif len(samples) > CLIP_SAMPLES:
        start = (len(samples) - CLIP_SAMPLES) // 2
        samples = samples[start : start + CLIP_SAMPLES]
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
    normalized = np.clip(librosa.power_to_db(mel, ref=np.max, top_db=80.0) / 80.0, -1.0, 0.0)
    return torch.from_numpy(normalized.astype(np.float32, copy=False)).unsqueeze(0)
