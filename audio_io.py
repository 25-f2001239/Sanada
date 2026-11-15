"""Upload validation, decoding (mono downmix), resampling and WAV writing for the web API."""
import io
from pathlib import Path

import av
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

ALLOWED_UPLOAD_EXTENSIONS = {".wav", ".mp3", ".flac"}


class AudioValidationError(ValueError):
    """The upload is not acceptable (type, size, duration or content)."""


def sniff_container(path):
    """Cheap magic-byte check so a renamed file cannot masquerade as wav/mp3/flac."""
    with open(path, "rb") as handle:
        head = handle.read(12)
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return ".wav"
    if head[:4] == b"fLaC":
        return ".flac"
    if head[:3] == b"ID3" or (len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return ".mp3"
    return None


def validate_extension(filename):
    suffix = Path(str(filename).replace("\\", "/")).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        raise AudioValidationError("Unsupported file type. Upload WAV, MP3 or FLAC.")
    return suffix


def decode_mono(path, max_seconds):
    """Decode to mono float32 at the file's own sample rate (PyAV downmixes).

    Returns (samples, sample_rate). Raises AudioValidationError when the content is not
    audio, is empty, or is longer than max_seconds (checked while decoding, so an
    oversized file is rejected without decoding all of it).
    """
    kind = sniff_container(path)
    if kind is None:
        raise AudioValidationError("The file content is not WAV, MP3 or FLAC audio.")
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise AudioValidationError("The file contains no audio stream.")
            stream = container.streams.audio[0]
            rate = int(stream.rate or 0)
            if rate <= 0:
                raise AudioValidationError("The audio stream has no valid sample rate.")
            resampler = av.AudioResampler(format="fltp", layout="mono")
            limit = int(max_seconds * rate)
            chunks, total = [], 0
            for frame in container.decode(stream):
                for out in resampler.resample(frame):
                    block = out.to_ndarray()[0].astype(np.float32, copy=False)
                    chunks.append(block)
                    total += len(block)
                if total > limit:
                    raise AudioValidationError(f"The recording is longer than the {max_seconds:g} second limit.")
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
        raise AudioValidationError(f"The recording is longer than the {max_seconds:g} second limit.")
    samples = np.nan_to_num(np.concatenate(chunks).astype(np.float32, copy=False))
    return samples, rate


def resample(samples, source_rate, target_rate):
    """Polyphase resample of a mono float32 signal."""
    source_rate, target_rate = int(source_rate), int(target_rate)
    if source_rate == target_rate:
        return np.asarray(samples, dtype=np.float32)
    divisor = gcd(source_rate, target_rate)
    out = resample_poly(np.asarray(samples, dtype=np.float64), target_rate // divisor, source_rate // divisor)
    return out.astype(np.float32)


def peak_normalise(samples, peak=0.89):
    """Scale so the absolute peak equals `peak` (about -1 dBFS); silence stays silence."""
    samples = np.asarray(samples, dtype=np.float32)
    top = float(np.max(np.abs(samples))) if samples.size else 0.0
    if top <= 1e-9:
        return samples.copy()
    return np.clip(samples * (peak / top), -1.0, 1.0).astype(np.float32)


def wav_bytes(samples, sample_rate):
    """Encode mono float samples as 16-bit PCM WAV bytes."""
    buffer = io.BytesIO()
    sf.write(buffer, np.clip(samples, -1.0, 1.0), int(sample_rate), format="WAV", subtype="PCM_16")
    return buffer.getvalue()
