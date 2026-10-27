"""CNN source-count model. Training requires labelled mixture audio."""
import os
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "checkpoints" / "source_count_cnn.pt"
SAMPLE_RATE = 16000
CLIP_SECONDS = 8.0
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
MAX_SOURCES = int(os.getenv("MAX_SOURCES", "32"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SourceCountCNN(nn.Module):
    """Predicts a discrete source count from a log-mel spectrogram."""
    def __init__(self, max_sources=MAX_SOURCES, n_mels=N_MELS):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 8)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 8, 128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, max_sources)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_audio(path):
    """Load audio, convert to mono/16 kHz, and return a fixed 8-second window."""
    # librosa supports common compressed formats as well as WAV/FLAC.
    audio, sr = librosa.load(str(path), sr=None, mono=True, dtype=np.float32)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    target_len = int(SAMPLE_RATE * CLIP_SECONDS)
    if len(audio) > target_len:
        # Center crop for long recordings, rather than silently taking only the beginning.
        start = max(0, (len(audio) - target_len) // 2)
        audio = audio[start:start + target_len]
    elif len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    return audio.astype(np.float32), sr


def waveform_to_logmel(waveform):
    mel = librosa.feature.melspectrogram(
        y=np.asarray(waveform, dtype=np.float32),
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=40,
        fmax=SAMPLE_RATE // 2,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    # Per-example normalisation makes the representation less sensitive to overall gain.
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    return torch.tensor(logmel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def predict_source_count(waveform):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    max_sources = int(checkpoint.get("max_sources", MAX_SOURCES))
    model = SourceCountCNN(max_sources=max_sources).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # To provide a timeline, we analyze the audio in sliding windows
    # Window size: 2 seconds, Hop size: 1 second
    window_size = int(SAMPLE_RATE * 2.0)
    hop_size = int(SAMPLE_RATE * 1.0)

    timeline = []

    # Ensure waveform is long enough for at least one window
    if len(waveform) < window_size:
        # Pad for very short audio
        waveform = np.pad(waveform, (0, window_size - len(waveform)))

    for start in range(0, len(waveform) - window_size + 1, hop_size):
        end = start + window_size
        chunk = waveform[start:end]

        # Process chunk to log-mel
        features = waveform_to_logmel(chunk).to(DEVICE)

        with torch.no_grad():
            logits = model(features)
            probs = torch.softmax(logits, dim=1)[0]
            confidence, idx = torch.max(probs, dim=0)

        timeline.append({
            "start": start / SAMPLE_RATE,
            "end": end / SAMPLE_RATE,
            "count": int(idx.item()) + 1,
            "confidence": round(float(confidence.item()), 4)
        })

    # Final aggregate prediction (the most frequent count in the timeline)
    counts = [t["count"] for t in timeline]
    final_count = max(set(counts), key=counts.count) if counts else 1

    # Top-k for the overall file (simplified as average of timeline)
    # For the "top predictions" in the UI, we'll use the final_count as most likely
    return {
        "estimated_sources": final_count,
        "timeline": timeline,
        "model_type": "CNN sliding-window classifier",
    }
