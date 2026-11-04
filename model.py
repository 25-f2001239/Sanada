import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import os
import tensorflow_hub as hub

# --- MODEL ARCHITECTURE ---
class SourceCountMLP(nn.Module):
    def __init__(self, input_dim=1024, num_classes=32):
        super(SourceCountMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# --- CONSTANTS ---
MODEL_PATH = 'best_sanada_model.pth'
SAMPLE_RATE = 16000
CLIP_SECONDS = 8.0

# Global YAMNet model for fast inference in Flask
_YAMNET_MODEL = None

def load_yamnet():
    global _YAMNET_MODEL
    if _YAMNET_MODEL is None:
        _YAMNET_MODEL = hub.load('https://tfhub.dev/google/yamnet/1')
    return _YAMNET_MODEL

def load_audio(path):
    """Loads audio, resamples, and normalizes to match YAMNet/MLP pipeline."""
    wav, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    # Standardize length
    fixed_length = int(SAMPLE_RATE * CLIP_SECONDS)
    if len(wav) < fixed_length:
        wav = np.pad(wav, (0, fixed_length - len(wav)))
    else:
        wav = wav[:fixed_length]

    return wav, sr

def predict_source_count(waveform):
    """
    Takes a raw waveform, extracts YAMNet embeddings, and predicts count
    with confidence and agreement metrics.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SourceCountMLP().to(device)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found.")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    yamnet = load_yamnet()
    wav_tensor = waveform.astype(np.float32)

    # Run YAMNet inference: scores, embeddings, spectrogram
    _, embeddings, _ = yamnet(wav_tensor)

    # Ensure embeddings is a numpy array and not a tensor before converting to torch
    if hasattr(embeddings, 'numpy'):
        embeddings = embeddings.numpy()

    # Process every window instead of averaging first
    tensor_embs = torch.FloatTensor(embeddings).to(device)

    with torch.no_grad():
        logits = model(tensor_embs)
        probs = F.softmax(logits, dim=1)

        # Prediction and confidence per window
        window_preds = torch.argmax(probs, dim=1).cpu().numpy()
        window_confidences = torch.max(probs, dim=1)[0].cpu().numpy()

    # Final prediction via majority vote
    counts = np.bincount(window_preds, minlength=32)
    winning_idx = np.argmax(counts)
    final_count = winning_idx + 1

    # Create timeline for the frontend
    timeline = []
    for i in range(num_windows):
        timeline.append({
            "start": i * 0.96,
            "end": (i + 1) * 0.96,
            "count": int(window_preds[i] + 1),
            "confidence": float(window_confidences[i])
        })

    # Calculate Metrics
    voted_windows = counts[winning_idx]
    agreement = (voted_windows / num_windows) * 100
    avg_confidence = np.mean(window_confidences) * 100

    return {
        "estimated_sources": int(final_count),
        "confidence": float(avg_confidence),
        "agreement": float(agreement),
        "windows_voted": int(voted_windows),
        "total_windows": num_windows,
        "timeline": timeline
    }
