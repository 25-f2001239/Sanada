# EchoCount architecture

EchoCount is a Flask application that estimates the number of overlapping sources in mono audio. Its CNN classifier operates on log-mel spectrograms and does not perform source separation.

## Request pipeline

1. The browser uploads a file to **POST /api/predict**.
2. Flask stores it under a unique temporary name and removes it when the request completes.
3. PyAV decodes WAV, FLAC, MP3, OGG, or M4A, downmixes to mono, and resamples to 16 kHz.
4. The audio is center-cropped or padded to eight seconds.
5. The full eight-second analysis clip is converted to one normalized 64-band log-mel spectrogram.
6. A PyTorch CNN predicts one source count for the full clip and reports its softmax confidence.

## Model and training

**model.py** owns the CNN and the complete feature pipeline used by both training and inference. Its adaptive pooling layer accepts the spectrogram frame dimensions generated for the full clip.

**train.py** reads **mixture_path,num_sources** and train, val, and test **split** labels from a CSV manifest. Each eight-second labelled mixture is used once, matching inference, and the best validation macro-MAE checkpoint is saved to **checkpoints/source_count_clip.pt**. The test split is excluded from model selection.

The old root-level **best_sanada_model.pth** contains YAMNet/MLP weights and is not compatible with this CNN. It is not used as an automatic fallback.

## Web application

- **app.py**: routes, health status, upload validation, JSON responses, cleanup.
- **templates/**: pages rendered by Flask/Jinja.
- **static/main.js**: shared navigation, theme, status, history storage.
- **static/analyze.js**: file preview, waveform, upload, and result rendering.
- **static/history.js**: local browser history and CSV export.
- **static/how.js**: explanatory illustrations.

## Deployment

The local Flask development server binds to **127.0.0.1**. Production runs through Gunicorn (Linux/container) or Waitress (Windows). The container listens on the **PORT** environment variable and needs a compatible checkpoint mounted at the configured **MODEL_PATH**.
