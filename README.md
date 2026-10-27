# EchoCount — Audio Source Count Estimator

A starter full-stack research prototype based on the supplied paper, **“SANADA: A Reproducible Synthetic Acoustic Dataset for Intractable Mixtures, Evaluated with Baseline Models in Separation, Counting, and Tagging.”**

The app includes:
- **Frontend:** responsive HTML/CSS/JavaScript interface for uploading audio and displaying results.
- **Backend:** Flask routes for health checks and audio inference.
- **AI model:** PyTorch CNN that classifies a log-mel spectrogram into a source-count class.
- **Training pipeline:** `train.py` trains the model from a CSV manifest containing mixture paths and known source counts.

## Important: no dataset / no trained weights yet

The UI and backend can run before the dataset arrives. The model architecture and training pipeline are included, but there is **no trained model checkpoint** in this starter project. Until you train it, the app deliberately reports “Model is not trained yet” instead of inventing an AI prediction.

A source-count model predicts how many sources are present; it does **not** separate those sources into individual audio files. Separation would be a separate model and project component.

## 1. Requirements

- Python 3.10 or 3.11 recommended
- A terminal (PowerShell is fine)
- Internet access for installing Python packages
- CPU works for development; a CUDA-compatible GPU can speed up training

## 2. Setup on Windows / PowerShell

Extract the ZIP, then open a terminal in the extracted `audio_source_count_estimator` folder.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this in the current terminal session:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Start the app:
```powershell
python app.py
```

Open `http://127.0.0.1:5000` in your browser. The app is intended for local development, not public deployment.

## 3. Dataset format when you receive it

The paper describes single-channel, 16 kHz, 16-bit PCM WAV audio with 8-second mixtures and ground-truth source counts. It also describes a metadata manifest with fields such as `mixture_path`, `num_sources`, and `source_files`.

For this starter trainer, create a CSV named `metadata.csv` with at least:

```csv
mixture_path,num_sources,split
audio/mix_0001.wav,2,train
audio/mix_0002.wav,3,train
audio/mix_0003.wav,2,val
audio/mix_0004.wav,4,test
```

- `mixture_path`: path to the mixture WAV file, relative to the CSV location or absolute.
- `num_sources`: the ground-truth count (integer).
- `split`: optional; use `train`, `val`, or `test`. If present, the trainer uses `train` and `val`; the test split is reserved for your final evaluation.
- The paths above are examples only. They are not real audio files.

The paper's variable-K setup includes organisational speech mixtures up to 9 speakers and environmental sound mixtures up to 32 sources/classes. Set `--max-sources` to the maximum count represented in your data. The starter defaults to 32 output classes.

**Avoid data leakage:** split original source recordings before mixing where possible, as the paper describes. Do not randomly split near-duplicate mixtures from the same stems across train and test. Keep the final test split untouched until evaluation.

## 4. Train the AI model

After the audio files and CSV are in place, from this project directory run:

```powershell
python train.py --metadata data\metadata.csv --epochs 25 --max-sources 32
```

The script saves the best validation-loss checkpoint to:
`checkpoints/source_count_cnn.pt`

The app automatically detects this checkpoint and enables inference on the next request. If your dataset contains counts only up to 9, use `--max-sources 9`.

Expected training data requirements:
- Every row must point to an existing labelled mixture.
- `num_sources` must be an integer between 1 and `--max-sources`.
- There must be both training and validation rows.
- Enough examples per count are needed for useful results. A tiny dataset is suitable only for testing the pipeline, not for reliable research results.

## 5. API endpoints

### `GET /api/health`
Returns whether the server is up and whether a trained checkpoint is present.

### `POST /api/predict`
Multipart form upload with field name `audio`. Returns audio metadata and, when trained, the predicted source count, confidence, and top three count classes.

Example:
```bash
curl -X POST -F "audio=@example.wav" http://127.0.0.1:5000/api/predict
```

## 6. Model design

1. Load audio and convert to mono.
2. Resample to 16 kHz.
3. Crop/pad to an 8-second analysis window.
4. Calculate a 64-bin log-mel spectrogram.
5. Feed the spectrogram to a small CNN.
6. Predict one of the count classes using cross-entropy loss.

This is a **baseline starter architecture**, not a reproduction of the paper's Conformer regressor. The paper reports a Conformer-based regressor for source counting. The included CNN is intentionally lighter and easier to train/modify as a first working implementation. You can later replace it with a Conformer to align more closely with the paper.

## 7. Evaluation checklist

When the dataset arrives, evaluate on the held-out `test` split and report:
- Exact count accuracy
- Mean absolute error (MAE)
- RMSE
- Relaxed accuracy within ±1 source
- Per-count results / confusion matrix

Do not treat training accuracy as proof that the model works. High-density mixtures can be much harder to count than low-density mixtures, as the paper discusses.

## 8. Project structure

```text
audio_source_count_estimator/
├── app.py
├── model.py
├── train.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── app.js
├── data/                 # add dataset and metadata CSV here later
├── checkpoints/          # trained weights saved here
└── uploads/              # temporary uploads; files are removed after requests
```

## 9. Current limitations

- The model is not trained until you add labelled data and run `train.py`.
- It estimates one count for an 8-second window. Long recordings are center-cropped; short recordings are padded.
- Confidence is the neural network's softmax score, not a guarantee of correctness.
- Source counting and source separation are different tasks.
- The supplied paper is identified as a preprint in the PDF; its reported numbers should be treated accordingly.
