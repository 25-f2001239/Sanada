# SANADA: Audio Source Count Estimator - Architecture & Technical Guide

## 1. Project Overview
This project is a research prototype designed to estimate the number of overlapping audio sources (speakers, sounds) in a single-channel mixture. It uses a deep learning approach to classify audio segments into discrete source-count categories.

## 2. System Architecture

### A. High-Level Pipeline
`Audio File` $\rightarrow$ `Preprocessing` $\rightarrow$ `Feature Extraction` $\rightarrow$ `CNN Model` $\rightarrow$ `Source Count Prediction`

### B. Technical Stack
- **Frontend**: HTML5, CSS3, JavaScript (ES6+), WaveSurfer.js (for audio visualization).
- **Backend**: Flask (Python), Flask-CORS.
- **AI Framework**: PyTorch.
- **Signal Processing**: Librosa, NumPy, SoundFile.

---

## 3. AI Model Architecture

### A. The Model: `SourceCountCNN`
The model is a Convolutional Neural Network (CNN) that treats audio as an image (spectrogram).

**Layer Breakdown:**
1.  **Feature Extractor**:
    - 3 Convolutional Layers with BatchNorm and ReLU.
    - Max-pooling after the first two layers to reduce dimensionality.
    - **Adaptive Average Pooling**: Forces the final feature map to a fixed size $(4 \times 8)$ regardless of input length.
2.  **Classifier**:
    - Flatten layer $\rightarrow$ Fully Connected (128 units) $\rightarrow$ ReLU $\rightarrow$ Dropout (0.25) $\rightarrow$ Output Layer.
    - **Output**: A vector of size `max_sources`, where each index represents the probability of that specific count (e.g., index 0 = 1 source).

### B. Audio Feature Engineering
The model does not "listen" to raw waves. It uses **Log-Mel Spectrograms**:
- **Sample Rate**: 16,000 Hz.
- **Window**: 8-second analysis window.
- **Mel Bins**: 64 bins (capturing human-centric frequency perception).
- **Normalization**: Per-example Z-score normalization (mean=0, std=1) to make the model invariant to recording volume.

### C. Inference Logic (Sliding Window)
To handle different audio lengths and improve stability, the project implements a **Sliding Window Inference**:
- The audio is split into 2-second windows with a 1-second hop.
- The CNN predicts a count for each window.
- The final result is the **Mode** (most frequent prediction) across the entire timeline.

---

## 4. Dataset & Training Guide

### A. Dataset Requirements
- **Format**: Mono WAV files, 16kHz.
- **Labels**: A `metadata.csv` containing:
  - `mixture_path`: Path to the audio file.
  - `num_sources`: Integer representing the ground truth count.
  - `split`: `train`, `val`, or `test`.

### B. Training Process
The model is trained using **Cross-Entropy Loss** and the **AdamW** optimizer.
**Command to train:**
```powershell
python train.py --metadata data/metadata.csv --epochs 25 --max-sources 4
```

---

## 5. Implementation Details for Team Members

### Key Files
- `model.py`: Contains the CNN architecture and `predict_source_count` logic.
- `train.py`: The pipeline for training and saving checkpoints.
- `app.py`: The Flask API that bridges the AI model and the web UI.
- `static/app.js`: Handles the WaveSurfer integration and API calls.

### Modifying the AI Model
If team members wish to experiment with different architectures (e.g., replacing the CNN with a Conformer or LSTM):
1.  Modify the `SourceCountCNN` class in `model.py`.
2.  Ensure the `forward` method still outputs a vector of size `max_sources`.
3.  Update `predict_source_count` to match the new model's input/output requirements.

### Deployment Note
The app uses a `.pt` checkpoint file. Ensure that any new models are saved to `checkpoints/source_count_cnn.pt` for the web app to detect them automatically.
