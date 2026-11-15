# EchoCount — Source Counting, Open-Vocabulary Sound Naming & Query-Based Audio Separation

EchoCount is a scalable audio analysis and source separation product. Given a recording containing a mixture of sounds (for example a hammer, a phone vibrating, a dog barking, musical instruments, or overlapping human voices), EchoCount:
1. **Counts** how many distinct overlapping sources are present (`SourceCountCNN` with Sparse Specialist routing $K \le 8$, validation-only isotonic calibration, and MC-dropout + softmax uncertainty decomposition).
2. **Names every sound** (`open_tagger.py`) with user-facing friendly labels (`"Hammer"`, `"Phone vibration"`, `"Dog bark"`, `"Cat meow"`, `"Cello"`, `"Scissors"`, `"Telephone ring"`, `"Human voice"`, `"Door knock"`, `"Keyboard typing"`, `"Water"`, `"Car horn"`, plus all 32 FSD classes and 527 AudioSet classes), confidence scores, frame-level time ranges (`start_s–end_s`), and co-occurring overlap detection.
3. **Separates each named sound** (`query_separation.py`, `speech_separation.py`) into its own labelled audio stem plus an exact **`Everything else (Residual)`** stem (`mixture − sum(stems)`), verified by three measured quality gates (self-consistency re-tagging, waveform correlation $< 0.90$, and energy conservation) and downloadable individually as WAV or together as a ZIP with `manifest.json`.

---

## 1. Pretrained Models, Datasets & Licences Table

Every pretrained model and dataset used by EchoCount has a permissive licence that allows commercial use and runs 100% offline on a laptop (16 GB RAM, CPU or NVIDIA RTX 4050 GPU):

| Component / Model | Repository / Path | Licence | Commercial Use Allowed? | Size on Disk | Offline Laptop Ready? |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Source Counter (General + Sparse Specialist $K \le 8$)** | `checkpoints/source_count_clip.pt` | MIT (Project-trained) | **Yes** | `5.8 MB` | **Yes** |
| **Domain Hierarchical Tagger (5 groups, 32 FSD classes)** | `checkpoints/tagger.pt` | MIT (Project-trained) | **Yes** | `4.2 MB` | **Yes** |
| **AudioSet 527-Class Audio Spectrogram Transformer (AST)** | `MIT/ast-finetuned-audioset-10-10-0.4593` (`models/ast-audioset`) | **BSD-3-Clause** | **Yes** | `346 MB` | **Yes** |
| **CLAP Open-Vocabulary Audio-Text Model** | `laion/clap-htsat-unfused` (`models/clap-htsat-unfused`) | **Apache-2.0 / MIT** | **Yes** | `615 MB` | **Yes** |
| **2-Speaker Speech Separator (SepFormer)** | `speechbrain/sepformer-wsj02mix` (`models/sepformer-wsj02mix`) | **Apache-2.0** | **Yes** | `113 MB` | **Yes** |
| **Speaker Embedding Verifier (ECAPA-TDNN)** | `speechbrain/spkrec-ecapa-voxceleb` (`models/spkrec-ecapa-voxceleb`) | **Apache-2.0** | **Yes** | `83 MB` | **Yes** |
| **Iterative Mask Separator (Trained on FSD50K Isolated Clips)** | `checkpoints/separator_iterative.pt` | MIT (Project-trained) | **Yes** | `0.8 MB` | **Yes** |
| **FSD50K Isolated Single-Source Subset (16 classes, 480 clips)** | `Fhrozen/FSD50k` / Zenodo `4060432` (`data/isolated_sources/`) | **CC0 / CC-BY-4.0** | **Yes** | `304 MB` | **Yes** |

---

## 2. Setup & Model Downloads

### 1. Python Virtual Environment (Windows PowerShell)
Verified on **Python 3.14.8** (Python 3.11+ supported):

```powershell
cd C:\coding\SANADA\Sanada
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Optional GPU Build of PyTorch (CUDA)
`requirements.txt` installs standard `torch==2.15.0` without hard-coding a CUDA wheel URL. To install a CUDA-enabled build on an NVIDIA GPU machine:

```powershell
.\.venv\Scripts\python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. Download Pretrained Models (`models/`)
All pretrained weights are stored locally in `models/` for offline execution:

```powershell
cd C:\coding\SANADA\Sanada
.\.venv\Scripts\python -c "from huggingface_hub import snapshot_download; snapshot_download('speechbrain/sepformer-wsj02mix', local_dir='models/sepformer-wsj02mix'); snapshot_download('speechbrain/spkrec-ecapa-voxceleb', local_dir='models/spkrec-ecapa-voxceleb'); snapshot_download('MIT/ast-finetuned-audioset-10-10-0.4593', local_dir='models/ast-audioset', ignore_patterns=['*.bin', '*.md']); snapshot_download('laion/clap-htsat-unfused', local_dir='models/clap-htsat-unfused', ignore_patterns=['*.md'])"
```

### 4. Download FSD50K Isolated Single-Source Clips (`data/isolated_sources/`)
To download 30 isolated clips per class across 16 target classes (`480` clips, `304 MB`, `CC0`/`CC-BY`) into `data/isolated_sources/<Class>/`:

```powershell
cd C:\coding\SANADA\Sanada
.\.venv\Scripts\python download_fsd50k_subset.py --clips-per-class 30
```

---

## 3. Running the Application, Benchmarks & Tests

```powershell
cd C:\coding\SANADA\Sanada

# Run unit tests, taxonomy single-source-of-truth test, quality gates, and 5-clip regression suite
.\.venv\Scripts\python test_templates.py --full

# Run full product benchmark (Tagger comparison, Held-out synthetic SI-SDRi, Real-eval, 8s & 60s latency profile)
.\.venv\Scripts\python evaluate_product_benchmarks.py

# Start the Flask production server (http://127.0.0.1:5000)
.\.venv\Scripts\python app.py
```

### Docker Deployment (CPU and GPU)

```powershell
# CPU Container
docker build -t echocount:latest .
docker run --rm -p 8080:8080 -e ECHOCOUNT_DEV=0 echocount:latest

# NVIDIA GPU Container
docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime -t echocount:gpu .
docker run --rm --gpus all -p 8080:8080 -e ECHOCOUNT_DEV=0 echocount:gpu
```

---

## 4. Measured Benchmarks

### A. Source Counting (`evaluation/results.json`)
- **Validation MAE**: `1.605` | **Overall Test Exact Accuracy**: `43.8%` | **Speech Test Exact Accuracy**: `53.1%`
- **Sparse Regime ($K \le 8$) Test Exact Accuracy**:
  - $K=2$: **`77.8%`** exact (`96.0%` within ±1)
  - $K=3$: **`73.0%`** exact (`96.6%` within ±1)
  - $K=4$: **`62.0%`** exact (`91.0%` within ±1)
  - $K=6$: **`89.0%`** exact (`96.0%` within ±1)
- **Held-Out Unseen Speaker Split (`Speaker_4_1`, `Speaker8`, `Speaker9`)**: MAE = `0.369`, Exact Accuracy = `69.9%`.

### B. Open-Vocabulary Sound Naming (`evaluation/product_benchmarks.json`)
- **12 Target Everyday Sounds Coverage**: `12/12 (100%)` (`Hammer`, `Phone vibration`, `Dog bark`, `Cat meow`, `Cello`, `Scissors`, `Telephone ring`, `Human voice`, `Door knock`, `Keyboard typing`, `Water`, `Car horn`).
- **32-Class FSD Hierarchical Tagger (`checkpoints/tagger.pt`)**: Coarse Group Test mAP = **`0.9926`**, Fine 32-Class Test mAP = **`0.8052`**.
- **Open-Vocabulary 16-Class Held-Out FSD50K Test Split**: Combined `AST-AudioSet-527 + CLAP + 32-CNN` outperforms either model alone by using the domain 32-CNN on in-vocabulary FSD classes and `AST-527 + CLAP` on open-vocabulary classes (`Hammer`, `Phone vibration`, `Door knock`, `Water`, `Car horn`, `Human voice`).

### C. Separation & Quality Gates (`evaluation/separation_results.json`)
- **2-Speaker Speech Separation (`sepformer-wsj02mix` + `spkrec-ecapa-voxceleb`)**: Pairwise stem correlation = `0.118` ($< 0.90$ threshold), ECAPA speaker embedding cosine similarity = `0.494` ($\le 0.82$ threshold, verified distinct speakers).
- **Multi-Sound Query-Conditioned Separation**:
  - Exact energy conservation (`sum(stems) + residual == mixture`, relative L2 error $< 10^{-5}$, reconstruction SNR $> 100\text{ dB}$).
  - Long-audio support via overlapping `8.0 s` windows with `2.0 s` raised-cosine (Hann) cross-fade stitching.

---

## 5. Known Limitations (Measured)

1. **Dense Mixtures ($K > 6$)**:
   - Measured counting exact accuracy drops from `89.0%` at $K=6$ to `0.0%` at $K=9$ (`evaluation/results.json`). For recordings with estimated $K > 6$, EchoCount displays `"Too many overlapping sounds to separate reliably"` and disables separation rather than producing degraded stems.
2. **Overlapping Speech Beyond 2 Speakers ($K \ge 3$)**:
   - The installed pretrained speech separator (`speechbrain/sepformer-wsj02mix`) supports up to **2 simultaneous speakers**. For $K \ge 3$ pure speech mixtures without `sepformer-wsj03mix`, EchoCount reports `"Separation is supported for up to 2 speakers with the installed model"`.
3. **Windowed Speaker Clustering Limit ($K \le 3$)**:
   - ECAPA-TDNN speaker clustering operates on `1.0 s` windows (`0.5 s` hop). When more than 3 speakers talk simultaneously in every window, single-window embeddings blend multiple voices; therefore the Distinct Voices card is shown only for $K \le 3$.
4. **Co-Channel Broadband Overlap in Non-Speech Separation**:
   - Query-conditioned spectral-temporal + HPSS Wiener masking (`query_separation.py`) separates sounds with distinct spectral-temporal or harmonic/percussive profiles (e.g. `Cello` + `Scissors` + `Telephone ring`, or `Hammer` + `Human voice`) and guarantees nothing is lost via the `Everything else (Residual)` stem, but two non-speech sounds with identical frequency bands occurring at the exact same millisecond share time-frequency bins and may be flagged `"Low confidence"` by the self-consistency re-tagging gate.
