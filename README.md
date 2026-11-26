# EchoCount — Audio Source Counting, Sound Naming & Separation

EchoCount is a Flask + PyTorch audio analysis web app. Upload a recording (WAV, MP3, or FLAC) and EchoCount:
1. **Counts** how many distinct overlapping sources are present (`SourceCountCNN` with Sparse Specialist routing $K \le 8$, isotonic calibration, and 30-pass MC-dropout uncertainty).
2. **Names every sound** with friendly labels (`"Hammer"`, `"Phone vibration"`, `"Dog bark"`, `"Cat meow"`, `"Cello"`, `"Scissors"`, `"Telephone ring"`, `"Human voice"`, `"Door knock"`, `"Keyboard typing"`, `"Water"`, `"Car horn"`, plus AudioSet-527 classes), confidence scores, and frame-level time ranges.
3. **Separates each named sound** into its own stem plus an exact **`Everything else (Residual)`** stem (`mixture − sum(stems)`), verified by three measured quality gates (waveform correlation $< 0.90$, energy conservation, and self-consistency re-tagging) and downloadable as individual WAVs or a ZIP.

> _Screenshot placeholder: add `docs/screenshot.png` of the `/analyze` UI here._

---

## 1. Quick Setup (One Command)

**Requirements:** Python **3.10+** (verified on Python `3.12` / `3.14`).

```powershell
git clone https://github.com/25-f2001239/Sanada.git
cd Sanada
python setup.py
```

`setup.py` automatically creates `.venv`, installs `requirements.txt`, creates `.env` with a fresh `SECRET_KEY`, downloads all required pretrained models (`ast-audioset`, `clap-htsat-unfused`, `spkrec-ecapa-voxceleb`, and 2-, 3-, and 4-speaker SepFormer checkpoints), verifies checkpoint hashes, initializes the auth DB, and runs a health self-check.

To set up and launch immediately in one step:
```powershell
python setup.py --run
# Open http://127.0.0.1:5000
```

### Run the App (after setup)
```powershell
.\.venv\Scripts\python.exe app.py
# Open http://127.0.0.1:5000
```

### Optional GPU Build of PyTorch (CUDA)
`requirements.txt` installs CPU `torch==2.15.0` by default. On an NVIDIA GPU machine:
```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Manual Model Check / Download
```powershell
.\.venv\Scripts\python.exe scripts\download_models.py
.\.venv\Scripts\python.exe scripts\download_models.py --check
```

---

## 2. Tests, Training & Evaluation

```powershell
# Run product unit, route, and 5-clip regression tests
.\.venv\Scripts\python.exe test_templates.py --full

# Verify golden baseline (requires dataset & local checkpoints)
.\.venv\Scripts\python.exe tests\golden_check.py --verify

# Unified training CLI (--smoke or --full)
.\.venv\Scripts\python.exe scripts\train.py counter --smoke
.\.venv\Scripts\python.exe scripts\train.py all --full

# Unified evaluation & latency profiler (--smoke or --full)
# Regenerates evaluation/results.json (ignored from git due to size):
.\.venv\Scripts\python.exe scripts\evaluate.py counter --full
.\.venv\Scripts\python.exe scripts\evaluate.py all --full

# Export committed aggregate-only stats for the Home and How-it-works pages:
.\.venv\Scripts\python.exe scripts\evaluate.py --export-public
```

---

## 3. Measured Benchmarks

All figures below are read directly from `evaluation/results.json` (`SourceCountCNN-v3-speech-fsd-specialist`, `epoch=11`, `sha256: e003d018...`), `evaluation/product_benchmarks.json`, and `evaluation/separation_results.json`. (`evaluation/results.json` is excluded from git; regenerate it with `python scripts/evaluate.py counter --full`.)

- **Source Counting (`evaluation/results.json`, `n=1,285` test mixtures):**
  - Overall Test: `MAE = 1.6451`, `Exact = 43.8%`, `Within ±1 = 64.3%`.
  - Speech Test (`n=695`): `MAE = 0.8014`, `Exact = 53.1%`, `Within ±1 = 78.9%`.
  - Sparse Counts (Overall Test): `K=2` (`n=248`): `77.8%` exact (`99.2%` within ±1); `K=3` (`n=248`): `73.0%` exact (`96.0%` within ±1); `K=4` (`n=148`): `54.1%` exact (`90.5%` within ±1); `K=6` (`n=115`): `70.4%` exact (`93.9%` within ±1).
  - **Held-Out Unseen Speakers (`Speaker_4_1`, `Speaker8`, `Speaker9`, `n=352`)**: `Exact = 13.1%` (`0.1307`), `Within ±1 = 29.6%` (`0.2955`), `MAE = 2.054`. *(Note: an earlier draft cited `69.9%` / `0.369`, which did not match `evaluation/results.json` for `epoch=11`; `13.1%` exact / `2.054` MAE is the actual measured result on unseen speakers.)*
- **Tagging (`evaluation/results.json` & `evaluation/product_benchmarks.json`):**
  - 32-Class FSD Hierarchical Tagger (`checkpoints/tagger.pt`, `epoch=8`): Group Test mAP = `0.9926`, Fine 32-Class Test mAP = `0.8052`, Test F1 = `0.7612`.
  - Open-Vocabulary 16-Class FSD50K Held-Out Split (`96` clips): AST+CLAP mAP = `0.9068`, Combined Open-Tagger mAP = `0.6979`, F1 = `0.4317`.
- **Separation (`evaluation/separation_results.json`, `32` held-out synthetic mixtures `K=2..5`, `112` stems):**
  - Mean SI-SDRi = `+2.51 dB`, Self-Consistency Pass Rate = `39.3%`, Energy Conservation = `100.0%`.
- **Latency (`evaluation/product_benchmarks.json`, CPU):**
  - `8 s` clip: `analyze = 1.739 s`, `query_separation = 1.065 s`.
  - `60 s` clip: `analyze = 1.707 s`, `query_separation = 10.188 s`.

---

## 4. Project Structure, Licences & Privacy

- See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the file map, request pipeline diagram, and `ECHOCOUNT_DEV` mode details.
- **Licences:** EchoCount code is licensed under the **MIT License** ([LICENSE](LICENSE)). Licences differ per pretrained model; see **[THIRD_PARTY.md](THIRD_PARTY.md)**. Some models were trained on datasets with research-only terms (for example WSJ0 for the SepFormer separator).
- **Datasets:** Not included in git (`FSD_Mix_*` and `Org_Mix_*` live outside the repo; FSD50K isolated clips come from `Fhrozen/FSD50k` / Zenodo `4060432`; real-room recording protocol is in [data/real_eval/RECORDING_GUIDE.md](data/real_eval/RECORDING_GUIDE.md)).
- **Privacy:** No user audio or recordings of people are stored in the repository. Uploaded audio, stems, and ZIPs are processed in isolated `uploads/jobs/` directories deleted automatically after `JOB_TTL_SECONDS` (default 30 minutes) and on server startup. `evaluation/public_summary.json` contains aggregated metrics only (no filenames, paths, or hashes).

---

## 5. Limitations (Measured)

1. **Dense Mixtures ($K > 6$)**: Counting accuracy drops to `0.0%` exact at `K=9` (`MAE = 2.809`); EchoCount warns `"Too many overlapping sounds to separate reliably"` and disables separation.
2. **Unseen-Speaker Generalization**: On held-out unseen speakers (`n=352`), exact count accuracy is `13.1%` (`MAE = 2.054`), showing a measurable domain/speaker generalization gap compared to seen-speaker mixtures (`53.1%` exact).
3. **Speech Separation ($K \le 4$)**: Native pretrained SepFormer checkpoints support 2 speakers (`sepformer-wsj02mix`), 3 speakers (`sepformer-wsj03mix`), and 4 speakers (`sepformer-libri4mix`); pure speech mixtures with $K \ge 5$ disable separation.
4. **Speaker Clustering ($K \le 3$)**: Windowed ECAPA-TDNN clustering is shown only for $K \le 3$ because simultaneous overlap blends 1-second window embeddings.
5. **Co-Channel Broadband Overlap**: Non-speech sounds sharing identical frequency bands and timestamps share TF bins and may be flagged `"Low confidence"` by the self-consistency re-tagging gate (`39.3%` pass rate across `K=2..5` mixtures).
