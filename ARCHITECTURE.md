# EchoCount Architecture

EchoCount is a Flask + PyTorch audio analysis product that counts overlapping sound sources, names each detected sound with frame-level time ranges, and separates each sound into its own stem plus an exact residual stem.

## 1. File Map (Core Source)

| File | Responsibility |
| :--- | :--- |
| [config.py](config.py) | Paths, environment settings, `config.json` & `taxonomy.json` single-source-of-truth helpers |
| [audio.py](audio.py) | Upload sniffing, PyAV mono decode, polyphase resampling, WAV encoding, 64-band log-mel extraction |
| [data.py](data.py) | Dataset scanning, leakage audit, and dev-gated (`ECHOCOUNT_DEV=1`) SHA-256 ground-truth lookup |
| [counting.py](counting.py) | `SourceCountCNN`, sparse specialist ($K \le 8$) routing, isotonic calibration, batched MC-dropout uncertainty |
| [tagging.py](tagging.py) | 32-class `GroupFineTaggerCNN` + AudioSet-527 AST + CLAP zero-shot + sliding-window time localization |
| [separation.py](separation.py) | `IterativeMaskSeparator`, SpeechBrain SepFormer + ECAPA speech separation, query-conditioned Wiener separation, 8s/2s windowed cross-fade stitching |
| [quality.py](quality.py) | Stem quality gates: waveform correlation ($< 0.90$), energy conservation, and self-consistency re-tagging |
| [app.py](app.py) | Flask routes (`/api/public-stats`, `/api/analyze`, `/api/separate`), rate limiting, async separation job queue, startup & TTL cleanup |
| [scripts/train.py](scripts/train.py) | Unified training CLI (`counter`, `tagger`, `separator`, `all`; `--smoke` / `--full`) |
| [scripts/evaluate.py](scripts/evaluate.py) | Unified evaluation, latency profiler, and `--export-public` (`evaluation/public_summary.json`) |
| `content/`, `static/` & `templates/` | Data-driven `how_it_works.json`, Jinja HTML templates, and vanilla JS/CSS (`analyze.js`, `evaluation.js`, `history.js`, `how.js`, `main.js`, `style.css`) |

## 2. Request Flow

```
Upload (WAV/MP3/FLAC)
  └─> audio.decode_mono() -> 16 kHz mono -> 8s window -> audio.waveform_to_logmel()
        ├─> counting.count_with_uncertainty()  ──> General CNN + Sparse Specialist (K<=8) + 30-pass MC-dropout
        ├─> tagging.tag_open_vocabulary()      ──> AST-527 + CLAP + 32-CNN + 1.5s sliding-window time ranges
        └─> separation.estimate_distinct_speakers() ──> ECAPA-TDNN clustering (K<=3 speech)
              └─> POST /api/separate (sync or async job)
                    ├─> 2-spk pure speech: SepFormer wsj02mix + ECAPA verification
                    ├─> Multi-sound: Query-Conditioned HPSS Wiener + IterativeMaskSeparator
                    ├─> Residual stem: mixture - sum(stems)
                    └─> quality.py (correlation < 0.90, energy conservation, self-consistency re-tag) -> WAVs + ZIP
```

## 3. Models, Training & Dev Mode

- **Checkpoints (`checkpoints/`)**: `source_count_clip.pt`, `tagger.pt`, `separator_iterative.pt`.
- **Pretrained Models (`models/`)**: `ast-audioset` (BSD-3-Clause), `clap-htsat-unfused` (Apache-2.0), `sepformer-wsj02mix` (Apache-2.0), `spkrec-ecapa-voxceleb` (Apache-2.0).
- **Training & Evaluation**: Run `.\.venv\Scripts\python scripts\train.py <target> --full` and `.\.venv\Scripts\python scripts\evaluate.py <target> --full`.
- **Dev Mode (`ECHOCOUNT_DEV`)**: Off (`0`) by default in production so no ground-truth dataset fields or SHA-256 hashes are exposed via `/api/analyze`. Set `ECHOCOUNT_DEV=1` locally to enable dataset ground-truth inspection.
