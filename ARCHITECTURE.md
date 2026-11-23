# EchoCount Architecture

EchoCount is a Flask + PyTorch audio analysis web application that counts overlapping sound sources, names detected sounds with frame-level time ranges, separates stems with quality gates, and enforces mandatory authentication with 3-role RBAC and per-user private history.

## 1. File Map (Hand-Written Source & Code Line Counts)

### Backend Python (`15` files, `6,277` code lines)
| File | Code Lines | Responsibility |
| :--- | ---: | :--- |
| [app.py](app.py) | 1,518 | Flask routes (45 total): auth/OAuth, per-user & admin history, `/api/analyze`, `/api/separate`, stems/ZIP, `/api/evaluation`, `/api/admin/*`, `/api/capabilities` |
| [separation.py](separation.py) | 871 | `IterativeMaskSeparator`, SpeechBrain SepFormer (2-mix, 3-mix, recursive) + ECAPA verification, query-conditioned Wiener separation, 8s/2s cross-fade stitching |
| [auth.py](auth.py) | 752 | SQLite schema (`users`, `sessions`, `oauth_identities`, `auth_tokens`, `user_jobs`), Argon2id hashing, CSRF, OIDC OAuth, RBAC decorators, per-user job isolation |
| [tagging.py](tagging.py) | 732 | 32-class `GroupFineTaggerCNN` + AudioSet-527 AST + CLAP zero-shot + sliding-window time localization |
| [counting.py](counting.py) | 270 | `SourceCountCNN`, sparse specialist ($K \le 8$) routing, isotonic calibration, batched MC-dropout uncertainty |
| [data.py](data.py) | 202 | Dataset scanning, leakage audit, and dev-gated (`ECHOCOUNT_DEV=1`) SHA-256 ground-truth lookup |
| [capabilities.py](capabilities.py) | 167 | Data-driven capability registry loader (`capabilities.json`) and policy classifier (`enabled` / `experimental` / `disabled`) |
| [quality.py](quality.py) | 164 | Stem quality gates: waveform correlation ($< 0.90$), energy conservation, and self-consistency re-tagging |
| [audio.py](audio.py) | 143 | Upload sniffing, PyAV mono decode, polyphase resampling, WAV encoding, 64-band log-mel extraction |
| [config.py](config.py) | 91 | Paths, environment settings, `config.json` & `taxonomy.json` single-source-of-truth helpers |
| [scripts/evaluate.py](scripts/evaluate.py) | 517 | Unified evaluation, latency profiler, and `--export-public` (`evaluation/public_summary.json`) |
| [scripts/benchmark_capabilities.py](scripts/benchmark_capabilities.py) | 361 | Held-out speaker-disjoint LibriSpeech & clip-disjoint FSD50K separation benchmark (`K=2..8`) |
| [scripts/train.py](scripts/train.py) | 327 | Unified training CLI (`counter`, `tagger`, `separator`, `all`; `--smoke` / `--full`) |
| [scripts/download_models.py](scripts/download_models.py) | 140 | Pretrained model & checkpoint downloader and integrity verifier |
| [scripts/migrate_history.py](scripts/migrate_history.py) | 22 | Idempotent per-user `user_jobs` migration & orphan audit |

### Frontend JavaScript, CSS & Templates (`20` files, `8,346` code lines)
| File | Code Lines | Responsibility |
| :--- | ---: | :--- |
| [static/style.css](static/style.css) | 2,993 | Light & dark theme tokens, responsive layouts, glass auth card, architecture inspector, admin & evaluation UI |
| [static/analyze.js](static/analyze.js) | 986 | Audio upload, waveform scrub/highlight, analysis rendering, async separation polling, stem players & ZIP download |
| [static/main.js](static/main.js) | 663 | Shared nav, command palette (`Ctrl K`), theme switcher, logout, legacy `localStorage` history purge & user-switch cache wipe |
| [static/waves.js](static/waves.js) | 518 | Shared Canvas 2D acoustic wave engine (far spectrogram waterfall, mid additive ribbons, near particles/ripples, calm variant, Lite mode) |
| [static/auth.js](static/auth.js) | 492 | Cinematic 2.5s intro, interactive Signal Decomposition "try it" demo, pipeline ticker, form/CSRF submit & convergence animation |
| [static/evaluation.js](static/evaluation.js) | 401 | Developer Evaluation Console charts, per-K tables, confusion matrix & CSV export |
| [static/archview.js](static/archview.js) | 374 | Interactive Analyze SVG architecture diagram & stage inspector |
| [static/history.js](static/history.js) | 235 | Server-backed per-user History page + admin-only All Users toggle, filter, delete & CSV export |
| [static/admin.js](static/admin.js) | 159 | Admin Console user management, role/status controls, session revoke & temp cleanup |
| [static/how.js](static/how.js) | 81 | Interactive How-It-Works window & spectrogram visualizer |
| `templates/*.html` (10 files) | 1,444 | Jinja templates (`base`, `index`, `analyze`, `auth`, `history`, `how`, `docs`, `evaluation`, `admin`, `403`) |

## 2. Request Flow

```
Upload (WAV/MP3/FLAC) -> Authenticated Session + Rate Limit
  └─> audio.decode_mono() -> 16 kHz mono -> 8s window -> audio.waveform_to_logmel()
        ├─> counting.count_with_uncertainty()     ──> General CNN + Sparse Specialist (K<=8) + 30-pass MC-dropout
        ├─> tagging.tag_open_vocabulary()         ──> AST-527 + CLAP + 32-CNN + 1.5s sliding-window time ranges
        └─> separation.estimate_distinct_speakers() ──> ECAPA-TDNN clustering (K<=3 speech)
              └─> POST /api/separate (sync or async job, owner/admin only)
                    ├─> Speech (K=2 / additive K=3 / recursive): SepFormer + ECAPA verification
                    ├─> Multi-sound: Query-Conditioned HPSS Wiener + IterativeMaskSeparator
                    ├─> Residual stem: mixture - sum(stems)
                    └─> quality.py (correlation < 0.90, energy conservation, self-consistency re-tag) -> WAVs + ZIP
```

## 3. Models, Checkpoints & Dev Mode

- **Active Checkpoints (`checkpoints/`)**: `source_count_clip.pt`, `source_count_cnn.pt`, `tagger.pt`, `separator_iterative.pt`.
- **Pretrained Models (`models/`)**: `ast-audioset` (BSD-3-Clause), `clap-htsat-unfused` (Apache-2.0), `sepformer-wsj02mix` (Apache-2.0), `sepformer-wsj03mix` (Apache-2.0), `spkrec-ecapa-voxceleb` (Apache-2.0).
- **Dev Mode (`ECHOCOUNT_DEV`)**: Off (`0`) by default so no ground-truth dataset fields or SHA-256 hashes are exposed via `/api/analyze`.
