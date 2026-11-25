# EchoCount Architecture

EchoCount is a Flask + PyTorch audio analysis web application that counts overlapping sound sources, names detected sounds with frame-level time ranges, separates stems with quality gates, and enforces mandatory authentication, email OTP verification, Google sign-in, single-admin locking, and per-user private history.

## 1. File Map (Hand-Written Source & Code Line Counts)

Measured with the `cloc` command documented in [README.md](README.md#5-line-counting-method) (excluding `.venv`, `models/`, `checkpoints/`, `data/`, `uploads/`, `tests/`, `evaluation/`, caches, and docs): **37 source files, 15,946 code lines** (`5` test files / `2,291` code lines tracked separately).

### Backend Python (`15` files, `6,833` code lines)
| File | Code Lines | Responsibility |
| :--- | ---: | :--- |
| [app.py](app.py) | 1,908 | Flask routes: auth/OTP/Google OAuth, per-user & admin history, `/api/analyze`, `/api/separate`, stems/ZIP, `/api/evaluation`, `/api/admin/*`, `/api/account*`, `/api/capabilities`, Same-Origin & security headers |
| [auth.py](auth.py) | 1,010 | SQLite schema (`users`, `sessions`, `oauth_identities`, `email_otps`, `user_jobs`, `audit_log`), Argon2id hashing, CSRF decorator, HMAC OTPs, Google OIDC + Firebase RS256, single-admin enforcement, RBAC & per-user job isolation |
| [separation.py](separation.py) | 847 | `IterativeMaskSeparator`, SpeechBrain SepFormer (2-mix, 3-mix, recursive) + ECAPA verification, query-conditioned Wiener separation, 8s/2s cross-fade stitching |
| [tagging.py](tagging.py) | 721 | 32-class `GroupFineTaggerCNN` + AudioSet-527 AST + CLAP zero-shot + sliding-window time localization & co-occurrence pairs |
| [scripts/evaluate.py](scripts/evaluate.py) | 572 | Unified evaluation, latency profiler, and `--export-public` (`evaluation/public_summary.json`) |
| [scripts/train.py](scripts/train.py) | 322 | Unified training CLI (`counter`, `tagger`, `separator`, `all`; `--smoke` / `--full`) |
| [counting.py](counting.py) | 262 | `SourceCountCNN`, sparse specialist ($K \le 8$) routing, isotonic calibration, batched MC-dropout uncertainty |
| [scripts/download_models.py](scripts/download_models.py) | 199 | First-run asset bootstrapper: pinned HF model downloader + GitHub Release `v1.0.0` SHA-256 checkpoint verifier |
| [data.py](data.py) | 195 | Dataset scanning, leakage audit, and dev-gated (`ECHOCOUNT_DEV=1`) SHA-256 ground-truth lookup |
| [mail.py](mail.py) | 184 | Transactional OTP email delivery (Brevo REST API, SMTP relay, dev-only console fallback, daily send budget, MX validation) |
| [capabilities.py](capabilities.py) | 162 | Data-driven capability registry loader (`capabilities.json`) and policy classifier (`enabled` / `experimental` / `disabled`) |
| [quality.py](quality.py) | 159 | Stem quality gates: waveform correlation ($< 0.90$), energy conservation, and self-consistency re-tagging |
| [audio.py](audio.py) | 132 | Upload sniffing, PyAV mono decode, polyphase resampling, WAV encoding, 64-band log-mel extraction |
| [config.py](config.py) | 87 | Paths, environment settings, `config.json` & `taxonomy.json` single-source-of-truth helpers |
| [scripts/setup_auth.py](scripts/setup_auth.py) | 75 | Interactive local `.env` setup CLI (`getpass` hidden prompts, masked confirmations, never prints secrets) |

### Frontend JavaScript, CSS & Templates (`22` files, `9,113` code lines)
| File | Code Lines | Responsibility |
| :--- | ---: | :--- |
| [static/style.css](static/style.css) | 3,137 | Light & dark theme tokens, responsive layouts, glass auth card, tabbed Analyze UI, architecture inspector, admin & evaluation UI |
| [static/analyze.js](static/analyze.js) | 1,018 | Audio upload, waveform scrub/highlight, compact tabbed results (Overview / Sounds & Timeline / Separation / Details), async separation polling, stem players & ZIP download |
| [static/auth.js](static/auth.js) | 723 | Auth tabs, email OTP modal/resend timer, forgot-password flow, Google OAuth/Firebase trigger, inline validation & motion |
| [static/main.js](static/main.js) | 663 | Shared nav, command palette (`Ctrl K`), theme switcher, logout, legacy `localStorage` history purge & user-switch cache wipe |
| [static/waves.js](static/waves.js) | 518 | Shared Canvas 2D acoustic wave engine (far spectrogram waterfall, mid additive ribbons, near particles/ripples, calm variant, Lite mode) |
| [static/evaluation.js](static/evaluation.js) | 401 | Developer Evaluation Console charts, per-K tables, confusion matrix & CSV export |
| [static/archview.js](static/archview.js) | 374 | Interactive Analyze SVG architecture diagram & stage inspector |
| [static/history.js](static/history.js) | 235 | Server-backed per-user History page + admin-only All Users toggle, filter, delete & CSV export |
| [static/admin.js](static/admin.js) | 190 | Admin Console user management, status controls, session revoke, audit log & temp cleanup |
| [static/how.js](static/how.js) | 81 | Interactive How-It-Works window & spectrogram visualizer |
| [static/account.js](static/account.js) | 77 | Account Settings export, deactivation, and permanent deletion confirmation UI |
| `templates/*.html` (11 files) | 1,696 | Jinja templates (`base`, `index`, `analyze`, `auth`, `account`, `history`, `how`, `docs`, `evaluation`, `admin`, `403`) |

### Why Source Exceeds 10,000 Lines (Feature Breakdown)
Keeping every active product feature intact (without minifying or hiding code in data files) accounts for the ~5,946 lines above 10k:
- **Design system & responsive themes** (`static/style.css`): `3,137` lines
- **Full auth + email OTP + Google OIDC/Firebase + Account lifecycle + Admin Console** (`auth.py`, `mail.py`, `static/auth.js`, `static/admin.js`, `static/account.js`, `templates/auth.html`, `admin.html`, `account.html`, `scripts/setup_auth.py`): `~2,850` lines
- **Developer Evaluation Console & Training/Eval CLIs** (`scripts/evaluate.py`, `scripts/train.py`, `static/evaluation.js`, `templates/evaluation.html`): `1,375` lines
- **Interactive Acoustic Wave Engine & Architecture Inspector** (`static/waves.js`, `static/archview.js`): `892` lines
- **How-It-Works & In-App Documentation** (`templates/how.html`, `templates/docs.html`, `static/how.js`): `468` lines

## 2. Request Flow

```
Upload (WAV/MP3/FLAC) -> Authenticated Session + Same-Origin Check + Rate Limit
  └─> audio.decode_mono() -> 16 kHz mono -> 8s window -> audio.waveform_to_logmel()
        ├─> counting.count_with_uncertainty()       ──> General CNN + Sparse Specialist (K<=8) + 30-pass MC-dropout
        ├─> tagging.tag_open_vocabulary()           ──> AST-527 + CLAP + 32-CNN + 1.5s sliding-window time ranges + top-12 pairs
        └─> separation.estimate_distinct_speakers() ──> ECAPA-TDNN clustering (K<=3 speech)
              └─> POST /api/separate (sync or async job, owner/admin only)
                    ├─> Speech (K=2 / additive K=3 / recursive): SepFormer + ECAPA verification
                    ├─> Multi-sound: Query-Conditioned HPSS Wiener + IterativeMaskSeparator
                    ├─> Residual stem: mixture - sum(stems)
                    └─> quality.py (correlation < 0.90, energy conservation, self-consistency re-tag) -> WAVs + ZIP
```

## 3. Models, Checkpoints & Dev Mode

- **Active Checkpoints (`checkpoints/`)**: `source_count_clip.pt`, `tagger.pt`, `separator_iterative.pt` (fetched from GitHub Release `v1.0.0` with SHA-256 verification on first run).
- **Pretrained Models (`models/`)**: `ast-audioset` (BSD-3-Clause), `clap-htsat-unfused` (Apache-2.0), `sepformer-wsj02mix` (Apache-2.0), `spkrec-ecapa-voxceleb` (Apache-2.0).
- **Dev Mode (`ECHOCOUNT_DEV`)**: Off (`0`) by default so no ground-truth dataset fields, SHA-256 hashes, or console OTP codes are exposed in production.
