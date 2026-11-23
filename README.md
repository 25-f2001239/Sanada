# EchoCount — Audio Source Counting, Sound Naming & Separation

EchoCount is a Flask + PyTorch audio analysis web app that estimates how many sounds overlap in a single-channel recording, names detected sounds with frame-level time ranges, separates stems with quality gates, and stores analysis history privately per account.

---

## 1. Quick Start (3 Steps)

**Python version:** `3.14.8` (see `.python-version`; compatible with Python `3.11 – 3.14`).

```powershell
# 1. Clone and create a virtual environment
git clone https://github.com/25-f2001239/Sanada.git
cd Sanada
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies (~250 MB+ including PyTorch)
pip install -r requirements.txt

# 3. Start the app (auto-creates .env, database, and downloads models on first run)
python app.py
```

Open **http://127.0.0.1:5000** and create an account with email + password.

### What Downloads Automatically on First Run
When you run `python app.py` on a fresh clone, EchoCount automatically creates `.env` (with a random `SECRET_KEY` for local development), initializes `data/echocount.db`, and fetches missing weights (~1.17 GB total, one-time download):

| Asset | Source | Size | Licence |
|---|---|---:|---|
| `source_count_clip.pt`, `tagger.pt`, `separator_iterative.pt` | GitHub Release `v1.0.0` (SHA-256 verified) | 12.84 MB | MIT |
| `MIT/ast-finetuned-audioset-10-10-0.4593` (pinned revision) | Hugging Face | ~346 MB | BSD-3-Clause |
| `laion/clap-htsat-unfused` (pinned revision) | Hugging Face | ~615 MB | Apache-2.0 / MIT |
| `speechbrain/sepformer-wsj02mix` (pinned revision) | Hugging Face | ~113 MB | Apache-2.0 |
| `speechbrain/spkrec-ecapa-voxceleb` (pinned revision) | Hugging Face | ~83 MB | Apache-2.0 |

*(If you are offline, `python app.py` still starts cleanly and tells you which assets are missing. To skip auto-download on startup, set `ECHOCOUNT_SKIP_AUTO_DOWNLOAD=1`. You can also verify or download manually anytime with `python scripts/download_models.py --check`.)*

---

## 2. Optional Configuration (Admin & OAuth)

- **Becoming an Admin**: Nobody is admin by default. Add your email(s) to `.env` and restart `python app.py`:
  ```env
  ADMIN_EMAILS=you@gmail.com
  ```
- **Google & Microsoft Sign-In (Optional)**: Email + password sign-up always works out of the box. Google and Microsoft OAuth buttons appear on `/login` **only** when their environment variables are set in `.env` (see `.env.example` for `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `MICROSOFT_CLIENT_ID`, `MICROSOFT_CLIENT_SECRET`).
- **Production Mode**: Set `ECHOCOUNT_ENV=production` and supply a strong `SECRET_KEY` (>= 32 chars) in the environment; production mode refuses default or missing secret keys.

---

## 3. Running Tests

```powershell
# Per-user history privacy & isolation tests (14 checks)
python tests/test_history_privacy.py

# Template, auth, RBAC, and product regression tests
python test_templates.py --full

# Golden regression check (requires local dataset & checkpoints)
python tests/golden_check.py --verify
```

---

## 4. Measured Benchmarks & Honest Limits

All figures are read from `evaluation/public_summary.json`, `evaluation/product_benchmarks.json`, and `evaluation/separation_results.json`:

- **Source Counting (`n = 1,285` test mixtures)**: Overall `MAE = 1.65`, `Exact = 43.8%`, `Within ±1 = 64.3%`; Speech Test (`n = 695`): `MAE = 0.80`, `Exact = 53.1%`, `Within ±1 = 78.9%`.
  - **Dense mixtures (`K > 6`)**: Accuracy drops sharply (`0.0%` exact at `K = 9`, `MAE = 2.81`); EchoCount warns users and disables separation above reliable sparse limits.
  - **Unseen speakers (`n = 352`)**: `13.1%` exact (`MAE = 2.05`), reflecting a real domain generalization gap vs. seen speakers.
- **Tagging**: 32-class FSD tagger Group mAP = `0.9926`, Fine-32 mAP = `0.8052`; AST + CLAP held-out 16-class macro AP = `0.9068`.
- **Separation**:
  - **2-speaker speech**: Enabled via pretrained `sepformer-wsj02mix` + ECAPA-TDNN speaker verification + 3 quality gates (correlation `< 0.90`, energy conservation, self-consistency re-tagging).
  - **General multi-sound separation**: Synthetic `K = 2..5` benchmark mean SI-SDRi = `+2.51 dB`, self-consistency `39.3%`, energy conservation `100.0%`; stems failing quality gates are flagged and withheld from ZIP export.

---

## 5. Line Counting Method

Hand-written source counts (Python, JavaScript, CSS, HTML/Jinja) exclude vendored code, virtual environments, models, checkpoints, datasets, caches, generated JSON, tests, and docs:

```powershell
cloc . --include-lang="Python,JavaScript,CSS,HTML" `
  --exclude-dir=.venv,venv,node_modules,.git,models,checkpoints,data,uploads,work,outputs,deliverables,__pycache__,.pytest_cache,.ruff_cache,.mypy_cache,.cache,feature_cache,tests,evaluation `
  --not-match-f="^test_.*\.py$"
```

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| First startup takes a few minutes | Downloading ~1.17 GB of pretrained Hugging Face models & release checkpoints | Wait for progress to finish; subsequent startups skip verified files |
| `"Some assets could not be downloaded"` on startup | Offline or network blocked | Re-run `python scripts/download_models.py` once online |
| `pip install` fails on `torch` or `av` | Unsupported Python version or old `pip` | Use Python `3.11 – 3.14` (`.python-version` = `3.14.8`) and run `python -m pip install --upgrade pip` |
| Google / Microsoft buttons not shown on `/login` | OAuth env vars not set | Intentional — configure `GOOGLE_CLIENT_ID`/`SECRET` or `MICROSOFT_CLIENT_ID`/`SECRET` in `.env` to enable |
| `403 Forbidden` on `/admin` | Account role is `user` | Add your email to `ADMIN_EMAILS` in `.env` and restart `python app.py` |
| Uploaded audio disappears after 30 minutes | Automatic privacy TTL cleanup | Expected behavior (`JOB_TTL_SECONDS=1800`); download ZIPs before expiry |

---

## 7. Architecture, Privacy & Licences

- See [ARCHITECTURE.md](ARCHITECTURE.md) for the file map and request pipeline.
- Code is licensed under [LICENSE](LICENSE) (MIT). Pretrained model licences are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
- Analysis history is stored per authenticated user in local SQLite (`data/echocount.db`), never shared across accounts, and never committed to Git.
