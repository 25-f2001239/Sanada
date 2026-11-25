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

Open **http://127.0.0.1:5000** and create an account with any valid email address (6-digit email verification code) or Google sign-in.

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

## 2. Authentication, Email OTP, Google Sign-In & Single Admin

Run the interactive local setup script (prompts with hidden input, writes `.env`, never prints secrets):

```powershell
python scripts/setup_auth.py
```

### Email OTP Verification (New Sign-Ups & Forgot Password Only)
- Any syntactically valid email address whose domain has an MX record can sign up (including temporary-mail domains).
- A 6-digit code is sent **only** when creating a new email account and for Forgot Password (existing users and the admin sign in directly with email and password with no OTP). Codes are generated with `secrets`, stored only as `HMAC-SHA256(secret, salt + code)`, expire after 10 minutes, are single-use, invalidate after 5 failed attempts, and enforce a 60 s resend cooldown, per-email/per-IP limits, and a daily global send budget (`OTP_DAILY_BUDGET=250`).
- **Live delivery**: Configure `BREVO_API_KEY` and `BREVO_SENDER_EMAIL` in `.env` (Brevo free tier: 300 transactional emails/day), or SMTP (`SMTP_HOST` / `SMTP_USER` / `SMTP_PASSWORD` / `SMTP_FROM`).
- **Dev-only console fallback**: Only when `ECHOCOUNT_DEV=1` (never in production), OTP codes print to the local server console. In production with no mail key configured, the app still starts cleanly and returns `"email verification is not available"` on email sign-up.

### Google Sign-In (Always Visible)
- **Default (`AUTH_GOOGLE_MODE=authlib`)**: Set `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` in `.env`. Both Sign In (`Continue with Google`) and Create Account (`Sign up with Google`) show the Google button; if credentials are not configured, the button renders in a disabled state with a helper note instead of shifting layout. Google users need no OTP, require `email_verified=true`, match by Google `sub`, and can never access the admin account.
- **Optional Firebase mode (`AUTH_GOOGLE_MODE=firebase`)**: Set `FIREBASE_PROJECT_ID`, `FIREBASE_WEB_API_KEY`, and `FIREBASE_AUTH_DOMAIN`. The server verifies Firebase Google ID tokens directly against Google's public X.509 certificates (no service-account key required).

### Single Locked Administrator & 5-Minute Rotation Procedure
- EchoCount enforces **exactly one** administrator account configured via `ADMIN_EMAILS` in `.env`. In production (`ECHOCOUNT_ENV=production`), the app refuses to start if `ADMIN_EMAILS` has 0 or more than 1 address.
- **No website button, API, or CLI can grant or change roles** (`POST /api/admin/users/<id>/role` always returns `403`).
- Admin sign-in uses email + password (14+ characters required, never Google OAuth), shorter sessions (30 min idle / 8 h absolute), progressive delay & lockout, and masks the admin email everywhere in the UI, APIs, CSV exports, and logs. Setting `ADMIN_REQUIRE_OTP=0` (default) lets the admin sign in without email latency at the tradeoff of relying solely on password strength, brute-force lockout, and the 30-minute idle timeout instead of a second factor; set `ADMIN_REQUIRE_OTP=1` to restore per-login admin email OTP step-up.
- **How to rotate the single admin address (~5 minutes)**:
  1. Ensure the new admin account exists and has a strong 14+ character password (or sign up and verify it as a regular user first).
  2. Open `.env` on the server (never commit `.env` to Git) and replace `ADMIN_EMAILS=<old-email>` with `ADMIN_EMAILS=<new-email>`.
  3. Restart `python app.py` — startup automatically demotes the previous admin to `user` (revoking its admin sessions) and promotes only the single address in `ADMIN_EMAILS`, logging only the masked address and demotion count.
  4. Sign in at `/login` with the new admin email and password (plus 6-digit email OTP if `ADMIN_REQUIRE_OTP=1`).
  5. Open `/admin` to verify only 1 admin is listed (masked) and clean up or disable the old account if no longer needed.

### Account Privacy, Deactivation & Permanent Deletion (`Settings → Account`)
- Users can export their own analysis history as JSON or CSV (`/settings/account`), **deactivate** their account temporarily (confirmed with current password or fresh Google sign-in; revoking all sessions immediately, hiding history from default views, reactivating directly by signing back in), or **permanently delete** their account (confirmed with current password or fresh Google sign-in within 5 min + typing `DELETE`, atomically removing DB records and uploaded audio/stem directories on disk).

---

## 3. Running Tests

```powershell
# OWASP security regression tests (weight-free, CI-safe)
python tests/test_security_audit.py

# Authentication overhaul, OTP, single locked admin & account lifecycle tests
python tests/test_auth_overhaul.py

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
| Google sign-in button is disabled on `/login` | Google env vars not set | Intentional — run `python scripts/setup_auth.py` or set `GOOGLE_CLIENT_ID`/`SECRET` in `.env` |
| `"Email verification is not available"` on sign-up | No Brevo/SMTP credentials in production | Configure `BREVO_API_KEY` & `BREVO_SENDER_EMAIL` via `python scripts/setup_auth.py` (or `ECHOCOUNT_DEV=1` locally) |
| `403 Forbidden` on `/admin` or role change | Roles are locked to a single `.env` admin | Set `ADMIN_EMAILS=<single-email>` in `.env` and restart `python app.py` |
| Uploaded audio disappears after 30 minutes | Automatic privacy TTL cleanup | Expected behavior (`JOB_TTL_SECONDS=1800`); download ZIPs before expiry |

---

## 7. Architecture, Privacy & Licences

- See [ARCHITECTURE.md](ARCHITECTURE.md) for the file map and request pipeline.
- Code is licensed under [LICENSE](LICENSE) (MIT). Pretrained model licences are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
- Analysis history is stored per authenticated user in local SQLite (`data/echocount.db`), never shared across accounts, and never committed to Git.
