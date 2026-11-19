"""Flask web application and API for EchoCount."""

from __future__ import annotations

import io
import json
import logging
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, make_response, redirect, render_template, request, send_file, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

import audio
import auth
import counting
import data as data_mod
import quality as quality_mod
import separation
import tagging
from config import (
    BASE_DIR,
    CLIP_SAMPLES,
    CLIP_SECONDS,
    EVAL_DIR,
    MODEL_PATH,
    SAMPLE_RATE,
)

load_dotenv(BASE_DIR / ".env")

UPLOAD_DIR = BASE_DIR / "uploads"
JOB_DIR = UPLOAD_DIR / "jobs"
EVAL_PATH = Path(os.getenv("EVAL_RESULTS", str(EVAL_DIR / "results.json")))
PUBLIC_SUMMARY_PATH = Path(os.getenv("PUBLIC_SUMMARY_PATH", str(EVAL_DIR / "public_summary.json")))
HOW_CONTENT_PATH = BASE_DIR / "content" / "how_it_works.json"
JOB_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(40 * 1024 * 1024)))
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(MAX_UPLOAD_BYTES + 1024 * 1024)))
MAX_DURATION_SECONDS = float(os.getenv("MAX_DURATION_SECONDS", "600"))
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "1800"))
MC_PASSES = int(os.getenv("MC_PASSES", "30"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s [echocount] %(message)s"
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = auth.get_or_create_secret_key()
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.getenv("ECHOCOUNT_COOKIE_SECURE", "0") == "1"

auth.init_db()
auth.init_oauth(app)

_jobs: dict = {}
_jobs_lock = threading.Lock()
_rate_buckets: dict[str, deque] = {}
_rate_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=int(os.getenv("SEPARATION_WORKERS", "2")))


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_") or "sound"


def check_rate_limit() -> bool:
    """Enforce per-IP sliding-window rate limit."""
    if RATE_LIMIT_PER_MINUTE <= 0:
        return True
    ip, now = request.remote_addr or "local", time.time()
    with _rate_lock:
        dq = _rate_buckets.setdefault(ip, deque())
        while dq and now - dq[0] > 60.0:
            dq.popleft()
        if len(dq) >= RATE_LIMIT_PER_MINUTE:
            return False
        dq.append(now)
    return True


def model_status() -> dict:
    """Aggregate readiness and metadata across counter, taggers, and separators."""
    ckpt = counting.checkpoint_status(MODEL_PATH) or {}
    return {
        "trained": bool(ckpt.get("trained", False)),
        "model_name": "EchoCount SourceCountCNN 8-second",
        "sample_rate": SAMPLE_RATE,
        "clip_seconds": CLIP_SECONDS,
        "min_sources": ckpt.get("min_sources"),
        "max_sources": ckpt.get("max_sources"),
        "val_mae": ckpt.get("val_mae"),
        "val_macro_mae": ckpt.get("val_macro_mae"),
        "val_exact_accuracy": ckpt.get("val_exact_accuracy"),
        "val_within_one": ckpt.get("val_within_one"),
        "model_error": ckpt.get("error"),
        "separator": separation.separator_status() or {},
        "speech_separator": separation.pretrained_speech_status() or {},
        "tagger": tagging.tagger_status() or {},
        "open_tagger": tagging.open_tagger_status() or {},
    }


def get_dense_thresholds() -> tuple[int, int, int]:
    """Read last reliable sparse K and first dense K accuracy from evaluation/results.json."""
    if not EVAL_PATH.is_file():
        return 6, 9, 43
    try:
        per_count = json.loads(EVAL_PATH.read_text(encoding="utf-8")).get("per_count") or []
        reliable = [
            p
            for p in per_count
            if not p.get("unreliable")
            and int(p.get("n", 0)) >= 30
            and int(p.get("true_k", 0)) <= 8
            and float(p.get("exact_accuracy", 0.0)) >= 0.5
        ]
        last_k = max((int(p["true_k"]) for p in reliable), default=6)
        dense = sorted(
            [
                p
                for p in per_count
                if not p.get("unreliable")
                and int(p.get("n", 0)) >= 30
                and int(p.get("true_k", 0)) > last_k
            ],
            key=lambda x: int(x["true_k"]),
        )
        return (
            (
                last_k,
                int(dense[0]["true_k"]),
                round(float(dense[0].get("exact_accuracy", 0.0)) * 100),
            )
            if dense
            else (last_k, last_k + 3, 40)
        )
    except Exception:
        return 6, 9, 43


def get_dense_mixture_warning(rounded_k: int) -> str | None:
    """Return dense-mixture reliability warning when K exceeds reliable sparse threshold."""
    if not EVAL_PATH.is_file():
        return None
    last_k, dense_k, dense_pct = get_dense_thresholds()
    if int(rounded_k) <= last_k:
        return None
    return f"Too many overlapping sounds to separate reliably: counts above about {last_k} are unreliable (measured exact accuracy at K={dense_k} is {dense_pct}%)"


def sweep_jobs(startup: bool = False) -> None:
    """Delete expired in-memory jobs and orphaned job directories (all on startup)."""
    now = time.time()
    with _jobs_lock:
        expired = [jid for jid, job in _jobs.items() if now - job["created"] > JOB_TTL_SECONDS]
        for jid in expired:
            _jobs.pop(jid, None)
    for jid in expired:
        shutil.rmtree(JOB_DIR / jid, ignore_errors=True)
    try:
        for p in JOB_DIR.iterdir():
            if not p.is_dir():
                continue
            if startup or (p.name not in _jobs and now - p.stat().st_mtime > JOB_TTL_SECONDS):
                shutil.rmtree(p, ignore_errors=True)
    except OSError:
        pass


sweep_jobs(startup=True)


def _read_json_file(path: Path) -> dict | None:
    """Read a JSON object from disk, returning None if absent or unreadable."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def load_public_summary() -> dict | None:
    """Load the committed public benchmark summary if present."""
    return _read_json_file(PUBLIC_SUMMARY_PATH)


@app.context_processor
def inject_site_globals() -> dict:
    """Expose public stats, auth state, dev flag, and job retention TTL to all Jinja templates."""
    return {
        "public_stats": load_public_summary(),
        "is_dev": data_mod.is_dev_mode(),
        "job_ttl_minutes": max(1, round(JOB_TTL_SECONDS / 60)),
        "current_user": auth.get_current_user(),
        "csrf_token": auth.get_or_create_form_csrf(),
        "oauth_providers": auth.provider_availability(),
    }


def get_job(job_id: str):
    """Look up an active job by hex job_id."""
    sweep_jobs()
    with _jobs_lock:
        job = _jobs.get(job_id)
    return job if (job is not None and all(c in "0123456789abcdef" for c in job_id)) else None


def read_upload():
    """Validate multipart upload, save to temp job dir, and decode to mono."""
    sweep_jobs()
    f = request.files.get("audio")
    if f is None or not f.filename:
        raise audio.AudioValidationError("Choose an audio file first.")
    suffix = audio.validate_extension(f.filename)
    name = secure_filename(f.filename) or f"audio{suffix}"
    job_id = uuid.uuid4().hex
    job_dir = JOB_DIR / job_id
    job_dir.mkdir(parents=True)
    try:
        path = job_dir / f"original{suffix}"
        f.save(path)
        if path.stat().st_size > MAX_UPLOAD_BYTES:
            raise audio.AudioValidationError(
                f"The upload exceeds the {MAX_UPLOAD_BYTES / 1048576:g} MB file limit."
            )
        samples, rate = audio.decode_mono(path, MAX_DURATION_SECONDS)
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    return job_id, job_dir, path, samples, rate, name


def analysis_clip(samples: np.ndarray, rate: int):
    """Resample to 16 kHz and extract centred 8-second window for counting."""
    model_rate = audio.resample(samples, rate, SAMPLE_RATE)
    if len(model_rate) > CLIP_SAMPLES:
        start = (len(model_rate) - CLIP_SAMPLES) // 2
        clip, crop_start = model_rate[start : start + CLIP_SAMPLES], start / SAMPLE_RATE
    else:
        clip, crop_start = np.pad(model_rate, (0, CLIP_SAMPLES - len(model_rate))), 0.0
    return clip.astype(np.float32), crop_start, len(model_rate) / SAMPLE_RATE


@app.get("/")
def index():
    return render_template("index.html", status=model_status())


@app.get("/analyze")
def analyze_page():
    return render_template("analyze.html", status=model_status())


@app.get("/evaluation")
def evaluation_page():
    return render_template("evaluation.html", status=model_status())


@app.get("/history")
def history_page():
    return render_template("history.html", status=model_status())


@app.get("/how")
def how_page():
    return render_template(
        "how.html",
        status=model_status(),
        how_content=_read_json_file(HOW_CONTENT_PATH) or {},
    )


@app.get("/docs")
def docs_page():
    return render_template("docs.html", status=model_status())


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "model": model_status()})


@app.get("/api/public-stats")
def public_stats_route():
    summary = load_public_summary()
    if summary is None:
        return jsonify({"available": False, "reason": "Public summary has not been exported yet."}), 404
    return jsonify(summary)


def _safe_next_url(raw: str | None) -> str:
    target = (raw or "/analyze").strip()
    return target if target.startswith("/") and not target.startswith("//") else "/analyze"


@app.get("/login")
def login_page():
    if auth.get_current_user():
        return redirect(_safe_next_url(request.args.get("next")))
    return render_template(
        "auth.html",
        status=model_status(),
        mode="login",
        next_url=_safe_next_url(request.args.get("next")),
    )


@app.get("/register")
def register_page():
    if auth.get_current_user():
        return redirect(_safe_next_url(request.args.get("next")))
    return render_template(
        "auth.html",
        status=model_status(),
        mode="register",
        next_url=_safe_next_url(request.args.get("next")),
    )


@app.get("/forgot-password")
def forgot_password_page():
    return render_template("auth.html", status=model_status(), mode="forgot", next_url="/analyze")


@app.get("/reset-password")
def reset_password_page():
    return render_template(
        "auth.html",
        status=model_status(),
        mode="reset",
        token=request.args.get("token", ""),
        next_url="/analyze",
    )


@app.get("/api/auth/me")
def auth_me():
    user = auth.get_current_user()
    if not user:
        return jsonify({"authenticated": False}), 200
    return jsonify(
        {
            "authenticated": True,
            "email": user["email"],
            "email_verified": user["email_verified"],
            "role": user.get("role", "user"),
        }
    )


@app.post("/auth/login")
def auth_login_route():
    if not auth.verify_csrf_request():
        return jsonify({"error": "Invalid or expired security token. Refresh the page and try again."}), 403
    if not auth.verify_captcha_hook(request):
        return jsonify({"error": "Verification check failed."}), 400
    data = request.get_json(silent=True) or request.form
    email = str(data.get("email", ""))
    password = str(data.get("password", ""))
    ip = request.remote_addr or "local"
    if not auth.check_auth_rate_limit(f"login:ip:{ip}") or not auth.check_auth_rate_limit(
        f"login:email:{auth.normalize_email(email)}", limit=5, window_s=300.0
    ):
        return jsonify({"error": "Too many sign-in attempts. Please wait a few minutes and try again."}), 429
    user, err = auth.authenticate_password_user(email, password)
    if err or not user:
        return jsonify({"error": err or "Invalid email or password."}), 401
    sid, _ = auth.create_session(user["id"])
    resp = make_response(
        jsonify({"ok": True, "email": user["email"], "redirect": _safe_next_url(data.get("next"))})
    )
    return auth.set_session_cookie(resp, sid)


@app.post("/auth/register")
def auth_register_route():
    if not auth.verify_csrf_request():
        return jsonify({"error": "Invalid or expired security token. Refresh the page and try again."}), 403
    if not auth.verify_captcha_hook(request):
        return jsonify({"error": "Verification check failed."}), 400
    data = request.get_json(silent=True) or request.form
    email = str(data.get("email", ""))
    password = str(data.get("password", ""))
    ip = request.remote_addr or "local"
    if not auth.check_auth_rate_limit(f"reg:ip:{ip}", limit=10, window_s=600.0):
        return jsonify({"error": "Too many registration attempts. Please try again later."}), 429
    uid, token, err = auth.register_password_user(email, password)
    if err or not uid or not token:
        return jsonify({"error": err or "Registration could not be completed."}), 400
    verify_url = url_for("auth_verify_route", token=token, _external=True)
    sent = auth.send_auth_email(
        auth.normalize_email(email),
        "Verify your EchoCount account",
        f"Click this link to verify your EchoCount account (valid for 30 minutes):\n\n{verify_url}\n",
    )
    is_local_or_dev = (
        (request.remote_addr in ("127.0.0.1", "::1", "localhost"))
        or data_mod.is_dev_mode()
        or os.getenv("ECHOCOUNT_TEST_VERIFY_LINK") == "1"
    )
    if not sent and not is_local_or_dev:
        return jsonify(
            {"error": "Email verification service (SMTP_HOST) is not configured on this server."}
        ), 503
    payload: dict = {
        "ok": True,
        "verification_required": True,
        "message": (
            "Verification email sent. Check your inbox to activate your account."
            if sent
            else "Account created. Click Verify email now to activate your account."
        ),
    }
    if not sent and is_local_or_dev:
        payload["verify_url"] = verify_url
    return jsonify(payload), 201


@app.get("/auth/verify")
def auth_verify_route():
    token = request.args.get("token", "")
    uid = auth.consume_auth_token(token, "verify_email")
    if not uid:
        return render_template(
            "auth.html",
            status=model_status(),
            mode="login",
            next_url="/analyze",
            auth_error="Verification link is invalid or expired. Please register or request a new link.",
        ), 400
    with auth._conn() as con:
        con.execute("UPDATE users SET email_verified = 1 WHERE id = ?", (uid,))
    sid, _ = auth.create_session(uid)
    resp = make_response(redirect(url_for("analyze_page")))
    return auth.set_session_cookie(resp, sid)


@app.post("/auth/forgot")
def auth_forgot_route():
    if not auth.verify_csrf_request():
        return jsonify({"error": "Invalid or expired security token."}), 403
    data = request.get_json(silent=True) or request.form
    email = auth.normalize_email(str(data.get("email", "")))
    ip = request.remote_addr or "local"
    if not auth.check_auth_rate_limit(f"forgot:ip:{ip}", limit=6, window_s=600.0):
        return jsonify({"error": "Too many reset requests. Please wait a few minutes."}), 429
    payload: dict = {
        "ok": True,
        "message": "If an eligible verified account exists for that email, a reset link has been prepared.",
    }
    with auth._conn() as con:
        row = con.execute("SELECT id, email_verified FROM users WHERE email = ?", (email,)).fetchone()
    if row and row["email_verified"]:
        token = auth.create_auth_token(int(row["id"]), "reset_password")
        reset_url = url_for("reset_password_page", token=token, _external=True)
        sent = auth.send_auth_email(
            email,
            "Reset your EchoCount password",
            f"Reset your EchoCount password (valid for 30 minutes):\n\n{reset_url}\n",
        )
        is_local_or_dev = (
            (request.remote_addr in ("127.0.0.1", "::1", "localhost"))
            or data_mod.is_dev_mode()
            or os.getenv("ECHOCOUNT_TEST_VERIFY_LINK") == "1"
        )
        if not sent and is_local_or_dev:
            payload["reset_url"] = reset_url
    return jsonify(payload)


@app.post("/auth/reset")
def auth_reset_route():
    if not auth.verify_csrf_request():
        return jsonify({"error": "Invalid or expired security token."}), 403
    data = request.get_json(silent=True) or request.form
    token = str(data.get("token", ""))
    password = str(data.get("password", ""))
    pw_err = auth.validate_password(password)
    if pw_err:
        return jsonify({"error": pw_err}), 400
    uid = auth.consume_auth_token(token, "reset_password")
    if not uid:
        return jsonify({"error": "Reset link is invalid or expired."}), 400
    pw_hash = auth.hash_password(password)
    with auth._conn() as con:
        con.execute("UPDATE users SET password_hash = ? WHERE id = ?", (pw_hash, uid))
        con.execute("UPDATE sessions SET revoked = 1 WHERE user_id = ?", (uid,))
    return jsonify({"ok": True, "message": "Password updated. You can now sign in.", "redirect": "/login"})


@app.post("/auth/logout")
def auth_logout_route():
    if not auth.verify_csrf_request():
        return jsonify({"error": "Invalid or expired security token."}), 403
    user = auth.get_current_user()
    if user:
        auth.revoke_session(user["sid"])
    resp = make_response(jsonify({"ok": True, "redirect": "/login"}))
    return auth.clear_session_cookie(resp)


@app.get("/auth/oauth/<provider>")
def oauth_start(provider: str):
    if provider not in auth.SUPPORTED_PROVIDERS:
        return jsonify({"error": "Unsupported authentication provider."}), 404
    client = auth._oauth.create_client(provider)
    if client is None:
        return render_template(
            "auth.html",
            status=model_status(),
            mode="login",
            next_url="/analyze",
            auth_error=f"{provider.title()} sign-in is not configured on this server yet.",
        ), 503
    redirect_uri = url_for("oauth_callback", provider=provider, _external=True)
    return client.authorize_redirect(redirect_uri)


@app.get("/auth/oauth/<provider>/callback")
def oauth_callback(provider: str):
    if provider not in auth.SUPPORTED_PROVIDERS:
        return jsonify({"error": "Unsupported authentication provider."}), 404
    client = auth._oauth.create_client(provider)
    if client is None:
        return redirect(url_for("login_page"))
    try:
        token = client.authorize_access_token()
        userinfo = token.get("userinfo") or client.userinfo()
    except Exception:
        auth.logger.exception("OAuth callback failed for %s", provider)
        return render_template(
            "auth.html",
            status=model_status(),
            mode="login",
            next_url="/analyze",
            auth_error=f"{provider.title()} sign-in could not be verified. Please try again.",
        ), 400
    sub = str(userinfo.get("sub") or userinfo.get("oid") or "").strip()
    raw_email = str(userinfo.get("email") or userinfo.get("preferred_username") or "").strip()
    verified = userinfo.get("email_verified")
    # Microsoft v2 enterprise tokens may omit email_verified when email is verified in tenant
    is_verified = bool(verified) if verified is not None else (provider == "microsoft" and bool(raw_email))
    if not sub or not raw_email:
        return render_template(
            "auth.html",
            status=model_status(),
            mode="login",
            next_url="/analyze",
            auth_error="Provider did not return a verified identity and email address.",
        ), 400
    user, err = auth.register_oauth_user(provider, sub, raw_email, is_verified)
    if err or not user:
        return render_template(
            "auth.html",
            status=model_status(),
            mode="login",
            next_url="/analyze",
            auth_error=err or "Could not sign in with provider.",
        ), 403
    sid, _ = auth.create_session(user["id"])
    resp = make_response(redirect(url_for("analyze_page")))
    return auth.set_session_cookie(resp, sid)


@app.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    return jsonify(
        {"error": f"The upload exceeds the {MAX_UPLOAD_BYTES / 1048576:g} MB file limit."}
    ), 413


@app.get("/api/evaluation")
def evaluation():
    if not EVAL_PATH.is_file():
        return jsonify({"available": False, "reason": "No evaluation results yet."})
    try:
        payload = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return jsonify({"available": False, "reason": f"Evaluation file could not be read: {exc}"})
    current = counting.checkpoint_fingerprint(MODEL_PATH)
    recorded = (payload.get("checkpoint") or {}).get("sha256")
    payload["stale"] = bool(current and recorded and current != recorded)
    payload["checkpoint_present"] = current is not None
    sep_eval = EVAL_DIR / "separation_results.json"
    if sep_eval.is_file():
        try:
            payload["separation_eval"] = json.loads(sep_eval.read_text(encoding="utf-8"))
        except Exception:
            pass
    return jsonify(payload)


def _separation_eligibility(
    k: int,
    last_reliable_k: int,
    is_pure_speech: bool,
    named_sounds: list,
    sep: dict,
    speech_sep: dict,
) -> dict:
    max_spk = int(speech_sep.get("max_supported_speakers") or 2)
    sep_trained = bool(sep.get("trained", False))
    if k > last_reliable_k:
        return {
            "can_separate": False,
            "trained": True,
            "speech_mode": is_pure_speech,
            "speech_separator": speech_sep,
            "max_separable": last_reliable_k,
            "reason": "Too many overlapping sounds to separate reliably.",
        }
    if is_pure_speech:
        supports_k = (k == 2 and bool(speech_sep.get("has_2spk_model"))) or (
            k == 3 and bool(speech_sep.get("has_3spk_model"))
        )
        reason = (
            None
            if supports_k
            else (
                f"Separation not available for this recording: only {k} source estimated."
                if k < 2
                else f"Separation not available for this recording: Separation is supported for up to {max_spk} speakers with the installed model."
            )
        )
        return {
            "can_separate": supports_k,
            "trained": bool(speech_sep.get("available") or sep_trained),
            "speech_mode": True,
            "speech_separator": speech_sep,
            "max_separable": max_spk,
            "reason": reason,
        }
    max_sep = min(last_reliable_k, separation.MAX_SEPARABLE)
    can_sep = bool(sep_trained and k >= 2 and len(named_sounds) >= 1)
    reason = (
        None
        if can_sep
        else (
            "Separation not available for this recording: only 1 source estimated."
            if k < 2
            else (
                "Separation not available for this recording: no trained general separator checkpoint is active."
                if not sep_trained
                else "Separation not available for this recording: no distinct sound queries detected above threshold."
            )
        )
    )
    return {
        "can_separate": can_sep,
        "trained": sep_trained,
        "speech_mode": False,
        "speech_separator": speech_sep,
        "max_separable": max_sep,
        "reason": reason,
    }


@app.post("/api/analyze")
@auth.login_required
def analyze():
    """Decode, count with uncertainty, run open-vocabulary tagging, and cluster speakers."""
    user = auth.get_current_user()
    if not check_rate_limit():
        return jsonify({"error": "Rate limit exceeded. Please wait a moment and try again."}), 429
    status = model_status()
    if not status["trained"]:
        return jsonify(
            {
                "trained": False,
                "message": "Trained checkpoint not found. Run `python scripts/download_models.py` (or train with `python scripts/train.py all --full`).",
                "model_error": status["model_error"],
            }
        ), 200

    timings = {}
    try:
        t0 = time.perf_counter()
        job_id, _, original_path, samples, rate, name = read_upload()
        auth.register_user_job(job_id, user["id"], name)
        clip, crop_start, duration = analysis_clip(samples, rate)
        timings["decode_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        tf = time.perf_counter()
        shared_logmel = audio.waveform_to_logmel(clip)
        timings["feature_ms"] = round((time.perf_counter() - tf) * 1000, 1)

        t1 = time.perf_counter()
        result = counting.count_with_uncertainty(
            clip, MODEL_PATH, MC_PASSES, precomputed_logmel=shared_logmel
        )
        timings["count_ms"] = round((time.perf_counter() - t1) * 1000, 1)

        t2 = time.perf_counter()
        gt = data_mod.lookup_ground_truth(original_path, allow_in_eval=True)
        timings["gt_ms"] = round((time.perf_counter() - t2) * 1000, 1)

        t3 = time.perf_counter()
        tags_open = tagging.tag_open_vocabulary(
            clip,
            sr=SAMPLE_RATE,
            precomputed_logmel=shared_logmel,
            use_clap_fallback=(os.getenv("ECHOCOUNT_FAST_TAG", "0") != "1"),
            localize_windows=True,
        )
        timings["tag_ms"] = round((time.perf_counter() - t3) * 1000, 1)

        t4 = time.perf_counter()
        speech_detected = bool(tags_open.get("speech_detected", False))
        speaker_clustering = separation.estimate_distinct_speakers(
            clip,
            model_count=result["rounded_count"],
            sample_rate=SAMPLE_RATE,
            speech_detected=speech_detected,
        )
        timings["speaker_cluster_ms"] = round((time.perf_counter() - t4) * 1000, 1)
    except audio.AudioValidationError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception:
        app.logger.exception("Analysis failed")
        return jsonify(
            {"error": "Could not run the trained model. Check the checkpoint and server logs."}
        ), 503

    named_sounds = tags_open.get("named_sounds", [])
    overlapping_sounds = tags_open.get("overlapping_sounds", [])
    k = int(result["rounded_count"])
    last_reliable_k, _, _ = get_dense_thresholds()
    is_pure_speech = bool(
        speech_detected and not any(s.get("group") != "Human voice" for s in named_sounds)
    )
    sep_state = _separation_eligibility(
        k,
        last_reliable_k,
        is_pure_speech,
        named_sounds,
        status["separator"],
        status["speech_separator"],
    )

    with _jobs_lock:
        _jobs[job_id] = {
            "created": time.time(),
            "name": name,
            "original": original_path,
            "input_rate": rate,
            "crop_start": crop_start,
            "duration": duration,
            "count": result,
            "speech_detected": is_pure_speech,
            "named_sounds": named_sounds,
            "speaker_clustering": speaker_clustering,
            "stems": None,
            "zip": None,
            "sep_status": "idle",
            "sep_result": None,
        }

    spread = int(np.ceil(result["total_std"] - 1e-9)) if result["total_std"] >= 0.5 else 0
    payload = {
        "trained": True,
        "job_id": job_id,
        "filename": name,
        "audio": {
            "duration_seconds": round(duration, 2),
            "input_sample_rate": rate,
            "analysed_window_seconds": CLIP_SECONDS,
            "crop_start_seconds": round(crop_start, 3),
        },
        "count": {
            "raw_regression_value": round(result["raw_regression_value"], 4),
            "calibrated_value": round(
                result.get("calibrated_value", result["raw_regression_value"]), 4
            ),
            "calibrated": result.get("calibrated", False),
            "sparse_regime": result.get("sparse_regime", False),
            "out_of_range": result.get("out_of_range", False),
            "range_warning": result.get("range_warning"),
            "dense_warning": get_dense_mixture_warning(k),
            "rounded_count": k,
            "uncertainty_std": round(result["total_std"], 3),
            "mc_dropout_std": round(result["mc_std"], 3),
            "softmax_std": round(
                result.get(
                    "softmax_std", max(0.0, result["total_std"] ** 2 - result["mc_std"] ** 2) ** 0.5
                ),
                3,
            ),
            "mc_passes": result["passes"],
            "display": f"≈{k} (±{spread})",
            "confidence_percent": round(result["confidence_percent"], 1),
            "min_sources": result["min_sources"],
            "max_sources": result["max_sources"],
        },
        "named_sounds": named_sounds,
        "overlapping_sounds": overlapping_sounds,
        "speaker_clustering": speaker_clustering,
        "separation": sep_state,
        "tags": {
            "available": bool(tags_open.get("available", True)),
            "hierarchical": True,
            "open_vocabulary": True,
            "groups": tags_open.get("groups", []),
            "active_groups": tags_open.get("active_groups", []),
            "named_sounds": named_sounds,
            "overlapping_sounds": overlapping_sounds,
            "max_simultaneous_sounds": tags_open.get("max_simultaneous_sounds", len(named_sounds)),
            "speech_detected": speech_detected,
            "tags": tags_open.get("tags", []),
            "caveat": tags_open.get(
                "caveat",
                "Named sounds include friendly labels, confidence, and frame-level time ranges.",
            ),
        },
        "timings": timings,
    }
    if gt is not None:
        payload["ground_truth"] = {
            "available": True,
            "true_k": gt["true_k"],
            "true_labels": gt["true_labels"],
            "true_groups": gt["true_groups"],
            "family": gt["family"],
            "dataset": gt["dataset"],
            "split": gt["split"],
            "match_method": "audio_hash",
            **({"sha256": gt["sha256"]} if data_mod.is_dev_mode() else {}),
        }
    return jsonify(payload)


def _execute_separation_job(job_id: str, job: dict) -> tuple[dict, int]:
    count, k, speech_mode = (
        job["count"],
        int(job["count"]["rounded_count"]),
        bool(job.get("speech_detected", False)),
    )
    sep = separation.separator_status()
    speech_sep_st = separation.pretrained_speech_status()
    max_spk = int(speech_sep_st.get("max_supported_speakers") or 2)
    last_reliable_k, _, _ = get_dense_thresholds()

    if not sep.get("trained", False) and not (speech_mode and speech_sep_st.get("available")):
        return {
            "error": "Separation not available for this recording: no trained separation model is active."
        }, 503
    if k < 2:
        return {"error": f"Separation not available for this recording: estimated {k} source."}, 409
    if k > last_reliable_k or k > separation.MAX_SEPARABLE:
        return {"error": "Too many overlapping sounds to separate reliably."}, 409
    if speech_mode and k > max_spk:
        return {
            "error": f"Separation is supported for up to {max_spk} speakers with the installed model."
        }, 409

    timings = {}
    try:
        t0 = time.perf_counter()
        samples, _ = audio.decode_mono(job["original"], MAX_DURATION_SECONDS)
        full_16k = audio.resample(samples, job["input_rate"], SAMPLE_RATE).astype(np.float32)
        q_out = separation.separate_by_queries(
            full_16k,
            k=k,
            named_sounds=job.get("named_sounds", []),
            speech_mode=speech_mode,
            sr=SAMPLE_RATE,
        )
        if not q_out.get("available") or not q_out.get("stems"):
            return {
                "error": q_out.get("reason") or "Separation not available for this recording."
            }, 409
        stems, residual_16k, queries, used_speech = (
            q_out["stems"],
            q_out["residual_stem"],
            q_out["queries"],
            bool(q_out.get("used_pretrained_speech", False)),
        )
        timings["separate_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    except Exception:
        app.logger.exception("Separation failed")
        return {"error": "Separation failed. Check the checkpoint and server logs."}, 503

    t_gate = time.perf_counter()
    sq = quality_mod.compute_stem_quality(stems, full_16k, residual=residual_16k)
    sc_gate = quality_mod.evaluate_stem_self_consistency(stems, queries)
    spk_ver = separation.verify_separated_speaker_stems(
        stems, sample_rate=SAMPLE_RATE, speech_mode=speech_mode
    )
    timings["quality_gate_ms"] = round((time.perf_counter() - t_gate) * 1000, 1)

    quality_failed = bool(sq["failed"] or (speech_mode and spk_ver["could_not_separate_speakers"]))
    quality_reason = sq["failure_reason"]
    if speech_mode and spk_ver["could_not_separate_speakers"]:
        quality_reason = (
            f"{spk_ver['reason']} Also: {quality_reason}" if quality_reason else spk_ver["reason"]
        )

    rate, job_dir = job["input_rate"], JOB_DIR / job_id
    all_waves = list(stems) + ([residual_16k] if residual_16k is not None else [])
    energies = np.array([float(np.sum(np.square(s))) for s in all_waves])
    shares = (energies / (float(energies.sum()) or 1.0)).tolist()

    t1 = time.perf_counter()
    stem_files, stem_entries = [], []
    for i, (stem, q_info, sc_info) in enumerate(zip(stems, queries, sc_gate["per_stem"]), 1):
        lbl = str(q_info.get("name") or q_info.get("friendly_name") or f"Sound {i}")
        fname = f"stem_{i:02d}_{_slugify(lbl)}.wav"
        (job_dir / fname).write_bytes(
            audio.wav_bytes(audio.peak_normalise(audio.resample(stem, SAMPLE_RATE, rate)), rate)
        )
        stem_files.append(fname)
        stem_entries.append(
            {
                "index": i,
                "label": lbl,
                "group": q_info.get("group", ""),
                "time_ranges": q_info.get("time_ranges", []),
                "file": fname,
                "is_residual": False,
                "energy_share": round(shares[i - 1], 4),
                "self_consistent": sc_info["self_consistent"],
                "low_confidence": sc_info["low_confidence"],
                "status_badge": sc_info["status_badge"],
                "top_predicted_sound": sc_info["top_predicted_sound"],
                "url": f"/api/stem/{job_id}/{fname}",
                "download_url": f"/api/stem/{job_id}/{fname}?download=1",
            }
        )

    if residual_16k is not None:
        res_fname = "residual.wav"
        (job_dir / res_fname).write_bytes(
            audio.wav_bytes(
                audio.peak_normalise(audio.resample(residual_16k, SAMPLE_RATE, rate)), rate
            )
        )
        stem_files.append(res_fname)
        stem_entries.append(
            {
                "index": len(stem_entries) + 1,
                "label": "Everything else (Residual)",
                "group": "Residual",
                "time_ranges": [],
                "file": res_fname,
                "is_residual": True,
                "energy_share": round(shares[-1], 4),
                "self_consistent": True,
                "low_confidence": False,
                "status_badge": "Residual (mix − stems)",
                "top_predicted_sound": "Residual",
                "url": f"/api/stem/{job_id}/{res_fname}",
                "download_url": f"/api/stem/{job_id}/{res_fname}?download=1",
            }
        )

    manifest = {
        "trained": True,
        "estimated_count": k,
        "raw_regression_value": count["raw_regression_value"],
        "uncertainty": {
            "std": count["total_std"],
            "mc_dropout_std": count["mc_std"],
            "method": "mc_dropout",
            "passes": count["passes"],
        },
        "stems_produced": len(stems),
        "includes_residual_stem": residual_16k is not None,
        "final_residual_energy_ratio": q_out.get("residual_energy_ratio", 0.0),
        "sample_rate": rate,
        "separation_processing_rate": SAMPLE_RATE,
        "duration_seconds": round(job["duration"], 2),
        "model_version": q_out.get("model_name", "QueryConditionedWiener"),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "stems": [
            {
                "file": e["file"],
                "label": e["label"],
                "is_residual": e["is_residual"],
                "energy_share": e["energy_share"],
                "self_consistent": e["self_consistent"],
                "low_confidence": e["low_confidence"],
            }
            for e in stem_entries
        ],
        "quality_check": sq,
        "self_consistency": sc_gate,
        "speaker_verification": spk_ver,
        "source_file": job["name"],
    }

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for fn in stem_files:
            archive.write(job_dir / fn, fn)
        archive.write(job["original"], f"original_mix{job['original'].suffix}")
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
    timings["zip_ms"] = round((time.perf_counter() - t1) * 1000, 1)

    result_payload = {
        "job_id": job_id,
        "status": "completed",
        "trained": True,
        "speech_mode": speech_mode,
        "used_pretrained_speech": used_speech,
        "pretrained_speech_status": speech_sep_st,
        "stems": stem_entries,
        "stopped_early": False,
        "quality_check": sq,
        "self_consistency": sc_gate,
        "speaker_verification": spk_ver,
        "quality_failed": quality_failed,
        "quality_reason": quality_reason,
        "download_url": f"/api/download/{job_id}.zip",
        "timings": timings,
    }
    with _jobs_lock:
        job.update(
            {
                "stems": stem_files,
                "zip": buffer.getvalue(),
                "used_pretrained_speech": used_speech,
                "sep_status": "completed",
                "sep_result": result_payload,
            }
        )
    return result_payload, 200


@app.post("/api/separate")
@auth.login_required
def separate_route():
    user = auth.get_current_user()
    if not check_rate_limit():
        return jsonify({"error": "Rate limit exceeded. Please wait a moment and try again."}), 429
    payload = request.get_json(silent=True) or {}
    job_id = str(payload.get("job_id", ""))
    if not auth.user_owns_job(job_id, user["id"]):
        return jsonify({"error": "That analysis expired or does not exist."}), 404
    job = get_job(job_id)
    if job is None:
        return jsonify(
            {"error": "That analysis expired or does not exist. Analyze the file again."}
        ), 404
    if payload.get("async"):
        with _jobs_lock:
            job["sep_status"] = "running"

        def _bg():
            res, code = _execute_separation_job(job_id, job)
            if code != 200:
                with _jobs_lock:
                    job["sep_status"], job["sep_result"] = "failed", res

        _executor.submit(_bg)
        return jsonify(
            {"job_id": job_id, "status": "running", "poll_url": f"/api/separate/{job_id}"}
        ), 202
    res, code = _execute_separation_job(job_id, job)
    return jsonify(res), code


@app.get("/api/separate/<job_id>")
@auth.login_required
def separate_status_route(job_id: str):
    user = auth.get_current_user()
    if not auth.user_owns_job(job_id, user["id"]):
        return jsonify({"error": "Job expired or does not exist."}), 404
    job = get_job(job_id)
    if job is None:
        return jsonify({"error": "Job expired or does not exist."}), 404
    st = job.get("sep_status", "idle")
    if st == "completed" and job.get("sep_result"):
        return jsonify(job["sep_result"]), 200
    if st == "failed":
        return jsonify({"job_id": job_id, "status": "failed", **(job.get("sep_result") or {})}), 409
    return jsonify({"job_id": job_id, "status": st}), 200


@app.get("/api/stem/<job_id>/<filename>")
@auth.login_required
def stem_preview(job_id: str, filename: str):
    user = auth.get_current_user()
    if not auth.user_owns_job(job_id, user["id"]):
        return jsonify({"error": "Stem not found or expired."}), 404
    job = get_job(job_id)
    if job is None or not job["stems"] or filename not in job["stems"]:
        return jsonify({"error": "Stem not found or expired."}), 404
    as_dl = request.args.get("download") == "1"
    return send_file(
        JOB_DIR / job_id / filename,
        mimetype="audio/wav",
        as_attachment=as_dl,
        download_name=filename if as_dl else None,
        max_age=0,
    )


@app.get("/api/download/<job_id>.zip")
@auth.login_required
def download(job_id: str):
    user = auth.get_current_user()
    if not auth.user_owns_job(job_id, user["id"]):
        return jsonify({"error": "This download has expired or was never created."}), 404
    job = get_job(job_id)
    if job is None or not job["zip"]:
        return jsonify({"error": "This download has expired or was never created."}), 404
    return send_file(
        io.BytesIO(job["zip"]),
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{Path(job['name']).stem or 'mix'}_stems.zip",
        max_age=0,
    )


@app.post("/api/predict")
@auth.login_required
def predict():
    audio_file = request.files.get("audio")
    if audio_file is None or not audio_file.filename:
        return jsonify({"error": "Choose an audio file first."}), 400
    suffix = Path(audio_file.filename.replace("\\", "/")).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported file type. Upload WAV, FLAC, MP3, OGG, or M4A."}), 400
    display_name = secure_filename(audio_file.filename) or f"audio{suffix}"
    temp_path = None
    try:
        fd, temp_name = tempfile.mkstemp(prefix="echocount_", suffix=suffix, dir=UPLOAD_DIR)
        os.close(fd)
        temp_path = Path(temp_name)
        audio_file.save(temp_path)
        if temp_path.stat().st_size > MAX_UPLOAD_BYTES:
            return jsonify(
                {"error": f"The upload exceeds the {MAX_UPLOAD_BYTES / 1048576:g} MB file limit."}
            ), 413
        waveform, info = audio.load_audio(temp_path)
        dur = info["original_duration_seconds"]
        meta = {
            "filename": display_name,
            "original_duration_seconds": round(dur, 2),
            "duration_seconds": round(dur, 2),
            "duration_formatted": f"{int(dur // 60)}:{int(dur % 60):02d}",
            "original_sample_rate": info["original_sample_rate"],
            "processed_sample_rate": SAMPLE_RATE,
            "channels_processed": 1,
            "peak_amplitude": round(float(np.max(np.abs(waveform))) if waveform.size else 0.0, 4),
            "rms_energy": round(
                float(np.sqrt(np.mean(np.square(waveform)))) if waveform.size else 0.0, 4
            ),
            "model_window_seconds": CLIP_SECONDS,
            "crop_start_seconds": round(info["crop_start_seconds"], 3),
        }
        if not model_status()["trained"]:
            return jsonify(
                {
                    "trained": False,
                    "status": "model_not_trained",
                    "message": "No compatible trained weights exist yet.",
                    "audio": meta,
                    "prediction": None,
                }
            )
        res = counting.predict_source_count(waveform, MODEL_PATH)
        crop = info["crop_start_seconds"]
        return jsonify(
            {
                "trained": True,
                "status": "ok",
                "message": "Prediction completed.",
                "audio": meta,
                "prediction": {
                    "estimated_sources": res["estimated_sources"],
                    "raw_value": round(res["raw_value"], 4),
                    "confidence": round(res["confidence"], 1),
                    "total_windows": res["total_windows"],
                    "min_sources": res["min_sources"],
                    "max_sources": res["max_sources"],
                    "timeline": [
                        {
                            **w,
                            "start": round(w["start"] + crop, 3),
                            "end": round(w["end"] + crop, 3),
                            "confidence": round(w["confidence"], 4),
                        }
                        for w in res["timeline"]
                    ],
                },
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception:
        app.logger.exception("Audio prediction failed")
        return jsonify({"error": "Could not run the trained model."}), 503
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    app.run(debug=False, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")))
