"""Flask web application and API for EchoCount."""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
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
from flask import (
    Flask,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    send_file,
    session as flask_session,
    url_for,
)
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename

import audio
import auth
import capabilities
import counting
import data as data_mod
import mail as mail_mod
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
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
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
    """Expose public stats, auth state, preview mode, dev flag, and job retention TTL to Jinja."""
    user = auth.get_current_user()
    admin_preview = bool(
        user and user.get("role") == "admin" and request.args.get("preview") == "user"
    )
    return {
        "public_stats": load_public_summary(),
        "capabilities": capabilities.load_registry(),
        "is_dev": data_mod.is_dev_mode(),
        "job_ttl_minutes": max(1, round(JOB_TTL_SECONDS / 60)),
        "current_user": user,
        "admin_preview": admin_preview,
        "effective_role": (
            "user" if admin_preview else (user.get("role", "user") if user else "guest")
        ),
        "csrf_token": auth.get_or_create_form_csrf(),
        "oauth_providers": auth.provider_availability(),
        "google_auth_mode": auth.google_auth_mode(),
        "firebase_web_config": auth.firebase_web_config(),
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
@auth.login_required
def index():
    user = auth.get_current_user()
    recent_jobs = auth.list_user_jobs(user["id"], limit=8) if user else []
    return render_template("index.html", status=model_status(), recent_jobs=recent_jobs)


@app.get("/analyze")
@auth.login_required
def analyze_page():
    return render_template("analyze.html", status=model_status())


@app.get("/evaluation")
@app.get("/admin/evaluation")
@auth.role_required("developer", "admin")
def evaluation_page():
    return render_template("evaluation.html", status=model_status())


@app.get("/admin")
@auth.role_required("admin")
def admin_page():
    return render_template("admin.html", status=model_status())


@app.get("/history")
@auth.login_required
def history_page():
    user = auth.get_current_user()
    server_jobs = auth.list_user_jobs(user["id"], limit=50) if user else []
    return render_template("history.html", status=model_status(), server_jobs=server_jobs)


@app.get("/how")
@auth.login_required
def how_page():
    return render_template(
        "how.html",
        status=model_status(),
        how_content=_read_json_file(HOW_CONTENT_PATH) or {},
    )


@app.get("/docs")
@auth.login_required
def docs_page():
    return render_template("docs.html", status=model_status())


@app.after_request
def apply_security_and_cache_headers(resp):
    """Apply baseline security headers to all responses and no-store to private routes."""
    p = request.path or ""
    if p.startswith(
        (
            "/api/history",
            "/api/my-jobs",
            "/api/admin/",
            "/api/account",
            "/api/auth/",
            "/api/me",
            "/api/analyze",
            "/api/jobs/",
            "/api/separate",
            "/api/stem/",
            "/api/download/",
            "/settings/account",
        )
    ):
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
        resp.headers["Pragma"] = "no-cache"

    g_connect = ""
    g_frame = ""
    if auth.provider_availability().get("google"):
        if auth.google_auth_mode() == "firebase":
            g_connect = " https://identitytoolkit.googleapis.com https://securetoken.googleapis.com https://www.googleapis.com"
            g_frame = " https://accounts.google.com https://*.firebaseapp.com"
        else:
            g_connect = " https://accounts.google.com https://oauth2.googleapis.com"

    resp.headers["Content-Security-Policy"] = (
        f"default-src 'self'; "
        f"script-src 'self' 'unsafe-inline'; "
        f"style-src 'self' 'unsafe-inline'; "
        f"img-src 'self' data: blob:; "
        f"media-src 'self' blob:; "
        f"connect-src 'self'{g_connect}; "
        f"frame-src 'self'{g_frame}; "
        f"frame-ancestors 'none'; "
        f"base-uri 'self'; "
        f"form-action 'self' https://accounts.google.com"
    )
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Permissions-Policy"] = "camera=(), geolocation=(), microphone=()"
    if auth.is_https_base() or request.is_secure:
        resp.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return resp


def _purge_job_files(job_ids: list[str]) -> tuple[int, bool]:
    """Remove in-memory job state and on-disk uploads/jobs/<job_id> dirs; verify gone."""
    if not job_ids:
        return 0, True
    with _jobs_lock:
        for jid in job_ids:
            _jobs.pop(jid, None)
    removed = 0
    all_gone = True
    for jid in job_ids:
        if jid and all(c in "0123456789abcdef" for c in jid):
            d = JOB_DIR / jid
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
            if d.exists():
                all_gone = False
    return removed, all_gone


def _csv_cell(val) -> str:
    """Neutralize spreadsheet formula injection in user-controlled CSV cells."""
    s = "" if val is None else str(val)
    return f"'{s}" if s and s[0] in ("=", "+", "-", "@", "\t", "\r") else s


def _history_csv_response(
    items: list[dict], include_user: bool = False, filename: str = "echocount-history.csv"
):
    buf = io.StringIO()
    w = csv.writer(buf)
    header = (
        ["created_at", "job_id"]
        + (["user_email"] if include_user else [])
        + [
            "filename",
            "estimated_sources",
            "confidence_percent",
            "duration_seconds",
            "top_sounds",
            "status",
        ]
    )
    w.writerow(header)
    for it in items:
        s = it.get("summary") or {}
        ts = datetime.fromtimestamp(float(it.get("created_at") or 0), UTC).isoformat(
            timespec="seconds"
        )
        sounds = "; ".join(str(x) for x in (s.get("top_sounds") or []))
        row = [ts, it.get("job_id", "")]
        if include_user:
            row.append(_csv_cell(it.get("user_email", "")))
        row.extend(
            [
                _csv_cell(it.get("filename", "")),
                s.get("rounded_count", ""),
                s.get("confidence_percent", ""),
                s.get("duration_seconds", ""),
                _csv_cell(sounds),
                it.get("status", "analyzed"),
            ]
        )
        w.writerow(row)
    resp = make_response(buf.getvalue())
    resp.headers["Content-Type"] = "text/csv; charset=utf-8"
    resp.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return resp


@app.get("/api/my-jobs")
@auth.login_required
def my_jobs_route():
    user = auth.get_current_user()
    return jsonify({"jobs": auth.list_user_jobs(user["id"], limit=30)})


@app.get("/api/history")
@auth.login_required
def history_list_route():
    """Return ONLY the authenticated user's own analysis history (never accepts client user_id)."""
    user = auth.get_current_user()
    return jsonify({"items": auth.list_user_jobs(user["id"], limit=100), "scope": "self"})


@app.delete("/api/history/<job_id>")
@auth.login_required
@auth.csrf_required
def history_delete_item_route(job_id: str):
    user = auth.get_current_user()
    if not auth.delete_user_job(job_id, user):
        return jsonify({"error": "History item not found."}), 404
    _purge_job_files([job_id])
    return jsonify({"ok": True, "deleted": job_id})


@app.delete("/api/history")
@auth.login_required
@auth.csrf_required
def history_clear_route():
    user = auth.get_current_user()
    deleted_ids = auth.clear_user_jobs(user["id"])
    _purge_job_files(deleted_ids)
    return jsonify({"ok": True, "deleted_count": len(deleted_ids)})


@app.get("/api/history/export.csv")
@auth.login_required
def history_export_csv_route():
    user = auth.get_current_user()
    items = auth.list_user_jobs(user["id"], limit=500)
    return _history_csv_response(items, include_user=False, filename="echocount-history.csv")


def _parse_optional_uid() -> tuple[int | None, tuple | None]:
    raw = (request.args.get("user_id") or "").strip()
    if not raw:
        return None, None
    try:
        return int(raw), None
    except ValueError:
        return None, (jsonify({"error": "Invalid user_id filter."}), 400)


@app.get("/api/admin/history")
@auth.role_required("admin")
def admin_history_list_route():
    """Admin-only view of all users' analysis history with optional user_id filter."""
    filter_uid, err = _parse_optional_uid()
    if err:
        return err
    items = auth.list_all_jobs(user_id=filter_uid, limit=250)
    users = [{"id": u["id"], "email": u["email"], "role": u["role"]} for u in auth.list_users()]
    return jsonify(
        {"items": items, "users": users, "scope": "all_users", "filter_user_id": filter_uid}
    )


@app.get("/api/admin/history/export.csv")
@auth.role_required("admin")
def admin_history_export_csv_route():
    filter_uid, err = _parse_optional_uid()
    if err:
        return err
    items = auth.list_all_jobs(user_id=filter_uid, limit=1000)
    return _history_csv_response(
        items, include_user=True, filename="echocount-all-users-history.csv"
    )


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "model": model_status()})


@app.get("/api/public-stats")
def public_stats_route():
    summary = load_public_summary()
    if summary is None:
        return jsonify(
            {"available": False, "reason": "Public summary has not been exported yet."}
        ), 404
    return jsonify(summary)


@app.get("/api/capabilities")
def capabilities_route():
    """Return the data-driven separation & counting capability registry."""
    return jsonify(capabilities.load_registry())


def _safe_next_url(raw: str | None, role: str | None = None) -> str:
    fallback = auth.default_landing_for_role(role)
    target = (raw or "").strip()
    if not target or target in ("/login", "/register"):
        return fallback
    return target if target.startswith("/") and not target.startswith("//") else fallback


@app.get("/login")
def login_page():
    user = auth.get_current_user()
    if user:
        return redirect(_safe_next_url(request.args.get("next"), user.get("role")))
    return render_template(
        "auth.html",
        status=model_status(),
        mode="login",
        next_url=request.args.get("next", ""),
    )


@app.get("/register")
def register_page():
    user = auth.get_current_user()
    if user:
        return redirect(_safe_next_url(request.args.get("next"), user.get("role")))
    return render_template(
        "auth.html",
        status=model_status(),
        mode="register",
        next_url=request.args.get("next", ""),
    )


@app.get("/forgot-password")
def forgot_password_page():
    return render_template("auth.html", status=model_status(), mode="forgot", next_url="/analyze")


@app.get("/api/auth/me")
@app.get("/api/me")
def auth_me():
    user = auth.get_current_user()
    if not user:
        return jsonify({"authenticated": False}), 200
    is_admin = user.get("role") == "admin"
    return jsonify(
        {
            "authenticated": True,
            "display_label": "Admin" if is_admin else user["email"],
            "email": user["masked_email"] if is_admin else user["email"],
            "masked_email": user["masked_email"],
            "email_verified": user["email_verified"],
            "has_password": user.get("has_password", True),
            "role": user.get("role", "user"),
        }
    )


def _issue_login_session(user_id: int, role: str, next_raw: str | None = None):
    prev_sid = request.cookies.get(auth.SESSION_COOKIE)
    sid, _ = auth.create_session(user_id, rotate_from_sid=prev_sid)
    is_admin = role == "admin"
    flask_session.pop("_otp_pending", None)
    resp = make_response(
        jsonify(
            {
                "ok": True,
                "role": role,
                "display_label": "Admin" if is_admin else "",
                "redirect": _safe_next_url(next_raw, role),
            }
        )
    )
    return auth.set_session_cookie(resp, sid, is_admin=is_admin)


@app.post("/auth/login")
@auth.csrf_required
def auth_login_route():
    if not auth.verify_captcha_hook(request):
        return jsonify({"error": "Verification check failed."}), 400
    data = request.get_json(silent=True) or request.form
    email = auth.normalize_email(str(data.get("email", "")))
    password = str(data.get("password", ""))
    ip = request.remote_addr or "local"
    ip_h = auth.hash_ip(ip)
    if (
        auth.is_locked_out(f"fail:email:{email}")
        or auth.is_locked_out(f"fail:ip:{ip_h}", max_failures=15)
        or not auth.check_auth_rate_limit(f"login:ip:{ip_h}", limit=12, window_s=300.0)
        or not auth.check_auth_rate_limit(f"login:email:{email}", limit=6, window_s=300.0)
    ):
        return jsonify(
            {"error": "Too many sign-in attempts. Please wait a few minutes and try again."}
        ), 429

    user, err = auth.authenticate_password_user(email, password)
    if err or not user:
        auth.record_login_failure(f"fail:email:{email}")
        auth.record_login_failure(f"fail:ip:{ip_h}")
        return jsonify({"error": "Invalid email or password."}), 401

    auth.clear_login_failures(f"fail:email:{email}")
    next_raw = data.get("next")

    # Optional admin step-up OTP (off by default: ADMIN_REQUIRE_OTP=0)
    if user["role"] == "admin":
        if not auth.admin_requires_otp():
            return _issue_login_session(user["id"], "admin", next_raw)
        ok_otp, otp_err, retry_s = auth.issue_email_otp(email, "admin_login", ip=ip)
        if not ok_otp:
            return jsonify(
                {"error": otp_err or "Email verification is not available right now."}
            ), 503 if retry_s == 0 else 429
        flask_session["_otp_pending"] = {
            "uid": user["id"],
            "email": email,
            "purpose": "admin_login",
            "role": "admin",
            "next": str(next_raw or ""),
            "ts": time.time(),
        }
        return jsonify(
            {
                "ok": True,
                "otp_required": True,
                "purpose": "admin_login",
                "cooldown": retry_s,
                "message": "Enter the 6-digit verification code sent to your email to continue.",
            }
        )

    # Unverified new sign-ups: send fresh code and open code-only verify view (no session until verified)
    if not user["email_verified"]:
        ok_otp, otp_err, retry_s = auth.issue_email_otp(email, "signup", ip=ip)
        if not ok_otp:
            return jsonify(
                {"error": otp_err or "Email verification is not available right now."}
            ), 503 if retry_s == 0 else 429
        flask_session["_otp_pending"] = {
            "uid": user["id"],
            "email": email,
            "purpose": "signup",
            "role": "user",
            "next": str(next_raw or "/analyze"),
            "ts": time.time(),
        }
        return jsonify(
            {
                "ok": True,
                "otp_required": True,
                "purpose": "signup",
                "masked_email": auth.mask_email(email),
                "cooldown": retry_s,
                "message": "Enter the 6-digit code sent to your email to verify your account.",
            }
        )

    # Deactivated accounts reactivate directly upon valid email + password sign-in (no OTP)
    if not user["is_active"]:
        auth.reactivate_user_account(user["id"])

    return _issue_login_session(user["id"], user["role"], next_raw)


def _render_auth_error(msg: str, code: int = 400):
    return (
        render_template(
            "auth.html", status=model_status(), mode="login", next_url="", auth_error=msg
        ),
        code,
    )


@app.post("/auth/register")
@auth.csrf_required
def auth_register_route():
    if not auth.verify_captcha_hook(request):
        return jsonify({"error": "Verification check failed."}), 400
    data = request.get_json(silent=True) or request.form
    email = auth.normalize_email(str(data.get("email", "")))
    password = str(data.get("password", ""))
    password_confirm = str(data.get("password_confirm", ""))
    ip = request.remote_addr or "local"
    if not auth.check_auth_rate_limit(f"reg:ip:{auth.hash_ip(ip)}", limit=10, window_s=600.0):
        return jsonify({"error": "Too many registration attempts. Please try again later."}), 429
    if not password_confirm or password != password_confirm:
        return jsonify({"error": "Passwords do not match."}), 400

    # Reject syntax/MX/weak-password errors cleanly, then use non-enumerating generic response
    uid, _legacy_tok, err = auth.register_password_user(email, password, check_mx=True)
    if err:
        return jsonify({"error": err}), 400

    generic_msg = "If this email is eligible, a 6-digit verification code has been sent."
    masked = auth.mask_email(email)
    if uid is not None:
        ok_otp, otp_err, retry_s = auth.issue_email_otp(email, "signup", ip=ip)
        if not ok_otp:
            code_status = 429 if retry_s > 0 else 503
            return jsonify(
                {"error": otp_err or "Email verification is not available right now."}
            ), code_status
        flask_session["_otp_pending"] = {
            "uid": uid,
            "email": email,
            "purpose": "signup",
            "role": "user",
            "next": "/analyze",
            "ts": time.time(),
        }
        return jsonify(
            {
                "ok": True,
                "otp_required": True,
                "purpose": "signup",
                "masked_email": masked,
                "cooldown": retry_s,
                "message": generic_msg,
            }
        ), 201

    # Existing verified account or admin email: no OTP created, generic reply + optional info notice
    existing = auth.get_user_by_email(email)
    if existing and existing["role"] != "admin" and existing["email_verified"]:
        mail_mod.send_transactional_email(
            email,
            "Sign-up attempt on your EchoCount account",
            "Someone tried to sign up for EchoCount with your address. Your existing account was not changed — please sign in instead.",
            db_path=auth.DB_PATH,
        )
    return jsonify(
        {
            "ok": True,
            "otp_required": True,
            "purpose": "signup",
            "masked_email": masked,
            "cooldown": auth.OTP_RESEND_COOLDOWN,
            "message": generic_msg,
        }
    ), 201


@app.post("/auth/signup/cancel")
@auth.csrf_required
def auth_signup_cancel_route():
    """Delete pending unverified account on 'Change email' and return the typed email."""
    pending = flask_session.pop("_otp_pending", None) or {}
    email = str(pending.get("email") or "")
    uid = pending.get("uid")
    deleted = bool(uid and email and auth.delete_unverified_signup(int(uid), email))
    return jsonify({"ok": True, "deleted": deleted, "email": email})


@app.post("/auth/otp/verify")
@auth.csrf_required
def auth_otp_verify_route():
    data = request.get_json(silent=True) or request.form
    purpose = str(data.get("purpose", "signup")).strip().lower()
    code = str(data.get("code", "")).strip()
    pending = flask_session.get("_otp_pending") or {}
    email = auth.normalize_email(str(data.get("email", "") or pending.get("email", "")))
    ip = request.remote_addr or "local"
    if not auth.check_auth_rate_limit(f"otpver:ip:{auth.hash_ip(ip)}", limit=25, window_s=300.0):
        return jsonify({"error": "Too many verification attempts. Please wait."}), 429

    if purpose == "admin_login":
        if (
            not auth.admin_requires_otp()
            or not pending
            or pending.get("purpose") != "admin_login"
            or (time.time() - float(pending.get("ts", 0))) > auth.OTP_TTL_SECONDS
        ):
            return jsonify({"error": "Verification session expired. Please sign in again."}), 400
        email = pending["email"]
    elif purpose not in ("signup", "forgot"):
        return jsonify({"error": "Unsupported verification flow."}), 400

    if purpose == "forgot":
        new_pw = str(data.get("password", ""))
        pw_conf = data.get("password_confirm")
        if pw_conf is not None and str(pw_conf) != new_pw:
            return jsonify({"error": "Passwords do not match."}), 400
    ok, err = auth.verify_email_otp(email, purpose, code)
    if not ok:
        return jsonify({"error": err or "Invalid or expired verification code."}), 400

    if purpose == "forgot":
        ok_pw, pw_err = auth.update_password_by_email(email, new_pw)
        if not ok_pw:
            return jsonify({"error": pw_err or "Could not update password."}), 400
        flask_session.pop("_otp_pending", None)
        return jsonify(
            {"ok": True, "message": "Password updated. You can now sign in.", "redirect": "/login"}
        )

    if purpose == "signup":
        user = auth.mark_email_verified(email)
        if not user or user["role"] == "admin":
            return jsonify({"error": "Invalid or expired verification code."}), 400
        return _issue_login_session(
            int(user["id"]), user["role"], pending.get("next") or "/analyze"
        )

    if purpose == "admin_login":
        uid = int(pending["uid"])
        return _issue_login_session(uid, "admin", pending.get("next") or "/admin")

    return jsonify({"error": "Unsupported verification flow."}), 400


@app.post("/auth/otp/resend")
@auth.csrf_required
def auth_otp_resend_route():
    data = request.get_json(silent=True) or request.form
    purpose = str(data.get("purpose", "signup")).strip().lower()
    pending = flask_session.get("_otp_pending") or {}
    email = auth.normalize_email(str(data.get("email", "") or pending.get("email", "")))
    ip = request.remote_addr or "local"

    if purpose not in ("signup", "forgot", "admin_login"):
        return jsonify({"error": "Unsupported verification flow."}), 400

    generic_msg = "If eligible, a new 6-digit verification code has been sent."
    u = auth.get_user_by_email(email)
    should_send = False
    if purpose == "signup":
        should_send = bool(u and not u["email_verified"] and u["role"] != "admin")
    elif purpose == "forgot":
        should_send = bool(u and u["role"] != "admin")
    elif purpose == "admin_login" and auth.admin_requires_otp():
        should_send = bool(
            pending and pending.get("purpose") == "admin_login" and pending.get("email") == email
        )

    if should_send:
        ok, err, retry_s = auth.issue_email_otp(email, purpose, ip=ip)
        if not ok:
            return jsonify(
                {"error": err or "Could not resend code.", "cooldown": retry_s}
            ), 429 if retry_s > 0 else 503
        return jsonify({"ok": True, "cooldown": retry_s, "message": generic_msg})

    return jsonify({"ok": True, "cooldown": auth.OTP_RESEND_COOLDOWN, "message": generic_msg})


@app.post("/auth/forgot")
@auth.csrf_required
def auth_forgot_route():
    data = request.get_json(silent=True) or request.form
    email = auth.normalize_email(str(data.get("email", "")))
    ip = request.remote_addr or "local"
    if not auth.check_auth_rate_limit(f"forgot:ip:{auth.hash_ip(ip)}", limit=6, window_s=600.0):
        return jsonify({"error": "Too many reset requests. Please wait a few minutes."}), 429
    generic_msg = (
        "If an eligible account exists for that email, a 6-digit reset code has been sent."
    )
    u = auth.get_user_by_email(email)
    # Never allow password reset via website for the locked admin account (generic non-enumerating reply)
    if u and u["role"] != "admin" and not auth.is_admin_email(email):
        ok, err, retry_s = auth.issue_email_otp(email, "forgot", ip=ip)
        if not ok and retry_s == 0:
            return jsonify({"error": err or "Email verification is not available right now."}), 503
        if not ok and retry_s > 0:
            return jsonify({"error": err, "cooldown": retry_s}), 429
    flask_session["_otp_pending"] = {"email": email, "purpose": "forgot", "ts": time.time()}
    return jsonify(
        {
            "ok": True,
            "otp_required": True,
            "purpose": "forgot",
            "masked_email": auth.mask_email(email),
            "cooldown": auth.OTP_RESEND_COOLDOWN,
            "message": generic_msg,
        }
    )


@app.post("/auth/logout")
@auth.csrf_required
def auth_logout_route():
    user = auth.get_current_user()
    if user:
        auth.revoke_session(user["session_id"])
    flask_session.pop("_otp_pending", None)
    resp = make_response(jsonify({"ok": True, "redirect": "/login"}))
    return auth.clear_session_cookie(resp)


@app.get("/auth/oauth/<provider>")
def oauth_start(provider: str):
    if provider != "google":
        return jsonify({"error": "Unsupported authentication provider."}), 404
    if auth.google_auth_mode() == "firebase":
        return redirect(url_for("login_page"))
    client = auth.get_oauth_client("google")
    if client is None:
        return _render_auth_error("Google sign-in is not set up on this server yet.", 503)
    next_raw = (request.args.get("next") or "").strip()
    if next_raw.startswith("/") and not next_raw.startswith("//"):
        flask_session["_oauth_next"] = next_raw
    redirect_uri = url_for("oauth_callback", provider="google", _external=True)
    return client.authorize_redirect(redirect_uri)


@app.get("/auth/oauth/<provider>/callback")
def oauth_callback(provider: str):
    if provider != "google":
        return jsonify({"error": "Unsupported authentication provider."}), 404
    client = auth.get_oauth_client("google")
    if client is None:
        return redirect(url_for("login_page"))
    try:
        token = client.authorize_access_token()
        userinfo = token.get("userinfo") or client.userinfo()
        sub = str(userinfo.get("sub") or "").strip()
        raw_email = str(userinfo.get("email") or "").strip()
        is_verified = bool(userinfo.get("email_verified") is True)
    except Exception:
        auth.logger.warning("Google OAuth callback failed.")
        return _render_auth_error("Google sign-in could not be verified. Please try again.", 400)
    if not sub or not raw_email or not is_verified:
        return _render_auth_error(
            "Google did not return a verified identity and email address.", 400
        )
    user, err = auth.register_oauth_user("google", sub, raw_email, is_verified)
    if err or not user:
        return _render_auth_error(err or "Could not sign in with Google.", 403)
    auth.mark_reauthenticated(int(user["id"]))
    prev_sid = request.cookies.get(auth.SESSION_COOKIE)
    sid, _ = auth.create_session(user["id"], rotate_from_sid=prev_sid)
    next_target = flask_session.pop("_oauth_next", None)
    resp = make_response(redirect(_safe_next_url(next_target, user.get("role"))))
    return auth.set_session_cookie(resp, sid)


@app.post("/auth/firebase/google")
def firebase_google_signin_route():
    """Optional Firebase Google ID-token verification endpoint (only when AUTH_GOOGLE_MODE=firebase)."""
    if auth.google_auth_mode() != "firebase":
        return jsonify({"error": "Firebase Google sign-in is not enabled."}), 404
    if not auth.verify_csrf_request():
        return jsonify({"error": "Invalid or expired security token."}), 403
    ip = request.remote_addr or "local"
    if not auth.check_auth_rate_limit(f"fb:ip:{auth.hash_ip(ip)}", limit=15, window_s=300.0):
        return jsonify({"error": "Too many sign-in attempts. Please wait."}), 429
    data = request.get_json(silent=True) or {}
    claims, err = auth.verify_firebase_google_token(str(data.get("id_token", "")))
    if err or not claims:
        return jsonify({"error": err or "Google sign-in could not be verified."}), 401
    user, uerr = auth.register_oauth_user("google", claims["sub"], claims["email"], True)
    if uerr or not user:
        return jsonify({"error": uerr or "Could not sign in with Google."}), 403
    auth.mark_reauthenticated(int(user["id"]))
    return _issue_login_session(int(user["id"]), user["role"], data.get("next"))


# ---------------------------------------------------------------------------
# Settings -> Account (Download My Data, Deactivate, Permanent Delete)
# ---------------------------------------------------------------------------


@app.get("/settings/account")
@auth.login_required
def account_settings_page():
    return render_template("account.html", status=model_status())


@app.get("/api/account/export")
@auth.login_required
def account_export_route():
    """Export ONLY the authenticated user's own history as JSON or CSV."""
    user = auth.get_current_user()
    items = auth.list_user_jobs(user["id"], limit=1000)
    fmt = (request.args.get("format") or "json").strip().lower()
    if fmt == "csv":
        return _history_csv_response(items, include_user=False, filename="my-echocount-data.csv")
    payload = {
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "account": {
            "display_label": user["display_label"],
            "role": user["role"],
            "email_verified": user["email_verified"],
            "created_at": user["created_at"],
        },
        "analyses": items,
    }
    resp = make_response(json.dumps(payload, indent=2))
    resp.headers["Content-Type"] = "application/json; charset=utf-8"
    resp.headers["Content-Disposition"] = 'attachment; filename="my-echocount-data.json"'
    return resp


@app.post("/api/account/deactivate")
@auth.login_required
@auth.csrf_required
def account_deactivate_route():
    user = auth.get_current_user()
    if user["role"] == "admin":
        return jsonify(
            {"error": "Administrator accounts cannot be deactivated from the website."}
        ), 403
    ip_h = auth.hash_ip(request.remote_addr or "local")
    if not auth.check_auth_rate_limit(
        f"deact:u:{user['id']}", limit=5, window_s=3600.0
    ) or not auth.check_auth_rate_limit(f"deact:ip:{ip_h}", limit=5, window_s=3600.0):
        return jsonify({"error": "Too many attempts. Please try again later."}), 429
    data = request.get_json(silent=True) or {}
    pw = str(data.get("password", ""))

    if user.get("has_password"):
        u_ok, err = auth.authenticate_password_user(user["email"], pw)
        if err or not u_ok or int(u_ok["id"]) != int(user["id"]):
            return jsonify({"error": "Invalid password confirmation."}), 401
    elif not auth.is_recently_reauthenticated(user["id"], window_s=300.0):
        return jsonify(
            {
                "error": "Please re-authenticate with Google within the last 5 minutes before deactivating."
            }
        ), 401

    ok, target_email, err = auth.deactivate_user_account(user["id"])
    if not ok:
        return jsonify({"error": err or "Could not deactivate account."}), 400
    if target_email:
        mail_mod.send_transactional_email(
            target_email,
            "Your EchoCount account was deactivated",
            "Your EchoCount account has been deactivated and all active sessions have been signed out.\n"
            "Sign in again with your email and password anytime to reactivate it.",
            db_path=auth.DB_PATH,
        )
    resp = make_response(
        jsonify({"ok": True, "status": "disabled", "redirect": "/login?deactivated=1"})
    )
    return auth.clear_session_cookie(resp)


@app.delete("/api/account")
@auth.login_required
@auth.csrf_required
def account_delete_self_route():
    user = auth.get_current_user()
    if user["role"] == "admin":
        return jsonify({"error": "Administrator accounts cannot be deleted from the website."}), 403
    ip_h = auth.hash_ip(request.remote_addr or "local")
    if not auth.check_auth_rate_limit(
        f"delacc:u:{user['id']}", limit=5, window_s=3600.0
    ) or not auth.check_auth_rate_limit(f"delacc:ip:{ip_h}", limit=5, window_s=3600.0):
        return jsonify({"error": "Too many deletion attempts. Please try again later."}), 429

    data = request.get_json(silent=True) or {}
    if str(data.get("confirmation", "")).strip() != "DELETE":
        return jsonify(
            {"error": 'Type "DELETE" exactly to confirm permanent account deletion.'}
        ), 400

    pw = str(data.get("password", ""))
    if user.get("has_password"):
        u_ok, err = auth.authenticate_password_user(user["email"], pw)
        if err or not u_ok or int(u_ok["id"]) != int(user["id"]):
            return jsonify(
                {"error": "Current password confirmation is required before permanent deletion."}
            ), 401
    elif not auth.is_recently_reauthenticated(user["id"], window_s=300.0):
        return jsonify(
            {
                "error": "Fresh Google sign-in within the last 5 minutes is required before permanent deletion."
            }
        ), 401

    ok, deleted_job_ids, deleted_email, err = auth.delete_user_account(user["id"], self_delete=True)
    if not ok:
        return jsonify({"error": err or "Could not delete account."}), 400
    removed_dirs, all_gone = _purge_job_files(deleted_job_ids)
    if deleted_email:
        mail_mod.send_transactional_email(
            deleted_email,
            "Your EchoCount account has been permanently deleted",
            "Your EchoCount account, analysis history, sessions, and uploaded audio files have been permanently removed.\n"
            "You may register again with this email address in the future if you wish.",
            db_path=auth.DB_PATH,
        )
    resp = make_response(
        jsonify(
            {
                "ok": True,
                "deleted_jobs": len(deleted_job_ids),
                "removed_job_dirs": removed_dirs,
                "files_verified_removed": all_gone,
                "redirect": "/login?deleted=1",
            }
        )
    )
    return auth.clear_session_cookie(resp)


@app.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    return jsonify(
        {"error": f"The upload exceeds the {MAX_UPLOAD_BYTES / 1048576:g} MB file limit."}
    ), 413


@app.errorhandler(404)
def not_found(_error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Resource not found."}), 404
    return _render_auth_error("Page not found.", 404)


@app.errorhandler(500)
def server_error(_error):
    app.logger.exception("Unhandled server error on %s", request.path)
    if request.path.startswith("/api/"):
        return jsonify({"error": "An internal error occurred. Please try again later."}), 500
    return _render_auth_error("An unexpected error occurred. Please try again later.", 500)


@app.get("/api/evaluation")
@auth.role_required("developer", "admin")
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
    pub = load_public_summary()
    if pub:
        payload["public_summary"] = pub
    return jsonify(payload)


@app.get("/api/evaluation/export.csv")
@auth.role_required("developer", "admin")
def evaluation_export_csv():
    """Export approved aggregate evaluation metrics from public_summary.json as CSV."""
    pub = load_public_summary()
    if not pub:
        return jsonify({"error": "Public evaluation summary not available."}), 404
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["section", "slice", "n", "exact_accuracy", "within_one", "mae", "bias", "notes"])
    c, t, sep = pub.get("counting", {}), pub.get("tagging", {}), pub.get("separation", {})
    for key, label in (
        ("overall", "Overall Test"),
        ("speech", "Speech Test"),
        ("fsd", "FSD Test"),
        ("held_out_unseen_speakers", "Held-Out Unseen Speakers"),
    ):
        s = c.get(key) or {}
        if s:
            w.writerow(
                [
                    "counting_summary",
                    label,
                    s.get("n", ""),
                    s.get("exact_accuracy", ""),
                    s.get("within_one", ""),
                    s.get("mae", ""),
                    s.get("bias", ""),
                    "",
                ]
            )
    for row in c.get("per_count_overall") or []:
        note = "unreliable (n<30)" if row.get("unreliable") else "reliable"
        w.writerow(
            [
                "counting_per_k",
                f"K={row.get('true_k')}",
                row.get("n", ""),
                row.get("exact_accuracy", ""),
                row.get("within_one", ""),
                row.get("mae", ""),
                row.get("bias", ""),
                note,
            ]
        )
    for slc, n_key, val_key in (
        ("Group Test Macro mAP", "", "group_test_macro_map"),
        ("Fine-32 Test Macro mAP", "", "fine_32_test_macro_map"),
        ("AST+CLAP Open-Vocab Macro AP", "open_vocab_held_out_clips", "ast_clap_macro_ap"),
    ):
        w.writerow(
            ["tagging", slc, t.get(n_key, "") if n_key else "", t.get(val_key, ""), "", "", "", ""]
        )
    w.writerow(
        [
            "separation",
            "Overall Synthetic Benchmark",
            sep.get("total_stems_evaluated", ""),
            sep.get("self_consistency_pass_rate", ""),
            sep.get("energy_conservation_pass_rate", ""),
            sep.get("overall_mean_sisdri_db", ""),
            "",
            "exact_accuracy=self_consistency_pass_rate; within_one=energy_pass_rate; mae=mean_sisdri_db",
        ]
    )
    resp = make_response(buf.getvalue())
    resp.headers["Content-Type"] = "text/csv; charset=utf-8"
    resp.headers["Content-Disposition"] = 'attachment; filename="echocount_evaluation_summary.csv"'
    return resp


@app.get("/api/admin/overview")
@auth.role_required("admin")
def admin_overview_api():
    """Return user list, active/failed job telemetry, temp-file disk usage, and model health."""
    sweep_jobs()
    users = auth.list_users()
    with _jobs_lock:
        active_jobs = len(_jobs)
        running_sep = sum(1 for j in _jobs.values() if j.get("sep_status") == "running")
        failed_sep = sum(1 for j in _jobs.values() if j.get("sep_status") == "failed")
        completed_sep = sum(1 for j in _jobs.values() if j.get("sep_status") == "completed")
    job_dirs = [p for p in JOB_DIR.iterdir() if p.is_dir()] if JOB_DIR.is_dir() else []
    disk_bytes = sum(f.stat().st_size for d in job_dirs for f in d.rglob("*") if f.is_file())
    return jsonify(
        {
            "ok": True,
            "users": users,
            "counts": {
                "total_users": len(users),
                "admins": sum(1 for u in users if u["role"] == "admin"),
                "developers": sum(1 for u in users if u["role"] == "developer"),
                "regular_users": sum(1 for u in users if u["role"] == "user"),
                "active_sessions": sum(u["active_sessions"] for u in users),
            },
            "system": {
                "model": model_status(),
                "active_jobs": active_jobs,
                "running_separations": running_sep,
                "completed_separations": completed_sep,
                "failed_separations": failed_sep,
                "temp_job_dirs": len(job_dirs),
                "temp_disk_mb": round(disk_bytes / 1048576, 2),
                "job_ttl_minutes": max(1, round(JOB_TTL_SECONDS / 60)),
                "separation_workers": int(os.getenv("SEPARATION_WORKERS", "2")),
                "oauth_providers": auth.provider_availability(),
            },
            "audit_log": auth.list_admin_audit(limit=50),
        }
    )


@app.post("/api/admin/users/<int:uid>/role")
@auth.role_required("admin")
@auth.csrf_required
def admin_update_role_api(uid: int):
    return jsonify(
        {
            "error": "Roles are locked and cannot be changed through the website. Edit ADMIN_EMAILS in .env and restart."
        }
    ), 403


@app.post("/api/admin/users/<int:uid>/status")
@auth.role_required("admin")
@auth.csrf_required
def admin_update_status_api(uid: int):
    actor = auth.get_current_user()
    data = request.get_json(silent=True) or request.form
    active = bool(data.get("is_active"))
    ok, err = auth.set_user_active(uid, active, actor["id"])
    if not ok:
        return jsonify({"error": err}), 400
    return jsonify({"ok": True})


@app.post("/api/admin/users/<int:uid>/revoke")
@auth.role_required("admin")
@auth.csrf_required
def admin_revoke_sessions_api(uid: int):
    revoked = auth.revoke_user_sessions(uid)
    return jsonify({"ok": True, "revoked": revoked})


@app.delete("/api/admin/users/<int:uid>")
@auth.role_required("admin")
@auth.csrf_required
def admin_delete_user_api(uid: int):
    actor = auth.get_current_user()
    ok, deleted_job_ids, _deleted_email, err = auth.delete_user_account(uid, actor["id"])
    if not ok:
        return jsonify({"error": err}), 400
    removed_dirs, all_gone = _purge_job_files(deleted_job_ids)
    return jsonify(
        {
            "ok": True,
            "deleted_jobs": len(deleted_job_ids),
            "removed_job_dirs": removed_dirs,
            "files_verified_removed": all_gone,
        }
    )


@app.post("/api/admin/cleanup")
@auth.role_required("admin")
@auth.csrf_required
def admin_cleanup_api():
    before = len([p for p in JOB_DIR.iterdir() if p.is_dir()]) if JOB_DIR.is_dir() else 0
    sweep_jobs(startup=False)
    after = len([p for p in JOB_DIR.iterdir() if p.is_dir()]) if JOB_DIR.is_dir() else 0
    return jsonify({"ok": True, "removed": max(0, before - after), "remaining": after})


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
        # Keep baseline can_separate strictly aligned with existing speech-2 golden behavior;
        # additive speech-3 and speech-recursive modes are exposed via extended_modes from capabilities.json.
        supports_k = k == 2 and bool(speech_sep.get("has_2spk_model"))
        ext_modes = capabilities.get_extended_speech_modes_for_k(k, is_pure_speech=True)
        reason = (
            None
            if supports_k
            else (
                f"Separation not available for this recording: only {k} source estimated."
                if k < 2
                else (
                    f"Baseline speech separation supports 2 speakers; additive mode available ({ext_modes[0]['label']})."
                    if ext_modes
                    else f"Separation not available for this recording: Separation is supported for up to {max_spk} speakers with the installed model."
                )
            )
        )
        return {
            "can_separate": supports_k,
            "trained": bool(speech_sep.get("available") or sep_trained),
            "speech_mode": True,
            "speech_separator": speech_sep,
            "max_separable": max_spk,
            "extended_modes": ext_modes,
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


def _can_see_ground_truth(user: dict | None) -> bool:
    """Server-side gate for reference ground truth: admin always; non-admin only when GROUND_TRUTH_PUBLIC=1."""
    if not user:
        return False
    return user.get("role") == "admin" or os.getenv("GROUND_TRUTH_PUBLIC", "0").strip() == "1"


def _rank_and_cap_pairs(
    named_sounds: list[dict], overlapping_sounds: list[dict], limit: int = 12
) -> tuple[list[dict], list[dict], int]:
    """Rank co-occurring pairs by overlap_duration * confidence and cap to top `limit`."""
    if not overlapping_sounds:
        return [], [], 0
    conf_map = {
        str(ns.get("name") or ns.get("friendly_name") or ns.get("class") or ""): float(
            ns.get("confidence", ns.get("probability", 0.5))
        )
        for ns in named_sounds
    }
    scored = []
    for o in overlapping_sounds:
        sounds = o.get("sounds") or []
        c1 = conf_map.get(str(sounds[0]), 0.5) if len(sounds) > 0 else 0.5
        c2 = conf_map.get(str(sounds[1]), 0.5) if len(sounds) > 1 else 0.5
        dur = float(o.get("total_overlap_seconds") or 0.0)
        pair_conf = round((c1 + c2) * 0.5, 4)
        score = round(dur * pair_conf, 4)
        scored.append((score, dur, {**o, "confidence": pair_conf, "score": score}))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    all_ranked = [item for _, _, item in scored]
    return all_ranked[:limit], all_ranked, len(all_ranked)


@app.get("/api/jobs/<job_id>/pairs")
@auth.login_required
def job_pairs_route(job_id: str):
    """Return full ranked co-occurring pairs for an authorized job on demand."""
    user = auth.get_current_user()
    if not auth.can_access_job(job_id, user):
        return jsonify({"error": "Job not found or expired."}), 404
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found or expired."}), 404
    all_pairs = job.get("all_overlapping_sounds") or []
    return jsonify({"pairs": all_pairs, "total": len(all_pairs)})


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
        gt = (
            data_mod.lookup_ground_truth(original_path, allow_in_eval=True)
            if _can_see_ground_truth(user)
            else None
        )
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

    named_sounds = sorted(
        tags_open.get("named_sounds", []),
        key=lambda x: -float(x.get("confidence", x.get("probability", 0.0))),
    )
    top_pairs, all_ranked_pairs, total_pairs = _rank_and_cap_pairs(
        named_sounds, tags_open.get("overlapping_sounds", []), limit=12
    )
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
            "all_overlapping_sounds": all_ranked_pairs,
            "speaker_clustering": speaker_clustering,
            "stems": None,
            "zip": None,
            "sep_status": "idle",
            "sep_result": None,
        }

    spread = int(np.ceil(result["total_std"] - 1e-9)) if result["total_std"] >= 0.5 else 0
    auth.update_user_job(
        job_id,
        "analyzed",
        {
            "rounded_count": k,
            "display": f"≈{k} (±{spread})",
            "confidence_percent": round(result["confidence_percent"], 1),
            "duration_seconds": round(duration, 2),
            "top_sounds": [s.get("name", "") for s in named_sounds[:4] if s.get("name")],
            "active_groups": list(tags_open.get("active_groups", []))[:5],
        },
    )
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
        "visible_default": 8,
        "overlapping_sounds": top_pairs,
        "overlapping_pairs_total": total_pairs,
        "speaker_clustering": speaker_clustering,
        "separation": sep_state,
        "tags": {
            "available": bool(tags_open.get("available", True)),
            "hierarchical": True,
            "open_vocabulary": True,
            "groups": tags_open.get("groups", []),
            "active_groups": tags_open.get("active_groups", []),
            "overlapping_sounds": top_pairs,
            "overlapping_pairs_total": total_pairs,
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


def _validate_separation_request(
    k: int, speech_mode: bool, req_mode: str, sep: dict, speech_sep_st: dict
) -> tuple[dict, int] | None:
    """Validate separation capability policy and return an error tuple if disallowed."""
    max_spk = int(speech_sep_st.get("max_supported_speakers") or 2)
    last_reliable_k, _, _ = get_dense_thresholds()
    if req_mode in ("speech-3", "speech-recursive"):
        allowed = {
            m["mode"] for m in capabilities.get_extended_speech_modes_for_k(k, is_pure_speech=True)
        }
        if req_mode not in allowed:
            return {
                "error": f"Separation mode '{req_mode}' is not enabled for K={k} under the capability policy."
            }, 409
        return None
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
    return None


def _write_stems_and_zip(
    job_id: str,
    job: dict,
    stems: list,
    residual_16k,
    queries: list,
    q_out: dict,
    sq: dict,
    sc_gate: dict,
    spk_ver: dict,
) -> tuple[list[str], list[dict], bytes]:
    """Export peak-normalised stem WAVs, residual, manifest.json, and compressed ZIP bundle."""
    rate, job_dir, count = job["input_rate"], JOB_DIR / job_id, job["count"]
    k = int(count["rounded_count"])
    all_waves = list(stems) + ([residual_16k] if residual_16k is not None else [])
    energies = np.array([float(np.sum(np.square(s))) for s in all_waves])
    shares = (energies / (float(energies.sum()) or 1.0)).tolist()

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
                fld: e[fld]
                for fld in (
                    "file",
                    "label",
                    "is_residual",
                    "energy_share",
                    "self_consistent",
                    "low_confidence",
                )
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
    return stem_files, stem_entries, buffer.getvalue()


def _execute_separation_job(
    job_id: str, job: dict, requested_mode: str | None = None
) -> tuple[dict, int]:
    """Run stem separation, evaluate 3 quality gates, and package downloadable WAV/ZIP artifacts."""
    k = int(job["count"]["rounded_count"])
    speech_mode = bool(job.get("speech_detected", False))
    sep, speech_sep_st = separation.separator_status(), separation.pretrained_speech_status()
    req_mode = (requested_mode or "").strip().lower()
    err_resp = _validate_separation_request(k, speech_mode, req_mode, sep, speech_sep_st)
    if err_resp:
        return err_resp

    timings: dict[str, float] = {}
    try:
        t0 = time.perf_counter()
        samples, _ = audio.decode_mono(job["original"], MAX_DURATION_SECONDS)
        full_16k = audio.resample(samples, job["input_rate"], SAMPLE_RATE).astype(np.float32)
        if req_mode == "speech-recursive":
            q_out = separation.separate_speech_recursive(full_16k, k=k, sample_rate=SAMPLE_RATE)
        else:
            q_out = separation.separate_by_queries(
                full_16k,
                k=k,
                named_sounds=job.get("named_sounds", []),
                speech_mode=(True if req_mode == "speech-3" else speech_mode),
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

    t1 = time.perf_counter()
    stem_files, stem_entries, zip_bytes = _write_stems_and_zip(
        job_id, job, stems, residual_16k, queries, q_out, sq, sc_gate, spk_ver
    )
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
                "zip": zip_bytes,
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
    if not auth.can_access_job(job_id, user):
        return jsonify({"error": "That analysis expired or does not exist."}), 404
    job = get_job(job_id)
    if job is None:
        return jsonify(
            {"error": "That analysis expired or does not exist. Analyze the file again."}
        ), 404
    req_mode = str(payload.get("mode", "")).strip() or None
    if payload.get("async"):
        with _jobs_lock:
            job["sep_status"] = "running"

        def _bg():
            res, code = _execute_separation_job(job_id, job, requested_mode=req_mode)
            if code != 200:
                with _jobs_lock:
                    job["sep_status"], job["sep_result"] = "failed", res

        _executor.submit(_bg)
        return jsonify(
            {"job_id": job_id, "status": "running", "poll_url": f"/api/separate/{job_id}"}
        ), 202
    res, code = _execute_separation_job(job_id, job, requested_mode=req_mode)
    return jsonify(res), code


@app.get("/api/separate/<job_id>")
@auth.login_required
def separate_status_route(job_id: str):
    user = auth.get_current_user()
    if not auth.can_access_job(job_id, user):
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
    if not auth.can_access_job(job_id, user):
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
    if not auth.can_access_job(job_id, user):
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


def _bootstrap_first_run() -> None:
    """Zero-config first-run setup: dev .env, folders, DB, and missing model/checkpoint download."""
    env_path = BASE_DIR / ".env"
    is_prod = (
        os.getenv("ECHOCOUNT_ENV", "").strip().lower() == "production"
        or os.getenv("FLASK_ENV", "").strip().lower() == "production"
    )
    if not env_path.is_file() and not is_prod:
        import secrets

        key = secrets.token_hex(32)
        env_path.write_text(
            f"# Auto-generated local development config (never commit .env)\n"
            f"SECRET_KEY={key}\n"
            f"# Optional: comma-separated emails that become admin on sign-up/restart\n"
            f"# ADMIN_EMAILS=you@gmail.com\n",
            encoding="utf-8",
        )
        os.environ.setdefault("SECRET_KEY", key)
        app.secret_key = key
        print("[EchoCount] Created local .env with a fresh random SECRET_KEY.")
        print(
            "[EchoCount] Nobody is admin by default. To promote an admin, set ADMIN_EMAILS=you@gmail.com in .env."
        )

    (BASE_DIR / "data").mkdir(parents=True, exist_ok=True)
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    auth.init_db()

    if os.getenv("ECHOCOUNT_SKIP_AUTO_DOWNLOAD", "0") != "1":
        try:
            from scripts.download_models import ensure_assets

            failures = ensure_assets()
            if failures:
                print(
                    "[EchoCount] Some assets could not be downloaded (offline or unavailable); "
                    "the app will still start with reduced functionality:"
                )
                for f in failures:
                    print(f"  - {f}")
        except Exception as exc:
            app.logger.warning("First-run asset check skipped: %s", exc)


if __name__ == "__main__":
    _bootstrap_first_run()
    app.run(debug=False, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")))
