"""Authentication, Email OTP, Google OIDC/Firebase, Single Locked Admin, and Account Lifecycle."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import shutil
import sqlite3
import subprocess
import time
from collections import deque
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from authlib.integrations.flask_client import OAuth
from flask import jsonify, redirect, render_template, request, url_for

import mail as mail_mod
from config import BASE_DIR, DATA_DIR

logger = logging.getLogger("echocount.auth")

DB_PATH = Path(os.getenv("ECHOCOUNT_AUTH_DB", str(DATA_DIR / "echocount.db")))
COMMON_PW_PATH = BASE_DIR / "common_passwords.txt"

SESSION_COOKIE = "echocount_session"
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(7 * 86400)))
ADMIN_IDLE_TTL_SECONDS = 30 * 60
ADMIN_ABS_TTL_SECONDS = 8 * 3600
OTP_TTL_SECONDS = 10 * 60
OTP_MAX_ATTEMPTS = 5
OTP_RESEND_COOLDOWN = 60
UNVERIFIED_PURGE_SECONDS = 24 * 3600
MIN_PASSWORD_LEN = 10
ADMIN_MIN_PASSWORD_LEN = 14
MAX_PASSWORD_LEN = 128


def admin_requires_otp() -> bool:
    """Return True only when ADMIN_REQUIRE_OTP=1 (default 0: password-only sign-in)."""
    return os.getenv("ADMIN_REQUIRE_OTP", "0").strip() == "1"


_EMAIL_RE = re.compile(r"^[a-z0-9._%+-]{1,64}@[a-z0-9.-]{1,253}\.[a-z]{2,24}$")
_ph = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=2, hash_len=32, salt_len=16)

_rate_buckets: dict[str, deque[float]] = {}
_fail_buckets: dict[str, deque[float]] = {}
_COMMON_PASSWORDS: set[str] | None = None
_oauth = OAuth()
_registered_providers: set[str] = set()


def _is_production() -> bool:
    return (
        os.getenv("ECHOCOUNT_ENV", "").strip().lower() == "production"
        or os.getenv("FLASK_ENV", "").strip().lower() == "production"
    )


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=15.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def _common_passwords() -> set[str]:
    global _COMMON_PASSWORDS
    if _COMMON_PASSWORDS is None:
        if COMMON_PW_PATH.is_file():
            _COMMON_PASSWORDS = {
                ln.strip().lower()
                for ln in COMMON_PW_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
                if ln.strip()
            }
        else:
            _COMMON_PASSWORDS = {"password123", "1234567890", "qwerty1234", "admin12345"}
    return _COMMON_PASSWORDS


def normalize_email(raw: str) -> str:
    return (raw or "").strip().lower()


def mask_email(email: str) -> str:
    return mail_mod.mask_email(email)


def hash_ip(ip: str) -> str:
    """Return a non-reversible salted short hash of an IP for logs (never log raw IPs)."""
    secret = get_or_create_secret_key().encode("utf-8")
    return hmac.new(secret, (ip or "local").encode("utf-8"), hashlib.sha256).hexdigest()[:12]


def is_https_base() -> bool:
    return os.getenv("BASE_URL", "").strip().lower().startswith("https://")


def get_or_create_secret_key(key_file: Path | None = None) -> str:
    """Return SECRET_KEY from env or persist a stable random key locally (dev only)."""
    env_key = os.getenv("SECRET_KEY", "").strip()
    if _is_production() and (not env_key or len(env_key) < 32 or "replace-with" in env_key.lower()):
        raise RuntimeError(
            "Production mode requires a strong SECRET_KEY (>=32 chars) in environment/.env."
        )
    if env_key and "replace-with" not in env_key.lower():
        return env_key
    target = key_file or (DB_PATH.parent / ".secret_key")
    if target.is_file():
        val = target.read_text(encoding="utf-8").strip()
        if len(val) >= 32:
            return val
    target.parent.mkdir(parents=True, exist_ok=True)
    val = secrets.token_hex(32)
    target.write_text(val, encoding="utf-8")
    return val


def get_configured_admin_email() -> str | None:
    """Resolve the single locked admin email from ADMIN_EMAILS (enforcing count == 1 in production)."""
    raw = os.getenv("ADMIN_EMAILS", "").strip()
    emails = [normalize_email(x) for x in raw.split(",") if x.strip()]
    if len(emails) > 1:
        raise RuntimeError("ADMIN_EMAILS must contain at most ONE administrator email address.")
    if emails:
        if not _EMAIL_RE.match(emails[0]):
            raise RuntimeError("ADMIN_EMAILS does not contain a valid email address.")
        return emails[0]
    if _is_production():
        raise RuntimeError(
            "Production mode requires ADMIN_EMAILS to contain exactly one admin email."
        )
    legacy = normalize_email(os.getenv("ECHOCOUNT_ADMIN_EMAIL", ""))
    if legacy and _EMAIL_RE.match(legacy):
        return legacy
    if DB_PATH.is_file():
        try:
            with _conn() as con:
                row = con.execute(
                    "SELECT email FROM users WHERE role = 'admin' ORDER BY id ASC LIMIT 1"
                ).fetchone()
                if row:
                    return normalize_email(row["email"])
        except sqlite3.Error:
            pass
    return None


def is_admin_email(raw_email: str) -> bool:
    cfg = get_configured_admin_email()
    return bool(cfg and normalize_email(raw_email) == cfg)


def enforce_single_admin() -> dict[str, Any]:
    """Ensure at most ONE admin account exists matching ADMIN_EMAILS; demote all others."""
    cfg_admin = get_configured_admin_email()
    demoted = 0
    with _conn() as con:
        admins = con.execute("SELECT id, email FROM users WHERE role = 'admin'").fetchall()
        for row in admins:
            em = normalize_email(row["email"])
            if cfg_admin is None or em != cfg_admin:
                con.execute("UPDATE users SET role = 'user' WHERE id = ?", (row["id"],))
                demoted += 1
        if cfg_admin:
            existing = con.execute("SELECT id FROM users WHERE email = ?", (cfg_admin,)).fetchone()
            if existing:
                con.execute(
                    "UPDATE users SET role = 'admin', email_verified = 1, is_active = 1, disabled_at = NULL WHERE id = ?",
                    (existing["id"],),
                )
        remaining = con.execute("SELECT id, email FROM users WHERE role = 'admin'").fetchall()
    logger.info(
        "Single-admin check complete: active_admins=%d demoted=%d configured=%s",
        len(remaining),
        demoted,
        mask_email(cfg_admin) if cfg_admin else "(none)",
    )
    return {
        "active_admins": len(remaining),
        "demoted": demoted,
        "configured_masked": mask_email(cfg_admin) if cfg_admin else None,
    }


def init_db() -> None:
    """Initialize SQLite auth/OTP/audit tables, run safe migrations, and enforce single admin."""
    with _conn() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT,
                email_verified INTEGER NOT NULL DEFAULT 0,
                role TEXT NOT NULL DEFAULT 'user',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL,
                last_login_at REAL,
                disabled_at REAL,
                reauth_at REAL
            );
            CREATE TABLE IF NOT EXISTS oauth_identities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                provider TEXT NOT NULL,
                provider_sub TEXT NOT NULL,
                email TEXT NOT NULL,
                email_verified INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                UNIQUE(provider, provider_sub)
            );
            CREATE TABLE IF NOT EXISTS auth_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                token_hash TEXT UNIQUE NOT NULL,
                purpose TEXT NOT NULL,
                expires_at REAL NOT NULL,
                used INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS email_otps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                purpose TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                expires_at REAL NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                used INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                csrf_token TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                last_seen_at REAL,
                revoked INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS user_jobs (
                job_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                filename TEXT NOT NULL,
                created_at REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'analyzed',
                summary_json TEXT
            );
            CREATE TABLE IF NOT EXISTS otp_daily_budget (
                day TEXT PRIMARY KEY,
                count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS admin_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                admin_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                target_id INTEGER NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name TEXT PRIMARY KEY,
                applied_at REAL NOT NULL
            );
            """
        )
        ucols = {r["name"] for r in con.execute("PRAGMA table_info(users)").fetchall()}
        for col, ddl in (
            ("role", "TEXT NOT NULL DEFAULT 'user'"),
            ("is_active", "INTEGER NOT NULL DEFAULT 1"),
            ("disabled_at", "REAL"),
            ("reauth_at", "REAL"),
        ):
            if col not in ucols:
                con.execute(f"ALTER TABLE users ADD COLUMN {col} {ddl}")
        scols = {r["name"] for r in con.execute("PRAGMA table_info(sessions)").fetchall()}
        if "last_seen_at" not in scols:
            con.execute("ALTER TABLE sessions ADD COLUMN last_seen_at REAL")
        jcols = {r["name"] for r in con.execute("PRAGMA table_info(user_jobs)").fetchall()}
        if "status" not in jcols:
            con.execute("ALTER TABLE user_jobs ADD COLUMN status TEXT NOT NULL DEFAULT 'analyzed'")
        if "summary_json" not in jcols:
            con.execute("ALTER TABLE user_jobs ADD COLUMN summary_json TEXT")
        # Phase 1 migration: remove legacy non-Google OAuth links while keeping user accounts intact
        con.execute("DELETE FROM oauth_identities WHERE provider != 'google'")
        # One-time migration: mark all pre-existing accounts as verified (OTP applies only to new sign-ups)
        mig_name = "mark_preexisting_users_verified_v1"
        if not con.execute(
            "SELECT 1 FROM schema_migrations WHERE name = ?", (mig_name,)
        ).fetchone():
            cur_v = con.execute("UPDATE users SET email_verified = 1 WHERE email_verified = 0")
            con.execute(
                "INSERT INTO schema_migrations (name, applied_at) VALUES (?, ?)",
                (mig_name, time.time()),
            )
            logger.info(
                "Migration %s marked %d existing account(s) as verified.",
                mig_name,
                int(cur_v.rowcount),
            )

    admin_email = os.getenv("ECHOCOUNT_ADMIN_EMAIL", "").strip()
    admin_pw = os.getenv("ECHOCOUNT_ADMIN_PASSWORD", "")
    if admin_email and admin_pw and is_admin_email(admin_email):
        ensure_admin_account(admin_email, admin_pw)
    enforce_single_admin()
    purge_expired_auth_data()


def ensure_admin_account(raw_email: str, password: str) -> int:
    """Idempotently ensure the configured single admin account exists with a password."""
    email = normalize_email(raw_email)
    if not email or not _EMAIL_RE.match(email):
        raise ValueError("Admin email must be a valid email address.")
    pw_err = validate_password(password, email, is_admin=True)
    if pw_err:
        raise ValueError(pw_err)
    pw_hash = hash_password(password)
    now = time.time()
    with _conn() as con:
        row = con.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if row:
            con.execute(
                "UPDATE users SET password_hash = ?, email_verified = 1, role = 'admin', is_active = 1, disabled_at = NULL WHERE id = ?",
                (pw_hash, row["id"]),
            )
            return int(row["id"])
        cur = con.execute(
            "INSERT INTO users (email, password_hash, email_verified, role, is_active, created_at) VALUES (?, ?, 1, 'admin', 1, ?)",
            (email, pw_hash, now),
        )
        return int(cur.lastrowid)


# ---------------------------------------------------------------------------
# Google-only OAuth (Authlib default + optional Firebase ID-token verification)
# ---------------------------------------------------------------------------


def google_auth_mode() -> str:
    mode = os.getenv("AUTH_GOOGLE_MODE", "authlib").strip().lower()
    return "firebase" if mode == "firebase" else "authlib"


def firebase_web_config() -> dict[str, str]:
    """Return public Firebase Web SDK config when AUTH_GOOGLE_MODE=firebase (non-secret)."""
    if google_auth_mode() != "firebase":
        return {}
    cfg = {
        "apiKey": os.getenv("FIREBASE_API_KEY", "").strip(),
        "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN", "").strip(),
        "projectId": os.getenv("FIREBASE_PROJECT_ID", "").strip(),
        "appId": os.getenv("FIREBASE_APP_ID", "").strip(),
    }
    return cfg if (cfg["apiKey"] and cfg["projectId"]) else {}


def init_oauth(app: Any) -> None:
    """Register the Google OIDC client when credentials are present."""
    _registered_providers.clear()
    _oauth.init_app(app)
    if google_auth_mode() == "firebase":
        if firebase_web_config():
            _registered_providers.add("google")
        return
    g_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    g_sec = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
    if g_id and g_sec:
        _oauth.register(
            name="google",
            client_id=g_id,
            client_secret=g_sec,
            server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email profile"},
        )
        _registered_providers.add("google")


def provider_availability() -> dict[str, bool]:
    return {"google": "google" in _registered_providers}


def get_oauth_client(provider: str) -> Any | None:
    if provider != "google" or provider not in _registered_providers:
        return None
    return _oauth.create_client("google")


def verify_firebase_google_token(
    id_token: str,
    *,
    project_id: str | None = None,
    verifier: Callable[[str], dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Verify a Firebase ID token (signature, issuer, audience, expiry, email_verified)."""
    pid = (project_id or os.getenv("FIREBASE_PROJECT_ID", "")).strip()
    if not id_token or not pid:
        return None, "Google sign-in could not be verified."
    try:
        if verifier is not None:
            claims = verifier(id_token)
        else:
            from google.auth.transport import requests as g_requests
            from google.oauth2 import id_token as g_id_token

            claims = g_id_token.verify_firebase_token(id_token, g_requests.Request(), audience=pid)
    except Exception:
        return None, "Google sign-in could not be verified."

    now = time.time()
    expected_iss = f"https://securetoken.google.com/{pid}"
    if not isinstance(claims, dict):
        return None, "Google sign-in could not be verified."
    if claims.get("iss") != expected_iss or claims.get("aud") != pid:
        return None, "Google sign-in could not be verified."
    if float(claims.get("exp") or 0) <= now:
        return None, "Google sign-in token expired. Please try again."
    if not claims.get("email_verified"):
        return None, "Your Google account email must be verified."
    sub = str(claims.get("sub") or claims.get("user_id") or "").strip()
    email = normalize_email(str(claims.get("email") or ""))
    if not sub or not email:
        return None, "Google sign-in could not be verified."
    return {"sub": sub, "email": email, "email_verified": True}, None


# ---------------------------------------------------------------------------
# Email Syntax + Optional MX Check & Hardened Password Policy
# ---------------------------------------------------------------------------


def _domain_has_mx(domain: str) -> bool:
    """Short-timeout MX check; if DNS lookup is unavailable, fall back to accepting valid syntax."""
    if os.getenv("ECHOCOUNT_SKIP_MX_CHECK", "0") == "1":
        return True
    try:
        import dns.resolver  # type: ignore

        ans = dns.resolver.resolve(domain, "MX", lifetime=1.5)
        return len(ans) > 0
    except ImportError:
        pass
    except Exception:
        return False
    # Stdlib fallback via nslookup (short timeout; fail open if DNS unreachable)
    nslookup = shutil.which("nslookup")
    if not nslookup:
        return True
    try:
        proc = subprocess.run(
            [nslookup, "-type=mx", domain],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
        out = (proc.stdout or "").lower()
        if "can't find" in out or "non-existent domain" in out or "nxdomain" in out:
            return False
        return True
    except Exception:
        return True


def validate_email_policy(
    raw_email: str, *, check_mx: bool = False
) -> tuple[str | None, str | None]:
    """Validate any syntactically valid email (including temporary-mail domains) + optional MX check."""
    email = normalize_email(raw_email)
    if not email or not _EMAIL_RE.match(email):
        return None, "Enter a valid email address."
    if check_mx:
        domain = email.rsplit("@", 1)[-1]
        if not _domain_has_mx(domain):
            return None, "That email domain cannot receive mail. Please check the address."
    return email, None


def validate_password(password: str, email: str = "", *, is_admin: bool = False) -> str | None:
    min_len = ADMIN_MIN_PASSWORD_LEN if is_admin else MIN_PASSWORD_LEN
    if not password or len(password) < min_len:
        label = "Admin password" if is_admin else "Password"
        return f"{label} must be at least {min_len} characters."
    if len(password) > MAX_PASSWORD_LEN:
        return f"Password must be {MAX_PASSWORD_LEN} characters or fewer."
    if email and password.strip().lower() == normalize_email(email):
        return "Password cannot be the same as your email address."
    if password.strip().lower() in _common_passwords():
        return "That password is too common. Choose a stronger passphrase."
    return None


def hash_password(password: str) -> str:
    return _ph.hash(password)


def verify_password(password: str, stored_hash: str) -> bool:
    if not stored_hash:
        return False
    try:
        return _ph.verify(stored_hash, password)
    except (VerifyMismatchError, Exception):
        return False


def verify_captcha_hook(_req: Any = None) -> bool:
    return True


def check_auth_rate_limit(key: str, limit: int = 8, window_s: float = 300.0) -> bool:
    now = time.time()
    dq = _rate_buckets.setdefault(key, deque())
    while dq and now - dq[0] > window_s:
        dq.popleft()
    if len(dq) >= limit:
        return False
    dq.append(now)
    return True


def record_login_failure(key: str, window_s: float = 900.0) -> int:
    """Record a failed auth attempt and apply a brief progressive delay (outside unit tests)."""
    now = time.time()
    dq = _fail_buckets.setdefault(key, deque())
    while dq and now - dq[0] > window_s:
        dq.popleft()
    dq.append(now)
    fails = len(dq)
    if os.getenv("ECHOCOUNT_NO_AUTH_DELAY", "0") != "1" and fails >= 2:
        time.sleep(min(0.4 * (fails - 1), 2.0))
    return fails


def is_locked_out(key: str, max_failures: int = 6, window_s: float = 900.0) -> bool:
    now = time.time()
    dq = _fail_buckets.get(key)
    if not dq:
        return False
    while dq and now - dq[0] > window_s:
        dq.popleft()
    return len(dq) >= max_failures


def clear_login_failures(key: str) -> None:
    _fail_buckets.pop(key, None)


# ---------------------------------------------------------------------------
# Email OTP Generation, HMAC-SHA256 Storage, Rate Limits & Purge
# ---------------------------------------------------------------------------


def _hash_otp(code: str, salt: str) -> str:
    key = get_or_create_secret_key().encode("utf-8")
    return hmac.new(key, f"{salt}:{code}".encode("utf-8"), hashlib.sha256).hexdigest()


def issue_email_otp(
    raw_email: str,
    purpose: str,
    *,
    ip: str = "local",
    send_email: bool = True,
) -> tuple[bool, str | None, int]:
    """Generate a 6-digit OTP, store HMAC-SHA256+salt, invalidate prior codes, and send email.

    Returns (ok, error_message, retry_after_seconds).
    """
    email = normalize_email(raw_email)
    now = time.time()
    if not check_auth_rate_limit(f"otp:ip:{hash_ip(ip)}", limit=20, window_s=3600.0):
        return False, "Too many verification code requests. Please wait and try again.", 60
    if not check_auth_rate_limit(f"otp:email:{email}", limit=5, window_s=3600.0):
        return (
            False,
            "Too many verification codes requested for this email. Please wait an hour.",
            60,
        )

    with _conn() as con:
        latest = con.execute(
            "SELECT created_at FROM email_otps WHERE email = ? AND purpose = ? ORDER BY id DESC LIMIT 1",
            (email, purpose),
        ).fetchone()
        if latest and (now - float(latest["created_at"])) < OTP_RESEND_COOLDOWN:
            wait_s = max(1, int(OTP_RESEND_COOLDOWN - (now - float(latest["created_at"]))))
            return False, f"Please wait {wait_s} seconds before requesting a new code.", wait_s

        code = f"{secrets.randbelow(1_000_000):06d}"
        salt = secrets.token_hex(16)
        code_hash = _hash_otp(code, salt)
        # Invalidate any prior unused codes for this (email, purpose)
        con.execute(
            "UPDATE email_otps SET used = 1 WHERE email = ? AND purpose = ? AND used = 0",
            (email, purpose),
        )
        con.execute(
            "INSERT INTO email_otps (email, purpose, code_hash, salt, expires_at, attempts, used, created_at) "
            "VALUES (?, ?, ?, ?, ?, 0, 0, ?)",
            (email, purpose, code_hash, salt, now + OTP_TTL_SECONDS, now),
        )

    if send_email:
        ok, err = mail_mod.send_otp_email(email, code, purpose, db_path=DB_PATH)
        if not ok:
            # Mark unused code invalidated so failed delivery doesn't leave dangling active code
            with _conn() as con:
                con.execute(
                    "UPDATE email_otps SET used = 1 WHERE email = ? AND purpose = ? AND used = 0",
                    (email, purpose),
                )
            return False, err or "Email verification is not available right now.", 0
    return True, None, OTP_RESEND_COOLDOWN


def verify_email_otp(raw_email: str, purpose: str, raw_code: str) -> tuple[bool, str | None]:
    """Constant-time verify a 6-digit OTP; enforces expiry, single use, and max 5 attempts."""
    email = normalize_email(raw_email)
    code = re.sub(r"\s+", "", str(raw_code or ""))
    if not re.fullmatch(r"\d{6}", code):
        return False, "Invalid or expired verification code."
    now = time.time()
    with _conn() as con:
        row = con.execute(
            "SELECT id, code_hash, salt, expires_at, attempts, used FROM email_otps "
            "WHERE email = ? AND purpose = ? AND used = 0 ORDER BY id DESC LIMIT 1",
            (email, purpose),
        ).fetchone()
        if not row or row["used"] or float(row["expires_at"]) < now:
            return False, "Invalid or expired verification code."
        if int(row["attempts"]) >= OTP_MAX_ATTEMPTS:
            con.execute("UPDATE email_otps SET used = 1 WHERE id = ?", (row["id"],))
            return (
                False,
                "Verification code invalidated after too many attempts. Request a new code.",
            )

        expected = _hash_otp(code, row["salt"])
        if not hmac.compare_digest(expected, row["code_hash"]):
            new_attempts = int(row["attempts"]) + 1
            invalidate = 1 if new_attempts >= OTP_MAX_ATTEMPTS else 0
            con.execute(
                "UPDATE email_otps SET attempts = ?, used = ? WHERE id = ?",
                (new_attempts, invalidate, row["id"]),
            )
            if invalidate:
                return (
                    False,
                    "Verification code invalidated after too many attempts. Request a new code.",
                )
            return False, "Invalid or expired verification code."

        con.execute("UPDATE email_otps SET used = 1 WHERE id = ?", (row["id"],))
    return True, None


def purge_expired_auth_data(job_dir: Path | None = None) -> dict[str, int]:
    """Delete unverified accounts (>24h), expired/used OTPs (>24h), and optionally old disabled accounts."""
    now = time.time()
    cutoff = now - UNVERIFIED_PURGE_SECONDS
    deleted_unverified = 0
    deleted_otps = 0
    deleted_disabled = 0
    with _conn() as con:
        cur_u = con.execute(
            "DELETE FROM users WHERE email_verified = 0 AND role != 'admin' AND created_at < ?",
            (cutoff,),
        )
        deleted_unverified = int(cur_u.rowcount)
        cur_o = con.execute(
            "DELETE FROM email_otps WHERE expires_at < ? OR (used = 1 AND created_at < ?)",
            (cutoff, cutoff),
        )
        deleted_otps = int(cur_o.rowcount)
        days = int(os.getenv("DISABLED_DELETE_DAYS", "0") or "0")
        if days > 0:
            d_cut = now - days * 86400
            rows = con.execute(
                "SELECT id FROM users WHERE is_active = 0 AND role != 'admin' AND disabled_at IS NOT NULL AND disabled_at < ?",
                (d_cut,),
            ).fetchall()
            for r in rows:
                jids = [
                    j["job_id"]
                    for j in con.execute(
                        "SELECT job_id FROM user_jobs WHERE user_id = ?", (r["id"],)
                    ).fetchall()
                ]
                con.execute("DELETE FROM users WHERE id = ?", (r["id"],))
                deleted_disabled += 1
                if job_dir:
                    for jid in jids:
                        shutil.rmtree(job_dir / jid, ignore_errors=True)
    return {
        "unverified_deleted": deleted_unverified,
        "otps_deleted": deleted_otps,
        "disabled_deleted": deleted_disabled,
    }


# ---------------------------------------------------------------------------
# Sessions (Rotation on Login + Admin 30m Idle / 8h Absolute) & CSRF
# ---------------------------------------------------------------------------


def create_session(user_id: int, *, rotate_from_sid: str | None = None) -> tuple[str, str]:
    """Create a fresh rotated session ID (revoking any previous session cookie to prevent fixation)."""
    sid = secrets.token_urlsafe(32)
    csrf = secrets.token_urlsafe(24)
    now = time.time()
    with _conn() as con:
        if rotate_from_sid:
            con.execute("UPDATE sessions SET revoked = 1 WHERE id = ?", (rotate_from_sid,))
        row = con.execute("SELECT role FROM users WHERE id = ?", (int(user_id),)).fetchone()
        role = (row["role"] if row else "user") or "user"
        ttl = ADMIN_ABS_TTL_SECONDS if role == "admin" else SESSION_TTL_SECONDS
        con.execute(
            "INSERT INTO sessions (id, user_id, csrf_token, created_at, expires_at, last_seen_at, revoked) "
            "VALUES (?, ?, ?, ?, ?, ?, 0)",
            (sid, int(user_id), csrf, now, now + ttl, now),
        )
        con.execute(
            "UPDATE users SET last_login_at = ?, reauth_at = ? WHERE id = ?",
            (now, now, int(user_id)),
        )
    return sid, csrf


def mark_reauthenticated(user_id: int) -> None:
    with _conn() as con:
        con.execute("UPDATE users SET reauth_at = ? WHERE id = ?", (time.time(), int(user_id)))


def is_recently_reauthenticated(user_id: int, window_s: float = 300.0) -> bool:
    with _conn() as con:
        row = con.execute("SELECT reauth_at FROM users WHERE id = ?", (int(user_id),)).fetchone()
    return bool(row and row["reauth_at"] and (time.time() - float(row["reauth_at"])) <= window_s)


def revoke_session(sid: str | None) -> None:
    if not sid:
        return
    with _conn() as con:
        con.execute("UPDATE sessions SET revoked = 1 WHERE id = ?", (sid,))


def get_current_user() -> dict[str, Any] | None:
    """Resolve authenticated active user from HttpOnly session cookie (enforces admin 30m idle)."""
    sid = request.cookies.get(SESSION_COOKIE)
    if not sid:
        return None
    now = time.time()
    with _conn() as con:
        row = con.execute(
            """
            SELECT u.id, u.email, u.password_hash, u.email_verified, u.role,
                   COALESCE(u.is_active, 1) AS is_active, u.created_at, u.reauth_at,
                   s.csrf_token, s.expires_at, s.last_seen_at, s.revoked
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.id = ?
            """,
            (sid,),
        ).fetchone()
        if not row or row["revoked"] or float(row["expires_at"]) < now:
            return None
        if not row["is_active"] or not row["email_verified"]:
            return None
        role = row["role"] or "user"
        if role == "admin":
            last_seen = float(row["last_seen_at"] or 0)
            if last_seen and (now - last_seen) > ADMIN_IDLE_TTL_SECONDS:
                con.execute("UPDATE sessions SET revoked = 1 WHERE id = ?", (sid,))
                return None
        con.execute("UPDATE sessions SET last_seen_at = ? WHERE id = ?", (now, sid))
    email = row["email"]
    masked = mask_email(email)
    return {
        "id": int(row["id"]),
        "email": email,
        "masked_email": masked,
        "display_label": "Admin" if role == "admin" else email,
        "email_verified": bool(row["email_verified"]),
        "has_password": bool(row["password_hash"]),
        "role": role,
        "is_active": True,
        "created_at": row["created_at"],
        "csrf_token": row["csrf_token"],
        "session_id": sid,
    }


def get_or_create_form_csrf() -> str:
    user = get_current_user()
    if user:
        return user["csrf_token"]
    from flask import session as flask_session

    tok = flask_session.get("_anon_csrf")
    if not tok:
        tok = secrets.token_urlsafe(24)
        flask_session["_anon_csrf"] = tok
    return tok


def verify_csrf_request() -> bool:
    expected = get_or_create_form_csrf()
    supplied = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token") or ""
    if not supplied and request.is_json:
        supplied = str((request.get_json(silent=True) or {}).get("csrf_token", ""))
    return bool(expected and supplied and hmac.compare_digest(expected, supplied))


def csrf_required(fn: Any) -> Any:
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not verify_csrf_request():
            return jsonify(
                {"error": "Invalid or expired security token. Refresh the page and try again."}
            ), 403
        return fn(*args, **kwargs)

    return wrapper


def set_session_cookie(resp: Any, sid: str, *, is_admin: bool = False) -> Any:
    secure = (
        os.getenv("ECHOCOUNT_COOKIE_SECURE", "0") == "1" or is_https_base() or request.is_secure
    )
    max_age = ADMIN_ABS_TTL_SECONDS if is_admin else SESSION_TTL_SECONDS
    resp.set_cookie(
        SESSION_COOKIE,
        sid,
        max_age=max_age,
        httponly=True,
        samesite="Lax",
        secure=secure,
        path="/",
    )
    return resp


def clear_session_cookie(resp: Any) -> Any:
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp


def default_landing_for_role(role: str | None) -> str:
    r = (role or "user").strip().lower()
    return "/admin" if r == "admin" else "/"


def _unauth_response() -> Any:
    if request.path.startswith("/api/"):
        return jsonify(
            {"error": "Authentication required. Please sign in.", "auth_required": True}
        ), 401
    return redirect(url_for("login_page", next=request.path))


def login_required(fn: Any) -> Any:
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs) if get_current_user() else _unauth_response()

    return wrapper


def role_required(*allowed_roles: str) -> Any:
    def decorator(fn: Any) -> Any:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            user = get_current_user()
            if not user:
                return _unauth_response()
            if user.get("role") not in allowed_roles:
                if request.path.startswith("/api/"):
                    return jsonify(
                        {"error": "You do not have permission to access this resource."}
                    ), 403
                return (
                    render_template(
                        "403.html",
                        required_roles=allowed_roles,
                        status={
                            "trained": True,
                            "sample_rate": 16000,
                            "clip_seconds": 8,
                            "max_sources": 31,
                        },
                    ),
                    403,
                )
            return fn(*args, **kwargs)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Per-User History Jobs, Account Lifecycle & Admin Audit Log
# ---------------------------------------------------------------------------


def register_user_job(
    job_id: str, user_id: int, filename: str, summary: dict | None = None
) -> None:
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO user_jobs (job_id, user_id, filename, created_at, status, summary_json) VALUES (?, ?, ?, ?, 'analyzed', ?)",
            (job_id, user_id, filename, time.time(), json.dumps(summary) if summary else None),
        )


def update_user_job(job_id: str, status: str, summary: dict | None = None) -> None:
    with _conn() as con:
        if summary is not None:
            con.execute(
                "UPDATE user_jobs SET status = ?, summary_json = ? WHERE job_id = ?",
                (status, json.dumps(summary), job_id),
            )
        else:
            con.execute("UPDATE user_jobs SET status = ? WHERE job_id = ?", (status, job_id))


def _format_job_row(r: sqlite3.Row) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if r["summary_json"]:
        try:
            parsed = json.loads(r["summary_json"])
            if isinstance(parsed, dict):
                summary = parsed
        except ValueError:
            summary = {}
    item: dict[str, Any] = {
        "id": r["job_id"],
        "job_id": r["job_id"],
        "user_id": int(r["user_id"]),
        "filename": r["filename"],
        "created_at": r["created_at"],
        "status": r["status"] or "analyzed",
        "summary": summary,
    }
    if "user_email" in r.keys():
        role = r["user_role"] if "user_role" in r.keys() else "user"
        item["user_email"] = mask_email(r["user_email"]) if role == "admin" else r["user_email"]
    return item


def list_user_jobs(user_id: int, limit: int = 100) -> list[dict[str, Any]]:
    """Return recent analysis jobs belonging solely to an active user_id."""
    with _conn() as con:
        rows = con.execute(
            """
            SELECT j.job_id, j.user_id, j.filename, j.created_at, j.status, j.summary_json
            FROM user_jobs j
            JOIN users u ON u.id = j.user_id
            WHERE j.user_id = ? AND COALESCE(u.is_active, 1) = 1
            ORDER BY j.created_at DESC LIMIT ?
            """,
            (int(user_id), int(limit)),
        ).fetchall()
    return [_format_job_row(r) for r in rows]


def list_all_jobs(user_id: int | None = None, limit: int = 200) -> list[dict[str, Any]]:
    """Return jobs across active users for admin-only views (hides disabled users; masks admin email)."""
    where = "WHERE COALESCE(u.is_active, 1) = 1" + (
        " AND j.user_id = ?" if user_id is not None else ""
    )
    params = (int(user_id), int(limit)) if user_id is not None else (int(limit),)
    with _conn() as con:
        rows = con.execute(
            f"""
            SELECT j.job_id, j.user_id, u.email AS user_email, u.role AS user_role,
                   j.filename, j.created_at, j.status, j.summary_json
            FROM user_jobs j
            JOIN users u ON u.id = j.user_id
            {where}
            ORDER BY j.created_at DESC LIMIT ?
            """,
            params,
        ).fetchall()
    return [_format_job_row(r) for r in rows]


def delete_user_job(job_id: str, user: dict[str, Any]) -> bool:
    with _conn() as con:
        if user.get("role") == "admin":
            cur = con.execute("DELETE FROM user_jobs WHERE job_id = ?", (job_id,))
        else:
            cur = con.execute(
                "DELETE FROM user_jobs WHERE job_id = ? AND user_id = ?", (job_id, int(user["id"]))
            )
    return cur.rowcount > 0


def clear_user_jobs(user_id: int) -> list[str]:
    with _conn() as con:
        ids = [
            r["job_id"]
            for r in con.execute(
                "SELECT job_id FROM user_jobs WHERE user_id = ?", (int(user_id),)
            ).fetchall()
        ]
        con.execute("DELETE FROM user_jobs WHERE user_id = ?", (int(user_id),))
    return ids


def log_admin_audit(admin_id: int, action: str, target_id: int) -> None:
    """Record admin moderation action with IDs only (never stores emails)."""
    with _conn() as con:
        con.execute(
            "INSERT INTO admin_audit_log (admin_id, action, target_id, created_at) VALUES (?, ?, ?, ?)",
            (int(admin_id), str(action), int(target_id), time.time()),
        )


def list_admin_audit(limit: int = 50) -> list[dict[str, Any]]:
    with _conn() as con:
        rows = con.execute(
            "SELECT id, admin_id, action, target_id, created_at FROM admin_audit_log ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [dict(r) for r in rows]


def deactivate_user_account(user_id: int) -> tuple[bool, str | None, str | None]:
    """Temporarily disable a non-admin account and immediately revoke all its sessions."""
    with _conn() as con:
        row = con.execute(
            "SELECT id, email, role FROM users WHERE id = ?", (int(user_id),)
        ).fetchone()
        if not row:
            return False, None, "Account not found."
        if row["role"] == "admin" or is_admin_email(row["email"]):
            return False, None, "Administrator accounts cannot be deactivated from the website."
        con.execute(
            "UPDATE users SET is_active = 0, disabled_at = ? WHERE id = ?",
            (time.time(), int(user_id)),
        )
        con.execute("UPDATE sessions SET revoked = 1 WHERE user_id = ?", (int(user_id),))
        email = row["email"]
    return True, email, None


def reactivate_user_account(user_id: int) -> bool:
    with _conn() as con:
        cur = con.execute(
            "UPDATE users SET is_active = 1, disabled_at = NULL, email_verified = 1 WHERE id = ?",
            (int(user_id),),
        )
    return cur.rowcount > 0


def delete_user_account(
    target_uid: int, actor_uid: int | None = None, *, self_delete: bool = False
) -> tuple[bool, list[str], str | None, str | None]:
    """Atomically delete a non-admin user, password hash, OAuth links, OTPs, sessions, and jobs.

    Returns (ok, deleted_job_ids, deleted_email, error).
    """
    if not self_delete and actor_uid is not None and int(target_uid) == int(actor_uid):
        return False, [], None, "You cannot delete your own administrator account."
    with _conn() as con:
        row = con.execute(
            "SELECT id, email, role FROM users WHERE id = ?", (int(target_uid),)
        ).fetchone()
        if not row:
            return False, [], None, "User account not found."
        if row["role"] == "admin" or is_admin_email(row["email"]):
            return False, [], None, "Administrator accounts cannot be deleted from the website."
        email = row["email"]
        ids = [
            r["job_id"]
            for r in con.execute(
                "SELECT job_id FROM user_jobs WHERE user_id = ?", (int(target_uid),)
            ).fetchall()
        ]
        con.execute("DELETE FROM email_otps WHERE email = ?", (email,))
        con.execute("DELETE FROM auth_tokens WHERE user_id = ?", (int(target_uid),))
        con.execute("DELETE FROM oauth_identities WHERE user_id = ?", (int(target_uid),))
        con.execute("DELETE FROM sessions WHERE user_id = ?", (int(target_uid),))
        con.execute("DELETE FROM user_jobs WHERE user_id = ?", (int(target_uid),))
        con.execute("DELETE FROM users WHERE id = ?", (int(target_uid),))
    if not self_delete and actor_uid is not None:
        log_admin_audit(actor_uid, "delete_user", target_uid)
    return True, ids, email, None


def list_users() -> list[dict[str, Any]]:
    """Return user accounts for Admin Dashboard; masks admin email, marks disabled accounts."""
    now = time.time()
    with _conn() as con:
        rows = con.execute(
            """
            SELECT u.id, u.email, u.role, u.email_verified, COALESCE(u.is_active, 1) AS is_active,
                   u.created_at, u.last_login_at, u.disabled_at,
                   (SELECT COUNT(*) FROM sessions s WHERE s.user_id = u.id AND s.revoked = 0 AND s.expires_at > ?) AS active_sessions,
                   (SELECT COUNT(*) FROM user_jobs j WHERE j.user_id = u.id) AS job_count
            FROM users u
            ORDER BY CASE u.role WHEN 'admin' THEN 0 ELSE 1 END, u.created_at DESC
            """,
            (now,),
        ).fetchall()
    out = []
    for r in rows:
        role = r["role"] or "user"
        out.append(
            {
                "id": int(r["id"]),
                "email": mask_email(r["email"]) if role == "admin" else r["email"],
                "role": role,
                "email_verified": bool(r["email_verified"]),
                "is_active": bool(r["is_active"]),
                "status": "active" if r["is_active"] else "disabled",
                "created_at": r["created_at"],
                "last_login_at": r["last_login_at"],
                "active_sessions": int(r["active_sessions"]),
                "job_count": int(r["job_count"]),
            }
        )
    return out


def set_user_active(target_uid: int, active: bool, actor_uid: int) -> tuple[bool, str | None]:
    if int(target_uid) == int(actor_uid):
        return False, "Administrator accounts cannot be disabled."
    with _conn() as con:
        row = con.execute(
            "SELECT id, email, role FROM users WHERE id = ?", (int(target_uid),)
        ).fetchone()
        if not row:
            return False, "User account not found."
        if row["role"] == "admin" or is_admin_email(row["email"]):
            return False, "Administrator accounts cannot be disabled."
        con.execute(
            "UPDATE users SET is_active = ?, disabled_at = ? WHERE id = ?",
            (1 if active else 0, None if active else time.time(), int(target_uid)),
        )
        if not active:
            con.execute("UPDATE sessions SET revoked = 1 WHERE user_id = ?", (int(target_uid),))
    log_admin_audit(actor_uid, "enable_user" if active else "disable_user", target_uid)
    return True, None


def revoke_user_sessions(target_uid: int) -> int:
    with _conn() as con:
        cur = con.execute(
            "UPDATE sessions SET revoked = 1 WHERE user_id = ? AND revoked = 0", (int(target_uid),)
        )
        return int(cur.rowcount)


def user_owns_job(job_id: str, user_id: int) -> bool:
    with _conn() as con:
        row = con.execute("SELECT user_id FROM user_jobs WHERE job_id = ?", (job_id,)).fetchone()
    return bool(row and int(row["user_id"]) == int(user_id))


def can_access_job(job_id: str, user: dict[str, Any] | None) -> bool:
    if not user or not job_id:
        return False
    if user.get("role") == "admin":
        with _conn() as con:
            row = con.execute("SELECT 1 FROM user_jobs WHERE job_id = ?", (job_id,)).fetchone()
        return row is not None
    return user_owns_job(job_id, int(user["id"]))


# ---------------------------------------------------------------------------
# Registration, Password Login, Google OAuth Login & Password Reset
# ---------------------------------------------------------------------------

_USER_SELECT = "SELECT id, email, password_hash, email_verified, role, COALESCE(is_active, 1) AS is_active FROM users"


def register_oauth_user(
    provider: str, provider_sub: str, raw_email: str, email_verified: bool
) -> tuple[dict[str, Any] | None, str | None]:
    """Authenticate or link a verified Google identity (never allows admin or unverified merge)."""
    if provider != "google":
        return None, "Unsupported authentication provider."
    if not email_verified:
        return None, "Your Google account email must be verified before signing in."
    email, err = validate_email_policy(raw_email, check_mx=False)
    if err or not email:
        return None, err or "Invalid email address."
    # Single locked admin: NEVER allow Google sign-in to create, merge, or sign into admin!
    if is_admin_email(email):
        return None, "Invalid email or password."

    now = time.time()
    with _conn() as con:
        existing_oauth = con.execute(
            "SELECT user_id FROM oauth_identities WHERE provider = 'google' AND provider_sub = ?",
            (provider_sub,),
        ).fetchone()
        if existing_oauth:
            u = con.execute(f"{_USER_SELECT} WHERE id = ?", (existing_oauth["user_id"],)).fetchone()
            if not u or u["role"] == "admin":
                return None, "Invalid email or password."
            if not u["is_active"]:
                return (
                    None,
                    "This account is deactivated. Sign in with email and password to reactivate.",
                )
            return dict(u), None

        existing_user = con.execute(f"{_USER_SELECT} WHERE email = ?", (email,)).fetchone()
        if existing_user:
            if existing_user["role"] == "admin":
                return None, "Invalid email or password."
            if not existing_user["is_active"]:
                return (
                    None,
                    "This account is deactivated. Sign in with email and password to reactivate.",
                )
            # Never auto-merge with an unverified password account
            if not existing_user["email_verified"]:
                return None, "Verify your existing password account by email before linking Google."
            uid = int(existing_user["id"])
        else:
            cur = con.execute(
                "INSERT INTO users (email, password_hash, email_verified, role, is_active, created_at) VALUES (?, NULL, 1, 'user', 1, ?)",
                (email, now),
            )
            uid = int(cur.lastrowid)

        con.execute(
            "INSERT INTO oauth_identities (user_id, provider, provider_sub, email, email_verified, created_at) "
            "VALUES (?, 'google', ?, ?, 1, ?)",
            (uid, provider_sub, email, now),
        )
        return dict(con.execute(f"{_USER_SELECT} WHERE id = ?", (uid,)).fetchone()), None


def register_password_user(
    raw_email: str, password: str, *, check_mx: bool = False
) -> tuple[int | None, str | None, str | None]:
    """Create an UNVERIFIED password account (never exposes admin or existing accounts)."""
    email, err = validate_email_policy(raw_email, check_mx=check_mx)
    if err or not email:
        return None, None, err
    pw_err = validate_password(password, email)
    if pw_err:
        return None, None, pw_err
    # Always hash before checking admin/existing to keep signup timing uniform
    pw_hash = hash_password(password)
    if is_admin_email(email):
        return None, None, None

    now = time.time()
    with _conn() as con:
        existing = con.execute(
            "SELECT id, email_verified, role FROM users WHERE email = ?", (email,)
        ).fetchone()
        if existing:
            if existing["role"] == "admin" or existing["email_verified"]:
                return None, None, None
            con.execute(
                "UPDATE users SET password_hash = ?, created_at = ? WHERE id = ?",
                (pw_hash, now, existing["id"]),
            )
            uid = int(existing["id"])
        else:
            cur = con.execute(
                "INSERT INTO users (email, password_hash, email_verified, role, is_active, created_at) VALUES (?, ?, 0, 'user', 1, ?)",
                (email, pw_hash, now),
            )
            uid = int(cur.lastrowid)
    return uid, None, None


def authenticate_password_user(
    raw_email: str, password: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Verify email + password (returns user dict with status flags; generic error on mismatch)."""
    email = normalize_email(raw_email)
    if not email or not password:
        return None, "Invalid email or password."
    with _conn() as con:
        row = con.execute(f"{_USER_SELECT} WHERE email = ?", (email,)).fetchone()
    if not row or not row["password_hash"] or not verify_password(password, row["password_hash"]):
        return None, "Invalid email or password."
    return {
        "id": int(row["id"]),
        "email": row["email"],
        "email_verified": bool(row["email_verified"]),
        "is_active": bool(row["is_active"]),
        "role": row["role"] or "user",
    }, None


def get_user_by_email(raw_email: str) -> dict[str, Any] | None:
    email = normalize_email(raw_email)
    if not email:
        return None
    with _conn() as con:
        row = con.execute(f"{_USER_SELECT} WHERE email = ?", (email,)).fetchone()
    return dict(row) if row else None


def delete_unverified_signup(user_id: int, raw_email: str) -> bool:
    """Delete an unverified non-admin user and its pending OTPs when user clicks 'Change email'."""
    email = normalize_email(raw_email)
    with _conn() as con:
        row = con.execute(
            "SELECT id, email, role, email_verified FROM users WHERE id = ? AND email = ?",
            (int(user_id), email),
        ).fetchone()
        if not row or row["role"] == "admin" or row["email_verified"]:
            return False
        con.execute("DELETE FROM email_otps WHERE email = ?", (email,))
        con.execute("DELETE FROM auth_tokens WHERE user_id = ?", (int(user_id),))
        con.execute(
            "DELETE FROM users WHERE id = ? AND email_verified = 0 AND role != 'admin'",
            (int(user_id),),
        )
    return True


def mark_email_verified(raw_email: str) -> dict[str, Any] | None:
    email = normalize_email(raw_email)
    with _conn() as con:
        con.execute("UPDATE users SET email_verified = 1 WHERE email = ?", (email,))
        row = con.execute(f"{_USER_SELECT} WHERE email = ?", (email,)).fetchone()
    return dict(row) if row else None


def update_password_by_email(raw_email: str, new_password: str) -> tuple[bool, str | None]:
    email = normalize_email(raw_email)
    if is_admin_email(email):
        return False, "Unable to reset password for this account through this form."
    pw_err = validate_password(new_password, email)
    if pw_err:
        return False, pw_err
    pw_hash = hash_password(new_password)
    with _conn() as con:
        row = con.execute("SELECT id, role FROM users WHERE email = ?", (email,)).fetchone()
        if not row or row["role"] == "admin":
            return False, "Unable to reset password."
        con.execute(
            "UPDATE users SET password_hash = ?, email_verified = 1 WHERE id = ?",
            (pw_hash, row["id"]),
        )
        con.execute("UPDATE sessions SET revoked = 1 WHERE user_id = ?", (row["id"],))
    return True, None
