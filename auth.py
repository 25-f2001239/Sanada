"""Authentication, OAuth 2.0/OIDC (Google, Microsoft, GitHub, Yahoo), RBAC, Argon2id, and sessions."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import smtplib
import sqlite3
import time
from collections import deque
from email.message import EmailMessage
from functools import wraps
from pathlib import Path
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from authlib.integrations.flask_client import OAuth
from flask import g, jsonify, redirect, render_template, request, session, url_for

from config import DATA_DIR

DB_PATH = Path(os.getenv("ECHOCOUNT_DB_PATH", str(DATA_DIR / "echocount.db")))
SESSION_COOKIE = "echocount_sid"
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(7 * 24 * 3600)))
TOKEN_TTL_SECONDS = int(os.getenv("AUTH_TOKEN_TTL_SECONDS", "1800"))
MIN_PASSWORD_LEN = 10
MAX_PASSWORD_LEN = 128

SUPPORTED_PROVIDERS = ("google", "microsoft", "github", "yahoo")
VALID_ROLES = ("user", "developer", "admin")

GOOGLE_DOMAINS = {"gmail.com", "googlemail.com"}
MICROSOFT_DOMAINS = {
    "outlook.com",
    "hotmail.com",
    "live.com",
    "msn.com",
    "passport.com",
    "outlook.co.uk",
    "hotmail.co.uk",
}
YAHOO_DOMAINS = {
    "yahoo.com",
    "ymail.com",
    "rocketmail.com",
    "yahoo.co.uk",
    "yahoo.co.in",
    "yahoo.ca",
    "yahoo.com.au",
    "yahoo.fr",
    "yahoo.de",
    "yahoo.es",
    "yahoo.it",
    "yahoo.co.jp",
}

DISPOSABLE_DOMAINS = {
    "mailinator.com",
    "guerrillamail.com",
    "guerrillamail.info",
    "10minutemail.com",
    "tempmail.com",
    "temp-mail.org",
    "yopmail.com",
    "trashmail.com",
    "sharklasers.com",
    "getnada.com",
    "dispostable.com",
    "throwawaymail.com",
    "maildrop.cc",
    "fakeinbox.com",
    "mintemail.com",
    "mohmal.com",
    "emailondeck.com",
    "burnermail.io",
}

_EMAIL_RE = re.compile(r"^[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+$")
_ph = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=2)
_oauth = OAuth()
_auth_attempts: dict[str, deque[float]] = {}

logger = logging.getLogger("echocount.auth")


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH), timeout=10.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db() -> None:
    """Create authentication, token, session, and job-ownership tables if missing."""
    with _conn() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT,
                email_verified INTEGER NOT NULL DEFAULT 0,
                role TEXT NOT NULL DEFAULT 'user',
                created_at REAL NOT NULL,
                last_login_at REAL
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
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                csrf_token TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                revoked INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS user_jobs (
                job_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                filename TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            """
        )
        cols = {r["name"] for r in con.execute("PRAGMA table_info(users)").fetchall()}
        if "role" not in cols:
            con.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
        if "is_active" not in cols:
            con.execute("ALTER TABLE users ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1")
        jcols = {r["name"] for r in con.execute("PRAGMA table_info(user_jobs)").fetchall()}
        if "status" not in jcols:
            con.execute("ALTER TABLE user_jobs ADD COLUMN status TEXT NOT NULL DEFAULT 'analyzed'")
        if "summary_json" not in jcols:
            con.execute("ALTER TABLE user_jobs ADD COLUMN summary_json TEXT")
    admin_email = os.getenv("ECHOCOUNT_ADMIN_EMAIL", "").strip()
    admin_pw = os.getenv("ECHOCOUNT_ADMIN_PASSWORD", "")
    if admin_email and admin_pw:
        ensure_admin_account(admin_email, admin_pw)
    cfg_admins = _admin_email_set()
    if cfg_admins:
        with _conn() as con:
            for em in cfg_admins:
                con.execute("UPDATE users SET role = 'admin' WHERE email = ?", (em,))


def _admin_email_set() -> set[str]:
    raw = os.getenv("ADMIN_EMAILS", "")
    return {normalize_email(x) for x in raw.split(",") if x.strip()}


def get_or_create_secret_key(key_file: Path | None = None) -> str:
    """Return SECRET_KEY from env or persist a stable random key locally (dev only)."""
    env_key = os.getenv("SECRET_KEY", "").strip()
    is_prod = (
        os.getenv("ECHOCOUNT_ENV", "").strip().lower() == "production"
        or os.getenv("FLASK_ENV", "").strip().lower() == "production"
    )
    if is_prod and (not env_key or len(env_key) < 32 or "replace-with" in env_key.lower()):
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


def ensure_admin_account(raw_email: str, password: str) -> int:
    """Idempotently create or update the permanent verified administrator account."""
    email = normalize_email(raw_email)
    if not email or not _EMAIL_RE.match(email):
        raise ValueError("Admin email must be a valid email address.")
    pw_err = validate_password(password, email)
    if pw_err:
        raise ValueError(pw_err)
    pw_hash = hash_password(password)
    now = time.time()
    with _conn() as con:
        row = con.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if row:
            uid = int(row["id"])
            con.execute(
                "UPDATE users SET password_hash = ?, email_verified = 1, role = 'admin' WHERE id = ?",
                (pw_hash, uid),
            )
            return uid
        cur = con.execute(
            "INSERT INTO users (email, password_hash, email_verified, role, created_at) VALUES (?, ?, 1, 'admin', ?)",
            (email, pw_hash, now),
        )
        return int(cur.lastrowid)


def init_oauth(app: Any) -> None:
    """Register Google, Microsoft, and Yahoo OIDC clients when configured via env vars."""
    _oauth.init_app(app)
    if os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET"):
        _oauth.register(
            name="google",
            client_id=os.getenv("GOOGLE_CLIENT_ID"),
            client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
            server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email"},
        )
    if os.getenv("MICROSOFT_CLIENT_ID") and os.getenv("MICROSOFT_CLIENT_SECRET"):
        tenant = os.getenv("MICROSOFT_TENANT_ID", "common").strip() or "common"
        _oauth.register(
            name="microsoft",
            client_id=os.getenv("MICROSOFT_CLIENT_ID"),
            client_secret=os.getenv("MICROSOFT_CLIENT_SECRET"),
            server_metadata_url=(
                f"https://login.microsoftonline.com/{tenant}/v2.0/.well-known/openid-configuration"
            ),
            client_kwargs={"scope": "openid email"},
        )
    if os.getenv("YAHOO_CLIENT_ID") and os.getenv("YAHOO_CLIENT_SECRET"):
        _oauth.register(
            name="yahoo",
            client_id=os.getenv("YAHOO_CLIENT_ID"),
            client_secret=os.getenv("YAHOO_CLIENT_SECRET"),
            server_metadata_url="https://api.login.yahoo.com/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email"},
        )
    if os.getenv("GITHUB_CLIENT_ID") and os.getenv("GITHUB_CLIENT_SECRET"):
        _oauth.register(
            name="github",
            client_id=os.getenv("GITHUB_CLIENT_ID"),
            client_secret=os.getenv("GITHUB_CLIENT_SECRET"),
            access_token_url="https://github.com/login/oauth/access_token",
            authorize_url="https://github.com/login/oauth/authorize",
            api_base_url="https://api.github.com/",
            client_kwargs={"scope": "read:user user:email"},
        )


def provider_availability() -> dict[str, bool]:
    """Return which supported OAuth providers have server credentials configured."""
    return {
        "google": bool(os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET")),
        "microsoft": bool(
            os.getenv("MICROSOFT_CLIENT_ID") and os.getenv("MICROSOFT_CLIENT_SECRET")
        ),
        "github": bool(os.getenv("GITHUB_CLIENT_ID") and os.getenv("GITHUB_CLIENT_SECRET")),
        "yahoo": bool(os.getenv("YAHOO_CLIENT_ID") and os.getenv("YAHOO_CLIENT_SECRET")),
    }


def normalize_email(raw: str) -> str:
    """Canonicalize an email address by stripping whitespace and lowercasing."""
    return (raw or "").strip().lower()


def _extra_allowed_domains() -> set[str]:
    raw = os.getenv("ECHOCOUNT_ALLOWED_DOMAINS", "")
    return {d.strip().lower() for d in raw.split(",") if d.strip()}


def validate_email_policy(
    raw_email: str, *, oauth_verified: bool = False
) -> tuple[str | None, str | None]:
    """Normalize and validate an email against disposable blocklist and provider policy."""
    email = normalize_email(raw_email)
    if not email or not _EMAIL_RE.match(email):
        return None, "Enter a valid email address."
    domain = email.rsplit("@", 1)[-1]
    if domain in DISPOSABLE_DOMAINS:
        return None, "Disposable or temporary email addresses are not permitted."
    if oauth_verified:
        return email, None
    allowed = GOOGLE_DOMAINS | MICROSOFT_DOMAINS | YAHOO_DOMAINS | _extra_allowed_domains()
    if domain not in allowed:
        return (
            None,
            "Use a legitimate Google, Microsoft, or Yahoo email address, or sign in with Google/Microsoft/Yahoo OAuth for corporate accounts.",
        )
    return email, None


def validate_password(password: str, email: str = "") -> str | None:
    """Enforce password length and basic safety constraints."""
    if not password or len(password) < MIN_PASSWORD_LEN:
        return f"Password must be at least {MIN_PASSWORD_LEN} characters."
    if len(password) > MAX_PASSWORD_LEN:
        return f"Password must be {MAX_PASSWORD_LEN} characters or fewer."
    if email and password.strip().lower() == normalize_email(email):
        return "Password cannot be the same as your email address."
    return None


def hash_password(password: str) -> str:
    """Hash a password using Argon2id with a unique random salt."""
    return _ph.hash(password)


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a plaintext password against a stored Argon2id hash."""
    if not stored_hash:
        return False
    try:
        return _ph.verify(stored_hash, password)
    except (VerifyMismatchError, Exception):
        return False


def verify_captcha_hook(_req: Any = None) -> bool:
    """Extensible CAPTCHA hook (no-op now; drop in Cloudflare Turnstile later without refactoring)."""
    return True


def check_auth_rate_limit(key: str, limit: int = 8, window_s: float = 300.0) -> bool:
    """Sliding-window brute-force limiter for login, registration, and password reset."""
    now = time.time()
    dq = _auth_attempts.setdefault(key, deque())
    while dq and now - dq[0] > window_s:
        dq.popleft()
    if len(dq) >= limit:
        return False
    dq.append(now)
    return True


def _hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def create_auth_token(user_id: int, purpose: str) -> str:
    """Generate a single-use time-limited opaque token stored only as SHA-256."""
    raw = secrets.token_urlsafe(32)
    now = time.time()
    with _conn() as con:
        con.execute(
            "INSERT INTO auth_tokens (user_id, token_hash, purpose, expires_at, used, created_at) VALUES (?, ?, ?, ?, 0, ?)",
            (user_id, _hash_token(raw), purpose, now + TOKEN_TTL_SECONDS, now),
        )
    return raw


def consume_auth_token(raw_token: str, purpose: str) -> int | None:
    """Validate and mark a single-use token as consumed; return user_id or None."""
    if not raw_token:
        return None
    h = _hash_token(raw_token)
    now = time.time()
    with _conn() as con:
        row = con.execute(
            "SELECT id, user_id, expires_at, used FROM auth_tokens WHERE token_hash = ? AND purpose = ?",
            (h, purpose),
        ).fetchone()
        if not row or row["used"] or float(row["expires_at"]) < now:
            return None
        con.execute("UPDATE auth_tokens SET used = 1 WHERE id = ?", (row["id"],))
        return int(row["user_id"])


def send_auth_email(to_email: str, subject: str, body: str) -> bool:
    """Send verification/reset email via SMTP if configured; return True if dispatched."""
    host = os.getenv("SMTP_HOST", "").strip()
    if not host:
        logger.info("SMTP_HOST not configured; auth email for %s prepared locally.", to_email)
        return False
    port = int(os.getenv("SMTP_PORT", "587"))
    sender = os.getenv("SMTP_FROM", "noreply@echocount.local")
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_email
    msg.set_content(body)
    with smtplib.SMTP(host, port, timeout=10) as smtp:
        if os.getenv("SMTP_TLS", "1") == "1":
            smtp.starttls()
        user, pwd = os.getenv("SMTP_USER", ""), os.getenv("SMTP_PASSWORD", "")
        if user and pwd:
            smtp.login(user, pwd)
        smtp.send_message(msg)
    return True


def create_session(user_id: int) -> tuple[str, str]:
    """Create a new server-side session and CSRF token for user_id."""
    sid = secrets.token_urlsafe(32)
    csrf = secrets.token_urlsafe(24)
    now = time.time()
    with _conn() as con:
        con.execute(
            "INSERT INTO sessions (id, user_id, csrf_token, created_at, expires_at, revoked) VALUES (?, ?, ?, ?, ?, 0)",
            (sid, user_id, csrf, now, now + SESSION_TTL_SECONDS),
        )
        con.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (now, user_id))
    return sid, csrf


def revoke_session(sid: str | None) -> None:
    """Invalidate a server-side session."""
    if not sid:
        return
    with _conn() as con:
        con.execute("UPDATE sessions SET revoked = 1 WHERE id = ?", (sid,))


def get_current_user() -> dict[str, Any] | None:
    """Load the authenticated active user and session CSRF token from the HttpOnly cookie."""
    if hasattr(g, "_auth_user"):
        return g._auth_user
    sid = request.cookies.get(SESSION_COOKIE)
    if not sid:
        g._auth_user = None
        return None
    now = time.time()
    with _conn() as con:
        row = con.execute(
            """
            SELECT u.id, u.email, u.email_verified, u.role, COALESCE(u.is_active, 1) AS is_active,
                   s.csrf_token, s.expires_at, s.revoked
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.id = ?
            """,
            (sid,),
        ).fetchone()
    if not row or row["revoked"] or not row["is_active"] or float(row["expires_at"]) < now:
        g._auth_user = None
        return None
    g._auth_user = {
        "id": int(row["id"]),
        "email": row["email"],
        "email_verified": bool(row["email_verified"]),
        "role": row["role"] or "user",
        "is_active": True,
        "csrf_token": row["csrf_token"],
        "sid": sid,
    }
    return g._auth_user


def get_or_create_form_csrf() -> str:
    """Return session CSRF token if logged in, else a per-browser pre-auth CSRF token."""
    user = get_current_user()
    if user:
        return user["csrf_token"]
    tok = session.get("_preauth_csrf")
    if not tok:
        tok = secrets.token_urlsafe(24)
        session["_preauth_csrf"] = tok
    return tok


def verify_csrf_request() -> bool:
    """Validate CSRF token from X-CSRF-Token header or form field."""
    submitted = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token") or ""
    user = get_current_user()
    expected = user["csrf_token"] if user else session.get("_preauth_csrf", "")
    return bool(expected and submitted and hmac.compare_digest(submitted, expected))


def set_session_cookie(resp: Any, sid: str) -> Any:
    """Attach secure HttpOnly session cookie to response."""
    secure = os.getenv("ECHOCOUNT_COOKIE_SECURE", "0") == "1" or request.is_secure
    resp.set_cookie(
        SESSION_COOKIE,
        sid,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite="Lax",
        secure=secure,
        path="/",
    )
    return resp


def clear_session_cookie(resp: Any) -> Any:
    """Delete session cookie on logout."""
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp


def default_landing_for_role(role: str | None) -> str:
    """Return the post-login landing path based on verified database role."""
    r = (role or "user").strip().lower()
    if r == "admin":
        return "/admin"
    if r == "developer":
        return "/admin/evaluation"
    return "/"


def _unauth_response() -> Any:
    if request.path.startswith("/api/"):
        return jsonify(
            {"error": "Authentication required. Please sign in.", "auth_required": True}
        ), 401
    return redirect(url_for("login_page", next=request.path))


def login_required(fn: Any) -> Any:
    """Require an authenticated active user for protected pages and private API endpoints."""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs) if get_current_user() else _unauth_response()

    return wrapper


def role_required(*allowed_roles: str) -> Any:
    """Require an authenticated user whose database role is in allowed_roles."""

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


def register_user_job(
    job_id: str, user_id: int, filename: str, summary: dict | None = None
) -> None:
    """Persist job ownership and optional summary so users can view their own history."""
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO user_jobs (job_id, user_id, filename, created_at, status, summary_json) VALUES (?, ?, ?, ?, 'analyzed', ?)",
            (job_id, user_id, filename, time.time(), json.dumps(summary) if summary else None),
        )


def update_user_job(job_id: str, status: str, summary: dict | None = None) -> None:
    """Update status and summary metadata for a persisted user job."""
    with _conn() as con:
        if summary is not None:
            con.execute(
                "UPDATE user_jobs SET status = ?, summary_json = ? WHERE job_id = ?",
                (status, json.dumps(summary), job_id),
            )
        else:
            con.execute("UPDATE user_jobs SET status = ? WHERE job_id = ?", (status, job_id))


def list_user_jobs(user_id: int, limit: int = 100) -> list[dict[str, Any]]:
    """Return recent analysis jobs belonging solely to user_id."""
    with _conn() as con:
        rows = con.execute(
            "SELECT job_id, user_id, filename, created_at, status, summary_json FROM user_jobs WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (int(user_id), int(limit)),
        ).fetchall()
    return [_format_job_row(r) for r in rows]


def list_all_jobs(user_id: int | None = None, limit: int = 200) -> list[dict[str, Any]]:
    """Return analysis jobs across all users (or filtered by user_id) for admin-only views."""
    where = "WHERE j.user_id = ?" if user_id is not None else ""
    params = (int(user_id), int(limit)) if user_id is not None else (int(limit),)
    with _conn() as con:
        rows = con.execute(
            f"""
            SELECT j.job_id, j.user_id, u.email AS user_email, j.filename, j.created_at, j.status, j.summary_json
            FROM user_jobs j
            JOIN users u ON u.id = j.user_id
            {where}
            ORDER BY j.created_at DESC LIMIT ?
            """,
            params,
        ).fetchall()
    return [_format_job_row(r) for r in rows]


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
        item["user_email"] = r["user_email"]
    return item


def delete_user_job(job_id: str, user: dict[str, Any]) -> bool:
    """Delete a job row only if owned by user or caller is admin."""
    with _conn() as con:
        if user.get("role") == "admin":
            cur = con.execute("DELETE FROM user_jobs WHERE job_id = ?", (job_id,))
        else:
            cur = con.execute(
                "DELETE FROM user_jobs WHERE job_id = ? AND user_id = ?",
                (job_id, int(user["id"])),
            )
    return cur.rowcount > 0


def clear_user_jobs(user_id: int) -> list[str]:
    """Delete all persisted jobs belonging solely to user_id and return deleted job_ids."""
    with _conn() as con:
        ids = [
            r["job_id"]
            for r in con.execute(
                "SELECT job_id FROM user_jobs WHERE user_id = ?", (int(user_id),)
            ).fetchall()
        ]
        con.execute("DELETE FROM user_jobs WHERE user_id = ?", (int(user_id),))
    return ids


def delete_user_account(target_uid: int, actor_uid: int) -> tuple[bool, list[str], str | None]:
    """Delete a user account and cascade its history/sessions; prevents self-deletion."""
    if int(target_uid) == int(actor_uid):
        return False, [], "You cannot delete your own administrator account."
    with _conn() as con:
        ids = [
            r["job_id"]
            for r in con.execute(
                "SELECT job_id FROM user_jobs WHERE user_id = ?", (int(target_uid),)
            ).fetchall()
        ]
        cur = con.execute("DELETE FROM users WHERE id = ?", (int(target_uid),))
        if cur.rowcount == 0:
            return False, [], "User account not found."
    return True, ids, None


def list_users() -> list[dict[str, Any]]:
    """Return all user accounts with active session counts and job counts (for Admin Dashboard)."""
    now = time.time()
    with _conn() as con:
        rows = con.execute(
            """
            SELECT u.id, u.email, u.role, u.email_verified, COALESCE(u.is_active, 1) AS is_active,
                   u.created_at, u.last_login_at,
                   (SELECT COUNT(*) FROM sessions s WHERE s.user_id = u.id AND s.revoked = 0 AND s.expires_at > ?) AS active_sessions,
                   (SELECT COUNT(*) FROM user_jobs j WHERE j.user_id = u.id) AS job_count
            FROM users u
            ORDER BY CASE u.role WHEN 'admin' THEN 0 WHEN 'developer' THEN 1 ELSE 2 END, u.created_at DESC
            """,
            (now,),
        ).fetchall()
    return [
        {
            "id": int(r["id"]),
            "email": r["email"],
            "role": r["role"] or "user",
            "email_verified": bool(r["email_verified"]),
            "is_active": bool(r["is_active"]),
            "created_at": r["created_at"],
            "last_login_at": r["last_login_at"],
            "active_sessions": int(r["active_sessions"]),
            "job_count": int(r["job_count"]),
        }
        for r in rows
    ]


def update_user_role(target_uid: int, new_role: str, actor_uid: int) -> tuple[bool, str | None]:
    """Update a user's role; prevents an admin from demoting their own account."""
    role = (new_role or "").strip().lower()
    if role not in VALID_ROLES:
        return False, f"Invalid role. Allowed roles: {', '.join(VALID_ROLES)}."
    if int(target_uid) == int(actor_uid) and role != "admin":
        return False, "You cannot remove administrator privileges from your own account."
    with _conn() as con:
        cur = con.execute("UPDATE users SET role = ? WHERE id = ?", (role, int(target_uid)))
        if cur.rowcount == 0:
            return False, "User account not found."
    return True, None


def set_user_active(target_uid: int, active: bool, actor_uid: int) -> tuple[bool, str | None]:
    """Enable or disable a user account; prevents self-deactivation and revokes sessions when disabled."""
    if int(target_uid) == int(actor_uid) and not active:
        return False, "You cannot deactivate your own administrator account."
    with _conn() as con:
        cur = con.execute(
            "UPDATE users SET is_active = ? WHERE id = ?", (1 if active else 0, int(target_uid))
        )
        if cur.rowcount == 0:
            return False, "User account not found."
        if not active:
            con.execute("UPDATE sessions SET revoked = 1 WHERE user_id = ?", (int(target_uid),))
    return True, None


def revoke_user_sessions(target_uid: int) -> int:
    """Revoke all active sessions for target_uid."""
    with _conn() as con:
        cur = con.execute(
            "UPDATE sessions SET revoked = 1 WHERE user_id = ? AND revoked = 0", (int(target_uid),)
        )
        return int(cur.rowcount)


def user_owns_job(job_id: str, user_id: int) -> bool:
    """Verify job_id belongs to user_id."""
    with _conn() as con:
        row = con.execute("SELECT user_id FROM user_jobs WHERE job_id = ?", (job_id,)).fetchone()
    return bool(row and int(row["user_id"]) == int(user_id))


def can_access_job(job_id: str, user: dict[str, Any] | None) -> bool:
    """Allow job access only to its owner or an administrator."""
    if not user or not job_id:
        return False
    if user.get("role") == "admin":
        with _conn() as con:
            row = con.execute("SELECT 1 FROM user_jobs WHERE job_id = ?", (job_id,)).fetchone()
        return row is not None
    return user_owns_job(job_id, int(user["id"]))


_USER_SELECT = "SELECT id, email, password_hash, email_verified, role, COALESCE(is_active, 1) AS is_active FROM users"


def register_oauth_user(
    provider: str, provider_sub: str, raw_email: str, email_verified: bool
) -> tuple[dict[str, Any] | None, str | None]:
    """Authenticate or safely link an OIDC identity (never linking unverified accounts)."""
    if provider not in SUPPORTED_PROVIDERS:
        return None, "Unsupported authentication provider."
    if not email_verified:
        return None, "Your provider account email must be verified before signing in."
    email, err = validate_email_policy(raw_email, oauth_verified=True)
    if err or not email:
        return None, err or "Invalid email address."

    now = time.time()
    with _conn() as con:
        existing_oauth = con.execute(
            "SELECT user_id FROM oauth_identities WHERE provider = ? AND provider_sub = ?",
            (provider, provider_sub),
        ).fetchone()
        if existing_oauth:
            user = con.execute(
                f"{_USER_SELECT} WHERE id = ?", (int(existing_oauth["user_id"]),)
            ).fetchone()
            if not user or not user["is_active"]:
                return None, "This account has been deactivated by an administrator."
            return dict(user), None

        existing_user = con.execute(f"{_USER_SELECT} WHERE email = ?", (email,)).fetchone()
        if existing_user:
            if not existing_user["is_active"]:
                return None, "This account has been deactivated by an administrator."
            if not existing_user["email_verified"]:
                return (
                    None,
                    "An unverified password account exists for this email. Verify it first before linking OAuth.",
                )
            uid = int(existing_user["id"])
        else:
            cur = con.execute(
                "INSERT INTO users (email, password_hash, email_verified, created_at) VALUES (?, NULL, 1, ?)",
                (email, now),
            )
            uid = int(cur.lastrowid)

        con.execute(
            "INSERT INTO oauth_identities (user_id, provider, provider_sub, email, email_verified, created_at) VALUES (?, ?, ?, ?, 1, ?)",
            (uid, provider, provider_sub, email, now),
        )
        return dict(con.execute(f"{_USER_SELECT} WHERE id = ?", (uid,)).fetchone()), None


def register_password_user(
    raw_email: str, password: str
) -> tuple[int | None, str | None, str | None]:
    """Create a pending password-based account and return (user_id, raw_verify_token, error)."""
    email, err = validate_email_policy(raw_email, oauth_verified=False)
    if err or not email:
        return None, None, err
    pw_err = validate_password(password, email)
    if pw_err:
        return None, None, pw_err

    pw_hash = hash_password(password)
    now = time.time()
    with _conn() as con:
        existing = con.execute(
            "SELECT id, email_verified FROM users WHERE email = ?", (email,)
        ).fetchone()
        if existing:
            if existing["email_verified"]:
                return (
                    None,
                    None,
                    "If this email is eligible, a verification link has been prepared.",
                )
            con.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?", (pw_hash, existing["id"])
            )
            uid = int(existing["id"])
        else:
            role = "admin" if email in _admin_email_set() else "user"
            cur = con.execute(
                "INSERT INTO users (email, password_hash, email_verified, role, created_at) VALUES (?, ?, 0, ?, ?)",
                (email, pw_hash, role, now),
            )
            uid = int(cur.lastrowid)
    return uid, create_auth_token(uid, "verify_email"), None


def authenticate_password_user(
    raw_email: str, password: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Verify email + password with constant generic failure message on invalid credentials."""
    email = normalize_email(raw_email)
    if not email or not password:
        return None, "Invalid email or password."
    with _conn() as con:
        row = con.execute(f"{_USER_SELECT} WHERE email = ?", (email,)).fetchone()
    if not row or not row["password_hash"] or not verify_password(password, row["password_hash"]):
        return None, "Invalid email or password."
    if not row["is_active"]:
        return None, "This account has been deactivated by an administrator."
    if not row["email_verified"]:
        return None, "Please verify your email address before signing in."
    return {
        "id": int(row["id"]),
        "email": row["email"],
        "email_verified": True,
        "role": row["role"] or "user",
    }, None
