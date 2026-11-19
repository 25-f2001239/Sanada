"""Authentication, OAuth 2.0/OIDC (Google, Microsoft, Yahoo), Argon2id, and session management."""

from __future__ import annotations

import hashlib
import hmac
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
from flask import g, jsonify, redirect, request, session, url_for

from config import DATA_DIR

DB_PATH = Path(os.getenv("ECHOCOUNT_DB_PATH", str(DATA_DIR / "echocount.db")))
SESSION_COOKIE = "echocount_sid"
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(7 * 24 * 3600)))
TOKEN_TTL_SECONDS = int(os.getenv("AUTH_TOKEN_TTL_SECONDS", "1800"))
MIN_PASSWORD_LEN = 10
MAX_PASSWORD_LEN = 128

SUPPORTED_PROVIDERS = ("google", "microsoft", "yahoo")

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
    admin_email = os.getenv("ECHOCOUNT_ADMIN_EMAIL", "").strip()
    admin_pw = os.getenv("ECHOCOUNT_ADMIN_PASSWORD", "")
    if admin_email and admin_pw:
        ensure_admin_account(admin_email, admin_pw)


def get_or_create_secret_key(key_file: Path | None = None) -> str:
    """Return SECRET_KEY from env or persist a stable random key locally across restarts."""
    env_key = os.getenv("SECRET_KEY", "").strip()
    if env_key:
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


def provider_availability() -> dict[str, bool]:
    """Return which of the three allowed OAuth providers have server credentials configured."""
    return {
        "google": bool(os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET")),
        "microsoft": bool(
            os.getenv("MICROSOFT_CLIENT_ID") and os.getenv("MICROSOFT_CLIENT_SECRET")
        ),
        "yahoo": bool(os.getenv("YAHOO_CLIENT_ID") and os.getenv("YAHOO_CLIENT_SECRET")),
    }


def normalize_email(raw: str) -> str:
    """Canonicalize an email address by stripping whitespace and lowercasing."""
    return (raw or "").strip().lower()


def _extra_allowed_domains() -> set[str]:
    raw = os.getenv("ECHOCOUNT_ALLOWED_DOMAINS", "")
    return {d.strip().lower() for d in raw.split(",") if d.strip()}


def validate_email_policy(raw_email: str, *, oauth_verified: bool = False) -> tuple[str | None, str | None]:
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
    """Load the authenticated user and session CSRF token from the HttpOnly cookie."""
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
            SELECT u.id, u.email, u.email_verified, u.role, s.csrf_token, s.expires_at, s.revoked
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.id = ?
            """,
            (sid,),
        ).fetchone()
    if not row or row["revoked"] or float(row["expires_at"]) < now:
        g._auth_user = None
        return None
    g._auth_user = {
        "id": int(row["id"]),
        "email": row["email"],
        "email_verified": bool(row["email_verified"]),
        "role": row["role"] or "user",
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


def login_required(fn: Any) -> Any:
    """Require an authenticated verified user for private API and analysis endpoints."""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        user = get_current_user()
        if not user:
            if request.path.startswith("/api/"):
                return jsonify({"error": "Authentication required. Please sign in.", "auth_required": True}), 401
            return redirect(url_for("login_page", next=request.path))
        return fn(*args, **kwargs)

    return wrapper


def register_user_job(job_id: str, user_id: int, filename: str) -> None:
    """Persist job ownership so users can never access another user's stems or ZIP."""
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO user_jobs (job_id, user_id, filename, created_at) VALUES (?, ?, ?, ?)",
            (job_id, user_id, filename, time.time()),
        )


def user_owns_job(job_id: str, user_id: int) -> bool:
    """Verify job_id belongs to user_id."""
    with _conn() as con:
        row = con.execute(
            "SELECT user_id FROM user_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
    return bool(row and int(row["user_id"]) == int(user_id))


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
            uid = int(existing_oauth["user_id"])
            user = con.execute("SELECT id, email, email_verified FROM users WHERE id = ?", (uid,)).fetchone()
            return dict(user), None

        existing_user = con.execute(
            "SELECT id, email, email_verified FROM users WHERE email = ?", (email,)
        ).fetchone()
        if existing_user:
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
        user = con.execute("SELECT id, email, email_verified FROM users WHERE id = ?", (uid,)).fetchone()
        return dict(user), None


def register_password_user(raw_email: str, password: str) -> tuple[int | None, str | None, str | None]:
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
        existing = con.execute("SELECT id, email_verified FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            if existing["email_verified"]:
                return None, None, "If this email is eligible, a verification link has been prepared."
            # Refresh unverified pending registration safely
            con.execute("UPDATE users SET password_hash = ? WHERE id = ?", (pw_hash, existing["id"]))
            uid = int(existing["id"])
        else:
            cur = con.execute(
                "INSERT INTO users (email, password_hash, email_verified, created_at) VALUES (?, ?, 0, ?)",
                (email, pw_hash, now),
            )
            uid = int(cur.lastrowid)
    token = create_auth_token(uid, "verify_email")
    return uid, token, None


def authenticate_password_user(raw_email: str, password: str) -> tuple[dict[str, Any] | None, str | None]:
    """Verify email + password with constant generic failure message on invalid credentials."""
    email = normalize_email(raw_email)
    if not email or not password:
        return None, "Invalid email or password."
    with _conn() as con:
        row = con.execute(
            "SELECT id, email, password_hash, email_verified, role FROM users WHERE email = ?", (email,)
        ).fetchone()
    if not row or not row["password_hash"] or not verify_password(password, row["password_hash"]):
        return None, "Invalid email or password."
    if not row["email_verified"]:
        return None, "Please verify your email address before signing in."
    return {
        "id": int(row["id"]),
        "email": row["email"],
        "email_verified": True,
        "role": row["role"] or "user",
    }, None
