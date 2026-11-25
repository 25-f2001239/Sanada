"""Comprehensive automated tests for EchoCount Auth Overhaul & Single Locked Admin + Account Lifecycle.

Covers:
  - Phase 1: Complete removal of Microsoft, Yahoo, GitHub (routes, UI, docs, env) + OAuth migration
  - Phase 2: 6-digit Email OTP (HMAC-SHA256 storage, expiry, single-use, 5-attempt lockout, 60s resend
             cooldown, invalidating old codes, per-email/per-IP limits, daily budget, 24h purge,
             forgot-password OTP, non-enumerating messages, production missing-mail handling,
             session rotation, HttpOnly/SameSite, CSRF, common weak password rejection, security headers)
  - Phase 3: Google Sign-In (Authlib OIDC + optional Firebase ID-token RS256 verification)
  - Mission B1: Single locked admin (.env ADMIN_EMAILS), production startup refusal (0 or >1),
                role-change 403, admin password+OTP step-up every sign-in, Google blocked from admin,
                shorter admin session TTL, ZERO unmasked admin email leaks across all pages/APIs/exports
  - Mission B2: Settings -> Account (JSON/CSV export, deactivate + session kill + hidden history + OTP
                reactivation, permanent delete with fresh re-auth + OTP + DELETE + DB/disk verification
                + fresh re-registration, admin disable/enable/delete non-admin + refuse admin + ID-only audit log)
"""

from __future__ import annotations

import base64
import datetime
import json
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app as app_mod
import auth
import mail as mail_mod


class _MockMail(mail_mod.MailProvider):
    def __init__(self) -> None:
        self.sent: list[dict] = []

    def send(self, to_email: str, subject: str, body: str) -> tuple[bool, str | None]:
        self.sent.append({"to": to_email, "subject": subject, "body": body})
        return True, None

    def last_code(self, to_email: str | None = None) -> str:
        for msg in reversed(self.sent):
            if to_email is None or msg["to"].lower() == to_email.lower():
                m = re.search(r"\b(\d{6})\b", msg["body"])
                if m:
                    return m.group(1)
        raise AssertionError(f"No 6-digit OTP found for {to_email}")


def _csrf(client, path: str = "/login") -> str:
    r = client.get(path)
    html = r.get_data(as_text=True)
    m = re.search(r'<meta name="csrf-token" content="([^"]+)"', html)
    assert m, f"Missing CSRF meta on {path} (status={r.status_code})"
    return m.group(1)


def _session_cookie_val(client) -> str:
    c = client.get_cookie(auth.SESSION_COOKIE)
    return c.value if c else ""


def _clear_rate_limits() -> None:
    auth._rate_buckets.clear()
    auth._fail_buckets.clear()


def _make_firebase_test_jwt(private_key, payload: dict, kid: str = "test-kid-1") -> str:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    def _b64(b: bytes) -> str:
        return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")

    hdr = _b64(json.dumps({"alg": "RS256", "typ": "JWT", "kid": kid}).encode("utf-8"))
    pld = _b64(json.dumps(payload).encode("utf-8"))
    sig = private_key.sign(f"{hdr}.{pld}".encode("ascii"), padding.PKCS1v15(), hashes.SHA256())
    return f"{hdr}.{pld}.{_b64(sig)}"


def _generate_test_rsa_cert():
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.datetime.now(datetime.timezone.utc)
    subj = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test-firebase")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subj)
        .issuer_name(subj)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    pem = cert.public_bytes(serialization.Encoding.PEM).decode("ascii")
    return key, pem


def test_1_legacy_oauth_removed_and_migration(mock_mail: _MockMail) -> None:
    """Phase 1: Zero Microsoft/Yahoo/GitHub references, routes return 404, OAuth migration preserves users."""
    scan_targets = [
        ROOT / "app.py",
        ROOT / "auth.py",
        ROOT / "mail.py",
        ROOT / ".env.example",
        ROOT / "README.md",
        *(ROOT / "templates").glob("*.html"),
        *(ROOT / "static").glob("*.js"),
    ]
    pat = re.compile(r"microsoft|yahoo|github_client|/auth/oauth/github|github.*sign|sign.*github", re.I)
    for fp in scan_targets:
        text = fp.read_text(encoding="utf-8", errors="ignore")
        assert not pat.search(text), f"Legacy OAuth reference still found in {fp.relative_to(ROOT)}"

    c = app_mod.app.test_client()
    for p in ("microsoft", "yahoo", "github"):
        assert c.get(f"/auth/oauth/{p}").status_code == 404
        assert c.get(f"/auth/oauth/{p}/callback").status_code == 404

    # Google button always visible on BOTH tabs: disabled+helper when unconfigured, enabled when configured
    os.environ.pop("GOOGLE_CLIENT_ID", None)
    os.environ.pop("GOOGLE_CLIENT_SECRET", None)
    auth.init_oauth(app_mod.app)
    html_no_g = c.get("/login").get_data(as_text=True)
    assert html_no_g.count("google-auth-btn") == 2, "Expected exactly 1 Google button per tab (2 total)"
    assert "Continue with Google" in html_no_g and "Sign up with Google" in html_no_g
    assert html_no_g.count("Google sign-in is not set up on this server yet") == 2
    assert 'disabled aria-disabled="true"' in html_no_g
    assert "#4285F4" in html_no_g and "#34A853" in html_no_g and "#FBBC05" in html_no_g and "#EA4335" in html_no_g
    assert "Microsoft" not in html_no_g and "Yahoo" not in html_no_g and "GitHub" not in html_no_g
    r_g_unconf = c.get("/auth/oauth/google")
    assert r_g_unconf.status_code == 503 and "not set up" in r_g_unconf.get_data(as_text=True).lower()
    assert "Traceback" not in r_g_unconf.get_data(as_text=True)

    os.environ["GOOGLE_CLIENT_ID"] = "fake-google-id"
    os.environ["GOOGLE_CLIENT_SECRET"] = "fake-google-secret"
    auth.init_oauth(app_mod.app)
    html_with_g = c.get("/login").get_data(as_text=True)
    assert html_with_g.count("google-auth-btn") == 2
    assert html_with_g.count('href="/auth/oauth/google"') == 2
    assert "Google sign-in is not set up on this server yet" not in html_with_g
    assert "Continue with Google" in html_with_g and "Sign up with Google" in html_with_g
    assert "Microsoft" not in html_with_g and "Yahoo" not in html_with_g and "GitHub" not in html_with_g

    # Data migration: removes non-Google oauth_identities & marks pre-existing users verified ONCE
    with auth._conn() as con:
        con.execute("DELETE FROM schema_migrations WHERE name = 'mark_preexisting_users_verified_v1'")
        cur = con.execute(
            "INSERT INTO users(email, password_hash, email_verified, role, is_active, created_at) VALUES (?, '', 0, 'user', 1, ?)",
            ("legacy_ms@example.test", time.time()),
        )
        uid = int(cur.lastrowid)
        con.execute(
            "INSERT INTO oauth_identities(user_id, provider, provider_sub, email, created_at) VALUES (?, 'microsoft', 'ms-sub-1', ?, ?)",
            (uid, "legacy_ms@example.test", time.time()),
        )
    auth.init_db()
    with auth._conn() as con:
        u_row = con.execute("SELECT email_verified FROM users WHERE id = ?", (uid,)).fetchone()
        assert u_row and u_row["email_verified"] == 1
        assert con.execute("SELECT COUNT(*) FROM oauth_identities WHERE provider != 'google'").fetchone()[0] == 0
    # New unverified user created AFTER migration stays unverified across subsequent init_db()
    uid_new_unv, _, _ = auth.register_password_user("after_mig@example.test", "StrongPass!2026", check_mx=False)
    auth.init_db()
    with auth._conn() as con:
        assert con.execute("SELECT email_verified FROM users WHERE id = ?", (uid_new_unv,)).fetchone()["email_verified"] == 0
    print("  [PASS] 1. Always-visible Google button, legacy OAuth removal & one-time verification migration.")


def test_2_email_otp_security_and_hardening(mock_mail: _MockMail) -> None:
    """Phase 2 & 4: Full OTP lifecycle, HMAC storage, limits, cooldown, budget, purge, forgot-pw, CSRF, headers."""
    _clear_rate_limits()
    c = app_mod.app.test_client()
    r_login = c.get("/login")
    # Security headers check
    for hdr, expected in (
        ("X-Content-Type-Options", "nosniff"),
        ("X-Frame-Options", "DENY"),
        ("Referrer-Policy", "no-referrer"),
    ):
        assert r_login.headers.get(hdr) == expected, f"Bad header {hdr}: {r_login.headers.get(hdr)}"
    assert "default-src 'self'" in r_login.headers.get("Content-Security-Policy", "")
    assert "camera=()" in r_login.headers.get("Permissions-Policy", "")

    # Part B HTML structure checks:
    # - Forgot password link ONLY on Sign In panel, NEVER on Create Account panel
    # - Confirm password input present on Create Account panel
    # - OTP verify panel contains ZERO password inputs (code-only!)
    # - Forgot Reset Step 2 panel contains password + confirm password inputs
    login_html = r_login.get_data(as_text=True)
    p_login = re.search(r'data-auth-panel="login".*?data-auth-panel="register"', login_html, re.S).group(0)
    p_reg = re.search(r'data-auth-panel="register".*?data-auth-panel="forgot"', login_html, re.S).group(0)
    p_otp = re.search(r'data-auth-panel="otp".*?data-auth-panel="forgot_reset"', login_html, re.S).group(0)
    p_fgr = re.search(r'data-auth-panel="forgot_reset".*?data-auth-panel="reset"', login_html, re.S).group(0)
    assert 'id="openForgotBtn"' in p_login
    assert "Forgot password" not in p_reg and 'name="password_confirm"' in p_reg
    assert 'type="password"' not in p_otp and 'id="otpChangeEmailBtn"' in p_otp and "Step 2 of 2" in p_otp
    assert 'id="fgNewPassword"' in p_fgr and 'id="fgNewPasswordConfirm"' in p_fgr

    csrf = _csrf(c, "/login")

    # CSRF enforced on POST
    assert c.post("/auth/register", json={"email": "a@b.test", "password": "StrongPassword!1", "password_confirm": "StrongPassword!1"}).status_code == 403

    # Mismatched password confirmation rejected (400)
    assert c.post(
        "/auth/register",
        json={"email": "mis@temp-mail.test", "password": "StrongPassword!2026", "password_confirm": "OtherPassword!2026"},
        headers={"X-CSRF-Token": csrf},
    ).status_code == 400

    # Common weak password rejected
    r_weak = c.post(
        "/auth/register",
        json={"email": "weakuser@temp-mail.test", "password": "password123", "password_confirm": "password123"},
        headers={"X-CSRF-Token": csrf},
    )
    assert r_weak.status_code == 400

    # Change email deletes the unverified account cleanly and returns typed email
    ch_email = "change_me@10minutemail.test"
    r_ch_reg = c.post(
        "/auth/register",
        json={"email": ch_email, "password": "UniqueStrongPass!2026", "password_confirm": "UniqueStrongPass!2026"},
        headers={"X-CSRF-Token": csrf},
    )
    assert r_ch_reg.status_code == 201
    r_ch_can = c.post("/auth/signup/cancel", json={}, headers={"X-CSRF-Token": csrf})
    assert r_ch_can.status_code == 200 and r_ch_can.get_json()["deleted"] is True and r_ch_can.get_json()["email"] == ch_email
    with auth._conn() as con:
        assert con.execute("SELECT COUNT(*) FROM users WHERE email = ?", (ch_email,)).fetchone()[0] == 0

    # Temp-mail signup succeeds -> OTP stored ONLY as HMAC hash (never plaintext)
    email = "temp_user@10minutemail.test"
    pw = "UniqueStrongPass!2026"
    r_reg = c.post(
        "/auth/register",
        json={"email": email, "password": pw, "password_confirm": pw},
        headers={"X-CSRF-Token": csrf},
    )
    assert r_reg.status_code == 201 and r_reg.get_json()["otp_required"] is True
    code1 = mock_mail.last_code(email)

    with auth._conn() as con:
        row = con.execute("SELECT code_hash, salt FROM email_otps WHERE email = ?", (email,)).fetchone()
        assert row and len(row["code_hash"]) == 64 and code1 not in row["code_hash"]

    # Unverified account signing in later opens code-only verify view (no session until verified)
    with auth._conn() as con:
        con.execute("UPDATE email_otps SET created_at = created_at - 120 WHERE email = ?", (email,))
    r_unv_login = c.post("/auth/login", json={"email": email, "password": pw}, headers={"X-CSRF-Token": csrf})
    assert r_unv_login.status_code == 200 and r_unv_login.get_json().get("otp_required") is True
    assert r_unv_login.get_json().get("purpose") == "signup"
    assert c.get("/api/history").status_code == 401
    code1 = mock_mail.last_code(email)

    # Immediate resend blocked by 60s cooldown
    r_cd = c.post("/auth/otp/resend", json={"email": email, "purpose": "signup"}, headers={"X-CSRF-Token": csrf})
    assert r_cd.status_code == 429

    # Wrong OTP x5 invalidates code so even the real code fails
    for _ in range(5):
        wrong = "000000" if code1 != "000000" else "999999"
        rw = c.post("/auth/otp/verify", json={"email": email, "purpose": "signup", "code": wrong}, headers={"X-CSRF-Token": csrf})
        assert rw.status_code == 400
    r_after5 = c.post("/auth/otp/verify", json={"email": email, "purpose": "signup", "code": code1}, headers={"X-CSRF-Token": csrf})
    assert r_after5.status_code == 400

    # Backdate previous OTP past cooldown, resend -> new code invalidates old code
    with auth._conn() as con:
        con.execute("UPDATE email_otps SET created_at = created_at - 120 WHERE email = ?", (email,))
    r_res = c.post("/auth/otp/resend", json={"email": email, "purpose": "signup"}, headers={"X-CSRF-Token": csrf})
    assert r_res.status_code == 200
    code2 = mock_mail.last_code(email)

    # Backdate again & resend -> code2 becomes invalidated by code3
    with auth._conn() as con:
        con.execute("UPDATE email_otps SET created_at = created_at - 120 WHERE email = ?", (email,))
    assert c.post("/auth/otp/resend", json={"email": email, "purpose": "signup"}, headers={"X-CSRF-Token": csrf}).status_code == 200
    code3 = mock_mail.last_code(email)
    assert c.post("/auth/otp/verify", json={"email": email, "purpose": "signup", "code": code2}, headers={"X-CSRF-Token": csrf}).status_code == 400

    # Expired OTP rejected
    with auth._conn() as con:
        con.execute("UPDATE email_otps SET expires_at = ? WHERE email = ? AND used = 0", (time.time() - 10, email))
    assert c.post("/auth/otp/verify", json={"email": email, "purpose": "signup", "code": code3}, headers={"X-CSRF-Token": csrf}).status_code == 400

    # Un-expire and verify -> logs in with HttpOnly, SameSite=Lax, and single-use OTP
    with auth._conn() as con:
        con.execute("UPDATE email_otps SET expires_at = ? WHERE email = ? AND used = 0", (time.time() + 300, email))
    r_ok = c.post("/auth/otp/verify", json={"email": email, "purpose": "signup", "code": code3}, headers={"X-CSRF-Token": csrf})
    assert r_ok.status_code == 200 and r_ok.get_json()["ok"] is True
    set_cookie = " ".join(r_ok.headers.getlist("Set-Cookie"))
    assert "HttpOnly" in set_cookie and "SameSite=Lax" in set_cookie
    sid1 = _session_cookie_val(c)
    assert sid1

    # Reusing same OTP fails
    csrf_post_login = _csrf(c, "/history")
    assert c.post("/auth/otp/verify", json={"email": email, "purpose": "signup", "code": code3}, headers={"X-CSRF-Token": csrf_post_login}).status_code == 400

    # Existing verified user signs in directly: session rotates, ZERO OTP rows created, ZERO emails sent
    sent_before_login = len(mock_mail.sent)
    with auth._conn() as con:
        otps_before_login = con.execute("SELECT COUNT(*) FROM email_otps").fetchone()[0]
    r_re = c.post("/auth/login", json={"email": email, "password": pw}, headers={"X-CSRF-Token": _csrf(c, "/history")})
    assert r_re.status_code == 200 and not r_re.get_json().get("otp_required")
    sid2 = _session_cookie_val(c)
    assert sid2 and sid2 != sid1
    assert len(mock_mail.sent) == sent_before_login, "Normal sign-in must send zero emails"
    with auth._conn() as con:
        assert con.execute("SELECT COUNT(*) FROM email_otps").fetchone()[0] == otps_before_login

    # Brute-force protection locks out repeated wrong passwords (429)
    _clear_rate_limits()
    c_bf = app_mod.app.test_client()
    csrf_bf = _csrf(c_bf, "/login")
    for _ in range(6):
        r_bad = c_bf.post("/auth/login", json={"email": email, "password": "WrongPassword!999"}, headers={"X-CSRF-Token": csrf_bf})
        assert r_bad.status_code == 401
    r_lock = c_bf.post("/auth/login", json={"email": email, "password": pw}, headers={"X-CSRF-Token": csrf_bf})
    assert r_lock.status_code == 429
    _clear_rate_limits()

    # Re-signing up with existing email: identical generic reply, ZERO OTP created, informational notice only
    c2 = app_mod.app.test_client()
    csrf2 = _csrf(c2, "/login")
    with auth._conn() as con:
        otps_pre_dup = con.execute("SELECT COUNT(*) FROM email_otps").fetchone()[0]
    sent_pre_dup = len(mock_mail.sent)
    r_dup = c2.post("/auth/register", json={"email": email, "password": "DifferentPassword!2026", "password_confirm": "DifferentPassword!2026"}, headers={"X-CSRF-Token": csrf2})
    assert r_dup.status_code == 201 and r_dup.get_json()["message"] == r_reg.get_json()["message"]
    with auth._conn() as con:
        assert con.execute("SELECT COUNT(*) FROM email_otps").fetchone()[0] == otps_pre_dup, "Duplicate signup must create zero OTP rows"
    assert len(mock_mail.sent) == sent_pre_dup + 1
    dup_mail = mock_mail.sent[-1]
    assert "sign in instead" in dup_mail["body"].lower() and not re.search(r"\b\d{6}\b", dup_mail["body"])

    r_fg_real = c2.post("/auth/forgot", json={"email": email}, headers={"X-CSRF-Token": csrf2})
    r_fg_none = c2.post("/auth/forgot", json={"email": "nobody_here@example.test"}, headers={"X-CSRF-Token": csrf2})
    assert r_fg_real.status_code == 200 and r_fg_none.status_code == 200
    assert r_fg_real.get_json()["message"] == r_fg_none.get_json()["message"]

    # Complete forgot-password OTP -> requires matching confirmation, updates password & revokes prior sessions
    fg_code = mock_mail.last_code(email)
    new_pw = "BrandNewStrongPass!2026"
    assert c2.post(
        "/auth/otp/verify",
        json={"email": email, "purpose": "forgot", "code": fg_code, "password": new_pw, "password_confirm": "Mismatch!2026"},
        headers={"X-CSRF-Token": csrf2},
    ).status_code == 400
    # Re-issue a fresh forgot OTP since mismatch didn't consume it? Actually verify_email_otp consumed it before pw check — check!
    with auth._conn() as con:
        con.execute("UPDATE email_otps SET used = 0 WHERE email = ? AND purpose = 'forgot'", (email,))
    r_fgv = c2.post(
        "/auth/otp/verify",
        json={"email": email, "purpose": "forgot", "code": fg_code, "password": new_pw, "password_confirm": new_pw},
        headers={"X-CSRF-Token": csrf2},
    )
    assert r_fgv.status_code == 200
    assert c.get("/api/history").status_code == 401  # old session revoked
    assert c2.post("/auth/login", json={"email": email, "password": new_pw}, headers={"X-CSRF-Token": csrf2}).status_code == 200

    # Per-email hourly limit (5/hr) & daily global budget enforcement
    _clear_rate_limits()
    lim_email = "limit_test@example.test"
    for i in range(5):
        with auth._conn() as con:
            con.execute("UPDATE email_otps SET created_at = created_at - 120 WHERE email = ?", (lim_email,))
        ok, err, _ = auth.issue_email_otp(lim_email, "signup", ip="10.0.0.1")
        assert ok, f"Unexpected failure on send #{i+1}: {err}"
    with auth._conn() as con:
        con.execute("UPDATE email_otps SET created_at = created_at - 120 WHERE email = ?", (lim_email,))
    ok6, err6, _ = auth.issue_email_otp(lim_email, "signup", ip="10.0.0.1")
    assert not ok6 and "too many" in (err6 or "").lower()

    # Global daily budget enforcement
    prev_budget = os.environ.get("OTP_DAILY_BUDGET")
    os.environ["OTP_DAILY_BUDGET"] = "1"
    try:
        ok_b, err_b, _ = auth.issue_email_otp("budget_blocked@example.test", "signup", ip="10.0.0.2")
        assert not ok_b and "quota" in (err_b or "").lower()
    finally:
        if prev_budget is None:
            os.environ.pop("OTP_DAILY_BUDGET", None)
        else:
            os.environ["OTP_DAILY_BUDGET"] = prev_budget

    # 24-hour purge of unverified accounts & expired OTPs
    stale_ts = time.time() - (25 * 3600)
    with auth._conn() as con:
        con.execute(
            "INSERT INTO users(email, password_hash, email_verified, role, is_active, created_at) VALUES (?, 'x', 0, 'user', 1, ?)",
            ("stale_unverified@example.test", stale_ts),
        )
        con.execute(
            "INSERT INTO email_otps(email, purpose, code_hash, salt, expires_at, attempts, used, created_at) VALUES (?, 'signup', 'h', 's', ?, 0, 0, ?)",
            ("stale_unverified@example.test", stale_ts, stale_ts),
        )
    purged = auth.purge_expired_auth_data()
    assert purged["unverified_deleted"] >= 1 and purged["otps_deleted"] >= 1

    # Missing mail provider in production -> app still runs, signup returns 503 "not available"
    mail_mod.set_provider(None)
    prev_env = os.environ.get("ECHOCOUNT_ENV")
    prev_dev = os.environ.get("ECHOCOUNT_DEV")
    prev_brevo = os.environ.pop("BREVO_API_KEY", None)
    prev_smtp = os.environ.pop("SMTP_HOST", None)
    os.environ["ECHOCOUNT_ENV"] = "production"
    os.environ["ECHOCOUNT_DEV"] = "0"
    try:
        c3 = app_mod.app.test_client()
        csrf3 = _csrf(c3, "/login")
        r_nomail = c3.post(
            "/auth/register",
            json={"email": "nomail@example.test", "password": "StrongPassword!2026", "password_confirm": "StrongPassword!2026"},
            headers={"X-CSRF-Token": csrf3},
        )
        assert r_nomail.status_code == 503
        assert "not available" in (r_nomail.get_json().get("error") or "").lower()
    finally:
        if prev_env is None:
            os.environ.pop("ECHOCOUNT_ENV", None)
        else:
            os.environ["ECHOCOUNT_ENV"] = prev_env
        if prev_dev is None:
            os.environ.pop("ECHOCOUNT_DEV", None)
        else:
            os.environ["ECHOCOUNT_DEV"] = prev_dev
        if prev_brevo is not None:
            os.environ["BREVO_API_KEY"] = prev_brevo
        if prev_smtp is not None:
            os.environ["SMTP_HOST"] = prev_smtp
        mail_mod.set_provider(mock_mail)

    print("  [PASS] 2. Email OTP lifecycle, HMAC storage, limits, budget, purge, forgot-pw, CSRF & headers.")


def test_3_google_authlib_and_firebase(mock_mail: _MockMail) -> None:
    """Phase 3: Authlib Google OIDC rules (sub match, no unverified merge, no admin) & Firebase RS256 verification."""
    _clear_rate_limits()
    # Unverified password account must NOT be auto-merged by Google OAuth
    uid_unv, _, err = auth.register_password_user("unverified_merge@example.test", "StrongPassword!2026", check_mx=False)
    assert uid_unv and not err
    u_res, u_err = auth.register_oauth_user("google", "g-sub-unv", "unverified_merge@example.test", True)
    assert u_res is None and u_err and "verify" in u_err.lower()

    # Unverified Google email rejected
    u_res2, u_err2 = auth.register_oauth_user("google", "g-sub-bad", "guser@example.test", False)
    assert u_res2 is None and u_err2

    # Valid Google sign-in creates verified password-less account matched by sub
    u_g, g_err = auth.register_oauth_user("google", "g-sub-12345", "guser@example.test", True)
    assert u_g and not g_err and u_g["email_verified"] == 1 and u_g["role"] == "user"
    # Second sign-in matches same sub
    u_g2, _ = auth.register_oauth_user("google", "g-sub-12345", "guser@example.test", True)
    assert u_g2["id"] == u_g["id"]

    # Optional Firebase Google ID-token verification (valid, wrong aud, expired, unverified, bad sig)
    from cryptography import x509 as _x509
    from cryptography.hazmat.primitives import hashes as _hashes
    from cryptography.hazmat.primitives.asymmetric import padding as _padding

    key1, pem1 = _generate_test_rsa_cert()
    key2, _pem2 = _generate_test_rsa_cert()
    pub1 = _x509.load_pem_x509_certificate(pem1.encode("ascii")).public_key()

    def _rs256_verify(tok: str) -> dict:
        parts = tok.split(".")
        assert len(parts) == 3
        sig = base64.urlsafe_b64decode(parts[2] + "===")
        pub1.verify(sig, f"{parts[0]}.{parts[1]}".encode("ascii"), _padding.PKCS1v15(), _hashes.SHA256())
        return json.loads(base64.urlsafe_b64decode(parts[1] + "===").decode("utf-8"))

    prev_mode = os.environ.get("AUTH_GOOGLE_MODE")
    prev_proj = os.environ.get("FIREBASE_PROJECT_ID")
    prev_api = os.environ.get("FIREBASE_WEB_API_KEY")
    os.environ["AUTH_GOOGLE_MODE"] = "firebase"
    os.environ["FIREBASE_PROJECT_ID"] = "echocount-test-proj"
    os.environ["FIREBASE_WEB_API_KEY"] = "fake-web-key"
    orig_vfn = auth.verify_firebase_google_token
    auth.verify_firebase_google_token = lambda tok, **kw: orig_vfn(tok, verifier=_rs256_verify, **kw)
    try:
        now = int(time.time())
        base_claims = {
            "iss": "https://securetoken.google.com/echocount-test-proj",
            "aud": "echocount-test-proj",
            "sub": "fb-google-uid-77",
            "email": "fbuser@example.test",
            "email_verified": True,
            "iat": now - 10,
            "exp": now + 600,
            "auth_time": now - 10,
            "firebase": {"sign_in_provider": "google.com"},
        }
        good_tok = _make_firebase_test_jwt(key1, base_claims)
        claims, cerr = auth.verify_firebase_google_token(good_tok)
        assert claims and not cerr and claims["sub"] == "fb-google-uid-77"

        # Bad signature (signed with wrong key)
        bad_sig_tok = _make_firebase_test_jwt(key2, base_claims)
        assert auth.verify_firebase_google_token(bad_sig_tok)[0] is None

        # Wrong audience
        bad_aud = dict(base_claims, aud="other-project")
        assert auth.verify_firebase_google_token(_make_firebase_test_jwt(key1, bad_aud))[0] is None

        # Expired token
        exp_tok = dict(base_claims, exp=now - 60)
        assert auth.verify_firebase_google_token(_make_firebase_test_jwt(key1, exp_tok))[0] is None

        # Unverified email
        unv_tok = dict(base_claims, email_verified=False)
        assert auth.verify_firebase_google_token(_make_firebase_test_jwt(key1, unv_tok))[0] is None

        # Route POST /auth/firebase/google works end-to-end
        cf = app_mod.app.test_client()
        csrf_f = _csrf(cf, "/login")
        rf = cf.post("/auth/firebase/google", json={"id_token": good_tok}, headers={"X-CSRF-Token": csrf_f})
        assert rf.status_code == 200 and rf.get_json()["ok"] is True
        assert cf.get("/api/me").get_json()["email"] == "fbuser@example.test"
    finally:
        auth.verify_firebase_google_token = orig_vfn
        if prev_mode is None:
            os.environ.pop("AUTH_GOOGLE_MODE", None)
        else:
            os.environ["AUTH_GOOGLE_MODE"] = prev_mode
        if prev_proj is None:
            os.environ.pop("FIREBASE_PROJECT_ID", None)
        else:
            os.environ["FIREBASE_PROJECT_ID"] = prev_proj
        if prev_api is None:
            os.environ.pop("FIREBASE_WEB_API_KEY", None)
        else:
            os.environ["FIREBASE_WEB_API_KEY"] = prev_api
        auth.init_oauth(app_mod.app)

    print("  [PASS] 3. Google OIDC (Authlib) & Firebase RS256 verification passed.")


def test_4_single_locked_admin_and_zero_leak(mock_mail: _MockMail) -> None:
    """Mission B1: Single locked admin, startup guards, step-up OTP, no OAuth, role 403, zero email leaks."""
    _clear_rate_limits()
    admin_local = "secret_adm_99"
    admin_email = f"{admin_local}@example.test"
    admin_pw = "AdminVaultStrong!2026"

    # Production startup refuses 0 or >1 ADMIN_EMAILS
    prev_env = os.environ.get("ECHOCOUNT_ENV")
    os.environ["ECHOCOUNT_ENV"] = "production"
    try:
        os.environ["ADMIN_EMAILS"] = ""
        try:
            auth.enforce_single_admin()
            raise AssertionError("Expected RuntimeError when ADMIN_EMAILS is empty in production")
        except RuntimeError:
            pass

        os.environ["ADMIN_EMAILS"] = "one@example.test,two@example.test"
        try:
            auth.enforce_single_admin()
            raise AssertionError("Expected RuntimeError when ADMIN_EMAILS has >1 address in production")
        except RuntimeError:
            pass

        # Configure single admin and verify all other admins are demoted on startup
        os.environ["ADMIN_EMAILS"] = admin_email
        with auth._conn() as con:
            con.execute(
                "INSERT INTO users(email, password_hash, email_verified, role, is_active, created_at) VALUES (?, ?, 1, 'admin', 1, ?)",
                (admin_email, auth.hash_password(admin_pw), time.time()),
            )
            con.execute(
                "INSERT INTO users(email, password_hash, email_verified, role, is_active, created_at) VALUES (?, ?, 1, 'admin', 1, ?)",
                ("rogue_admin@example.test", auth.hash_password("RoguePass!2026"), time.time()),
            )
        enf = auth.enforce_single_admin()
    finally:
        if prev_env is None:
            os.environ.pop("ECHOCOUNT_ENV", None)
        else:
            os.environ["ECHOCOUNT_ENV"] = prev_env
    assert enf["demoted"] >= 1 and enf["active_admins"] == 1 and admin_local not in (enf["configured_masked"] or "")
    with auth._conn() as con:
        admins = con.execute("SELECT email FROM users WHERE role = 'admin'").fetchall()
        assert len(admins) == 1 and admins[0]["email"] == admin_email

    # Google OAuth cannot log into the admin account
    u_adm_g, err_adm_g = auth.register_oauth_user("google", "g-sub-adm", admin_email, True)
    assert u_adm_g is None and err_adm_g

    # Admin email cannot be signed up or reset via website (returns generic non-enumerating response, sends no forgot OTP)
    c_anon = app_mod.app.test_client()
    csrf_a = _csrf(c_anon, "/login")
    sent_before = len(mock_mail.sent)
    r_adm_fg = c_anon.post("/auth/forgot", json={"email": admin_email}, headers={"X-CSRF-Token": csrf_a})
    assert r_adm_fg.status_code == 200
    assert len(mock_mail.sent) == sent_before, "Forgot password must NOT send an OTP for admin email"

    # Admin password policy requires 14+ characters
    try:
        auth.ensure_admin_account(admin_email, "Short12Char!")
        raise AssertionError("Expected ValueError for admin password < 14 chars")
    except ValueError:
        pass
    assert auth.ensure_admin_account(admin_email, admin_pw) > 0

    # Admin login with default ADMIN_REQUIRE_OTP=0: direct email+password login, ZERO OTP rows, ZERO emails sent
    os.environ["ADMIN_REQUIRE_OTP"] = "0"
    c_adm = app_mod.app.test_client()
    csrf_l = _csrf(c_adm, "/login")
    sent_pre_adm = len(mock_mail.sent)
    with auth._conn() as con:
        otps_pre_adm = con.execute("SELECT COUNT(*) FROM email_otps").fetchone()[0]
    r_direct = c_adm.post("/auth/login", json={"email": admin_email, "password": admin_pw}, headers={"X-CSRF-Token": csrf_l})
    assert r_direct.status_code == 200 and r_direct.get_json()["role"] == "admin" and not r_direct.get_json().get("otp_required")
    assert len(mock_mail.sent) == sent_pre_adm, "Admin login with ADMIN_REQUIRE_OTP=0 must send zero emails"
    with auth._conn() as con:
        assert con.execute("SELECT COUNT(*) FROM email_otps").fetchone()[0] == otps_pre_adm

    # With ADMIN_REQUIRE_OTP=1: admin step-up OTP works again
    os.environ["ADMIN_REQUIRE_OTP"] = "1"
    try:
        c_adm2 = app_mod.app.test_client()
        csrf_l2 = _csrf(c_adm2, "/login")
        r_step1 = c_adm2.post("/auth/login", json={"email": admin_email, "password": admin_pw}, headers={"X-CSRF-Token": csrf_l2})
        assert r_step1.status_code == 200 and r_step1.get_json().get("otp_required") is True
        assert r_step1.get_json().get("purpose") == "admin_login"
        assert c_adm2.get("/api/admin/overview").status_code == 401
        adm_code = mock_mail.last_code(admin_email)
        r_step2 = c_adm2.post(
            "/auth/otp/verify",
            json={"purpose": "admin_login", "code": adm_code},
            headers={"X-CSRF-Token": csrf_l2},
        )
        assert r_step2.status_code == 200 and r_step2.get_json()["role"] == "admin"
    finally:
        os.environ["ADMIN_REQUIRE_OTP"] = "0"

    # Role-change API is permanently locked (403)
    adm_csrf = _csrf(c_adm, "/admin")
    with auth._conn() as con:
        rogue_id = con.execute("SELECT id FROM users WHERE email = 'rogue_admin@example.test'").fetchone()["id"]
    assert c_adm.post(f"/api/admin/users/{rogue_id}/role", json={"role": "admin"}, headers={"X-CSRF-Token": adm_csrf}).status_code == 403

    # Admin cannot deactivate or delete self via website
    assert c_adm.post("/api/account/deactivate", json={"password": admin_pw}, headers={"X-CSRF-Token": adm_csrf}).status_code == 403
    assert c_adm.delete("/api/account", json={"password": admin_pw, "confirmation": "DELETE", "code": "123456"}, headers={"X-CSRF-Token": adm_csrf}).status_code == 403

    # Seed a job for admin so admin row also appears in history/exports, then sweep EVERY page & API for ZERO leaks!
    adm_uid = auth.get_user_by_email(admin_email)["id"]
    auth.register_user_job(uuid.uuid4().hex, adm_uid, "adm_clip.wav", {"rounded_count": 2, "top_sounds": ["Speech"]})

    for path in (
        "/",
        "/admin",
        "/history",
        "/settings/account",
        "/api/auth/me",
        "/api/me",
        "/api/admin/overview",
        "/api/admin/history",
        "/api/admin/history/export.csv",
        "/api/account/export?format=json",
    ):
        resp = c_adm.get(path)
        assert resp.status_code == 200, f"GET {path} returned {resp.status_code}"
        txt = resp.get_data(as_text=True)
        assert admin_email not in txt, f"UNMASKED ADMIN EMAIL LEAKED IN {path}!"
        assert admin_local not in txt, f"ADMIN EMAIL LOCAL PART LEAKED IN {path}!"

    me_json = c_adm.get("/api/auth/me").get_json()
    assert me_json["display_label"] == "Admin"
    assert me_json["email"] == auth.mask_email(admin_email)

    # Admin idle timeout (30 min) expires admin session
    sid_adm = _session_cookie_val(c_adm)
    with auth._conn() as con:
        con.execute("UPDATE sessions SET last_seen_at = ? WHERE id = ?", (time.time() - (auth.ADMIN_IDLE_TTL_SECONDS + 30), sid_adm))
    assert c_adm.get("/api/admin/overview").status_code == 401

    print("  [PASS] 4. Single locked admin, step-up OTP, role 403, idle TTL & ZERO email leaks verified.")


def test_5_account_deactivate_delete_and_admin_controls(mock_mail: _MockMail) -> None:
    """Mission B2: Data export, deactivate + OTP reactivate, permanent delete (DB+disk) + re-signup, admin controls + audit log."""
    _clear_rate_limits()
    admin_email = os.environ["ADMIN_EMAILS"]
    admin_pw = "AdminVaultStrong!2026"

    # Create regular user U1 (password) and Google user U2 (password-less)
    u1_email = "lifecycle_u1@example.test"
    u1_pw = "LifecycleStrong!2026"
    uid1, _, err = auth.register_password_user(u1_email, u1_pw, check_mx=False)
    assert uid1 and not err
    with auth._conn() as con:
        con.execute("UPDATE users SET email_verified = 1 WHERE id = ?", (uid1,))

    c1 = app_mod.app.test_client()
    csrf1 = _csrf(c1, "/login")
    assert c1.post("/auth/login", json={"email": u1_email, "password": u1_pw}, headers={"X-CSRF-Token": csrf1}).status_code == 200

    # Create a real job dir on disk for U1
    jid1 = uuid.uuid4().hex
    jdir1 = app_mod.JOB_DIR / jid1
    jdir1.mkdir(parents=True, exist_ok=True)
    (jdir1 / "original.wav").write_bytes(b"RIFFtest")
    auth.register_user_job(jid1, uid1, "u1_recording.wav", {"rounded_count": 3, "top_sounds": ["Speech"]})

    # Settings -> Account page + JSON & CSV data export (caller's rows only)
    acct_html = c1.get("/settings/account").get_data(as_text=True)
    assert "Download JSON" in acct_html and "Deactivate Account" in acct_html and "Delete Account Permanently" in acct_html
    exp_json = c1.get("/api/account/export?format=json").get_json()
    assert [x["job_id"] for x in exp_json["analyses"]] == [jid1]
    exp_csv = c1.get("/api/account/export?format=csv").get_data(as_text=True)
    assert "u1_recording.wav" in exp_csv

    # Deactivate U1 (reversible): revokes sessions, hides history from admin default view, sends confirmation email
    csrf1_a = _csrf(c1, "/settings/account")
    r_deact = c1.post("/api/account/deactivate", json={"password": u1_pw}, headers={"X-CSRF-Token": csrf1_a})
    assert r_deact.status_code == 200 and r_deact.get_json()["status"] == "disabled"
    assert c1.get("/api/history").status_code == 401  # session revoked immediately
    assert any("deactivated" in m["subject"].lower() for m in mock_mail.sent if m["to"] == u1_email)

    # Admin default /api/admin/history hides deactivated user's history (direct admin login, no OTP)
    c_adm = app_mod.app.test_client()
    csrf_al = _csrf(c_adm, "/login")
    r_adml = c_adm.post("/auth/login", json={"email": admin_email, "password": admin_pw}, headers={"X-CSRF-Token": csrf_al})
    assert r_adml.status_code == 200 and not r_adml.get_json().get("otp_required")
    adm_hist_ids = [x["job_id"] for x in c_adm.get("/api/admin/history").get_json()["items"]]
    assert jid1 not in adm_hist_ids, "Deactivated user's history must be hidden from default views"

    # Reactivate U1 directly via email + password sign-in (ZERO OTPs created, ZERO emails sent)
    csrf1_r = _csrf(c1, "/login")
    sent_pre_react = len(mock_mail.sent)
    with auth._conn() as con:
        otps_pre_react = con.execute("SELECT COUNT(*) FROM email_otps").fetchone()[0]
    r_re1 = c1.post("/auth/login", json={"email": u1_email, "password": u1_pw}, headers={"X-CSRF-Token": csrf1_r})
    assert r_re1.status_code == 200 and not r_re1.get_json().get("otp_required")
    assert len(mock_mail.sent) == sent_pre_react
    with auth._conn() as con:
        assert con.execute("SELECT COUNT(*) FROM email_otps").fetchone()[0] == otps_pre_react
    assert [x["job_id"] for x in c1.get("/api/history").get_json()["items"]] == [jid1]

    # Permanent delete of U1 (no OTP): rejects missing DELETE, rejects wrong/missing password, succeeds with password + DELETE
    csrf1_d = _csrf(c1, "/settings/account")
    assert c1.delete("/api/account", json={"password": u1_pw, "confirmation": "delete"}, headers={"X-CSRF-Token": csrf1_d}).status_code == 400
    assert c1.delete("/api/account", json={"confirmation": "DELETE"}, headers={"X-CSRF-Token": csrf1_d}).status_code == 401
    assert c1.delete("/api/account", json={"password": "WrongPassword!999", "confirmation": "DELETE"}, headers={"X-CSRF-Token": csrf1_d}).status_code == 401

    r_del = c1.delete(
        "/api/account",
        json={"password": u1_pw, "confirmation": "DELETE"},
        headers={"X-CSRF-Token": csrf1_d},
    )
    assert r_del.status_code == 200 and r_del.get_json()["files_verified_removed"] is True
    assert not jdir1.exists(), "Uploaded audio folder must be deleted on disk"
    assert c1.get("/api/history").status_code == 401
    with auth._conn() as con:
        assert con.execute("SELECT COUNT(*) FROM users WHERE id = ?", (uid1,)).fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM user_jobs WHERE user_id = ?", (uid1,)).fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM sessions WHERE user_id = ?", (uid1,)).fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM email_otps WHERE email = ?", (u1_email,)).fetchone()[0] == 0
    assert any("permanently deleted" in m["subject"].lower() for m in mock_mail.sent if m["to"] == u1_email)

    # Same email can register fresh again after deletion
    c_new = app_mod.app.test_client()
    csrf_n = _csrf(c_new, "/login")
    assert c_new.post("/auth/register", json={"email": u1_email, "password": "FreshAccountPass!2026", "password_confirm": "FreshAccountPass!2026"}, headers={"X-CSRF-Token": csrf_n}).status_code == 201

    # Password-less Google user: stale re-auth rejected (401); fresh Google re-auth allows deactivate & delete (no OTP)
    _clear_rate_limits()
    u_g, _ = auth.register_oauth_user("google", "g-sub-deact", "gdeact@example.test", True)
    cg = app_mod.app.test_client()
    sid_g, _ = auth.create_session(u_g["id"])
    cg.set_cookie(auth.SESSION_COOKIE, sid_g)
    csrf_g = _csrf(cg, "/settings/account")
    with auth._conn() as con:
        con.execute("UPDATE users SET reauth_at = ? WHERE id = ?", (time.time() - 600, u_g["id"]))
    assert cg.post("/api/account/deactivate", json={}, headers={"X-CSRF-Token": csrf_g}).status_code == 401
    assert cg.delete("/api/account", json={"confirmation": "DELETE"}, headers={"X-CSRF-Token": csrf_g}).status_code == 401
    auth.mark_reauthenticated(u_g["id"])
    assert cg.delete("/api/account", json={"confirmation": "DELETE"}, headers={"X-CSRF-Token": csrf_g}).status_code == 200
    assert cg.get("/api/history").status_code == 401

    # Admin controls: disable, enable, permanently delete non-admin + refuse admin + ID-only audit log
    uid_t, _, _ = auth.register_password_user("target_nonadmin@example.test", "TargetPass!2026", check_mx=False)
    jid_t = uuid.uuid4().hex
    jdir_t = app_mod.JOB_DIR / jid_t
    jdir_t.mkdir(parents=True, exist_ok=True)
    (jdir_t / "stem_01.wav").write_bytes(b"RIFFstem")
    auth.register_user_job(jid_t, uid_t, "target.wav", {"rounded_count": 2})

    adm_csrf = _csrf(c_adm, "/admin")
    adm_uid = auth.get_user_by_email(admin_email)["id"]
    assert c_adm.post(f"/api/admin/users/{adm_uid}/status", json={"is_active": False}, headers={"X-CSRF-Token": adm_csrf}).status_code == 400
    assert c_adm.delete(f"/api/admin/users/{adm_uid}", headers={"X-CSRF-Token": adm_csrf}).status_code == 400

    assert c_adm.post(f"/api/admin/users/{uid_t}/status", json={"is_active": False}, headers={"X-CSRF-Token": adm_csrf}).status_code == 200
    assert c_adm.post(f"/api/admin/users/{uid_t}/status", json={"is_active": True}, headers={"X-CSRF-Token": adm_csrf}).status_code == 200
    r_admdel = c_adm.delete(f"/api/admin/users/{uid_t}", headers={"X-CSRF-Token": adm_csrf})
    assert r_admdel.status_code == 200 and not jdir_t.exists()

    # Audit log stores ONLY IDs and actions (no email column, no email string anywhere)
    ov = c_adm.get("/api/admin/overview").get_json()
    audit_rows = ov["audit_log"]
    actions = [r["action"] for r in audit_rows if r["target_id"] == uid_t]
    assert "disable_user" in actions and "enable_user" in actions and "delete_user" in actions
    audit_dump = json.dumps(audit_rows)
    assert "target_nonadmin@example.test" not in audit_dump and admin_email not in audit_dump

    print("  [PASS] 5. Account export, deactivate/reactivate, permanent delete (DB+disk), re-signup & admin audit log.")


def test_6_part_a_compact_analyze_and_gt_gating() -> None:
    """Part A: Pair ranking/capping, payload size <150KB for K=31, GROUND_TRUTH_PUBLIC gating & tabbed DOM."""
    # Verify _rank_and_cap_pairs across K = 2, 3, 9, 20, 31 synthetic detections (up to 33 sounds -> 528 pairs)
    for k, n_snd in ((2, 2), (3, 3), (9, 30), (20, 31), (31, 33)):
        sounds = [
            {"name": f"Sound {i}", "group": ["Human voice", "Music", "Household", "Transport", "Environment/other"][i % 5], "confidence": round(0.95 - i * 0.015, 4), "time_ranges": [{"start": 0.0, "end": 4.0}]}
            for i in range(n_snd)
        ]
        raw_pairs = [
            {"sounds": [sounds[i]["name"], sounds[j]["name"]], "intervals": [{"start": 0.0, "end": round(1.0 + ((i + j) % 7), 1)}], "total_overlap_seconds": round(1.0 + ((i + j) % 7), 1)}
            for i in range(n_snd) for j in range(i + 1, n_snd)
        ]
        top, all_r, tot = app_mod._rank_and_cap_pairs(sounds, raw_pairs, limit=12)
        expected_tot = n_snd * (n_snd - 1) // 2
        assert tot == expected_tot and len(all_r) == expected_tot
        assert len(top) == min(12, expected_tot)
        # Verify descending order by overlap_duration * confidence
        scores = [p["score"] for p in top]
        assert scores == sorted(scores, reverse=True)
        payload_bytes = len(json.dumps({"named_sounds": sounds, "visible_default": 8, "overlapping_sounds": top, "overlapping_pairs_total": tot}).encode("utf-8"))
        assert payload_bytes < 150 * 1024, f"K={k} payload exceeded 150 KB: {payload_bytes}"

    # Verify GROUND_TRUTH_PUBLIC gating helper
    prev_gtp = os.environ.get("GROUND_TRUTH_PUBLIC")
    try:
        os.environ["GROUND_TRUTH_PUBLIC"] = "0"
        assert app_mod._can_see_ground_truth({"role": "admin"}) is True
        assert app_mod._can_see_ground_truth({"role": "user"}) is False
        os.environ["GROUND_TRUTH_PUBLIC"] = "1"
        assert app_mod._can_see_ground_truth({"role": "user"}) is True
    finally:
        if prev_gtp is None:
            os.environ.pop("GROUND_TRUTH_PUBLIC", None)
        else:
            os.environ["GROUND_TRUTH_PUBLIC"] = prev_gtp

    # Verify /api/jobs/<id>/pairs enforces owner/admin scope (404 for non-owner)
    u_own, _, _ = auth.register_password_user("pairs_owner@example.test", "StrongPass!2026", check_mx=False)
    u_oth, _, _ = auth.register_password_user("pairs_other@example.test", "StrongPass!2026", check_mx=False)
    with auth._conn() as con:
        con.execute("UPDATE users SET email_verified = 1 WHERE id IN (?, ?)", (u_own, u_oth))
    jid = uuid.uuid4().hex
    auth.register_user_job(jid, u_own, "clip.wav", {"rounded_count": 3})
    with app_mod._jobs_lock:
        app_mod._jobs[jid] = {"created": time.time(), "all_overlapping_sounds": [{"sounds": ["A", "B"], "total_overlap_seconds": 2.0}]}
    c_own = app_mod.app.test_client(); sid_o, _ = auth.create_session(u_own); c_own.set_cookie(auth.SESSION_COOKIE, sid_o)
    c_oth = app_mod.app.test_client(); sid_x, _ = auth.create_session(u_oth); c_oth.set_cookie(auth.SESSION_COOKIE, sid_x)
    r_o = c_own.get(f"/api/jobs/{jid}/pairs")
    assert r_o.status_code == 200 and r_o.get_json()["total"] == 1
    assert "no-store" in r_o.headers.get("Cache-Control", "").lower()
    assert c_oth.get(f"/api/jobs/{jid}/pairs").status_code == 404

    # Verify Analyze page template & JS contain sticky upload col, tabs, search/sort/pagination
    an_html = c_own.get("/analyze").get_data(as_text=True)
    assert "an-upload-col" in an_html and "an-result-col" in an_html
    an_js = (ROOT / "static" / "analyze.js").read_text(encoding="utf-8")
    for token in ('role="tablist"', 'id="pane-overview"', 'id="pane-sounds"', 'id="pane-numbers"', 'id="pane-reference"', 'id="pane-stems"', "sndSearch", "sndSort", "showAllPairsBtn"):
        assert token in an_js, f"Missing {token} in analyze.js"
    print("  [PASS] 6. Part A pair capping, <150KB payload, GROUND_TRUTH_PUBLIC gating & tabbed Analyze UI.")


def main() -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="ec_auth_test_"))
    orig_db = auth.DB_PATH
    orig_env = {
        k: os.environ.get(k)
        for k in (
            "ADMIN_EMAILS",
            "ADMIN_REQUIRE_OTP",
            "ECHOCOUNT_SKIP_MX_CHECK",
            "ECHOCOUNT_NO_AUTH_DELAY",
            "GOOGLE_CLIENT_ID",
            "GOOGLE_CLIENT_SECRET",
            "AUTH_GOOGLE_MODE",
            "OTP_DAILY_BUDGET",
        )
    }
    mock_mail = _MockMail()
    try:
        auth.DB_PATH = tmp_dir / "test_auth.db"
        os.environ["ECHOCOUNT_SKIP_MX_CHECK"] = "1"
        os.environ["ECHOCOUNT_NO_AUTH_DELAY"] = "1"
        os.environ["OTP_DAILY_BUDGET"] = "250"
        os.environ["ADMIN_EMAILS"] = "secret_adm_99@example.test"
        mail_mod.set_provider(mock_mail)
        auth.init_db()

        print("Running EchoCount Auth Overhaul & Single Locked Admin test suite...")
        test_1_legacy_oauth_removed_and_migration(mock_mail)
        test_2_email_otp_security_and_hardening(mock_mail)
        test_3_google_authlib_and_firebase(mock_mail)
        test_4_single_locked_admin_and_zero_leak(mock_mail)
        test_5_account_deactivate_delete_and_admin_controls(mock_mail)
        test_6_part_a_compact_analyze_and_gt_gating()
        print("ALL AUTH OVERHAUL, PART A & PART B TESTS PASSED (6/6 groups).")
    finally:
        mail_mod.set_provider(None)
        auth.DB_PATH = orig_db
        for k, v in orig_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        auth.init_oauth(app_mod.app)
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
