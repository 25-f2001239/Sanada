"""Transactional email sender (Brevo REST API, SMTP fallback, dev-only console) with daily quota."""

from __future__ import annotations

import json
import logging
import os
import smtplib
import sqlite3
import urllib.error
import urllib.request
from datetime import UTC, datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Protocol

from config import DATA_DIR

logger = logging.getLogger("echocount.mail")
DB_PATH = Path(os.getenv("ECHOCOUNT_AUTH_DB", str(DATA_DIR / "echocount.db")))


def mask_email(email: str) -> str:
    """Return a privacy-safe masked address (e.g. a***@e***.com)."""
    e = (email or "").strip().lower()
    if "@" not in e:
        return "***"
    local, dom = e.split("@", 1)
    parts = dom.split(".")
    m_local = (local[:1] + "***") if local else "***"
    m_dom = f"{parts[0][:1]}***.{'.'.join(parts[1:])}" if len(parts) > 1 else "***"
    return f"{m_local}@{m_dom}"


def is_dev_mode() -> bool:
    return os.getenv("ECHOCOUNT_DEV", "0").strip() == "1"


def daily_budget_limit() -> int:
    try:
        return max(1, int(os.getenv("OTP_DAILY_BUDGET", "250")))
    except ValueError:
        return 250


def _consume_daily_budget(db_path: Path | None = None) -> tuple[bool, int]:
    """Atomically check and increment today's global send count. Returns (allowed, new_count)."""
    limit = daily_budget_limit()
    day = datetime.now(UTC).strftime("%Y-%m-%d")
    target = db_path or DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(target, timeout=10.0) as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS otp_daily_budget (day TEXT PRIMARY KEY, count INTEGER NOT NULL DEFAULT 0)"
        )
        row = con.execute("SELECT count FROM otp_daily_budget WHERE day = ?", (day,)).fetchone()
        cur = int(row[0]) if row else 0
        if cur >= limit:
            return False, cur
        con.execute(
            "INSERT INTO otp_daily_budget (day, count) VALUES (?, 1) "
            "ON CONFLICT(day) DO UPDATE SET count = count + 1",
            (day,),
        )
        return True, cur + 1


class MailProvider(Protocol):
    name: str

    def send(self, to_email: str, subject: str, body: str) -> tuple[bool, str | None]: ...


class BrevoProvider:
    name = "brevo"

    def __init__(self, api_key: str, sender_email: str, sender_name: str = "EchoCount") -> None:
        self.api_key = api_key
        self.sender_email = sender_email
        self.sender_name = sender_name

    def send(self, to_email: str, subject: str, body: str) -> tuple[bool, str | None]:
        payload = json.dumps(
            {
                "sender": {"name": self.sender_name, "email": self.sender_email},
                "to": [{"email": to_email}],
                "subject": subject,
                "textContent": body,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            "https://api.brevo.com/v3/smtp/email",
            data=payload,
            headers={
                "api-key": self.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if 200 <= resp.status < 300:
                    return True, None
                return False, f"Provider returned HTTP {resp.status}."
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("Brevo send failed for %s: %s", mask_email(to_email), type(exc).__name__)
            return False, "Email delivery failed. Please try again shortly."


class SmtpProvider:
    name = "smtp"

    def __init__(self, host: str, port: int, user: str, password: str, sender: str, use_tls: bool) -> None:
        self.host, self.port, self.user, self.password, self.sender, self.use_tls = (
            host,
            port,
            user,
            password,
            sender,
            use_tls,
        )

    def send(self, to_email: str, subject: str, body: str) -> tuple[bool, str | None]:
        msg = EmailMessage()
        msg["Subject"], msg["From"], msg["To"] = subject, self.sender, to_email
        msg.set_content(body)
        try:
            with smtplib.SMTP(self.host, self.port, timeout=10) as srv:
                if self.use_tls:
                    srv.starttls()
                if self.user and self.password:
                    srv.login(self.user, self.password)
                srv.send_message(msg)
            return True, None
        except Exception as exc:
            logger.warning("SMTP send failed for %s: %s", mask_email(to_email), type(exc).__name__)
            return False, "Email delivery failed. Please try again shortly."


class DevConsoleProvider:
    name = "dev_console"

    def send(self, to_email: str, subject: str, body: str) -> tuple[bool, str | None]:
        # Development mode ONLY (ECHOCOUNT_DEV=1): never prints raw email; logs code to local console
        print(f"[EchoCount DEV MAIL] to={mask_email(to_email)} | subject={subject}\n{body}\n")
        return True, None


_CUSTOM_PROVIDER: MailProvider | None = None


def set_provider(provider: MailProvider | None) -> None:
    """Override mail provider (used in automated tests or custom deployments)."""
    global _CUSTOM_PROVIDER
    _CUSTOM_PROVIDER = provider


def get_provider() -> MailProvider | None:
    if _CUSTOM_PROVIDER is not None:
        return _CUSTOM_PROVIDER
    brevo_key = os.getenv("BREVO_API_KEY", "").strip()
    brevo_sender = os.getenv("BREVO_SENDER_EMAIL", "").strip()
    if brevo_key and brevo_sender:
        return BrevoProvider(brevo_key, brevo_sender, os.getenv("BREVO_SENDER_NAME", "EchoCount"))
    smtp_host = os.getenv("SMTP_HOST", "").strip()
    if smtp_host:
        return SmtpProvider(
            smtp_host,
            int(os.getenv("SMTP_PORT", "587")),
            os.getenv("SMTP_USER", "").strip(),
            os.getenv("SMTP_PASSWORD", ""),
            os.getenv("SMTP_FROM", "").strip()
            or os.getenv("SMTP_SENDER_EMAIL", "").strip()
            or os.getenv("SMTP_USER", "").strip()
            or "no-reply@echocount.local",
            os.getenv("SMTP_TLS", "1") != "0",
        )
    if is_dev_mode():
        return DevConsoleProvider()
    return None


def send_transactional_email(
    to_email: str, subject: str, body: str, *, db_path: Path | None = None
) -> tuple[bool, str | None]:
    """Enforce daily quota and dispatch email through the active provider (never logs secrets)."""
    provider = get_provider()
    if provider is None:
        return False, "Email verification is not available right now. Please try again later."
    allowed, count = _consume_daily_budget(db_path=db_path)
    if not allowed:
        logger.warning("Daily email send budget reached (count=%d).", count)
        return False, "Daily email verification quota reached. Please try again later."
    ok, err = provider.send(to_email, subject, body)
    if ok:
        logger.info("Sent email via %s (daily_count=%d, to=%s).", getattr(provider, "name", "custom"), count, mask_email(to_email))
    return ok, err


def send_otp_email(to_email: str, code: str, purpose: str, *, db_path: Path | None = None) -> tuple[bool, str | None]:
    labels = {
        "signup": "Verify your EchoCount account",
        "login_verify": "Verify your EchoCount email",
        "admin_login": "EchoCount sign-in verification code",
        "forgot": "Reset your EchoCount password",
        "reactivate": "Reactivate your EchoCount account",
        "deactivate": "Confirm EchoCount account deactivation",
        "delete_account": "Confirm permanent EchoCount account deletion",
    }
    subject = labels.get(purpose, "Your EchoCount verification code")
    body = (
        f"Your EchoCount verification code is: {code}\n\n"
        "This code expires in 10 minutes and can only be used once.\n"
        "If you did not request this, you can safely ignore this email."
    )
    return send_transactional_email(to_email, subject, body, db_path=db_path)
