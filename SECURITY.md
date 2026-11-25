# Security Policy

## Supported Versions

Security fixes are maintained on the `main` branch of EchoCount (`25-f2001239/Sanada`).

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for suspected security vulnerabilities, credential leaks, or privacy bypasses.

Instead, report vulnerabilities privately through GitHub Security Advisories for this repository (`Security → Advisories → Report a vulnerability`). Include:

1. A clear description of the issue and affected route or file.
2. Minimal reproduction steps (without real user data or secrets).
3. Impact assessment and any suggested mitigation.

## Security Architecture & Controls

EchoCount enforces the following controls by default:

- **Secrets & Environment**: All credentials (`SECRET_KEY`, `ADMIN_EMAILS`, `BREVO_API_KEY`, SMTP, Google OAuth) live strictly in git-ignored `.env`, provisioned locally via `python scripts/setup_auth.py` (hidden input). In production (`ECHOCOUNT_ENV=production`), startup refuses to run with the default or short (`< 32` chars) `SECRET_KEY`.
- **Passwords & Sessions**: Passwords are hashed with Argon2id (`argon2-cffi`), rejected against `common_passwords.txt`, and sessions use `HttpOnly`, `SameSite=Lax`, `Secure` (on HTTPS) cookies with rotation on login and idle/absolute TTLs.
- **Email OTP**: 6-digit codes generated with `secrets`, stored only as `HMAC-SHA256(secret, salt + code)`, 10-minute expiry, single use, 5-attempt invalidation, 60 s resend cooldown, per-email/per-IP rate limits, and a daily global budget.
- **Single Locked Administrator**: Exactly one admin address is allowed via `ADMIN_EMAILS` in `.env`. Role-change APIs always return `403`, admin emails are masked across UI/API/CSV/logs, and Google OAuth can never access the admin account.
- **Defense-in-Depth**: CSRF tokens on state-changing routes, strict Same-Origin `Origin` header validation on mutations, CSP (`frame-ancestors 'none'`), `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, CSV formula neutralization, open-redirect blocking, path-traversal guards, and automatic 30-minute TTL cleanup for uploaded audio/stems.
- **Per-User History Privacy**: Every analysis job is bound to `user_id`; cross-user access returns `404`, and `Cache-Control: no-store` prevents shared-browser leakage.
