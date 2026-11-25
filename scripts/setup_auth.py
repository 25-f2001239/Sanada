"""Interactive local .env setup script using getpass (never prints raw secrets)."""

from __future__ import annotations

import getpass
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"

FIELDS = [
    ("ADMIN_EMAILS", "Single administrator email (exactly ONE email)", False),
    (
        "BASE_URL",
        "Public base URL (e.g. http://127.0.0.1:5000 or https://your.ngrok-free.app)",
        False,
    ),
    ("BREVO_API_KEY", "Brevo transactional API key (hidden input)", True),
    ("BREVO_SENDER_EMAIL", "Verified Brevo sender email", False),
    ("GOOGLE_CLIENT_ID", "Google OAuth Client ID (leave blank to skip)", False),
    (
        "GOOGLE_CLIENT_SECRET",
        "Google OAuth Client Secret (hidden input, leave blank to skip)",
        True,
    ),
    ("AUTH_GOOGLE_MODE", "Google auth mode ('authlib' or 'firebase', default authlib)", False),
]


def _mask(val: str) -> str:
    if not val:
        return "(unset)"
    if "@" in val:
        u, d = val.split("@", 1)
        return f"{u[:1]}***@{d[:1]}***"
    return f"{val[:3]}***{val[-2:]}" if len(val) > 6 else "***"


def _load_existing(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main() -> None:
    print(
        "EchoCount Local Auth Setup -> writes git-ignored .env (press Enter to keep existing value)\n"
    )
    env = _load_existing(ENV_PATH)
    if not env.get("SECRET_KEY") or "replace-with" in env.get("SECRET_KEY", "").lower():
        env["SECRET_KEY"] = secrets.token_hex(32)
        print("  SECRET_KEY: generated fresh 64-char random hex key.")

    for key, prompt, secret in FIELDS:
        cur = env.get(key, "")
        label = f"{prompt} [{_mask(cur)}]: "
        val = (getpass.getpass(label) if secret else input(label)).strip()
        if val:
            env[key] = val
        elif key == "AUTH_GOOGLE_MODE" and not cur:
            env[key] = "authlib"

    lines = ["# EchoCount local configuration (git-ignored; never commit)"]
    for k, v in env.items():
        lines.append(f"{k}={v}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nSaved .env (masked summary):")
    for k in (
        "SECRET_KEY",
        "ADMIN_EMAILS",
        "BASE_URL",
        "BREVO_API_KEY",
        "BREVO_SENDER_EMAIL",
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET",
        "AUTH_GOOGLE_MODE",
    ):
        print(f"  {k:<22} = {_mask(env.get(k, ''))}")


if __name__ == "__main__":
    main()
