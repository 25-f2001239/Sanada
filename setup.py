#!/usr/bin/env python3
"""One-command setup script for EchoCount (Sanada).

After cloning the repository on any machine (Windows, macOS, or Linux), run:

    python setup.py

What this does automatically:
  1. Verifies Python >= 3.10.
  2. Creates a local virtual environment (`.venv`) unless already inside one
     or `--no-venv` is passed.
  3. Installs all dependencies from `requirements.txt`.
  4. Creates `.env` from `.env.example` (if missing) with a fresh random `SECRET_KEY`.
  5. Ensures runtime folders exist (`models/`, `checkpoints/`, `data/`, `jobs/`).
  6. Downloads all required pretrained HuggingFace models (AST, CLAP, ECAPA-TDNN,
     and 2-, 3-, and 4-speaker SepFormer checkpoints) and verifies trained
     checkpoint hashes via `scripts/download_models.py`.
  7. Initializes the SQLite auth database and runs a startup self-check.

Flags:
  --run        Start the EchoCount web server immediately after setup finishes.
  --no-venv    Install into the current Python interpreter instead of `.venv`.
  --skip-deps  Skip `pip install -r requirements.txt` (useful if already installed).
  --check      Verify an existing installation without reinstalling packages.
"""

from __future__ import annotations

import argparse
import os
import secrets
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQ_FILE = ROOT / "requirements.txt"
ENV_EXAMPLE = ROOT / ".env.example"
ENV_FILE = ROOT / ".env"


def _banner(step: str) -> None:
    print(f"\n==> {step}", flush=True)


def _in_venv() -> bool:
    return (
        hasattr(sys, "real_prefix")
        or (getattr(sys, "base_prefix", sys.prefix) != sys.prefix)
        or bool(os.environ.get("VIRTUAL_ENV"))
    )


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _run(cmd: list[str], *, check: bool = True) -> int:
    printable = " ".join(str(c) for c in cmd)
    print(f"    $ {printable}", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if check and proc.returncode != 0:
        raise SystemExit(
            f"\n[ERROR] Command failed with exit code {proc.returncode}: {printable}"
        )
    return proc.returncode


def ensure_python_version() -> None:
    _banner("Checking Python version")
    v = sys.version_info
    print(f"    Detected Python {v.major}.{v.minor}.{v.micro} ({sys.executable})")
    if (v.major, v.minor) < (3, 10):
        raise SystemExit(
            "[ERROR] Python 3.10 or newer is required to run EchoCount."
        )


def resolve_python(no_venv: bool) -> Path:
    if no_venv or _in_venv():
        py = Path(sys.executable)
        _banner(f"Using active Python interpreter: {py}")
        return py

    py = _venv_python()
    if py.is_file():
        _banner(f"Reusing existing virtual environment: {VENV_DIR}")
        return py

    _banner(f"Creating virtual environment at {VENV_DIR}")
    venv.create(str(VENV_DIR), with_pip=True)
    if not py.is_file():
        raise SystemExit(f"[ERROR] Virtual environment created but {py} was not found.")
    return py


def install_dependencies(py: Path) -> None:
    _banner("Installing Python dependencies from requirements.txt")
    if not REQ_FILE.is_file():
        raise SystemExit(f"[ERROR] Missing {REQ_FILE}")
    # Upgrade pip best-effort (never fail setup if pip self-upgrade is blocked)
    _run([str(py), "-m", "pip", "install", "--upgrade", "pip"], check=False)
    _run([str(py), "-m", "pip", "install", "-r", str(REQ_FILE)])


def ensure_env_file() -> None:
    _banner("Configuring environment (.env)")
    if ENV_FILE.is_file():
        print(f"    [OK] {ENV_FILE.name} already exists (leaving untouched).")
        return
    if not ENV_EXAMPLE.is_file():
        print("    [WARN] .env.example not found; skipping .env creation.")
        return

    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    placeholder = "SECRET_KEY=replace-with-a-long-random-secret-key"
    if placeholder in text:
        text = text.replace(placeholder, f"SECRET_KEY={secrets.token_hex(32)}", 1)
    ENV_FILE.write_text(text, encoding="utf-8")
    print("    [OK] Created .env from .env.example with a newly generated SECRET_KEY.")


def ensure_directories() -> None:
    _banner("Ensuring runtime directories exist")
    for rel in ("models", "checkpoints", "data", "jobs"):
        d = ROOT / rel
        d.mkdir(parents=True, exist_ok=True)
    print("    [OK] Verified models/, checkpoints/, data/, jobs/.")


def download_and_verify_models(py: Path, check_only: bool) -> None:
    dl_script = ROOT / "scripts" / "download_models.py"
    if not dl_script.is_file():
        raise SystemExit(f"[ERROR] Missing {dl_script}")

    if not check_only:
        _banner("Downloading pretrained HuggingFace models (AST, CLAP, ECAPA, SepFormer 2/3/4-spk)")
        _run([str(py), str(dl_script)])

    _banner("Verifying all pretrained models and trained checkpoints")
    _run([str(py), str(dl_script), "--check"])


def verify_app_boot(py: Path) -> None:
    _banner("Initializing auth database and running startup self-check")
    check_code = (
        "import auth, separation, app; "
        "auth.init_db(); "
        "sep = separation.separator_status(); "
        "sp = separation.pretrained_speech_status(); "
        "assert sep.get('available'), f'Separator unavailable: {sep}'; "
        "assert sp.get('has_2spk_model') and sp.get('has_3spk_model') and sp.get('has_4spk_model'), "
        "f'Missing speech separation models: {sp}'; "
        "assert sp.get('max_supported_speakers') == 4, f'Expected 4-speaker support: {sp}'; "
        "c = app.app.test_client(); "
        "r = c.get('/api/health'); "
        "assert r.status_code == 200, f'/api/health returned {r.status_code}'; "
        "print('    [OK] Auth DB initialized, /api/health=200, speech separation ready up to 4 speakers.')"
    )
    _run([str(py), "-c", check_code])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set up EchoCount (venv, dependencies, .env, all pretrained models, and self-check)."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Start the EchoCount web app immediately after setup completes.",
    )
    parser.add_argument(
        "--no-venv",
        action="store_true",
        help="Use the current Python interpreter instead of creating/using .venv.",
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip pip install -r requirements.txt.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify models, checkpoints, and app startup without reinstalling or downloading.",
    )
    args = parser.parse_args()

    print("=== EchoCount (Sanada) One-Command Setup ===", flush=True)
    ensure_python_version()
    py = resolve_python(no_venv=args.no_venv)
    ensure_directories()
    ensure_env_file()

    if not args.check and not args.skip_deps:
        install_dependencies(py)

    download_and_verify_models(py, check_only=args.check)
    verify_app_boot(py)

    try:
        rel_py = py.relative_to(ROOT)
        launch_cmd = f".\\{rel_py} app.py" if os.name == "nt" else f"./{rel_py} app.py"
    except ValueError:
        launch_cmd = f"{py} app.py"

    print("\n============================================================")
    print(" EchoCount setup complete! Everything is verified and ready.")
    print(f" Start the server anytime with:\n   {launch_cmd}")
    print(" Then open: http://127.0.0.1:5000")
    print("============================================================\n", flush=True)

    if args.run:
        _banner("Starting EchoCount web server")
        _run([str(py), str(ROOT / "app.py")])


if __name__ == "__main__":
    main()
