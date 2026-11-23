"""Download pretrained HuggingFace models and hosted trained checkpoints for EchoCount."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "checkpoints_manifest.json"
MODELS_DIR = ROOT / "models"
CHECKPOINTS_DIR = ROOT / "checkpoints"

PRETRAINED_MODELS: Dict[str, Dict[str, str]] = {
    "ast": {
        "repo_id": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "revision": "f826b80d28226b62986cc218e5cec390b1096902",
        "local_dir": "models/ast-audioset",
        "size": "~346 MB",
        "license": "BSD-3-Clause (commercial use allowed)",
        "marker": "model.safetensors",
    },
    "clap": {
        "repo_id": "laion/clap-htsat-unfused",
        "revision": "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a",
        "local_dir": "models/clap-htsat-unfused",
        "size": "~615 MB",
        "license": "Apache-2.0 / MIT (commercial use allowed)",
        "marker": "pytorch_model.bin",
    },
    "sepformer": {
        "repo_id": "speechbrain/sepformer-wsj02mix",
        "revision": "3a2826343a10e2d2e8a75f79aeab5ff3a2473531",
        "local_dir": "models/sepformer-wsj02mix",
        "size": "~113 MB",
        "license": "Apache-2.0 (commercial use allowed)",
        "marker": "masknet.ckpt",
    },
    "ecapa": {
        "repo_id": "speechbrain/spkrec-ecapa-voxceleb",
        "revision": "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
        "local_dir": "models/spkrec-ecapa-voxceleb",
        "size": "~83 MB",
        "license": "Apache-2.0 (commercial use allowed)",
        "marker": "embedding_model.ckpt",
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _print_inventory(only: str | None = None) -> None:
    print("=== Pretrained Models (HuggingFace Hub) ===")
    for key, spec in PRETRAINED_MODELS.items():
        if only and only not in (key, "pretrained", "all"):
            continue
        print(f"  - {key:<10} | {spec['repo_id']:<42} | {spec['size']:<9} | {spec['license']}")
    if MANIFEST_PATH.is_file():
        ckpts = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")).get("checkpoints", {})
        print("\n=== Trained Checkpoints (checkpoints_manifest.json) ===")
        for name, info in ckpts.items():
            if only and only not in (name, "checkpoints", "all"):
                continue
            print(f"  - {name:<26} | {info['size_mb']:>5.2f} MB | sha256={info['sha256'][:12]}... | MIT")


def check_all(only: str | None = None) -> bool:
    """Verify all expected models and checkpoints exist and match manifest SHA-256 hashes."""
    ok = True
    for key, spec in PRETRAINED_MODELS.items():
        if only and only not in (key, "pretrained", "all"):
            continue
        marker = ROOT / spec["local_dir"] / spec["marker"]
        exists = marker.is_file()
        ok = ok and exists
        print(f"[{'OK' if exists else 'MISSING'}] {key:<10} -> {marker.relative_to(ROOT)}")

    if MANIFEST_PATH.is_file():
        ckpts = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")).get("checkpoints", {})
        for name, info in ckpts.items():
            if only and only not in (name, "checkpoints", "all"):
                continue
            p = ROOT / info["path"]
            if not p.is_file():
                print(f"[MISSING] {name} -> {info['path']}")
                ok = False
                continue
            actual = _sha256(p)
            match = actual == info["sha256"]
            ok = ok and match
            print(f"[{'OK' if match else 'HASH MISMATCH'}] {name} ({info['size_mb']} MB)")
    if not ok:
        print("\nMissing files detected. Run `python scripts/download_models.py` to fetch pretrained models and release checkpoints.")
        print("If the GitHub Release is not published yet, train locally with `python scripts/train.py all --full`.")
    return ok


def download_all(only: str | None = None, base_url: str | None = None) -> list[str]:
    """Download pinned HuggingFace base models and release checkpoints with SHA-256 verification."""
    _print_inventory(only)
    print()
    failures: list[str] = []
    for key, spec in PRETRAINED_MODELS.items():
        if only and only not in (key, "pretrained", "all"):
            continue
        target_dir = ROOT / spec["local_dir"]
        if (target_dir / spec["marker"]).is_file():
            print(f"[SKIP] {key} already present at {spec['local_dir']}")
            continue
        try:
            from huggingface_hub import snapshot_download

            print(
                f"[DOWNLOAD] {spec['repo_id']}@{spec['revision'][:8]} ({spec['size']}) -> {spec['local_dir']}"
            )
            ignore = (
                ["*.bin", "*.md", ".gitattributes"] if key == "ast" else ["*.md", ".gitattributes"]
            )
            snapshot_download(
                spec["repo_id"],
                revision=spec["revision"],
                local_dir=str(target_dir),
                ignore_patterns=ignore,
            )
        except Exception as exc:
            failures.append(f"{key} ({spec['repo_id']}): {exc}")
            print(f"[WARN] Could not download {key} ({exc}). App will still start.", file=sys.stderr)

    if MANIFEST_PATH.is_file():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        url_root = (
            base_url or os.getenv("ECHOCOUNT_CHECKPOINT_URL") or manifest.get("default_release_base_url", "")
        ).rstrip("/")
        for name, info in manifest.get("checkpoints", {}).items():
            if only and only not in (name, "checkpoints", "all"):
                continue
            dest = ROOT / info["path"]
            if dest.is_file() and _sha256(dest) == info["sha256"]:
                print(f"[SKIP] {name} already verified.")
                continue
            url = f"{url_root}/{name}" if url_root else info.get("release_asset_url", "")
            print(f"[DOWNLOAD] {url} ({info['size_mb']} MB) -> {info['path']}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                urllib.request.urlretrieve(url, dest)
                actual = _sha256(dest)
                if actual != info["sha256"]:
                    dest.unlink(missing_ok=True)
                    msg = f"{name}: SHA-256 mismatch (got {actual[:12]}..., expected {info['sha256'][:12]}...)"
                    failures.append(msg)
                    print(f"[WARN] {msg}; removed corrupted file.", file=sys.stderr)
                else:
                    print(f"[OK] {name} verified (sha256={actual[:12]}...).")
            except Exception as exc:
                failures.append(f"{name}: {exc}")
                print(
                    f"[WARN] Could not download {name} from {url} ({exc}). App will still start.",
                    file=sys.stderr,
                )
    return failures


def ensure_assets() -> list[str]:
    """Check local models & checkpoints; automatically download any missing items on first run."""
    missing = False
    for spec in PRETRAINED_MODELS.values():
        if not (ROOT / spec["local_dir"] / spec["marker"]).is_file():
            missing = True
            break
    if not missing and MANIFEST_PATH.is_file():
        for info in json.loads(MANIFEST_PATH.read_text(encoding="utf-8")).get("checkpoints", {}).values():
            p = ROOT / info["path"]
            if not p.is_file():
                missing = True
                break
    if not missing:
        return []
    print("[EchoCount] First-run setup: downloading missing models/checkpoints (~1.17 GB total)...")
    return download_all()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="Verify local models and checkpoint hashes")
    p.add_argument("--only", type=str, default=None, help="Download/check one item (ast, clap, sepformer, ecapa, or checkpoint filename)")
    p.add_argument("--base-url", type=str, default=None, help="Override base URL for hosted checkpoints")
    args = p.parse_args()
    if args.check:
        sys.exit(0 if check_all(args.only) else 1)
    download_all(only=args.only, base_url=args.base_url)


if __name__ == "__main__":
    main()
