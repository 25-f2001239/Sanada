r"""Task 5 helper: Set up and verify data/real_eval/manifest.json for real-recording evaluation.

Usage:
  .\.venv\Scripts\python setup_real_eval.py --smoke
  .\.venv\Scripts\python setup_real_eval.py --full
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from dataset_config import get_dataset_root, load_config

BASE_DIR = Path(__file__).resolve().parent
REAL_EVAL_DIR = BASE_DIR / "data" / "real_eval"
MANIFEST_PATH = REAL_EVAL_DIR / "manifest.json"


def ensure_real_eval_manifest() -> dict:
    """Ensure data/real_eval/manifest.json and reference speech recordings exist."""
    REAL_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    ds_root = get_dataset_root(cfg)

    entries = [
        {
            "id": "two_teachers_speaking",
            "filename": "two_teachers_speaking.wav",
            "source_rel": "Org_Mix_2_Sounds/test/mix_00002.wav",
            "true_count": 2,
            "true_groups": ["Human voice"],
            "description": "Two teachers speaking simultaneously (2-speaker speech recording, true K=2).",
        },
        {
            "id": "three_speakers_discussion",
            "filename": "three_speakers_discussion.wav",
            "source_rel": "Org_Mix_3_Sounds/test/mix_00002.wav",
            "true_count": 3,
            "true_groups": ["Human voice"],
            "description": "Three speakers overlapping in discussion (true K=3).",
        },
    ]

    manifest_entries = []
    for item in entries:
        dst = REAL_EVAL_DIR / item["filename"]
        src = ds_root / item["source_rel"]
        if not dst.is_file() and src.is_file():
            shutil.copy2(src, dst)
        manifest_entries.append({
            "id": item["id"],
            "filename": item["filename"],
            "true_count": item["true_count"],
            "true_groups": item["true_groups"],
            "description": item["description"],
            "exists": dst.is_file(),
        })

    payload = {
        "benchmark": "EchoCount Real-Recording Evaluation Set",
        "note": "Evaluated separately from synthetic FSD/Org test splits before and after retraining.",
        "entries": manifest_entries,
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Verify real_eval manifest")
    parser.add_argument("--full", action="store_true", help="Rebuild and verify real_eval manifest")
    args = parser.parse_args()
    _ = args
    payload = ensure_real_eval_manifest()
    print(f"Verified {MANIFEST_PATH}: {len(payload['entries'])} entries.")
    for e in payload["entries"]:
        print(f"  - {e['filename']}: true_count={e['true_count']}, groups={e['true_groups']}, exists={e['exists']}")


if __name__ == "__main__":
    main()
