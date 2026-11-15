"""Ground truth registry for SANADA/EchoCount mixtures (FSD + Speech Org_*).

Indexes all pre-rendered mixture clips across both FSD_Mix_*_Sounds and Org_Mix_*
datasets by SHA-256 and by relative path/filename, mapping each to its true source
count (K), source family ('fsd' or 'speech'), and true source labels.
For speech mixtures, speaker IDs (Speaker_4_X / SpeakerN) are mapped to neutral
labels ('Human 1 (Human voice)', 'Human 2 (Human voice)', ...) and never exposed as
speaker-identity tags.

Used by the Analyze page to display 'Ground truth (if known)' ONLY when an uploaded
file matches a known dataset mixture clip. Never shown for unknown user files.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional

import dataset_config

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_PATH = DATA_DIR / "ground_truth_index.json"

_REGISTRY_HASH: Optional[Dict[str, Dict[str, Any]]] = None
_REGISTRY_NAME: Optional[Dict[str, Dict[str, Any]]] = None


def build_and_save_index(output_path: Path = INDEX_PATH, limit: Optional[int] = None) -> Dict[str, Any]:
    """Scan all configured FSD + Org metadata CSVs and build mapping by file sha256 and relpath."""
    by_hash: Dict[str, Dict[str, Any]] = {}
    by_relpath: Dict[str, Dict[str, Any]] = {}
    by_filename: Dict[str, list] = {}

    tax = dataset_config.load_taxonomy()
    records = dataset_config.scan_all_mixtures(include_duplicates=False)
    if limit is not None:
        records = records[:limit]

    for r in records:
        disk_file = Path(r["path"])
        if not disk_file.is_file():
            continue
        folder = r["rel_folder"]
        split = r["split"]
        fname = r["filename"]
        k = int(r["num_sources"])
        family = r["family"]
        if family == "speech":
            # Neutral human voice labels (never expose Speaker_4_X identity strings as class labels)
            labels = [f"Human {i} (Human voice)" for i in range(1, k + 1)]
            groups = ["Human voice"]
        else:
            labels = list(r["labels"])
            groups = sorted({dataset_config.get_group_for_class(lbl, tax) for lbl in labels})

        info = {
            "dataset": folder,
            "split": split,
            "filename": fname,
            "family": family,
            "true_k": k,
            "true_labels": labels,
            "true_groups": groups,
            "rel_path": f"{folder}/{split}/{fname}",
            "size_bytes": disk_file.stat().st_size,
        }
        file_bytes = disk_file.read_bytes()
        h = hashlib.sha256(file_bytes).hexdigest()
        info["sha256"] = h
        by_hash[h] = info

        rel_key = f"{folder}/{split}/{fname}".lower()
        by_relpath[rel_key] = info

        by_filename.setdefault(fname.lower(), []).append(info)

    payload = {
        "generated_at": time.time(),
        "total_files": len(by_relpath),
        "unique_hashes": len(by_hash),
        "by_hash": by_hash,
        "by_relpath": by_relpath,
        "by_filename": by_filename,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_registry():
    global _REGISTRY_HASH, _REGISTRY_NAME
    if _REGISTRY_HASH is not None:
        return _REGISTRY_HASH, _REGISTRY_NAME

    if not INDEX_PATH.is_file():
        build_and_save_index(INDEX_PATH)

    try:
        data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        _REGISTRY_HASH = data.get("by_hash", {})
        _REGISTRY_NAME = data.get("by_filename", {})
    except Exception:
        data = build_and_save_index(INDEX_PATH)
        _REGISTRY_HASH = data.get("by_hash", {})
        _REGISTRY_NAME = data.get("by_filename", {})

    return _REGISTRY_HASH, _REGISTRY_NAME


def is_dev_mode() -> bool:
    """Ground-truth lookup in the web app is strictly gated behind ECHOCOUNT_DEV=1 (off by default)."""
    return os.getenv("ECHOCOUNT_DEV", "0").strip() == "1"


def lookup_ground_truth(
    file_path: Path,
    upload_filename: Optional[str] = None,
    allow_in_eval: bool = False,
) -> Optional[Dict[str, Any]]:
    """Look up ground truth for a file strictly by SHA-256 audio hash.

    Phase 0.1 rule:
    - Disabled in production by default (`ECHOCOUNT_DEV != "1"`), returning None so
      no ground-truth card, SHA-256 dataset matching, or "In-distribution test file"
      note is ever exposed in production API responses or UI.
    - Only active when `ECHOCOUNT_DEV=1` or when called from an offline evaluation
      script with `allow_in_eval=True`.
    """
    if not allow_in_eval and not is_dev_mode():
        return None

    file_path = Path(file_path)
    if not file_path.is_file():
        return None

    by_hash, _ = load_registry()
    tax = dataset_config.load_taxonomy()

    # Match strictly by SHA-256 content hash (never by filename alone)
    try:
        h = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if h in by_hash:
            match = by_hash[h]
            labels = match.get("true_labels", [])
            if match.get("family") == "speech":
                true_groups = ["Human voice"]
            else:
                true_groups = sorted({dataset_config.get_group_for_class(lbl, tax) for lbl in labels})
            return {
                "known": True,
                "card_label": "Dataset file (known ground truth)",
                "in_distribution_test": match.get("split") == "test",
                "in_distribution_note": "In-distribution test file",
                "dataset": match["dataset"],
                "split": match["split"],
                "filename": match["filename"],
                "family": match.get("family", "fsd"),
                "true_k": match["true_k"],
                "true_labels": labels,
                "true_groups": true_groups,
                "sha256": h,
                "match_type": "exact_sha256",
            }
    except Exception:
        pass

    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Quick check on existing index or 50-file sample")
    parser.add_argument("--full", action="store_true", help="Build full index across all FSD + Org mixtures")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of data/ground_truth_index.json")
    args = parser.parse_args()

    if args.rebuild or args.full or not INDEX_PATH.is_file():
        limit = 50 if (args.smoke and not args.full and not INDEX_PATH.is_file()) else None
        print("Building ground-truth index across all FSD + Org metadata.csv files...")
        payload = build_and_save_index(INDEX_PATH, limit=limit)
        print(
            f"Indexed {payload['total_files']} dataset files "
            f"({payload['unique_hashes']} unique SHA-256 hashes) into {INDEX_PATH}"
        )
    else:
        by_hash, _ = load_registry()
        print(f"Loaded ground-truth index: {len(by_hash)} unique SHA-256 dataset files.")

    # Verify lookup on both FSD and Org mixtures
    real_eval_teacher = DATA_DIR / "real_eval" / "two_teachers_speaking.wav"
    if real_eval_teacher.is_file():
        gt = lookup_ground_truth(real_eval_teacher, real_eval_teacher.name, allow_in_eval=True)
        if gt:
            print(
                f"  [Real Eval two_teachers_speaking.wav] -> {gt['dataset']}/{gt['split']}/{gt['filename']}: "
                f"True K={gt['true_k']}, Groups={gt.get('true_groups')}, Labels={gt['true_labels']}"
            )


if __name__ == "__main__":
    main()
