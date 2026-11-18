"""Dataset scanning, leakage audit, and dev-gated ground-truth SHA-256 lookup."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    DATA_DIR,
    get_dataset_root,
    get_group_for_class,
    load_config,
    load_taxonomy,
)

INDEX_PATH = DATA_DIR / "ground_truth_index.json"
_REGISTRY_HASH: dict[str, dict[str, Any]] | None = None


def is_dev_mode() -> bool:
    """Return True only when ECHOCOUNT_DEV=1 (ground-truth lookup is disabled in production)."""
    return os.getenv("ECHOCOUNT_DEV", "0").strip() == "1"


def scan_all_mixtures(
    cfg: dict[str, Any] | None = None,
    include_duplicates: bool = False,
) -> list[dict[str, Any]]:
    """Scan metadata.csv files under dataset_path and remap mixture_path by split + filename."""
    cfg = cfg or load_config()
    root = get_dataset_root(cfg)
    excluded = set(cfg.get("exclude_duplicate_folders", []))
    held_out_spks = set(cfg.get("held_out_speakers", ["Speaker_4_1", "Speaker8", "Speaker9"]))
    records: list[dict[str, Any]] = []

    for mf in sorted(root.rglob("metadata.csv")):
        split_dir = mf.parent
        split_name = split_dir.name
        if split_name not in ("train", "val", "test"):
            continue
        rel_folder = str(split_dir.parent.relative_to(root)).replace("\\", "/")
        if not include_duplicates and rel_folder in excluded:
            continue
        if not (
            rel_folder.startswith("FSD_Mix_")
            or rel_folder.startswith("Org_Mix_")
            or "Org_Mix_" in rel_folder
        ):
            continue

        family = "speech" if "Org_" in rel_folder else "fsd"
        df = pd.read_csv(mf)
        col_k = "num_speakers" if "num_speakers" in df.columns else "num_sources"
        for _, row in df.iterrows():
            fname = Path(str(row["mixture_path"]).replace("\\", "/")).name
            disk_path = split_dir / fname
            if not disk_path.is_file():
                continue
            try:
                labels = (
                    json.loads(row["speaker_labels"])
                    if isinstance(row.get("speaker_labels"), str)
                    else []
                )
            except Exception:
                labels = []
            try:
                sfiles = (
                    json.loads(row["source_files"])
                    if isinstance(row.get("source_files"), str)
                    else []
                )
            except Exception:
                sfiles = []
            has_ho = family == "speech" and any(lbl in held_out_spks for lbl in labels)
            records.append(
                {
                    "path": disk_path,
                    "rel_folder": rel_folder,
                    "family": family,
                    "split": split_name,
                    "held_out_speaker": has_ho,
                    "held_out_speaker_split": "test_unseen_speaker" if has_ho else split_name,
                    "filename": fname,
                    "num_sources": int(row[col_k]),
                    "labels": labels,
                    "source_files": sfiles,
                }
            )
    return records


def audit_dataset() -> dict[str, Any]:
    """Return summary counts across all FSD + Org mixtures and held-out speakers."""
    cfg = load_config()
    all_recs = scan_all_mixtures(cfg, include_duplicates=True)
    dedup_recs = scan_all_mixtures(cfg, include_duplicates=False)
    fam_split = Counter((r["family"], r["split"]) for r in dedup_recs)
    return {
        "dataset_root": str(get_dataset_root(cfg)),
        "total_all": len(all_recs),
        "total_dedup": len(dedup_recs),
        "family_split": {f"{k[0]}_{k[1]}": v for k, v in fam_split.items()},
    }


def build_and_save_index(
    output_path: Path = INDEX_PATH, limit: int | None = None
) -> dict[str, Any]:
    """Build SHA-256 ground-truth registry across all configured FSD + Org mixtures."""
    by_hash, by_relpath, by_filename = {}, {}, {}
    tax = load_taxonomy()
    records = scan_all_mixtures(include_duplicates=False)
    if limit is not None:
        records = records[:limit]

    for r in records:
        disk_file = Path(r["path"])
        if not disk_file.is_file():
            continue
        k, family = int(r["num_sources"]), r["family"]
        labels = (
            [f"Human {i} (Human voice)" for i in range(1, k + 1)]
            if family == "speech"
            else list(r["labels"])
        )
        groups = (
            ["Human voice"]
            if family == "speech"
            else sorted({get_group_for_class(lbl, tax) for lbl in labels})
        )
        h = hashlib.sha256(disk_file.read_bytes()).hexdigest()
        rel_path = f"{r['rel_folder']}/{r['split']}/{r['filename']}"
        info = {
            "dataset": r["rel_folder"],
            "split": r["split"],
            "filename": r["filename"],
            "family": family,
            "true_k": k,
            "true_labels": labels,
            "true_groups": groups,
            "rel_path": rel_path,
            "size_bytes": disk_file.stat().st_size,
            "sha256": h,
        }
        by_hash[h] = info
        by_relpath[rel_path.lower()] = info
        by_filename.setdefault(r["filename"].lower(), []).append(info)

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


def load_registry() -> dict[str, dict[str, Any]]:
    """Load or build the SHA-256 ground-truth hash registry."""
    global _REGISTRY_HASH
    if _REGISTRY_HASH is not None:
        return _REGISTRY_HASH
    try:
        data = (
            json.loads(INDEX_PATH.read_text(encoding="utf-8"))
            if INDEX_PATH.is_file()
            else build_and_save_index(INDEX_PATH)
        )
    except Exception:
        data = build_and_save_index(INDEX_PATH)
    _REGISTRY_HASH = data.get("by_hash", {})
    return _REGISTRY_HASH


def lookup_ground_truth(file_path: Path, allow_in_eval: bool = False) -> dict[str, Any] | None:
    """Look up ground truth strictly by SHA-256 hash when ECHOCOUNT_DEV=1 or allow_in_eval=True."""
    if not allow_in_eval and not is_dev_mode():
        return None
    file_path = Path(file_path)
    if not file_path.is_file():
        return None
    by_hash = load_registry()
    tax = load_taxonomy()
    try:
        h = hashlib.sha256(file_path.read_bytes()).hexdigest()
        match = by_hash.get(h)
        if not match:
            return None
        labels = match.get("true_labels", [])
        true_groups = (
            ["Human voice"]
            if match.get("family") == "speech"
            else sorted({get_group_for_class(lbl, tax) for lbl in labels})
        )
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
        return None
