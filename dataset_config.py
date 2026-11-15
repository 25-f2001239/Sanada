"""Configuration and dataset loader for EchoCount / SANADA.

Reads config.json for the external dataset path ("C:\\coding\\SANADA\\sanada dataset")
without copying WAV files. Remaps CSV mixture_path entries by split + filename,
excludes exact duplicate folders (Org_Fsd_Mix/Org_Mix_4_Sounds_Variable), and builds
both the standard (train/val/test) split and a Held-Out-Speaker split.

Usage:
    python dataset_config.py --smoke
    python dataset_config.py --full
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
TAXONOMY_PATH = BASE_DIR / "taxonomy.json"


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "dataset_path": r"C:\coding\SANADA\sanada dataset",
        "local_data_path": str(BASE_DIR / "data"),
        "real_eval_manifest": str(BASE_DIR / "data" / "real_eval" / "manifest.json"),
        "models_dir": str(BASE_DIR / "models"),
        "checkpoints_dir": str(BASE_DIR / "checkpoints"),
        "exclude_duplicate_folders": ["Org_Fsd_Mix/Org_Mix_4_Sounds_Variable"],
        "held_out_speakers": ["Speaker_4_1", "Speaker8", "Speaker9"],
        "seed": 42,
        "sample_rate": 16000,
        "clip_seconds": 8.0,
        "max_separable": 8,
        "quality_corr_threshold": 0.9,
        "speaker_cosine_similarity_max": 0.82,
    }


def load_taxonomy(path: Path = TAXONOMY_PATH) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    # Ensure class_to_group is always populated and consistent with group_to_fine_classes
    c2g = dict(data.get("class_to_group", {}))
    for grp, fcs in data.get("group_to_fine_classes", {}).items():
        for fc in fcs:
            c2g[fc] = grp
    data["class_to_group"] = c2g
    return data


def get_class_to_group(tax: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Return canonical class -> coarse group mapping from taxonomy.json."""
    tax = tax or load_taxonomy()
    return dict(tax.get("class_to_group", {}))


def get_group_for_class(label: str, tax: Optional[Dict[str, Any]] = None) -> str:
    """Resolve any fine class, friendly sound name, or target sound to its canonical group."""
    c2g = get_class_to_group(tax)
    if label in c2g:
        return c2g[label]
    norm = label.replace(" ", "_")
    if norm in c2g:
        return c2g[norm]
    norm_space = label.replace("_", " ")
    if norm_space in c2g:
        return c2g[norm_space]
    return "Environment/other"


def get_friendly_name(label: str, tax: Optional[Dict[str, Any]] = None) -> str:
    """Map an internal class label (e.g. 'Telephone', 'Computer_keyboard', 'Bark') to its friendly UI name."""
    tax = tax or load_taxonomy()
    fmap = tax.get("friendly_names", {})
    if label in fmap:
        return fmap[label]
    norm = label.replace(" ", "_")
    if norm in fmap:
        return fmap[norm]
    return label.replace("_", " ")


def get_dataset_root(cfg: Optional[Dict[str, Any]] = None) -> Path:
    cfg = cfg or load_config()
    ext = Path(cfg["dataset_path"])
    if ext.is_dir():
        return ext
    return Path(cfg.get("local_data_path", str(BASE_DIR / "data")))


def scan_all_mixtures(
    cfg: Optional[Dict[str, Any]] = None,
    include_duplicates: bool = False,
) -> List[Dict[str, Any]]:
    """Scan all metadata.csv files under dataset_path and remap mixture_path by split + filename.

    Each returned item has:
        path: Path
        rel_folder: str  (e.g. 'FSD_Mix_2_Sounds', 'Org_Mix_2_Sounds')
        family: str      ('fsd' or 'speech')
        split: str       ('train', 'val', 'test')
        held_out_speaker_split: str ('train', 'test_unseen_speaker')
        filename: str
        num_sources: int
        labels: list[str]
        source_files: list[str]
    """
    cfg = cfg or load_config()
    root = get_dataset_root(cfg)
    excluded = set(cfg.get("exclude_duplicate_folders", []))
    held_out_spks = set(cfg.get("held_out_speakers", ["Speaker_4_1", "Speaker8", "Speaker9"]))

    records: List[Dict[str, Any]] = []
    meta_paths = sorted(root.rglob("metadata.csv"))

    for mf in meta_paths:
        split_dir = mf.parent
        split_name = split_dir.name
        if split_name not in ("train", "val", "test"):
            continue
        rel_folder = str(split_dir.parent.relative_to(root)).replace("\\", "/")
        if not include_duplicates and rel_folder in excluded:
            continue
        if not (rel_folder.startswith("FSD_Mix_") or rel_folder.startswith("Org_Mix_") or "Org_Mix_" in rel_folder):
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
                labels = json.loads(row["speaker_labels"]) if isinstance(row.get("speaker_labels"), str) else []
            except Exception:
                labels = []
            try:
                sfiles = json.loads(row["source_files"]) if isinstance(row.get("source_files"), str) else []
            except Exception:
                sfiles = []

            k = int(row[col_k])
            has_held_out = family == "speech" and any(lbl in held_out_spks for lbl in labels)
            ho_split = "test_unseen_speaker" if has_held_out else ("train" if split_name == "train" else split_name)

            records.append({
                "path": disk_path,
                "rel_folder": rel_folder,
                "family": family,
                "split": split_name,
                "held_out_speaker": has_held_out,
                "held_out_speaker_split": ho_split,
                "filename": fname,
                "num_sources": k,
                "labels": labels,
                "source_files": sfiles,
            })

    return records


def audit_dataset(smoke: bool = False) -> Dict[str, Any]:
    """Run Task 1 verification audit and print summary."""
    cfg = load_config()
    root = get_dataset_root(cfg)
    print("=" * 84)
    print(f"TASK 1 AUDIT — Dataset Root: \"{root}\"")
    print("=" * 84)

    all_recs = scan_all_mixtures(cfg, include_duplicates=True)
    dedup_recs = scan_all_mixtures(cfg, include_duplicates=False)
    print(f"Total resolved mixtures (all metadata.csv): {len(all_recs)}")
    print(f"Total resolved mixtures (excluding duplicate Org_Fsd_Mix/...): {len(dedup_recs)}")

    fam_split = Counter((r["family"], r["split"]) for r in dedup_recs)
    for (fam, sp), cnt in sorted(fam_split.items()):
        print(f"  Family={fam:<8} Split={sp:<6} : {cnt:5d} mixtures")

    ho_counts = Counter((r["family"], r["held_out_speaker"]) for r in dedup_recs)
    print("\nHeld-Out-Speaker Split (held_out_speakers =", cfg.get("held_out_speakers"), "):")
    print(f"  Speech mixtures without held-out speakers (seen-speaker pool): {ho_counts[('speech', False)]}")
    print(f"  Speech mixtures WITH held-out speakers (unseen-speaker eval):  {ho_counts[('speech', True)]}")

    return {
        "dataset_root": str(root),
        "total_all": len(all_recs),
        "total_dedup": len(dedup_recs),
        "family_split": {f"{k[0]}_{k[1]}": v for k, v in fam_split.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Quick audit check")
    parser.add_argument("--full", action="store_true", help="Full audit check")
    args = parser.parse_args()
    audit_dataset(smoke=args.smoke)


if __name__ == "__main__":
    main()
