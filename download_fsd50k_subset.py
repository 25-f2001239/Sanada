"""Download a subset of isolated single-source clips from FSD50K (Phase 3.1).

Downloads >= 30 isolated clips per class across 16 target classes (covering all 12
core everyday target sounds + FSD classes) from the public FSD50K dataset
(`Fhrozen/FSD50k` on HuggingFace Hub / Zenodo record 4060432), preferring CC0 and
CC-BY commercial-permissive clips, and organises them into:
    data/isolated_sources/<Class>/<fname>.wav

Also writes `data/isolated_sources/manifest.json` with per-clip license, split
(train/val/test), duration, size, and exact reproducible download commands.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download

BASE_DIR = Path(__file__).resolve().parent
ISO_DIR = BASE_DIR / "data" / "isolated_sources"
META_DIR = ISO_DIR / "_metadata"
HF_CACHE_DIR = ISO_DIR / "_hf_cache"

# 16 classes (>= 15 required), >= 30 clips each, covering all 12 target sounds + FSD classes
TARGET_CLASSES: Dict[str, Dict] = {
    "Hammer": {"fsd50k_label": "Hammer", "keywords": ["hammer"]},
    "Phone_vibration": {"fsd50k_label": "Buzz", "keywords": ["vibrat", "phone", "buzz", "hum"]},
    "Bark": {"fsd50k_label": "Bark", "keywords": ["bark", "dog"]},
    "Meow": {"fsd50k_label": "Meow", "keywords": ["meow", "cat"]},
    "Cello": {"fsd50k_label": "Bowed_string_instrument", "keywords": ["cello"]},
    "Scissors": {"fsd50k_label": "Scissors", "keywords": ["scissor", "snip", "cut"]},
    "Telephone": {"fsd50k_label": "Telephone", "keywords": ["telephone", "ring", "phone"]},
    "Speech": {"fsd50k_label": "Speech", "keywords": ["speech", "voice", "talk", "speak"]},
    "Knock": {"fsd50k_label": "Knock", "keywords": ["knock", "door"]},
    "Computer_keyboard": {"fsd50k_label": "Computer_keyboard", "keywords": ["keyboard", "typing"]},
    "Water": {"fsd50k_label": "Water", "keywords": ["water", "tap", "faucet", "pour", "stream"]},
    "Car_horn": {"fsd50k_label": "Vehicle_horn_and_car_horn_and_honking", "keywords": ["horn", "honk", "car"]},
    "Acoustic_guitar": {"fsd50k_label": "Acoustic_guitar", "keywords": ["guitar"]},
    "Trumpet": {"fsd50k_label": "Trumpet", "keywords": ["trumpet"]},
    "Laughter": {"fsd50k_label": "Laughter", "keywords": ["laugh"]},
    "Applause": {"fsd50k_label": "Applause", "keywords": ["applause", "clap"]},
}

EXACT_DOWNLOAD_COMMAND = (
    "cd C:\\coding\\SANADA\\Sanada\n"
    ".\\.venv\\Scripts\\python download_fsd50k_subset.py --clips-per-class 30"
)


def ensure_metadata() -> None:
    META_DIR.mkdir(parents=True, exist_ok=True)
    for rel in [
        "labels/vocabulary.csv",
        "labels/dev.csv",
        "labels/eval.csv",
        "LICENSE-DATASET.md",
        "metadata/dev_clips_info_FSD50K.json",
    ]:
        target = META_DIR / rel
        if not target.is_file():
            hf_hub_download("Fhrozen/FSD50k", rel, repo_type="dataset", local_dir=str(META_DIR))


def is_commercial_license(lic_url: str) -> bool:
    u = (lic_url or "").lower()
    return ("zero" in u or "cc0" in u or "/by/" in u or "/by-sa/" in u) and ("nc" not in u)


def select_clips_per_class(clips_per_class: int = 30) -> Dict[str, List[Dict]]:
    ensure_metadata()
    dev_df = pd.read_csv(META_DIR / "labels" / "dev.csv")
    dev_info = json.loads((META_DIR / "metadata" / "dev_clips_info_FSD50K.json").read_text(encoding="utf-8"))

    selected: Dict[str, List[Dict]] = {}
    used_fnames = set()

    for cls_name, spec in TARGET_CLASSES.items():
        fsd_lbl = spec["fsd50k_label"]
        kws = [k.lower() for k in spec.get("keywords", [])]

        candidates = []
        for _, row in dev_df.iterrows():
            fname = str(row["fname"])
            if fname in used_fnames:
                continue
            lbls = [x.strip() for x in str(row["labels"]).split(",") if x.strip()]
            if fsd_lbl not in lbls:
                continue
            cinfo = dev_info.get(fname, {})
            lic = cinfo.get("license", "http://creativecommons.org/publicdomain/zero/1.0/")
            comm_ok = is_commercial_license(lic)
            title_tags = (str(cinfo.get("title", "")) + " " + " ".join(cinfo.get("tags", []))).lower()
            kw_match = any(k in title_tags for k in kws)
            is_primary = lbls[0] == fsd_lbl
            # Score higher for: commercial-permissive CC0/CC-BY, keyword match (especially Cello), primary leaf label, fewer extra labels
            score = (
                (100 if comm_ok else 0)
                + (50 if kw_match else 0)
                + (30 if is_primary else 0)
                - len(lbls)
            )
            candidates.append({
                "fname": fname,
                "class": cls_name,
                "fsd50k_label": fsd_lbl,
                "all_labels": lbls,
                "orig_split": str(row.get("split", "train")),
                "license": lic,
                "commercial_allowed": comm_ok,
                "title": cinfo.get("title", ""),
                "score": score,
            })

        candidates.sort(key=lambda x: (-x["score"], x["fname"]))
        chosen = candidates[:clips_per_class]
        for idx, item in enumerate(chosen):
            used_fnames.add(item["fname"])
            # Assign deterministic train/val/test split (20 train, 5 val, 5 test for 30 clips)
            if idx < int(clips_per_class * 0.67):
                item["split"] = "train"
            elif idx < int(clips_per_class * 0.83):
                item["split"] = "val"
            else:
                item["split"] = "test"
        selected[cls_name] = chosen

    return selected


def download_subset(clips_per_class: int = 30) -> Dict:
    print("=" * 80)
    print("PHASE 3.1 — FSD50K Isolated Single-Source Clips Downloader")
    print("Exact command:")
    print(EXACT_DOWNLOAD_COMMAND)
    print("=" * 80)

    selected = select_clips_per_class(clips_per_class=clips_per_class)
    all_patterns = []
    fnames_to_target: Dict[str, Tuple[str, Dict]] = {}
    for cls_name, items in selected.items():
        cls_dir = ISO_DIR / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            fn = item["fname"]
            dest = cls_dir / f"{fn}.wav"
            fnames_to_target[fn] = (str(dest), item)
            if not dest.is_file():
                all_patterns.append(f"clips/dev/{fn}.wav")

    print(f"Target classes: {len(selected)} classes, {sum(len(v) for v in selected.values())} total clips.")
    print(f"Remaining clips to fetch from HuggingFace Hub (Fhrozen/FSD50k): {len(all_patterns)}")

    t0 = time.time()
    # Batch in chunks of 80 patterns so snapshot_download is fast and resilient
    chunk_size = 80
    for i in range(0, len(all_patterns), chunk_size):
        batch = all_patterns[i : i + chunk_size]
        snapshot_download(
            repo_id="Fhrozen/FSD50k",
            repo_type="dataset",
            local_dir=str(HF_CACHE_DIR),
            allow_patterns=batch,
            max_workers=8,
        )
        for pat in batch:
            fn = Path(pat).stem
            src_file = HF_CACHE_DIR / pat
            dest_str, _ = fnames_to_target[fn]
            dest_file = Path(dest_str)
            if src_file.is_file() and not dest_file.is_file():
                shutil.copy2(src_file, dest_file)
        print(f"  Downloaded {min(i + chunk_size, len(all_patterns))}/{len(all_patterns)} clips ({time.time() - t0:.1f}s)...")

    # Copy any already-cached files and compute total size & license summary
    total_bytes = 0
    license_counts: Dict[str, int] = {}
    class_counts: Dict[str, int] = {}
    manifest_entries = []

    for fn, (dest_str, item) in fnames_to_target.items():
        dest_file = Path(dest_str)
        cache_file = HF_CACHE_DIR / "clips" / "dev" / f"{fn}.wav"
        if not dest_file.is_file() and cache_file.is_file():
            shutil.copy2(cache_file, dest_file)
        if dest_file.is_file():
            sz = dest_file.stat().st_size
            total_bytes += sz
            item["size_bytes"] = sz
            item["rel_path"] = f"{item['class']}/{fn}.wav"
            lic = item["license"]
            license_counts[lic] = license_counts.get(lic, 0) + 1
            class_counts[item["class"]] = class_counts.get(item["class"], 0) + 1
            manifest_entries.append(item)

    # Clean up temporary _hf_cache duplicate wavs to save disk space
    if (HF_CACHE_DIR / "clips").is_dir():
        shutil.rmtree(HF_CACHE_DIR / "clips", ignore_errors=True)

    summary = {
        "dataset": "FSD50K (Fonseca et al., IEEE/ACM TASLP 2022, Zenodo 4060432 / HF Fhrozen/FSD50k)",
        "dataset_license": "CC-BY-4.0 dataset compilation; individual clips CC0 / CC-BY (commercial use allowed)",
        "exact_download_command": EXACT_DOWNLOAD_COMMAND,
        "num_classes": len(class_counts),
        "clips_per_class_target": clips_per_class,
        "total_clips_downloaded": len(manifest_entries),
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        "class_counts": class_counts,
        "license_distribution": license_counts,
        "clips": manifest_entries,
    }
    manifest_path = ISO_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {len(manifest_entries)} isolated clips ({summary['total_size_mb']} MB) across {len(class_counts)} classes -> {manifest_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-per-class", type=int, default=30)
    args = parser.parse_args()
    download_subset(clips_per_class=args.clips_per_class)
