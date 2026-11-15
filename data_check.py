"""Data availability checker for SANADA/EchoCount (Task 1 & Task 4).

Reports:
1. All mixture folders resolved from config.json ("C:\\coding\\SANADA\\sanada dataset") across
   both FSD_* and Org_* speech families.
2. Whether Multichannel4SpeakersTrim, data/isolated_sources, or data/FSD_Master has usable
   isolated clips and how many per speaker/class.
3. Exact requirements for blocked non-speech and supervised speech separation training.

Usage:
    python data_check.py --smoke
    python data_check.py --full
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from dataset_config import get_dataset_root, load_config, load_taxonomy, scan_all_mixtures

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def count_clips_by_subdir(directory: Path) -> dict:
    if not directory.is_dir():
        return {"total": 0, "per_class": {}}
    per_class: dict[str, int] = Counter()
    total = 0
    for f in directory.rglob("*"):
        if f.suffix.lower() in AUDIO_EXTS and not f.name.startswith("mix_"):
            per_class[f.parent.name] += 1
            total += 1
    return {"total": total, "per_class": dict(sorted(per_class.items()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Quick check")
    parser.add_argument("--full", action="store_true", help="Full check")
    parser.parse_args()

    cfg = load_config()
    ext_root = get_dataset_root(cfg)
    tax = load_taxonomy()

    print("=" * 78)
    print("SANADA / EchoCount — Data & Checkpoint Availability Report")
    print("=" * 78)
    print(f"Configured Dataset Root: \"{ext_root}\"")

    recs = scan_all_mixtures(cfg, include_duplicates=False)
    by_folder = Counter((r["family"], r["rel_folder"]) for r in recs)
    print("\n=== 1. Pre-rendered Mixture Datasets (Deduped) ===")
    for (fam, folder), cnt in sorted(by_folder.items()):
        print(f"  [{fam.upper():<6}] {folder:<36s} {cnt:5d} WAV files")
    print(f"  {'TOTAL UNIQUE RESOLVED MIXTURES':<45s} {len(recs):5d} WAV files")

    print("\n=== 2. Isolated Single-Source Clips (for Supervised Separator Training) ===")
    candidates = [
        ext_root / "Multichannel4SpeakersTrim",
        Path(r"C:\coding\SANADA\Multichannel4SpeakersTrim"),
        DATA_DIR / "Multichannel4SpeakersTrim",
        DATA_DIR / "isolated_sources",
        DATA_DIR / "FSD_Master",
        DATA_DIR / "FSD50K.dev_audio",
    ]
    found_any = False
    for cand in candidates:
        info = count_clips_by_subdir(cand)
        if info["total"] > 0:
            found_any = True
            print(f"  FOUND: {cand} ({info['total']} clips across {len(info['per_class'])} classes/speakers)")
            for cls, c in info["per_class"].items():
                print(f"    {cls:<32s}: {c:4d} clips")
        else:
            print(f"  {str(cand):<62s} : 0 usable clips (NOT FOUND)")

    if not found_any:
        print("\n  STATUS: ZERO isolated single-source clips exist on disk.")
        print("  - Supervised speech separation training requires:")
        print("      C:\\coding\\SANADA\\sanada dataset\\Multichannel4SpeakersTrim\\<split>\\Speaker_4_X\\*.wav")
        print("  - Pretrained speech separation (SpeechBrain SepFormer) requires weights in:")
        print("      C:\\coding\\SANADA\\Sanada\\models\\sepformer-wsj02mix\\")
        print("  - Non-speech separation (Animal, Musical instrument, Household/tools, Environment/other)")
        print("    remains BLOCKED until isolated clips for the 32 FSD classes are placed in:")
        print("      C:\\coding\\SANADA\\Sanada\\data\\isolated_sources\\<ClassName>\\*.wav")
        print(f"    Required 32 FSD classes: {', '.join(sorted(c for cs in tax['group_to_fine_classes'].values() for c in cs))}")

    print("\n=== 3. Checkpoint Status ===")
    ckpt_dir = BASE_DIR / "checkpoints"
    for name in (
        "source_count_clip.pt",
        "source_count_clip_backup.pt",
        "source_count_clip_v1_backup.pt",
        "tagger.pt",
        "tagger_backup.pt",
        "separator_iterative.pt",
    ):
        p = ckpt_dir / name
        if p.is_file():
            try:
                import torch
                ck = torch.load(str(p), map_location="cpu", weights_only=False)
                trained = ck.get("trained", "?") if isinstance(ck, dict) else "?"
                arch = ck.get("architecture", "?") if isinstance(ck, dict) else "?"
                ver = ck.get("version", ck.get("epoch", "?")) if isinstance(ck, dict) else "?"
                print(f"  {name:<36s} EXISTS | trained={str(trained):<5s} | arch={arch} | ver/ep={ver}")
            except Exception as exc:
                print(f"  {name:<36s} EXISTS | error={exc}")
        else:
            print(f"  {name:<36s} MISSING")


if __name__ == "__main__":
    main()
