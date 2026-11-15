"""Evaluate the counting checkpoint on a held-out set and write the JSON the Evaluation tab reads.

    python evaluate.py
    python evaluate.py --metadata data/FSD_Mix_Variable_Sounds/test/metadata.csv --output evaluation/results.json

Every number in the output comes from running the real checkpoint on real audio. Nothing is
hard-coded: the "density saturation" finding is computed from the per-clip predictions.
Exit code is non-zero (and no JSON is written) if the checkpoint or the data are missing.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from counting import DEFAULT_MC_PASSES, checkpoint_fingerprint, count_with_uncertainty
from model import MODEL_PATH, checkpoint_status, load_audio

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_METADATA = BASE_DIR / "data" / "FSD_Mix_Variable_Sounds" / "test" / "metadata.csv"
DEFAULT_OUTPUT = BASE_DIR / "evaluation" / "results.json"
BUCKET_EDGES = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 31), (32, 10_000)]


def resolve_audio(csv_path, raw):
    raw = Path(str(raw).replace("\\", "/"))
    for candidate in (raw, csv_path.parent / raw, csv_path.parent / raw.name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Audio listed in {csv_path} was not found: {raw}")


def load_manifest(csv_path):
    frame = pd.read_csv(csv_path)
    column = "num_sources" if "num_sources" in frame else "num_speakers" if "num_speakers" in frame else None
    if column is None or "mixture_path" not in frame:
        raise ValueError(f"{csv_path} needs mixture_path and num_sources (or num_speakers) columns.")
    return [(resolve_audio(csv_path, p), int(n)) for p, n in zip(frame["mixture_path"], frame[column])], frame


def leakage_check(frame, csv_path):
    """Count test source files that also appear in a sibling train split (when it exists)."""
    train_csv = csv_path.parent.parent / "train" / "metadata.csv"
    if "source_files" not in frame or not train_csv.is_file():
        return {"checked": False, "overlapping_source_files": None}
    train = pd.read_csv(train_csv)
    if "source_files" not in train:
        return {"checked": False, "overlapping_source_files": None}
    flatten = lambda col: {s for row in col for s in json.loads(row)}
    return {"checked": True, "overlapping_source_files": len(flatten(frame["source_files"]) & flatten(train["source_files"]))}


def summarise(rows):
    true = np.array([r["true"] for r in rows], dtype=float)
    pred = np.array([r["pred"] for r in rows], dtype=float)
    raw = np.array([r["raw"] for r in rows], dtype=float)
    err = pred - true
    return {
        "n": len(rows),
        "mae": float(np.mean(np.abs(err))),
        "mae_raw": float(np.mean(np.abs(raw - true))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "exact_accuracy": float(np.mean(err == 0)),
        "within_one": float(np.mean(np.abs(err) <= 1)),
        "mean_total_std": float(np.mean([r["std"] for r in rows])),
        "mc_std_coverage": float(np.mean([abs(r["raw"] - r["true"]) <= r["std"] for r in rows])),
    }


def group(rows, key):
    out = []
    for label, members in key:
        if not members:
            continue
        stats = summarise(members)
        stats.update(label=label, mean_true=float(np.mean([r["true"] for r in members])),
                     mean_pred=float(np.mean([r["pred"] for r in members])),
                     mean_raw=float(np.mean([r["raw"] for r in members])))
        out.append(stats)
    return out


def saturation_finding(rows, high_count, tolerance):
    """Derive the density-saturation finding from the data. Direction is whatever the data show."""
    high = [r for r in rows if r["true"] >= high_count]
    if len(high) < 3:
        return {"evaluated": False, "reason": f"Fewer than 3 held-out clips have {high_count}+ sources."}
    bias = float(np.mean([r["pred"] - r["true"] for r in high]))
    low = [r for r in rows if r["true"] < high_count]
    low_bias = float(np.mean([r["pred"] - r["true"] for r in low])) if low else None
    direction = "overcounting" if bias > tolerance else "undercounting" if bias < -tolerance else "none"
    ranked = sorted(rows, key=lambda r: r["true"])
    mean_pred_by_true = {}
    for r in ranked:
        mean_pred_by_true.setdefault(r["true"], []).append(r["pred"])
    curve = sorted((t, float(np.mean(v))) for t, v in mean_pred_by_true.items())
    peak_pred = max(v for _, v in curve)
    plateau_from = next((t for t, v in curve if v >= peak_pred - 1.0), None)
    return {
        "evaluated": True,
        "high_count_threshold": high_count,
        "tolerance": tolerance,
        "n_high": len(high),
        "mean_bias_high": bias,
        "mean_bias_low": low_bias,
        "direction": direction,
        "max_mean_predicted": peak_pred,
        "predictions_plateau_from_true_count": plateau_from,
        "worst_examples": [
            {"file": r["file"], "true": r["true"], "pred": r["pred"], "raw": round(r["raw"], 3)}
            for r in sorted(high, key=lambda r: -abs(r["pred"] - r["true"]))[:5]
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--passes", type=int, default=DEFAULT_MC_PASSES, help="MC-dropout passes per clip")
    parser.add_argument("--high-count", type=int, default=20, help="true count at/above which the saturation check applies")
    parser.add_argument("--tolerance", type=float, default=1.0, help="mean bias (sources) that counts as a systematic error")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    status = checkpoint_status(args.checkpoint)
    if not status["trained"]:
        sys.exit(f"Counting checkpoint unavailable ({args.checkpoint}): {status['error'] or 'file not found'}")
    metadata = args.metadata.expanduser().resolve()
    if not metadata.is_file():
        sys.exit(f"Held-out metadata not found: {metadata}")

    import torch

    torch.manual_seed(args.seed)
    items, frame = load_manifest(metadata)
    rows = []
    for index, (path, true_count) in enumerate(items, 1):
        waveform, _ = load_audio(path)
        result = count_with_uncertainty(waveform, args.checkpoint, args.passes)
        rows.append({"file": path.name, "true": true_count, "pred": result["rounded_count"],
                     "raw": result["raw_regression_value"], "std": result["total_std"]})
        if index % 10 == 0 or index == len(items):
            print(f"{index}/{len(items)} clips", flush=True)

    per_count = group(rows, [(str(c), [r for r in rows if r["true"] == c]) for c in sorted({r["true"] for r in rows})])
    buckets = group(rows, [(f"{lo}-{hi}" if hi < 10_000 else f"{lo}+", [r for r in rows if lo <= r["true"] <= hi]) for lo, hi in BUCKET_EDGES])
    payload = {
        "available": True,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": {"path": str(args.checkpoint), "sha256": checkpoint_fingerprint(args.checkpoint),
                       "epoch": status.get("epoch"), "min_sources": status["min_sources"], "max_sources": status["max_sources"]},
        "dataset": {"metadata": str(metadata), "name": metadata.parent.parent.name, "split": metadata.parent.name,
                    "leakage_check": leakage_check(frame, metadata)},
        "uncertainty": {"method": "mc_dropout", "passes": args.passes, "seed": args.seed},
        "overall": summarise(rows),
        "buckets": buckets,
        "per_count": per_count,
        "saturation": saturation_finding(rows, args.high_count, args.tolerance),
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    overall = payload["overall"]
    print(f"Wrote {args.output}\n n={overall['n']} MAE={overall['mae']:.3f} exact={overall['exact_accuracy']:.3f} "
          f"within1={overall['within_one']:.3f} bias={overall['bias']:+.3f}")
    print(" saturation:", payload["saturation"].get("direction", payload["saturation"].get("reason")))


if __name__ == "__main__":
    main()
