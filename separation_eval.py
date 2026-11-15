"""Evaluate the separation checkpoint on held-out mixtures with ground truth stems.

Produces evaluation/separation_results.json for the Evaluation tab.

Usage:
    python separation_eval.py
    python separation_eval.py --checkpoint checkpoints/separator_iterative.pt --output evaluation/separation_results.json
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from counting import file_sha256
from separation import (
    DEVICE,
    HOP_LENGTH,
    MAX_SEPARABLE,
    N_FFT,
    SEPARATION_SAMPLE_RATE,
    SEPARATOR_ARCHITECTURE,
    IterativeMaskSeparator,
    _read_checkpoint,
    separator_status,
)
from train_separator import calculate_sisdr, pit_loss, separate_forward

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = BASE_DIR / "checkpoints" / "separator_iterative.pt"
DEFAULT_OUTPUT = BASE_DIR / "evaluation" / "separation_results.json"
DEFAULT_TEST_DIR = BASE_DIR / "data" / "isolated_sources" / "test"


def evaluate_separation(checkpoint_path, test_dir=None, output_path=DEFAULT_OUTPUT, max_eval=60, seed=42):
    checkpoint_path = Path(checkpoint_path)
    status = separator_status(checkpoint_path)
    if not status["available"]:
        sys.exit(f"Separation checkpoint unavailable ({checkpoint_path}): {status.get('reason')}")

    checkpoint, config = _read_checkpoint(checkpoint_path)
    model = IterativeMaskSeparator(config["channels"], config["blocks"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    is_trained = bool(checkpoint.get("trained", False)) and checkpoint.get("version") != "UNTRAINED-INIT"
    window = torch.hann_window(N_FFT, device=DEVICE)

    # Check for test data
    from train_separator import DynamicMixtureDataset, SyntheticSmokeDataset, find_isolated_clips
    test_clips = find_isolated_clips(test_dir) if test_dir and Path(test_dir).is_dir() else []

    if test_clips:
        print(f"Evaluating on {len(test_clips)} isolated test clips from {test_dir}...")
        ds = DynamicMixtureDataset(test_clips, num_mixtures=max_eval, seed=seed)
        data_source = str(test_dir)
    else:
        print("Note: No isolated test clips found; evaluating on synthetic test mixtures for pipeline benchmark.")
        ds = SyntheticSmokeDataset(num_samples=min(max_eval, 30), seed=seed)
        data_source = "synthetic_benchmark"

    results_by_k = {k: [] for k in range(2, MAX_SEPARABLE + 1)}
    all_results = []

    print(f"\nEvaluating separation checkpoint: {checkpoint_path.name}")
    print(f"Trained: {is_trained} | Version: {checkpoint.get('version')} | Architecture: {SEPARATOR_ARCHITECTURE}")
    print("-" * 70)

    with torch.no_grad():
        for idx, (mix, true_stems, k) in enumerate(ds, 1):
            mix = mix.to(DEVICE)
            true_stems_time = torch.stack([s.to(DEVICE) for s in true_stems])
            true_stems_spec = torch.stack([torch.stft(s, N_FFT, HOP_LENGTH, window=window, return_complex=True) for s in true_stems_time])
            pred_stems_time, pred_stems_spec = separate_forward(model, mix, k, window)
            _, perm = pit_loss(pred_stems_time, pred_stems_spec, true_stems_time, true_stems_spec)

            stems_eval = []
            for p_idx, t_idx in perm:
                p_stem = pred_stems_time[p_idx]
                t_stem = true_stems_time[t_idx]
                sdr_pred = calculate_sisdr(p_stem, t_stem).item()
                sdr_mix = calculate_sisdr(mix, t_stem).item()
                sisdri = sdr_pred - sdr_mix
                stems_eval.append({
                    "stem_index": p_idx + 1,
                    "target_index": t_idx + 1,
                    "si_sdr_db": round(sdr_pred, 2),
                    "si_sdri_db": round(sisdri, 2),
                })
                all_results.append(sisdri)
                results_by_k[k].append(sisdri)

            if idx % 10 == 0 or idx == len(ds):
                print(f"[{idx}/{len(ds)}] mixtures evaluated...", flush=True)

    overall_mean = float(np.mean(all_results)) if all_results else 0.0
    overall_median = float(np.median(all_results)) if all_results else 0.0

    by_k_summary = []
    for k in range(2, MAX_SEPARABLE + 1):
        if results_by_k[k]:
            vals = results_by_k[k]
            by_k_summary.append({
                "k": k,
                "n_stems": len(vals),
                "mean_sisdri_db": round(float(np.mean(vals)), 2),
                "median_sisdri_db": round(float(np.median(vals)), 2),
                "min_sisdri_db": round(float(np.min(vals)), 2),
                "max_sisdri_db": round(float(np.max(vals)), 2),
            })

    finding = "poor" if overall_mean < 1.0 else "moderate" if overall_mean < 6.0 else "good"

    payload = {
        "available": True,
        "trained": is_trained,
        "warning": None if is_trained else "Untrained demo weights: stems are NOT real separations",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
            "version": checkpoint.get("version"),
            "architecture": SEPARATOR_ARCHITECTURE,
            "best_train_sisdri": checkpoint.get("best_val_sisdri"),
        },
        "dataset": {
            "source": data_source,
            "evaluated_mixtures": len(ds),
            "total_stems_evaluated": len(all_results),
        },
        "overall": {
            "mean_sisdri_db": round(overall_mean, 2),
            "median_sisdri_db": round(overall_median, 2),
            "performance_level": finding,
        },
        "by_source_count": by_k_summary,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote separation evaluation results to {output_path}")
    print(f"Overall Mean SI-SDRi: {overall_mean:+.2f} dB (Level: {finding})")
    for row in by_k_summary:
        print(f"  K={row['k']}: Mean SI-SDRi = {row['mean_sisdri_db']:+.2f} dB across {row['n_stems']} stems")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-eval", type=int, default=40)
    args = parser.parse_args()

    evaluate_separation(args.checkpoint, args.test_dir, args.output, args.max_eval)


if __name__ == "__main__":
    main()
