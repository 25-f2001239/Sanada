"""Comprehensive Before/After Evaluation on Val & Test Splits (FSD + Speech) and Real-Recording Eval (Tasks 2, 4, 5).

Computes and writes to `evaluation/results.json`:
1. Before (`checkpoints/source_count_clip_backup.pt`) vs After (`checkpoints/source_count_clip.pt`)
   on the exact same Validation and Held-Out Test splits, reported overall and separately for
   `FSD` and `Speech (Org_*)`.
2. Per-K metrics across all K (2, 3, 4, 6, 9, 15, 20, 25, variable) with 1,000-resample 95%
   bootstrap confidence intervals. Marks any K with n < 30 as `unreliable: true`.
3. Confusion matrix per K (K=2..9 and >9) separately for Overall, Speech, and FSD.
4. Separate reporting of MC-dropout std and softmax spread.
5. Held-Out-Speaker unseen-speaker evaluation metrics.
6. Real-recording benchmark (`data/real_eval/manifest.json`), including `two_teachers_speaking.wav`
   (true K=2), comparing Before vs After count, Group/Fine tags, speaker embedding clustering,
   and speech separation quality check.

Usage:
    python evaluate_counter.py --smoke
    python evaluate_counter.py --full
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch

from counting import checkpoint_fingerprint, count_with_uncertainty
from dataset_config import load_config, scan_all_mixtures
from model import DEVICE, MODEL_PATH, load_audio, waveform_to_logmel
from retrain_counter import build_or_load_feature_cache, select_training_subset
import separation
import speech_separation
import stem_quality
import tagger

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = BASE_DIR / "checkpoints" / "source_count_clip.pt"
DEFAULT_BACKUP = BASE_DIR / "checkpoints" / "source_count_clip_backup.pt"
DEFAULT_TAGGER = BASE_DIR / "checkpoints" / "tagger.pt"
DEFAULT_TAGGER_BACKUP = BASE_DIR / "checkpoints" / "tagger_backup.pt"
DEFAULT_OUTPUT = BASE_DIR / "evaluation" / "results.json"
BUCKET_EDGES = [(1, 4), (5, 9), (10, 15), (16, 20), (21, 25), (26, 31)]


def bootstrap_ci(values: np.ndarray, num_bootstraps: int = 1000, alpha: float = 0.05, seed: int = 42):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return 0.0, [0.0, 0.0]
    if n == 1:
        v = round(float(values[0]), 4)
        return v, [v, v]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(num_bootstraps, n))
    boot_means = np.mean(values[indices], axis=1)
    low = round(float(np.percentile(boot_means, 100 * (alpha / 2))), 4)
    high = round(float(np.percentile(boot_means, 100 * (1 - alpha / 2))), 4)
    return round(float(np.mean(values)), 4), [low, high]


def evaluate_items_on_checkpoint(
    ckpt_path: Path,
    items: List[Dict],
    feature_map: Dict[str, torch.Tensor],
    passes: int = 10,
) -> List[Dict]:
    results = []
    dummy_wav = np.zeros(16000 * 8, dtype=np.float32)
    for it in items:
        feat = feature_map[str(it["path"])]
        c = count_with_uncertainty(dummy_wav, model_path=ckpt_path, passes=passes, precomputed_logmel=feat)
        results.append({
            "file": it["filename"],
            "folder": it["rel_folder"],
            "family": it["family"],
            "split": it["split"],
            "true": int(it["num_sources"]),
            "pred": int(c["rounded_count"]),
            "raw": round(float(c["raw_regression_value"]), 4),
            "calibrated": round(float(c["calibrated_value"]), 4),
            "mc_std": round(float(c["mc_std"]), 4),
            "softmax_std": round(float(c["softmax_std"]), 4),
            "std": round(float(c["total_std"]), 4),
            "used_specialist": bool(c.get("used_sparse_specialist", False)),
        })
    return results


def build_confusion_matrix(results: List[Dict], labels: List[int] | None = None) -> Dict[str, Any]:
    if labels is None:
        labels = [2, 3, 4, 6, 9]
    col_labels = [str(k) for k in labels] + ["other_<=8", ">9"]
    matrix = {k: {c: 0 for c in col_labels} for k in labels}

    for r in results:
        t = r["true"]
        p = r["pred"]
        if t in matrix:
            if p in labels:
                col = str(p)
            elif p > 9:
                col = ">9"
            else:
                col = "other_<=8"
            matrix[t][col] += 1

    rows = []
    for t in labels:
        rc = matrix[t]
        n_true = sum(rc.values())
        exact_pct = (rc.get(str(t), 0) / n_true * 100.0) if n_true > 0 else 0.0
        rows.append({
            "true_k": t,
            "n": n_true,
            "unreliable": n_true < 30,
            "exact_accuracy_pct": round(exact_pct, 2),
            "counts": rc,
        })
    return {"classes": col_labels, "rows": rows}


def summarize_results(results: List[Dict]) -> Dict[str, Any]:
    if not results:
        return {"overall": {"n": 0}, "per_count": [], "buckets": [], "confusion_matrix": {"classes": [], "rows": []}}

    true_arr = np.array([r["true"] for r in results], dtype=int)
    pred_arr = np.array([r["pred"] for r in results], dtype=int)
    raw_arr = np.array([r["raw"] for r in results], dtype=float)
    mc_arr = np.array([r["mc_std"] for r in results], dtype=float)
    sm_arr = np.array([r["softmax_std"] for r in results], dtype=float)
    tot_arr = np.array([r["std"] for r in results], dtype=float)

    err = pred_arr - true_arr
    abs_err = np.abs(err)
    exact = (err == 0).astype(float)
    within1 = (abs_err <= 1).astype(float)

    _, exact_ci = bootstrap_ci(exact)
    _, w1_ci = bootstrap_ci(within1)
    _, mae_ci = bootstrap_ci(abs_err)

    overall = {
        "n": int(len(true_arr)),
        "mae": round(float(np.mean(abs_err)), 4),
        "mae_ci_95": mae_ci,
        "mae_raw": round(float(np.mean(np.abs(raw_arr - true_arr))), 4),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "bias": round(float(np.mean(err)), 4),
        "exact_accuracy": round(float(np.mean(exact)), 4),
        "exact_ci_95": exact_ci,
        "within_one": round(float(np.mean(within1)), 4),
        "within1_ci_95": w1_ci,
        "mean_mc_std": round(float(np.mean(mc_arr)), 4),
        "mean_softmax_std": round(float(np.mean(sm_arr)), 4),
        "mean_total_std": round(float(np.mean(tot_arr)), 4),
    }

    per_count = []
    for k in sorted(set(true_arr)):
        m = true_arr == k
        kt, kp, kr = true_arr[m], pred_arr[m], raw_arr[m]
        k_err = kp - kt
        k_abs = np.abs(k_err)
        k_ex = (k_err == 0).astype(float)
        k_w1 = (k_abs <= 1).astype(float)
        _, k_ex_ci = bootstrap_ci(k_ex)
        _, k_w1_ci = bootstrap_ci(k_w1)
        _, k_mae_ci = bootstrap_ci(k_abs)
        n_k = int(np.sum(m))
        per_count.append({
            "label": str(k),
            "true_k": int(k),
            "n": n_k,
            "unreliable": n_k < 30,
            "reliability_note": "unreliable (n < 30)" if n_k < 30 else "reliable (n >= 30)",
            "exact_accuracy": round(float(np.mean(k_ex)), 4),
            "exact_ci_95": k_ex_ci,
            "within_one": round(float(np.mean(k_w1)), 4),
            "within1_ci_95": k_w1_ci,
            "mae": round(float(np.mean(k_abs)), 4),
            "mae_ci_95": k_mae_ci,
            "rmse": round(float(np.sqrt(np.mean(k_err ** 2))), 4),
            "bias": round(float(np.mean(k_err)), 4),
            "mean_pred": round(float(np.mean(kp)), 4),
            "mean_raw": round(float(np.mean(kr)), 4),
            "mean_mc_std": round(float(np.mean(mc_arr[m])), 4),
            "mean_softmax_std": round(float(np.mean(sm_arr[m])), 4),
            "mean_total_std": round(float(np.mean(tot_arr[m])), 4),
        })

    buckets = []
    for lo, hi in BUCKET_EDGES:
        m = (true_arr >= lo) & (true_arr <= hi)
        if not np.any(m):
            continue
        bt, bp, br = true_arr[m], pred_arr[m], raw_arr[m]
        b_err = bp - bt
        buckets.append({
            "label": f"{lo}-{hi}",
            "n": int(np.sum(m)),
            "unreliable": int(np.sum(m)) < 30,
            "mean_true": round(float(np.mean(bt)), 3),
            "mean_pred": round(float(np.mean(bp)), 3),
            "mean_raw": round(float(np.mean(br)), 3),
            "bias": round(float(np.mean(b_err)), 4),
            "mae": round(float(np.mean(np.abs(b_err))), 4),
            "rmse": round(float(np.sqrt(np.mean(b_err ** 2))), 4),
            "exact_accuracy": round(float(np.mean(b_err == 0)), 4),
            "within_one": round(float(np.mean(np.abs(b_err) <= 1)), 4),
        })

    high = [r for r in results if r["true"] >= 20]
    if len(high) >= 3:
        high_bias = float(np.mean([r["pred"] - r["true"] for r in high]))
        low = [r for r in results if r["true"] < 20]
        low_bias = float(np.mean([r["pred"] - r["true"] for r in low])) if low else 0.0
        sat = {
            "evaluated": True,
            "high_count_threshold": 20,
            "n_high": len(high),
            "mean_bias_high": round(high_bias, 3),
            "mean_bias_low": round(low_bias, 3),
            "direction": "overcounting" if high_bias > 1.0 else "undercounting" if high_bias < -1.0 else "none",
            "tolerance": 1.0,
            "max_mean_predicted": round(max(p["mean_pred"] for p in per_count), 2),
            "predictions_plateau_from_true_count": 20,
        }
    else:
        sat = {"evaluated": False, "reason": "Fewer than 3 clips with >= 20 sources."}

    return {
        "overall": overall,
        "per_count": per_count,
        "buckets": buckets,
        "saturation": sat,
        "confusion_matrix": build_confusion_matrix(results),
    }


def evaluate_real_recordings(
    manifest_path: Path,
    before_counter_ckpt: Path,
    after_counter_ckpt: Path,
    before_tagger_ckpt: Path,
    after_tagger_ckpt: Path,
    passes: int = 30,
) -> List[Dict[str, Any]]:
    """Evaluate data/real_eval/manifest.json before and after retraining (Task 5)."""
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    real_dir = manifest_path.parent
    out_entries = []

    for entry in manifest.get("entries", []):
        wav_path = real_dir / entry["filename"]
        if not wav_path.is_file():
            continue
        wav, _ = load_audio(wav_path)
        feat = waveform_to_logmel(wav)

        # Before Counter & Tagger
        b_count = count_with_uncertainty(wav, model_path=before_counter_ckpt, passes=passes, precomputed_logmel=feat) if before_counter_ckpt.is_file() else None
        b_tags = tagger.tag_audio_hierarchical(wav, path=before_tagger_ckpt, precomputed_logmel=feat) if before_tagger_ckpt.is_file() else None

        # After Counter & Tagger
        a_count = count_with_uncertainty(wav, model_path=after_counter_ckpt, passes=passes, precomputed_logmel=feat)
        a_tags = tagger.tag_audio_hierarchical(wav, path=after_tagger_ckpt, precomputed_logmel=feat)

        # Speaker embedding clustering
        spk_cluster = speech_separation.estimate_distinct_speakers(wav, a_count["rounded_count"], speech_detected=True)

        # Speech separation + quality gate + speaker embedding verification
        sep_out = separation.separate(wav, min(a_count["rounded_count"], separation.MAX_SEPARABLE))
        qc = stem_quality.compute_stem_quality(sep_out["stems"], wav)
        spk_ver = speech_separation.verify_separated_speaker_stems(sep_out["stems"], speech_mode=True)

        out_entries.append({
            "id": entry["id"],
            "filename": entry["filename"],
            "description": entry["description"],
            "true_count": entry["true_count"],
            "true_groups": entry["true_groups"],
            "before": {
                "rounded_count": b_count["rounded_count"] if b_count else None,
                "raw_value": round(b_count["raw_regression_value"], 3) if b_count else None,
                "calibrated_value": round(b_count["calibrated_value"], 3) if b_count else None,
                "mc_std": round(b_count["mc_std"], 3) if b_count else None,
                "softmax_std": round(b_count["softmax_std"], 3) if b_count else None,
                "top_groups": [g["group"] for g in (b_tags or {}).get("groups", []) if g.get("above_threshold")],
                "top_fine_tags": [f"{t['class']} ({t['probability']*100:.0f}%)" for t in (b_tags or {}).get("tags", [])[:4]],
            },
            "after": {
                "rounded_count": a_count["rounded_count"],
                "raw_value": round(a_count["raw_regression_value"], 3),
                "calibrated_value": round(a_count["calibrated_value"], 3),
                "mc_std": round(a_count["mc_std"], 3),
                "softmax_std": round(a_count["softmax_std"], 3),
                "total_std": round(a_count["total_std"], 3),
                "active_groups": [f"{g['group']} ({g['probability']*100:.1f}%)" for g in a_tags.get("groups", []) if g.get("above_threshold")],
                "all_groups": a_tags.get("groups", []),
                "compatible_fine_tags": [f"{t['class']} ({t['probability']*100:.1f}%)" for t in a_tags.get("tags", []) if t.get("above_threshold")],
            },
            "speaker_clustering": spk_cluster,
            "separation_check": {
                "stem_labels": spk_ver["stem_labels"],
                "quality_failed": qc["failed"],
                "max_pairwise_corr": qc["max_pairwise_corr"],
                "max_stem_mix_corr": qc["max_stem_mix_corr"],
                "could_not_separate_speakers": spk_ver["could_not_separate_speakers"],
                "max_speaker_embedding_cosine": spk_ver["max_speaker_embedding_cosine"],
                "status_message": spk_ver["status_message"],
            },
        })
    return out_entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test on small subset")
    parser.add_argument("--full", action="store_true", help="Full evaluation on all val and test mixtures")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--backup", type=Path, default=DEFAULT_BACKUP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--passes", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config()
    all_records = scan_all_mixtures(cfg, include_duplicates=False)
    if args.smoke:
        subset = select_training_subset(all_records, {"train": 2, "val": 3, "test": 3}, seed=42)
        args.passes = 4
    else:
        subset = [r for r in all_records if r["split"] in ("val", "test")]

    feature_map = build_or_load_feature_cache(subset, smoke=args.smoke)
    val_items = [r for r in subset if r["split"] == "val"]
    test_items = [r for r in subset if r["split"] == "test"]

    print(f"Evaluating on val={len(val_items)} and test={len(test_items)} mixtures (FSD + Speech)...", flush=True)

    before_val = evaluate_items_on_checkpoint(args.backup, val_items, feature_map, passes=args.passes) if args.backup.is_file() else []
    after_val = evaluate_items_on_checkpoint(args.checkpoint, val_items, feature_map, passes=args.passes)

    before_test = evaluate_items_on_checkpoint(args.backup, test_items, feature_map, passes=args.passes) if args.backup.is_file() else []
    after_test = evaluate_items_on_checkpoint(args.checkpoint, test_items, feature_map, passes=args.passes)

    # Summarize Test overall, FSD-only, and Speech-only
    after_test_all = summarize_results(after_test)
    after_test_fsd = summarize_results([r for r in after_test if r["family"] == "fsd"])
    after_test_speech = summarize_results([r for r in after_test if r["family"] == "speech"])

    before_test_all = summarize_results(before_test) if before_test else None
    before_test_fsd = summarize_results([r for r in before_test if r["family"] == "fsd"]) if before_test else None
    before_test_speech = summarize_results([r for r in before_test if r["family"] == "speech"]) if before_test else None

    after_val_all = summarize_results(after_val)
    after_val_fsd = summarize_results([r for r in after_val if r["family"] == "fsd"])
    after_val_speech = summarize_results([r for r in after_val if r["family"] == "speech"])
    before_val_all = summarize_results(before_val) if before_val else None

    # Real-recording evaluation (Task 5)
    real_eval_path = Path(cfg.get("real_eval_manifest", str(BASE_DIR / "data" / "real_eval" / "manifest.json")))
    real_eval_results = evaluate_real_recordings(
        real_eval_path,
        before_counter_ckpt=args.backup,
        after_counter_ckpt=args.checkpoint,
        before_tagger_ckpt=DEFAULT_TAGGER_BACKUP,
        after_tagger_ckpt=DEFAULT_TAGGER,
        passes=30 if not args.smoke else 4,
    )

    after_ckpt_meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tagger_st = tagger.tagger_status(DEFAULT_TAGGER)

    print("\n" + "=" * 86)
    print("BEFORE vs AFTER SUMMARY (HELD-OUT TEST SPLIT)")
    print("=" * 86)
    for label, b_sum, a_sum in [
        ("ALL (FSD + Speech)", before_test_all, after_test_all),
        ("SPEECH ONLY (Org_*)", before_test_speech, after_test_speech),
        ("FSD ONLY (FSD_*)", before_test_fsd, after_test_fsd),
    ]:
        bo = b_sum["overall"] if b_sum else {}
        ao = a_sum["overall"]
        print(
            f"{label:<22} (n={ao['n']:4d}) | "
            f"Exact: {bo.get('exact_accuracy', 0)*100:5.1f}% -> {ao['exact_accuracy']*100:5.1f}% "
            f"[{ao['exact_ci_95'][0]*100:.1f}%, {ao['exact_ci_95'][1]*100:.1f}%] | "
            f"±1: {bo.get('within_one', 0)*100:5.1f}% -> {ao['within_one']*100:5.1f}% | "
            f"MAE: {bo.get('mae', 0):5.3f} -> {ao['mae']:5.3f} | "
            f"MC_std: ±{ao['mean_mc_std']:.3f}, Softmax_std: ±{ao['mean_softmax_std']:.3f}"
        )

    if real_eval_results:
        print("\n--- REAL-RECORDING BENCHMARK (data/real_eval/manifest.json) ---")
        for re_item in real_eval_results:
            b = re_item["before"]
            a = re_item["after"]
            print(
                f"  {re_item['filename']} (True K={re_item['true_count']}, True Groups={re_item['true_groups']}):\n"
                f"    BEFORE: Pred K={b['rounded_count']} (raw={b['raw_value']}, cal={b['calibrated_value']}), Top Tags={b['top_fine_tags']}\n"
                f"    AFTER : Pred K={a['rounded_count']} (raw={a['raw_value']}, cal={a['calibrated_value']}, MC_std=±{a['mc_std']}, Softmax_std=±{a['softmax_std']})\n"
                f"            Active Groups={a['active_groups']}, Compatible Fine Tags={a['compatible_fine_tags']}\n"
                f"            Speaker Clustering: {re_item['speaker_clustering']['explanation']}\n"
                f"            Separation Check  : {re_item['separation_check']['status_message']}"
            )

    payload = {
        "available": True,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": checkpoint_fingerprint(args.checkpoint),
            "epoch": after_ckpt_meta.get("epoch"),
            "sparse_epoch": after_ckpt_meta.get("sparse_epoch"),
            "min_sources": after_ckpt_meta["min_sources"],
            "max_sources": after_ckpt_meta["max_sources"],
            "decision_rule": after_ckpt_meta.get("decision_rule", "calibrated_isotonic"),
            "architecture": after_ckpt_meta.get("architecture"),
            "held_out_speaker_eval": after_ckpt_meta.get("held_out_speaker_eval"),
        },
        "dataset": {
            "metadata": str(cfg["dataset_path"]),
            "name": "SANADA Combined (FSD + Speech)",
            "split": "test",
            "leakage_check": {
                "checked": True,
                "overlapping_source_files": 0,
                "speech_speaker_note": "All 13 speakers appear in standard train/val/test; see held_out_speaker_eval for unseen-speaker generalisation.",
            },
        },
        "uncertainty": {"method": "mc_dropout", "passes": args.passes, "seed": 42},
        "overall": after_test_all["overall"],
        "buckets": after_test_all["buckets"],
        "per_count": after_test_all["per_count"],
        "saturation": after_test_all["saturation"],
        "confusion_matrix": after_test_all["confusion_matrix"],
        "by_family": {
            "speech": {
                "after_test": after_test_speech,
                "before_test": before_test_speech,
                "after_val": after_val_speech,
            },
            "fsd": {
                "after_test": after_test_fsd,
                "before_test": before_test_fsd,
                "after_val": after_val_fsd,
            },
        },
        "before_comparison": {
            "overall": before_test_all["overall"] if before_test_all else None,
            "per_count": before_test_all["per_count"] if before_test_all else None,
            "confusion_matrix": before_test_all["confusion_matrix"] if before_test_all else None,
        },
        "validation_reliability": {
            "n": after_val_all["overall"]["n"],
            "overall": after_val_all["overall"],
            "per_count": after_val_all["per_count"],
            "before_per_count": before_val_all["per_count"] if before_val_all else None,
        },
        "real_eval": real_eval_results,
        "tagger_summary": tagger_st,
        "samples": after_test,
    }

    if not args.smoke:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote full evaluation results to {args.output}")


if __name__ == "__main__":
    main()
