"""Unified EchoCount evaluation and profiling CLI (`counter`, `benchmarks`, `profile`, `all`)."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import audio
import quality
import separation
import tagging
from config import (
    DATA_DIR,
    MODEL_PATH,
    SAMPLE_RATE,
    get_friendly_name,
    get_group_for_class,
    load_taxonomy,
)
from counting import checkpoint_fingerprint, count_with_uncertainty
from data import scan_all_mixtures


def _sisdr_np(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    n = min(len(pred), len(target))
    p, t = (
        pred[:n].astype(np.float64) - np.mean(pred[:n]),
        target[:n].astype(np.float64) - np.mean(target[:n]),
    )
    alpha = float(np.sum(p * t)) / (float(np.sum(t**2)) + eps)
    return float(
        10.0
        * np.log10(
            (float(np.sum((alpha * t) ** 2)) + eps) / (float(np.sum((p - alpha * t) ** 2)) + eps)
            + eps
        )
    )


def _bootstrap_ci(vals: np.ndarray, stat_fn, n_boot: int = 500, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    stats = [float(stat_fn(vals[rng.integers(0, len(vals), len(vals))])) for _ in range(n_boot)]
    return [round(float(np.percentile(stats, 2.5)), 4), round(float(np.percentile(stats, 97.5)), 4)]


def eval_counter(smoke: bool = False) -> dict[str, Any]:
    """Evaluate SourceCountCNN on the held-out test split with bootstrap 95% CIs."""
    test_recs = [r for r in scan_all_mixtures(include_duplicates=False) if r["split"] == "test"]
    if smoke:
        test_recs = test_recs[:40]
    errs, exacts, w1s = [], [], []
    for r in test_recs:
        wav, _ = audio.load_audio(Path(r["path"]))
        pred = count_with_uncertainty(wav, MODEL_PATH, passes=10)["rounded_count"]
        err = abs(pred - int(r["num_sources"]))
        errs.append(err)
        exacts.append(int(err == 0))
        w1s.append(int(err <= 1))
    e_arr, ex_arr, w_arr = (
        np.array(errs, dtype=np.float64),
        np.array(exacts, dtype=np.float64),
        np.array(w1s, dtype=np.float64),
    )
    summary = {
        "n_test": len(test_recs),
        "mae": round(float(np.mean(e_arr)), 4),
        "mae_ci95": _bootstrap_ci(e_arr, np.mean),
        "exact_accuracy": round(float(np.mean(ex_arr)), 4),
        "exact_ci95": _bootstrap_ci(ex_arr, np.mean),
        "within_one": round(float(np.mean(w_arr)), 4),
        "checkpoint": {"sha256": checkpoint_fingerprint(MODEL_PATH)},
    }
    print(
        f"[Eval Counter] n={summary['n_test']} | MAE={summary['mae']} | Exact={summary['exact_accuracy'] * 100:.1f}% | Within-1={summary['within_one'] * 100:.1f}%"
    )
    return summary


def eval_benchmarks(smoke: bool = False) -> dict[str, Any]:
    """Run Phase 1-3 product benchmarks on held-out FSD50K test clips and real_eval."""
    iso_dir = DATA_DIR / "isolated_sources"
    mpath = iso_dir / "manifest.json"
    if not mpath.is_file():
        return {"available": False}
    mdata = json.loads(mpath.read_text(encoding="utf-8"))
    test_clips = [c for c in mdata.get("clips", []) if c.get("split") == "test"]
    if smoke:
        test_clips = test_clips[:16]
    tax = load_taxonomy()
    classes = sorted({c["class"] for c in test_clips})
    y_true, y_comb = {c: [] for c in classes}, {c: [] for c in classes}

    for item in test_clips:
        p = iso_dir / item["rel_path"]
        if not p.is_file():
            continue
        samples, sr = audio.decode_mono(p, 30.0)
        y16 = audio.resample(samples, sr, SAMPLE_RATE)
        y16 = (
            y16[: SAMPLE_RATE * 8]
            if len(y16) >= SAMPLE_RATE * 8
            else np.pad(y16, (0, SAMPLE_RATE * 8 - len(y16)))
        ).astype(np.float32)
        res = tagging.tag_open_vocabulary(
            y16, sr=SAMPLE_RATE, use_clap_fallback=True, localize_windows=False
        )
        probs = res.get("all_probabilities", {})
        for c in classes:
            y_true[c].append(int(c == item["class"]))
            y_comb[c].append(probs.get(get_friendly_name(c, tax), 0.0))

    aps = [
        float(average_precision_score(y_true[c], y_comb[c]))
        for c in classes
        if sum(y_true[c]) > 0 and max(y_comb[c]) > 0
    ]
    f1s = [
        float(f1_score(y_true[c], (np.array(y_comb[c]) >= 0.26).astype(int), zero_division=0))
        for c in classes
        if sum(y_true[c]) > 0
    ]

    # Synthetic K=2..5 separation benchmark on held-out test clips
    by_cls: dict[str, list[Path]] = {}
    for c in test_clips:
        p = iso_dir / c["rel_path"]
        if p.is_file():
            by_cls.setdefault(c["class"], []).append(p)
    rng, sisdris, sc_pass, sc_tot = random.Random(123), [], 0, 0
    n_mix = 4 if smoke else 16
    avail_classes = sorted(by_cls.keys())
    if len(avail_classes) >= 2:
        for m_idx in range(n_mix):
            k = min(2 + (m_idx % 4), len(avail_classes))
            true_stems, queries = [], []
            for cls in rng.sample(avail_classes, k):
                s, sr = audio.decode_mono(rng.choice(by_cls[cls]), 30.0)
                y16 = audio.resample(s, sr, SAMPLE_RATE)
                y16 = (
                    y16[: SAMPLE_RATE * 6]
                    if len(y16) >= SAMPLE_RATE * 6
                    else np.pad(y16, (0, SAMPLE_RATE * 6 - len(y16)))
                )
                scaled = (
                    (y16 / (float(np.sqrt(np.mean(y16**2))) + 1e-6))
                    * 0.12
                    * (10.0 ** (-rng.uniform(0, 5) / 20.0))
                )
                true_stems.append(scaled.astype(np.float32))
                fn = get_friendly_name(cls, tax)
                queries.append(
                    {
                        "name": fn,
                        "friendly_name": fn,
                        "class": cls,
                        "group": get_group_for_class(cls, tax),
                        "confidence": 0.85,
                        "time_ranges": [{"start": 0.0, "end": 6.0}],
                    }
                )
            mix = np.sum(np.stack(true_stems, axis=0), axis=0)
            scale = 0.89 / max(float(np.max(np.abs(mix))), 1e-6)
            mix = (mix * scale).astype(np.float32)
            true_stems = [(s * scale).astype(np.float32) for s in true_stems]
            sep_out = separation.separate_by_queries(
                mix, k=k, named_sounds=queries, speech_mode=False, sr=SAMPLE_RATE
            )
            sc = quality.evaluate_stem_self_consistency(sep_out["stems"], queries)
            sc_pass += sc["passed_count"]
            sc_tot += sc["total_stems"]
            for ps, ts in zip(sep_out["stems"], true_stems):
                sisdris.append(_sisdr_np(ps, ts) - _sisdr_np(mix, ts))

    out = {
        "tagger_macro_ap": round(float(np.mean(aps)) if aps else 0.0, 4),
        "tagger_macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 4),
        "mean_sisdri_db": round(float(np.mean(sisdris)) if sisdris else 0.0, 2),
        "self_consistency_pass_rate": round(sc_pass / max(1, sc_tot), 4),
    }
    print(
        f"[Eval Benchmarks] Tagger mAP={out['tagger_macro_ap']} | Mean SI-SDRi={out['mean_sisdri_db']:+.2f} dB | Self-Consistency={out['self_consistency_pass_rate'] * 100:.1f}%"
    )
    return out


def profile_latency() -> dict[str, Any]:
    """Profile analyze and separation latency on an 8-second and 60-second clip."""
    ref = DATA_DIR / "real_eval" / "two_teachers_speaking.wav"
    samples, sr = audio.decode_mono(ref, 60.0)
    y8 = audio.resample(samples, sr, SAMPLE_RATE)[: SAMPLE_RATE * 8].astype(np.float32)
    if len(y8) < SAMPLE_RATE * 8:
        y8 = np.pad(y8, (0, SAMPLE_RATE * 8 - len(y8)))
    y60 = np.tile(y8, 8)[: SAMPLE_RATE * 60].astype(np.float32)

    # Warmup
    lm = audio.waveform_to_logmel(y8)
    _ = count_with_uncertainty(y8, MODEL_PATH, 10, precomputed_logmel=lm)
    _ = tagging.tag_open_vocabulary(
        y8, sr=SAMPLE_RATE, precomputed_logmel=lm, use_clap_fallback=True, localize_windows=False
    )

    res = {}
    for label, wav in (("clip_8s", y8), ("clip_60s", y60)):
        t0 = time.perf_counter()
        lm = audio.waveform_to_logmel(wav[: SAMPLE_RATE * 8])
        _ = count_with_uncertainty(wav[: SAMPLE_RATE * 8], MODEL_PATH, 30, precomputed_logmel=lm)
        _ = tagging.tag_open_vocabulary(
            wav[: SAMPLE_RATE * 8],
            sr=SAMPLE_RATE,
            precomputed_logmel=lm,
            use_clap_fallback=True,
            localize_windows=True,
        )
        t_an = round(time.perf_counter() - t0, 3)

        t1 = time.perf_counter()
        qs = [
            {"name": "Cello", "group": "Musical instrument", "time_ranges": []},
            {"name": "Scissors", "group": "Household/tools", "time_ranges": []},
        ]
        _ = separation.separate_by_queries(
            wav, k=2, named_sounds=qs, speech_mode=False, sr=SAMPLE_RATE
        )
        t_sep = round(time.perf_counter() - t1, 3)
        res[label] = {"analyze_s": t_an, "separate_s": t_sep}
        print(f"[Profile {label}] analyze={t_an:.3f}s | separate={t_sep:.3f}s")
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="target", required=True)
    for name in ("counter", "benchmarks", "profile", "all"):
        sp = sub.add_parser(name)
        sp.add_argument("--smoke", action="store_true")
        sp.add_argument("--full", action="store_true")
    args = p.parse_args()
    smoke = bool(args.smoke and not args.full)
    if args.target in ("counter", "all"):
        eval_counter(smoke=smoke)
    if args.target in ("benchmarks", "all"):
        eval_benchmarks(smoke=smoke)
    if args.target in ("profile", "all"):
        profile_latency()


if __name__ == "__main__":
    main()
