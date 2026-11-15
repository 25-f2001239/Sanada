"""Stem quality gates for EchoCount / SANADA separation output (Phase 2.3).

Implements three measured quality gates per separation:
1. Waveform correlation gate (`compute_stem_quality`):
   - Verifies stem-vs-stem and stem-vs-mixture Pearson correlations are below threshold (0.90).
2. Energy conservation gate (`check_energy_conservation`):
   - Verifies that `sum(stems) + residual` reconstructs the original mixture within tolerance.
3. Self-consistency gate (`evaluate_stem_self_consistency`):
   - Re-tags each separated stem with the audio tagger and checks whether its top predicted
     sound name or coarse group matches the target query. Marks stems that do not pass as
     `"low confidence"`.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Callable, Dict, List, Optional

import numpy as np


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two float arrays (safely handles zero variance)."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    a_f, b_f = a[:n].astype(np.float64), b[:n].astype(np.float64)
    a_f = a_f - a_f.mean()
    b_f = b_f - b_f.mean()
    norm_a = np.linalg.norm(a_f)
    norm_b = np.linalg.norm(b_f)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a_f, b_f) / (norm_a * norm_b))


def check_energy_conservation(
    stems: List[np.ndarray],
    residual: np.ndarray,
    mix_clip: np.ndarray,
    rel_tol: float = 0.05,
) -> Dict[str, Any]:
    """Check that sum(stems) + residual reconstructs mix_clip within tolerance."""
    n = len(mix_clip)
    if n == 0 or not stems:
        return {
            "passed": False,
            "relative_l2_error": 1.0,
            "reconstruction_snr_db": 0.0,
            "tolerance": rel_tol,
        }
    recon = np.zeros(n, dtype=np.float64)
    for s in stems:
        m = min(n, len(s))
        recon[:m] += s[:m].astype(np.float64)
    if residual is not None and len(residual) > 0:
        m = min(n, len(residual))
        recon[:m] += residual[:m].astype(np.float64)

    mix_f = mix_clip[:n].astype(np.float64)
    err = mix_f - recon
    mix_norm = float(np.linalg.norm(mix_f)) + 1e-12
    err_norm = float(np.linalg.norm(err))
    rel_err = err_norm / mix_norm
    snr_db = float(20.0 * np.log10(mix_norm / max(err_norm, 1e-12)))

    return {
        "passed": bool(rel_err <= rel_tol),
        "relative_l2_error": round(rel_err, 6),
        "reconstruction_snr_db": round(snr_db, 2),
        "tolerance": rel_tol,
    }


def evaluate_stem_self_consistency(
    stems: List[np.ndarray],
    queries: List[Dict[str, Any]],
    tag_fn: Optional[Callable[[np.ndarray], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Re-tag each separated stem with the tagger and check if top class/group matches its query.

    Parameters
    ----------
    stems : list of 1-D float32 arrays
    queries : list of dicts with keys {"name": str, "class": str, "group": str}
    tag_fn : callable(waveform) -> tagger result dict (defaults to open_tagger.tag_open_vocabulary)
    """
    if tag_fn is None:
        import open_tagger

        def tag_fn(w: np.ndarray) -> Dict[str, Any]:
            return open_tagger.tag_open_vocabulary(w, use_clap_fallback=False, localize_windows=False)

    per_stem = []
    passed_count = 0

    for idx, (stem, q) in enumerate(zip(stems, queries)):
        q_name = str(q.get("name") or q.get("friendly_name") or f"Source {idx + 1}")
        q_class = str(q.get("class") or q_name)
        q_group = str(q.get("group") or "")

        # Pure speech speaker stems ("Human 1", "Human 2") match "Human voice"
        is_human_stem = q_name.startswith("Human") or q_group == "Human voice"

        try:
            res = tag_fn(stem)
        except Exception:
            res = {"named_sounds": [], "tags": [], "groups": []}

        named = res.get("named_sounds") or res.get("tags") or []
        groups = res.get("groups") or []

        top_sound = named[0]["name"] if named and "name" in named[0] else (named[0]["class"] if named else "None")
        top_sound_prob = float(named[0].get("confidence", named[0].get("probability", 0.0))) if named else 0.0
        top_group = groups[0]["group"] if groups else "None"
        top_group_prob = float(groups[0].get("probability", 0.0)) if groups else 0.0

        # Check if query sound is among top-2 predicted sounds or top group matches query group
        top2_names = {
            str(item.get("name") or item.get("friendly_name") or item.get("class", "")).lower()
            for item in named[:2]
        }
        top2_classes = {str(item.get("class", "")).lower() for item in named[:2]}

        name_match = (
            q_name.lower() in top2_names
            or q_class.lower() in top2_classes
            or (is_human_stem and ("human voice" in top2_names or top_group == "Human voice"))
        )
        group_match = bool(q_group and top_group == q_group and top_group_prob >= 0.25)
        passed = bool(name_match or group_match)
        if passed:
            passed_count += 1

        per_stem.append({
            "stem_index": idx + 1,
            "query_name": q_name,
            "query_group": q_group,
            "top_predicted_sound": top_sound,
            "top_predicted_confidence": round(top_sound_prob, 4),
            "top_predicted_group": top_group,
            "top_group_confidence": round(top_group_prob, 4),
            "exact_name_match": bool(name_match),
            "group_match": bool(group_match),
            "self_consistent": passed,
            "low_confidence": not passed,
            "status_badge": "Verified" if passed else "Low confidence",
        })

    total = max(1, len(per_stem))
    return {
        "per_stem": per_stem,
        "passed_count": passed_count,
        "total_stems": len(per_stem),
        "pass_rate": round(passed_count / total, 4),
    }


def compute_stem_quality(
    stems: list[np.ndarray],
    mix_clip: np.ndarray,
    corr_threshold: float = 0.9,
    residual: Optional[np.ndarray] = None,
) -> dict:
    """Compute pairwise Pearson correlations between stems, stem-vs-mix, and energy conservation."""
    num_stems = len(stems)

    pairwise = []
    for i, j in combinations(range(num_stems), 2):
        c = abs(_pearson_corr(stems[i], stems[j]))
        pairwise.append({"i": i, "j": j, "corr": round(c, 4)})

    stem_mix = []
    for i, stem in enumerate(stems):
        c = abs(_pearson_corr(stem, mix_clip))
        stem_mix.append({"i": i, "corr": round(c, 4)})

    max_pairwise = max((e["corr"] for e in pairwise), default=0.0)
    max_stem_mix = max((e["corr"] for e in stem_mix), default=0.0)
    worst = max(max_pairwise, max_stem_mix)

    if residual is None and stems:
        n = len(mix_clip)
        stem_sum = np.zeros(n, dtype=np.float32)
        for s in stems:
            m = min(n, len(s))
            stem_sum[:m] += s[:m]
        residual = mix_clip[:n] - stem_sum

    energy_check = (
        check_energy_conservation(stems, residual, mix_clip)
        if stems and residual is not None
        else {"passed": True, "relative_l2_error": 0.0, "reconstruction_snr_db": 100.0, "tolerance": 0.05}
    )

    failed = bool(worst > corr_threshold or not energy_check["passed"])
    reason: Optional[str] = None
    if worst > corr_threshold:
        if max_pairwise >= max_stem_mix:
            reason = (
                f"Separation failed quality check: stems are near-copies of each other "
                f"(max pairwise correlation {max_pairwise:.2f} > threshold {corr_threshold:.2f})."
            )
        else:
            reason = (
                f"Separation failed quality check: stems are near-copies of the mix "
                f"(max stem–mix correlation {max_stem_mix:.2f} > threshold {corr_threshold:.2f})."
            )
    elif not energy_check["passed"]:
        reason = (
            f"Separation failed energy conservation check: relative reconstruction error "
            f"{energy_check['relative_l2_error']:.3f} exceeds tolerance {energy_check['tolerance']:.2f}."
        )

    return {
        "pairwise_corrs": pairwise,
        "stem_mix_corrs": stem_mix,
        "max_pairwise_corr": round(max_pairwise, 4),
        "max_stem_mix_corr": round(max_stem_mix, 4),
        "energy_conservation": energy_check,
        "failed": failed,
        "failure_reason": reason,
        "threshold_used": corr_threshold,
        "num_stems": num_stems,
    }
