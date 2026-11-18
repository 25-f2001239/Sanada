"""Stem quality gates: waveform correlation, energy conservation, and self-consistency re-tagging."""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import Any

import numpy as np


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute absolute-safe Pearson correlation between two 1-D arrays."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    af, bf = a[:n].astype(np.float64), b[:n].astype(np.float64)
    af, bf = af - af.mean(), bf - bf.mean()
    na, nb = np.linalg.norm(af), np.linalg.norm(bf)
    return 0.0 if (na < 1e-12 or nb < 1e-12) else float(np.dot(af, bf) / (na * nb))


def check_energy_conservation(
    stems: list[np.ndarray],
    residual: np.ndarray | None,
    mix_clip: np.ndarray,
    rel_tol: float = 0.05,
) -> dict[str, Any]:
    """Verify sum(stems) + residual reconstructs mix_clip within relative L2 tolerance."""
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
    mix_norm = float(np.linalg.norm(mix_f)) + 1e-12
    err_norm = float(np.linalg.norm(mix_f - recon))
    rel_err = err_norm / mix_norm
    snr_db = float(20.0 * np.log10(mix_norm / max(err_norm, 1e-12)))
    return {
        "passed": bool(rel_err <= rel_tol),
        "relative_l2_error": round(rel_err, 6),
        "reconstruction_snr_db": round(snr_db, 2),
        "tolerance": rel_tol,
    }


def evaluate_stem_self_consistency(
    stems: list[np.ndarray],
    queries: list[dict[str, Any]],
    tag_fn: Callable[[np.ndarray], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Re-tag each separated stem and check if its top predicted sound/group matches the query."""
    if tag_fn is None:
        import tagging

        def tag_fn(w: np.ndarray) -> dict[str, Any]:
            return tagging.tag_open_vocabulary(w, use_clap_fallback=False, localize_windows=False)

    per_stem, passed_count = [], 0
    for idx, (stem, q) in enumerate(zip(stems, queries)):
        q_name = str(q.get("name") or q.get("friendly_name") or f"Source {idx + 1}")
        q_class, q_group = str(q.get("class") or q_name), str(q.get("group") or "")
        is_human = q_name.startswith("Human") or q_group == "Human voice"
        try:
            res = tag_fn(stem)
        except Exception:
            res = {"named_sounds": [], "tags": [], "groups": []}
        named = res.get("named_sounds") or res.get("tags") or []
        groups = res.get("groups") or []
        top_sound = named[0].get("name", named[0].get("class", "None")) if named else "None"
        top_prob = (
            float(named[0].get("confidence", named[0].get("probability", 0.0))) if named else 0.0
        )
        top_grp = groups[0]["group"] if groups else "None"
        top_grp_prob = float(groups[0].get("probability", 0.0)) if groups else 0.0

        top2_names = {
            str(x.get("name") or x.get("friendly_name") or x.get("class", "")).lower()
            for x in named[:2]
        }
        top2_cls = {str(x.get("class", "")).lower() for x in named[:2]}
        name_ok = (
            q_name.lower() in top2_names
            or q_class.lower() in top2_cls
            or (is_human and ("human voice" in top2_names or top_grp == "Human voice"))
        )
        grp_ok = bool(q_group and top_grp == q_group and top_grp_prob >= 0.25)
        passed = bool(name_ok or grp_ok)
        passed_count += int(passed)
        per_stem.append(
            {
                "stem_index": idx + 1,
                "query_name": q_name,
                "query_group": q_group,
                "top_predicted_sound": top_sound,
                "top_predicted_confidence": round(top_prob, 4),
                "top_predicted_group": top_grp,
                "top_group_confidence": round(top_grp_prob, 4),
                "exact_name_match": bool(name_ok),
                "group_match": bool(grp_ok),
                "self_consistent": passed,
                "low_confidence": not passed,
                "status_badge": "Verified" if passed else "Low confidence",
            }
        )

    return {
        "per_stem": per_stem,
        "passed_count": passed_count,
        "total_stems": len(per_stem),
        "pass_rate": round(passed_count / max(1, len(per_stem)), 4),
    }


def compute_stem_quality(
    stems: list[np.ndarray],
    mix_clip: np.ndarray,
    corr_threshold: float = 0.9,
    residual: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute pairwise stem correlations, stem-vs-mix correlations, and energy conservation."""
    pairwise = [
        {"i": i, "j": j, "corr": round(abs(_pearson_corr(stems[i], stems[j])), 4)}
        for i, j in combinations(range(len(stems)), 2)
    ]
    stem_mix = [
        {"i": i, "corr": round(abs(_pearson_corr(s, mix_clip)), 4)} for i, s in enumerate(stems)
    ]
    max_pw = max((e["corr"] for e in pairwise), default=0.0)
    max_sm = max((e["corr"] for e in stem_mix), default=0.0)

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
        else {
            "passed": True,
            "relative_l2_error": 0.0,
            "reconstruction_snr_db": 100.0,
            "tolerance": 0.05,
        }
    )

    worst = max(max_pw, max_sm)
    failed = bool(worst > corr_threshold or not energy_check["passed"])
    reason = None
    if worst > corr_threshold:
        if max_pw >= max_sm:
            reason = f"Separation failed quality check: stems are near-copies of each other (max pairwise correlation {max_pw:.2f} > threshold {corr_threshold:.2f})."
        else:
            reason = f"Separation failed quality check: stems are near-copies of the mix (max stem–mix correlation {max_sm:.2f} > threshold {corr_threshold:.2f})."
    elif not energy_check["passed"]:
        reason = f"Separation failed energy conservation check: relative reconstruction error {energy_check['relative_l2_error']:.3f} exceeds tolerance {energy_check['tolerance']:.2f}."

    return {
        "pairwise_corrs": pairwise,
        "stem_mix_corrs": stem_mix,
        "max_pairwise_corr": round(max_pw, 4),
        "max_stem_mix_corr": round(max_sm, 4),
        "energy_conservation": energy_check,
        "failed": failed,
        "failure_reason": reason,
        "threshold_used": corr_threshold,
        "num_stems": len(stems),
    }
