"""Data-driven capability registry for EchoCount separation & counting limits (Part C)."""

from __future__ import annotations

import json
from typing import Any

from config import BASE_DIR

REGISTRY_PATH = BASE_DIR / "capabilities.json"

# Policy thresholds (C3): data-driven, never hand-waved
ENABLED_MIN_SISDRI_DB = 8.0
ENABLED_MIN_SELF_CONSISTENCY = 0.80
EXPERIMENTAL_MIN_SISDRI_DB = 3.0


def classify_k_status(
    mean_sisdri_db: float | None,
    self_consistency: float | None,
    *,
    available: bool = True,
) -> str:
    """Classify a (mode, K) cell strictly from measured held-out metrics."""
    if not available or mean_sisdri_db is None:
        return "disabled"
    sc = 0.0 if self_consistency is None else float(self_consistency)
    s = float(mean_sisdri_db)
    if s >= ENABLED_MIN_SISDRI_DB and sc >= ENABLED_MIN_SELF_CONSISTENCY:
        return "enabled"
    if s >= EXPERIMENTAL_MIN_SISDRI_DB:
        return "experimental"
    return "disabled"


def _default_registry() -> dict[str, Any]:
    """Conservative fallback if capabilities.json has not been generated yet."""
    specs = [
        (
            "speech-2",
            "2-speaker speech (SepFormer WSJ0-2mix)",
            "speech",
            "speechbrain/sepformer-wsj02mix",
            "Apache-2.0",
            "enabled",
            [2],
        ),
        (
            "speech-3",
            "3-speaker speech (SepFormer WSJ0-3mix)",
            "speech",
            "speechbrain/sepformer-wsj03mix",
            "Apache-2.0",
            "disabled",
            [],
        ),
        (
            "speech-recursive",
            "Recursive speech separation (up to 8 speakers)",
            "speech",
            "sepformer-wsj02mix + sepformer-wsj03mix (recursive)",
            "Apache-2.0",
            "disabled",
            [],
        ),
        (
            "sound-query",
            "Text-queried general sound separation (up to 8)",
            "general",
            "No licensed pretrained AudioSep; evaluated QueryConditionedWiener",
            "N/A",
            "disabled",
            [],
        ),
    ]
    modes = {
        mid: {
            "label": lbl,
            "family": fam,
            "model": mdl,
            "licence": lic,
            "status": st,
            "enabled_k": en_k,
            "experimental_k": [],
            "per_k": {},
            **({"pretrained_available": False} if mid == "sound-query" else {}),
        }
        for mid, lbl, fam, mdl, lic, st, en_k in specs
    }
    return {
        "policy": {
            "enabled_min_sisdri_db": ENABLED_MIN_SISDRI_DB,
            "enabled_min_self_consistency": ENABLED_MIN_SELF_CONSISTENCY,
            "experimental_min_sisdri_db": EXPERIMENTAL_MIN_SISDRI_DB,
        },
        "benchmark_meta": {"status": "not_benchmarked"},
        "modes": modes,
        "counting_limits": {"reliable_sparse_max_k": 6, "warning_above_k": 6},
    }


def load_registry() -> dict[str, Any]:
    """Load capabilities.json and enrich with computed UI summary fields."""
    data = _default_registry()
    if REGISTRY_PATH.is_file():
        try:
            loaded = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, ValueError):
            pass

    modes = data.get("modes") or {}
    enabled_speech: set[int] = set()
    exp_speech: set[int] = set()
    for mid in ("speech-2", "speech-3", "speech-recursive"):
        m = modes.get(mid) or {}
        enabled_speech.update(int(k) for k in (m.get("enabled_k") or []))
        exp_speech.update(int(k) for k in (m.get("experimental_k") or []))
    exp_speech -= enabled_speech

    max_en = max(enabled_speech) if enabled_speech else 2
    max_exp = max(exp_speech) if exp_speech else 0

    sq = modes.get("sound-query") or {}
    sq_en = sorted(int(k) for k in (sq.get("enabled_k") or []))
    sq_exp = sorted(int(k) for k in (sq.get("experimental_k") or []))

    if max_exp > max_en:
        headline_detail = (
            f"Verified speech separation up to {max_en} speakers "
            f"(experimental up to {max_exp}; policy-gated)"
        )
    else:
        headline_detail = f"Verified speech separation up to {max_en} speakers (SepFormer + ECAPA)"

    data["ui_summary"] = {
        "max_enabled_speech_k": max_en,
        "max_experimental_speech_k": max_exp,
        "enabled_speech_k": sorted(enabled_speech),
        "experimental_speech_k": sorted(exp_speech),
        "sound_query_status": sq.get("status", "disabled"),
        "sound_query_enabled_k": sq_en,
        "sound_query_experimental_k": sq_exp,
        "headline_speech_count": max_en,
        "headline_speech_detail": headline_detail,
    }
    return data


def get_extended_speech_modes_for_k(k: int, is_pure_speech: bool) -> list[dict[str, Any]]:
    """Return additive speech separation options allowed by the registry for this K."""
    if not is_pure_speech or k < 2 or k > 8:
        return []
    reg = load_registry()
    modes = reg.get("modes") or {}
    options: list[dict[str, Any]] = []

    if k == 3:
        m3 = modes.get("speech-3") or {}
        cell = (m3.get("per_k") or {}).get("3") or {}
        st = cell.get("status") or m3.get("status", "disabled")
        if st in ("enabled", "experimental"):
            options.append(
                {
                    "mode": "speech-3",
                    "status": st,
                    "label": m3.get("label", "3-speaker speech (SepFormer-3mix)"),
                    "mean_sisdri_db": cell.get("mean_sisdri_db"),
                    "self_consistency": cell.get("self_consistency"),
                }
            )

    if k >= 3:
        mr = modes.get("speech-recursive") or {}
        cell = (mr.get("per_k") or {}).get(str(k)) or {}
        st = cell.get("status") or "disabled"
        if st in ("enabled", "experimental"):
            options.append(
                {
                    "mode": "speech-recursive",
                    "status": st,
                    "label": f"Recursive speech separation (K={k}, {st})",
                    "mean_sisdri_db": cell.get("mean_sisdri_db"),
                    "self_consistency": cell.get("self_consistency"),
                }
            )
    return options
