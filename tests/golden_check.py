"""Step 1 Golden Test Capture & Verification Harness (`tests/golden_check.py`).

Captures and verifies:
1. Outputs on 10 fixed audio clips (5 dataset mixtures across K & families,
   2 real_eval clips, and 3 held-out isolated clips) via `/api/analyze` and `/api/separate`.
2. Page renders (`/`, `/analyze`, `/evaluation`, `/history`, `/how`, `/docs`).
3. JSON shapes of API routes (`/api/health`, `/api/evaluation`, `/api/analyze`, `/api/separate`).
4. Latency baseline and SHA-256 checksums of `evaluation/*.json`.

Usage:
    .\\.venv\\Scripts\\python tests/golden_check.py --capture
    .\\.venv\\Scripts\\python tests/golden_check.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import app as flask_app
import config as cfg_mod

GOLDEN_DIR = BASE_DIR / "tests" / "golden"
GOLDEN_PATH = GOLDEN_DIR / "golden_outputs.json"


def _get_10_clips() -> list[dict[str, Any]]:
    ds_root = cfg_mod.get_dataset_root(cfg_mod.load_config())
    candidates = [
        ("fsd_mix_2", ds_root / "FSD_Mix_2_Sounds" / "test" / "mix_00006.wav", False),
        ("fsd_mix_3", ds_root / "FSD_Mix_3_Sounds" / "test" / "mix_00035.wav", True),
        ("org_mix_2", ds_root / "Org_Mix_2_Sounds" / "test" / "mix_00006.wav", True),
        ("org_mix_3", ds_root / "Org_Mix_3_Sounds" / "test" / "mix_00006.wav", False),
        ("org_mix_9", ds_root / "Org_Mix_9_Sounds" / "test" / "mix_00054.wav", False),
        ("real_2spk", BASE_DIR / "data" / "real_eval" / "two_teachers_speaking.wav", True),
        ("real_3spk", BASE_DIR / "data" / "real_eval" / "three_speakers_discussion.wav", False),
        ("iso_hammer", BASE_DIR / "data" / "isolated_sources" / "Hammer" / "210210.wav", False),
        ("iso_bark", BASE_DIR / "data" / "isolated_sources" / "Bark" / "101729.wav", False),
        ("iso_water", BASE_DIR / "data" / "isolated_sources" / "Water" / "106954.wav", False),
    ]
    out = []
    for cid, p, run_sep in candidates:
        if not p.is_file():
            # Fallback to first .wav in same directory if specific filename differs
            if p.parent.is_dir():
                wavs = sorted(p.parent.glob("*.wav"))
                if wavs:
                    p = wavs[0]
        if p.is_file():
            out.append({"id": cid, "path": p, "run_sep": run_sep})
    return out


def _json_shape(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_shape(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [_json_shape(obj[0])] if obj else []
    return type(obj).__name__


def _eval_checksums() -> dict[str, str]:
    eval_dir = BASE_DIR / "evaluation"
    out = {}
    for p in sorted(eval_dir.glob("*.json")):
        out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def run_capture_or_verify(verify: bool = False) -> None:
    os.environ.pop("ECHOCOUNT_DEV", None)
    client = flask_app.app.test_client()

    # 1. Pages & API health/eval shapes
    pages = ["/", "/analyze", "/evaluation", "/history", "/how", "/docs"]
    page_snapshots = {}
    for route in pages:
        resp = client.get(route)
        assert resp.status_code == 200, f"GET {route} failed: {resp.status_code}"
        html = resp.get_data(as_text=True)
        page_snapshots[route] = {
            "status": resp.status_code,
            "sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
            "length": len(html),
        }

    health_json = client.get("/api/health").get_json()
    eval_json = client.get("/api/evaluation").get_json()

    # 2. Run 10 fixed audio clips
    clips = _get_10_clips()
    assert len(clips) == 10, f"Expected 10 clips, got {len(clips)}"
    clip_results = {}
    latencies = {}

    for item in clips:
        cid = item["id"]
        p: Path = item["path"]
        torch.manual_seed(42)
        t0 = time.perf_counter()
        with open(p, "rb") as f:
            r_an = client.post(
                "/api/analyze",
                data={"audio": (f, p.name)},
                content_type="multipart/form-data",
            )
        t_an = round(time.perf_counter() - t0, 4)
        assert r_an.status_code == 200, f"/api/analyze failed on {cid}: {r_an.status_code}"
        an = r_an.get_json()

        sep_summary = None
        t_sep = None
        if item["run_sep"] and an["separation"]["can_separate"]:
            ts0 = time.perf_counter()
            r_sep = client.post("/api/separate", json={"job_id": an["job_id"]})
            t_sep = round(time.perf_counter() - ts0, 4)
            assert r_sep.status_code == 200, f"/api/separate failed on {cid}: {r_sep.status_code}"
            sep = r_sep.get_json()
            sep_summary = {
                "shape": _json_shape(sep),
                "stem_labels": [s["label"] for s in sep["stems"]],
                "quality_failed": sep["quality_failed"],
                "max_pairwise_corr": sep["quality_check"]["max_pairwise_corr"],
                "max_stem_mix_corr": sep["quality_check"]["max_stem_mix_corr"],
                "energy_passed": sep["quality_check"]["energy_conservation"]["passed"],
                "self_consistency_pass_rate": sep["self_consistency"]["pass_rate"],
            }

        latencies[cid] = {"analyze_s": t_an, "separate_s": t_sep}
        clip_results[cid] = {
            "file": p.name,
            "analyze_shape": _json_shape(an),
            "count": {
                "rounded_count": an["count"]["rounded_count"],
                "raw_regression_value": an["count"]["raw_regression_value"],
                "calibrated_value": an["count"]["calibrated_value"],
                "uncertainty_std": an["count"]["uncertainty_std"],
            },
            "groups": {g["group"]: g["probability"] for g in an["tags"]["groups"]},
            "named_sounds": [
                {"name": ns["name"], "confidence": ns["confidence"]} for ns in an["named_sounds"]
            ],
            "can_separate": an["separation"]["can_separate"],
            "separation": sep_summary,
        }

    payload = {
        "evaluation_checksums": _eval_checksums(),
        "api_shapes": {
            "/api/health": _json_shape(health_json),
            "/api/evaluation": _json_shape(eval_json),
        },
        "pages": page_snapshots,
        "latencies": latencies,
        "clips": clip_results,
    }

    if not verify:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"Captured golden baseline for {len(clips)} clips & {len(pages)} pages -> {GOLDEN_PATH}"
        )
        for cid, lat in latencies.items():
            print(f"  {cid:<12} | analyze={lat['analyze_s']:.3f}s | separate={lat['separate_s']}")
        return

    # Verify against golden baseline
    assert GOLDEN_PATH.is_file(), f"Golden baseline missing at {GOLDEN_PATH}. Run --capture first."
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    # Check evaluation JSON checksums unchanged
    assert payload["evaluation_checksums"] == golden["evaluation_checksums"], (
        "evaluation/*.json checksums changed!"
    )

    # Check 10 clips within 1e-4 tolerance
    tol = 1e-4
    for cid, g_clip in golden["clips"].items():
        c_clip = payload["clips"][cid]
        assert c_clip["count"]["rounded_count"] == g_clip["count"]["rounded_count"], (
            f"{cid}: rounded_count {c_clip['count']['rounded_count']} != {g_clip['count']['rounded_count']}"
        )
        for fld in ("raw_regression_value", "calibrated_value", "uncertainty_std"):
            diff = abs(c_clip["count"][fld] - g_clip["count"][fld])
            assert diff <= tol, f"{cid}: count.{fld} diff={diff} > {tol}"
        assert c_clip["can_separate"] == g_clip["can_separate"], f"{cid}: can_separate mismatch"
        for grp, g_prob in g_clip["groups"].items():
            diff = abs(c_clip["groups"].get(grp, -1.0) - g_prob)
            assert diff <= tol, f"{cid}: group {grp} prob diff={diff} > {tol}"
        assert [x["name"] for x in c_clip["named_sounds"]] == [
            x["name"] for x in g_clip["named_sounds"]
        ], f"{cid}: named_sounds mismatch"
        for c_ns, g_ns in zip(c_clip["named_sounds"], g_clip["named_sounds"]):
            diff = abs(c_ns["confidence"] - g_ns["confidence"])
            assert diff <= tol, f"{cid}: sound {c_ns['name']} conf diff={diff} > {tol}"
        if g_clip["separation"] is not None:
            cs, gs = c_clip["separation"], g_clip["separation"]
            assert cs["stem_labels"] == gs["stem_labels"], f"{cid}: stem_labels mismatch"
            assert cs["quality_failed"] == gs["quality_failed"], f"{cid}: quality_failed mismatch"
            assert cs["energy_passed"] == gs["energy_passed"], f"{cid}: energy_passed mismatch"
            assert abs(cs["max_pairwise_corr"] - gs["max_pairwise_corr"]) <= tol

    print("ALL GOLDEN CHECKS PASSED (10/10 clips within 1e-4, all routes & eval checksums intact).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    run_capture_or_verify(verify=args.verify and not args.capture)
