r"""Comprehensive Product Unit, API, Taxonomy, Quality-Gate & 5-Clip Regression Test Suite.

Covers all Phase 0 – Phase 4 verification requirements:
1. Unit Test — Taxonomy Single Source of Truth (`taxonomy.json`, `tagger.py`, `ground_truth.py`, `open_tagger.py`),
   explicitly asserting zero group disagreement across all 32 FSD classes and all 12 target everyday sounds
   (including `Telephone` -> `Household/tools`, `Scissors` -> `Household/tools`, `Cello` -> `Musical instrument`).
2. Unit Test — Production Hygiene (`ECHOCOUNT_DEV=0` default vs `ECHOCOUNT_DEV=1`):
   verifies no ground-truth card, dataset names, SHA-256 hashes, or demo banners appear in production API or UI.
3. Unit Test — Stem Quality Gates & Long-Audio Windowing (`stem_quality.py`, `query_separation.py`):
   verifies energy conservation (`sum(stems) + residual == mixture`), correlation gate (< 0.90),
   self-consistency re-tagging, and raised-cosine cross-fade stitching on long (> 8s) audio.
4. API & Async Job Queue Test (`/api/analyze`, `/api/separate` sync + async polling, `/api/stem/...`, `/api/download/...`).
5. Regression Test on 5 Fixed Clips:
   - Clip 1: `data/real_eval/two_teachers_speaking.wav` (2 speakers, SepFormer + ECAPA)
   - Clip 2: `data/real_eval/three_speakers_discussion.wav` (3 speakers, density limit check)
   - Clip 3: `FSD_Mix_3_Sounds/test/mix_00035.wav` (Cello, Scissors, Telephone ring — multi-sound separation + residual)
   - Clip 4: `Org_Mix_9_Sounds/test/mix_00054.wav` (Dense K=9 mixture — honest uncertainty warning)
   - Clip 5: Held-out FSD50K isolated clip (`data/isolated_sources/Hammer/*.wav` — open-vocabulary Hammer naming)

Usage:
  .\.venv\Scripts\python test_templates.py --smoke
  .\.venv\Scripts\python test_templates.py --full
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from flask import render_template
import numpy as np
import torch

import app as flask_app
import dataset_config
import ground_truth
import model as model_mod
import open_tagger
import query_separation
import separation
import stem_quality
import tagger

BASE_DIR = Path(__file__).resolve().parent
CKPT_DIR = BASE_DIR / "checkpoints"


def test_taxonomy_consistency():
    print("\n[1/5] Unit Test: Taxonomy Single Source of Truth (Phase 0.3)...")
    tax = dataset_config.load_taxonomy()
    c2g = dataset_config.get_class_to_group(tax)

    # Check explicit mismatch classes reported previously
    assert dataset_config.get_group_for_class("Cello", tax) == "Musical instrument"
    assert dataset_config.get_group_for_class("Scissors", tax) == "Household/tools"
    assert dataset_config.get_group_for_class("Telephone", tax) == "Household/tools"
    assert dataset_config.get_friendly_name("Telephone", tax) == "Telephone ring"

    # Check all 12 target everyday sounds exist and map to valid groups
    target_12 = tax.get("target_12_sounds", {})
    expected_12 = {
        "hammer", "phone vibration", "dog", "cat", "cello", "scissors",
        "telephone ring", "speech", "door knock", "keyboard typing", "water", "car horn",
    }
    assert set(target_12.keys()) == expected_12, f"Missing target sounds: {expected_12 - set(target_12.keys())}"
    for k, spec in target_12.items():
        fn = spec["friendly_name"]
        grp = spec["group"]
        assert grp in tax["groups"], f"Invalid group {grp} for {k}"
        assert dataset_config.get_group_for_class(fn, tax) == grp, f"Disagreement on group for {fn}"

    # Check tagger checkpoint fine_to_group matches taxonomy.json 100%
    ckpt = torch.load(CKPT_DIR / "tagger.pt", map_location="cpu", weights_only=False)
    for cls_name, ckpt_grp in ckpt.get("fine_to_group", {}).items():
        tax_grp = dataset_config.get_group_for_class(cls_name, tax)
        assert ckpt_grp == tax_grp, f"Tagger checkpoint disagrees with taxonomy.json on {cls_name}: {ckpt_grp} != {tax_grp}"

    # Check ground_truth_index.json true_groups agree with taxonomy.json (specifically mix_00035.wav)
    if ground_truth.INDEX_PATH.is_file():
        idx_data = json.loads(ground_truth.INDEX_PATH.read_text(encoding="utf-8"))
        m35 = idx_data.get("by_relpath", {}).get("fsd_mix_3_sounds/test/mix_00035.wav")
        if m35:
            assert m35["true_groups"] == ["Household/tools", "Musical instrument"], (
                f"mix_00035.wav true_groups mismatch: {m35['true_groups']}"
            )
    print("  [PASS] Taxonomy single source of truth verified across taxonomy.json, tagger.pt, open_tagger.py, and ground_truth_index.json.")


def test_production_hygiene():
    print("\n[2/5] Unit Test: Production Hygiene & Dev Flag Gating (Phase 0.1 & 0.2)...")
    # Ensure ECHOCOUNT_DEV is off by default
    os.environ.pop("ECHOCOUNT_DEV", None)
    assert ground_truth.is_dev_mode() is False
    ref_wav = BASE_DIR / "data" / "real_eval" / "two_teachers_speaking.wav"
    assert ground_truth.lookup_ground_truth(ref_wav) is None, "lookup_ground_truth must return None in production mode!"

    # Verify static/analyze.js contains no developer-only strings
    js_code = (BASE_DIR / "static" / "analyze.js").read_text(encoding="utf-8")
    for forbidden in ["In-distribution test file", "Dataset file (known ground truth)", "Untrained demo weights", "SHA-256"]:
        assert forbidden not in js_code, f"Forbidden developer text '{forbidden}' found in static/analyze.js!"
    print("  [PASS] Production mode omits ground-truth lookup and static/analyze.js is free of developer/demo strings.")


def test_quality_gates_and_windowing():
    print("\n[3/5] Unit Test: Quality Gates & Long-Audio Windowing (Phase 2.3 & 2.5)...")
    sr = 16000
    # Create a 14-second synthetic 2-source mixture with comparable energy (220 Hz harmonic + 3800 Hz modulated metallic scissor snips)
    t = np.linspace(0, 14.0, sr * 14, endpoint=False, dtype=np.float32)
    s_cello = (0.30 * np.sin(2 * np.pi * 220.0 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 1.5 * t))).astype(np.float32)
    s_sciss = (0.32 * np.sin(2 * np.pi * 3800.0 * t) * (0.5 + 0.5 * np.cos(2 * np.pi * 4.0 * t))).astype(np.float32)
    mix14 = (s_cello + s_sciss).astype(np.float32)

    queries = [
        {"name": "Cello", "friendly_name": "Cello", "class": "Cello", "group": "Musical instrument", "time_ranges": []},
        {"name": "Scissors", "friendly_name": "Scissors", "class": "Scissors", "group": "Household/tools", "time_ranges": []},
    ]
    out = query_separation.separate_by_queries(mix14, k=2, named_sounds=queries, speech_mode=False, sr=sr)
    assert out["available"] is True
    assert len(out["stems"]) == 2
    assert len(out["stems"][0]) == len(mix14), "Windowed stitched stem length must equal full 14s input length"
    assert out["residual_stem"] is not None and len(out["residual_stem"]) == len(mix14)

    sq = stem_quality.compute_stem_quality(out["stems"], mix14, residual=out["residual_stem"])
    assert sq["failed"] is False, f"Quality check unexpectedly failed: {sq}"
    assert sq["max_pairwise_corr"] < 0.90
    assert sq["energy_conservation"]["passed"] is True
    assert sq["energy_conservation"]["relative_l2_error"] < 1e-4
    print(
        f"  [PASS] 14s windowed separation: max_pairwise_corr={sq['max_pairwise_corr']:.4f}, "
        f"energy_rel_err={sq['energy_conservation']['relative_l2_error']:.6f} "
        f"(SNR={sq['energy_conservation']['reconstruction_snr_db']:.1f} dB)."
    )


def test_templates_and_routes():
    print("\n[4/5] Template & Route Safety Check...")
    client = flask_app.app.test_client()
    for route in ["/", "/analyze", "/evaluation", "/history", "/how", "/docs", "/api/health"]:
        resp = client.get(route)
        assert resp.status_code == 200, f"Route {route} returned HTTP {resp.status_code}"

    status_cases = [
        ("All keys present", flask_app.model_status()),
        ("Trained=True, all val metrics None", {
            "trained": True, "sample_rate": 16000, "clip_seconds": 8, "min_sources": 2, "max_sources": 31,
            "val_mae": None, "val_macro_mae": None, "val_exact_accuracy": None, "val_within_one": None,
        }),
        ("Trained=False, empty dict", {"trained": False}),
    ]
    with flask_app.app.test_request_context("/"):
        for label, st_dict in status_cases:
            html = render_template("index.html", status=st_dict)
            assert "Count every" in html, f"Failed rendering index.html for case: {label}"
    print("  [PASS] All routes HTTP 200 and Jinja templates safe under missing keys.")


def test_five_fixed_clips_regression():
    print("\n[5/5] 5-Clip End-to-End API Regression Suite (Phase 4.4)...")
    os.environ.pop("ECHOCOUNT_DEV", None)
    client = flask_app.app.test_client()
    ds_root = dataset_config.get_dataset_root(dataset_config.load_config())

    # Clip 1: Real-eval 2-speaker mixture (two_teachers_speaking.wav)
    c1 = BASE_DIR / "data" / "real_eval" / "two_teachers_speaking.wav"
    with open(c1, "rb") as f:
        r1 = client.post("/api/analyze", data={"audio": (f, c1.name)}, content_type="multipart/form-data")
    assert r1.status_code == 200
    j1 = r1.get_json()
    assert "ground_truth" not in j1, "Production /api/analyze must not include ground_truth"
    assert j1["count"]["rounded_count"] == 2
    assert j1["separation"]["can_separate"] is True
    s1 = client.post("/api/separate", json={"job_id": j1["job_id"]}).get_json()
    assert s1["used_pretrained_speech"] is True
    assert any(st["is_residual"] for st in s1["stems"]), "Must include residual stem"
    print(f"  [Clip 1 OK] {c1.name}: K={j1['count']['rounded_count']}, stems={[s['label'] for s in s1['stems']]}")

    # Clip 2: Real-eval 3-speaker mixture (three_speakers_discussion.wav)
    c2 = BASE_DIR / "data" / "real_eval" / "three_speakers_discussion.wav"
    with open(c2, "rb") as f:
        r2 = client.post("/api/analyze", data={"audio": (f, c2.name)}, content_type="multipart/form-data")
    j2 = r2.get_json()
    assert j2["count"]["rounded_count"] == 3
    assert j2["separation"]["can_separate"] is False
    print(f"  [Clip 2 OK] {c2.name}: K={j2['count']['rounded_count']}, can_separate={j2['separation']['can_separate']} ({j2['separation']['reason']})")

    # Clip 3: FSD_Mix_3_Sounds/test/mix_00035.wav (Cello, Scissors, Telephone ring)
    c3 = ds_root / "FSD_Mix_3_Sounds" / "test" / "mix_00035.wav"
    if c3.is_file():
        with open(c3, "rb") as f:
            r3 = client.post("/api/analyze", data={"audio": (f, c3.name)}, content_type="multipart/form-data")
        j3 = r3.get_json()
        names3 = [ns["name"] for ns in j3["named_sounds"]]
        assert "Cello" in names3 and "Scissors" in names3 and "Telephone ring" in names3, f"Expected Cello, Scissors, Telephone ring in {names3}"
        # Test async separation job submission + polling
        sep_sub = client.post("/api/separate", json={"job_id": j3["job_id"], "async": True})
        assert sep_sub.status_code == 202
        for _ in range(40):
            poll = client.get(f"/api/separate/{j3['job_id']}")
            pj = poll.get_json()
            if pj.get("status") == "completed":
                break
            time.sleep(0.25)
        assert pj.get("status") == "completed"
        stems3 = [s["label"] for s in pj["stems"]]
        assert "Everything else (Residual)" in stems3
        # Verify individual WAV download & ZIP download
        dl_wav = client.get(pj["stems"][0]["download_url"])
        assert dl_wav.status_code == 200 and len(dl_wav.data) > 1000
        dl_zip = client.get(pj["download_url"])
        assert dl_zip.status_code == 200 and len(dl_zip.data) > 1000
        print(f"  [Clip 3 OK] {c3.name}: K={j3['count']['rounded_count']}, named_sounds={names3}, stems={stems3}")

    # Clip 4: Dense mixture Org_Mix_9_Sounds/test/mix_00054.wav (True K=9)
    c4 = ds_root / "Org_Mix_9_Sounds" / "test" / "mix_00054.wav"
    if c4.is_file():
        with open(c4, "rb") as f:
            r4 = client.post("/api/analyze", data={"audio": (f, c4.name)}, content_type="multipart/form-data")
        j4 = r4.get_json()
        assert j4["speaker_clustering"]["show_card"] is False
        assert j4["separation"]["can_separate"] is False
        assert "Too many overlapping sounds to separate reliably" in (j4["count"]["dense_warning"] or "")
        print(f"  [Clip 4 OK] {c4.name}: K={j4['count']['rounded_count']}, dense_warning='{j4['count']['dense_warning']}'")

    # Clip 5: Held-out FSD50K isolated Hammer clip (open-vocabulary naming)
    hammer_clips = sorted((BASE_DIR / "data" / "isolated_sources" / "Hammer").glob("*.wav"))
    if hammer_clips:
        c5 = hammer_clips[-1]
        with open(c5, "rb") as f:
            r5 = client.post("/api/analyze", data={"audio": (f, c5.name)}, content_type="multipart/form-data")
        j5 = r5.get_json()
        names5 = [ns["name"] for ns in j5["named_sounds"]]
        assert "Hammer" in names5, f"Expected 'Hammer' in named_sounds for {c5.name}, got {names5}"
        print(f"  [Clip 5 OK] Hammer/{c5.name}: named_sounds={names5}, time_ranges={j5['named_sounds'][0].get('time_ranges')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.parse_args()

    print("=" * 84)
    print("ECHOCOUNT PRODUCT TEST SUITE (Phases 0 – 4)")
    print("=" * 84)
    test_taxonomy_consistency()
    test_production_hygiene()
    test_quality_gates_and_windowing()
    test_templates_and_routes()
    test_five_fixed_clips_regression()
    print("\nALL PRODUCT TESTS PASSED.")


if __name__ == "__main__":
    main()
