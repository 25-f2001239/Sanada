"""Product unit, API, taxonomy, quality-gate, and 5-clip regression test suite."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from flask import render_template

import app as flask_app
import config as cfg_mod
import data as data_mod
import quality as quality_mod
import separation

BASE_DIR = Path(__file__).resolve().parent
CKPT_DIR = BASE_DIR / "checkpoints"


def test_taxonomy_consistency() -> None:
    """Verify taxonomy.json, tagger.pt, and ground_truth_index.json agree on every class group."""
    tax = cfg_mod.load_taxonomy()
    assert cfg_mod.get_group_for_class("Cello", tax) == "Musical instrument"
    assert cfg_mod.get_group_for_class("Scissors", tax) == "Household/tools"
    assert cfg_mod.get_group_for_class("Telephone", tax) == "Household/tools"
    assert cfg_mod.get_friendly_name("Telephone", tax) == "Telephone ring"

    target_12 = tax.get("target_12_sounds", {})
    expected_12 = {
        "hammer",
        "phone vibration",
        "dog",
        "cat",
        "cello",
        "scissors",
        "telephone ring",
        "speech",
        "door knock",
        "keyboard typing",
        "water",
        "car horn",
    }
    assert set(target_12.keys()) == expected_12
    for spec in target_12.values():
        assert spec["group"] in tax["groups"]
        assert cfg_mod.get_group_for_class(spec["friendly_name"], tax) == spec["group"]

    ckpt = torch.load(CKPT_DIR / "tagger.pt", map_location="cpu", weights_only=False)
    for cls_name, ckpt_grp in ckpt.get("fine_to_group", {}).items():
        assert ckpt_grp == cfg_mod.get_group_for_class(cls_name, tax)

    if data_mod.INDEX_PATH.is_file():
        m35 = (
            json.loads(data_mod.INDEX_PATH.read_text(encoding="utf-8"))
            .get("by_relpath", {})
            .get("fsd_mix_3_sounds/test/mix_00035.wav")
        )
        if m35:
            assert m35["true_groups"] == ["Household/tools", "Musical instrument"]
    print("  [PASS] Taxonomy single source of truth.")


def test_production_hygiene() -> None:
    """Verify ground truth is gated off by default and static/analyze.js has no dev/demo strings."""
    os.environ.pop("ECHOCOUNT_DEV", None)
    assert data_mod.is_dev_mode() is False
    assert (
        data_mod.lookup_ground_truth(BASE_DIR / "data" / "real_eval" / "two_teachers_speaking.wav")
        is None
    )
    js_code = (BASE_DIR / "static" / "analyze.js").read_text(encoding="utf-8")
    for forbidden in (
        "In-distribution test file",
        "Dataset file (known ground truth)",
        "Untrained demo weights",
        "SHA-256",
    ):
        assert forbidden not in js_code
    print("  [PASS] Production hygiene.")


def test_quality_gates_and_windowing() -> None:
    """Verify 14s windowed separation, correlation gate, and energy conservation."""
    sr = 16000
    t = np.linspace(0, 14.0, sr * 14, endpoint=False, dtype=np.float32)
    s_cello = (
        0.30 * np.sin(2 * np.pi * 220.0 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 1.5 * t))
    ).astype(np.float32)
    s_sciss = (
        0.32 * np.sin(2 * np.pi * 3800.0 * t) * (0.5 + 0.5 * np.cos(2 * np.pi * 4.0 * t))
    ).astype(np.float32)
    mix14 = (s_cello + s_sciss).astype(np.float32)
    queries = [
        {
            "name": "Cello",
            "friendly_name": "Cello",
            "class": "Cello",
            "group": "Musical instrument",
            "time_ranges": [],
        },
        {
            "name": "Scissors",
            "friendly_name": "Scissors",
            "class": "Scissors",
            "group": "Household/tools",
            "time_ranges": [],
        },
    ]
    out = separation.separate_by_queries(mix14, k=2, named_sounds=queries, speech_mode=False, sr=sr)
    assert out["available"] and len(out["stems"]) == 2 and len(out["stems"][0]) == len(mix14)
    sq = quality_mod.compute_stem_quality(out["stems"], mix14, residual=out["residual_stem"])
    assert (
        sq["failed"] is False
        and sq["max_pairwise_corr"] < 0.90
        and sq["energy_conservation"]["passed"]
    )
    print(
        f"  [PASS] 14s windowed separation (corr={sq['max_pairwise_corr']:.4f}, SNR={sq['energy_conservation']['reconstruction_snr_db']:.1f} dB)."
    )


def test_templates_and_routes() -> None:
    """Verify all pages and API health route return HTTP 200 and templates tolerate missing keys."""
    client = flask_app.app.test_client()
    for route in ("/", "/analyze", "/evaluation", "/history", "/how", "/docs", "/api/health"):
        assert client.get(route).status_code == 200
    cases = [
        flask_app.model_status(),
        {
            "trained": True,
            "sample_rate": 16000,
            "clip_seconds": 8,
            "min_sources": 2,
            "max_sources": 31,
            "val_mae": None,
            "val_macro_mae": None,
            "val_exact_accuracy": None,
            "val_within_one": None,
        },
        {"trained": False},
    ]
    with flask_app.app.test_request_context("/"):
        for st_dict in cases:
            assert "Count every" in render_template("index.html", status=st_dict)
    print("  [PASS] Templates & routes.")


def test_five_fixed_clips_regression() -> None:
    """Run 5-clip end-to-end API regression including async separation and WAV/ZIP downloads."""
    os.environ.pop("ECHOCOUNT_DEV", None)
    client = flask_app.app.test_client()
    ds_root = cfg_mod.get_dataset_root(cfg_mod.load_config())

    c1 = BASE_DIR / "data" / "real_eval" / "two_teachers_speaking.wav"
    with open(c1, "rb") as f:
        j1 = client.post(
            "/api/analyze", data={"audio": (f, c1.name)}, content_type="multipart/form-data"
        ).get_json()
    assert (
        "ground_truth" not in j1
        and j1["count"]["rounded_count"] == 2
        and j1["separation"]["can_separate"]
    )
    s1 = client.post("/api/separate", json={"job_id": j1["job_id"]}).get_json()
    assert s1["used_pretrained_speech"] and any(st["is_residual"] for st in s1["stems"])

    c2 = BASE_DIR / "data" / "real_eval" / "three_speakers_discussion.wav"
    with open(c2, "rb") as f:
        j2 = client.post(
            "/api/analyze", data={"audio": (f, c2.name)}, content_type="multipart/form-data"
        ).get_json()
    assert j2["count"]["rounded_count"] == 3 and j2["separation"]["can_separate"] is False

    c3 = ds_root / "FSD_Mix_3_Sounds" / "test" / "mix_00035.wav"
    if c3.is_file():
        with open(c3, "rb") as f:
            j3 = client.post(
                "/api/analyze", data={"audio": (f, c3.name)}, content_type="multipart/form-data"
            ).get_json()
        names3 = [ns["name"] for ns in j3["named_sounds"]]
        assert "Cello" in names3 and "Scissors" in names3 and "Telephone ring" in names3
        assert (
            client.post("/api/separate", json={"job_id": j3["job_id"], "async": True}).status_code
            == 202
        )
        pj = {}
        for _ in range(40):
            pj = client.get(f"/api/separate/{j3['job_id']}").get_json()
            if pj.get("status") == "completed":
                break
            time.sleep(0.25)
        assert pj.get("status") == "completed" and "Everything else (Residual)" in [
            s["label"] for s in pj["stems"]
        ]
        assert client.get(pj["stems"][0]["download_url"]).status_code == 200
        assert client.get(pj["download_url"]).status_code == 200

    c4 = ds_root / "Org_Mix_9_Sounds" / "test" / "mix_00054.wav"
    if c4.is_file():
        with open(c4, "rb") as f:
            j4 = client.post(
                "/api/analyze", data={"audio": (f, c4.name)}, content_type="multipart/form-data"
            ).get_json()
        assert (
            j4["speaker_clustering"]["show_card"] is False
            and j4["separation"]["can_separate"] is False
        )
        assert "Too many overlapping sounds to separate reliably" in (
            j4["count"]["dense_warning"] or ""
        )

    hammer = sorted((BASE_DIR / "data" / "isolated_sources" / "Hammer").glob("*.wav"))
    if hammer:
        with open(hammer[-1], "rb") as f:
            j5 = client.post(
                "/api/analyze",
                data={"audio": (f, hammer[-1].name)},
                content_type="multipart/form-data",
            ).get_json()
        assert "Hammer" in [ns["name"] for ns in j5["named_sounds"]]
    print("  [PASS] 5-clip regression.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true")
    p.parse_args()
    test_taxonomy_consistency()
    test_production_hygiene()
    test_quality_gates_and_windowing()
    test_templates_and_routes()
    test_five_fixed_clips_regression()
    print("ALL PRODUCT TESTS PASSED.")


if __name__ == "__main__":
    main()
