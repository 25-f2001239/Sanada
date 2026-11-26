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
    """Verify public pages return HTTP 200, admin pages are gated, and templates tolerate missing keys."""
    client = flask_app.app.test_client()
    for route in ("/", "/analyze", "/history", "/how", "/docs", "/api/health"):
        assert client.get(route).status_code == 200
    # Evaluation & Admin are hidden from anonymous users
    for route in ("/evaluation", "/admin"):
        assert client.get(route).status_code == 302
    assert client.get("/api/evaluation").status_code == 401
    assert client.get("/api/admin/overview").status_code == 401
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


def test_admin_access_and_management() -> None:
    """Verify admin-only evaluation, user promotion/demotion, invites, disabling, and audit log."""
    import tempfile
    import auth as auth_mod

    orig_db = auth_mod.DB_PATH
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        auth_mod.DB_PATH = Path(td) / "test_admin.db"
        auth_mod.init_db()
        try:
            admin_uid = auth_mod.ensure_admin_account("root_admin@gmail.com", "AdminStrong#2026")
            c_admin = flask_app.app.test_client()
            a_sid, a_csrf = auth_mod.create_session(admin_uid)
            c_admin.set_cookie(auth_mod.SESSION_COOKIE, a_sid)

            c_user = flask_app.app.test_client()
            user_uid, _ = _login_test_user(c_user, "regular.member@gmail.com")

            # Regular user cannot see /evaluation, /admin, or admin APIs
            assert c_user.get("/evaluation").status_code == 404
            assert c_user.get("/admin").status_code == 404
            assert c_user.get("/api/evaluation").status_code == 403
            assert c_user.get("/api/admin/overview").status_code == 403
            user_home = c_user.get("/").get_data(as_text=True)
            assert 'href="/evaluation"' not in user_home and 'href="/admin"' not in user_home

            # Admin can see /evaluation, /admin, and all admin APIs
            assert c_admin.get("/evaluation").status_code == 200
            assert c_admin.get("/admin").status_code == 200
            assert c_admin.get("/api/evaluation").status_code == 200
            assert c_admin.get("/api/admin/overview").status_code == 200
            assert c_admin.get("/api/admin/users").status_code == 200
            assert c_admin.get("/api/admin/users.csv").status_code == 200
            assert c_admin.get("/api/admin/analyses").status_code == 200
            admin_home = c_admin.get("/").get_data(as_text=True)
            assert 'href="/evaluation"' in admin_home and 'href="/admin"' in admin_home

            # CSRF required on mutations
            assert c_admin.post(f"/api/admin/users/{user_uid}/role", json={"role": "admin"}).status_code == 403

            # Promote user -> admin -> now user can access /evaluation
            h = {"X-CSRF-Token": a_csrf}
            r = c_admin.post(f"/api/admin/users/{user_uid}/role", json={"role": "admin"}, headers=h)
            assert r.status_code == 200 and r.get_json()["ok"] is True
            assert c_user.get("/evaluation").status_code == 200

            # Admin cannot demote themselves
            r_self = c_admin.post(f"/api/admin/users/{admin_uid}/role", json={"role": "user"}, headers=h)
            assert r_self.status_code == 400

            # Demote user back to regular
            r_dem = c_admin.post(f"/api/admin/users/{user_uid}/role", json={"role": "user"}, headers=h)
            assert r_dem.status_code == 200
            assert c_user.get("/evaluation").status_code == 404

            # Pre-authorize future admin by email invite -> auto-promoted on first sign-in
            r_inv = c_admin.post("/api/admin/invites", json={"email": "future.admin@gmail.com"}, headers=h)
            assert r_inv.status_code == 200
            c_future = flask_app.app.test_client()
            _login_test_user(c_future, "future.admin@gmail.com")
            assert c_future.get("/admin").status_code == 200

            # Disable user -> immediately invalidates session
            r_dis = c_admin.post(f"/api/admin/users/{user_uid}/disabled", json={"disabled": True}, headers=h)
            assert r_dis.status_code == 200
            assert c_user.get("/api/auth/me").get_json()["authenticated"] is False

            # Audit log recorded actions
            audit = c_admin.get("/api/admin/audit").get_json()["audit"]
            actions = {row["action"] for row in audit}
            assert {"promote", "demote", "invite_admin", "invite_accepted", "disable"}.issubset(actions)

            # Per-user history isolation: each user (and admin) only sees their own history;
            # admin Activity sees everyone's even after a user clears their own history.
            auth_mod.record_analysis(admin_uid, "j1", "admin_clip.wav", 8.0, 2, 91.0, None, None, 120.0)
            auth_mod.record_analysis(user_uid, "j2", "user_clip.wav", 8.0, 4, 82.0, 4, "Org_Mix_4", 150.0)
            # Re-enable user to verify history API
            assert c_admin.post(f"/api/admin/users/{user_uid}/disabled", json={"disabled": False}, headers=h).status_code == 200
            c_user2 = flask_app.app.test_client()
            u_sid, u_csrf = auth_mod.create_session(user_uid)
            c_user2.set_cookie(auth_mod.SESSION_COOKIE, u_sid)

            adm_hist = c_admin.get("/api/history").get_json()["items"]
            usr_hist = c_user2.get("/api/history").get_json()["items"]
            assert [x["filename"] for x in adm_hist] == ["admin_clip.wav"]
            assert [x["filename"] for x in usr_hist] == ["user_clip.wav"]

            # User cannot delete admin's history item
            assert c_user2.delete(f"/api/history/{adm_hist[0]['id']}", headers={"X-CSRF-Token": u_csrf}).status_code == 404
            # User clears own history -> gone from /api/history, still visible in /api/admin/analyses
            assert c_user2.delete("/api/history", headers={"X-CSRF-Token": u_csrf}).status_code == 200
            assert c_user2.get("/api/history").get_json()["items"] == []
            all_act = [x["filename"] for x in c_admin.get("/api/admin/analyses").get_json()["analyses"]]
            assert "admin_clip.wav" in all_act and "user_clip.wav" in all_act
        finally:
            auth_mod.DB_PATH = orig_db
    print("  [PASS] Admin access control, role management, invites, and audit log.")


def _login_test_user(client, email: str = "qa.tester@gmail.com"):
    import auth as auth_mod

    auth_mod.init_db()
    with auth_mod._conn() as con:
        row = con.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if row:
            uid = int(row["id"])
            con.execute("UPDATE users SET email_verified = 1 WHERE id = ?", (uid,))
        else:
            cur = con.execute(
                "INSERT INTO users (email, password_hash, email_verified, created_at) VALUES (?, ?, 1, ?)",
                (email, auth_mod.hash_password("TestPass123!"), time.time()),
            )
            uid = int(cur.lastrowid)
    sid, csrf = auth_mod.create_session(uid)
    client.set_cookie(auth_mod.SESSION_COOKIE, sid)
    return uid, csrf


def test_five_fixed_clips_regression() -> None:
    """Run 5-clip end-to-end API regression including async separation and WAV/ZIP downloads."""
    os.environ.pop("ECHOCOUNT_DEV", None)
    client = flask_app.app.test_client()
    _login_test_user(client, "regression.tester@gmail.com")
    ds_root = cfg_mod.get_dataset_root(cfg_mod.load_config())

    c1 = BASE_DIR / "data" / "real_eval" / "two_teachers_speaking.wav"
    with open(c1, "rb") as f:
        j1 = client.post(
            "/api/analyze", data={"audio": (f, c1.name)}, content_type="multipart/form-data"
        ).get_json()
    assert (
        j1.get("ground_truth", {}).get("true_k") == 2
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


def test_expo_ready_pages_and_privacy() -> None:
    """Verify public_summary.json privacy, Home/How rendering with/without stats, and job cleanup."""
    import re
    import shutil
    import app as app_mod

    pub_path = BASE_DIR / "evaluation" / "public_summary.json"
    assert pub_path.is_file(), "evaluation/public_summary.json missing"
    raw_pub = pub_path.read_text(encoding="utf-8")
    pub = json.loads(raw_pub)
    assert "ground_truth" not in raw_pub.lower()
    assert not re.search(r"\b[0-9a-f]{64}\b", raw_pub)
    assert not re.search(r"[A-Za-z]:\\|/coding/|mix_\d+\.wav", raw_pub)
    assert pub["counting"]["overall"]["n"] == 1285

    client = app_mod.app.test_client()
    assert client.get("/api/public-stats").status_code == 200

    home_html = client.get("/").get_data(as_text=True)
    for forbidden in ("podcast_ep12", "cafe_corner", "duet_take4", "87%", ">Example<", "Model ready"):
        assert forbidden not in home_html, f"Found forbidden string {forbidden!r} on Home"
    assert "Measured, not promised" in home_html
    assert "What it can hear" in home_html
    assert "Fast and checked" in home_html

    how_html = client.get("/how").get_data(as_text=True)
    for anchor in ("#problem", "#data", "#pipeline", "#quality-gates", "#train", "#evaluate", "#findings", "#facts", "#keep-in-mind", "#reproducibility", "#references"):
        assert anchor in how_html, f"Missing TOC anchor {anchor} on /how"

    orig_path = app_mod.PUBLIC_SUMMARY_PATH
    try:
        app_mod.PUBLIC_SUMMARY_PATH = BASE_DIR / "evaluation" / "__nonexistent__.json"
        assert client.get("/api/public-stats").status_code == 404
        home_no_stats = client.get("/").get_data(as_text=True)
        assert "Measured, not promised" not in home_no_stats
        assert client.get("/how").status_code == 200
    finally:
        app_mod.PUBLIC_SUMMARY_PATH = orig_path

    # Startup cleanup and TTL sweep test
    orphan = app_mod.JOB_DIR / "test_orphan_startup"
    orphan.mkdir(parents=True, exist_ok=True)
    (orphan / "dummy.wav").write_bytes(b"0")
    app_mod.sweep_jobs(startup=True)
    assert not orphan.exists(), "sweep_jobs(startup=True) did not remove orphan job directory"

    expired = app_mod.JOB_DIR / "test_orphan_ttl"
    expired.mkdir(parents=True, exist_ok=True)
    old_ts = time.time() - (app_mod.JOB_TTL_SECONDS + 60)
    os.utime(expired, (old_ts, old_ts))
    app_mod.sweep_jobs(startup=False)
    assert not expired.exists(), "sweep_jobs(startup=False) did not remove expired job directory"
    shutil.rmtree(orphan, ignore_errors=True)
    shutil.rmtree(expired, ignore_errors=True)
    print("  [PASS] Expo-ready pages, public_summary privacy, and job cleanup.")


def test_auth_security_and_ground_truth() -> None:
    """Verify OAuth provider restrictions, email policy, Argon2id, CSRF, authz, and Ground Truth."""
    import auth as auth_mod

    auth_mod.init_db()
    anon = flask_app.app.test_client()

    # Protected endpoints reject unauthenticated callers
    assert anon.post("/api/analyze").status_code == 401
    assert anon.post("/api/separate", json={"job_id": "abc"}).status_code == 401
    assert anon.post("/api/predict").status_code == 401
    assert anon.get("/api/download/abc.zip").status_code == 401

    # Unsupported OAuth provider rejected
    assert anon.get("/auth/oauth/facebook").status_code == 404

    # Auth pages render without username fields or CAPTCHA, and include eye toggle
    login_html = anon.get("/login").get_data(as_text=True)
    assert "data-pw-toggle" in login_html
    assert 'name="username"' not in login_html
    assert "turnstile" not in login_html.lower()
    assert "/auth/oauth/google" in login_html
    assert "/auth/oauth/microsoft" not in login_html and "/auth/oauth/yahoo" not in login_html

    # Fetch CSRF token from /login
    import re

    m = re.search(r'name="csrf_token"\s+value="([^"]+)"', login_html)
    assert m, "CSRF token missing from /login"
    csrf = m.group(1)

    # Disposable & unsupported emails rejected
    r_disp = anon.post(
        "/auth/register",
        json={"email": "bad@mailinator.com", "password": "StrongPassword123!", "csrf_token": csrf},
        headers={"X-CSRF-Token": csrf},
    )
    assert r_disp.status_code == 400

    r_dom = anon.post(
        "/auth/register",
        json={"email": "bad@unknown-domain.example", "password": "StrongPassword123!", "csrf_token": csrf},
        headers={"X-CSRF-Token": csrf},
    )
    assert r_dom.status_code == 400

    # Valid Google email registration -> Argon2id hash -> verification required before login
    # (localhost remote_addr automatically surfaces verify_url when SMTP_HOST is unset)
    test_email = f"verify_{int(time.time() * 1000)}@gmail.com"
    r_ok = anon.post(
        "/auth/register",
        json={"email": test_email, "password": "StrongPassword123!", "csrf_token": csrf},
        headers={"X-CSRF-Token": csrf},
        environ_base={"REMOTE_ADDR": "127.0.0.1"},
    )
    assert r_ok.status_code == 201
    v_url = r_ok.get_json()["verify_url"]
    with auth_mod._conn() as con:
        row = con.execute("SELECT password_hash, email_verified FROM users WHERE email = ?", (test_email,)).fetchone()
        assert row["password_hash"].startswith("$argon2id$")
        assert row["email_verified"] == 0

    # Cannot log in before verifying email
    r_pre = anon.post(
        "/auth/login",
        json={"email": test_email, "password": "StrongPassword123!", "csrf_token": csrf},
        headers={"X-CSRF-Token": csrf},
    )
    assert r_pre.status_code == 401

    # Verify via token link -> activates account & logs in
    from urllib.parse import urlparse

    r_ver = anon.get(urlparse(v_url).path + "?" + urlparse(v_url).query)
    assert r_ver.status_code in (200, 302)
    assert anon.get("/api/auth/me").get_json()["authenticated"] is True

    # Logout -> re-login with verified credentials
    me_page = anon.get("/login")
    assert me_page.status_code in (200, 302)
    r_out = anon.post("/auth/logout", headers={"X-CSRF-Token": csrf})
    # Note: after verification, session has its own CSRF token; test fresh login on new client
    c_fresh = flask_app.app.test_client()
    h_fresh = c_fresh.get("/login").get_data(as_text=True)
    csrf_fresh = re.search(r'name="csrf_token"\s+value="([^"]+)"', h_fresh).group(1)
    r_in = c_fresh.post(
        "/auth/login",
        json={"email": test_email, "password": "StrongPassword123!", "csrf_token": csrf_fresh},
        headers={"X-CSRF-Token": csrf_fresh},
    )
    assert r_in.status_code == 200 and r_in.get_json()["ok"] is True
    assert c_fresh.get("/api/auth/me").get_json()["authenticated"] is True

    # Permanent admin account provisioning is idempotent & verified
    admin_uid = auth_mod.ensure_admin_account("qa_admin@outlook.com", "AdminStrong#2026")
    assert admin_uid > 0
    with auth_mod._conn() as con:
        adm = con.execute("SELECT role, email_verified FROM users WHERE id = ?", (admin_uid,)).fetchone()
        assert adm["role"] == "admin" and adm["email_verified"] == 1

    # Cross-user job isolation: User B cannot access User A's job
    user_a = flask_app.app.test_client()
    user_b = flask_app.app.test_client()
    _login_test_user(user_a, "user_a@outlook.com")
    _login_test_user(user_b, "user_b@yahoo.com")
    c_real = BASE_DIR / "data" / "real_eval" / "two_teachers_speaking.wav"
    with open(c_real, "rb") as f:
        ja = user_a.post("/api/analyze", data={"audio": (f, c_real.name)}, content_type="multipart/form-data").get_json()
    jid = ja["job_id"]
    assert ja.get("ground_truth", {}).get("available") is True
    assert user_b.post("/api/separate", json={"job_id": jid}).status_code == 404
    assert user_b.get(f"/api/separate/{jid}").status_code == 404
    assert user_b.get(f"/api/download/{jid}.zip").status_code == 404

    # Non-dataset audio has no ground_truth match (never fabricated)
    import io, audio as audio_mod
    synth_wav = audio_mod.wav_bytes(np.zeros(16000, dtype=np.float32), 16000)
    j_synth = user_a.post(
        "/api/analyze", data={"audio": (io.BytesIO(synth_wav), "custom_room.wav")}, content_type="multipart/form-data"
    ).get_json()
    assert "ground_truth" not in j_synth

    # Ground Truth SHA-256 match on known dataset file
    ds_root = cfg_mod.get_dataset_root(cfg_mod.load_config())
    c_gt = ds_root / "FSD_Mix_3_Sounds" / "test" / "mix_00035.wav"
    if c_gt.is_file():
        with open(c_gt, "rb") as f:
            jgt = user_a.post("/api/analyze", data={"audio": (f, c_gt.name)}, content_type="multipart/form-data").get_json()
        assert jgt.get("ground_truth", {}).get("available") is True
        assert jgt["ground_truth"]["true_k"] == 3
    print("  [PASS] Auth security, OAuth/email policy, Argon2id, user isolation, and Ground Truth.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true")
    p.parse_args()
    test_taxonomy_consistency()
    test_production_hygiene()
    test_quality_gates_and_windowing()
    test_templates_and_routes()
    test_admin_access_and_management()
    test_expo_ready_pages_and_privacy()
    test_auth_security_and_ground_truth()
    test_five_fixed_clips_regression()
    print("ALL PRODUCT TESTS PASSED.")


if __name__ == "__main__":
    main()
