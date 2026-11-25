"""Automated security & privacy tests for per-user history isolation (Part A)."""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app as app_mod
import auth
import mail as mail_mod


class _CaptureMail(mail_mod.MailProvider):
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []

    def send(self, to_email: str, subject: str, body: str) -> tuple[bool, str | None]:
        self.sent.append((to_email, subject, body))
        return True, None


_TEST_MAIL = _CaptureMail()


def _extract_csrf(client, path="/login") -> str:
    r = client.get(path)
    m = re.search(r'<meta name="csrf-token" content="([^"]+)"', r.get_data(as_text=True))
    assert m, f"Missing csrf-token meta on {path}"
    return m.group(1)


def _seed_verified_user(email: str, password: str, role: str = "user") -> int:
    uid, _tok, err = auth.register_password_user(email, password, check_mx=False)
    assert uid and not err, f"Failed creating {email}: {err}"
    with auth._conn() as con:
        con.execute(
            "UPDATE users SET email_verified = 1, is_active = 1, role = ? WHERE id = ?", (role, uid)
        )
    return int(uid)


def _login(client, email: str, password: str) -> str:
    csrf = _extract_csrf(client, "/login")
    r = client.post(
        "/auth/login",
        json={"email": email, "password": password},
        headers={"X-CSRF-Token": csrf},
    )
    assert r.status_code == 200, (
        f"Login failed for {email}: {r.status_code} {r.get_data(as_text=True)}"
    )
    body = r.get_json() or {}
    if body.get("otp_required"):
        assert _TEST_MAIL.sent, f"Expected OTP email for {email}"
        code_m = re.search(r"\b(\d{6})\b", _TEST_MAIL.sent[-1][2])
        assert code_m, "Missing 6-digit OTP in email"
        r2 = client.post(
            "/auth/otp/verify",
            json={"email": email, "purpose": body.get("purpose", "admin_login"), "code": code_m.group(1)},
            headers={"X-CSRF-Token": csrf},
        )
        assert r2.status_code == 200, f"OTP verify failed for {email}: {r2.status_code} {r2.get_data(as_text=True)}"
    return _extract_csrf(client, "/history")


def _logout(client, csrf: str) -> None:
    r = client.post("/auth/logout", headers={"X-CSRF-Token": csrf})
    assert r.status_code == 200


def _seed_job_for_user(uid: int, filename: str, count: int = 2) -> str:
    job_id = uuid.uuid4().hex
    job_dir = app_mod.JOB_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    stem_path = job_dir / "stem_01_voice.wav"
    res_path = job_dir / "residual.wav"
    orig_path = job_dir / "original.wav"
    for p in (stem_path, res_path, orig_path):
        p.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zf:
        zf.writestr("manifest.json", "{}")
        zf.writestr("original_mix.wav", b"RIFF\x24\x00\x00\x00WAVEfmt ")
    auth.register_user_job(
        job_id,
        uid,
        filename,
        {
            "rounded_count": count,
            "display": f"≈{count} (±0)",
            "confidence_percent": 91.5,
            "duration_seconds": 8.0,
            "top_sounds": ["Speech"],
        },
    )
    with app_mod._jobs_lock:
        app_mod._jobs[job_id] = {
            "created": 1e12,
            "name": filename,
            "original": orig_path,
            "input_rate": 16000,
            "duration": 8.0,
            "count": {
                "rounded_count": count,
                "raw_regression_value": float(count),
                "total_std": 0.1,
                "mc_std": 0.05,
                "passes": 8,
            },
            "named_sounds": [],
            "speech_detected": True,
            "stems": ["stem_01_voice.wav", "residual.wav"],
            "zip": zbuf.getvalue(),
            "sep_status": "completed",
            "sep_result": {"job_id": job_id, "status": "completed"},
        }
    return job_id


def _verify_client_storage_purge_via_node() -> None:
    """Execute static/main.js in Node VM to prove legacy purge & cache clearing on switch/logout."""
    script = r"""
const fs = require("fs");
const vm = require("vm");
const code = fs.readFileSync("static/main.js", "utf8");

function makeStore(init = {}) {
  const m = new Map(Object.entries(init));
  return {
    get length() { return m.size; },
    key(i) { return [...m.keys()][i] ?? null; },
    getItem(k) { return m.has(k) ? m.get(k) : null; },
    setItem(k, v) { m.set(k, String(v)); },
    removeItem(k) { m.delete(k); },
    _keys() { return [...m.keys()]; },
  };
}

function bootMain(uid, localInit, sessionInit) {
  const ls = makeStore(localInit);
  const ss = makeStore(sessionInit);
  const listeners = {};
  const fakeEl = (id) => ({
    id,
    content: "csrf-tok",
    addEventListener(ev, fn) { listeners[id + ":" + ev] = fn; },
    classList: { add(){}, remove(){}, toggle(){} },
    setAttribute(){},
    appendChild(){},
    querySelectorAll: () => [],
  });
  const ctx = {
    localStorage: ls,
    sessionStorage: ss,
    matchMedia: () => ({ matches: false, addEventListener(){} }),
    location: { href: "/history", pathname: "/history" },
    fetch: async () => ({ ok: true, json: async () => ({}) }),
    setTimeout, clearTimeout,
    document: {
      documentElement: { dataset: { theme: "dark" }, classList: { add(){} } },
      body: { dataset: { uid: String(uid || ""), page: "history" } },
      querySelector: (s) => (s === "#logoutBtn" ? fakeEl("#logoutBtn") : s === 'meta[name="csrf-token"]' ? fakeEl("csrf") : null),
      querySelectorAll: () => [],
      addEventListener() {},
    },
    window: {},
    CustomEvent: class CustomEvent { constructor(t, o) { this.type = t; this.detail = o?.detail; } },
    IntersectionObserver: class IntersectionObserver { observe(){} unobserve(){} disconnect(){} },
    requestAnimationFrame: () => 0,
    cancelAnimationFrame() {},
    dispatchEvent() {},
    addEventListener() {},
  };
  ctx.window = ctx;
  vm.runInNewContext(code, ctx);
  return { ctx, ls, ss, listeners };
}

// 1. Legacy echocount.history.v1 is purged immediately on load and never imported
const r1 = bootMain("101", { "echocount.history.v1": JSON.stringify([{ id: "old", filename: "admin.wav" }]) }, {});
if (r1.ls.getItem("echocount.history.v1") !== null) throw new Error("Legacy key was not purged on load");
if (r1.ctx.window.EC.H.all().length !== 0) throw new Error("Legacy history was imported into H.all()");

// 2. Switching user uid in the same browser clears any echocount.history* keys
const r2 = bootMain(
  "202",
  { "echocount.history.cache": "stale_a" },
  { "echocount.active_uid": "101", "echocount.history.tmp": "stale_a" }
);
if (r2.ls.getItem("echocount.history.cache") !== null) throw new Error("localStorage history cache survived user switch");
if (r2.ss.getItem("echocount.history.tmp") !== null) throw new Error("sessionStorage history cache survived user switch");
if (r2.ss.getItem("echocount.active_uid") !== "202") throw new Error("active_uid did not update");

// 3. Logout clears history cache and active_uid
r2.ls.setItem("echocount.history.u202", "secret");
r2.listeners["#logoutBtn:click"]().then(() => {
  if (r2.ls.getItem("echocount.history.u202") !== null) throw new Error("History cache survived logout");
  if (r2.ss.getItem("echocount.active_uid") !== null) throw new Error("active_uid survived logout");
  console.log("NODE_CLIENT_STORAGE_CHECKS_OK");
}).catch((e) => { console.error(e); process.exit(1); });
"""
    proc = subprocess.run(
        ["node", "-e", script],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0 and "NODE_CLIENT_STORAGE_CHECKS_OK" in proc.stdout, (
        f"Client storage verification failed: {proc.stdout}\n{proc.stderr}"
    )


def run_tests() -> None:
    auth.init_db()
    mail_mod.set_provider(_TEST_MAIL)
    prev_skip = os.environ.get("ECHOCOUNT_SKIP_MX_CHECK")
    prev_delay = os.environ.get("ECHOCOUNT_NO_AUTH_DELAY")
    os.environ["ECHOCOUNT_SKIP_MX_CHECK"] = "1"
    os.environ["ECHOCOUNT_NO_AUTH_DELAY"] = "1"
    tag = uuid.uuid4().hex[:8]
    email_a = f"priv_user_a_{tag}@gmail.com"
    email_b = f"priv_user_b_{tag}@gmail.com"
    email_adm = f"priv_admin_{tag}@gmail.com"
    pw = "StrongTestPass!2026"

    created_uids: list[int] = []
    created_jobs: list[str] = []

    try:
        uid_a = _seed_verified_user(email_a, pw, role="user")
        uid_b = _seed_verified_user(email_b, pw, role="user")
        uid_adm = _seed_verified_user(email_adm, pw, role="admin")
        created_uids.extend([uid_a, uid_b, uid_adm])

        client_a = app_mod.app.test_client()
        client_b = app_mod.app.test_client()
        client_adm = app_mod.app.test_client()

        # 1. User A logs in and owns a job
        csrf_a = _login(client_a, email_a, pw)
        job_a = _seed_job_for_user(uid_a, f"secret_a_{tag}.wav", count=2)
        created_jobs.append(job_a)

        r = client_a.get("/api/history")
        assert r.status_code == 200
        assert "no-store" in r.headers.get("Cache-Control", "").lower(), (
            "Missing Cache-Control: no-store"
        )
        items_a = r.get_json()["items"]
        assert len(items_a) == 1 and items_a[0]["job_id"] == job_a
        _logout(client_a, csrf_a)

        # 2. User B logs in -> empty history & 404 on User A's job/stems/zip/delete
        csrf_b = _login(client_b, email_b, pw)
        r_b = client_b.get("/api/history")
        assert r_b.status_code == 200
        assert r_b.get_json()["items"] == [], f"New user B saw leaked history: {r_b.get_json()}"
        assert client_b.get("/api/my-jobs").get_json()["jobs"] == []

        # Forged user_id query param and header must be ignored
        r_forged = client_b.get(f"/api/history?user_id={uid_a}", headers={"X-User-Id": str(uid_a)})
        assert r_forged.status_code == 200
        assert r_forged.get_json()["items"] == [], "Forged user_id parameter was not ignored!"

        # Object-level auth: 404 (not 403) for non-owner on job, stem, residual, original & ZIP
        assert client_b.post("/api/separate", json={"job_id": job_a}).status_code == 404
        r_sep_st = client_b.get(f"/api/separate/{job_a}")
        assert r_sep_st.status_code == 404
        assert "no-store" in r_sep_st.headers.get("Cache-Control", "").lower()
        assert client_b.get(f"/api/stem/{job_a}/stem_01_voice.wav").status_code == 404
        assert client_b.get(f"/api/stem/{job_a}/residual.wav").status_code == 404
        assert client_b.get(f"/api/stem/{job_a}/original.wav").status_code == 404
        assert client_b.get(f"/api/download/{job_a}.zip").status_code == 404
        assert (
            client_b.delete(f"/api/history/{job_a}", headers={"X-CSRF-Token": csrf_b}).status_code
            == 404
        )

        # Same-browser (shared cookie jar) User A -> logout -> User B switch test
        shared = app_mod.app.test_client()
        csrf_sh_a = _login(shared, email_a, pw)
        html_a = shared.get("/").get_data(as_text=True)
        assert f'data-uid="{uid_a}"' in html_a and f"secret_a_{tag}.wav" in html_a
        _logout(shared, csrf_sh_a)
        _login(shared, email_b, pw)
        html_b_home = shared.get("/").get_data(as_text=True)
        html_b_hist = shared.get("/history").get_data(as_text=True)
        assert f'data-uid="{uid_b}"' in html_b_hist
        assert f"secret_a_{tag}.wav" not in html_b_home
        assert f"secret_a_{tag}.wav" not in html_b_hist
        assert shared.get("/api/history").get_json()["items"] == []

        # Normal user cannot access admin history endpoints -> 403
        assert client_b.get("/api/admin/history").status_code == 403
        assert client_b.get("/api/admin/history/export.csv").status_code == 403

        # Seed a job for B so we can test admin view & account deletion cascade
        job_b = _seed_job_for_user(uid_b, f"secret_b_{tag}.wav", count=3)
        created_jobs.append(job_b)
        assert len(client_b.get("/api/history").get_json()["items"]) == 1

        # 3. User A logs back in -> sees ONLY A's items, CSV export has ONLY A's rows
        csrf_a = _login(client_a, email_a, pw)
        items_a2 = client_a.get("/api/history").get_json()["items"]
        assert [x["job_id"] for x in items_a2] == [job_a]
        csv_a = client_a.get("/api/history/export.csv")
        assert csv_a.status_code == 200
        assert "no-store" in csv_a.headers.get("Cache-Control", "").lower()
        csv_a_text = csv_a.get_data(as_text=True)
        assert f"secret_a_{tag}.wav" in csv_a_text
        assert f"secret_b_{tag}.wav" not in csv_a_text
        assert "user_email" not in csv_a_text.splitlines()[0]

        # 4. Admin logs in -> default /api/history is empty (only admin's own runs),
        #    /api/admin/history shows A & B with user_email, and admin can access A's stem/zip
        csrf_adm = _login(client_adm, email_adm, pw)
        assert client_adm.get("/api/history").get_json()["items"] == []

        r_adm_all = client_adm.get("/api/admin/history")
        assert r_adm_all.status_code == 200
        assert "no-store" in r_adm_all.headers.get("Cache-Control", "").lower()
        adm_items = r_adm_all.get_json()["items"]
        by_id = {x["job_id"]: x for x in adm_items}
        assert job_a in by_id and by_id[job_a]["user_email"] == email_a
        assert job_b in by_id and by_id[job_b]["user_email"] == email_b

        r_adm_filt = client_adm.get(f"/api/admin/history?user_id={uid_a}")
        assert [x["job_id"] for x in r_adm_filt.get_json()["items"]] == [job_a]

        csv_adm = client_adm.get("/api/admin/history/export.csv")
        assert csv_adm.status_code == 200
        csv_adm_text = csv_adm.get_data(as_text=True)
        assert "user_email" in csv_adm_text.splitlines()[0]
        assert email_a in csv_adm_text and email_b in csv_adm_text

        # Admin can access owner's stem, residual & zip (original.wav never exposed directly)
        assert client_adm.get(f"/api/stem/{job_a}/stem_01_voice.wav").status_code == 200
        assert client_adm.get(f"/api/stem/{job_a}/residual.wav").status_code == 200
        assert client_adm.get(f"/api/stem/{job_a}/original.wav").status_code == 404
        assert client_adm.get(f"/api/download/{job_a}.zip").status_code == 200

        # 5. User A deletes own history item -> DB row & disk folder removed
        assert (app_mod.JOB_DIR / job_a).is_dir()
        r_del = client_a.delete(f"/api/history/{job_a}", headers={"X-CSRF-Token": csrf_a})
        assert r_del.status_code == 200
        assert client_a.get("/api/history").get_json()["items"] == []
        assert not (app_mod.JOB_DIR / job_a).exists(), (
            "Job directory was not deleted on item delete"
        )

        # 6. Admin deletes User B's account -> cascades B's history & removes B's disk folder
        assert (app_mod.JOB_DIR / job_b).is_dir()
        r_del_u = client_adm.delete(f"/api/admin/users/{uid_b}", headers={"X-CSRF-Token": csrf_adm})
        assert r_del_u.status_code == 200
        assert not (app_mod.JOB_DIR / job_b).exists(), (
            "User B job directory was not deleted on account delete"
        )
        assert client_b.get("/api/history").status_code == 401, "Deleted user B still authenticated"

        # 7. Session revocation -> 401 on /api/history
        auth.revoke_user_sessions(uid_a)
        assert client_a.get("/api/history").status_code == 401

        # 8. Client-side JS verification: legacy key purge, user-switch & logout cache clearing
        _verify_client_storage_purge_via_node()

        print(
            "ALL PART A HISTORY PRIVACY TESTS PASSED (14/14 checks, including same-browser switch & JS storage purge)."
        )
    finally:
        mail_mod.set_provider(None)
        if prev_skip is None:
            os.environ.pop("ECHOCOUNT_SKIP_MX_CHECK", None)
        else:
            os.environ["ECHOCOUNT_SKIP_MX_CHECK"] = prev_skip
        if prev_delay is None:
            os.environ.pop("ECHOCOUNT_NO_AUTH_DELAY", None)
        else:
            os.environ["ECHOCOUNT_NO_AUTH_DELAY"] = prev_delay
        for jid in created_jobs:
            with app_mod._jobs_lock:
                app_mod._jobs.pop(jid, None)
            shutil.rmtree(app_mod.JOB_DIR / jid, ignore_errors=True)
        with auth._conn() as con:
            for uid in created_uids:
                con.execute("DELETE FROM admin_audit_log WHERE admin_id = ? OR target_id = ?", (uid, uid))
                con.execute("DELETE FROM users WHERE id = ?", (uid,))
            for em in (email_a, email_b, email_adm):
                con.execute("DELETE FROM email_otps WHERE email = ?", (em,))


if __name__ == "__main__":
    run_tests()
