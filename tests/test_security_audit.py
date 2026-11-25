"""OWASP security regression tests (weight-free, CI-safe)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import app as flask_app
import auth as auth_mod


def test_open_redirect_hardening() -> None:
    """Reject protocol-relative, backslash, scheme, and control-char redirect targets."""
    bad_targets = [
        "https://evil.example/phish",
        "http://evil.example",
        "//evil.example/path",
        r"/\evil.example",
        "/analyze\r\nSet-Cookie: a=1",
        "/analyze\x00foo",
        "javascript:alert(1)",
        "/login",
        "/register",
    ]
    expected = auth_mod.default_landing_for_role("user")
    for target in bad_targets:
        out = flask_app._safe_next_url(target, "user")
        assert out == expected, f"Expected fallback {expected!r} for {target!r}, got {out!r}"
    assert flask_app._safe_next_url("/history?page=2", "user") == "/history?page=2"
    print("  [PASS] 1. Open-redirect hardening (scheme, //, /\\, CRLF).")


def test_cross_origin_mutation_rejection() -> None:
    """Reject cross-origin and null-origin Origin headers on all mutating endpoints."""
    client = flask_app.app.test_client()
    for path in ("/auth/login", "/api/analyze", "/api/separate", "/api/predict"):
        for bad_origin in ("https://evil.example", "http://evil.example:5000", "null"):
            r = client.post(path, headers={"Origin": bad_origin})
            assert r.status_code == 403, (
                f"Expected 403 for {path} with Origin={bad_origin}, got {r.status_code}"
            )
            assert "Cross-origin" in (r.get_json() or {}).get("error", "")
    # Same-origin Origin header must pass the origin check (reaching auth/CSRF layer, not 403 origin block)
    r_same = client.post("/api/analyze", headers={"Origin": "http://localhost"})
    assert r_same.status_code == 401, (
        f"Expected 401 (auth gate) for same-origin POST, got {r_same.status_code}"
    )
    print("  [PASS] 2. Cross-origin and null-origin POST rejection.")


def test_auth_js_no_innerhtml_feedback() -> None:
    """Verify static/auth.js uses textContent (never innerHTML) in showFeedback."""
    js = (BASE_DIR / "static" / "auth.js").read_text(encoding="utf-8")
    assert "feedback.innerHTML" not in js
    assert "feedback.textContent" in js
    print("  [PASS] 3. DOM XSS fix verified in static/auth.js (textContent).")


def test_csv_formula_injection_and_path_traversal() -> None:
    """Verify CSV cells neutralize formula prefixes and stem preview rejects traversal."""
    for dangerous in ("=CMD|' /C calc'!A0", "+1+1", "-2+3", "@SUM(A1)", "\t=1", "\r=1"):
        safe = flask_app._csv_cell(dangerous)
        assert safe.startswith("'"), f"Formula cell not escaped: {dangerous!r} -> {safe!r}"
    assert flask_app._csv_cell("normal_clip.wav") == "normal_clip.wav"

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        orig_db = auth_mod.DB_PATH
        try:
            auth_mod.DB_PATH = Path(td) / "sec.db"
            auth_mod.init_db()
            uid, err, _ = auth_mod.register_password_user(
                "sec.tester@example.test", "StrongPass!2026"
            )
            assert uid and not err
            auth_mod.mark_email_verified("sec.tester@example.test")
            sid, _csrf = auth_mod.create_session(uid)
            client = flask_app.app.test_client()
            client.set_cookie(auth_mod.SESSION_COOKIE, sid)
            for bad in ("../app.py", "..\\app.py", "nested/file.wav"):
                r = client.get(f"/api/stem/{'a' * 32}/{bad}")
                assert r.status_code in (400, 404), f"Unexpected status {r.status_code} for {bad}"
            assert client.get("/api/stem/not-hex-id/stem.wav").status_code == 404
        finally:
            auth_mod.DB_PATH = orig_db
    print("  [PASS] 4. CSV formula neutralization and path-traversal guards.")


def test_security_headers_no_cors_and_generic_404() -> None:
    """Verify CSP, anti-clickjacking, no CORS wildcard, and clean 404 error responses."""
    client = flask_app.app.test_client()
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
    assert r.headers.get("X-Frame-Options") == "DENY"
    assert r.headers.get("Referrer-Policy") == "no-referrer"
    assert "frame-ancestors 'none'" in (r.headers.get("Content-Security-Policy") or "")
    assert "Access-Control-Allow-Origin" not in r.headers

    r404 = client.get("/api/does-not-exist-xyz")
    assert r404.status_code == 404
    body = r404.get_data(as_text=True)
    assert "Traceback" not in body and 'File "' not in body
    assert flask_app.app.debug is False
    print("  [PASS] 5. Security headers, no CORS, generic 404, debug=False.")


def main() -> None:
    print("Running OWASP Security Regression Suite...")
    test_open_redirect_hardening()
    test_cross_origin_mutation_rejection()
    test_auth_js_no_innerhtml_feedback()
    test_csv_formula_injection_and_path_traversal()
    test_security_headers_no_cors_and_generic_404()
    print("ALL OWASP SECURITY REGRESSION TESTS PASSED (5/5).")


if __name__ == "__main__":
    main()
