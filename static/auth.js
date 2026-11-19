/* EchoCount — Authentication UI, password visibility, strength meter, and signature wave canvas */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const toast = (window.EC && window.EC.toast) || (() => {});

  const tabs = $$("[data-auth-tab]");
  const panels = $$("[data-auth-panel]");
  const feedback = $("#authFeedback");
  const subtitle = $("#authSubtitle");

  const showFeedback = (msg, kind = "info", linkHtml = "") => {
    if (!feedback) return;
    feedback.className = `notice ${kind} auth-alert`;
    feedback.innerHTML = linkHtml ? `<span>${msg}</span> ${linkHtml}` : msg;
    feedback.classList.remove("hidden");
  };

  const switchPanel = (name) => {
    panels.forEach((p) => (p.hidden = p.dataset.authPanel !== name));
    tabs.forEach((t) => t.setAttribute("aria-selected", t.dataset.authTab === name ? "true" : "false"));
    if (feedback) feedback.classList.add("hidden");
    if (subtitle) {
      subtitle.textContent =
        name === "register"
          ? "Create an EchoCount account with your verified email."
          : name === "forgot"
            ? "Request a password reset link."
            : "Welcome back. Sign in to analyze and separate recordings.";
    }
  };

  tabs.forEach((btn) => btn.addEventListener("click", () => switchPanel(btn.dataset.authTab)));
  $("#openForgotBtn")?.addEventListener("click", () => switchPanel("forgot"));
  $("#backToLoginBtn")?.addEventListener("click", () => switchPanel("login"));

  /* ---- Password visibility toggle (keyboard accessible, preserves cursor) ---- */
  $$("[data-pw-toggle]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const input = document.getElementById(btn.dataset.pwToggle);
      if (!input) return;
      const start = input.selectionStart;
      const end = input.selectionEnd;
      const show = input.type === "password";
      input.type = show ? "text" : "password";
      btn.setAttribute("aria-pressed", show ? "true" : "false");
      btn.setAttribute("aria-label", show ? "Hide password" : "Show password");
      input.focus({ preventScroll: true });
      if (typeof start === "number" && typeof end === "number") {
        try {
          input.setSelectionRange(start, end);
        } catch {}
      }
    });
  });

  /* ---- Password strength meter ---- */
  const regPw = $("#regPassword");
  const pwBar = $("#pwStrengthBar");
  const pwText = $("#pwStrengthText");
  if (regPw && pwBar && pwText) {
    regPw.addEventListener("input", () => {
      const v = regPw.value;
      let score = 0;
      if (v.length >= 10) score++;
      if (v.length >= 14) score++;
      if (/[a-z]/.test(v) && /[A-Z]/.test(v)) score++;
      if (/\d/.test(v) && /[^a-zA-Z0-9]/.test(v)) score++;
      const pct = Math.max(8, score * 25);
      pwBar.style.width = v ? `${pct}%` : "0%";
      pwBar.dataset.level = String(score);
      const labels = [
        "Too short — use at least 10 characters.",
        "Fair — add uppercase, numbers, or symbols.",
        "Good — solid length.",
        "Strong password.",
        "Very strong password.",
      ];
      pwText.textContent = v ? labels[score] : "Use 10+ characters with uppercase, lowercase, numbers, or symbols.";
    });
  }

  /* ---- AJAX form handlers ---- */
  const wireForm = (formId, onSuccess) => {
    const form = $(formId);
    if (!form) return;
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      if (formId === "#registerForm") {
        const p1 = $("#regPassword")?.value || "";
        const p2 = $("#regPasswordConfirm")?.value || "";
        if (p1 !== p2) {
          showFeedback("Passwords do not match.", "warn");
          $("#regPasswordConfirm")?.focus();
          return;
        }
      }
      const btn = $(".auth-submit", form);
      const orig = btn ? btn.textContent : "";
      if (btn) {
        btn.disabled = true;
        btn.textContent = "Working…";
      }
      try {
        const body = Object.fromEntries(new FormData(form).entries());
        const csrf = body.csrf_token || "";
        const res = await fetch(form.action, {
          method: "POST",
          credentials: "same-origin",
          headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf },
          body: JSON.stringify(body),
        });
        const data = await res.json();
        if (!res.ok) {
          showFeedback(data.error || "Request failed.", "warn");
          return;
        }
        onSuccess(data);
      } catch {
        showFeedback("Network error. Please try again.", "warn");
      } finally {
        if (btn) {
          btn.disabled = false;
          btn.textContent = orig;
        }
      }
    });
  };

  wireForm("#loginForm", (data) => {
    toast("Signed in.");
    location.href = data.redirect || "/analyze";
  });

  wireForm("#registerForm", (data) => {
    const link = data.verify_url
      ? `<a class="btn sm" href="${data.verify_url}">Verify email now</a>`
      : "";
    showFeedback(data.message || "Check your inbox to verify your email.", "info", link);
  });

  wireForm("#forgotForm", (data) => {
    const link = data.reset_url
      ? `<a class="btn sm" href="${data.reset_url}">Open reset link</a>`
      : "";
    showFeedback(data.message || "Reset instructions prepared.", "info", link);
  });

  wireForm("#resetForm", (data) => {
    showFeedback(data.message || "Password updated.", "info");
    setTimeout(() => (location.href = data.redirect || "/login"), 1200);
  });

  /* ---- Signature layered audio-wave canvas (respects prefers-reduced-motion) ---- */
  const cv = $("#authWaveCanvas");
  if (cv) {
    const ctx = cv.getContext("2d");
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    let w = 0;
    let h = 0;
    let t = 0;
    let rafId = 0;

    const fit = () => {
      const r = cv.getBoundingClientRect();
      const d = Math.min(devicePixelRatio || 1, 2);
      w = Math.max(1, r.width);
      h = Math.max(1, r.height);
      cv.width = Math.round(w * d);
      cv.height = Math.round(h * d);
      ctx.setTransform(d, 0, 0, d, 0, 0);
    };

    const draw = () => {
      ctx.clearRect(0, 0, w, h);
      const waves = [
        { amp: 24, freq: 1.8, speed: 0.9, col: "rgba(255, 90, 31, 0.55)", lw: 2.2 },
        { amp: 18, freq: 2.7, speed: 1.3, col: "rgba(200, 240, 74, 0.45)", lw: 1.8 },
        { amp: 14, freq: 3.9, speed: 0.65, col: "rgba(139, 92, 246, 0.42)", lw: 1.4 },
      ];
      waves.forEach((wv, idx) => {
        ctx.beginPath();
        ctx.strokeStyle = wv.col;
        ctx.lineWidth = wv.lw;
        for (let x = 0; x <= w; x += 4) {
          const env = Math.sin((x / w) * Math.PI);
          const y =
            h * 0.52 +
            Math.sin((x / w) * Math.PI * 2 * wv.freq + t * wv.speed + idx) * wv.amp * env;
          x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
      });
    };

    const loop = () => {
      t += 0.02;
      draw();
      rafId = requestAnimationFrame(loop);
    };

    fit();
    addEventListener("resize", fit);
    if (reduced) {
      draw();
    } else {
      rafId = requestAnimationFrame(loop);
    }
    addEventListener("pagehide", () => cancelAnimationFrame(rafId));
  }
})();
