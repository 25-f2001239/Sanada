/* EchoCount — "Sound Made Visible" Authentication, Coordinated Signal Decomposition & Wave Field */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const toast = (window.EC && window.EC.toast) || (() => {});
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  const tabsWrap = $(".auth-mode-tabs");
  const tabs = $$("[data-auth-tab]");
  const panels = $$("[data-auth-panel]");
  const formBox = $("#authFormBox");
  const feedback = $("#authFeedback");
  const heading = $("#authHeading");
  const subtitle = $("#authSubtitle");

  $("#authThemeBtn")?.addEventListener("click", () => window.EC?.toggleTheme?.());

  /* ---- Full-Page Live Acoustic Scene + 2.5-Second Skippable Intro ---- */
  const authRoot = $(".auth-editorial");
  const curtain = $("#authIntroCurtain");
  const INTRO_KEY = "echocount.login_intro_seen";
  let introSeen = false;
  try {
    introSeen = sessionStorage.getItem(INTRO_KEY) === "1";
  } catch {}
  const playIntro = !reduced && !introSeen;

  const bgCanvas = $("#loginWaveField");
  const bgBtn = $("#bgWaveBtn");
  const liteBtn = $("#bgLiteBtn");
  const waveCtrl =
    bgCanvas && window.initWaveField
      ? window.initWaveField(bgCanvas, {
          intensity: "full",
          pointerInteraction: true,
          intro: playIntro,
          colorMode: document.documentElement.dataset.theme || "dark",
          onQualityChange: () => {
            if (liteBtn) {
              liteBtn.hidden = false;
              liteBtn.textContent = "Lite mode: ON";
            }
          },
        })
      : null;
  window.__ecLoginWaveCtrl = waveCtrl;

  if (liteBtn && waveCtrl) {
    liteBtn.addEventListener("click", () => {
      const next = !waveCtrl.isLite();
      waveCtrl.setLite(next);
      liteBtn.textContent = next ? "Lite mode: ON" : "Lite mode: OFF";
    });
  }

  if (playIntro && authRoot) {
    authRoot.classList.add("auth-intro-playing");
    requestAnimationFrame(() => curtain?.classList.add("is-fading"));
    let introDone = false;
    const finishIntro = (skipped) => {
      if (introDone) return;
      introDone = true;
      try {
        sessionStorage.setItem(INTRO_KEY, "1");
      } catch {}
      curtain?.remove();
      authRoot.classList.remove("auth-intro-playing");
      if (skipped) {
        authRoot.classList.add("auth-intro-skip");
        waveCtrl?.skipIntro?.();
      }
      authRoot.classList.add("auth-intro-done");
      window.removeEventListener("pointerdown", onSkip);
      window.removeEventListener("keydown", onSkip);
    };
    const onSkip = () => finishIntro(true);
    window.addEventListener("pointerdown", onSkip, { once: true, passive: true });
    window.addEventListener("keydown", onSkip, { once: true });
    setTimeout(() => finishIntro(false), 2500);
  } else {
    curtain?.remove();
    if (authRoot && introSeen) {
      authRoot.classList.add("auth-intro-skip", "auth-intro-done");
    }
  }

  if (bgBtn && waveCtrl) {
    const syncBgBtn = () => {
      const p = waveCtrl.isPaused();
      bgBtn.textContent = p ? "Play background" : "Pause background";
      bgBtn.setAttribute("aria-label", p ? "Play background wave field" : "Pause background wave field");
    };
    syncBgBtn();
    bgBtn.addEventListener("click", () => {
      waveCtrl.toggle();
      syncBgBtn();
    });
  }

  /* ---- Count-up statistics once on page load ---- */
  if (!reduced) {
    $$("[data-auth-count]").forEach((el) => {
      const target = Number(el.dataset.authCount || 0);
      if (!Number.isFinite(target) || target <= 0) return;
      const t0 = performance.now();
      const dur = 680;
      const step = (now) => {
        const p = Math.min(1, (now - t0) / dur);
        const eased = 1 - Math.pow(1 - p, 3);
        el.textContent = String(Math.round(target * eased));
        if (p < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
  }

  /* ---- Typing waveform underline + whole-scene amplitude reactivity ---- */
  $$(".input-wave-wrap input").forEach((inp) => {
    let timer = 0;
    inp.addEventListener("input", () => {
      const wrap = inp.closest(".input-wave-wrap");
      if (!wrap) return;
      wrap.classList.add("is-typing");
      waveCtrl?.boostTyping?.(0.45);
      clearTimeout(timer);
      timer = setTimeout(() => wrap.classList.remove("is-typing"), 520);
    });
  });

  const showFeedback = (msg, kind = "info") => {
    if (!feedback) return;
    feedback.className = `notice ${kind} auth-alert is-visible`;
    feedback.textContent = String(msg || "");
    feedback.classList.remove("hidden");
    if (kind === "warn" && formBox && !reduced) {
      formBox.classList.remove("is-shake");
      void formBox.offsetWidth;
      formBox.classList.add("is-shake");
    }
  };

  const switchPanel = (name) => {
    const showTabs = name === "login" || name === "register";
    if (tabsWrap) {
      tabsWrap.hidden = !showTabs;
      if (showTabs) tabsWrap.dataset.activeTab = name;
    }
    panels.forEach((p) => {
      const active = p.dataset.authPanel === name;
      p.hidden = !active;
      if (active) {
        p.classList.remove("panel-fade-in");
        void p.offsetWidth;
        p.classList.add("panel-fade-in");
      }
    });
    tabs.forEach((t) => t.setAttribute("aria-selected", t.dataset.authTab === name ? "true" : "false"));
    if (feedback) feedback.classList.add("hidden");
    if (heading) {
      heading.textContent =
        name === "register"
          ? "Create your account"
          : name === "forgot" || name === "forgot_reset"
            ? "Reset your password"
            : name === "otp"
              ? "Verify your email"
              : "Sign in to EchoCount";
    }
    if (subtitle) {
      subtitle.textContent =
        name === "register"
          ? "Register with a verified email address to analyze recordings and track your history."
          : name === "forgot"
            ? "Enter your account email to receive a 6-digit password reset code."
            : name === "forgot_reset"
              ? "Enter the 6-digit code and choose a new password."
              : name === "otp"
                ? "Enter the 6-digit verification code sent to your email address."
                : "Enter your email and password to access the acoustic analyzer.";
    }
  };

  tabs.forEach((btn) => btn.addEventListener("click", () => switchPanel(btn.dataset.authTab)));
  $("#openForgotBtn")?.addEventListener("click", () => switchPanel("forgot"));
  $("#backToLoginBtn")?.addEventListener("click", () => switchPanel("login"));
  $("#otpBackBtn")?.addEventListener("click", () => switchPanel("login"));
  $("#fgBackBtn")?.addEventListener("click", () => switchPanel("login"));
  if ($(".auth-panel:not([hidden])")?.dataset.authPanel === "forgot" && tabsWrap) tabsWrap.hidden = true;

  // Google OAuth button loading feedback (keyboard & pointer)
  $$("[data-google-link]").forEach((el) => {
    el.addEventListener("click", () => {
      el.classList.add("is-loading");
      el.setAttribute("aria-busy", "true");
    });
  });

  // Friendly status banners after account deactivation or deletion
  const qs = new URLSearchParams(location.search);
  if (qs.get("deactivated") === "1") {
    showFeedback("Your account has been deactivated and all sessions were signed out. Sign in anytime to reactivate it.", "info");
  } else if (qs.get("deleted") === "1") {
    showFeedback("Your account and all associated data have been permanently deleted.", "info");
  }

  /* ---- Reusable 6-Digit OTP Box Wiring & Countdown Timers ---- */
  const wireOtpBoxes = (boxes, onSixDigits) => {
    const readCode = () => boxes.map((b) => (b.value || "").replace(/\D/g, "")).join("");
    boxes.forEach((box, idx) => {
      box.addEventListener("input", () => {
        const digits = (box.value || "").replace(/\D/g, "");
        if (digits.length > 1) {
          digits.slice(0, 6).split("").forEach((d, j) => { if (boxes[idx + j]) boxes[idx + j].value = d; });
          boxes[Math.min(5, idx + digits.length)]?.focus();
        } else {
          box.value = digits;
          if (digits && idx < boxes.length - 1) boxes[idx + 1].focus();
        }
        if (onSixDigits && readCode().length === 6) onSixDigits();
      });
      box.addEventListener("keydown", (e) => {
        if (e.key === "Backspace" && !box.value && idx > 0) {
          boxes[idx - 1].value = "";
          boxes[idx - 1].focus();
        } else if (e.key === "ArrowLeft" && idx > 0) boxes[idx - 1].focus();
        else if (e.key === "ArrowRight" && idx < boxes.length - 1) boxes[idx + 1].focus();
      });
      box.addEventListener("paste", (e) => {
        const txt = (e.clipboardData?.getData("text") || "").replace(/\D/g, "");
        if (!txt) return;
        e.preventDefault();
        txt.slice(0, 6).split("").forEach((d, j) => { if (boxes[j]) boxes[j].value = d; });
        boxes[Math.min(5, txt.length - 1)]?.focus();
        if (onSixDigits && readCode().length === 6) onSixDigits();
      });
    });
    return {
      code: readCode,
      clear: () => { boxes.forEach((b) => (b.value = "")); boxes[0]?.focus(); },
    };
  };

  let expTimer = 0, cdTimer = 0;
  const startTimers = (timerEl, resendEl, cooldownSec = 60, ttlSec = 600) => {
    clearInterval(expTimer);
    clearInterval(cdTimer);
    let expLeft = Math.max(0, Number(ttlSec) || 600);
    let cdLeft = Math.max(0, Number(cooldownSec) || 60);
    const drawExp = () => {
      if (!timerEl) return;
      const m = Math.floor(expLeft / 60), s = String(expLeft % 60).padStart(2, "0");
      timerEl.textContent = expLeft > 0 ? `Expires in ${m}:${s}` : "Code expired — resend a new code";
    };
    const drawCd = () => {
      if (!resendEl) return;
      resendEl.disabled = cdLeft > 0;
      resendEl.textContent = cdLeft > 0 ? `Resend code (${cdLeft}s)` : "Resend code";
    };
    drawExp(); drawCd();
    expTimer = setInterval(() => { expLeft = Math.max(0, expLeft - 1); drawExp(); if (expLeft <= 0) clearInterval(expTimer); }, 1000);
    cdTimer = setInterval(() => { cdLeft = Math.max(0, cdLeft - 1); drawCd(); if (cdLeft <= 0) clearInterval(cdTimer); }, 1000);
  };

  const otpForm = $("#otpForm");
  const otpPurposeIn = $("#otpPurpose");
  const otpEmailIn = $("#otpEmail");
  const otpMaskedEl = $("#otpMaskedEmail");
  const otpChangeBtn = $("#otpChangeEmailBtn");
  const otpCtrl = wireOtpBoxes($$("#otpGrid .otp-box"), () => otpForm?.requestSubmit());

  const fgForm = $("#forgotResetForm");
  const fgEmailIn = $("#fgEmail");
  const fgMaskedEl = $("#fgMaskedEmail");
  const fgCtrl = wireOtpBoxes($$("#fgOtpGrid .fg-otp-box"), null);

  const maskClientEmail = (em) => {
    if (!em || !em.includes("@")) return "your email";
    const [loc, dom] = em.split("@", 2);
    const parts = dom.split(".");
    return `${loc[0]}***@${parts[0][0]}***.${parts.slice(1).join(".")}`;
  };

  const openOtpPanel = (purpose, email, message, cooldown = 60, masked = "") => {
    if (otpPurposeIn) otpPurposeIn.value = purpose || "signup";
    if (otpEmailIn) otpEmailIn.value = email || "";
    if (otpMaskedEl) otpMaskedEl.textContent = masked || maskClientEmail(email);
    if (otpChangeBtn) otpChangeBtn.hidden = purpose === "admin_login";
    switchPanel("otp");
    if (message) showFeedback(message, "info");
    otpCtrl.clear();
    startTimers($("#otpTimerText"), $("#otpResendBtn"), cooldown, 600);
  };

  const openForgotResetPanel = (email, message, cooldown = 60, masked = "") => {
    if (fgEmailIn) fgEmailIn.value = email || "";
    if (fgMaskedEl) fgMaskedEl.textContent = masked || maskClientEmail(email);
    const pw1 = $("#fgNewPassword"), pw2 = $("#fgNewPasswordConfirm");
    if (pw1) pw1.value = "";
    if (pw2) pw2.value = "";
    switchPanel("forgot_reset");
    if (message) showFeedback(message, "info");
    fgCtrl.clear();
    startTimers($("#fgTimerText"), $("#fgResendBtn"), cooldown, 600);
  };

  const wireResend = (btnId, getForm, getPurpose, getEmail, timerEl, ctrl) => {
    $(btnId)?.addEventListener("click", async () => {
      const form = getForm();
      const btn = $(btnId);
      if (!form || !btn) return;
      const csrf = $('input[name="csrf_token"]', form)?.value || "";
      btn.disabled = true;
      try {
        const res = await fetch("/auth/otp/resend", {
          method: "POST",
          credentials: "same-origin",
          headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf },
          body: JSON.stringify({ purpose: getPurpose(), email: getEmail() }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          showFeedback(data.error || "Could not resend code.", "warn");
          if (data.cooldown) startTimers(timerEl, btn, data.cooldown, 600);
          else btn.disabled = false;
          return;
        }
        showFeedback(data.message || "A new 6-digit code has been sent.", "info");
        ctrl.clear();
        startTimers(timerEl, btn, data.cooldown || 60, 600);
      } catch {
        showFeedback("Network error. Please try again.", "warn");
        btn.disabled = false;
      }
    });
  };
  wireResend("#otpResendBtn", () => otpForm, () => otpPurposeIn?.value || "signup", () => otpEmailIn?.value || "", $("#otpTimerText"), otpCtrl);
  wireResend("#fgResendBtn", () => fgForm, () => "forgot", () => fgEmailIn?.value || "", $("#fgTimerText"), fgCtrl);

  otpChangeBtn?.addEventListener("click", async () => {
    const csrf = $('input[name="csrf_token"]', otpForm)?.value || "";
    const typed = otpEmailIn?.value || $("#regEmail")?.value || "";
    try {
      await fetch("/auth/signup/cancel", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf },
        body: JSON.stringify({}),
      });
    } catch {}
    switchPanel("register");
    const regEm = $("#regEmail");
    if (regEm && typed) regEm.value = typed;
    if ($("#regPassword")) $("#regPassword").value = "";
    if ($("#regPasswordConfirm")) $("#regPasswordConfirm").value = "";
    regEm?.focus();
  });

  /* ---- Password visibility toggle with smooth icon rotation ---- */
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
      btn.classList.add("is-rotating");
      setTimeout(() => btn.classList.remove("is-rotating"), 260);
      input.focus({ preventScroll: true });
      if (typeof start === "number" && typeof end === "number") {
        try { input.setSelectionRange(start, end); } catch {}
      }
    });
  });

  /* ---- Password strength meter & live confirm-password matching ---- */
  const regPw = $("#regPassword"), regPw2 = $("#regPasswordConfirm");
  const pwBar = $("#pwStrengthBar"), pwText = $("#pwStrengthText");
  if (regPw && pwBar && pwText) {
    regPw.addEventListener("input", () => {
      const v = regPw.value;
      let score = 0;
      if (v.length >= 10) score++;
      if (v.length >= 14) score++;
      if (/[a-z]/.test(v) && /[A-Z]/.test(v)) score++;
      if (/\d/.test(v) && /[^a-zA-Z0-9]/.test(v)) score++;
      pwBar.style.width = v ? `${Math.max(8, score * 25)}%` : "0%";
      pwBar.dataset.level = String(score);
      const labels = ["Too short — use at least 10 characters.", "Fair — add uppercase, numbers, or symbols.", "Good — solid length.", "Strong password.", "Very strong password."];
      pwText.textContent = v ? labels[score] : "Use 10+ characters; common weak passwords are rejected.";
    });
  }
  const wirePwMatch = (a, b, msgEl) => {
    if (!a || !b || !msgEl) return;
    const sync = () => {
      if (!b.value) { msgEl.textContent = ""; return; }
      msgEl.textContent = a.value === b.value ? "Passwords match." : "Passwords do not match.";
    };
    a.addEventListener("input", sync);
    b.addEventListener("input", sync);
  };
  wirePwMatch(regPw, regPw2, $("#pwMatchText"));
  wirePwMatch($("#fgNewPassword"), $("#fgNewPasswordConfirm"), $("#fgMatchText"));

  /* ---- AJAX form handlers with inline waveform loader & <=400ms success transition ---- */
  const wireForm = (formId, onSuccess, preCheck) => {
    const form = $(formId);
    if (!form) return;
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      if (preCheck) {
        const err = preCheck(form);
        if (err) { showFeedback(err, "warn"); return; }
      }
      const btn = $(".auth-submit", form);
      const lbl = $(".auth-action-label", btn || form);
      const orig = lbl ? lbl.textContent : "";
      if (btn) { btn.disabled = true; btn.classList.add("is-loading"); }
      if (lbl) lbl.textContent = "Working…";
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
        if (!res.ok) { showFeedback(data.error || "Request failed.", "warn"); return; }
        onSuccess(data);
      } catch {
        showFeedback("Network error. Please try again.", "warn");
      } finally {
        if (btn) { btn.disabled = false; btn.classList.remove("is-loading"); }
        if (lbl) lbl.textContent = orig;
      }
    });
  };

  wireForm("#loginForm", (data) => {
    if (data.otp_required) {
      openOtpPanel(data.purpose || "signup", $("#loginEmail")?.value || "", data.message, data.cooldown || 60, data.masked_email || "");
      return;
    }
    toast("Signed in.");
    formBox?.classList.add("is-success");
    waveCtrl?.convergeSuccess?.();
    setTimeout(() => { location.href = data.redirect || "/"; }, reduced ? 40 : 420);
  });

  wireForm(
    "#registerForm",
    (data) => {
      if (data.otp_required) {
        openOtpPanel(data.purpose || "signup", $("#regEmail")?.value || "", data.message, data.cooldown || 60, data.masked_email || "");
        return;
      }
      showFeedback(data.message || "Check your inbox for your 6-digit verification code.", "info");
    },
    () => (regPw && regPw2 && regPw.value !== regPw2.value ? "Passwords do not match." : null),
  );

  wireForm("#forgotForm", (data) => {
    if (data.otp_required) {
      openForgotResetPanel($("#forgotEmail")?.value || "", data.message, data.cooldown || 60, data.masked_email || "");
      return;
    }
    showFeedback(data.message || "If an eligible account exists, a reset code was sent.", "info");
  });

  otpForm?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const code = otpCtrl.code();
    if (code.length !== 6) {
      showFeedback("Please enter all 6 digits of your verification code.", "warn");
      return;
    }
    const btn = $(".auth-submit", otpForm);
    const lbl = $(".auth-action-label", btn || otpForm);
    const orig = lbl ? lbl.textContent : "";
    if (btn) { btn.disabled = true; btn.classList.add("is-loading"); }
    if (lbl) lbl.textContent = "Verifying…";
    try {
      const csrf = $('input[name="csrf_token"]', otpForm)?.value || "";
      const res = await fetch(otpForm.action, {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf },
        body: JSON.stringify({ purpose: otpPurposeIn?.value || "signup", email: otpEmailIn?.value || "", code }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) { showFeedback(data.error || "Invalid or expired verification code.", "warn"); return; }
      toast("Verified and signed in.");
      formBox?.classList.add("is-success");
      waveCtrl?.convergeSuccess?.();
      setTimeout(() => { location.href = data.redirect || "/analyze"; }, reduced ? 40 : 420);
    } catch {
      showFeedback("Network error. Please try again.", "warn");
    } finally {
      if (btn) { btn.disabled = false; btn.classList.remove("is-loading"); }
      if (lbl) lbl.textContent = orig;
    }
  });

  fgForm?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const code = fgCtrl.code();
    const pw1 = $("#fgNewPassword")?.value || "";
    const pw2 = $("#fgNewPasswordConfirm")?.value || "";
    if (code.length !== 6) { showFeedback("Please enter all 6 digits of your reset code.", "warn"); return; }
    if (pw1 !== pw2) { showFeedback("Passwords do not match.", "warn"); return; }
    const btn = $(".auth-submit", fgForm);
    const lbl = $(".auth-action-label", btn || fgForm);
    const orig = lbl ? lbl.textContent : "";
    if (btn) { btn.disabled = true; btn.classList.add("is-loading"); }
    if (lbl) lbl.textContent = "Updating…";
    try {
      const csrf = $('input[name="csrf_token"]', fgForm)?.value || "";
      const res = await fetch(fgForm.action, {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf },
        body: JSON.stringify({ purpose: "forgot", email: fgEmailIn?.value || "", code, password: pw1, password_confirm: pw2 }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) { showFeedback(data.error || "Could not update password.", "warn"); return; }
      switchPanel("login");
      showFeedback(data.message || "Password updated. You can now sign in.", "info");
    } catch {
      showFeedback("Network error. Please try again.", "warn");
    } finally {
      if (btn) { btn.disabled = false; btn.classList.remove("is-loading"); }
      if (lbl) lbl.textContent = orig;
    }
  });

  /* ========================================================================
     LEFT PANEL: COORDINATED SIGNAL DECOMPOSITION CYCLE (#authScopeCanvas)
     Cycle (10.0s):
       0.00 - 0.34: SEPARATE (3 individual sources enter 1 -> 2 -> 3)
       0.34 - 0.66: SUMMED   (signals converge into composite Σ(t) + ±σ band)
       0.66 - 1.00: SPLIT    (composite separates back into 3 source tracks)
     ======================================================================== */
  const cv = $("#authScopeCanvas");
  const motionBtn = $("#authMotionBtn");
  const autoBtn = $("#scopeAutoBtn");
  const phaseLbl = $("#scopePhaseLabel");
  const countLbl = $("#scopeSourceCount");
  const legSpans = $$("#authScopeLegend [data-leg]");
  const srcToggles = $$("#authScopeLegend [data-src-toggle]");

  if (cv) {
    const ctx = cv.getContext("2d");
    let paused = reduced;
    let w = 0;
    let h = 0;
    let t = 1.6;
    let cycle = 0.08; // 0..1 across 10s
    let rafId = 0;
    let lastTs = 0;
    let interactive = false;
    let idleTimer = 0;
    const enabledSrc = [true, true, true];

    const syncHandles = () => {
      srcToggles.forEach((btn) => {
        const idx = Number(btn.dataset.srcToggle);
        const on = Boolean(enabledSrc[idx]);
        btn.classList.toggle("is-on", on);
        btn.setAttribute("aria-checked", on ? "true" : "false");
      });
      if (autoBtn) autoBtn.hidden = !interactive;
    };

    const toggleSource = (idx) => {
      if (idx < 0 || idx > 2) return;
      interactive = true;
      enabledSrc[idx] = !enabledSrc[idx];
      syncHandles();
      clearTimeout(idleTimer);
      idleTimer = setTimeout(() => {
        interactive = false;
        enabledSrc[0] = enabledSrc[1] = enabledSrc[2] = true;
        syncHandles();
        draw();
      }, 12000);
      draw();
    };

    srcToggles.forEach((btn) =>
      btn.addEventListener("click", () => toggleSource(Number(btn.dataset.srcToggle))),
    );
    autoBtn?.addEventListener("click", () => {
      clearTimeout(idleTimer);
      interactive = false;
      enabledSrc[0] = enabledSrc[1] = enabledSrc[2] = true;
      syncHandles();
      draw();
    });
    cv.addEventListener("click", (e) => {
      const r = cv.getBoundingClientRect();
      if (!r.height) return;
      const ry = (e.clientY - r.top) / r.height;
      if (ry < 0.30) toggleSource(0);
      else if (ry < 0.54) toggleSource(1);
      else if (ry < 0.78) toggleSource(2);
    });

    const syncBtn = () => {
      if (!motionBtn) return;
      motionBtn.textContent = paused ? "Play signal" : "Pause signal";
      motionBtn.setAttribute("aria-label", paused ? "Play waveform animation" : "Pause waveform animation");
    };

    const fit = () => {
      const r = cv.getBoundingClientRect();
      const d = Math.min(devicePixelRatio || 1, 2);
      w = Math.max(1, r.width);
      h = Math.max(1, r.height);
      cv.width = Math.round(w * d);
      cv.height = Math.round(h * d);
      ctx.setTransform(d, 0, 0, d, 0, 0);
      draw();
    };

    const s1 = (u, ph) =>
      Math.sin(u * 20 - ph * 1.4) * Math.exp(-(((u - 0.28) / 0.16) ** 2)) +
      0.7 * Math.sin(u * 26 - ph * 1.1) * Math.exp(-(((u - 0.72) / 0.14) ** 2));
    const s2 = (u, ph) =>
      (0.75 * Math.sin(u * 14 + ph * 0.9) + 0.35 * Math.cos(u * 28 - ph * 0.6)) *
      Math.sin(u * Math.PI);
    const s3 = (u, ph) =>
      0.85 * Math.cos(u * 38 - ph * 1.9) * Math.exp(-(((u - 0.52) / 0.11) ** 2));

    function computeState(c) {
      // Returns mergeFactor (0 = 3 separate lanes, 1 = converged at center), activeSources (1..3), stateName
      if (c < 0.34) {
        const sub = c / 0.34;
        const activeCount = sub < 0.33 ? 1 : sub < 0.66 ? 2 : 3;
        return { merge: 0, count: activeCount, state: "SEPARATE", sumGlow: 0.28 };
      }
      if (c < 0.66) {
        const sub = (c - 0.34) / 0.32;
        const merge = sub < 0.35 ? sub / 0.35 : 1;
        return { merge, count: 3, state: "SUMMED · Σ(t)", sumGlow: 1.0 };
      }
      const sub = (c - 0.66) / 0.34;
      const merge = Math.max(0, 1 - sub / 0.55);
      return { merge, count: 3, state: "SPLIT", sumGlow: 0.35 + 0.65 * merge };
    }

    function draw() {
      if (!w || !h) return;
      ctx.clearRect(0, 0, w, h);

      const st = interactive
        ? {
            merge: 0,
            count: enabledSrc.filter(Boolean).length,
            state: "TRY IT · LIVE MIX",
            sumGlow: enabledSrc.some(Boolean) ? 0.85 : 0.12,
          }
        : computeState(cycle);
      const activeMask = interactive
        ? enabledSrc
        : [st.count >= 1, st.count >= 2, st.count >= 3];

      if (phaseLbl && phaseLbl.textContent !== st.state) phaseLbl.textContent = st.state;
      if (countLbl && countLbl.textContent !== String(st.count)) countLbl.textContent = String(st.count);

      legSpans.forEach((sp) => {
        const k = sp.dataset.leg;
        const on =
          k === "sum"
            ? st.sumGlow > 0.6
            : activeMask[Number(k)] && st.merge < 0.75;
        sp.classList.toggle("is-lit", on);
      });

      const baseRows = [0.18, 0.42, 0.66];
      const sumY = h * 0.52;
      const standY = h * 0.86;

      // Graticule lines
      ctx.strokeStyle = "rgba(148, 163, 184, 0.08)";
      ctx.lineWidth = 1;
      [0.18, 0.42, 0.66, 0.86].forEach((ry) => {
        ctx.beginPath();
        ctx.moveTo(0, h * ry);
        ctx.lineTo(w, h * ry);
        ctx.stroke();
      });

      const fns = [
        { col: "#FF5A1F", fn: (u) => s1(u, t), idx: 0 },
        { col: "#12B5A6", fn: (u) => s2(u, t), idx: 1 },
        { col: "#8B5CF6", fn: (u) => s3(u, t), idx: 2 },
      ];

      const step = Math.max(3, Math.round(w / 150));

      // Draw 3 individual sources (interpolating toward sumY as st.merge -> 1)
      fns.forEach((tr) => {
        const active = activeMask[tr.idx];
        const laneY = h * baseRows[tr.idx] * (1 - st.merge) + sumY * st.merge;
        const amp = h * 0.072 * (active ? 1 - st.merge * 0.55 : 0.12);
        ctx.beginPath();
        for (let x = 0; x <= w; x += step) {
          const u = x / w;
          const y = laneY - tr.fn(u) * amp;
          if (x === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = tr.col;
        ctx.globalAlpha = active ? 0.92 - st.merge * 0.55 : 0.16;
        ctx.lineWidth = 1.6;
        ctx.stroke();
      });

      // Draw Composite Mixture Σ(t) + Uncertainty Band (±σ shimmer)
      const compY = standY * (1 - st.merge) + sumY * st.merge;
      const compAmp = h * (0.055 + 0.038 * st.sumGlow);
      const uncSpread = (st.count > 0 ? 4 : 1.5) + 5 * st.sumGlow * (0.65 + 0.35 * Math.sin(t * 2.4));

      // Uncertainty band fill
      ctx.beginPath();
      const pts = [];
      for (let x = 0; x <= w; x += step) {
        const u = x / w;
        const val =
          ((activeMask[0] ? s1(u, t) : 0) +
            (activeMask[1] ? s2(u, t) : 0) +
            (activeMask[2] ? s3(u, t) : 0)) *
          0.56;
        pts.push([x, compY - val * compAmp]);
      }
      pts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y - uncSpread) : ctx.lineTo(x, y - uncSpread)));
      for (let i = pts.length - 1; i >= 0; i--) ctx.lineTo(pts[i][0], pts[i][1] + uncSpread);
      ctx.closePath();
      ctx.globalAlpha = 0.12 + 0.14 * st.sumGlow;
      ctx.fillStyle = "#FFB800";
      ctx.fill();

      // Composite centerline
      ctx.beginPath();
      pts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
      ctx.globalAlpha = 0.55 + 0.45 * st.sumGlow;
      ctx.strokeStyle = "#FFB800";
      ctx.lineWidth = 1.6 + 0.9 * st.sumGlow;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    const tickerStages = $$("#authPipelineTicker [data-pt]");
    let lastTickerIdx = -1;
    function syncTicker() {
      if (!tickerStages.length) return;
      const idx = Math.floor(cycle * tickerStages.length) % tickerStages.length;
      if (idx === lastTickerIdx) return;
      lastTickerIdx = idx;
      tickerStages.forEach((el, i) => el.classList.toggle("is-active", i === idx));
    }

    function loop(ts) {
      if (paused || document.hidden) {
        rafId = 0;
        return;
      }
      if (!lastTs) lastTs = ts;
      const dt = Math.min(0.08, (ts - lastTs) / 1000);
      lastTs = ts;
      t += dt * 1.15;
      cycle = (cycle + dt / 10.0) % 1;
      syncTicker();
      draw();
      rafId = requestAnimationFrame(loop);
    }

    // Expose helper so browser tests can inspect or jump to separate / summed / split states
    window.__ecSetScopePhase = (p) => {
      cycle = Number(p) % 1;
      syncTicker();
      draw();
    };

    motionBtn?.addEventListener("click", () => {
      paused = !paused;
      syncBtn();
      cancelAnimationFrame(rafId);
      lastTs = 0;
      if (!paused) rafId = requestAnimationFrame(loop);
      else draw();
    });

    syncBtn();
    fit();
    addEventListener("resize", fit);
    if (!paused) rafId = requestAnimationFrame(loop);
    addEventListener("pagehide", () => cancelAnimationFrame(rafId));
  }
})();
