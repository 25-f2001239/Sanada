/* EchoCount — "Live Acoustic Scene" Shared Canvas 2D Engine (Far Waterfall + Mid Additive Waves + Near Particles/Ripples) */
(() => {
  const PALETTE = {
    dark: [
      { rgb: [255, 90, 31], a: 0.56, amp: 32, freq: 1.35, spd: 0.78, lw: 2.2, depth: 1.0 },
      { rgb: [18, 181, 166], a: 0.52, amp: 27, freq: 1.95, spd: -0.62, lw: 2.0, depth: 0.94 },
      { rgb: [139, 92, 246], a: 0.48, amp: 23, freq: 2.55, spd: 0.52, lw: 1.85, depth: 0.86 },
      { rgb: [56, 189, 248], a: 0.46, amp: 19, freq: 3.05, spd: -0.45, lw: 1.7, depth: 0.78 },
      { rgb: [255, 184, 0], a: 0.42, amp: 16, freq: 1.75, spd: 0.39, lw: 1.55, depth: 0.70 },
      { rgb: [255, 112, 67], a: 0.36, amp: 13, freq: 3.55, spd: -0.32, lw: 1.4, depth: 0.60 },
      { rgb: [45, 212, 191], a: 0.30, amp: 11, freq: 4.10, spd: 0.26, lw: 1.25, depth: 0.50 },
      { rgb: [167, 139, 250], a: 0.24, amp: 9, freq: 2.30, spd: -0.21, lw: 1.1, depth: 0.42 },
    ],
    light: [
      { rgb: [234, 72, 16], a: 0.38, amp: 28, freq: 1.35, spd: 0.78, lw: 2.0, depth: 1.0 },
      { rgb: [13, 148, 136], a: 0.35, amp: 24, freq: 1.95, spd: -0.62, lw: 1.85, depth: 0.94 },
      { rgb: [124, 58, 237], a: 0.32, amp: 20, freq: 2.55, spd: 0.52, lw: 1.7, depth: 0.86 },
      { rgb: [2, 132, 199], a: 0.30, amp: 17, freq: 3.05, spd: -0.45, lw: 1.55, depth: 0.78 },
      { rgb: [217, 119, 6], a: 0.28, amp: 14, freq: 1.75, spd: 0.39, lw: 1.4, depth: 0.70 },
      { rgb: [225, 29, 72], a: 0.23, amp: 11, freq: 3.55, spd: -0.32, lw: 1.25, depth: 0.60 },
      { rgb: [15, 118, 110], a: 0.19, amp: 9, freq: 4.10, spd: 0.26, lw: 1.15, depth: 0.50 },
      { rgb: [109, 40, 217], a: 0.16, amp: 8, freq: 2.30, spd: -0.21, lw: 1.0, depth: 0.42 },
    ],
  };

  const WF_RGB = [
    [255, 90, 31],
    [18, 181, 166],
    [139, 92, 246],
    [255, 184, 0],
    [56, 189, 248],
  ];

  const lerp = (a, b, t) => a + (b - a) * t;
  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

  function initWaveField(canvas, options = {}) {
    if (!canvas || !canvas.getContext) return null;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;

    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const isTouch = matchMedia("(hover: none)").matches;
    const intensity = options.intensity === "calm" ? "calm" : "full";
    const pointerOn = options.pointerInteraction !== false && intensity === "full";
    const onQualityChange = typeof options.onQualityChange === "function" ? options.onQualityChange : null;

    let mode = options.colorMode || document.documentElement.dataset.theme || "dark";
    let fromMode = mode;
    let themeBlend = 1;
    let themeBlendStart = 0;

    let paused = reduced;
    let inView = true;
    let qualityReduced = false;
    let liteManual = false;
    let reducedAtMs = null;
    let slowAccumMs = 0;

    // 2.5s cinematic intro (full mode only, once per session)
    let introActive = Boolean(options.intro && intensity === "full" && !reduced);
    let introMs = introActive ? 0 : 2500;
    let introBurstFired = false;

    // Typing reactivity + success convergence flash (single smooth flash, <= 1/s WCAG)
    let typingBoost = 0;
    let surgeEnergy = 0;
    let nextSurgeIn = 3.6;
    let convergeActive = false;
    let convergeP = 0;

    let w = 0;
    let h = 0;
    let phase = 1.6;
    let scanPhase = 0;
    let wfScroll = 0;
    let rafId = 0;
    let lastRafTs = 0;
    let lastDrawTs = 0;
    const startWallMs = performance.now();
    const minFrameInterval = 1000 / (Number(options.fpsCap) || (intensity === "calm" ? 30 : 60));

    const frameTimes = [];
    let worstFrameMs = 0;

    const ptr = { x: 0.5, y: 0.5, active: false, strength: 0 };
    const ripples = []; // Pointer/ghost ripples (cap 6)
    const bursts = []; // Sound bursts (cap 3)
    let nextGhostIn = 2.6;

    const maxParticles =
      intensity === "calm"
        ? 12
        : window.innerWidth < 640
          ? 56
          : window.innerWidth < 1100
            ? 96
            : 160;
    const particles = Array.from({ length: maxParticles }, (_, i) => ({
      u: ((i * 37.3) % 100) / 100,
      v: 0.1 + (((i * 29.1) % 80) / 100),
      vu: (i % 2 === 0 ? 1 : -1) * (0.00024 + (i % 5) * 0.00007),
      layer: i % 8,
      r: 1.0 + (i % 3) * 0.65,
    }));

    const isLite = () => qualityReduced || liteManual;

    function spawnRipple(x, y, maxR = 72, rgb = [139, 92, 246]) {
      if (ripples.length >= 6) ripples.shift();
      ripples.push({ x, y, age: 0, born: performance.now(), maxR, rgb });
    }

    function spawnBurst(x, y, maxR = 320, rgb = [255, 90, 31]) {
      if (bursts.length >= 3) bursts.shift();
      bursts.push({ x, y, age: 0, maxR, rgb });
    }

    function fit() {
      const rect = canvas.getBoundingClientRect();
      const d = Math.min(window.devicePixelRatio || 1, isLite() ? 1 : 2);
      w = Math.max(1, rect.width);
      h = Math.max(1, rect.height);
      canvas.width = Math.round(w * d);
      canvas.height = Math.round(h * d);
      ctx.setTransform(d, 0, 0, d, 0, 0);
      drawFrame(performance.now());
    }

    function waveHeight(u, layerCfg, ph, idx, yBase) {
      const env = Math.sin(u * Math.PI);
      const breathe = 0.74 + 0.26 * Math.sin(ph * 0.55 + idx * 1.1) + surgeEnergy * 0.42 + typingBoost * 0.38;
      const y1 = Math.sin(u * Math.PI * 2 * layerCfg.freq + ph * layerCfg.spd) * layerCfg.amp * breathe;
      const y2 = Math.cos(u * Math.PI * 3.5 - ph * layerCfg.spd * 0.66 + idx * 0.7) * (layerCfg.amp * 0.36 * breathe);
      const y3 = Math.sin(u * Math.PI * 5.1 + ph * (layerCfg.spd * 0.45)) * (layerCfg.amp * 0.14);

      let deform = 0;
      if (pointerOn && ptr.strength > 0.01) {
        const dx = u - ptr.x;
        const dy = (ptr.y * h - yBase) * 0.22;
        deform += Math.exp(-(dx * dx) / 0.028) * clamp(dy, -46, 46) * ptr.strength * layerCfg.depth;
      }
      for (let b = 0; b < bursts.length; b++) {
        const bt = bursts[b];
        const rx = (u - bt.x) * w;
        const ry = yBase - bt.y * h;
        const dist = Math.hypot(rx, ry);
        const front = bt.age * bt.maxR;
        const band = Math.abs(dist - front);
        if (band < 56) {
          const k = (1 - band / 56) * (1 - bt.age);
          deform += Math.sin((dist - front) * 0.15) * 22 * k * layerCfg.depth;
        }
      }
      const raw = (y1 + y2 + y3) * (0.28 + 0.72 * env) + deform;
      return convergeActive ? raw * (1 - convergeP) : raw;
    }

    /* ---- FAR LAYER: Scrolling Spectrogram Waterfall (25-35% opacity cells) ---- */
    function drawWaterfall(isLight) {
      if (intensity !== "full") return;
      const lite = isLite();
      const cols = lite ? 18 : Math.min(46, Math.max(22, Math.floor(w / 32)));
      const rows = lite ? 9 : 16;
      const cellW = w / cols;
      const cellH = h / rows;
      const scrollIdx = Math.floor(wfScroll);
      const fracX = (wfScroll - scrollIdx) * cellW;
      const alphaMax = isLight ? 0.26 : 0.33;

      for (let c = -1; c <= cols; c++) {
        const gc = c + scrollIdx;
        const x = c * cellW - fracX;
        for (let r = 0; r < rows; r++) {
          // Organic banded harmonic energy pattern
          const f1 = Math.sin(gc * 0.31 + r * 0.58) * Math.cos(gc * 0.14 - r * 0.34);
          const f2 = Math.sin(gc * 0.19 - r * 0.85 + phase * 0.35);
          const energy = 0.5 + 0.32 * f1 + 0.22 * f2 + surgeEnergy * 0.14 + typingBoost * 0.1;
          if (energy < 0.62) continue;
          const norm = clamp((energy - 0.62) / 0.42, 0, 1);
          const rgb = WF_RGB[Math.abs(gc + r * 2) % WF_RGB.length];
          const a = (0.08 + norm * (alphaMax - 0.08)) * (lite ? 0.7 : 1);
          ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a.toFixed(3)})`;
          ctx.fillRect(x + 2, r * cellH + 2, Math.max(3, cellW - 5), Math.max(3, cellH - 5));
        }
      }
    }

    function drawFrame(nowTs) {
      if (!w || !h) return;
      ctx.clearRect(0, 0, w, h);

      if (themeBlend < 1) {
        themeBlend = Math.min(1, (nowTs - themeBlendStart) / 400);
      }
      const palA = PALETTE[fromMode] || PALETTE.dark;
      const palB = PALETTE[mode] || PALETTE.dark;
      const isLight = mode === "light";

      // 1. FAR LAYER: Scrolling spectrogram waterfall
      drawWaterfall(isLight);

      // Intro radial sweep factor: waves sweep outward from logo burst origin
      const originX = w < 900 ? 0.5 : 0.22;
      const sweep = introActive ? clamp((introMs - 180) / 1100, 0, 1) : 1;
      const easeSweep = 1 - Math.pow(1 - sweep, 3);

      // 2. MID LAYER: 8 glowing wave ribbons with additive blending (optical interference)
      const totalLayers = intensity === "calm" ? 3 : isLite() ? 4 : 8;
      const xStep = isLite() ? 10 : 5;
      const cy = h * 0.50;

      ctx.save();
      if (!isLight && intensity === "full") {
        ctx.globalCompositeOperation = "lighter";
      }
      for (let i = 0; i < totalLayers; i++) {
        const a = palA[i];
        const b = palB[i];
        const r = Math.round(lerp(a.rgb[0], b.rgb[0], themeBlend));
        const g = Math.round(lerp(a.rgb[1], b.rgb[1], themeBlend));
        const bl = Math.round(lerp(a.rgb[2], b.rgb[2], themeBlend));
        const alpha = lerp(a.a, b.a, themeBlend) * (intensity === "calm" ? 0.62 : 1);
        const spreadOff = (i - (totalLayers - 1) * 0.5) * (intensity === "calm" ? 14 : 26);
        const yOff = cy + spreadOff * (convergeActive ? 1 - convergeP : 1);

        const xLo = easeSweep < 0.999 ? Math.max(0, w * originX - w * 1.05 * easeSweep) : 0;
        const xHi = easeSweep < 0.999 ? Math.min(w, w * originX + w * 1.05 * easeSweep) : w;
        if (xHi - xLo < 4) continue;

        ctx.beginPath();
        let started = false;
        for (let x = xLo; x <= xHi; x += xStep) {
          const u = x / w;
          const y = yOff + waveHeight(u, b, phase + i * 0.72, i, yOff);
          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
          }
        }
        // Soft ribbon glow pass (compositor-friendly, no shadowBlur)
        if (intensity === "full" && !isLite()) {
          ctx.strokeStyle = `rgba(${r},${g},${bl},${(alpha * 0.30).toFixed(3)})`;
          ctx.lineWidth = b.lw * 4.2;
          ctx.stroke();
        }
        ctx.strokeStyle = `rgba(${r},${g},${bl},${alpha.toFixed(3)})`;
        ctx.lineWidth = b.lw;
        ctx.stroke();
      }

      // Single convergence flash line on auth success
      if (convergeActive && convergeP > 0.35) {
        const flashA = Math.sin(((convergeP - 0.35) / 0.65) * Math.PI) * 0.85;
        ctx.beginPath();
        ctx.moveTo(0, cy);
        ctx.lineTo(w, cy);
        ctx.strokeStyle = `rgba(255,255,255,${clamp(flashA, 0, 0.9).toFixed(3)})`;
        ctx.lineWidth = 3 + flashA * 6;
        ctx.stroke();
      }
      ctx.restore();

      // 3. NEAR LAYER: Wave-riding particles + analyzer scan line + bursts + pointer/ghost ripples
      const pCount = isLite() ? Math.min(28, particles.length) : particles.length;
      for (let i = 0; i < pCount; i++) {
        const p = particles[i];
        if (!paused) p.u = (p.u + p.vu * (1 + typingBoost * 1.6) + 1) % 1;
        const lIdx = p.layer % totalLayers;
        const cfg = palB[lIdx];
        const yOff = cy + (lIdx - (totalLayers - 1) * 0.5) * (intensity === "calm" ? 14 : 26) * (convergeActive ? 1 - convergeP : 1);
        const px = p.u * w;
        const py = yOff + waveHeight(p.u, cfg, phase + lIdx * 0.72, lIdx, yOff) + (p.v - 0.5) * h * 0.12 * (convergeActive ? 1 - convergeP : 1);
        ctx.beginPath();
        ctx.arc(px, py, p.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${cfg.rgb[0]},${cfg.rgb[1]},${cfg.rgb[2]},${isLight ? "0.42" : "0.62"})`;
        ctx.fill();
      }

      // Analyzer scan line with fading trail
      if (intensity === "full") {
        const sx = (scanPhase % 1) * w;
        const grad = ctx.createLinearGradient(sx - 56, 0, sx + 2, 0);
        grad.addColorStop(0, "transparent");
        grad.addColorStop(1, isLight ? "rgba(18,181,166,0.22)" : "rgba(56,189,248,0.28)");
        ctx.fillStyle = grad;
        ctx.fillRect(sx - 56, 0, 58, h);
      }

      // Radial sonic burst rings
      for (let i = bursts.length - 1; i >= 0; i--) {
        const bt = bursts[i];
        if (!paused) bt.age += 0.015;
        if (bt.age >= 1) {
          bursts.splice(i, 1);
          continue;
        }
        const rad = 14 + bt.age * bt.maxR;
        const fade = (1 - bt.age) * (isLight ? 0.34 : 0.52);
        ctx.beginPath();
        ctx.arc(bt.x * w, bt.y * h, rad, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(${bt.rgb[0]},${bt.rgb[1]},${bt.rgb[2]},${fade.toFixed(3)})`;
        ctx.lineWidth = 2.0 * (1 - bt.age * 0.5);
        ctx.stroke();
      }

      // Pointer & autonomous ghost ripples (cap 6)
      for (let i = ripples.length - 1; i >= 0; i--) {
        const rp = ripples[i];
        if (!paused) rp.age += 0.022;
        if (rp.age >= 1) {
          ripples.splice(i, 1);
          continue;
        }
        ctx.beginPath();
        ctx.arc(rp.x * w, rp.y * h, 10 + rp.age * rp.maxR, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(${rp.rgb[0]},${rp.rgb[1]},${rp.rgb[2]},${((1 - rp.age) * (isLight ? 0.30 : 0.45)).toFixed(3)})`;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }

    function tick(ts) {
      if (paused || document.hidden || !inView) {
        rafId = 0;
        return;
      }
      if (!lastRafTs) lastRafTs = ts;
      const dt = ts - lastRafTs;
      lastRafTs = ts;

      if (dt > 0 && dt < 250) {
        frameTimes.push(dt);
        if (frameTimes.length > 240) frameTimes.shift();
        if (dt > worstFrameMs) worstFrameMs = dt;

        // Adaptive quality: if avg frame time > 24 ms for 2 seconds, auto-reduce & offer Lite mode
        if (dt > 24) {
          slowAccumMs += dt;
          if (!qualityReduced && slowAccumMs >= 2000) {
            qualityReduced = true;
            reducedAtMs = Math.round(ts - startWallMs);
            fit();
            onQualityChange?.(true);
          }
        } else {
          slowAccumMs = Math.max(0, slowAccumMs - dt * 0.5);
        }
      }

      if (ts - lastDrawTs >= minFrameInterval) {
        const stepSec = Math.min(0.08, (ts - (lastDrawTs || ts)) / 1000) || 0.016;
        lastDrawTs = ts;
        phase += stepSec * (0.92 + typingBoost * 0.35);
        scanPhase = (scanPhase + stepSec / 6.5) % 1;
        wfScroll += stepSec * (isLite() ? 1.4 : 2.6);
        ptr.strength *= ptr.active ? 0.96 : 0.88;
        typingBoost *= 0.92;
        surgeEnergy *= 0.95;

        if (convergeActive) {
          convergeP = Math.min(1, convergeP + stepSec * 2.7);
        }

        if (introActive) {
          introMs += stepSec * 1000;
          if (!introBurstFired && introMs >= 120) {
            introBurstFired = true;
            const bx = w < 900 ? 0.5 : 0.14;
            const by = w < 900 ? 0.12 : 0.16;
            spawnBurst(bx, by, Math.max(w, h) * 0.65, [255, 90, 31]);
            spawnBurst(bx, by, Math.max(w, h) * 0.48, [18, 181, 166]);
          }
          if (introMs >= 2500) {
            introActive = false;
          }
        } else if (intensity === "full") {
          nextSurgeIn -= stepSec;
          if (nextSurgeIn <= 0) {
            nextSurgeIn = 3.2 + Math.random() * 2.6;
            surgeEnergy = 1.0;
            const rgb = WF_RGB[Math.floor(Math.random() * WF_RGB.length)];
            spawnBurst(0.18 + Math.random() * 0.64, 0.22 + Math.random() * 0.56, 260 + Math.random() * 110, rgb);
          }
          if (isTouch) {
            nextGhostIn -= stepSec;
            if (nextGhostIn <= 0) {
              nextGhostIn = 2.6 + Math.random() * 1.8;
              spawnRipple(0.18 + Math.random() * 0.64, 0.22 + Math.random() * 0.56, 68, [18, 181, 166]);
            }
          }
        }

        drawFrame(ts);
        if (frameTimes.length % 15 === 0) {
          const avg = frameTimes.reduce((s, v) => s + v, 0) / frameTimes.length;
          canvas.dataset.metrics = JSON.stringify({
            avgFrameMs: Number(avg.toFixed(2)),
            worstFrameMs: Number(worstFrameMs.toFixed(2)),
            qualityReduced,
            liteMode: isLite(),
            reducedAtMs,
            samples: frameTimes.length,
          });
        }
      }
      rafId = requestAnimationFrame(tick);
    }

    function start() {
      cancelAnimationFrame(rafId);
      lastRafTs = 0;
      lastDrawTs = 0;
      if (!paused && !document.hidden && inView) {
        rafId = requestAnimationFrame(tick);
      } else {
        drawFrame(performance.now());
      }
    }

    if (pointerOn) {
      window.addEventListener(
        "pointermove",
        (e) => {
          const rect = canvas.getBoundingClientRect();
          if (!rect.width || !rect.height) return;
          ptr.x = clamp((e.clientX - rect.left) / rect.width, 0, 1);
          ptr.y = clamp((e.clientY - rect.top) / rect.height, 0, 1);
          ptr.active = true;
          ptr.strength = 1;
          if (!paused && (ripples.length === 0 || performance.now() - ripples[ripples.length - 1].born > 280)) {
            spawnRipple(ptr.x, ptr.y, 68, [139, 92, 246]);
          }
        },
        { passive: true },
      );
      window.addEventListener(
        "pointerdown",
        (e) => {
          if (paused) return;
          const rect = canvas.getBoundingClientRect();
          if (!rect.width || !rect.height) return;
          const rx = clamp((e.clientX - rect.left) / rect.width, 0, 1);
          const ry = clamp((e.clientY - rect.top) / rect.height, 0, 1);
          spawnRipple(rx, ry, 86, [255, 90, 31]);
        },
        { passive: true },
      );
    }

    canvas.addEventListener("contentvisibilityautostatechange", (event) => {
      inView = !event.skipped;
      if (inView && !paused) start();
      else {
        cancelAnimationFrame(rafId);
        rafId = 0;
      }
    });

    const io = new IntersectionObserver(
      (entries) => {
        inView = Boolean(entries[0]?.isIntersecting);
        if (inView && !paused) start();
        else {
          cancelAnimationFrame(rafId);
          rafId = 0;
        }
      },
      { threshold: 0.02 },
    );
    io.observe(canvas);

    document.addEventListener("visibilitychange", () => {
      if (document.hidden) {
        cancelAnimationFrame(rafId);
        rafId = 0;
      } else if (!paused && inView) {
        start();
      }
    });

    window.addEventListener("resize", fit);
    window.addEventListener("ec:theme", (e) => {
      const next = e.detail || document.documentElement.dataset.theme || "dark";
      if (next !== mode) {
        fromMode = mode;
        mode = next;
        themeBlend = 0;
        themeBlendStart = performance.now();
        if (paused) drawFrame(performance.now());
      }
    });

    const boot = () => {
      canvas.classList.add("wave-ready");
      fit();
      start();
    };
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", boot, { once: true });
    } else {
      boot();
    }

    return {
      pause() {
        paused = true;
        cancelAnimationFrame(rafId);
        rafId = 0;
      },
      play() {
        if (reduced) return;
        paused = false;
        start();
      },
      toggle() {
        if (paused) this.play();
        else this.pause();
        return !paused;
      },
      setLite(val) {
        liteManual = Boolean(val);
        fit();
        return isLite();
      },
      isLite,
      skipIntro() {
        introActive = false;
        introMs = 2500;
        drawFrame(performance.now());
      },
      boostTyping(amount = 0.85) {
        if (!paused) typingBoost = Math.min(1.4, typingBoost + amount);
      },
      convergeSuccess() {
        if (reduced) return;
        convergeActive = true;
        convergeP = 0;
        spawnBurst(0.5, 0.5, Math.max(w, h) * 0.5, [18, 181, 166]);
      },
      pulse(x = 0.5, y = 0.5) {
        if (!paused) spawnBurst(x, y, 240, [18, 181, 166]);
      },
      isPaused: () => paused,
      setColorMode(next) {
        fromMode = mode;
        mode = next;
        themeBlend = 0;
        themeBlendStart = performance.now();
        drawFrame(performance.now());
      },
      getMetrics() {
        const avg = frameTimes.length
          ? frameTimes.reduce((s, v) => s + v, 0) / frameTimes.length
          : 0;
        return {
          avgFrameMs: Number(avg.toFixed(2)),
          worstFrameMs: Number(worstFrameMs.toFixed(2)),
          qualityReduced,
          liteMode: isLite(),
          reducedAtMs,
          samples: frameTimes.length,
          paused,
        };
      },
    };
  }

  window.initWaveField = initWaveField;

  /* Auto-init calm background variants on Home hero & Analyze header */
  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("canvas[data-wave-calm]").forEach((cv) => {
      const ctrl = initWaveField(cv, {
        intensity: "calm",
        pointerInteraction: false,
        fpsCap: 30,
      });
      if (ctrl && cv.id) window[`__ecWave_${cv.id}`] = ctrl;
    });
  });
})();
