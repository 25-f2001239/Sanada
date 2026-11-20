/* EchoCount — Live 2D Neural Architecture View */
(() => {
  const { $, esc, fmtTime } = window.EC;
  const viewAnalysis = $("#analysisView"),
    viewArch = $("#archView"),
    tabAnalysis = $("#viewTabAnalysis"),
    tabArch = $("#viewTabArch"),
    canvas = $("#archCanvas"),
    inspector = $("#archInspector"),
    fileLbl = $("#archFileLbl"),
    pickBtn = $("#archPickBtn"),
    runBtn = $("#archRunBtn"),
    playBtn = $("#archPlayBtn"),
    muteBtn = $("#archMuteBtn");

  if (!canvas || !viewArch) return;

  const W = 1200,
    H = 620;
  const ctx = canvas.getContext("2d");
  const bellSrc = viewArch.dataset.bellSrc || "/static/arch-bell.mp3";
  const doneSrc = viewArch.dataset.doneSrc || "/static/arch-done.mp3";

  // Pipeline state: "off" (unlit Ferris wheel) | "play" (4-step choreographed flow, 1s per step at 1x)
  const STEP_SEC = 1.0,
    TOTAL_SEC = 4.0;
  let activeView = "analysis";
  let flowState = "off";
  let serverPending = false; // true while /api/analyze request is in flight
  let speed = 1;
  let elapsed = 0; // 0..4.0 seconds (1s per step at 1x)
  let animTime = 0;
  let selectedId = "counter";
  let fileInfo = null;
  let realSpec = null;
  let realWave = null;
  let lastResult = null;
  let rafId = 0,
    lastTs = 0;
  let soundOn = true;
  let rungSteps = new Set();

  function fmtCount(c) {
    if (!c) return "≈ K";
    const k = Number(c.rounded_count);
    if (Number.isFinite(k) && k <= 4) return `≈${k}`;
    return c.display || `≈${k}`;
  }

  // Sound engine: arch-bell.mp3 for steps 1-3, arch-done.mp3 for step 4 (prediction shown).
  // ONLY plays when activeView === "arch" and soundOn === true.
  let audioCtx = null;
  const soundBufs = { step: null, done: null };
  const soundLoading = { step: false, done: false };
  const soundUrls = { step: bellSrc, done: doneSrc };
  const activeSources = new Set();

  function loadSound(kind) {
    if (soundBufs[kind] || soundLoading[kind]) return;
    soundLoading[kind] = true;
    fetch(soundUrls[kind])
      .then((r) => r.arrayBuffer())
      .then((ab) => {
        const AC = window.AudioContext || window.webkitAudioContext;
        if (!AC) return;
        if (!audioCtx) audioCtx = new AC();
        return audioCtx.decodeAudioData(ab);
      })
      .then((buf) => {
        if (buf) soundBufs[kind] = buf;
      })
      .catch(() => {})
      .finally(() => {
        soundLoading[kind] = false;
      });
  }

  function ensureBellLoaded() {
    loadSound("step");
    loadSound("done");
  }

  function stopAllBells() {
    activeSources.forEach((s) => {
      try {
        if (typeof s.stop === "function") s.stop();
        else if (typeof s.pause === "function") {
          s.pause();
          s.currentTime = 0;
        }
      } catch {}
    });
    activeSources.clear();
  }

  function playSound(kind = "step") {
    if (activeView !== "arch" || !soundOn) return;
    const buf = soundBufs[kind];
    if (buf && audioCtx) {
      if (audioCtx.state === "suspended") audioCtx.resume().catch(() => {});
      try {
        const src = audioCtx.createBufferSource();
        const gain = audioCtx.createGain();
        src.buffer = buf;
        gain.gain.value = 0.48;
        src.connect(gain);
        gain.connect(audioCtx.destination);
        activeSources.add(src);
        src.onended = () => activeSources.delete(src);
        src.start(0);
        return;
      } catch {}
    }
    try {
      const a = new Audio(soundUrls[kind]);
      a.volume = 0.48;
      activeSources.add(a);
      a.addEventListener("ended", () => activeSources.delete(a));
      a.play().catch(() => {});
    } catch {}
  }

  function ringStep(stepKey) {
    if (rungSteps.has(stepKey)) return;
    rungSteps.add(stepKey);
    playSound(stepKey === "step4" ? "done" : "step");
  }

  // Stage boxes in logical 1200x620 coordinates
  const NODES = {
    decode: { id: "decode", x: 22, y: 206, w: 190, h: 208, title: "1. Audio Decode", sub: "soundfile · librosa" },
    mel: { id: "mel", x: 248, y: 206, w: 206, h: 208, title: "2. Log-Mel Spectrogram", sub: "librosa.melspectrogram" },
    counter: { id: "counter", x: 520, y: 22, w: 422, h: 176, title: "3A. Source Counter", sub: "SourceCountCNN + 30× MC-Dropout MLP" },
    tagger: { id: "tagger", x: 520, y: 222, w: 422, h: 176, title: "3B. Open-Vocab Tagger", sub: "GroupFineTaggerCNN + AST + CLAP" },
    speaker: { id: "speaker", x: 520, y: 422, w: 422, h: 176, title: "3C. Speakers & Separation", sub: "ECAPA-TDNN + SepFormer / MaskNet" },
    fusion: { id: "fusion", x: 984, y: 186, w: 194, h: 248, title: "4. Output Fusion", sub: "Count · Tags · GT Hash" },
  };

  const COUNTER_MLP = buildLayers([5, 7, 5], 736, 66, 184, 108);
  const TAGGER_MLP = buildLayers([4, 5, 4], 542, 268, 120, 96);
  const ECAPA_NET = buildLayers([4, 5, 3], 542, 468, 114, 96);

  function buildLayers(counts, x0, y0, w, h) {
    const layers = counts.map((n, li) => {
      const lx = x0 + (li / Math.max(1, counts.length - 1)) * w;
      const pts = [];
      for (let i = 0; i < n; i++) {
        const ly = y0 + ((i + 0.5) / n) * h;
        pts.push({ x: lx, y: ly, seed: (li * 13 + i * 7) * 0.91 });
      }
      return pts;
    });
    const edges = [];
    for (let l = 0; l < layers.length - 1; l++) {
      for (const a of layers[l]) {
        for (const b of layers[l + 1]) {
          edges.push({ a, b, w: Math.sin(a.seed * 3.1 + b.seed * 2.3) * 0.5 + 0.5 });
        }
      }
    }
    return { layers, edges };
  }

  function computeSpectrogram(samples, sr) {
    if (!samples || !samples.length) return null;
    const cols = 40,
      bins = 16,
      win = 256;
    const grid = new Float32Array(cols * bins);
    const maxStart = Math.max(1, samples.length - win);
    const fMin = 80,
      fMax = Math.min(7500, (sr || 16000) / 2);
    const freqs = [];
    for (let b = 0; b < bins; b++) freqs.push(fMin * Math.pow(fMax / fMin, b / (bins - 1)));
    let peak = 1e-6;
    for (let c = 0; c < cols; c++) {
      const start = Math.floor((c / cols) * maxStart);
      for (let b = 0; b < bins; b++) {
        const omega = (2 * Math.PI * freqs[b]) / (sr || 16000);
        let re = 0,
          im = 0;
        for (let k = 0; k < win; k += 2) {
          const s = samples[start + k] || 0;
          const hann = 0.5 * (1 - Math.cos((2 * Math.PI * k) / win));
          const ang = omega * k;
          re += s * hann * Math.cos(ang);
          im -= s * hann * Math.sin(ang);
        }
        const mag = Math.log1p(Math.sqrt(re * re + im * im) * 1.4);
        grid[c * bins + b] = mag;
        if (mag > peak) peak = mag;
      }
    }
    for (let i = 0; i < grid.length; i++) grid[i] = Math.min(1, grid[i] / peak);
    return { cols, bins, grid };
  }

  function palette() {
    const dark = document.documentElement.dataset.theme === "dark";
    return dark
      ? {
          dark: true,
          bg: "#121216",
          card: "#19191f",
          cardHi: "#23232c",
          border: "rgba(255,255,255,0.10)",
          borderSel: "rgba(255,255,255,0.42)",
          ink: "#F2F0EB",
          mute: "rgba(242,240,235,0.55)",
          dim: "rgba(242,240,235,0.20)",
          wire: "rgba(255,255,255,0.12)",
          orange: "#FF5A1F",
          amber: "#fbbf24",
          lime: "#C8F04A",
          teal: "#12b5a6",
          violet: "#a78bfa",
        }
      : {
          dark: false,
          bg: "#F6F5F0",
          card: "#FFFFFF",
          cardHi: "#FFF9F4",
          border: "rgba(12,12,14,0.12)",
          borderSel: "rgba(12,12,14,0.45)",
          ink: "#0C0C0E",
          mute: "rgba(12,12,14,0.55)",
          dim: "rgba(12,12,14,0.18)",
          wire: "rgba(12,12,14,0.13)",
          orange: "#FF5A1F",
          amber: "#eab308",
          lime: "#78a800",
          teal: "#0d9488",
          violet: "#7c3aed",
        };
  }

  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, r);
  }

  function heatColor(v, lit, dark) {
    const t = Math.max(0, Math.min(1, v));
    if (!lit) {
      // Powered-off monochrome dim spectrogram
      const g = dark ? Math.round(26 + t * 34) : Math.round(230 - t * 48);
      return `rgb(${g}, ${g}, ${g + (dark ? 4 : -2)})`;
    }
    if (t < 0.35) {
      const k = t / 0.35;
      return `rgb(${Math.round(26 + k * 145)}, ${Math.round(24 + k * 56)}, ${Math.round(32 + k * 8)})`;
    }
    if (t < 0.72) {
      const k = (t - 0.35) / 0.37;
      return `rgb(${Math.round(171 + k * 80)}, ${Math.round(80 + k * 111)}, ${Math.round(40 - k * 4)})`;
    }
    const k = (t - 0.72) / 0.28;
    return `rgb(${Math.round(251)}, ${Math.round(191 + k * 48)}, ${Math.round(36 + k * 90)})`;
  }

  function bezierPt(p0, p1, p2, p3, t) {
    const u = 1 - t;
    return {
      x: u * u * u * p0.x + 3 * u * u * t * p1.x + 3 * u * t * t * p2.x + t * t * t * p3.x,
      y: u * u * u * p0.y + 3 * u * u * t * p1.y + 3 * u * t * t * p2.y + t * t * t * p3.y,
    };
  }

  function drawBus(p0, p3, active, color, pulseOffset, pal) {
    const dx = (p3.x - p0.x) * 0.5;
    const p1 = { x: p0.x + dx, y: p0.y };
    const p2 = { x: p3.x - dx, y: p3.y };

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(p0.x, p0.y);
    ctx.bezierCurveTo(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
    ctx.strokeStyle = active ? color : pal.wire;
    ctx.globalAlpha = active ? 0.7 : 0.75;
    ctx.lineWidth = active ? 2.6 : 1.5;
    ctx.stroke();

    if (active) {
      ctx.globalAlpha = 1;
      for (let i = 0; i < 3; i++) {
        const t = (pulseOffset + i / 3) % 1;
        const pt = bezierPt(p0, p1, p2, p3, t);
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 4.2, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.shadowColor = color;
        ctx.shadowBlur = 10;
        ctx.fill();
      }
    }
    ctx.restore();
  }

  function stageEnergy(stageId) {
    // Turned-off Ferris wheel: zero lights when flowState === "off"
    if (flowState === "off") return 0;

    // 1.0s per step at 1x: [0..1] decode, [1..2] mel, [2..3] 3 models, [3..4] fusion
    const ranges = {
      decode: [0.0, 1.12],
      mel: [1.0, 2.12],
      counter: [2.0, 3.12],
      tagger: [2.0, 3.12],
      speaker: [2.0, 3.12],
      fusion: [3.0, 4.0],
    };
    const [a, b] = ranges[stageId];
    if (elapsed < a || elapsed > b) return 0;
    const len = b - a;
    const t = (elapsed - a) / len;
    // Quick ramp-in, steady bright plateau, gentle ramp-out
    if (t < 0.12) return 0.35 + (t / 0.12) * 0.65;
    if (t > 0.85) return Math.max(0.25, 1 - ((t - 0.85) / 0.15) * 0.75);
    return 1.0;
  }

  function drawNodeFrame(node, energy, pal, accent) {
    const isSel = selectedId === node.id;
    const lit = energy > 0.05;
    ctx.save();
    roundRect(node.x, node.y, node.w, node.h, 14);
    ctx.fillStyle = lit ? pal.cardHi : pal.card;
    ctx.fill();
    ctx.lineWidth = lit ? 2.1 : isSel ? 1.7 : 1.1;
    ctx.strokeStyle = lit ? accent : isSel ? pal.borderSel : pal.border;
    if (lit) {
      ctx.shadowColor = accent;
      ctx.shadowBlur = 12;
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    ctx.fillStyle = lit ? pal.ink : pal.mute;
    ctx.font = "700 13px Geist, system-ui, sans-serif";
    ctx.fillText(node.title, node.x + 12, node.y + 22);
    ctx.fillStyle = pal.mute;
    ctx.font = "500 10.5px 'Geist Mono', monospace";
    ctx.fillText(node.sub, node.x + 12, node.y + 37);

    // LED indicator (completely dark when off)
    ctx.beginPath();
    ctx.arc(node.x + node.w - 14, node.y + 18, 4.2, 0, Math.PI * 2);
    ctx.fillStyle = lit ? accent : pal.wire;
    ctx.fill();
    ctx.restore();
  }

  function drawDecodeContent(node, energy, pal) {
    const lit = energy > 0.05;
    const x = node.x + 12,
      y = node.y + 52,
      w = node.w - 24,
      h = 84;
    ctx.save();
    roundRect(x, y, w, h, 8);
    ctx.fillStyle = pal.dark ? "rgba(255,255,255,0.03)" : "rgba(12,12,14,0.04)";
    ctx.fill();

    const midY = y + h / 2;
    const bars = 46;
    const scan = (animTime * 0.9) % 1;
    for (let i = 0; i < bars; i++) {
      const u = i / bars;
      let amp;
      if (realWave && realWave.length) {
        const idx = Math.floor(u * (realWave.length - 1));
        amp = Math.max(0.08, realWave[idx]);
      } else {
        const tShift = lit ? animTime * 0.5 : 0;
        amp =
          0.22 +
          0.42 * Math.abs(Math.sin((u + tShift) * Math.PI * 5)) *
            (0.6 + 0.4 * Math.cos(u * 13 - tShift * 4));
      }
      const bh = amp * (h * 0.76);
      const bx = x + 6 + u * (w - 12);
      const near = lit && Math.abs(u - scan) < 0.14;
      ctx.fillStyle = !lit ? pal.dim : near ? pal.lime : pal.orange;
      ctx.fillRect(bx, midY - bh / 2, 2.4, Math.max(2, bh));
    }

    ctx.fillStyle = pal.mute;
    ctx.font = "500 10px 'Geist Mono', monospace";
    const srTxt = fileInfo ? `${Math.round((fileInfo.sampleRate || 16000) / 1000)}kHz → 16kHz mono` : "WAV/MP3 → 16kHz mono";
    const durTxt = fileInfo ? `Window: 8.0s (of ${fmtTime(fileInfo.duration)})` : "Window: 8.0s (128,000 pts)";
    ctx.fillText(srTxt, x, y + h + 18);
    ctx.fillText(durTxt, x, y + h + 34);
    const ms = lastResult?.timings?.decode_ms;
    ctx.fillStyle = lit ? pal.orange : pal.mute;
    ctx.fillText(ms != null ? `Measured: ${ms} ms` : "decode_mono() + resample()", x, y + h + 50);
    ctx.restore();
  }

  function drawMelContent(node, energy, pal) {
    const lit = energy > 0.05;
    const x = node.x + 12,
      y = node.y + 52,
      w = node.w - 24,
      h = 92;
    const cols = 40,
      bins = 16;
    const cw = w / cols,
      ch = h / bins;
    const scanCol = Math.floor(((animTime * 0.85) % 1) * cols);

    ctx.save();
    roundRect(x, y, w, h, 8);
    ctx.clip();
    for (let c = 0; c < cols; c++) {
      for (let b = 0; b < bins; b++) {
        let v;
        if (realSpec) {
          v = realSpec.grid[c * bins + (bins - 1 - b)];
        } else {
          const t = c / cols,
            f = 1 - b / bins;
          const sh = lit ? animTime * 1.6 : 0;
          v =
            0.12 +
            0.55 * Math.exp(-Math.pow((f - 0.28 - 0.12 * Math.sin(t * 7 + sh)) * 4, 2)) +
            0.42 * Math.exp(-Math.pow((f - 0.65 - 0.08 * Math.cos(t * 11 - sh)) * 5, 2));
        }
        const boost = lit && Math.abs(c - scanCol) <= 2 ? 0.24 : 0;
        ctx.fillStyle = heatColor(v * (lit ? 0.72 + 0.28 * energy : 0.7) + boost, lit, pal.dark);
        ctx.fillRect(x + c * cw, y + b * ch, Math.ceil(cw), Math.ceil(ch));
      }
    }
    if (lit) {
      ctx.fillStyle = "rgba(255,255,255,0.75)";
      ctx.fillRect(x + scanCol * cw, y, 2, h);
    }
    ctx.restore();

    ctx.save();
    ctx.fillStyle = pal.mute;
    ctx.font = "500 10px 'Geist Mono', monospace";
    ctx.fillText("n_fft=1024 · hop=256 · 64 mels", x, y + h + 16);
    ctx.fillStyle = lit ? pal.amber : pal.mute;
    ctx.fillText("Tensor: [1, 64, 501]  [-1, 0]", x, y + h + 32);
    const ms = lastResult?.timings?.feature_ms;
    ctx.fillStyle = pal.mute;
    ctx.fillText(ms != null ? `Measured: ${ms} ms` : "Shared across all 3 branches →", x, y + h + 47);
    ctx.restore();
  }

  function drawNeuralNet(net, energy, accent, pal, dropoutBlink) {
    const lit = energy > 0.05;
    const wave = (animTime * 1.4) % 1;
    ctx.save();
    for (const e of net.edges) {
      const firing = lit && Math.sin(animTime * 6 + e.a.seed + e.b.seed) > 0.15;
      ctx.beginPath();
      ctx.moveTo(e.a.x, e.a.y);
      ctx.lineTo(e.b.x, e.b.y);
      ctx.strokeStyle = firing ? accent : pal.wire;
      ctx.globalAlpha = firing ? 0.3 + 0.55 * e.w * energy : 0.3;
      ctx.lineWidth = firing ? 1.5 : 0.8;
      ctx.stroke();
    }
    net.layers.forEach((layer, li) => {
      const layerAct = Math.max(0.2, 1 - Math.abs(wave - li / Math.max(1, net.layers.length - 1)) * 1.6);
      layer.forEach((n, ni) => {
        const dropped = lit && dropoutBlink && li === 1 && Math.sin(animTime * 9 + ni * 4.7) < -0.35;
        const glow = lit ? energy * layerAct * (0.6 + 0.4 * Math.sin(animTime * 5 + n.seed)) : 0;
        ctx.beginPath();
        ctx.arc(n.x, n.y, dropped ? 3.2 : 5.0, 0, Math.PI * 2);
        if (!lit || dropped) {
          ctx.fillStyle = pal.card;
          ctx.globalAlpha = dropped ? 0.3 : 0.85;
          ctx.fill();
          ctx.lineWidth = 1.2;
          ctx.strokeStyle = pal.wire;
          ctx.stroke();
        } else {
          ctx.globalAlpha = 0.96;
          ctx.fillStyle = glow > 0.42 ? accent : pal.card;
          ctx.fill();
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = accent;
          ctx.stroke();
        }
      });
    });
    ctx.restore();
  }

  function drawCounterContent(node, energy, pal) {
    const lit = energy > 0.05;
    const convs = [
      { lbl: "16", h: 66 },
      { lbl: "32", h: 56 },
      { lbl: "64", h: 46 },
      { lbl: "64", h: 38 },
    ];
    let cx = node.x + 16;
    const baseY = node.y + 118;
    const activeConv = Math.floor(((animTime * 1.8) % 1) * 4);
    ctx.save();
    convs.forEach((c, i) => {
      const act = lit && i === activeConv;
      const bw = 28,
        by = baseY - c.h / 2;
      for (let d = 2; d >= 0; d--) {
        roundRect(cx + d * 3, by - d * 3, bw, c.h, 4);
        ctx.fillStyle = act ? pal.orange : pal.card;
        ctx.globalAlpha = act ? 0.25 + (2 - d) * 0.25 : 0.85;
        ctx.fill();
        ctx.strokeStyle = act ? pal.orange : pal.border;
        ctx.lineWidth = 1.2;
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      ctx.fillStyle = pal.mute;
      ctx.font = "600 9.5px 'Geist Mono', monospace";
      ctx.fillText(`C${c.lbl}`, cx + 4, baseY + c.h / 2 + 13);
      cx += 44;
    });

    ctx.strokeStyle = lit ? pal.orange : pal.wire;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(node.x + 194, node.y + 118);
    ctx.lineTo(node.x + 212, node.y + 118);
    ctx.stroke();

    drawNeuralNet(COUNTER_MLP, energy, pal.orange, pal, true);

    ctx.fillStyle = pal.mute;
    ctx.font = "500 9.5px 'Geist Mono', monospace";
    const passNum = lit ? Math.floor(((animTime * 8) % 30) + 1) : 30;
    ctx.fillText("Conv2D Trunk (Pool 4×16)", node.x + 16, node.y + 62);
    ctx.fillText(`MLP 4096→256→30 · ${lit ? `MC pass ${passNum}/30` : "30× MC-Dropout"}`, node.x + 216, node.y + 56);

    const kTxt = lastResult?.count
      ? `Output: ${fmtCount(lastResult.count)} (${lastResult.count.confidence_percent}% · ${lastResult.timings?.count_ms ?? "—"} ms)`
      : "Output: ≈K ordinal regression";
    ctx.fillStyle = lit ? pal.orange : pal.mute;
    ctx.font = "600 10.5px 'Geist Mono', monospace";
    ctx.fillText(kTxt, node.x + 16, node.y + node.h - 10);
    ctx.restore();
  }

  function drawTaggerContent(node, energy, pal) {
    const lit = energy > 0.05;
    ctx.save();
    ctx.fillStyle = pal.mute;
    ctx.font = "600 9.5px 'Geist Mono', monospace";
    ctx.fillText("FSD CNN+MLP (9+32)", node.x + 14, node.y + 58);
    drawNeuralNet(TAGGER_MLP, energy, pal.teal, pal, false);

    // AST Patch Attention 4x4 grid
    const ax = node.x + 168,
      ay = node.y + 68,
      cell = 20;
    ctx.fillStyle = pal.mute;
    ctx.fillText("AST ViT (527 cls)", ax, node.y + 58);
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        const w = lit ? 0.25 + 0.75 * Math.abs(Math.sin(animTime * 4 + r * 1.7 + c * 2.3)) : 0.25;
        roundRect(ax + c * (cell + 3), ay + r * (cell + 3), cell, cell, 4);
        ctx.fillStyle = lit ? pal.teal : pal.dim;
        ctx.globalAlpha = lit ? 0.14 + w * 0.72 * energy : 0.35;
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;

    // CLAP Zero-shot bars
    const lx = node.x + 286,
      ly = node.y + 68;
    ctx.fillStyle = pal.mute;
    ctx.fillText("CLAP Audio↔Text", lx, node.y + 58);
    const demoTags = lastResult?.named_sounds?.length
      ? lastResult.named_sounds.slice(0, 4).map((s) => ({
          name: s.friendly_name || s.name || s.class,
          p: Math.min(1, Number(s.confidence ?? s.prob ?? 0.7)),
        }))
      : [
          { name: "Speech", p: 0.88 },
          { name: "Music", p: 0.64 },
          { name: "Vehicle", p: 0.42 },
          { name: "Ambient", p: 0.31 },
        ];
    demoTags.forEach((t, i) => {
      const y = ly + i * 22;
      const pulse = lit ? 0.85 + 0.15 * Math.sin(animTime * 5 + i) : 1;
      ctx.fillStyle = lit ? pal.ink : pal.mute;
      ctx.font = "500 9.5px 'Geist Mono', monospace";
      ctx.fillText(String(t.name).slice(0, 12), lx, y + 10);
      roundRect(lx + 74, y + 2, 48, 9, 4);
      ctx.fillStyle = pal.wire;
      ctx.fill();
      roundRect(lx + 74, y + 2, Math.max(4, 48 * t.p * pulse), 9, 4);
      ctx.fillStyle = lit ? pal.teal : pal.dim;
      ctx.fill();
    });

    const ms = lastResult?.timings?.tag_ms;
    const tagSummary = lastResult?.named_sounds?.length
      ? `Detected: ${lastResult.named_sounds.slice(0, 3).map((s) => s.friendly_name || s.name).join(", ")} (${ms ?? "—"} ms)`
      : "Fuses FSD-32 + AST-527 + CLAP zero-shot";
    ctx.fillStyle = lit ? pal.teal : pal.mute;
    ctx.font = "600 10.5px 'Geist Mono', monospace";
    ctx.fillText(tagSummary.slice(0, 56), node.x + 14, node.y + node.h - 10);
    ctx.restore();
  }

  function drawSpeakerContent(node, energy, pal) {
    const lit = energy > 0.05;
    ctx.save();
    ctx.fillStyle = pal.mute;
    ctx.font = "600 9.5px 'Geist Mono', monospace";
    ctx.fillText("ECAPA-TDNN Emb.", node.x + 14, node.y + 58);
    drawNeuralNet(ECAPA_NET, energy, pal.violet, pal, false);

    // Speaker cosine clusters
    const cx = node.x + 168,
      cy = node.y + 68,
      cw = 100,
      ch = 88;
    ctx.fillStyle = pal.mute;
    ctx.fillText("Cosine Clusters", cx, node.y + 58);
    roundRect(cx, cy, cw, ch, 8);
    ctx.fillStyle = lit ? "rgba(124,58,237,0.09)" : pal.dark ? "rgba(255,255,255,0.03)" : "rgba(12,12,14,0.04)";
    ctx.fill();
    const centers = [
      { x: cx + 28, y: cy + 28, col: pal.violet },
      { x: cx + 72, y: cy + 56, col: pal.orange },
      { x: cx + 68, y: cy + 24, col: pal.teal },
    ];
    centers.forEach((ctr, ci) => {
      for (let j = 0; j < 4; j++) {
        const ang = (lit ? animTime * 1.4 : 0) + ci * 2 + j * 1.57;
        const r = lit ? 6 + 4 * Math.sin(animTime * 2.5 + j) : 8;
        ctx.beginPath();
        ctx.arc(ctr.x + Math.cos(ang) * r, ctr.y + Math.sin(ang) * r, 3.2, 0, Math.PI * 2);
        ctx.fillStyle = lit ? ctr.col : pal.dim;
        ctx.globalAlpha = lit ? 0.5 + 0.5 * energy : 0.6;
        ctx.fill();
      }
    });
    ctx.globalAlpha = 1;

    // Separation MaskNet / Quality Gates
    const mx = node.x + 286,
      my = node.y + 68;
    ctx.fillStyle = pal.mute;
    ctx.fillText("SepFormer / MaskNet", mx, node.y + 58);
    const gates = ["STFT Mask", "Corr <0.90", "Energy OK", "Re-tag OK"];
    gates.forEach((g, i) => {
      const y = my + i * 21;
      const gLit = lit && ((animTime * 2.5 + i) % 2) > 0.5;
      roundRect(mx, y, 120, 15, 6);
      ctx.fillStyle = gLit
        ? "rgba(124,58,237,0.2)"
        : pal.dark
          ? "rgba(255,255,255,0.04)"
          : "rgba(12,12,14,0.04)";
      ctx.fill();
      ctx.fillStyle = gLit ? pal.violet : pal.mute;
      ctx.font = "600 9.5px 'Geist Mono', monospace";
      ctx.fillText(`● ${g}`, mx + 8, y + 11);
    });

    const spk = lastResult?.speaker_clustering;
    const sep = lastResult?.separation;
    const spkTxt = spk
      ? `Speakers: ${spk.distinct_speakers ?? "—"} · Sep: ${sep?.eligible ? "Ready" : "Gated"} (${lastResult.timings?.speaker_cluster_ms ?? "—"} ms)`
      : "Speaker count + on-demand stem separation";
    ctx.fillStyle = lit ? pal.violet : pal.mute;
    ctx.font = "600 10.5px 'Geist Mono', monospace";
    ctx.fillText(spkTxt.slice(0, 56), node.x + 14, node.y + node.h - 10);
    ctx.restore();
  }

  function drawFusionContent(node, energy, pal) {
    const lit = energy > 0.05;
    const res = flowState === "off" || elapsed >= STEP_SEC * 3 ? lastResult : null;
    const x = node.x + 14,
      w = node.w - 28;
    ctx.save();
    roundRect(x, node.y + 52, w, 58, 10);
    ctx.fillStyle = lit
      ? "rgba(255,90,31,0.14)"
      : pal.dark
        ? "rgba(255,255,255,0.04)"
        : "rgba(12,12,14,0.04)";
    ctx.fill();
    ctx.fillStyle = pal.mute;
    ctx.font = "600 9.5px 'Geist Mono', monospace";
    ctx.fillText("ESTIMATED SOURCES", x + 10, node.y + 68);
    ctx.fillStyle = lit || res?.count ? pal.orange : pal.mute;
    ctx.font = "800 22px Unbounded, Geist, sans-serif";
    const bigK = res?.count ? fmtCount(res.count) : "≈ K";
    ctx.fillText(bigK, x + 10, node.y + 98);

    const rows = [
      ["Confidence", res?.count ? `${res.count.confidence_percent}%` : "MC + Softmax"],
      ["Named sounds", res?.named_sounds ? `${res.named_sounds.length} detected` : "Open-vocab"],
      [
        "Ground Truth",
        res?.ground_truth?.available
          ? `K=${res.ground_truth.true_k} (SHA-256)`
          : res
            ? "No hash match"
            : "SHA-256 lookup",
      ],
      [
        "Total latency",
        res?.timings
          ? `${Math.round(Object.values(res.timings).reduce((a, b) => a + (Number(b) || 0), 0))} ms`
          : "—",
      ],
    ];
    rows.forEach(([k, v], i) => {
      const y = node.y + 132 + i * 27;
      ctx.fillStyle = pal.mute;
      ctx.font = "500 9.5px 'Geist Mono', monospace";
      ctx.fillText(k, x, y);
      ctx.fillStyle = lit || res ? pal.ink : pal.mute;
      ctx.font = "600 10.5px 'Geist Mono', monospace";
      ctx.fillText(v, x, y + 12);
    });
    ctx.restore();
  }

  function renderFrame() {
    const pal = palette();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
      canvas.width = W * dpr;
      canvas.height = H * dpr;
    }
    ctx.save();
    ctx.scale(dpr, dpr);

    ctx.fillStyle = pal.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = pal.wire;
    for (let gx = 20; gx < W; gx += 28) {
      for (let gy = 20; gy < H; gy += 28) {
        ctx.fillRect(gx, gy, 1.2, 1.2);
      }
    }

    const eDec = stageEnergy("decode"),
      eMel = stageEnergy("mel"),
      eCnt = stageEnergy("counter"),
      eTag = stageEnergy("tagger"),
      eSpk = stageEnergy("speaker"),
      eFus = stageEnergy("fusion");

    const pulse = animTime * 0.9;

    // 1 -> 2: Audio / Waveform (orange)
    drawBus(
      { x: NODES.decode.x + NODES.decode.w, y: 310 },
      { x: NODES.mel.x, y: 310 },
      eDec > 0.25 || eMel > 0.25,
      pal.orange,
      pulse,
      pal,
    );

    // 2 -> 1->3 Splitter -> ALL 3 models: strictly yellowish (pal.amber) coming out of Log-Mel
    const jx = 486,
      jy = 310;
    const melOutActive = eMel > 0.25 || eCnt > 0.25 || eTag > 0.25 || eSpk > 0.25;
    drawBus({ x: NODES.mel.x + NODES.mel.w, y: 310 }, { x: jx, y: jy }, melOutActive, pal.amber, pulse * 1.1, pal);
    drawBus({ x: jx, y: jy }, { x: NODES.counter.x, y: 110 }, eCnt > 0.25, pal.amber, pulse * 1.1, pal);
    drawBus({ x: jx, y: jy }, { x: NODES.tagger.x, y: 310 }, eTag > 0.25, pal.amber, pulse * 1.1, pal);
    drawBus({ x: jx, y: jy }, { x: NODES.speaker.x, y: 510 }, eSpk > 0.25, pal.amber, pulse * 1.1, pal);

    // Splitter hub circle (yellowish when lit, dark unlit when off)
    ctx.save();
    ctx.beginPath();
    ctx.arc(jx, jy, 11, 0, Math.PI * 2);
    ctx.fillStyle = pal.card;
    ctx.fill();
    ctx.lineWidth = melOutActive ? 2.2 : 1.3;
    ctx.strokeStyle = melOutActive ? pal.amber : pal.border;
    if (melOutActive) {
      ctx.shadowColor = pal.amber;
      ctx.shadowBlur = 10;
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.fillStyle = melOutActive ? pal.amber : pal.mute;
    ctx.font = "700 9px 'Geist Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText("1→3", jx, jy + 3);
    ctx.restore();

    // 3 Models -> 4 Fusion: each model keeps its distinct output color (orange, teal, violet)
    const outActive = eFus > 0.2 || (flowState === "play" && elapsed >= 2.65 && elapsed < 3.9);
    drawBus({ x: NODES.counter.x + NODES.counter.w, y: 110 }, { x: NODES.fusion.x, y: 248 }, outActive, pal.orange, pulse, pal);
    drawBus({ x: NODES.tagger.x + NODES.tagger.w, y: 310 }, { x: NODES.fusion.x, y: 310 }, outActive, pal.teal, pulse, pal);
    drawBus({ x: NODES.speaker.x + NODES.speaker.w, y: 510 }, { x: NODES.fusion.x, y: 372 }, outActive, pal.violet, pulse, pal);

    // Stage nodes
    drawNodeFrame(NODES.decode, eDec, pal, pal.orange);
    drawDecodeContent(NODES.decode, eDec, pal);

    drawNodeFrame(NODES.mel, eMel, pal, pal.amber);
    drawMelContent(NODES.mel, eMel, pal);

    drawNodeFrame(NODES.counter, eCnt, pal, pal.orange);
    drawCounterContent(NODES.counter, eCnt, pal);

    drawNodeFrame(NODES.tagger, eTag, pal, pal.teal);
    drawTaggerContent(NODES.tagger, eTag, pal);

    drawNodeFrame(NODES.speaker, eSpk, pal, pal.violet);
    drawSpeakerContent(NODES.speaker, eSpk, pal);

    drawNodeFrame(NODES.fusion, eFus, pal, pal.lime);
    drawFusionContent(NODES.fusion, eFus, pal);

    ctx.restore();
  }

  function renderInspector() {
    const t = lastResult?.timings || {};
    const specs = {
      decode: {
        title: "Stage 1 — Audio Ingest & WAV Decode",
        module: "audio.py → decode_mono() + app.py → analysis_clip()",
        libs: "soundfile (sf.read), librosa.resample, NumPy",
        shape: "Raw file → float32 mono 16,000 Hz → centred 8.0s clip [128,000]",
        latency: t.decode_ms != null ? `${t.decode_ms} ms` : "—",
        detail: fileInfo
          ? `Loaded ${fileInfo.filename} (${fmtTime(fileInfo.duration)}, ${fileInfo.sampleRate} Hz). Resampled to 16 kHz mono and centred on an 8-second window for inference.`
          : "Decodes WAV, MP3, FLAC, OGG, or M4A to mono float32 PCM, resamples to 16 kHz, and extracts the centred 8-second analysis clip.",
      },
      mel: {
        title: "Stage 2 — STFT & Log-Mel Spectrogram",
        module: "audio.py → waveform_to_logmel()",
        libs: "librosa.feature.melspectrogram, librosa.power_to_db, PyTorch",
        shape: "[128,000] → STFT (n_fft=1024, hop=256) → Log-Mel [1, 64, 501] in [-1, 0]",
        latency: t.feature_ms != null ? `${t.feature_ms} ms` : "—",
        detail:
          "Computes one shared 64-band log-mel spectrogram (40 Hz–8 kHz, 80 dB dynamic range) and fans it out to the 3 parallel neural branches.",
      },
      counter: {
        title: "Branch 3A — Source Counter (SourceCountCNN + MC-Dropout MLP)",
        module: "counting.py → SourceCountCNN + count_with_uncertainty()",
        libs: "PyTorch Conv2D Trunk (1→16→32→64→64, Pool 4×16) + MLP (4096→256→Drop 0.4→30)",
        shape: "[1, 64, 501] → Conv [64, 4, 16] → Flatten 4096 → FC 256 → 30 ordinal logits × 30 MC passes",
        latency: t.count_ms != null ? `${t.count_ms} ms` : "—",
        detail: lastResult?.count
          ? `Predicted ${fmtCount(lastResult.count)} sources (raw ${lastResult.count.raw_regression_value}, calibrated ${lastResult.count.calibrated_value}, confidence ${lastResult.count.confidence_percent}%, MC std ${lastResult.count.mc_dropout_std}).`
          : "Runs 30 stochastic Monte Carlo Dropout passes through the MLP head to measure both epistemic (MC) and aleatoric (softmax) uncertainty.",
      },
      tagger: {
        title: "Branch 3B — Open-Vocabulary Sound Tagger (3-Model Fusion)",
        module: "tagging.py → GroupFineTaggerCNN + AST + CLAP → tag_open_vocabulary()",
        libs: "PyTorch (tagger.pt), HuggingFace Transformers (MIT/ast-finetuned-audioset, laion/clap-htsat-unfused)",
        shape: "Log-Mel → Dual MLP (9 coarse groups + 32 FSD) + 16 kHz → AST (527 AudioSet) + CLAP zero-shot",
        latency: t.tag_ms != null ? `${t.tag_ms} ms` : "—",
        detail: lastResult?.named_sounds?.length
          ? `Detected ${lastResult.named_sounds.length} named sound(s): ${lastResult.named_sounds.slice(0, 6).map((s) => `${s.friendly_name || s.name} (${Math.round((s.confidence ?? s.prob ?? 0) * 100)}%)`).join(", ")}.`
          : "Combines the trained dual-head FSD CNN+MLP with pretrained Audio Spectrogram Transformer (AST) and CLAP embeddings, plus frame-level time ranges.",
      },
      speaker: {
        title: "Branch 3C — Speaker Diarization & Separation Gate",
        module: "separation.py → estimate_distinct_speakers() + IterativeMaskSeparator / SepFormer",
        libs: "SpeechBrain ECAPA-TDNN (spkrec-ecapa-voxceleb), SepFormer (wsj02mix), IterativeMaskSeparator + Quality Gates",
        shape: "16 kHz → ECAPA embeddings → cosine clustering; STFT MaskNet → separated stems + quality checks",
        latency: t.speaker_cluster_ms != null ? `${t.speaker_cluster_ms} ms` : "—",
        detail: lastResult?.speaker_clustering
          ? `Distinct speakers estimated: ${lastResult.speaker_clustering.distinct_speakers ?? "—"}. Separation eligible: ${lastResult.separation?.eligible ? "Yes" : "No (" + (lastResult.separation?.reason || "gated") + ")"}.`
          : "Clusters speaker embeddings with ECAPA-TDNN and routes eligible mixtures to SepFormer (2-speaker speech) or Iterative STFT MaskNet with correlation & energy quality gates.",
      },
      fusion: {
        title: "Stage 4 — Output Fusion & Ground Truth Verification",
        module: "app.py → /api/analyze + data.py → lookup_ground_truth()",
        libs: "Flask JSON API, SHA-256 content hash registry (15,559 dataset files), SQLite history",
        shape: "Branch A + Branch B + Branch C + SHA-256 GT → Unified JSON response",
        latency: Object.keys(t).length
          ? `${Math.round(Object.values(t).reduce((a, b) => a + (Number(b) || 0), 0))} ms total`
          : "—",
        detail: lastResult
          ? `Completed for ${lastResult.filename}. ${lastResult.ground_truth?.available ? `Matched dataset ground truth: K=${lastResult.ground_truth.true_k} (${lastResult.ground_truth.dataset}/${lastResult.ground_truth.split}).` : "No dataset SHA-256 match (showing model prediction only)."}`
          : "Merges count, uncertainty, named sound timelines, speaker count, separation eligibility, and optional SHA-256 dataset ground truth into the Analysis result.",
      },
    };
    const s = specs[selectedId] || specs.counter;
    inspector.innerHTML = `
      <div class="arch-insp-head">
        <div><strong>${esc(s.title)}</strong><div class="fine"><code>${esc(s.module)}</code></div></div>
        <span class="adm-pill ok">${esc(s.latency)}</span>
      </div>
      <div class="arch-insp-grid">
        <div><small class="fine">Libraries / Models</small><div>${esc(s.libs)}</div></div>
        <div><small class="fine">Data / Tensor Flow</small><div><code>${esc(s.shape)}</code></div></div>
      </div>
      <p style="margin-top:10px;font-size:0.9rem">${esc(s.detail)}</p>
    `;
  }

  function updateBar() {
    runBtn.disabled = !fileInfo || serverPending;
    fileLbl.textContent = fileInfo
      ? `${fileInfo.filename} · ${fmtTime(fileInfo.duration)} · ${fileInfo.sampleRate} Hz`
      : "No audio loaded";
    const isPlaying = flowState === "play";
    playBtn.innerHTML = isPlaying
      ? `<svg class="ic"><use href="#i-pause"/></svg><span>Stop flow</span>`
      : `<svg class="ic"><use href="#i-play"/></svg><span>Play flow</span>`;
    playBtn.disabled = serverPending;
    muteBtn.textContent = soundOn ? "Sound: On" : "Sound: Off";
    muteBtn.setAttribute("aria-pressed", String(soundOn));
  }

  function startLoop() {
    if (rafId || activeView !== "arch" || flowState === "off") return;
    lastTs = 0;
    rafId = requestAnimationFrame(tick);
  }

  function stopFlow() {
    flowState = "off";
    elapsed = 0;
    rungSteps.clear();
    stopAllBells();
    if (rafId) {
      cancelAnimationFrame(rafId);
      rafId = 0;
    }
    updateBar();
    if (activeView === "arch") renderFrame();
  }

  function startPlayFlow() {
    ensureBellLoaded();
    flowState = "play";
    elapsed = 0;
    rungSteps = new Set();
    updateBar();
    if (activeView === "arch") {
      ringStep("step1");
      startLoop();
    }
  }

  function tick(ts) {
    if (activeView !== "arch" || flowState === "off") {
      rafId = 0;
      if (activeView === "arch") renderFrame();
      return;
    }
    if (!lastTs) lastTs = ts;
    const dt = Math.min(0.05, (ts - lastTs) / 1000);
    lastTs = ts;
    animTime += dt * speed;
    elapsed += dt * speed;

    // If live server inference is still running past Step 3, hold smoothly inside Step 3 until result arrives
    if (serverPending && elapsed > STEP_SEC * 2.92) {
      elapsed = STEP_SEC * 2.92;
    }

    // Ring bell cleanly once at the exact 1-second boundary of each step (0s, 1s, 2s, 3s at 1x)
    if (elapsed >= 0) ringStep("step1");
    if (elapsed >= STEP_SEC) ringStep("step2");
    if (elapsed >= STEP_SEC * 2) ringStep("step3");
    if (elapsed >= STEP_SEC * 3) ringStep("step4");

    if (elapsed >= TOTAL_SEC) {
      flowState = "off";
      elapsed = 0;
      rungSteps.clear();
      rafId = 0;
      updateBar();
      renderFrame(); // Powered-off final state
      return;
    }

    renderFrame();
    rafId = requestAnimationFrame(tick);
  }

  function setView(v) {
    activeView = v;
    const isArch = v === "arch";
    viewAnalysis.classList.toggle("hidden", isArch);
    viewArch.classList.toggle("hidden", !isArch);
    tabAnalysis.setAttribute("aria-selected", String(!isArch));
    tabArch.setAttribute("aria-selected", String(isArch));
    if (isArch) {
      ensureBellLoaded();
      updateBar();
      renderInspector();
      renderFrame();
      if (flowState !== "off") startLoop();
    } else {
      stopAllBells();
      if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = 0;
      }
    }
  }

  tabAnalysis?.addEventListener("click", () => setView("analysis"));
  tabArch?.addEventListener("click", () => setView("arch"));

  pickBtn?.addEventListener("click", () => $("#audioInput")?.click());
  runBtn?.addEventListener("click", () => $("#goBtn")?.click());

  playBtn?.addEventListener("click", () => {
    if (flowState === "play") stopFlow();
    else startPlayFlow();
  });

  muteBtn?.addEventListener("click", () => {
    soundOn = !soundOn;
    if (!soundOn) stopAllBells();
    else ensureBellLoaded();
    updateBar();
  });

  document.querySelectorAll(".arch-speed button").forEach((b) => {
    b.addEventListener("click", () => {
      speed = Number(b.dataset.speed) || 1;
      document.querySelectorAll(".arch-speed button").forEach((x) => x.classList.toggle("active", x === b));
    });
  });

  canvas.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * W;
    const y = ((e.clientY - rect.top) / rect.height) * H;
    for (const n of Object.values(NODES)) {
      if (x >= n.x && x <= n.x + n.w && y >= n.y && y <= n.y + n.h) {
        selectedId = n.id;
        renderInspector();
        renderFrame();
        break;
      }
    }
  });

  // Repaint on theme change even while powered off
  new MutationObserver(() => {
    if (activeView === "arch") renderFrame();
  }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  // Sync with analyze.js events (normal Analysis view stays instant; Architecture view runs smooth 1s/step)
  window.addEventListener("ec:file", (e) => {
    const d = e.detail || {};
    fileInfo = { filename: d.filename, duration: d.duration, sampleRate: d.sampleRate };
    realWave = d.peaks || null;
    realSpec = computeSpectrogram(d.samples, d.sampleRate);
    lastResult = null;
    updateBar();
    renderInspector();
    if (activeView === "arch") renderFrame();
  });
  window.addEventListener("ec:reset", () => {
    fileInfo = realWave = realSpec = lastResult = null;
    serverPending = false;
    stopFlow();
    renderInspector();
  });
  window.addEventListener("ec:run-start", () => {
    serverPending = true;
    lastResult = null;
    if (activeView === "arch") {
      startPlayFlow();
    } else {
      updateBar();
    }
  });
  window.addEventListener("ec:run-stage", () => {
    // Choreographed 1s-per-step timeline handles stage transitions smoothly
  });
  window.addEventListener("ec:run-result", (e) => {
    serverPending = false;
    lastResult = e.detail || null;
    renderInspector();
    updateBar();
    // If Architecture view is currently running its 4s pass, let it continue smoothly into Step 4 with real results
    if (activeView === "arch" && flowState === "off") renderFrame();
  });
  window.addEventListener("ec:run-error", () => {
    serverPending = false;
    stopFlow();
  });

  updateBar();
  renderInspector();
})();
