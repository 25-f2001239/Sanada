/* EchoCount — Interactive SVG Data-Flow Architecture & Technical Inspector (/analyze) */
(() => {
  const { $, esc } = window.EC;
  const viewAnalysis = $("#analysisView");
  const viewArch = $("#archView");
  const tabAnalysis = $("#viewTabAnalysis");
  const tabArch = $("#viewTabArch");
  const svg = $("#archSvg");
  const packetsLayer = $("#archPackets");
  const inspector = $("#archInspector");
  const playBtn = $("#archPlayBtn");
  const playIconUse = $("#archPlayIcon use");
  const playText = $("#archPlayText");
  const restartBtn = $("#archRestartBtn");
  const phasePill = $("#archPhasePill");
  const fileLbl = $("#archFileLbl");

  if (!svg || !viewArch || !inspector) return;

  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---- Verified Technical Metadata from Actual Python Source Code ---- */
  const NODE_SPECS = {
    input: {
      title: "01 · Audio Input & Container Verification",
      module: "audio.py → sniff_container(), validate_extension()",
      inputDim: "Multipart upload (.wav, .mp3, .flac · ≤ 40 MB)",
      outputDim: "Verified binary audio container stream (≤ 600.0 s)",
      library: "Python stdlib magic-byte inspection + Werkzeug secure_filename",
      role: "Validates file extension and inspects container magic bytes (RIFF/WAVE, fLaC, ID3/MPEG frame sync) before any decoder touches the file.",
      checks: "Rejects empty files, non-audio payloads, uploads > 40 MB, and recordings > 600 s with HTTP 422/413.",
      nextStage: "Streams verified container into PyAV decode_mono() for mono float32 conversion.",
    },
    decode: {
      title: "02 · PyAV Streaming Decode & 16 kHz Windowing",
      module: "audio.py → decode_mono(), resample(), app.py → analysis_clip()",
      inputDim: "Arbitrary sample rate (8–96 kHz), mono or multi-channel",
      outputDim: "float32 [128,000] (16,000 Hz mono · centered 8.0 s clip)",
      library: "PyAV (av.AudioResampler) + SciPy polyphase resample_poly",
      role: "Decodes audio frames in streaming chunks to mono float32, resamples to 16 kHz, and extracts the centered 8.0-second analysis window (or zero-pads shorter clips).",
      checks: "NaN/Inf sanitization via np.nan_to_num; preserves full-length 16 kHz waveform on disk for windowed stem separation.",
      nextStage: "Passes float32 [128,000] clip to shared log-mel feature extraction and open-vocabulary taggers.",
    },
    logmel: {
      title: "03 · Shared 64-Band Log-Mel Spectrogram",
      module: "audio.py → waveform_to_logmel()",
      inputDim: "Waveform float32 [128,000] (8.0 s @ 16 kHz)",
      outputDim: "Tensor float32 [1, 64, 501] (32,064 time-frequency cells)",
      library: "librosa.feature.melspectrogram + PyTorch tensor normalization",
      role: "Computes a 64-band power mel spectrogram (N_FFT=1024 / 64 ms window, HOP_LENGTH=256 / 16 ms hop, 40–8000 Hz) converted to decibel log scale and normalized per clip.",
      checks: "Computed once per upload and shared across SourceCountCNN and GroupFineTaggerCNN to eliminate redundant STFT work.",
      nextStage: "Broadcasts [1, 64, 501] simultaneously into the parallel Counting and Domain Tagging branches.",
    },
    counter: {
      title: "04A · SourceCountCNN + Sparse Specialist & 30× MC-Dropout",
      module: "counting.py → SourceCountCNN, conv_trunk(), count_with_uncertainty()",
      inputDim: "Log-Mel Tensor float32 [1, 1, 64, 501]",
      outputDim: "30 ordinal count logits (K ∈ [2..31]) + calibrated K̂ ± σ_total",
      library: "PyTorch (2,153,989 total parameters: 1.08M general + 1.07M K≤8 specialist)",
      role: "Runs a 3-block 2D CNN trunk (Conv2d 1→16→32→64 + BatchNorm + ReLU + AdaptiveAvgPool2d 4×16) into a 256-unit classifier with Dropout(p=0.4) across 30 stochastic forward passes.",
      checks: "Combines MC-dropout pass variance (σ_mc) with softmax distribution spread (σ_softmax), applies isotonic calibration, and flags dense mixtures (K > 6).",
      nextStage: "Forwards estimated source count K̂ and uncertainty bounds to Speaker Clustering, Separation eligibility, and Output Fusion.",
    },
    tagger: {
      title: "04B · Open-Vocabulary Audio Tagger (GroupFineCNN + AST + CLAP)",
      module: "tagging.py → GroupFineTaggerCNN, compute_ast_scores(), compute_clap_scores()",
      inputDim: "Log-Mel [1, 64, 501] + 16 kHz waveform (AST) + 48 kHz waveform (CLAP)",
      outputDim: "5 Coarse Groups + 32 Domain Classes + 527 AudioSet concepts + frame time ranges",
      library: "PyTorch GroupFineTaggerCNN (4.14 MB) + HuggingFace AST (346 MB) + CLAP HTSAT-Unfused (614 MB)",
      role: "Combines domain CNN probabilities (Group test mAP 0.993, Fine-32 mAP 0.805) with AST AudioSet-527 and zero-shot CLAP audio-text similarity (held-out macro AP 0.907), localizing active time intervals.",
      checks: "Applies per-class validation thresholds and reconciles fine sound detections with parent coarse groups and co-occurring overlap intervals.",
      nextStage: "Supplies named sound queries, active time windows, and speech_detected flag to Query/Speech Separation and Output Fusion.",
    },
    separator: {
      title: "04C · Speaker Clustering & Source Separation (ECAPA + SepFormer / Wiener)",
      module: "separation.py → estimate_distinct_speakers(), separate_by_queries(), IterativeMaskSeparator",
      inputDim: "16 kHz mono waveform + estimated count K̂ + named sound queries",
      outputDim: "K̂ isolated WAV stems + residual track (mix − Σ stems)",
      library: "SpeechBrain ECAPA-TDNN (192-D, 83 MB) + SepFormer WSJ0-2mix (113 MB) + Query-Conditioned HPSS Wiener",
      role: "Clusters 192-D ECAPA speaker embeddings for sparse speech (K ≤ 3). Routes 2-speaker speech to pretrained SepFormer and general sparse mixtures to windowed (8.0 s window, 2.0 s overlap) query-conditioned masking.",
      checks: "Enforces sparse reliability limit (K ≤ 6) and computes explicit residual stem so total mixture energy is preserved.",
      nextStage: "Sends candidate separated stems and residual waveform to the 3 Quality Gates in quality.py.",
    },
    fusion: {
      title: "05 · Output Fusion, 3 Quality Gates & ZIP Bundle",
      module: "quality.py → compute_stem_quality(), evaluate_stem_self_consistency() · app.py",
      inputDim: "Count K̂ ± σ + Named Sounds + Candidate Stems + Residual Waveform",
      outputDim: "JSON analysis payload + verified WAV stems + manifest.json ZIP archive",
      library: "NumPy / SciPy Pearson & L2 energy verification + re-tagging consistency gate",
      role: "Fuses count, uncertainty, sound tags, and speaker clustering into the analysis response, and subjects separated stems to three quantitative quality gates before enabling ZIP export.",
      checks: "Gate 1: max pairwise stem correlation < 0.90 & speaker embedding cosine < 0.82. Gate 2: energy conservation within ±5% relative L2 tolerance. Gate 3: re-tagger self-consistency verification.",
      nextStage: "Renders interactive waveform, detected sound time pills, Ground Truth comparison (when matched), and downloadable WAV/ZIP stems.",
    },
  };

  /* ---- Packet definitions along the 8 SVG paths ---- */
  const PATH_DEFS = [
    { id: "p-in-dec", flowId: "f-in-dec", color: "#38bdf8", phase: [0.0, 0.12] },
    { id: "p-dec-mel", flowId: "f-dec-mel", color: "#38bdf8", phase: [0.12, 0.25] },
    { id: "p-mel-cnt", flowId: "f-mel-cnt", color: "#f59e0b", phase: [0.25, 0.50] },
    { id: "p-mel-tag", flowId: "f-mel-tag", color: "#12b5a6", phase: [0.27, 0.52] },
    { id: "p-mel-sep", flowId: "f-mel-sep", color: "#a78bfa", phase: [0.29, 0.54] },
    { id: "p-cnt-fus", flowId: "f-cnt-fus", color: "#f59e0b", phase: [0.54, 0.82] },
    { id: "p-tag-fus", flowId: "f-tag-fus", color: "#12b5a6", phase: [0.56, 0.84] },
    { id: "p-sep-fus", flowId: "f-sep-fus", color: "#a78bfa", phase: [0.58, 0.86] },
  ];

  const STAGE_WINDOWS = [
    { nodes: ["input"], range: [0.0, 0.13], label: "STAGE 1 · AUDIO INPUT & VALIDATION" },
    { nodes: ["decode"], range: [0.12, 0.26], label: "STAGE 2 · PYAV 16 KHZ DECODE & 8S WINDOW" },
    { nodes: ["logmel"], range: [0.24, 0.42], label: "STAGE 3 · SHARED LOG-MEL SPECTROGRAM [1, 64, 501]" },
    { nodes: ["counter", "tagger", "separator"], range: [0.36, 0.76], label: "STAGE 4 · PARALLEL 3-BRANCH INFERENCE" },
    { nodes: ["fusion"], range: [0.74, 1.0], label: "STAGE 5 · OUTPUT FUSION & 3 QUALITY GATES" },
  ];

  const NS = "http://www.w3.org/2000/svg";
  const pathItems = PATH_DEFS.map((d) => {
    const pathEl = document.getElementById(d.id);
    const flowEl = document.getElementById(d.flowId);
    const g = document.createElementNS(NS, "g");
    g.setAttribute("opacity", "0");

    const halo = document.createElementNS(NS, "circle");
    halo.setAttribute("r", "8");
    halo.setAttribute("fill", d.color);
    halo.setAttribute("opacity", "0.26");

    const trail = document.createElementNS(NS, "circle");
    trail.setAttribute("r", "3.2");
    trail.setAttribute("fill", d.color);
    trail.setAttribute("opacity", "0.45");

    const core = document.createElementNS(NS, "circle");
    core.setAttribute("r", "4.4");
    core.setAttribute("fill", "#ffffff");
    core.setAttribute("stroke", d.color);
    core.setAttribute("stroke-width", "2");

    g.appendChild(halo);
    g.appendChild(trail);
    g.appendChild(core);
    packetsLayer.appendChild(g);

    return {
      ...d,
      pathEl,
      flowEl,
      g,
      halo,
      trail,
      core,
      len: pathEl ? pathEl.getTotalLength() : 0,
    };
  });

  const nodeEls = {};
  svg.querySelectorAll(".arch-node[data-node]").forEach((el) => {
    nodeEls[el.dataset.node] = el;
  });
  const waveBars = [...svg.querySelectorAll("#vizInputWave .wbar")];
  const melCells = [...svg.querySelectorAll("#vizMelGrid .mel-cell")];

  let activeView = "analysis";
  let selectedNode = "counter";
  let running = !reducedMotion;
  let speed = 1;
  let cycleProgress = 0.32; // Start with visible activity in parallel branches if static/reduced-motion
  let dashOffset = 0;
  let rafId = 0;
  let lastTs = 0;
  let liveSummary = null;

  function renderInspector(nodeKey) {
    const spec = NODE_SPECS[nodeKey] || NODE_SPECS.counter;
    const liveBanner = liveSummary
      ? `<div class="arch-live-banner">
          <span><strong>Last analyzed clip:</strong> ${esc(liveSummary.filename)}</span>
          <span>Count: <code>${esc(liveSummary.display)}</code></span>
          <span>Top sounds: <code>${esc(liveSummary.sounds || "—")}</code></span>
          <span>Latency: <code>${esc(liveSummary.latency)}</code></span>
        </div>`
      : "";
    inspector.innerHTML = `
      ${liveBanner}
      <div class="arch-insp-head">
        <div>
          <span class="portal-kicker">Technical Stage Inspector</span>
          <h3 id="archInspTitle">${esc(spec.title)}</h3>
        </div>
        <code class="arch-mod-badge">${esc(spec.module)}</code>
      </div>
      <div class="arch-insp-grid">
        <div class="arch-insp-card">
          <small>Input Representation</small>
          <code>${esc(spec.inputDim)}</code>
        </div>
        <div class="arch-insp-card">
          <small>Output Representation</small>
          <code>${esc(spec.outputDim)}</code>
        </div>
        <div class="arch-insp-card">
          <small>Model / Engine</small>
          <span>${esc(spec.library)}</span>
        </div>
        <div class="arch-insp-card">
          <small>Downstream Connection</small>
          <span>${esc(spec.nextStage)}</span>
        </div>
      </div>
      <div class="arch-insp-notes">
        <p><strong>Processing responsibility:</strong> ${esc(spec.role)}</p>
        <p class="fine"><strong>Verification &amp; quality checks:</strong> ${esc(spec.checks)}</p>
      </div>
    `;
  }

  function selectNode(nodeKey) {
    if (!NODE_SPECS[nodeKey]) return;
    selectedNode = nodeKey;
    Object.entries(nodeEls).forEach(([k, el]) => {
      const sel = k === nodeKey;
      el.classList.toggle("is-selected", sel);
      el.setAttribute("aria-pressed", sel ? "true" : "false");
    });
    renderInspector(nodeKey);
  }

  Object.entries(nodeEls).forEach(([k, el]) => {
    el.addEventListener("click", () => selectNode(k));
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        selectNode(k);
      }
    });
  });

  function syncPlayButton() {
    if (!playBtn) return;
    playText.textContent = running ? "Pause flow" : "Play flow";
    playBtn.setAttribute("aria-label", running ? "Pause data-flow animation" : "Play data-flow animation");
    if (playIconUse) playIconUse.setAttribute("href", running ? "#i-pause" : "#i-play");
  }

  function updateVisualFrame(p, timeSec) {
    // 1. Determine active pipeline stage(s)
    const activeNodes = new Set();
    let currentLabel = STAGE_WINDOWS[0].label;
    for (const sw of STAGE_WINDOWS) {
      if (p >= sw.range[0] && p <= sw.range[1]) {
        sw.nodes.forEach((n) => activeNodes.add(n));
        currentLabel = sw.label;
      }
    }
    if (phasePill) phasePill.textContent = currentLabel;

    Object.entries(nodeEls).forEach(([k, el]) => {
      el.classList.toggle("is-active-stage", activeNodes.has(k));
    });

    // 2. Position packets along SVG Bézier paths
    pathItems.forEach((item) => {
      if (!item.pathEl || !item.len) return;
      const [p0, p1] = item.phase;
      if (item.flowEl) {
        const flowActive = p >= p0 - 0.04 && p <= p1 + 0.08;
        item.flowEl.style.opacity = flowActive ? "0.85" : "0.18";
        item.flowEl.style.strokeDashoffset = String(-dashOffset);
      }
      if (p < p0 || p > p1) {
        item.g.setAttribute("opacity", "0");
        return;
      }
      const localT = Math.max(0, Math.min(1, (p - p0) / (p1 - p0)));
      // Smooth ease-in-out along each connector path
      const eased = localT < 0.5 ? 2 * localT * localT : 1 - Math.pow(-2 * localT + 2, 2) / 2;
      const pt = item.pathEl.getPointAtLength(eased * item.len);
      const trailPt = item.pathEl.getPointAtLength(Math.max(0, eased * item.len - 10));
      item.halo.setAttribute("cx", pt.x.toFixed(1));
      item.halo.setAttribute("cy", pt.y.toFixed(1));
      item.core.setAttribute("cx", pt.x.toFixed(1));
      item.core.setAttribute("cy", pt.y.toFixed(1));
      item.trail.setAttribute("cx", trailPt.x.toFixed(1));
      item.trail.setAttribute("cy", trailPt.y.toFixed(1));
      const edgeAlpha = Math.min(1, Math.sin(localT * Math.PI) * 1.6);
      item.g.setAttribute("opacity", edgeAlpha.toFixed(2));
    });

    // 3. Subtle input waveform & log-mel spectrogram evolution
    waveBars.forEach((bar, i) => {
      const amp = 0.3 + 0.7 * Math.abs(Math.sin(timeSec * 3.4 + i * 0.65) * Math.cos(timeSec * 1.8 - i * 0.4));
      const h = Math.round(6 + amp * 20);
      bar.setAttribute("height", String(h));
      bar.setAttribute("y", String(17 - Math.floor(h / 2)));
    });

    melCells.forEach((cell) => {
      const r = Number(cell.dataset.mr || 0);
      const c = Number(cell.dataset.mc || 0);
      const wave = 0.22 + 0.72 * Math.max(0, Math.sin(c * 0.62 - timeSec * 2.8 + r * 0.85));
      cell.setAttribute("opacity", wave.toFixed(2));
    });
  }

  function step(ts) {
    if (!running || activeView !== "arch") {
      rafId = 0;
      return;
    }
    if (!lastTs) lastTs = ts;
    const dt = Math.min(0.08, (ts - lastTs) / 1000);
    lastTs = ts;

    // Full cycle = 6.0 seconds at 1x speed
    cycleProgress = (cycleProgress + (dt * speed) / 6.0) % 1;
    dashOffset = (dashOffset + dt * speed * 42) % 280;
    updateVisualFrame(cycleProgress, ts / 1000);
    rafId = requestAnimationFrame(step);
  }

  function startLoop() {
    cancelAnimationFrame(rafId);
    lastTs = 0;
    // Measure path lengths now that SVG is visible
    pathItems.forEach((item) => {
      if (item.pathEl) item.len = item.pathEl.getTotalLength();
    });
    if (running && activeView === "arch" && !document.hidden) {
      rafId = requestAnimationFrame(step);
    } else {
      updateVisualFrame(cycleProgress, 1.2);
    }
  }

  function setView(viewName) {
    activeView = viewName === "arch" ? "arch" : "analysis";
    const isArch = activeView === "arch";
    viewAnalysis?.classList.toggle("hidden", isArch);
    viewArch.classList.toggle("hidden", !isArch);
    tabAnalysis?.setAttribute("aria-selected", isArch ? "false" : "true");
    tabArch?.setAttribute("aria-selected", isArch ? "true" : "false");
    if (isArch) {
      startLoop();
    } else {
      cancelAnimationFrame(rafId);
      rafId = 0;
    }
  }

  tabAnalysis?.addEventListener("click", () => setView("analysis"));
  tabArch?.addEventListener("click", () => setView("arch"));

  playBtn?.addEventListener("click", () => {
    running = !running;
    syncPlayButton();
    if (running) {
      startLoop();
    } else {
      cancelAnimationFrame(rafId);
      rafId = 0;
    }
  });

  restartBtn?.addEventListener("click", () => {
    cycleProgress = 0;
    dashOffset = 0;
    updateVisualFrame(0, 0);
    if (running && !rafId) startLoop();
  });

  viewArch.querySelectorAll(".arch-speed button[data-speed]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const next = Number(btn.dataset.speed || 1);
      if (!Number.isFinite(next) || next <= 0) return;
      speed = next;
      viewArch.querySelectorAll(".arch-speed button[data-speed]").forEach((b) => {
        const on = b === btn;
        b.classList.toggle("active", on);
        b.setAttribute("aria-pressed", on ? "true" : "false");
      });
    });
  });

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      cancelAnimationFrame(rafId);
      rafId = 0;
    } else if (running && activeView === "arch") {
      startLoop();
    }
  });

  window.addEventListener("ec:analysis-result", (e) => {
    const d = e.detail || {};
    if (!d.count) return;
    const t = d.timings || {};
    const names = (d.named_sounds || []).slice(0, 3).map((s) => s.name || s.friendly_name).filter(Boolean).join(", ");
    liveSummary = {
      filename: d.filename || "Uploaded recording",
      display: d.count.display || `≈${d.count.rounded_count}`,
      sounds: names || "None above threshold",
      latency: `decode ${t.decode_ms ?? "—"}ms · count ${t.count_ms ?? "—"}ms · tag ${t.tag_ms ?? "—"}ms`,
    };
    const outTxt = document.getElementById("vizCountOut");
    if (outTxt) outTxt.textContent = `K̂ = ${liveSummary.display}`;
    if (fileLbl) fileLbl.textContent = `Last run: ${liveSummary.filename} (${liveSummary.display})`;
    renderInspector(selectedNode);
  });

  // Initial setup
  syncPlayButton();
  selectNode("counter");
  updateVisualFrame(cycleProgress, 1.0);
  if (location.hash === "#arch" || new URLSearchParams(location.search).get("view") === "arch") {
    setView("arch");
  }
})();
