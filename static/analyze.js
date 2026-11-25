/* EchoCount — Analyze page: upload, waveform player, API call, result rendering */
(() => {
  const { $, esc, fmtTime, toast, H } = window.EC;
  const EXT = [".wav", ".flac", ".mp3", ".ogg", ".m4a"],
    MAX = 40 * 1024 * 1024,
    WIN = 8;
  const el = {
    dz: $("#dropzone"),
    input: $("#audioInput"),
    fc: $("#fileCard"),
    fName: $("#fName"),
    fMeta: $("#fMeta"),
    clear: $("#clearBtn"),
    player: $("#player"),
    cv: $("#wave"),
    play: $("#playBtn"),
    tNow: $("#tNow"),
    tTot: $("#tTot"),
    go: $("#goBtn"),
    cancel: $("#cancelBtn"),
    empty: $("#sEmpty"),
    load: $("#sLoad"),
    loadTxt: $("#loadTxt"),
    bar: $("#loadBar"),
    err: $("#sErr"),
    errTxt: $("#errTxt"),
    res: $("#sRes"),
    retry: $("#retryBtn"),
    stages: $("#stages"),
  };
  let file = null,
    url = null,
    audio = null,
    peaks = null,
    dur = 0,
    crop = null,
    hl = null,
    raf = 0,
    busy = false,
    xhr = null,
    runId = 0;

  const show = (k) =>
    ["empty", "load", "err", "res"].forEach((n) =>
      el[n].classList.toggle("hidden", n !== k),
    );
  const bytes = (b) =>
    b < 1048576
      ? `${Math.round(b / 1024)} KB`
      : `${(b / 1048576).toFixed(2)} MB`;

  /* ---------- file handling ---------- */
  function setFile(f) {
    if (!f) return;
    const ext = "." + f.name.split(".").pop().toLowerCase();
    if (!EXT.includes(ext))
      return toast(
        "Unsupported file type. Use WAV, FLAC, MP3, OGG or M4A.",
        "err",
      );
    if (f.size === 0) return toast("That file is empty.", "err");
    if (f.size > MAX) return toast("That file is larger than 40 MB.", "err");
    reset(true);
    file = f;
    url = URL.createObjectURL(f);
    audio = new Audio(url);
    audio.preload = "metadata";
    audio.addEventListener("ended", () => {
      setPlayIcon(false);
      draw();
    });
    audio.addEventListener("loadedmetadata", () => {
      if (!dur && isFinite(audio.duration)) {
        dur = audio.duration;
        info();
      }
    });
    el.fName.textContent = f.name;
    el.fMeta.textContent = bytes(f.size);
    el.fc.classList.remove("hidden");
    el.player.classList.remove("hidden");
    el.go.disabled = false;
    show("empty");
    requestAnimationFrame(draw);
    decode(f);
  }
  function reset(keepInput) {
    runId++;
    if (xhr) {
      xhr.abort();
      xhr = null;
    }
    busy = false;
    if (audio) {
      audio.pause();
      audio = null;
    }
    if (url) URL.revokeObjectURL(url);
    cancelAnimationFrame(raf);
    file = url = peaks = crop = hl = null;
    dur = 0;
    el.fc.classList.add("hidden");
    el.player.classList.add("hidden");
    el.go.disabled = true;
    setPlayIcon(false);
    el.tNow.textContent = el.tTot.textContent = "0:00";
    if (!keepInput) {
      el.input.value = "";
      show("empty");
    }
  }
  function info() {
    el.tTot.textContent = fmtTime(dur);
    el.fMeta.textContent = `${bytes(file.size)} · ${fmtTime(dur)}${dur > WIN ? " · long-audio windowed separation enabled" : ""}`;
    crop = dur > WIN ? { start: (dur - WIN) / 2, end: (dur + WIN) / 2 } : null;
  }
  async function decode(f) {
    try {
      const AC = window.AudioContext || window.webkitAudioContext,
        ac = new AC();
      const buf = await ac.decodeAudioData(await f.arrayBuffer());
      ac.close && ac.close();
      if (f !== file) return;
      dur = buf.duration;
      const n = 900,
        d = buf.getChannelData(0),
        step = Math.max(1, Math.floor(d.length / n)),
        skip = Math.max(1, Math.ceil(step / 64));
      const p = new Float32Array(n);
      let mx = 0;
      for (let i = 0; i < n; i++) {
        let m = 0;
        for (
          let j = i * step;
          j < Math.min(d.length, (i + 1) * step);
          j += skip
        ) {
          const v = Math.abs(d[j]);
          if (v > m) m = v;
        }
        p[i] = m;
        if (m > mx) mx = m;
      }
      peaks = p.map((v) => v / (mx || 1));
    } catch {
      peaks = null;
      toast(
        "Waveform preview is not available for this file. You can still analyze it.",
      );
    }
    if (f === file) {
      info();
      draw();
    }
  }

  /* ---------- waveform ---------- */
  function draw() {
    const c = el.cv,
      r = c.getBoundingClientRect(),
      d = Math.min(devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(r.width * d)),
      h = Math.max(1, Math.round(r.height * d));
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
    const g = c.getContext("2d");
    g.clearRect(0, 0, w, h);
    const now = audio ? audio.currentTime : 0;
    if (!peaks || !dur) {
      g.fillStyle = "rgba(242,240,235,.3)";
      g.fillRect(0, h / 2 - d, w, 2 * d);
      return;
    }
    const bars = Math.max(1, Math.floor(w / (4 * d))),
      bw = w / bars;
    for (let i = 0; i < bars; i++) {
      const p = peaks[Math.floor((i / bars) * peaks.length)],
        tm = ((i + 0.5) / bars) * dur,
        bh = Math.max(3 * d, p * h * 0.92);
      const inCrop = !crop || (tm >= crop.start && tm <= crop.end),
        inHl = hl && tm >= hl.start && tm <= hl.end;
      g.fillStyle = inHl
        ? "#C8F04A"
        : tm <= now
          ? "#FF5A1F"
          : inCrop
            ? "#F2F0EB"
            : "rgba(242,240,235,.22)";
      const x = i * bw + bw * 0.19,
        y = (h - bh) / 2,
        rr = Math.min(bw * 0.31, 4 * d);
      g.beginPath();
      g.roundRect
        ? g.roundRect(x, y, bw * 0.62, bh, rr)
        : g.rect(x, y, bw * 0.62, bh);
      g.fill();
    }
    g.fillStyle = "#fff";
    g.fillRect(Math.min(w - 2 * d, (now / dur) * w), 0, 2 * d, h);
  }
  const setPlayIcon = (on) => {
    el.play.innerHTML = `<svg class="ic"><use href="#i-${on ? "pause" : "play"}"/></svg><span>${on ? "Pause" : "Play"}</span>`;
  };
  function tick() {
    draw();
    el.tNow.textContent = fmtTime(audio.currentTime);
    if (!audio.paused) raf = requestAnimationFrame(tick);
  }
  el.play.addEventListener("click", () => {
    if (!audio) return;
    if (audio.paused) {
      audio
        .play()
        .then(() => {
          setPlayIcon(true);
          tick();
        })
        .catch(() => toast("This browser cannot play that file.", "err"));
    } else {
      audio.pause();
      setPlayIcon(false);
      cancelAnimationFrame(raf);
      draw();
    }
  });
  el.cv.addEventListener("pointerdown", (e) => {
    if (!audio || !dur) return;
    const r = el.cv.getBoundingClientRect();
    audio.currentTime = Math.min(
      dur,
      Math.max(0, ((e.clientX - r.left) / r.width) * dur),
    );
    el.tNow.textContent = fmtTime(audio.currentTime);
    draw();
  });
  new ResizeObserver(() => file && draw()).observe(el.cv);

  /* ---------- dropzone ---------- */
  el.input.addEventListener("change", () => {
    setFile(el.input.files[0]);
    el.input.value = "";
  });
  el.dz.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      el.input.click();
    }
  });
  ["dragenter", "dragover"].forEach((n) =>
    el.dz.addEventListener(n, (e) => {
      e.preventDefault();
      el.dz.classList.add("over");
    }),
  );
  ["dragleave", "drop"].forEach((n) =>
    el.dz.addEventListener(n, (e) => {
      e.preventDefault();
      el.dz.classList.remove("over");
    }),
  );
  el.dz.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
  el.clear.addEventListener("click", () => reset(false));

  /* ---------- synthetic test mixtures ---------- */
  function synth(n) {
    const sr = 16000,
      len = sr * 8,
      d = new Float32Array(len),
      f0 = [120, 185, 250, 320, 150, 210];
    for (let s = 0; s < n; s++) {
      const f = f0[s % 6] * (1 + 0.03 * s),
        ph = Math.random() * 6.28,
        rate = 2.2 + s * 0.7;
      for (let i = 0; i < len; i++) {
        const t = i / sr,
          env =
            Math.pow(Math.max(0, Math.sin(6.283 * rate * t + ph)), 0.7) *
            (0.6 + 0.4 * Math.sin(t * 1.3 + s)),
          vib = 1 + 0.012 * Math.sin(t * 5.5 + s);
        let v = 0;
        for (let h = 1; h <= 6; h++) v += Math.sin(6.283 * f * h * vib * t) / h;
        d[i] += v * env * 0.22;
      }
    }
    let mx = 0;
    for (let i = 0; i < len; i++) {
      d[i] += (Math.random() - 0.5) * 0.01;
      mx = Math.max(mx, Math.abs(d[i]));
    }
    const buf = new ArrayBuffer(44 + len * 2),
      v = new DataView(buf),
      str = (o, t) =>
        [...t].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
    str(0, "RIFF");
    v.setUint32(4, 36 + len * 2, true);
    str(8, "WAVEfmt ");
    v.setUint32(16, 16, true);
    v.setUint16(20, 1, true);
    v.setUint16(22, 1, true);
    v.setUint32(24, sr, true);
    v.setUint32(28, sr * 2, true);
    v.setUint16(32, 2, true);
    v.setUint16(34, 16, true);
    str(36, "data");
    v.setUint32(40, len * 2, true);
    for (let i = 0; i < len; i++)
      v.setInt16(
        44 + i * 2,
        Math.max(-1, Math.min(1, (d[i] / (mx || 1)) * 0.9)) * 32767,
        true,
      );
    return new File(
      [buf],
      `synthetic-${n}-${n === 1 ? "source" : "sources"}.wav`,
      { type: "audio/wav" },
    );
  }
  document.querySelectorAll("[data-synth]").forEach((b) =>
    b.addEventListener("click", () => {
      toast("Generating a synthetic mixture");
      setTimeout(() => setFile(synth(+b.dataset.synth)), 30);
    }),
  );

  /* ---------- analysis pipeline: decode -> count -> tag -> separate -> zip ---------- */
  const STAGES = [
    ["decode", "Decode"],
    ["count", "Count"],
    ["tag", "Tag"],
    ["separate", "Separate"],
    ["zip", "ZIP"],
  ];
  let jobId = null,
    wavesurfers = [],
    sepState = null,
    tagDoneState = "done",
    lastNamedSounds = [];
  const stageHtml = (state) =>
    `<ol class="stages" aria-label="Pipeline progress">${STAGES.map(([k, l]) => `<li data-stage="${k}" data-state="${state[k] || "idle"}"><i></i>${l}<span class="sr-only"> ${state[k] || "idle"}</span></li>`).join("")}</ol>`;
  const setStages = (state) => {
    el.stages.innerHTML = stageHtml(state);
  };
  const destroyStems = () => {
    wavesurfers.forEach((w) => {
      try {
        w.destroy();
      } catch {}
    });
    wavesurfers = [];
  };

  function run() {
    if (!file || busy) return;
    busy = true;
    el.go.disabled = true;
    show("load");
    const id = ++runId;
    jobId = null;
    destroyStems();
    el.bar.classList.remove("indet");
    el.bar.style.width = "0%";
    el.loadTxt.textContent = "Uploading";
    setStages({ decode: "active" });
    const fd = new FormData();
    fd.append("audio", file);
    const x = new XMLHttpRequest();
    xhr = x;
    x.open("POST", "/api/analyze");
    x.timeout = 180000;
    x.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        const p = (e.loaded / e.total) * 100;
        el.bar.style.width = p + "%";
        el.loadTxt.textContent =
          p < 100
            ? `Uploading ${Math.round(p)}%`
            : "Decoding, counting and naming sounds";
      }
    };
    x.upload.onload = () => {
      el.loadTxt.textContent = "Decoding, counting and naming sounds";
      el.bar.classList.add("indet");
      setStages({ decode: "done", count: "active", tag: "active" });
    };
    x.onload = () => {
      if (id !== runId) return;
      let data = {};
      try {
        data = JSON.parse(x.responseText);
      } catch {}
      done();
      if (x.status === 401 || data.auth_required) {
        window.EC.clearHistoryCache?.();
        fail("Please sign in to analyze recordings. Redirecting to sign in…");
        setTimeout(() => (location.href = "/login?next=/analyze"), 900);
        return;
      }
      if (x.status >= 200 && x.status < 300) render(data);
      else fail(data.error || `The server answered with status ${x.status}.`);
    };
    x.onerror = () => {
      if (id !== runId) return;
      done();
      fail(
        "Could not reach the server. Check that python app.py is running, then try again.",
      );
    };
    x.ontimeout = () => {
      if (id !== runId) return;
      done();
      fail("The analysis took too long. Try a shorter file.");
    };
    x.send(fd);
  }
  const done = () => {
    busy = false;
    xhr = null;
    el.go.disabled = !file;
  };
  el.cancel.addEventListener("click", () => {
    runId++;
    if (xhr) xhr.abort();
    done();
    show("empty");
    toast("Analysis cancelled.");
  });
  function fail(msg) {
    el.errTxt.textContent = msg;
    show("err");
  }
  el.go.addEventListener("click", run);
  el.retry.addEventListener("click", run);

  /* ---------- compact tabbed results ---------- */
  const tip = (t) =>
    `<span class="q-tip" tabindex="0" role="note" aria-label="${esc(t)}" title="${esc(t)}">?</span>`;
  const metricTile = (l, v, help) =>
    `<div class="metric res-tile"><small>${esc(l)} ${help ? tip(help) : ""}</small><b>${esc(v ?? "—")}</b></div>`;
  const fmtRange = (r) => `${fmtTime(r.start || 0)}–${fmtTime(r.end || 0)}`;
  const soundRowHtml = (ns, winSec) => {
    const pct = Math.min(100, Math.max(0, (ns.confidence ?? ns.probability ?? 0) * 100));
    const trs = Array.isArray(ns.time_ranges) && ns.time_ranges.length
      ? ns.time_ranges.slice(0, 2).map((r) => `<button type="button" class="time-pill" data-start="${r.start}" data-end="${r.end}">${esc(fmtRange(r))}</button>`).join("")
      : `<span class="fine">0:00–${fmtTime(winSec || 8)}</span>`;
    return `<div class="snd-row">
      <div class="snd-meta"><strong>${esc(ns.name || ns.friendly_name || ns.class)}</strong><span class="snd-grp">${esc(ns.group || "")}</span></div>
      <div class="snd-times">${trs}</div>
      <div class="snd-conf"><span class="snd-bar"><i style="width:${pct.toFixed(0)}%"></i></span><code>${pct.toFixed(1)}%</code></div>
    </div>`;
  };

  function wireTimePills(root) {
    root.querySelectorAll(".time-pill").forEach((btn) => {
      btn.addEventListener("click", () => {
        const s0 = parseFloat(btn.dataset.start || "0"), s1 = parseFloat(btn.dataset.end || "0");
        hl = { start: s0, end: s1 };
        if (audio && isFinite(s0)) {
          audio.currentTime = Math.min(dur || s0, s0);
          el.tNow.textContent = fmtTime(audio.currentTime);
        }
        draw();
      });
    });
  }

  function render(data) {
    window.dispatchEvent(new CustomEvent("ec:analysis-result", { detail: data }));
    if (!data.trained || !data.count) {
      el.res.innerHTML = `<div class="result"><h3>No counting model available</h3><p class="fine">${esc(data.message || "The counting checkpoint is missing or incompatible.")}</p></div>`;
      show("res");
      return;
    }
    const c = data.count, a = data.audio || {}, s = data.separation || {}, tg = data.tags || {}, sc = data.speaker_clustering || null;
    const namedSounds = (Array.isArray(data.named_sounds) ? data.named_sounds : Array.isArray(tg.named_sounds) ? tg.named_sounds : [])
      .slice().sort((x, y) => (y.confidence ?? y.probability ?? 0) - (x.confidence ?? x.probability ?? 0));
    const topPairs = Array.isArray(data.overlapping_sounds) ? data.overlapping_sounds : Array.isArray(tg.overlapping_sounds) ? tg.overlapping_sounds : [];
    const totalPairs = Number(data.overlapping_pairs_total ?? tg.overlapping_pairs_total ?? topPairs.length);
    const visDef = Number(data.visible_default || 8);
    const winSec = a.analysed_window_seconds || 8;
    const gt = data.ground_truth && data.ground_truth.available ? data.ground_truth : null;
    lastNamedSounds = namedSounds;
    jobId = data.job_id;
    tagDoneState = tg.available ? "done" : "idle";

    const spread = c.uncertainty_std >= 0.5 ? Math.ceil(c.uncertainty_std - 1e-9) : 0;
    const mcStd = typeof c.mc_dropout_std === "number" ? c.mc_dropout_std : 0;
    const smStd = typeof c.softmax_std === "number" ? c.softmax_std : Math.sqrt(Math.max(0, c.uncertainty_std * c.uncertainty_std - mcStd * mcStd));
    const domFactor = smStd >= mcStd ? "softmax spread" : "MC-dropout variance";

    // Persistent verdict & 3-5 key chips
    let verdictHtml = "";
    if (gt) {
      const d = c.rounded_count - gt.true_k;
      verdictHtml = d === 0
        ? `<span class="gt-badge ok">Exact count match</span>`
        : `<span class="gt-badge diff">Off by ${Math.abs(d)} (${d > 0 ? "+" + d : d})</span>`;
    } else if (c.rounded_count > 20 || namedSounds.length > 20) {
      verdictHtml = `<span class="gt-badge diff">${c.rounded_count} sources detected; counts above 6 are unreliable</span>`;
    }
    const groups = Array.isArray(tg.groups) ? tg.groups : [];
    const activeGroups = groups.filter((g) => g.above_threshold);
    const keyChips = [
      `<span class="res-chip">Conf <b>${c.confidence_percent.toFixed(0)}%</b></span>`,
      `<span class="res-chip"><b>${namedSounds.length}</b> sound${namedSounds.length === 1 ? "" : "s"}</span>`,
      `<span class="res-chip"><b>${activeGroups.length}</b> group${activeGroups.length === 1 ? "" : "s"}</span>`,
      `<span class="res-chip">${fmtTime(a.duration_seconds || dur || 0)}</span>`,
      c.dense_warning ? `<span class="res-chip warn" title="${esc(c.dense_warning)}">Dense · unreliable ${tip(c.dense_warning)}</span>` : "",
      sc && sc.show_card ? `<span class="res-chip"><b>${esc(sc.distinct_speakers)}</b> voices</span>` : "",
    ].filter(Boolean).slice(0, 5).join("");

    // Tab 1: Overview (top 8 sounds, active groups, main separation CTA)
    const topSoundsHtml = namedSounds.length
      ? namedSounds.slice(0, visDef).map((ns) => soundRowHtml(ns, winSec)).join("")
      : `<p class="fine">No individual sound crossed the detection threshold.</p>`;
    const moreSoundsBtn = namedSounds.length > visDef
      ? `<button type="button" class="link-btn" id="jumpSoundsBtn">View all ${namedSounds.length} sounds →</button>`
      : "";
    const groupChipsHtml = activeGroups.length
      ? activeGroups.map((g) => `<span class="tag-chip above" title="Threshold ${(g.threshold * 100).toFixed(0)}%"><strong>${esc(g.group)}</strong> <small>${(g.probability * 100).toFixed(0)}%</small></span>`).join("")
      : `<span class="fine">No coarse group above threshold</span>`;
    const canSepNow = Boolean((s.can_separate && s.trained) || (Array.isArray(s.extended_modes) && s.extended_modes.length));
    const ovSepBtn = canSepNow
      ? `<button type="button" class="btn primary sm" id="ovSepBtn">Separate into ${s.speech_mode ? `${c.rounded_count} speakers` : `${Math.min(namedSounds.length || c.rounded_count, c.rounded_count)} stems`}</button>`
      : `<button type="button" class="btn sm" id="ovStemsTabBtn">Separation status →</button>`;

    // Tab 2: Sounds (full list in <=420px scroll box with search, sort, group filter, pagination >40, & pairs table)
    const uniqGroups = [...new Set(namedSounds.map((ns) => ns.group).filter(Boolean))];
    const grpFilters = [`<button type="button" class="flt-chip active" data-grp="">All (${namedSounds.length})</button>`]
      .concat(uniqGroups.map((g) => `<button type="button" class="flt-chip" data-grp="${esc(g)}">${esc(g)}</button>`))
      .join("");
    const renderPairRows = (pairs) => pairs.map((o) => {
      const pn = (o.sounds || []).map(esc).join(" + ");
      const ivs = (o.intervals || []).slice(0, 2).map((iv) => esc(fmtRange(iv))).join(", ");
      const durS = typeof o.total_overlap_seconds === "number" ? `${o.total_overlap_seconds.toFixed(1)}s` : "—";
      const cf = typeof o.confidence === "number" ? `${(o.confidence * 100).toFixed(0)}%` : "—";
      return `<tr><td><strong>${pn}</strong></td><td>${ivs || "—"}</td><td>${durS}</td><td>${cf}</td></tr>`;
    }).join("");
    const pairsSection = totalPairs > 0
      ? `<div class="res-subhead"><strong>Co-occurring pairs</strong><span class="fine">Top ${topPairs.length} of ${totalPairs}</span>${totalPairs > topPairs.length ? `<button type="button" class="link-btn" id="showAllPairsBtn">Show all ${totalPairs} pairs</button>` : ""}</div>
         <div class="pairs-scroll"><table class="kv pairs-tbl"><thead><tr><th>Pair</th><th>Intervals</th><th>Overlap</th><th>Conf</th></tr></thead><tbody id="pairsBody">${renderPairRows(topPairs)}</tbody></table></div>`
      : "";
    const fineAbove = (Array.isArray(tg.tags) ? tg.tags : []).filter((t) => t.above_threshold);
    const fineDisclosure = fineAbove.length
      ? `<details class="res-help"><summary>Compatible fine classes (${fineAbove.length})</summary><div class="tag-chips" style="margin-top:6px">${fineAbove.map((t) => `<span class="tag-chip above">${esc(t.friendly_name || t.name || t.class)} <small>${((t.probability ?? t.confidence ?? 0) * 100).toFixed(0)}%</small></span>`).join("")}</div></details>`
      : "";

    // Tab 3: Numbers (all raw/calibrated/uncertainty tiles with "?" tooltips + regime chips)
    const numTiles = [
      metricTile("Raw value", c.raw_regression_value.toFixed(2), "Direct SourceCountCNN regression estimate before calibration."),
      c.calibrated ? metricTile("Calibrated", c.calibrated_value.toFixed(2), "Validation-calibrated regression estimate.") : "",
      metricTile("Rounded count", c.rounded_count, "Nearest integer count from calibrated output."),
      sc && sc.show_card && typeof sc.distinct_speakers === "number"
        ? metricTile(sc.using_ecapa ? "Distinct voices" : "Voices (approx.)", sc.distinct_speakers, sc.limit_explanation || "Speaker clustering estimate (reliable for K<=3).")
        : "",
      metricTile("MC-dropout std", `±${mcStd.toFixed(2)}`, `Epistemic spread across ${c.mc_passes} stochastic forward passes.`),
      metricTile("Softmax spread", `±${smStd.toFixed(2)}`, "Aleatoric spread from the discrete class distribution."),
      metricTile("Total std", `±${c.uncertainty_std.toFixed(2)}`, `Combined uncertainty (dominated by ${domFactor}).`),
      metricTile("Class confidence", `${c.confidence_percent.toFixed(1)}%`, "Peak probability of the rounded count class."),
    ].join("");
    const infoChips = [
      c.sparse_regime && !c.dense_warning ? `<span class="res-chip">Sparse regime (K ≤ 8) ${tip("Validation-calibrated sparse specialist mapping active.")}</span>` : "",
      c.dense_warning ? `<span class="res-chip warn">Dense mixture limit ${tip(`${c.dense_warning} See /evaluation for error by K.`)}</span>` : "",
      c.out_of_range || c.rounded_count < c.min_sources ? `<span class="res-chip warn">Outside range ${tip(c.range_warning || `Calibrated for ${c.min_sources}–${c.max_sources} sources.`)}</span>` : "",
      a.duration_seconds > winSec + 0.05 ? `<span class="res-chip">Central ${winSec}s window ${tip(`Count/tags use ${fmtTime(a.crop_start_seconds)}–${fmtTime(a.crop_start_seconds + winSec)}; separation processes full ${fmtTime(a.duration_seconds)}.`)}</span>` : "",
    ].filter(Boolean).join("");

    // Tab 4: Reference (ground truth) — ONLY rendered when server explicitly includes ground_truth
    let refPanelHtml = "";
    if (gt) {
      const gtL = Array.isArray(gt.true_labels) ? gt.true_labels : [];
      const prL = namedSounds.map((ns) => `${ns.name || ns.friendly_name || ns.class} (${((ns.confidence ?? 0) * 100).toFixed(0)}%)`);
      const chipBox = (arr, idPrefix) => {
        if (!arr.length) return `<span class="fine">None</span>`;
        const init = arr.slice(0, 12).map((x) => `<span class="tag-chip">${esc(x)}</span>`).join("");
        const more = arr.length > 12 ? `<button type="button" class="link-btn" data-expand-chips="${idPrefix}">+${arr.length - 12} more</button>` : "";
        return `<div class="chip-scroll" id="${idPrefix}Box">${init}</div>${more}`;
      };
      refPanelHtml = `<div class="res-pane" id="pane-reference" role="tabpanel" aria-labelledby="rtab-reference" hidden>
        <div class="res-subhead">
          <span class="fine">Matched by audio hash (${esc(gt.dataset || "corpus")} · ${esc(gt.split || "split")})</span>
          <button type="button" class="btn sm" id="copyGtBtn">Copy labels</button>
        </div>
        <div class="gt-compare-grid">
          <div class="gt-col">
            <div class="gt-col-head">REFERENCE</div>
            <div class="gt-stat"><span>Count</span><strong>${esc(gt.true_k)}</strong></div>
            <div class="gt-stat"><span>Groups</span><span>${(gt.true_groups || []).map(esc).join(", ") || "—"}</span></div>
            <div class="gt-stat-block"><span>Labels (${gtL.length})</span>${chipBox(gtL, "gtRef")}</div>
          </div>
          <div class="gt-col">
            <div class="gt-col-head pred">MODEL</div>
            <div class="gt-stat"><span>Count</span><strong>≈${esc(c.rounded_count)}</strong></div>
            <div class="gt-stat"><span>Groups</span><span>${(tg.active_groups || []).map(esc).join(", ") || "None"}</span></div>
            <div class="gt-stat-block"><span>Detected (${prL.length})</span>${chipBox(prL, "gtMod")}</div>
          </div>
        </div>
      </div>`;
    }

    const tabsMeta = [
      ["overview", "Overview"],
      ["sounds", `Sounds (${namedSounds.length})`],
      ["numbers", "Numbers"],
      ...(gt ? [["reference", "Reference"]] : []),
      ["stems", "Stems"],
    ];

    el.res.innerHTML = `<div class="result res-compact">
      <header class="res-head">
        <div class="res-head-top">
          <div class="count-main"><b id="bigN" class="count-ripple">≈${c.rounded_count}</b><span class="count-pm">(±${spread})</span><span class="count-unit">${c.rounded_count === 1 ? "source" : "sources"}</span></div>
          ${verdictHtml}
        </div>
        <div class="res-chips">${keyChips}</div>
        <details class="res-help"><summary>How to read this</summary><p class="fine">Raw <code>${c.raw_regression_value.toFixed(2)}</code>${c.calibrated ? ` → calibrated <code>${c.calibrated_value.toFixed(2)}</code>` : ""} (${c.mc_passes} MC passes, total std ±${c.uncertainty_std.toFixed(2)}, dominated by ${domFactor}). Click any time pill to highlight &amp; seek.</p></details>
      </header>

      <div class="res-tabs" role="tablist" aria-label="Analysis result sections">
        ${tabsMeta.map(([id, lbl], i) => `<button type="button" role="tab" id="rtab-${id}" data-rtab="${id}" aria-controls="pane-${id}" aria-selected="${i === 0 ? "true" : "false"}" tabindex="${i === 0 ? "0" : "-1"}">${esc(lbl)}</button>`).join("")}
      </div>

      <div class="res-pane" id="pane-overview" role="tabpanel" aria-labelledby="rtab-overview">
        <div class="res-subhead"><strong>Top detected sounds</strong>${moreSoundsBtn}</div>
        <div class="snd-list">${topSoundsHtml}</div>
        <div class="res-subhead" style="margin-top:10px"><strong>Active sound groups</strong></div>
        <div class="tag-chips">${groupChipsHtml}</div>
        <div class="res-ov-cta">${ovSepBtn}</div>
      </div>

      <div class="res-pane" id="pane-sounds" role="tabpanel" aria-labelledby="rtab-sounds" hidden>
        <div class="snd-toolbar">
          <input type="search" id="sndSearch" class="search" placeholder="Filter sounds…" aria-label="Filter detected sounds">
          <select id="sndSort" class="admin-select" aria-label="Sort detected sounds"><option value="conf">Sort: Confidence</option><option value="time">Sort: Start time</option></select>
        </div>
        <div class="flt-bar" id="sndGrpBar" role="group" aria-label="Filter by group">${grpFilters}</div>
        <div class="snd-scroll" id="sndFullList"></div>
        ${pairsSection}
        ${fineDisclosure}
      </div>

      <div class="res-pane" id="pane-numbers" role="tabpanel" aria-labelledby="rtab-numbers" hidden>
        <div class="metrics res-num-grid">${numTiles}</div>
        ${infoChips ? `<div class="res-chips" style="margin-top:10px">${infoChips}</div>` : ""}
      </div>

      ${refPanelHtml}

      <div class="res-pane" id="pane-stems" role="tabpanel" aria-labelledby="rtab-stems" hidden>
        <div id="stemsBody"></div>
      </div>
    </div>`;

    // Tab switching + arrow-key navigation + URL hash persistence
    const tabBtns = [...el.res.querySelectorAll("[data-rtab]")];
    const activateTab = (id, setHash = true) => {
      if (!tabBtns.some((b) => b.dataset.rtab === id)) id = "overview";
      tabBtns.forEach((b) => {
        const on = b.dataset.rtab === id;
        b.setAttribute("aria-selected", on ? "true" : "false");
        b.tabIndex = on ? 0 : -1;
        const pane = $(`#pane-${b.dataset.rtab}`);
        if (pane) pane.hidden = !on;
      });
      if (setHash) {
        try { history.replaceState(null, "", `#tab=${id}`); } catch {}
      }
    };
    tabBtns.forEach((b, idx) => {
      b.addEventListener("click", () => activateTab(b.dataset.rtab));
      b.addEventListener("keydown", (e) => {
        if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
          e.preventDefault();
          const next = tabBtns[(idx + (e.key === "ArrowRight" ? 1 : tabBtns.length - 1)) % tabBtns.length];
          next.focus();
          activateTab(next.dataset.rtab);
        }
      });
    });
    const hashMatch = /^#tab=([a-z]+)$/.exec(location.hash || "");
    if (hashMatch) activateTab(hashMatch[1], false);

    $("#jumpSoundsBtn")?.addEventListener("click", () => activateTab("sounds"));
    $("#ovStemsTabBtn")?.addEventListener("click", () => activateTab("stems"));
    $("#ovSepBtn")?.addEventListener("click", () => {
      activateTab("stems");
      if (s.can_separate && s.trained) separate(null);
      else if (Array.isArray(s.extended_modes) && s.extended_modes.length) separate(s.extended_modes[0].mode);
    });

    // Sounds tab: live filter + sort + pagination (>40 rows)
    let activeGrp = "", pageSize = 40;
    const renderFullSounds = () => {
      const box = $("#sndFullList");
      if (!box) return;
      const q = ($("#sndSearch")?.value || "").trim().toLowerCase();
      const sortBy = $("#sndSort")?.value || "conf";
      let list = namedSounds.filter((ns) => {
        if (activeGrp && ns.group !== activeGrp) return false;
        if (q && !(ns.name || ns.friendly_name || ns.class || "").toLowerCase().includes(q) && !(ns.group || "").toLowerCase().includes(q)) return false;
        return true;
      });
      if (sortBy === "time") {
        list = list.slice().sort((x, y) => (x.time_ranges?.[0]?.start ?? 999) - (y.time_ranges?.[0]?.start ?? 999));
      }
      if (!list.length) {
        box.innerHTML = `<p class="fine">No matching sounds.</p>`;
        return;
      }
      const shown = list.slice(0, pageSize);
      box.innerHTML = shown.map((ns) => soundRowHtml(ns, winSec)).join("") +
        (list.length > shown.length ? `<button type="button" class="btn sm block" id="loadMoreSndBtn">Show more (${shown.length} of ${list.length})</button>` : "");
      wireTimePills(box);
      $("#loadMoreSndBtn")?.addEventListener("click", () => { pageSize += 40; renderFullSounds(); });
    };
    renderFullSounds();
    $("#sndSearch")?.addEventListener("input", () => { pageSize = 40; renderFullSounds(); });
    $("#sndSort")?.addEventListener("change", renderFullSounds);
    $("#sndGrpBar")?.addEventListener("click", (e) => {
      const b = e.target.closest("[data-grp]");
      if (!b) return;
      activeGrp = b.dataset.grp || "";
      el.res.querySelectorAll("#sndGrpBar .flt-chip").forEach((x) => x.classList.toggle("active", x === b));
      pageSize = 40;
      renderFullSounds();
    });

    // Co-occurring pairs "Show all N pairs" on-demand fetch
    $("#showAllPairsBtn")?.addEventListener("click", async (e) => {
      const btn = e.currentTarget;
      btn.disabled = true;
      btn.textContent = "Loading…";
      try {
        const r = await fetch(`/api/jobs/${jobId}/pairs`, { credentials: "same-origin" });
        const d = await r.json();
        if (r.ok && Array.isArray(d.pairs)) {
          $("#pairsBody").innerHTML = renderPairRows(d.pairs);
          btn.remove();
        } else {
          btn.textContent = "Unavailable";
        }
      } catch {
        btn.disabled = false;
        btn.textContent = `Show all ${totalPairs} pairs`;
      }
    });

    // Reference tab expand chips & copy labels
    if (gt) {
      const refMap = {
        gtRef: Array.isArray(gt.true_labels) ? gt.true_labels : [],
        gtMod: namedSounds.map((ns) => `${ns.name || ns.friendly_name || ns.class} (${((ns.confidence ?? 0) * 100).toFixed(0)}%)`),
      };
      el.res.querySelectorAll("[data-expand-chips]").forEach((btn) => {
        btn.addEventListener("click", () => {
          const k = btn.dataset.expandChips;
          const bx = $(`#${k}Box`);
          if (bx) bx.innerHTML = (refMap[k] || []).map((x) => `<span class="tag-chip">${esc(x)}</span>`).join("");
          btn.remove();
        });
      });
      $("#copyGtBtn")?.addEventListener("click", () => {
        navigator.clipboard?.writeText((gt.true_labels || []).join(", ")).then(() => toast("Reference labels copied.")).catch(() => {});
      });
    }

    wireTimePills($("#pane-overview"));
    show("res");
    window.__ecWave_analyzeHeaderWave?.pulse?.(0.5, 0.5);
    sepState = s;
    stemsEmpty(s, c.rounded_count, namedSounds);
  }

  function stemsEmpty(s, roundedCount, namedSounds) {
    const body = $("#stemsBody");

    let speechOfflineHtml = "";
    if (s.speech_mode && s.speech_separator && !s.speech_separator.available) {
      const cmds = (
        s.speech_separator.powershell_download_cmd ||
        s.speech_separator.download_commands ||
        ""
      ).trim();
      speechOfflineHtml = `<div class="notice info" style="margin-bottom:12px">
        <div><strong>Offline fallback (Speech separation):</strong> ${esc(s.speech_separator.offline_fallback_message || s.speech_separator.reason || "Pretrained SepFormer weights not found in models/sepformer-wsj02mix.")}</div>
        <pre class="term" style="margin-top:8px;padding:10px 12px;overflow-x:auto;user-select:all;font-size:.82rem;line-height:1.45"><code>${esc(cmds)}</code></pre>
      </div>`;
    }

    // Phase 0.2 & Phase 2: Enable Separate button ONLY when a real trained separator supports the request.
    const extModes = Array.isArray(s.extended_modes) ? s.extended_modes : [];
    const extBtnsHtml = extModes
      .map((m) => {
        const expTag = m.status === "experimental" ? " [Experimental]" : "";
        const metricTxt =
          typeof m.mean_sisdri_db === "number"
            ? ` (held-out ${m.mean_sisdri_db >= 0 ? "+" : ""}${m.mean_sisdri_db} dB SI-SDRi)`
            : "";
        return `<button class="btn ${m.status === "enabled" ? "primary" : ""}" data-sep-mode="${esc(m.mode)}" type="button">${esc(m.label)}${esc(expTag)}${esc(metricTxt)}</button>`;
      })
      .join(" ");

    if ((s.can_separate && s.trained) || extModes.length > 0) {
      let btnHtml = "";
      if (s.can_separate && s.trained) {
        let btnLabel = `Separate into ${roundedCount} stems`;
        if (s.speech_mode) {
          btnLabel = `Separate into ${roundedCount} speaker stems (Human 1, Human 2)`;
        } else if (Array.isArray(namedSounds) && namedSounds.length > 0) {
          const namesPreview = namedSounds
            .slice(0, 3)
            .map((x) => x.name || x.friendly_name)
            .join(", ");
          btnLabel = `Separate ${namedSounds.length} named sound${namedSounds.length === 1 ? "" : "s"} (${namesPreview}) + Residual`;
        }
        btnHtml = `<button class="btn primary" id="sepBtn" type="button">${esc(btnLabel)}</button> `;
      }
      body.innerHTML = `${speechOfflineHtml}<p class="fine">Extracts one audio stem per detected sound plus an <strong>Everything else (Residual)</strong> stem so nothing is lost. Runs in the background with quality-gate verification.</p>
        <div style="display:flex;gap:8px;flex-wrap:wrap">${btnHtml}${extBtnsHtml}</div>`;
      $("#sepBtn")?.addEventListener("click", () => separate(null));
      body.querySelectorAll("[data-sep-mode]").forEach((btn) =>
        btn.addEventListener("click", () => separate(btn.dataset.sepMode)),
      );
    } else {
      const msg = s.reason || "Separation not available for this recording.";
      body.innerHTML = `${speechOfflineHtml}<div class="state-inline" role="status"><strong>Separation not available for this recording</strong><p>${esc(msg)}</p></div>`;
    }
  }

  function separate(requestedMode = null) {
    const body = $("#stemsBody"),
      my = runId;
    setStages({
      decode: "done",
      count: "done",
      tag: tagDoneState,
      separate: "active",
    });
    const modeNote = requestedMode ? ` (${esc(requestedMode)})` : "";
    body.innerHTML = `<div role="status" aria-live="polite"><div class="eq"><i></i><i></i><i></i><i></i><i></i><i></i></div><p>Running background separation${modeNote} &amp; verifying stem quality gates…</p></div>${stageHtml({ decode: "done", count: "done", tag: tagDoneState, separate: "active" })}`;

    const payload = { job_id: jobId, async: true };
    if (requestedMode) payload.mode = requestedMode;

    const fetchAuthJson = (url, opts) =>
      fetch(url, { credentials: "same-origin", cache: "no-store", ...opts }).then((r) => {
        if (r.status === 401) {
          window.EC?.clearHistoryCache?.();
          location.href = "/login?next=/analyze";
          throw new Error("Session expired.");
        }
        return r.json().then((d) => ({ ok: r.ok, status: r.status, d }));
      });

    const pollJob = (pollUrl) =>
      fetchAuthJson(pollUrl).then(({ ok, d }) => {
        if (my !== runId) return null;
        if (!ok || d.status === "failed") throw new Error(d.error || "Separation failed.");
        if (d.status === "running" || d.status === "idle") {
          return new Promise((res) => setTimeout(res, 650)).then(() => pollJob(pollUrl));
        }
        return d;
      });

    fetchAuthJson("/api/separate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then(({ ok, status, d }) => {
        if (my !== runId) return null;
        if (!ok) throw new Error(d.error || "Separation failed.");
        if (status === 202 && d.poll_url) return pollJob(d.poll_url);
        return d;
      })
      .then((d) => {
        if (!d || my !== runId) return;
        const qFail = Boolean(d.quality_failed);
        const qc = d.quality_check || {};
        const sv = d.speaker_verification || {};
        const sc = d.self_consistency || {};
        const ec = qc.energy_conservation || {};
        const headline = sv.could_not_separate_speakers
          ? "Could not separate speakers — Separation failed quality check: stems are near-copies"
          : "Separation failed quality check: stems are near-copies";
        const embSimTxt =
          typeof sv.max_embedding_cosine_similarity === "number"
            ? `, max speaker embedding cos = ${sv.max_embedding_cosine_similarity.toFixed(3)}`
            : "";
        const qBanner = qFail
          ? `<div class="banner-quality-fail" role="alert">
          <div>${esc(headline)}</div>
          <small style="font-weight:400">${esc(d.quality_reason || "")} (max pairwise r = ${typeof qc.max_pairwise_corr === "number" ? qc.max_pairwise_corr.toFixed(3) : "—"}, max stem–mix r = ${typeof qc.max_stem_mix_corr === "number" ? qc.max_stem_mix_corr.toFixed(3) : "—"}${embSimTxt})</small>
        </div>`
          : "";
        const gateSummary = `<p class="fine" style="margin-bottom:10px">Quality gates: Self-consistency <strong>${sc.passed_count ?? 0}/${sc.total_stems ?? 0}</strong> verified · Max pairwise corr <code>${typeof qc.max_pairwise_corr === "number" ? qc.max_pairwise_corr.toFixed(3) : "0.000"}</code> (&lt; 0.90) · Energy conservation SNR <code>${typeof ec.reconstruction_snr_db === "number" ? ec.reconstruction_snr_db.toFixed(1) + " dB" : "exact"}</code></p>`;
        const zipStageState = qFail ? "idle" : "done";
        const zipActions = qFail
          ? `<div class="stem-actions"><button class="btn" id="zipBtn" type="button" disabled aria-disabled="true"><svg class="ic"><use href="#i-download"/></svg><span>ZIP unavailable (failed quality check)</span></button><span class="fine" id="zipMsg" role="status" aria-live="polite">Stems failed the quality check and are not offered as a successful separation.</span></div>`
          : `<div class="stem-actions"><button class="btn primary" id="zipBtn" type="button"><svg class="ic"><use href="#i-download"/></svg><span>Download all as ZIP</span></button><span class="fine" id="zipMsg" role="status" aria-live="polite"></span></div>`;
        body.innerHTML =
          qBanner +
          stageHtml({
            decode: "done",
            count: "done",
            tag: tagDoneState,
            separate: "done",
            zip: zipStageState,
          }) +
          gateSummary +
          `<div id="stemList" class="stem-list"></div>
          ${zipActions}`;
        buildStems(d.stems);
        if (!qFail) {
          $("#zipBtn").addEventListener("click", () =>
            downloadZip(d.download_url, false),
          );
        }
      })
      .catch((err) => {
        if (my !== runId) return;
        body.innerHTML = `<div class="state-inline err" role="alert"><strong>Separation not available for this recording</strong><p>${esc(err.message)}</p></div>`;
      });
  }

  function buildStems(stems) {
    const list = $("#stemList");
    destroyStems();
    const rows = stems.map((st) => {
      const row = document.createElement("div");
      row.className = "stem";
      row.dataset.index = st.index;
      const stemTitle = st.label || `Stem ${st.index}`;
      const badgeClass = st.low_confidence ? "uncertain" : "above";
      const badgeTxt =
        st.status_badge || (st.low_confidence ? "Low confidence" : "Verified");
      const dlUrl = st.download_url || `${st.url}?download=1`;
      row.innerHTML = `<div class="stem-head">
          <div><strong>${esc(stemTitle)}</strong> <span class="tag-chip ${badgeClass}" style="margin-left:6px;padding:2px 8px;font-size:.75rem">${esc(badgeTxt)}</span></div>
          <small>${(st.energy_share * 100).toFixed(1)}% of energy</small>
        </div>
        <div class="stem-wave" id="sw${st.index}" aria-label="Waveform of ${esc(stemTitle)}"></div>
        <div class="stem-ctl">
          <button class="btn sm" data-a="play" type="button" aria-label="Play ${esc(stemTitle)}"><svg class="ic"><use href="#i-play"/></svg><span>Play</span></button>
          <button class="btn sm" data-a="mute" type="button" aria-pressed="false" aria-label="Mute ${esc(stemTitle)}">Mute</button>
          <button class="btn sm" data-a="solo" type="button" aria-pressed="false" aria-label="Solo ${esc(stemTitle)}">Solo</button>
          <a class="btn sm" href="${esc(dlUrl)}" download="${esc(st.file || `stem_${st.index}.wav`)}" aria-label="Download ${esc(stemTitle)} WAV"><svg class="ic"><use href="#i-download"/></svg><span>Download WAV</span></a>
        </div>`;
      list.appendChild(row);
      return { st, row };
    });
    if (!window.WaveSurfer) {
      rows.forEach(({ st, row }) => {
        const dlUrl = st.download_url || `${st.url}?download=1`;
        row.querySelector(".stem-wave").innerHTML =
          `<audio controls src="${st.url}"></audio>`;
        row.querySelector(".stem-ctl").innerHTML =
          `<a class="btn sm" href="${esc(dlUrl)}" download="${esc(st.file || `stem_${st.index}.wav`)}">Download WAV</a>`;
      });
      toast("WaveSurfer.js could not be loaded; using basic players.");
      return;
    }
    const state = rows.map(() => ({ mute: false, solo: false }));
    const apply = () => {
      const anySolo = state.some((s) => s.solo);
      wavesurfers.forEach((w, i) =>
        w.setMuted(state[i].mute || (anySolo && !state[i].solo)),
      );
    };
    rows.forEach(({ st, row }, i) => {
      const ws = WaveSurfer.create({
        container: row.querySelector(".stem-wave"),
        url: st.url,
        height: 56,
        barWidth: 2,
        barGap: 1,
        barRadius: 2,
        waveColor: "rgba(120,118,112,.6)",
        progressColor: "#FF5A1F",
        cursorColor: "#FF5A1F",
        normalize: true,
      });
      wavesurfers.push(ws);
      ws.on("error", () => {
        row.querySelector(".stem-wave").innerHTML =
          '<p class="fine">This stem could not be loaded (it may have expired).</p>';
      });
      const playBtn = row.querySelector("[data-a=play]");
      const label = (on) => {
        playBtn.querySelector("span").textContent = on ? "Pause" : "Play";
        playBtn
          .querySelector("use")
          .setAttribute("href", `#i-${on ? "pause" : "play"}`);
      };
      ws.on("play", () => label(true));
      ws.on("pause", () => label(false));
      ws.on("finish", () => label(false));
      playBtn.addEventListener("click", () => ws.playPause());
      row.querySelector("[data-a=mute]").addEventListener("click", (e) => {
        state[i].mute = !state[i].mute;
        e.currentTarget.setAttribute("aria-pressed", state[i].mute);
        apply();
      });
      row.querySelector("[data-a=solo]").addEventListener("click", (e) => {
        state[i].solo = !state[i].solo;
        e.currentTarget.setAttribute("aria-pressed", state[i].solo);
        apply();
      });
    });
  }

  function downloadZip(url, qFail) {
    const btn = $("#zipBtn"),
      msg = $("#zipMsg");
    const origText = "Download all as ZIP";
    btn.disabled = true;
    msg.textContent = "Preparing download…";
    btn.querySelector("span").textContent = "Downloading…";
    const sBox = document.querySelector("#stemsBody .stages");
    if (sBox && !qFail)
      sBox.outerHTML = stageHtml({
        decode: "done",
        count: "done",
        tag: tagDoneState,
        separate: "done",
        zip: "active",
      });
    fetch(url)
      .then((r) => {
        if (!r.ok)
          return r.json().then((d) => {
            throw new Error(d.error || `HTTP ${r.status}`);
          });
        return r.blob().then((b) => ({ b, r }));
      })
      .then(({ b, r }) => {
        const m = /filename="?([^";]+)"?/.exec(
            r.headers.get("Content-Disposition") || "",
          ),
          a = document.createElement("a");
        a.href = URL.createObjectURL(b);
        a.download = m ? m[1] : "stems.zip";
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(a.href), 4000);
        msg.textContent = "Download started.";
        const sDone = document.querySelector("#stemsBody .stages");
        if (sDone)
          sDone.outerHTML = stageHtml({
            decode: "done",
            count: "done",
            tag: tagDoneState,
            separate: "done",
            zip: "done",
          });
      })
      .catch((err) => {
        msg.textContent = "Download failed: " + err.message;
        const sErr = document.querySelector("#stemsBody .stages");
        if (sErr)
          sErr.outerHTML = stageHtml({
            decode: "done",
            count: "done",
            tag: tagDoneState,
            separate: "done",
            zip: "idle",
          });
      })
      .finally(() => {
        btn.disabled = false;
        btn.querySelector("span").textContent = origText;
      });
  }
})();
