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

  /* ---------- results ---------- */
  const metric = (l, v) =>
    `<div class="metric"><small>${l}</small><b>${esc(v ?? "—")}</b></div>`;
  const fmtRange = (r) => `${fmtTime(r.start || 0)}–${fmtTime(r.end || 0)}`;

  function render(data) {
    if (!data.trained || !data.count) {
      el.res.innerHTML = `<div class="result"><div><h3>No counting model available</h3><p>${esc(data.message || "The counting checkpoint is missing or incompatible.")}</p></div>
        ${data.model_error ? `<div class="notice">${esc(data.model_error)}</div>` : ""}
        <div class="notice info">Add a compatible checkpoint (see <code>retrain_counter.py</code>). No value is shown without real model inference.</div></div>`;
      show("res");
      return;
    }
    const c = data.count,
      a = data.audio || {},
      s = data.separation || {},
      tg = data.tags || {},
      sc = data.speaker_clustering || null;
    const namedSounds = Array.isArray(data.named_sounds)
      ? data.named_sounds
      : Array.isArray(tg.named_sounds)
        ? tg.named_sounds
        : [];
    const overlaps = Array.isArray(data.overlapping_sounds)
      ? data.overlapping_sounds
      : Array.isArray(tg.overlapping_sounds)
        ? tg.overlapping_sounds
        : [];
    lastNamedSounds = namedSounds;
    jobId = data.job_id;
    tagDoneState = tg.available ? "done" : "idle";
    const spread =
      c.uncertainty_std >= 0.5 ? Math.ceil(c.uncertainty_std - 1e-9) : 0;
    const mcStd = typeof c.mc_dropout_std === "number" ? c.mc_dropout_std : 0;
    const smStd =
      typeof c.softmax_std === "number"
        ? c.softmax_std
        : Math.sqrt(
            Math.max(0, c.uncertainty_std * c.uncertainty_std - mcStd * mcStd),
          );
    const dominantFactor =
      smStd >= mcStd
        ? "softmax distribution spread"
        : "MC-dropout pass variance";
    let warn = "";
    if (a.duration_seconds > a.analysed_window_seconds + 0.05) {
      warn += `<div class="notice info">Count &amp; sound tagging used the central ${a.analysed_window_seconds} s window (${fmtTime(a.crop_start_seconds)} to ${fmtTime(a.crop_start_seconds + a.analysed_window_seconds)}). Full-length separation processes the entire ${fmtTime(a.duration_seconds)} recording in overlapping 8 s windows with 2 s cross-fade.</div>`;
    }
    if (c.dense_warning) {
      warn += `<div class="notice warn" role="alert"><strong>${esc(c.dense_warning)}</strong></div>`;
    }
    if (c.out_of_range || c.rounded_count < c.min_sources) {
      warn += `<div class="notice warn"><strong>Input outside model range:</strong> ${esc(c.range_warning || `Model calibrated for ${c.min_sources}–${c.max_sources} overlapping sources.`)}</div>`;
    } else if (c.sparse_regime && !c.dense_warning) {
      warn += `<div class="notice info"><strong>Sparse regime active (K ≤ 8):</strong> Validation-calibrated mapping applied to distinguish sparse mixtures (e.g., K=2 vs K=3).</div>`;
    }
    const calibText = c.calibrated
      ? `Raw model output <code>${c.raw_regression_value.toFixed(2)}</code> calibrated to <code>${c.calibrated_value.toFixed(2)}</code> (rounds to <strong>${c.rounded_count}</strong>).`
      : `Raw model output <code>${c.raw_regression_value.toFixed(2)}</code> rounds to <strong>${c.rounded_count}</strong>.`;

    // Distinct-voices card (shown only when sc.show_card is true: K<=3, non-contradictory)
    let spkClusterHtml = "";
    if (sc && sc.show_card) {
      const methodBadge = sc.using_ecapa
        ? `<code>ECAPA-TDNN speaker embeddings</code>`
        : `<code>Approximate (not real speaker embeddings)</code>`;
      const labelsTxt =
        Array.isArray(sc.labels) && sc.labels.length
          ? ` (${sc.labels.map((l) => esc(l)).join(", ")})`
          : "";
      spkClusterHtml = `<div class="notice info" role="status" aria-label="Distinct speaker clustering">
        <div><strong>Distinct voices: ${esc(sc.distinct_speakers)}</strong>${labelsTxt} · ${methodBadge}</div>
        <small>${esc(sc.limit_explanation || "Windowed speaker clustering is only reliable for non-overlapping speech or sparse mixtures (K ≤ 3).")}</small>
      </div>`;
    }

    // Named Sounds section (friendly name, confidence, coarse group, and frame-level time ranges)
    let namedSoundsHtml = "";
    if (namedSounds.length > 0) {
      const rows = namedSounds
        .map((ns) => {
          const confPct =
            ((ns.confidence ?? ns.probability ?? 0) * 100).toFixed(1) + "%";
          const trs =
            Array.isArray(ns.time_ranges) && ns.time_ranges.length
              ? ns.time_ranges
                  .map(
                    (r) =>
                      `<button type="button" class="btn sm time-pill" data-start="${r.start}" data-end="${r.end}" style="padding:2px 8px;font-size:.78rem">${esc(fmtRange(r))}</button>`,
                  )
                  .join(" ")
              : `<span class="fine">0:00–${fmtTime(a.analysed_window_seconds || 8)}</span>`;
          return `<div class="named-sound-row" style="display:flex;align-items:center;justify-content:space-between;gap:12px;padding:8px 12px;border:1px solid rgba(255,255,255,.1);border-radius:8px;margin-bottom:6px">
          <div>
            <strong>${esc(ns.name || ns.friendly_name || ns.class)}</strong>
            <small style="margin-left:8px;opacity:.75">${esc(ns.group || "")}</small>
          </div>
          <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
            <span>${trs}</span>
            <span class="tag-chip above" style="margin:0"><strong>${confPct}</strong></span>
          </div>
        </div>`;
        })
        .join("");

      let overlapHtml = "";
      if (overlaps.length > 0) {
        const ovItems = overlaps
          .map((o) => {
            const pairName = (o.sounds || []).map((x) => esc(x)).join(" + ");
            const ivs = (o.intervals || [])
              .map((iv) => esc(fmtRange(iv)))
              .join(", ");
            return `<span><strong>${pairName}</strong> (${ivs})</span>`;
          })
          .join(" &nbsp;·&nbsp; ");
        overlapHtml = `<p class="fine" style="margin-top:8px"><strong>Co-occurring overlapping sounds:</strong> ${ovItems}</p>`;
      }

      namedSoundsHtml = `<section class="tag-section" aria-labelledby="namedH">
        <h3 id="namedH" class="sub">Detected sounds <small>${namedSounds.length} named sound${namedSounds.length === 1 ? "" : "s"} with time ranges</small></h3>
        <div class="named-sounds-list">${rows}</div>
        ${overlapHtml}
      </section>`;
    }

    // Sound groups & compatible fine classes section
    let tagsHtml = "";
    if (
      tg &&
      tg.available &&
      ((Array.isArray(tg.groups) && tg.groups.length > 0) ||
        (Array.isArray(tg.tags) && tg.tags.length > 0))
    ) {
      const groups = Array.isArray(tg.groups) ? tg.groups : [];
      const aboveGroups = groups.filter((g) => g.above_threshold);
      const visibleGroups = groups.filter(
        (g) => g.above_threshold || g.probability * 100 >= 0.5,
      );
      const zeroGroups = groups.filter(
        (g) => !g.above_threshold && g.probability * 100 < 0.5,
      );

      const groupChips = visibleGroups
        .map((g) => {
          const isUncertain = !g.above_threshold;
          const pct = (g.probability * 100).toFixed(1) + "%";
          return `<span class="tag-chip ${isUncertain ? "uncertain" : "above"}" title="Validation threshold: ${(g.threshold * 100).toFixed(0)}%"><strong>${esc(g.group)}</strong> <small>${pct}${isUncertain ? " · uncertain" : ""}</small></span>`;
        })
        .join("");

      const collapsedZeroHtml =
        zeroGroups.length > 0
          ? `<p class="fine" style="margin-top:6px">Other groups (not detected): ${zeroGroups.map((g) => esc(g.group)).join(", ")}</p>`
          : "";

      const fineTags = Array.isArray(tg.tags) ? tg.tags : [];
      const aboveFine = fineTags.filter((t) => t.above_threshold);

      let fineBlock = "";
      if (aboveFine.length > 0) {
        const fineChips = aboveFine
          .map((t) => {
            const pct =
              ((t.probability ?? t.confidence ?? 0) * 100).toFixed(1) + "%";
            const dispName =
              t.friendly_name || t.name || t.class.replace(/_/g, " ");
            return `<span class="tag-chip above" title="Group: ${esc(t.group || "")} · Threshold: ${(t.threshold * 100).toFixed(0)}%">${esc(dispName)} <small>${pct}</small></span>`;
          })
          .join("");
        fineBlock = `<h4 class="sub" style="margin-top:12px;font-size:.9rem">Compatible fine classes <small>${aboveFine.length} above threshold</small></h4><div class="tag-chips">${fineChips}</div>`;
      } else {
        fineBlock = `<p class="fine" style="margin-top:10px"><strong>Compatible fine classes:</strong> No fine class above threshold.</p>`;
      }

      tagsHtml = `<section class="tag-section" aria-labelledby="tagsH">
        <h3 id="tagsH" class="sub">Sound groups (primary) <small>${aboveGroups.length} active above validation threshold</small></h3>
        <div class="tag-chips">${groupChips || '<span class="fine">No group above threshold</span>'}</div>
        ${collapsedZeroHtml}
        ${fineBlock}
        <p class="fine">${esc(tg.caveat || "Coarse sound groups are shown first. Fine classes are secondary details and never contradict active groups.")}</p>
      </section>`;
    } else {
      tagsHtml = `<section class="tag-section" aria-labelledby="tagsH">
        <h3 id="tagsH" class="sub">Sound groups &amp; classes</h3>
        <div class="state-inline" role="status"><strong>Tagger not available</strong><p>${esc((tg && tg.reason) || "No trained tagger checkpoint found.")}</p></div>
      </section>`;
    }

    const spkMetric =
      sc && sc.show_card && typeof sc.distinct_speakers === "number"
        ? metric(
            sc.using_ecapa
              ? "Distinct voices (ECAPA)"
              : "Distinct voices (Approx.)",
            sc.distinct_speakers,
          )
        : "";

    el.res.innerHTML = `<div class="result">
      <div id="resStages" class="res-stages">${stageHtml({ decode: "done", count: "done", tag: tagDoneState, separate: "idle", zip: "idle" })}</div>
      <div class="count-card" role="group" aria-label="Estimated source count">
        <div class="count-main"><b id="bigN">≈${c.rounded_count}</b><span class="count-pm">(±${spread})</span><span class="count-unit">${c.rounded_count === 1 ? "source" : "sources"}</span></div>
        <p>${calibText} Uncertainty band decomposition (${c.mc_passes} MC-dropout passes): MC-dropout std <code>±${mcStd.toFixed(2)}</code> + softmax spread <code>±${smStd.toFixed(2)}</code> → combined std <code>±${c.uncertainty_std.toFixed(2)}</code> (dominated by ${dominantFactor}).</p>
      </div>
      ${spkClusterHtml}
      ${namedSoundsHtml}
      <div class="metrics">${metric("Raw value", c.raw_regression_value.toFixed(2))}${c.calibrated ? metric("Calibrated", c.calibrated_value.toFixed(2)) : ""}${metric("Rounded count", c.rounded_count)}${spkMetric}${metric("MC-dropout std", "±" + mcStd.toFixed(2))}${metric("Softmax spread", "±" + smStd.toFixed(2))}${metric("Total std", "±" + c.uncertainty_std.toFixed(2))}${metric("Class confidence", c.confidence_percent.toFixed(1) + "%")}</div>
      ${warn}
      ${tagsHtml}
      <div class="notice info">Known behaviour: the counter is least reliable on dense mixtures. See the <a href="/evaluation">Evaluation</a> tab for measured error by source count.</div>
      <section class="stems" aria-labelledby="stemsH"><h3 id="stemsH" class="sub">Separated stems</h3><div id="stemsBody"></div></section>
    </div>`;

    // Interactive time-range pills highlight the waveform and seek audio
    el.res.querySelectorAll(".time-pill").forEach((btn) => {
      btn.addEventListener("click", () => {
        const s0 = parseFloat(btn.dataset.start || "0"),
          s1 = parseFloat(btn.dataset.end || "0");
        hl = { start: s0, end: s1 };
        if (audio && isFinite(s0)) {
          audio.currentTime = Math.min(dur || s0, s0);
          el.tNow.textContent = fmtTime(audio.currentTime);
        }
        draw();
      });
    });

    show("res");
    sepState = s;
    stemsEmpty(s, c.rounded_count, namedSounds);
    H.add({
      id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
      ts: Date.now(),
      filename: data.filename || file.name,
      count: c.rounded_count,
      conf: +(c.confidence_percent / 100).toFixed(3),
      duration: a.duration_seconds || dur,
      counts: [c.rounded_count],
    });
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
    // Otherwise hide the Separate button and show a plain "Separation not available for this recording" message.
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
      body.innerHTML = `${speechOfflineHtml}<p class="fine">Extracts one audio stem per detected sound plus an <strong>Everything else (Residual)</strong> stem so nothing is lost.</p>
        <button class="btn primary" id="sepBtn" type="button">${esc(btnLabel)}</button>`;
      $("#sepBtn").addEventListener("click", separate);
    } else {
      const msg = s.reason || "Separation not available for this recording.";
      body.innerHTML = `${speechOfflineHtml}<div class="state-inline" role="status"><strong>Separation not available for this recording</strong><p>${esc(msg)}</p></div>`;
    }
  }

  function separate() {
    const body = $("#stemsBody"),
      my = runId;
    setStages({
      decode: "done",
      count: "done",
      tag: tagDoneState,
      separate: "active",
    });
    body.innerHTML = `<div role="status" aria-live="polite"><div class="eq"><i></i><i></i><i></i><i></i><i></i><i></i></div><p>Separating named sounds &amp; verifying stem quality gates…</p></div>${stageHtml({ decode: "done", count: "done", tag: tagDoneState, separate: "active" })}`;
    fetch("/api/separate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: jobId }),
    })
      .then((r) => r.json().then((d) => ({ ok: r.ok, d })))
      .then(({ ok, d }) => {
        if (my !== runId) return;
        if (!ok) throw new Error(d.error || "Separation failed.");
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
