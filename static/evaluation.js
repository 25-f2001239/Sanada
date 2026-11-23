/* EchoCount - Evaluation tab: renders /api/evaluation (produced by evaluate_counter.py). No values are hard-coded. */
(() => {
  const { $, esc } = window.EC;
  const pct = (v) => (v * 100).toFixed(1) + "%",
    f2 = (v) => (+v).toFixed(2),
    sg = (v) => (v > 0 ? "+" : "") + (+v).toFixed(2);
  const state = $("#evState"),
    body = $("#evBody");
  let data = null;

  fetch("/api/evaluation")
    .then((r) => r.json())
    .then((d) => {
      if (!d.available) {
        state.innerHTML = `<h3>Evaluation not available</h3><p>${esc(d.reason || "No results.")}</p>`;
        return;
      }
      data = d;
      state.classList.add("hidden");
      body.classList.remove("hidden");
      render(d);
    })
    .catch(() => {
      state.innerHTML =
        "<h3>Evaluation not available</h3><p>The server could not be reached.</p>";
    });

  const metric = (l, v, s = "") =>
    `<div class="metric"><small>${l}</small><b>${esc(v)}</b>${s ? `<small>${esc(s)}</small>` : ""}</div>`;

  function render(d) {
    const o = d.overall;
    const ho =
      d.held_out_speaker_evaluation ||
      (d.checkpoint && d.checkpoint.held_out_speaker_eval) ||
      {};
    const hoMetric =
      ho.exact_accuracy != null
        ? metric(
            "Held-out-speaker Exact",
            pct(ho.exact_accuracy),
            `MAE ${f2(ho.mae || 0)} · ±1 ${pct(ho.within_one || 0)}`,
          )
        : "";
    $("#evMetrics").innerHTML =
      metric("MAE", f2(o.mae), "sources") +
      metric("Exact-count accuracy", pct(o.exact_accuracy)) +
      metric("Within ±1", pct(o.within_one)) +
      metric("Mean bias", sg(o.bias), "predicted − true") +
      hoMetric +
      metric("Clips", o.n, d.dataset.name + " / " + d.dataset.split);

    const notes = [];
    if (d.stale)
      notes.push(
        '<div class="notice">These results were produced by a different checkpoint than the one currently loaded. Re-run <code>python evaluate_counter.py --full</code>.</div>',
      );
    if (!d.checkpoint_present)
      notes.push(
        '<div class="notice">The counting checkpoint is not present on this server; results are from an earlier run.</div>',
      );
    if (ho && ho.exact_accuracy != null) {
      notes.push(
        `<div class="notice info"><strong>Held-Out-Speaker Generalisation (unseen speakers ${(ho.held_out_speakers || []).join(", ")}, n=${ho.n || 352}):</strong> Exact accuracy <b>${pct(ho.exact_accuracy)}</b>, Within ±1 <b>${pct(ho.within_one || 0)}</b>, MAE <b>${f2(ho.mae || 0)}</b>, Bias <b>${sg(ho.bias || 0)}</b>.</div>`,
      );
    }
    if (d.separation_eval && d.separation_eval.available) {
      const se = d.separation_eval;
      const seNotice = !se.trained
        ? `<div class="banner-untrained" role="alert"><strong>Separation Evaluation: ${esc(se.warning || "Untrained demo weights")}</strong></div>`
        : `<div class="notice info"><strong>Separation Evaluation:</strong> Overall Mean SI-SDRi: ${sg(se.overall.mean_sisdri_db)} dB (${esc(se.overall.performance_level)}).</div>`;
      notes.push(seNotice);
    }
    const lk = d.dataset.leakage_check;
    if (lk && lk.speech_speaker_note) {
      notes.push(
        `<div class="notice info"><strong>Speaker split audit:</strong> ${esc(lk.speech_speaker_note)}</div>`,
      );
    }
    const s = d.saturation;
    if (s && s.evaluated && s.direction !== "none") {
      const dir = s.direction === "overcounting" ? "over" : "under";
      notes.push(
        `<div class="notice" role="note"><strong>Known limitation: density saturation.</strong> On the ${s.n_high} test clips with ${s.high_count_threshold}+ true sources the model ${dir}counts by ${f2(Math.abs(s.mean_bias_high))} sources on average` +
          (s.mean_bias_low != null
            ? `, versus ${sg(s.mean_bias_low)} on sparser clips`
            : "") +
          `. Mean predictions stop rising at about ${f2(s.max_mean_predicted)} sources, from true count ${s.predictions_plateau_from_true_count} onward. Computed from this evaluation run.</div>`,
      );
    } else if (s && s.evaluated) {
      notes.push(
        `<div class="notice info">No systematic density-driven bias beyond ±${s.tolerance} sources was found at ${s.high_count_threshold}+ true sources in this run.</div>`,
      );
    }
    $("#evNotices").innerHTML = notes.join("");

    $("#evTable").innerHTML =
      '<thead><tr><th scope="col">True count</th><th scope="col">n</th><th scope="col">Mean pred.</th><th scope="col">Bias</th><th scope="col">MAE</th><th scope="col">Exact</th><th scope="col">±1</th></tr></thead><tbody>' +
      d.buckets
        .map((b) => {
          const unrel =
            b.unreliable || b.n < 30
              ? ' <small title="Fewer than 30 clips">(unreliable: n&lt;30)</small>'
              : "";
          return `<tr><th scope="row">${esc(b.label)}${unrel}</th><td>${b.n}</td><td>${f2(b.mean_pred)}</td><td>${sg(b.bias)}</td><td>${f2(b.mae)}</td><td>${pct(b.exact_accuracy)}</td><td>${pct(b.within_one)}</td></tr>`;
        })
        .join("") +
      "</tbody>";

    // Render Before vs After Comparison Table (Overall + Speech + FSD)
    const cmpTbl = $("#evCmpTable");
    if (cmpTbl) {
      const bComp = d.before_comparison || {};
      const bPerK = bComp.per_count || [];
      const aPerK = d.per_count || [];
      const keyK = [2, 3, 4, 6, 9, 15, 20, 25];
      let rowsHtml = "";
      keyK.forEach((k) => {
        const aItem = aPerK.find((p) => p.true_k === k);
        const bItem = bPerK.find((p) => p.true_k === k);
        if (!aItem) return;
        const n = aItem.n;
        const unrel =
          aItem.unreliable || n < 30
            ? " <small>(unreliable n&lt;30)</small>"
            : "";
        const bExact = bItem ? pct(bItem.exact_accuracy) : "N/A";
        const aExact = `${pct(aItem.exact_accuracy)} [${pct(aItem.exact_ci_95[0])}, ${pct(aItem.exact_ci_95[1])}]`;
        const bW1 = bItem ? pct(bItem.within_one) : "N/A";
        const aW1 = `${pct(aItem.within_one)} [${pct(aItem.within1_ci_95[0])}, ${pct(aItem.within1_ci_95[1])}]`;
        const bMae = bItem ? f2(bItem.mae) : "N/A";
        const aMae = `${f2(aItem.mae)} [${f2(aItem.mae_ci_95[0])}, ${f2(aItem.mae_ci_95[1])}]`;

        rowsHtml += `<tr><th scope="row" rowspan="3">K = ${k}${unrel}</th><td rowspan="3">${n}</td><td>Exact Acc</td><td>${bExact}</td><td><strong>${aExact}</strong></td></tr>
          <tr><td>Within ±1</td><td>${bW1}</td><td><strong>${aW1}</strong></td></tr>
          <tr><td>MAE</td><td>${bMae}</td><td><strong>${aMae}</strong></td></tr>`;
      });

      const spk =
        (d.by_family && d.by_family.speech) || d.speech_breakdown || {};
      const spkAfter = spk.after_test || spk.test_after;
      const spkBefore = spk.before_test || spk.test_before;
      if (spkAfter && spkBefore) {
        const sb = spkBefore.overall,
          sa = spkAfter.overall;
        rowsHtml += `<tr><th scope="row" rowspan="3">SPEECH ONLY (Org_*)</th><td rowspan="3">${sa.n}</td><td>Exact Acc</td><td>${pct(sb.exact_accuracy)}</td><td><strong>${pct(sa.exact_accuracy)} [${pct(sa.exact_ci_95[0])}, ${pct(sa.exact_ci_95[1])}]</strong></td></tr>
          <tr><td>Within ±1</td><td>${pct(sb.within_one)}</td><td><strong>${pct(sa.within_one)}</strong></td></tr>
          <tr><td>MAE / Bias</td><td>${f2(sb.mae)} (${sg(sb.bias)})</td><td><strong>${f2(sa.mae)} (${sg(sa.bias)})</strong></td></tr>`;
      }
      const fsd = (d.by_family && d.by_family.fsd) || d.fsd_breakdown || {};
      const fsdAfter = fsd.after_test || fsd.test_after;
      const fsdBefore = fsd.before_test || fsd.test_before;
      if (fsdAfter && fsdBefore) {
        const fb = fsdBefore.overall,
          fa = fsdAfter.overall;
        rowsHtml += `<tr><th scope="row" rowspan="3">FSD ONLY (FSD_*)</th><td rowspan="3">${fa.n}</td><td>Exact Acc</td><td>${pct(fb.exact_accuracy)}</td><td><strong>${pct(fa.exact_accuracy)} [${pct(fa.exact_ci_95[0])}, ${pct(fa.exact_ci_95[1])}]</strong></td></tr>
          <tr><td>Within ±1</td><td>${pct(fb.within_one)}</td><td><strong>${pct(fa.within_one)}</strong></td></tr>
          <tr><td>MAE / Bias</td><td>${f2(fb.mae)} (${sg(fb.bias)})</td><td><strong>${f2(fa.mae)} (${sg(fa.bias)})</strong></td></tr>`;
      }

      const bAll = bComp.overall;
      if (bAll && o) {
        rowsHtml += `<tr><th scope="row" rowspan="4">OVERALL (All Test)</th><td rowspan="4">${o.n}</td><td>Exact Acc</td><td>${pct(bAll.exact_accuracy)}</td><td><strong>${pct(o.exact_accuracy)} [${pct(o.exact_ci_95[0])}, ${pct(o.exact_ci_95[1])}]</strong></td></tr>
          <tr><td>Within ±1</td><td>${pct(bAll.within_one)}</td><td><strong>${pct(o.within_one)}</strong></td></tr>
          <tr><td>MAE</td><td>${f2(bAll.mae)}</td><td><strong>${f2(o.mae)}</strong></td></tr>
          <tr><td>Mean Bias</td><td>${sg(bAll.bias)}</td><td><strong>${sg(o.bias)}</strong></td></tr>`;
      }
      cmpTbl.innerHTML =
        '<thead><tr><th scope="col">Slice / True K</th><th scope="col">n</th><th scope="col">Metric</th><th scope="col">Before (FSD-only)</th><th scope="col">After (Speech + FSD) + 95% CI</th></tr></thead><tbody>' +
        rowsHtml +
        "</tbody>";
    }

    // Render Confusion Matrix Heatmap Table
    const cmTbl = $("#evCmTable");
    if (cmTbl && d.confusion_matrix) {
      const cm = d.confusion_matrix;
      const thCols = cm.classes
        .map((c) => `<th scope="col">Pred ${esc(c)}</th>`)
        .join("");
      const rowsHtml = cm.rows
        .map((r) => {
          const denom = Math.max(1, r.n || 1);
          const counts = cm.classes
            .map((c) => {
              const v = r.counts[c] || 0;
              const ratio = Math.min(1, v / denom);
              const isDiag = String(c) === String(r.true_k);
              const bg = v
                ? isDiag
                  ? `rgba(18, 181, 166, ${(0.14 + ratio * 0.52).toFixed(2)})`
                  : `rgba(255, 90, 31, ${(0.08 + ratio * 0.42).toFixed(2)})`
                : "transparent";
              return `<td style="background:${bg}">${v}</td>`;
            })
            .join("");
          return `<tr><th scope="row">True K = ${r.true_k}</th><td>${r.n}</td>${counts}<td><strong>${r.exact_accuracy_pct.toFixed(1)}%</strong></td></tr>`;
        })
        .join("");
      cmTbl.innerHTML = `<thead><tr><th scope="col">True Count</th><th scope="col">n</th>${thCols}<th scope="col">Exact %</th></tr></thead><tbody>${rowsHtml}</tbody>`;
    }

    // Render Tagging & Separation panels from approved public_summary
    const pub = d.public_summary || {};
    const tag = pub.tagging || {};
    const tagMetrics = $("#evTagMetrics");
    const tagTbl = $("#evTagTable");
    if (tagMetrics && tag.group_test_macro_map != null) {
      tagMetrics.innerHTML =
        metric("Group mAP (test)", tag.group_test_macro_map.toFixed(3), "5 sound groups") +
        metric("Fine-32 mAP (test)", tag.fine_32_test_macro_map.toFixed(3), `F1 ${tag.fine_32_test_macro_f1.toFixed(3)}`) +
        metric("AST + CLAP AP", tag.ast_clap_macro_ap.toFixed(3), `${tag.open_vocab_classes_evaluated} classes / ${tag.open_vocab_held_out_clips} clips`);
    }
    if (tagTbl && tag.per_class_held_out) {
      const entries = Object.entries(tag.per_class_held_out);
      tagTbl.innerHTML =
        '<thead><tr><th scope="col">Sound Class</th><th scope="col">AST+CLAP AP</th><th scope="col">Combined AP</th><th scope="col">Combined F1</th></tr></thead><tbody>' +
        entries
          .map(
            ([cls, m]) =>
              `<tr><th scope="row">${esc(cls)}</th><td><strong>${Number(m.ast_clap_ap).toFixed(3)}</strong></td><td>${Number(m.combined_ap).toFixed(3)}</td><td>${Number(m.combined_f1).toFixed(3)}</td></tr>`,
          )
          .join("") +
        "</tbody>";
    }

    const sep = pub.separation || {};
    const gates = pub.quality_gates || {};
    const sepMetrics = $("#evSepMetrics");
    const sepGates = $("#evSepGates");
    const sepTbl = $("#evSepTable");
    if (sepMetrics && sep.overall_mean_sisdri_db != null) {
      sepMetrics.innerHTML =
        metric("Mean SI-SDRi", `${sg(sep.overall_mean_sisdri_db)} dB`, `${sep.total_stems_evaluated} stems`) +
        metric("Self-consistency", pct(sep.self_consistency_pass_rate || 0), "re-tag gate") +
        metric("Energy pass", pct(sep.energy_conservation_pass_rate || 0), "with residual");
    }
    if (sepGates && gates.max_correlation_threshold != null) {
      sepGates.textContent = `SepFormer 2-speaker speech + ECAPA-TDNN verification + 3 quality gates: pairwise corr < ${gates.max_correlation_threshold}, speaker cosine < ${gates.max_speaker_cosine_threshold}, energy L2 tolerance ±${Math.round(gates.energy_relative_l2_tolerance * 100)}%.`;
    }
    if (sepTbl && sep.by_k) {
      const kRows = Object.entries(sep.by_k)
        .map(
          ([k, v]) =>
            `<tr><th scope="row">${esc(k)}</th><td>${v.n_stems} stems</td><td><strong>${sg(v.mean_sisdri_db)} dB</strong></td></tr>`,
        )
        .join("");
      const clsRows = Object.entries(sep.per_class_sisdri || {})
        .slice(0, 8)
        .map(
          ([cls, v]) =>
            `<tr><th scope="row">${esc(cls)}</th><td>${v.n_stems} stems</td><td>${sg(v.mean_sisdri_db)} dB (med ${sg(v.median_sisdri_db)} dB)</td></tr>`,
        )
        .join("");
      sepTbl.innerHTML =
        '<thead><tr><th scope="col">Slice / Class</th><th scope="col">Stems</th><th scope="col">Mean SI-SDRi</th></tr></thead><tbody>' +
        kRows +
        clsRows +
        "</tbody>";
    }

    // Render Validation Reliability & Uncertainty Composition Table
    const valTbl = $("#evValTable");
    if (valTbl && d.validation_reliability) {
      const vr = d.validation_reliability;
      const vPerK = vr.per_count || [];
      const vbPerK = vr.before_per_count || [];
      const valRows = [2, 3, 4, 6, 9]
        .map((k) => {
          const a = vPerK.find((x) => x.true_k === k);
          const b = vbPerK.find((x) => x.true_k === k);
          if (!a) return "";
          const unrel =
            a.unreliable || a.n < 30 ? " <small>(n&lt;30)</small>" : "";
          const bEx = b ? pct(b.exact_accuracy) : "—";
          const aEx = pct(a.exact_accuracy);
          const aW1 = pct(a.within_one);
          const mcS =
            typeof a.mean_mc_std === "number"
              ? "±" + a.mean_mc_std.toFixed(3)
              : "—";
          const smS =
            typeof a.mean_softmax_std === "number"
              ? "±" + a.mean_softmax_std.toFixed(3)
              : "—";
          const totS =
            typeof a.mean_total_std === "number"
              ? "±" + a.mean_total_std.toFixed(3)
              : "—";
          return `<tr><th scope="row">Val K = ${k}${unrel}</th><td>${a.n}</td><td>${bEx}</td><td><strong>${aEx}</strong></td><td>${aW1}</td><td>${f2(a.mae)}</td><td><code>${mcS}</code></td><td><code>${smS}</code></td><td><code>${totS}</code></td></tr>`;
        })
        .join("");
      valTbl.innerHTML = `<thead><tr><th scope="col">Validation Slice</th><th scope="col">n</th><th scope="col">Before Exact</th><th scope="col">Calibrated Exact</th><th scope="col">Within ±1</th><th scope="col">MAE</th><th scope="col">MC-Dropout Std</th><th scope="col">Softmax Spread</th><th scope="col">Combined Std</th></tr></thead><tbody>${valRows}</tbody>`;
    }

    // Render Real-Recording Benchmark Table (data/real_eval/)
    const realTbl = $("#evRealTable");
    if (realTbl) {
      const recs = d.real_eval || d.real_recording_eval || [];
      if (recs.length === 0) {
        realTbl.innerHTML =
          "<tbody><tr><td>No real_eval recordings found in <code>data/real_eval/manifest.json</code>.</td></tr></tbody>";
      } else {
        const rRows = recs
          .map((r) => {
            const bRaw = r.before
              ? (r.before.raw_value ?? r.before.raw ?? 0)
              : 0;
            const bCal = r.before
              ? (r.before.calibrated_value ?? r.before.cal ?? 0)
              : 0;
            const aRaw = r.after ? (r.after.raw_value ?? r.after.raw ?? 0) : 0;
            const aCal = r.after
              ? (r.after.calibrated_value ?? r.after.cal ?? 0)
              : 0;
            const bCnt = r.before
              ? `${r.before.rounded_count} (raw ${f2(bRaw)}, cal ${f2(bCal)})`
              : "—";
            const aCnt = r.after
              ? `<strong>${r.after.rounded_count}</strong> (raw ${f2(aRaw)}, cal ${f2(aCal)})`
              : "—";
            const bTag = r.before
              ? (
                  r.before.top_fine_tags ||
                  r.before.tags_above_threshold ||
                  []
                ).join(", ") || "none"
              : "—";
            const aGrp = r.after
              ? (r.after.active_groups || []).join(", ") || "none"
              : "—";
            const sc = r.speaker_clustering || {};
            const spkN =
              sc.distinct_speakers ?? sc.distinct_speakers_found ?? "—";
            const spk = `${spkN} (${(sc.labels || []).join(", ")})`;
            const fname = r.filename || r.file || "recording.wav";
            return `<tr>
            <th scope="row"><code>${esc(fname)}</code><br><small>${esc(r.description || "")}</small></th>
            <td>K = ${r.true_count}<br><small>${esc((r.true_groups || []).join(", "))}</small></td>
            <td>Count: ${bCnt}<br><small>Tags: ${esc(bTag)}</small></td>
            <td>Count: ${aCnt}<br><small>Groups: <strong>${esc(aGrp)}</strong></small></td>
            <td>${esc(spk)}</td>
          </tr>`;
          })
          .join("");
        realTbl.innerHTML = `<thead><tr><th scope="col">Recording</th><th scope="col">True</th><th scope="col">Before Retraining</th><th scope="col">After Retraining</th><th scope="col">Distinct Voices (VAD)</th></tr></thead><tbody>${rRows}</tbody>`;
      }
    }

    $("#evMeta").textContent =
      `Checkpoint ${d.checkpoint.path.split(/[\\/]/).pop()} (sha256 ${String(d.checkpoint.sha256).slice(0, 12)}), epoch ${d.checkpoint.epoch}. Uncertainty: ${d.uncertainty.method}, ${d.uncertainty.passes} passes. Generated ${d.generated_at}.`;
    draw();
  }

  function draw() {
    if (!data) return;
    const c = $("#evScatter"),
      r = c.getBoundingClientRect(),
      dpr = Math.min(devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(r.width * dpr)),
      h = Math.max(1, Math.round(r.height * dpr));
    c.width = w;
    c.height = h;
    const g = c.getContext("2d"),
      cs = getComputedStyle(document.documentElement);
    const ink = cs.getPropertyValue("--ink").trim() || "#111",
      mute = cs.getPropertyValue("--mute").trim() || "#777",
      line = cs.getPropertyValue("--line2").trim() || "#ccc";
    const rows = data.samples,
      max = Math.max(5, ...rows.map((s) => Math.max(s.true, s.pred))) + 1;
    const pad = 38 * dpr,
      X = (v) => pad + (v / max) * (w - pad - 10 * dpr),
      Y = (v) => h - pad - (v / max) * (h - pad - 10 * dpr);
    g.font = `${11 * dpr}px sans-serif`;
    g.fillStyle = mute;
    g.strokeStyle = line;
    g.lineWidth = dpr;
    const step = max > 20 ? 5 : 2;
    for (let v = 0; v <= max; v += step) {
      g.beginPath();
      g.moveTo(X(v), Y(0));
      g.lineTo(X(v), Y(max));
      g.moveTo(X(0), Y(v));
      g.lineTo(X(max), Y(v));
      g.globalAlpha = 0.35;
      g.stroke();
      g.globalAlpha = 1;
      g.fillText(v, X(v) - 4 * dpr, h - pad + 14 * dpr);
      g.fillText(v, pad - 22 * dpr, Y(v) + 4 * dpr);
    }
    g.fillText("true count", w / 2 - 20 * dpr, h - 4 * dpr);
    g.save();
    g.translate(10 * dpr, h / 2 + 30 * dpr);
    g.rotate(-Math.PI / 2);
    g.fillText("predicted count", 0, 0);
    g.restore();
    g.strokeStyle = ink;
    g.setLineDash([6 * dpr, 5 * dpr]);
    g.beginPath();
    g.moveTo(X(0), Y(0));
    g.lineTo(X(max), Y(max));
    g.stroke();
    g.setLineDash([]);
    const jitter = (i) => (((i * 2654435761) % 1000) / 1000 - 0.5) * 0.5;
    g.fillStyle = "rgba(139,92,246,.6)";
    rows.forEach((s, i) => {
      g.beginPath();
      g.arc(X(s.true + jitter(i)), Y(s.pred), 3.2 * dpr, 0, 7);
      g.fill();
    });
    g.strokeStyle = "#FF5A1F";
    g.lineWidth = 2.2 * dpr;
    g.beginPath();
    data.per_count.forEach((p, i) => {
      const x = X(+p.label),
        y = Y(p.mean_pred);
      i ? g.lineTo(x, y) : g.moveTo(x, y);
    });
    g.stroke();
  }
  new ResizeObserver(draw).observe($("#evScatter"));
  addEventListener("sanada:themechange", draw);
})();
