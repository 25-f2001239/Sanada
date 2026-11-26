/* EchoCount — Per-user History page (loaded from /api/history for the signed-in account) */
(() => {
  const { $, esc, fmtTime, toast } = window.EC;
  const list = $("#hlist"),
    stats = $("#hstats"),
    q = $("#q");
  if (!list || list.dataset.authed !== "true") return;

  const csrf = () => $('meta[name="csrf-token"]')?.content || "";
  let all = [];

  const when = (ts) =>
    ts
      ? new Date(ts * 1000).toLocaleString([], {
          dateStyle: "medium",
          timeStyle: "short",
        })
      : "—";

  function render() {
    const term = (q?.value || "").trim().toLowerCase();
    const items = all.filter((h) =>
      (h.filename || "").toLowerCase().includes(term),
    );
    const freq = {};
    all.forEach((h) => {
      if (h.predicted_k != null) freq[h.predicted_k] = (freq[h.predicted_k] || 0) + 1;
    });
    const common = Object.entries(freq).sort((a, b) => b[1] - a[1])[0];
    const validK = all.filter((h) => Number.isFinite(Number(h.predicted_k)));
    const avg = validK.length
      ? validK.reduce((s, h) => s + Number(h.predicted_k), 0) / validK.length
      : 0;
    stats.innerHTML = [
      ["Analyses", all.length],
      ["Average count", validK.length ? avg.toFixed(1) : "—"],
      ["Most common count", common ? common[0] : "—"],
    ]
      .map(
        ([l, v]) => `<div class="metric"><small>${l}</small><b>${v}</b></div>`,
      )
      .join("");

    if (!all.length) {
      list.innerHTML = `<div class="panel state"><div class="orbit"><svg class="ic"><use href="#i-wave"/></svg></div><h3>Nothing here yet</h3><p>Results appear here after you analyze a recording.</p><a class="btn primary" href="/analyze">Analyze a recording</a></div>`;
      return;
    }
    if (!items.length) {
      list.innerHTML = `<div class="panel state"><h3>No match</h3><p>No saved result has “${esc(q.value)}” in its file name.</p></div>`;
      return;
    }
    list.innerHTML = items
      .map((h, i) => {
        const k = h.predicted_k ?? "—";
        const conf =
          h.confidence != null ? `${Math.round(Number(h.confidence))}% model confidence` : "";
        const gt =
          h.gt_true_k != null ? ` · GT: ${h.gt_true_k}` : "";
        const dur = h.duration_s != null ? fmtTime(h.duration_s) : "—";
        return `<article class="hrow" style="animation-delay:${Math.min(i, 10) * 40}ms">
          <div class="badge" aria-label="${esc(k)} sources">${esc(k)}</div>
          <div class="hinfo"><strong title="${esc(h.filename)}">${esc(h.filename)}</strong><small>${when(h.created_at)} · ${dur}${conf ? " · " + conf : ""}${gt}</small></div>
          <button class="icon-btn" data-del="${esc(h.id)}" aria-label="Delete ${esc(h.filename)}"><svg class="ic"><use href="#i-trash"/></svg></button>
        </article>`;
      })
      .join("");
  }

  async function load() {
    try {
      const r = await fetch("/api/history", { credentials: "same-origin" });
      if (r.status === 401) {
        location.href = "/login?next=/history";
        return;
      }
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || "Could not load history.");
      all = Array.isArray(d.items) ? d.items : [];
      render();
    } catch (e) {
      list.innerHTML = `<div class="panel state err"><h3>Could not load history</h3><p>${esc(e.message)}</p></div>`;
    }
  }

  list.addEventListener("click", async (e) => {
    const b = e.target.closest("[data-del]");
    if (!b) return;
    const id = b.dataset.del;
    try {
      const r = await fetch(`/api/history/${encodeURIComponent(id)}`, {
        method: "DELETE",
        credentials: "same-origin",
        headers: { "X-CSRF-Token": csrf() },
      });
      const d = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(d.error || "Could not delete item.");
      all = all.filter((x) => String(x.id) !== String(id));
      render();
      toast("Result deleted.");
    } catch (err) {
      toast(err.message, "err");
    }
  });

  q?.addEventListener("input", render);

  $("#clearAll")?.addEventListener("click", async () => {
    if (!all.length) return;
    if (!confirm("Clear all saved results from your history?")) return;
    try {
      const r = await fetch("/api/history", {
        method: "DELETE",
        credentials: "same-origin",
        headers: { "X-CSRF-Token": csrf() },
      });
      const d = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(d.error || "Could not clear history.");
      all = [];
      render();
      toast("History cleared.");
    } catch (err) {
      toast(err.message, "err");
    }
  });

  $("#exportBtn")?.addEventListener("click", () => {
    if (!all.length) return toast("There is nothing to export yet.");
    const cell = (v) => {
      let s = String(v ?? "");
      if (/^[=+\-@\t\r]/.test(s)) s = "'" + s;
      return `"${s.replace(/"/g, '""')}"`;
    };
    const rows = [
      ["file", "date", "estimated_sources", "model_confidence_pct", "duration_s", "ground_truth_k"],
    ].concat(
      all.map((h) => [
        h.filename,
        h.created_at ? new Date(h.created_at * 1000).toISOString() : "",
        h.predicted_k ?? "",
        h.confidence ?? "",
        h.duration_s ?? "",
        h.gt_true_k ?? "",
      ]),
    );
    const a = document.createElement("a");
    a.href = URL.createObjectURL(
      new Blob([rows.map((r) => r.map(cell).join(",")).join("\n")], {
        type: "text/csv",
      }),
    );
    a.download = "echocount-history.csv";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });

  load();
})();
