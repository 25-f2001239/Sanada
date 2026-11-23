/* EchoCount — Server-backed per-user History page (with Admin All-Users view) */
(() => {
  const { $, esc, fmtTime, toast, clearHistoryCache } = window.EC;
  const list = $("#hlist"),
    stats = $("#hstats"),
    q = $("#q"),
    clearBtn = $("#clearAll"),
    exportBtn = $("#exportBtn"),
    scopeSelfBtn = $("#scopeSelfBtn"),
    scopeAllBtn = $("#scopeAllBtn"),
    userFilter = $("#adminUserFilter"),
    scopeBanner = $("#adminScopeBanner");

  const csrf = () => $('meta[name="csrf-token"]')?.content || "";
  const when = (sec) =>
    new Date((Number(sec) || 0) * 1000).toLocaleString([], {
      dateStyle: "medium",
      timeStyle: "short",
    });

  let scope = "self"; // "self" | "all"
  let selectedUid = "";
  let cachedItems = [];
  let loaded = false;

  function handleUnauthorized() {
    clearHistoryCache?.();
    location.href = "/login?next=" + encodeURIComponent(location.pathname);
  }

  function syncScopeUI() {
    if (!scopeSelfBtn || !scopeAllBtn) return;
    const isAll = scope === "all";
    scopeSelfBtn.classList.toggle("primary", !isAll);
    scopeSelfBtn.setAttribute("aria-pressed", String(!isAll));
    scopeAllBtn.classList.toggle("primary", isAll);
    scopeAllBtn.setAttribute("aria-pressed", String(isAll));
    if (userFilter) userFilter.hidden = !isAll;
    if (scopeBanner) scopeBanner.hidden = !isAll;
    if (clearBtn) clearBtn.hidden = isAll;
  }

  function populateUserFilter(users) {
    if (!userFilter || !Array.isArray(users)) return;
    const prev = selectedUid;
    userFilter.innerHTML =
      `<option value="">All users (${users.length})</option>` +
      users
        .map(
          (u) =>
            `<option value="${esc(u.id)}">${esc(u.email)} (${esc(u.role)})</option>`,
        )
        .join("");
    userFilter.value = prev;
  }

  async function loadHistory() {
    syncScopeUI();
    list.innerHTML = `<div class="panel state"><p>Loading history…</p></div>`;
    let url = "/api/history";
    if (scope === "all") {
      url =
        "/api/admin/history" +
        (selectedUid ? `?user_id=${encodeURIComponent(selectedUid)}` : "");
    }
    try {
      const res = await fetch(url, {
        credentials: "same-origin",
        cache: "no-store",
      });
      if (res.status === 401) {
        handleUnauthorized();
        return;
      }
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        toast(data.error || "Could not load history.", "err");
        cachedItems = [];
        loaded = true;
        render();
        return;
      }
      cachedItems = Array.isArray(data.items) ? data.items : [];
      if (scope === "all" && Array.isArray(data.users)) {
        populateUserFilter(data.users);
      }
      loaded = true;
      render();
    } catch {
      toast("Network error loading history.", "err");
      list.innerHTML = `<div class="panel state"><h3>Could not load history</h3><p>Check your connection and refresh the page.</p></div>`;
    }
  }

  function render() {
    const all = cachedItems;
    const term = q.value.trim().toLowerCase();
    const items = all.filter((h) => {
      if (!term) return true;
      const fn = String(h.filename || "").toLowerCase();
      const em = String(h.user_email || "").toLowerCase();
      const sounds = Array.isArray(h.summary?.top_sounds)
        ? h.summary.top_sounds.join(" ").toLowerCase()
        : "";
      return fn.includes(term) || em.includes(term) || sounds.includes(term);
    });

    const counted = all.filter((h) =>
      Number.isFinite(Number(h.summary?.rounded_count)),
    );
    const freq = {};
    counted.forEach((h) => {
      const k = Number(h.summary.rounded_count);
      freq[k] = (freq[k] || 0) + 1;
    });
    const common = Object.entries(freq).sort((a, b) => b[1] - a[1])[0];
    const avg = counted.length
      ? counted.reduce((s, h) => s + Number(h.summary.rounded_count), 0) /
        counted.length
      : 0;

    stats.innerHTML = [
      ["Analyses", all.length],
      ["Average count", counted.length ? avg.toFixed(1) : "—"],
      ["Most common count", common ? common[0] : "—"],
    ]
      .map(
        ([l, v]) => `<div class="metric"><small>${l}</small><b>${v}</b></div>`,
      )
      .join("");

    if (!loaded) return;

    if (!all.length) {
      list.innerHTML =
        scope === "all"
          ? `<div class="panel state"><h3>No analyses found</h3><p>No analysis runs match this user filter yet.</p></div>`
          : `<div class="panel state"><div class="orbit"><svg class="ic"><use href="#i-wave"/></svg></div><h3>Nothing here yet</h3><p>Your private analysis runs will appear here after you analyze a recording.</p><a class="btn primary" href="/analyze">Analyze a recording</a></div>`;
      return;
    }

    if (!items.length) {
      list.innerHTML = `<div class="panel state"><h3>No match</h3><p>No saved result matches “${esc(q.value)}”.</p></div>`;
      return;
    }

    list.innerHTML = items
      .map((h, i) => {
        const s = h.summary || {};
        const count = Number.isFinite(Number(s.rounded_count))
          ? Number(s.rounded_count)
          : "—";
        const conf = Number.isFinite(Number(s.confidence_percent))
          ? `${Math.round(Number(s.confidence_percent))}% confidence`
          : h.status || "analyzed";
        const dur = Number.isFinite(Number(s.duration_seconds))
          ? fmtTime(Number(s.duration_seconds))
          : "—";
        const sounds = Array.isArray(s.top_sounds) && s.top_sounds.length
          ? ` · ${esc(s.top_sounds.slice(0, 3).join(", "))}`
          : "";
        const userTag =
          scope === "all" && h.user_email
            ? `<span class="mini-chip" style="margin-right:8px">${esc(h.user_email)}</span>`
            : "";
        return `<article class="hrow" style="animation-delay:${Math.min(i, 10) * 35}ms" data-job-id="${esc(h.job_id)}">
      <div class="badge" aria-label="${esc(count)} sources">${esc(count)}</div>
      <div class="hinfo">
        <strong title="${esc(h.filename)}">${userTag}${esc(h.filename)}</strong>
        <small>${when(h.created_at)} · ${dur} · ${esc(conf)}${sounds}</small>
      </div>
      <button class="icon-btn" data-del="${esc(h.job_id)}" aria-label="Delete ${esc(h.filename)}"><svg class="ic"><use href="#i-trash"/></svg></button>
    </article>`;
      })
      .join("");
  }

  async function apiDelete(url, fallbackErr) {
    const res = await fetch(url, {
      method: "DELETE",
      credentials: "same-origin",
      headers: { "X-CSRF-Token": csrf() },
    });
    if (res.status === 401) {
      handleUnauthorized();
      return false;
    }
    if (!res.ok) {
      const d = await res.json().catch(() => ({}));
      toast(d.error || fallbackErr, "err");
      return false;
    }
    return true;
  }

  list.addEventListener("click", async (e) => {
    const b = e.target.closest("[data-del]");
    if (!b) return;
    const jid = b.dataset.del;
    b.disabled = true;
    try {
      if (!(await apiDelete(`/api/history/${encodeURIComponent(jid)}`, "Could not delete result."))) {
        b.disabled = false;
        return;
      }
      cachedItems = cachedItems.filter((x) => x.job_id !== jid);
      render();
      toast("Result deleted.");
    } catch {
      toast("Network error deleting result.", "err");
      b.disabled = false;
    }
  });

  q.addEventListener("input", render);

  clearBtn?.addEventListener("click", async () => {
    if (!cachedItems.length || !confirm("Delete all of your saved analysis history?")) return;
    try {
      if (!(await apiDelete("/api/history", "Could not clear history."))) return;
      cachedItems = [];
      render();
      toast("Your history was cleared.");
    } catch {
      toast("Network error clearing history.", "err");
    }
  });

  exportBtn?.addEventListener("click", () => {
    if (!cachedItems.length) return toast("There is nothing to export yet.");
    const url =
      scope === "all"
        ? "/api/admin/history/export.csv" +
          (selectedUid ? `?user_id=${encodeURIComponent(selectedUid)}` : "")
        : "/api/history/export.csv";
    location.href = url;
  });

  scopeSelfBtn?.addEventListener("click", () => {
    if (scope === "self") return;
    scope = "self";
    selectedUid = "";
    loadHistory();
  });

  scopeAllBtn?.addEventListener("click", () => {
    if (scope === "all") return;
    scope = "all";
    loadHistory();
  });

  userFilter?.addEventListener("change", () => {
    selectedUid = userFilter.value || "";
    loadHistory();
  });

  loadHistory();
})();
