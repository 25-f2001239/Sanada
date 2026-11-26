/* EchoCount — Admin dashboard */
(() => {
  const { $, esc, toast } = window.EC;
  const csrf = () => $('meta[name="csrf-token"]')?.content || "";
  let allUsers = [],
    meId = null;

  const when = (ts) =>
    ts
      ? new Date(ts * 1000).toLocaleString([], {
          dateStyle: "medium",
          timeStyle: "short",
        })
      : "—";

  const fmtUptime = (s) => {
    const h = Math.floor(s / 3600),
      m = Math.floor((s % 3600) / 60);
    return h ? `${h}h ${m}m` : `${m}m ${s % 60}s`;
  };

  async function api(url, opts = {}) {
    const headers = { ...(opts.headers || {}) };
    if (opts.method && opts.method !== "GET") {
      headers["X-CSRF-Token"] = csrf();
      if (opts.body && !headers["Content-Type"])
        headers["Content-Type"] = "application/json";
    }
    const r = await fetch(url, { credentials: "same-origin", ...opts, headers });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.error || `Request failed (${r.status})`);
    return d;
  }

  function metricGrid(el, rows) {
    el.innerHTML = rows
      .map(
        ([label, val, sub]) =>
          `<div class="metric"><small>${esc(label)}</small><b>${esc(val)}</b>${sub ? `<span class="fine">${esc(sub)}</span>` : ""}</div>`,
      )
      .join("");
  }

  function renderBars(el, items, labelKey, valKey, emptyMsg) {
    if (!items || !items.length) {
      el.innerHTML = `<p class="fine">${esc(emptyMsg)}</p>`;
      return;
    }
    const max = Math.max(1, ...items.map((x) => Number(x[valKey]) || 0));
    el.innerHTML = items
      .map((x) => {
        const v = Number(x[valKey]) || 0;
        const pct = Math.max(4, Math.round((v / max) * 100));
        return `<div class="adm-bar-row"><span class="adm-bar-lbl">${esc(x[labelKey])}</span><div class="adm-bar-track"><i style="width:${pct}%"></i></div><b>${v}</b></div>`;
      })
      .join("");
  }

  async function loadOverview() {
    try {
      const d = await api("/api/admin/overview");
      const u = d.users || {},
        a = d.analyses || {},
        s = d.system || {};
      metricGrid($("#admUserStats"), [
        ["Total users", u.total ?? 0, `${u.verified ?? 0} verified`],
        ["Admins", u.admins ?? 0, `${u.disabled ?? 0} disabled`],
        ["Active (24h)", u.active_24h ?? 0, `${u.active_7d ?? 0} in 7d`],
        ["Online sessions", u.online_sessions ?? 0, `${u.new_7d ?? 0} new in 7d`],
      ]);
      const gtAcc =
        a.gt_exact_accuracy != null
          ? `${(a.gt_exact_accuracy * 100).toFixed(1)}% exact`
          : "No GT matches yet";
      metricGrid($("#admUsageStats"), [
        ["Total analyses", a.total ?? 0, `${a.last_24h ?? 0} in 24h · ${a.last_7d ?? 0} in 7d`],
        ["Avg predicted K", a.avg_predicted_k != null ? a.avg_predicted_k : "—"],
        ["Avg latency", a.avg_latency_ms != null ? `${a.avg_latency_ms} ms` : "—"],
        ["GT matched", a.with_ground_truth ?? 0, gtAcc],
      ]);
      renderBars($("#admDaily"), a.daily_14d, "day", "analyses", "No analyses recorded in the last 14 days.");
      renderBars($("#admSignups"), d.signups_14d, "day", "users", "No new sign-ups in the last 14 days.");
      renderBars(
        $("#admKHist"),
        (a.predicted_k_histogram || []).map((r) => ({ k: `K=${r.k}`, n: r.n })),
        "k",
        "n",
        "No predictions recorded yet.",
      );

      const provs = Object.entries(d.providers || {})
        .map(([k, v]) => `<span class="adm-pill">${esc(k)}: <b>${v}</b></span>`)
        .join("");
      const top = (a.top_users || [])
        .map((t) => `<tr><td>${esc(t.email)}</td><td><b>${t.analyses}</b></td></tr>`)
        .join("");
      $("#admProviders").innerHTML = `
        <div class="adm-pills">${provs || '<span class="fine">No users yet.</span>'}</div>
        ${top ? `<table class="kv adm-mini"><thead><tr><th>Top users</th><th>Analyses</th></tr></thead><tbody>${top}</tbody></table>` : '<p class="fine" style="margin-top:10px">No analyses yet.</p>'}
      `;

      renderSystem(s);
    } catch (e) {
      toast(e.message, "err");
    }
  }

  function renderSystem(s) {
    const ck = Object.entries(s.checkpoints || {})
      .map(([name, ok]) => `<span class="adm-pill ${ok ? "ok" : "warn"}">${esc(name)}: ${ok ? "present" : "missing"}</span>`)
      .join(" ");
    const oa = Object.entries(s.oauth_providers || {})
      .map(([name, ok]) => `<span class="adm-pill ${ok ? "ok" : ""}">${esc(name)}: ${ok ? "on" : "off"}</span>`)
      .join(" ");
    const rows = [
      ["Uptime", fmtUptime(s.uptime_s || 0)],
      ["Dev mode", s.dev_mode ? "Enabled (ECHOCOUNT_DEV=1)" : "Off (production)"],
      ["Counter model", s.model_trained ? "Loaded" : "Not trained / missing"],
      ["Checkpoints", ck || "—"],
      ["Evaluation results.json", s.evaluation_results ? "Present" : "Missing"],
      [
        "Ground-truth index",
        s.ground_truth_index_files != null ? `${s.ground_truth_index_files.toLocaleString()} files indexed` : "Not built",
      ],
      ["Dataset root", `<code>${esc(s.dataset_root || "")}</code> (${s.dataset_root_found ? "found" : "not found"})`],
      ["Active jobs", `${s.active_jobs ?? 0} in memory (${s.separations_running ?? 0} separating) · ${s.job_storage_mb ?? 0} MB on disk`],
      ["SQLite database", `${s.db_size_mb ?? 0} MB`],
      ["OAuth providers", oa || "—"],
      ["SMTP email", s.smtp_configured ? "Configured" : "Not configured (local verification links)"],
      [
        "Limits",
        s.limits
          ? `${s.limits.max_upload_mb} MB max upload · ${s.limits.max_duration_s}s max duration · ${s.limits.job_ttl_min} min job TTL`
          : "—",
      ],
    ];
    $("#admSystem").innerHTML = `<tbody>${rows
      .map(([k, v]) => `<tr><th>${esc(k)}</th><td>${v}</td></tr>`)
      .join("")}</tbody>`;
  }

  function renderUsers() {
    const q = ($("#admUserQ").value || "").trim().toLowerCase();
    const f = $("#admUserFilter").value;
    const rows = allUsers.filter((u) => {
      if (q && !u.email.toLowerCase().includes(q)) return false;
      if (f === "admin" && u.role !== "admin") return false;
      if (f === "user" && u.role !== "user") return false;
      if (f === "disabled" && !u.disabled) return false;
      if (f === "online" && !u.active_sessions) return false;
      return true;
    });
    const tbl = $("#admUsers");
    if (!rows.length) {
      tbl.innerHTML = `<tbody><tr><td colspan="7" class="fine">No matching users.</td></tr></tbody>`;
      return;
    }
    tbl.innerHTML = `
      <thead><tr>
        <th>Email</th><th>Role</th><th>Status</th><th>Sign-in</th><th>Last login</th><th>Analyses</th><th>Actions</th>
      </tr></thead>
      <tbody>${rows
        .map((u) => {
          const isMe = u.id === meId;
          const status = u.disabled
            ? '<span class="adm-pill warn">Disabled</span>'
            : u.email_verified
              ? '<span class="adm-pill ok">Verified</span>'
              : '<span class="adm-pill">Unverified</span>';
          const rolePill =
            u.role === "admin"
              ? `<span class="adm-pill admin">Admin${u.protected ? " · cfg" : ""}</span>`
              : `<span class="adm-pill">User</span>`;
          const sess = u.active_sessions
            ? `<div class="fine">${u.active_sessions} active session${u.active_sessions === 1 ? "" : "s"}</div>`
            : "";
          const btns = [];
          if (u.role === "admin") {
            if (!isMe && !u.protected)
              btns.push(`<button class="btn sm" data-act="demote" data-id="${u.id}">Demote</button>`);
          } else {
            btns.push(`<button class="btn sm primary" data-act="promote" data-id="${u.id}">Make admin</button>`);
          }
          if (!isMe && !u.protected) {
            btns.push(
              u.disabled
                ? `<button class="btn sm" data-act="enable" data-id="${u.id}">Enable</button>`
                : `<button class="btn sm" data-act="disable" data-id="${u.id}">Disable</button>`,
            );
          }
          if (u.active_sessions > (isMe ? 1 : 0)) {
            btns.push(`<button class="btn sm" data-act="revoke" data-id="${u.id}">Sign out</button>`);
          }
          if (!isMe && !u.protected) {
            btns.push(`<button class="btn sm danger" data-act="delete" data-id="${u.id}">Delete</button>`);
          }
          return `<tr>
            <td><strong>${esc(u.email)}</strong>${isMe ? ' <span class="fine">(you)</span>' : ""}${sess}</td>
            <td>${rolePill}</td>
            <td>${status}</td>
            <td>${esc((u.sign_in_methods || []).join(", ") || "—")}</td>
            <td>${esc(when(u.last_login_at))}<div class="fine">Joined ${esc(when(u.created_at))}</div></td>
            <td>${u.analyses ?? 0}</td>
            <td><div class="adm-actions">${btns.join("") || '<span class="fine">—</span>'}</div></td>
          </tr>`;
        })
        .join("")}</tbody>`;
  }

  function renderInvites(invites) {
    const box = $("#admInvites");
    if (!invites || !invites.length) {
      box.innerHTML = "";
      return;
    }
    box.innerHTML = `
      <div class="fine" style="margin-top:10px">Pending admin invites (granted on first sign-in):</div>
      <div class="adm-pills" style="margin-top:6px">${invites
        .map(
          (i) =>
            `<span class="adm-pill">${esc(i.email)} <button type="button" class="adm-x" data-cancel-invite="${esc(i.email)}" aria-label="Cancel invite for ${esc(i.email)}">×</button></span>`,
        )
        .join("")}</div>`;
  }

  async function loadUsers() {
    try {
      const d = await api("/api/admin/users");
      allUsers = d.users || [];
      meId = d.me;
      renderUsers();
      renderInvites(d.invites || []);
    } catch (e) {
      toast(e.message, "err");
    }
  }

  async function loadAnalyses() {
    try {
      const d = await api("/api/admin/analyses?limit=50");
      const rows = d.analyses || [];
      const tbl = $("#admAnalyses");
      if (!rows.length) {
        tbl.innerHTML = `<tbody><tr><td class="fine">No analyses recorded yet.</td></tr></tbody>`;
        return;
      }
      tbl.innerHTML = `
        <thead><tr>
          <th>Time</th><th>User</th><th>File</th><th>Predicted K</th><th>Ground truth</th><th>Latency</th>
        </tr></thead>
        <tbody>${rows
          .map((r) => {
            const gt =
              r.gt_true_k != null
                ? `K=${r.gt_true_k} <span class="fine">(${esc(r.gt_dataset || "")})</span>`
                : '<span class="fine">—</span>';
            return `<tr>
              <td>${esc(when(r.created_at))}</td>
              <td>${esc(r.email)}</td>
              <td><code>${esc(r.filename)}</code>${r.duration_s != null ? ` <span class="fine">${r.duration_s}s</span>` : ""}</td>
              <td><b>${r.predicted_k ?? "—"}</b>${r.confidence != null ? ` <span class="fine">${r.confidence}%</span>` : ""}</td>
              <td>${gt}</td>
              <td>${r.total_ms != null ? `${Math.round(r.total_ms)} ms` : "—"}</td>
            </tr>`;
          })
          .join("")}</tbody>`;
    } catch (e) {
      toast(e.message, "err");
    }
  }

  async function loadAudit() {
    try {
      const d = await api("/api/admin/audit?limit=100");
      const rows = d.audit || [];
      const tbl = $("#admAudit");
      if (!rows.length) {
        tbl.innerHTML = `<tbody><tr><td class="fine">No admin actions recorded yet.</td></tr></tbody>`;
        return;
      }
      tbl.innerHTML = `
        <thead><tr><th>Time</th><th>Actor</th><th>Action</th><th>Target</th><th>Detail</th></tr></thead>
        <tbody>${rows
          .map(
            (r) => `<tr>
              <td>${esc(when(r.created_at))}</td>
              <td>${esc(r.actor_email)}</td>
              <td><code>${esc(r.action)}</code></td>
              <td>${esc(r.target_email || "—")}</td>
              <td>${esc(r.detail || "—")}</td>
            </tr>`,
          )
          .join("")}</tbody>`;
    } catch (e) {
      toast(e.message, "err");
    }
  }

  $("#admUserQ")?.addEventListener("input", renderUsers);
  $("#admUserFilter")?.addEventListener("change", renderUsers);

  $("#admInviteForm")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const inp = $("#admInviteEmail");
    const email = inp.value.trim();
    if (!email) return;
    try {
      const r = await api("/api/admin/invites", {
        method: "POST",
        body: JSON.stringify({ email }),
      });
      inp.value = "";
      toast(r.message || "Updated.");
      loadUsers();
      loadOverview();
      loadAudit();
    } catch (err) {
      toast(err.message, "err");
    }
  });

  $("#admInvites")?.addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-cancel-invite]");
    if (!btn) return;
    const email = btn.dataset.cancelInvite;
    try {
      const r = await api(`/api/admin/invites/${encodeURIComponent(email)}`, {
        method: "DELETE",
      });
      toast(r.message || "Invite cancelled.");
      loadUsers();
      loadAudit();
    } catch (err) {
      toast(err.message, "err");
    }
  });

  $("#admUsers")?.addEventListener("click", async (e) => {
    const btn = e.target.closest("button[data-act]");
    if (!btn) return;
    const act = btn.dataset.act,
      id = btn.dataset.id;
    const user = allUsers.find((u) => String(u.id) === String(id));
    const email = user ? user.email : `#${id}`;
    if (act === "delete" && !confirm(`Permanently delete ${email} and their sessions?`)) return;
    if (act === "demote" && !confirm(`Remove admin role from ${email}?`)) return;
    if (act === "disable" && !confirm(`Disable ${email} and sign them out immediately?`)) return;
    try {
      let r;
      if (act === "promote")
        r = await api(`/api/admin/users/${id}/role`, { method: "POST", body: JSON.stringify({ role: "admin" }) });
      else if (act === "demote")
        r = await api(`/api/admin/users/${id}/role`, { method: "POST", body: JSON.stringify({ role: "user" }) });
      else if (act === "disable")
        r = await api(`/api/admin/users/${id}/disabled`, { method: "POST", body: JSON.stringify({ disabled: true }) });
      else if (act === "enable")
        r = await api(`/api/admin/users/${id}/disabled`, { method: "POST", body: JSON.stringify({ disabled: false }) });
      else if (act === "revoke")
        r = await api(`/api/admin/users/${id}/revoke-sessions`, { method: "POST" });
      else if (act === "delete")
        r = await api(`/api/admin/users/${id}`, { method: "DELETE" });
      toast((r && r.message) || "Done.");
      loadUsers();
      loadOverview();
      loadAudit();
    } catch (err) {
      toast(err.message, "err");
    }
  });

  // Load tab data on demand + initial overview
  document.querySelectorAll('.admin-tabs [role="tab"]').forEach((btn) => {
    btn.addEventListener("click", () => {
      const t = btn.dataset.tab;
      if (t === "overview" || t === "system") loadOverview();
      if (t === "users") loadUsers();
      if (t === "activity") loadAnalyses();
      if (t === "audit") loadAudit();
    });
  });

  loadOverview();
  loadUsers();
})();
