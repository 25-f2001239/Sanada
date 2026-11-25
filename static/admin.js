/* EchoCount — Administrator Console: user governance, locked single admin, session revocation, audit log & telemetry */
(() => {
  const { $, esc, toast } = window.EC;
  const wrap = $(".admin-wrap");
  const selfId = Number(wrap?.dataset.selfId || 0);
  const csrf = document.querySelector('meta[name="csrf-token"]')?.content || "";
  const feedback = $("#adminFeedback");
  const searchIn = $("#adminUserSearch");
  const delDialog = $("#adminDeleteDialog");
  const delDesc = $("#adminDelDesc");
  let allUsers = [];
  let pendingDeleteUid = null;

  const showMsg = (msg, kind = "info") => {
    if (!feedback) return;
    feedback.className = `notice ${kind}`;
    feedback.textContent = msg;
    feedback.classList.remove("hidden");
  };

  const metric = (l, v, s = "") =>
    `<div class="metric"><small>${esc(l)}</small><b>${esc(v)}</b>${s ? `<small>${esc(s)}</small>` : ""}</div>`;

  const reqJson = async (url, method = "POST", body = {}) => {
    const res = await fetch(url, {
      method,
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf },
      body: method === "GET" ? undefined : JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || "Action failed.");
    return data;
  };

  const renderUsers = () => {
    const body = $("#adminUsersBody");
    if (!body) return;
    const q = (searchIn?.value || "").trim().toLowerCase();
    const list = q
      ? allUsers.filter((u) => u.email.toLowerCase().includes(q) || u.role.toLowerCase().includes(q))
      : allUsers;
    if (!list.length) {
      body.innerHTML = '<tr><td colspan="7">No matching user accounts.</td></tr>';
      return;
    }
    body.innerHTML = list
      .map((u) => {
        const isLockedAdmin = u.role === "admin" || u.id === selfId;
        const label = isLockedAdmin
          ? `Admin <span class="fine">(${esc(u.email)})</span>`
          : esc(u.email);
        const actionsHtml = isLockedAdmin
          ? '<span class="mini-chip">Locked (.env)</span>'
          : `<button type="button" class="btn sm" data-revoke="${u.id}" ${u.active_sessions === 0 ? "disabled" : ""}>Revoke sessions</button>
             <button type="button" class="btn sm ${u.is_active ? "danger" : ""}" data-toggle-active="${u.id}" data-next-active="${u.is_active ? "0" : "1"}">
               ${u.is_active ? "Disable" : "Enable"}
             </button>
             <button type="button" class="btn sm danger" data-delete-user="${u.id}" data-delete-label="${esc(u.email)}">Delete</button>`;
        return `<tr data-uid="${u.id}">
          <th scope="row">${label}${u.id === selfId ? ' <span class="mini-chip">You</span>' : ""}</th>
          <td><span class="role-badge role-${esc(u.role)}">${esc(u.role)}</span></td>
          <td><span class="status-dot ${u.email_verified ? "ok" : "warn"}">${u.email_verified ? "Verified" : "Pending"}</span></td>
          <td><span class="status-dot ${u.is_active ? "ok" : "err"}">${u.is_active ? "Active" : "Disabled"}</span></td>
          <td>${u.active_sessions}</td>
          <td>${u.job_count}</td>
          <td class="admin-actions-cell">${actionsHtml}</td>
        </tr>`;
      })
      .join("");
  };

  const renderAudit = (rows) => {
    const body = $("#adminAuditBody");
    if (!body) return;
    if (!rows || !rows.length) {
      body.innerHTML = '<tr><td colspan="4">No admin account actions recorded yet.</td></tr>';
      return;
    }
    body.innerHTML = rows
      .map((r) => {
        const ts = r.created_at ? new Date(r.created_at * 1000).toISOString().replace("T", " ").slice(0, 19) : "—";
        return `<tr>
          <td><code>${esc(ts)}</code></td>
          <td>#${esc(r.admin_id)}</td>
          <td><code>${esc(r.action)}</code></td>
          <td>#${esc(r.target_id)}</td>
        </tr>`;
      })
      .join("");
  };

  const loadOverview = async () => {
    try {
      const res = await fetch("/api/admin/overview", { credentials: "same-origin" });
      if (!res.ok) throw new Error("Failed to load administrator overview.");
      const data = await res.json();
      allUsers = data.users || [];

      const c = data.counts || {};
      const s = data.system || {};
      $("#adminMetrics").innerHTML =
        metric("Total Accounts", c.total_users ?? 0, `${c.admins ?? 0} admin · ${c.regular_users ?? 0} user`) +
        metric("Active Sessions", c.active_sessions ?? 0, "server-side HttpOnly") +
        metric("Jobs (In-Memory)", s.active_jobs ?? 0, `${s.running_separations ?? 0} running · ${s.failed_separations ?? 0} failed`) +
        metric("Temp Storage", `${s.temp_disk_mb ?? 0} MB`, `${s.temp_job_dirs ?? 0} dirs · TTL ${s.job_ttl_minutes ?? 30}m`);

      renderUsers();
      renderAudit(data.audit_log || []);

      const m = s.model || {};
      $("#adminSysBody").innerHTML = [
        ["Counter Checkpoint", m.trained ? `Ready (${m.model_name})` : "Not trained"],
        ["Open-Vocab Tagger", m.open_tagger?.available ? "Ready (AST + CLAP)" : "Domain CNN only"],
        ["Speech Separator", m.speech_separator?.available ? `SepFormer (up to ${m.speech_separator.max_supported_speakers} spk)` : "Unavailable"],
        ["General Separator", m.separator?.trained ? "Trained checkpoint loaded" : "Inactive"],
        ["Worker Pool", `${s.separation_workers ?? 2} background threads`],
        ["Completed Separations", String(s.completed_separations ?? 0)],
      ]
        .map(([k, v]) => `<tr><th scope="row">${esc(k)}</th><td>${esc(v)}</td></tr>`)
        .join("");

      const provs = s.oauth_providers || {};
      const googleOn = Boolean(provs.google);
      $("#adminOAuthGrid").innerHTML = `<div class="oauth-status-item">
        <strong>Google</strong>
        <span class="status-dot ${googleOn ? "ok" : "warn"}">${googleOn ? "Configured" : "Not configured"}</span>
      </div>`;
    } catch (err) {
      showMsg(err.message || "Could not load overview.", "warn");
    }
  };

  $("#adminUsersBody")?.addEventListener("click", async (e) => {
    const revBtn = e.target.closest("[data-revoke]");
    if (revBtn) {
      const uid = revBtn.dataset.revoke;
      try {
        const out = await reqJson(`/api/admin/users/${uid}/revoke`, "POST");
        toast(`Revoked ${out.revoked || 0} session(s).`);
        loadOverview();
      } catch (err) {
        showMsg(err.message, "warn");
      }
      return;
    }
    const togBtn = e.target.closest("[data-toggle-active]");
    if (togBtn) {
      const uid = togBtn.dataset.toggleActive;
      const nextActive = togBtn.dataset.nextActive === "1";
      try {
        await reqJson(`/api/admin/users/${uid}/status`, "POST", { is_active: nextActive });
        toast(nextActive ? "Account enabled." : "Account disabled & sessions revoked.");
        loadOverview();
      } catch (err) {
        showMsg(err.message, "warn");
      }
      return;
    }
    const delBtn = e.target.closest("[data-delete-user]");
    if (delBtn) {
      pendingDeleteUid = delBtn.dataset.deleteUser;
      const lbl = delBtn.dataset.deleteLabel || `user #${pendingDeleteUid}`;
      if (delDesc) {
        delDesc.textContent = `Permanently delete ${lbl} (#${pendingDeleteUid}), all sessions, history, and audio files on disk? This cannot be undone.`;
      }
      delDialog?.showModal();
    }
  });

  $("#adminDelCancelBtn")?.addEventListener("click", () => {
    pendingDeleteUid = null;
    delDialog?.close();
  });

  $("#adminDelConfirmBtn")?.addEventListener("click", async () => {
    if (!pendingDeleteUid) return;
    const uid = pendingDeleteUid;
    pendingDeleteUid = null;
    delDialog?.close();
    try {
      const out = await reqJson(`/api/admin/users/${uid}`, "DELETE");
      toast(`Deleted account #${uid} (${out.deleted_jobs || 0} job(s) removed).`);
      loadOverview();
    } catch (err) {
      showMsg(err.message, "warn");
    }
  });

  $("#adminCleanupBtn")?.addEventListener("click", async () => {
    try {
      const out = await reqJson("/api/admin/cleanup", "POST");
      toast(`Cleanup complete (${out.removed} expired dir(s) removed).`);
      loadOverview();
    } catch (err) {
      showMsg(err.message, "warn");
    }
  });

  $("#adminRefreshBtn")?.addEventListener("click", () => {
    toast("Refreshing telemetry…");
    loadOverview();
  });

  searchIn?.addEventListener("input", renderUsers);
  loadOverview();
})();
