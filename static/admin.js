/* EchoCount — Administrator Console: user governance, role management, session revocation, system health */
(() => {
  const { $, esc, toast } = window.EC;
  const wrap = $(".admin-wrap");
  const selfId = Number(wrap?.dataset.selfId || 0);
  const csrf = document.querySelector('meta[name="csrf-token"]')?.content || "";
  const feedback = $("#adminFeedback");
  const searchIn = $("#adminUserSearch");
  let allUsers = [];

  const showMsg = (msg, kind = "info") => {
    if (!feedback) return;
    feedback.className = `notice ${kind}`;
    feedback.textContent = msg;
    feedback.classList.remove("hidden");
  };

  const metric = (l, v, s = "") =>
    `<div class="metric"><small>${esc(l)}</small><b>${esc(v)}</b>${s ? `<small>${esc(s)}</small>` : ""}</div>`;

  const postJson = async (url, body = {}) => {
    const res = await fetch(url, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf },
      body: JSON.stringify(body),
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
        const isSelf = u.id === selfId;
        const roleOpts = ["user", "developer", "admin"]
          .map((r) => `<option value="${r}"${u.role === r ? " selected" : ""}>${r}</option>`)
          .join("");
        return `<tr data-uid="${u.id}">
          <th scope="row">${esc(u.email)}${isSelf ? ' <span class="mini-chip">You</span>' : ""}</th>
          <td>
            <select class="admin-select" data-role-select="${u.id}" aria-label="Role for ${esc(u.email)}" ${isSelf ? "disabled" : ""}>
              ${roleOpts}
            </select>
          </td>
          <td><span class="status-dot ${u.email_verified ? "ok" : "warn"}">${u.email_verified ? "Verified" : "Pending"}</span></td>
          <td><span class="status-dot ${u.is_active ? "ok" : "err"}">${u.is_active ? "Active" : "Suspended"}</span></td>
          <td>${u.active_sessions}</td>
          <td>${u.job_count}</td>
          <td class="admin-actions-cell">
            <button type="button" class="btn sm" data-revoke="${u.id}" ${u.active_sessions === 0 ? "disabled" : ""}>Revoke sessions</button>
            <button type="button" class="btn sm ${u.is_active ? "danger" : ""}" data-toggle-active="${u.id}" data-next-active="${u.is_active ? "0" : "1"}" ${isSelf ? "disabled" : ""}>
              ${u.is_active ? "Suspend" : "Activate"}
            </button>
          </td>
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
        metric("Total Accounts", c.total_users ?? 0, `${c.admins ?? 0} admin · ${c.developers ?? 0} dev · ${c.regular_users ?? 0} user`) +
        metric("Active Sessions", c.active_sessions ?? 0, "server-side HttpOnly") +
        metric("Jobs (In-Memory)", s.active_jobs ?? 0, `${s.running_separations ?? 0} running · ${s.failed_separations ?? 0} failed`) +
        metric("Temp Storage", `${s.temp_disk_mb ?? 0} MB`, `${s.temp_job_dirs ?? 0} dirs · TTL ${s.job_ttl_minutes ?? 30}m`);

      renderUsers();

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
      $("#adminOAuthGrid").innerHTML = ["google", "microsoft", "github", "yahoo"]
        .map((p) => {
          const on = Boolean(provs[p]);
          return `<div class="oauth-status-item">
            <strong>${esc(p.charAt(0).toUpperCase() + p.slice(1))}</strong>
            <span class="status-dot ${on ? "ok" : "warn"}">${on ? "Configured" : "Not configured"}</span>
          </div>`;
        })
        .join("");
    } catch (err) {
      showMsg(err.message || "Could not load overview.", "warn");
    }
  };

  $("#adminUsersBody")?.addEventListener("change", async (e) => {
    const sel = e.target.closest("[data-role-select]");
    if (!sel) return;
    const uid = sel.dataset.roleSelect;
    try {
      await postJson(`/api/admin/users/${uid}/role`, { role: sel.value });
      toast(`Updated role to ${sel.value}.`);
      loadOverview();
    } catch (err) {
      showMsg(err.message, "warn");
      loadOverview();
    }
  });

  $("#adminUsersBody")?.addEventListener("click", async (e) => {
    const revBtn = e.target.closest("[data-revoke]");
    if (revBtn) {
      const uid = revBtn.dataset.revoke;
      try {
        const out = await postJson(`/api/admin/users/${uid}/revoke`);
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
        await postJson(`/api/admin/users/${uid}/status`, { is_active: nextActive });
        toast(nextActive ? "Account activated." : "Account suspended & sessions revoked.");
        loadOverview();
      } catch (err) {
        showMsg(err.message, "warn");
      }
    }
  });

  $("#adminCleanupBtn")?.addEventListener("click", async () => {
    try {
      const out = await postJson("/api/admin/cleanup");
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
