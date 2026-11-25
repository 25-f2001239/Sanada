/* EchoCount — Settings -> Account: Data Export, Deactivate & Permanent Delete */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const csrf = document.querySelector('meta[name="csrf-token"]')?.content || "";

  const showErr = (el, msg) => {
    if (!el) return;
    el.textContent = msg || "";
    el.classList.toggle("hidden", !msg);
  };

  const apiJson = async (url, method, body) => {
    const res = await fetch(url, {
      method,
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf },
      body: JSON.stringify(body || {}),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || "Request failed.");
    return data;
  };

  const deactDlg = $("#deactivateDialog");
  const delDlg = $("#deleteDialog");

  $("#openDeactivateBtn")?.addEventListener("click", () => {
    showErr($("#deactErr"), "");
    deactDlg?.showModal();
  });
  $("#openDeleteBtn")?.addEventListener("click", () => {
    showErr($("#delErr"), "");
    delDlg?.showModal();
  });
  document.querySelectorAll("[data-close-dialog]").forEach((btn) => {
    btn.addEventListener("click", () => document.getElementById(btn.dataset.closeDialog)?.close());
  });

  $("#deactivateForm")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const errEl = $("#deactErr");
    const btn = $("#confirmDeactivateBtn");
    showErr(errEl, "");
    if (btn) btn.disabled = true;
    try {
      const data = await apiJson("/api/account/deactivate", "POST", {
        password: $("#deactPassword")?.value || "",
      });
      try {
        sessionStorage.clear();
      } catch {}
      location.href = data.redirect || "/login?deactivated=1";
    } catch (err) {
      showErr(errEl, err.message);
      if (btn) btn.disabled = false;
    }
  });

  $("#deleteForm")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const errEl = $("#delErr");
    const btn = $("#confirmDeleteBtn");
    showErr(errEl, "");
    const confirmation = ($("#delConfirm")?.value || "").trim();
    if (confirmation !== "DELETE") {
      showErr(errEl, 'Type "DELETE" exactly to confirm permanent account deletion.');
      return;
    }
    if (btn) btn.disabled = true;
    try {
      const data = await apiJson("/api/account", "DELETE", {
        password: $("#delPassword")?.value || "",
        confirmation,
      });
      try {
        sessionStorage.clear();
      } catch {}
      location.href = data.redirect || "/login?deleted=1";
    } catch (err) {
      showErr(errEl, err.message);
      if (btn) btn.disabled = false;
    }
  });
})();
