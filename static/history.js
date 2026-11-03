/* EchoCount — History page */
(() => {
  const { $, esc, fmtTime, toast, H } = window.EC;
  const list = $('#hlist'), stats = $('#hstats'), q = $('#q');
  const when = ts => new Date(ts).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });

  function spark(counts) {
    if (!counts || !counts.length) return '';
    const max = Math.max(...counts), min = Math.min(...counts), span = Math.max(1, max - min), w = 120, h = 32;
    const pts = counts.map((c, i) => `${counts.length === 1 ? w / 2 : (i / (counts.length - 1) * (w - 6) + 3).toFixed(1)},${(h - 4 - (c - min) / span * (h - 8)).toFixed(1)}`).join(' ');
    return `<svg class="spark" viewBox="0 0 ${w} ${h}" aria-hidden="true"><polyline points="${pts}" fill="none" stroke="#FF5A1F" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  }
  function render() {
    const all = H.all().filter(h => h && typeof h.filename === 'string' && Number.isFinite(+h.count)), term = q.value.trim().toLowerCase(), items = all.filter(h => h.filename.toLowerCase().includes(term));
    const freq = {}; all.forEach(h => freq[h.count] = (freq[h.count] || 0) + 1);
    const common = Object.entries(freq).sort((a, b) => b[1] - a[1])[0];
    const avg = all.length ? all.reduce((s, h) => s + h.count, 0) / all.length : 0;
    stats.innerHTML = [['Analyses', all.length], ['Average count', all.length ? avg.toFixed(1) : '—'], ['Most common count', common ? common[0] : '—']]
      .map(([l, v]) => `<div class="metric"><small>${l}</small><b>${v}</b></div>`).join('');
    if (!all.length) { list.innerHTML = `<div class="panel state"><div class="orbit"><svg class="ic"><use href="#i-wave"/></svg></div><h3>Nothing here yet</h3><p>Results appear after you analyze a recording with a trained model.</p><a class="btn primary" href="/analyze">Analyze a recording</a></div>`; return; }
    if (!items.length) { list.innerHTML = `<div class="panel state"><h3>No match</h3><p>No saved result has “${esc(q.value)}” in its file name.</p></div>`; return; }
    list.innerHTML = items.map((h, i) => `<article class="hrow" style="animation-delay:${Math.min(i, 10) * 40}ms">
      <div class="badge" aria-label="${h.count} sources">${h.count}</div>
      <div class="hinfo"><strong title="${esc(h.filename)}">${esc(h.filename)}</strong><small>${when(h.ts)} · ${fmtTime(h.duration)} · ${Math.round((h.conf || 0) * 100)}% confident · ${Math.round((h.agree || 0) * 100)}% agreement</small></div>
      ${spark(h.counts)}
      <button class="icon-btn" data-del="${h.id}" aria-label="Delete ${esc(h.filename)}"><svg class="ic"><use href="#i-trash"/></svg></button></article>`).join('');
  }
  list.addEventListener('click', e => { const b = e.target.closest('[data-del]'); if (b) { H.remove(b.dataset.del); render(); toast('Result deleted.'); } });
  q.addEventListener('input', render);
  $('#clearAll').addEventListener('click', () => { if (H.all().length && confirm('Delete all saved results from this browser?')) { H.clear(); render(); toast('History cleared.'); } });
  $('#exportBtn').addEventListener('click', () => {
    const all = H.all(); if (!all.length) return toast('There is nothing to export yet.');
    const cell = v => { let s = String(v ?? ''); if (/^[=+\-@\t\r]/.test(s)) s = "'" + s; return `"${s.replace(/"/g, '""')}"`; };
    const rows = [['file', 'date', 'estimated_sources', 'avg_confidence', 'window_agreement', 'duration_s', 'window_counts']]
      .concat(all.map(h => [h.filename, new Date(h.ts).toISOString(), h.count, h.conf, h.agree, h.duration, (h.counts || []).join(' ')]));
    const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([rows.map(r => r.map(cell).join(',')).join('\n')], { type: 'text/csv' }));
    a.download = 'echocount-history.csv'; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });
  render();
})();
