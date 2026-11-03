/* EchoCount — shared behaviour: nav, command menu, motion helpers, toasts, tabs, copy, status, history store, hero */
(() => {
  const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmtTime = s => { s = Math.max(0, s || 0); return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`; };
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches, fine = matchMedia('(hover:hover) and (pointer:fine)').matches;

  function toast(msg, type = '') {
    const t = document.createElement('div'); t.className = 'toast ' + type; t.textContent = msg; $('#toasts').appendChild(t);
    setTimeout(() => { t.classList.add('out'); setTimeout(() => t.remove(), 360); }, 3400);
  }
  const KEY = 'echocount.history.v1';
  const H = {
    all() { try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch { return []; } },
    save(a) { try { localStorage.setItem(KEY, JSON.stringify(a)); } catch { toast('Could not save history in this browser.', 'err'); } },
    add(e) { const a = H.all(); a.unshift(e); H.save(a.slice(0, 200)); },
    remove(id) { H.save(H.all().filter(x => x.id !== id)); },
    clear() { H.save([]); }
  };
  window.EC = { $, esc, fmtTime, toast, H };

  /* ---- theme: saved choice wins, otherwise follow the OS; first paint is handled by the inline script in base.html ---- */
  const root = document.documentElement, TKEY = 'echocount.theme', themeBtn = $('#themeBtn'), themeMeta = $('#themeColor');
  const stored = () => { try { const v = localStorage.getItem(TKEY); return v === 'light' || v === 'dark' ? v : null; } catch { return null; } };
  const applyTheme = t => {
    root.dataset.theme = t;
    themeBtn.setAttribute('aria-label', t === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
    if (themeMeta) themeMeta.content = t === 'dark' ? '#121214' : '#EEEDE9';
  };
  const toggleTheme = () => {
    const next = root.dataset.theme === 'dark' ? 'light' : 'dark';
    root.classList.add('theme-anim'); applyTheme(next);
    try { localStorage.setItem(TKEY, next); } catch {}
    setTimeout(() => root.classList.remove('theme-anim'), 450);
  };
  applyTheme(root.dataset.theme || 'light');
  themeBtn.addEventListener('click', toggleTheme);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => { if (!stored()) applyTheme(e.matches ? 'dark' : 'light'); });
  addEventListener('storage', e => { if (e.key === TKEY && (e.newValue === 'light' || e.newValue === 'dark')) applyTheme(e.newValue); });

  /* ---- nav: sliding indicator, scroll progress, mobile sheet ---- */
  const nav = $('#nav'), links = $('#links'), ind = $('#ind'), bar = $('#scrollbar');
  const move = a => { if (!a || innerWidth <= 860) return; ind.style.opacity = 1; ind.style.left = a.offsetLeft + 'px'; ind.style.width = a.offsetWidth + 'px'; };
  const active = () => $('a.active', links);
  const home = () => active() ? move(active()) : (ind.style.opacity = 0);
  $$('a', links).forEach(a => a.addEventListener('pointerenter', () => move(a)));
  links.addEventListener('pointerleave', home); addEventListener('resize', home); home();
  document.fonts && document.fonts.ready.then(home);
  const onScroll = () => { nav.classList.toggle('scrolled', scrollY > 10); const m = document.documentElement.scrollHeight - innerHeight; bar.style.transform = `scaleX(${m > 0 ? scrollY / m : 0})`; };
  addEventListener('scroll', onScroll, { passive: true }); onScroll();
  const sheet = $('#sheet'), burger = $('#burger');
  const setSheet = on => {
    sheet.classList.toggle('open', on); sheet.setAttribute('aria-hidden', !on); sheet.inert = !on; burger.setAttribute('aria-expanded', on);
    burger.innerHTML = `<svg class="ic"><use href="#i-${on ? 'x' : 'menu'}"/></svg>`; document.body.style.overflow = on ? 'hidden' : '';
  };
  burger.addEventListener('click', () => setSheet(!sheet.classList.contains('open')));
  $$('a', sheet).forEach(a => { if (a.pathname === location.pathname) a.classList.add('active'); a.addEventListener('click', () => setSheet(false)); });

  /* ---- command menu (Ctrl/Cmd + K) ---- */
  const cmdk = $('#cmdk'), cin = $('#cmdIn'), clist = $('#cmdList');
  const cmds = [
    { t: 'Toggle light / dark theme', k: '◐', s: 'Action', run: toggleTheme },
    { t: 'Go to Home', k: 'H', s: 'Page', run: () => location.href = '/' },
    { t: 'Analyze a recording', k: 'A', s: 'Page', run: () => location.href = '/analyze' },
    { t: 'Open history', k: 'Y', s: 'Page', run: () => location.href = '/history' },
    { t: 'How it works', k: '?', s: 'Page', run: () => location.href = '/how-it-works' },
    { t: 'Read the docs', k: 'D', s: 'Page', run: () => location.href = '/docs' },
    { t: 'Check server health', k: '♥', s: 'Action', run: () => fetch('/api/health').then(r => r.json()).then(d => toast(d.model.trained ? 'Server is up and a model is trained.' : 'Server is up. No trained model yet.')).catch(() => toast('The server did not respond.', 'err')) },
    { t: 'Copy training command', k: '⌘', s: 'Action', run: () => navigator.clipboard.writeText('python train.py --metadata data/metadata.csv --epochs 25 --max-sources 32').then(() => toast('Training command copied.')).catch(() => toast('Could not access the clipboard.', 'err')) },
    { t: 'Clear saved history', k: '×', s: 'Action', run: () => { if (H.all().length && confirm('Delete all saved results from this browser?')) { H.clear(); toast('History cleared.'); if (location.pathname === '/history') location.reload(); } } }
  ];
  let sel = 0, shown = cmds;
  function paint() {
    const q = cin.value.trim().toLowerCase(); shown = cmds.filter(c => c.t.toLowerCase().includes(q)); sel = Math.min(sel, Math.max(0, shown.length - 1));
    clist.innerHTML = shown.length ? shown.map((c, i) => `<li role="option" data-i="${i}" aria-selected="${i === sel}"><b>${c.k}</b>${esc(c.t)}<small>${c.s}</small></li>`).join('') : '<li>No results</li>';
  }
  const openCmd = on => { cmdk.classList.toggle('open', on); cmdk.setAttribute('aria-hidden', !on); cmdk.inert = !on; if (on) { cin.value = ''; sel = 0; paint(); setTimeout(() => cin.focus(), 30); } };
  const exec = i => { const c = shown[i]; if (c) { openCmd(false); c.run(); } };
  $('#cmdBtn').addEventListener('click', () => openCmd(true));
  cmdk.addEventListener('pointerdown', e => { if (e.target === cmdk) openCmd(false); });
  cin.addEventListener('input', () => { sel = 0; paint(); });
  clist.addEventListener('click', e => { const li = e.target.closest('li[data-i]'); if (li) exec(+li.dataset.i); });
  clist.addEventListener('pointermove', e => { const li = e.target.closest('li[data-i]'); if (li && +li.dataset.i !== sel) { sel = +li.dataset.i; paint(); } });
  addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); openCmd(!cmdk.classList.contains('open')); return; }
    if (e.key === 'Escape') { openCmd(false); setSheet(false); }
    if (!cmdk.classList.contains('open')) return;
    if (e.key === 'ArrowDown') { e.preventDefault(); sel = (sel + 1) % shown.length; paint(); }
    if (e.key === 'ArrowUp') { e.preventDefault(); sel = (sel - 1 + shown.length) % shown.length; paint(); }
    if (e.key === 'Enter') { e.preventDefault(); exec(sel); }
  });

  /* ---- motion helpers ---- */
  document.addEventListener('pointermove', e => { const c = e.target.closest && e.target.closest('[data-spot]'); if (c) { const r = c.getBoundingClientRect(); c.style.setProperty('--mx', e.clientX - r.left + 'px'); c.style.setProperty('--my', e.clientY - r.top + 'px'); } });
  if (fine && !reduce) {
    $$('[data-magnetic]').forEach(b => {
      b.addEventListener('pointermove', e => { const r = b.getBoundingClientRect(); b.style.transform = `translate(${(e.clientX - r.left - r.width / 2) * .22}px,${(e.clientY - r.top - r.height / 2) * .3}px)`; });
      b.addEventListener('pointerleave', () => b.style.transform = '');
    });
    $$('[data-tilt]').forEach(t => {
      t.addEventListener('pointermove', e => { const r = t.getBoundingClientRect(), x = (e.clientX - r.left) / r.width - .5, y = (e.clientY - r.top) / r.height - .5; t.style.transform = `perspective(1200px) rotateY(${x * 4}deg) rotateX(${-y * 4}deg)`; });
      t.addEventListener('pointerleave', () => t.style.transform = '');
    });
  }
  $$('[data-split]').forEach(h => {
    let n = 0; const frag = document.createDocumentFragment(), wrap = el => { const w = document.createElement('span'); w.className = 'w'; const s = document.createElement('span'); s.style.setProperty('--i', n++); s.appendChild(el); w.appendChild(s); return w; };
    [...h.childNodes].forEach(node => {
      if (node.nodeType === 3) node.textContent.split(/(\s+)/).forEach(p => { if (!p) return; frag.appendChild(/^\s+$/.test(p) ? document.createTextNode(' ') : wrap(document.createTextNode(p))); });
      else frag.appendChild(wrap(node));
    });
    h.textContent = ''; h.appendChild(frag);
  });
  $$('[data-marquee] .marquee-track').forEach(t => { t.innerHTML += t.innerHTML; });
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return; io.unobserve(e.target); const el = e.target;
    if (el.classList.contains('rv')) el.classList.add('in');
    if (el.dataset.count) {
      const end = +el.dataset.count, suf = el.dataset.suffix || '', t0 = performance.now();
      (function f(n) { const k = Math.min(1, (n - t0) / 1400); el.textContent = Math.round(end * (1 - Math.pow(1 - k, 4))) + suf; if (k < 1) requestAnimationFrame(f); })(t0);
    }
  }), { threshold: .2 });
  $$('.rv,[data-count]').forEach(el => io.observe(el));

  /* ---- tabs, copy ---- */
  $$('[data-tabs]').forEach(root => {
    const btns = $$('[data-tab]', root), panels = $$('[data-panel]', root);
    btns.forEach(b => b.addEventListener('click', () => { btns.forEach(x => x.setAttribute('aria-selected', x === b)); panels.forEach(p => p.hidden = p.dataset.panel !== b.dataset.tab); }));
  });
  document.addEventListener('click', async e => {
    const b = e.target.closest('[data-copy]'); if (!b) return; const src = $(b.dataset.copy); if (!src) return;
    try { await navigator.clipboard.writeText(src.textContent.trim()); } catch { const r = document.createRange(); r.selectNodeContents(src); getSelection().removeAllRanges(); getSelection().addRange(r); document.execCommand('copy'); }
    const old = b.textContent; b.textContent = 'Copied'; setTimeout(() => b.textContent = old, 1400);
  });

  /* ---- footer wordmark: scale so the whole word always fits the page width ---- */
  const fw = $('.foot-word');
  if (fw) {
    const fit = () => {
      const box = fw.parentElement.clientWidth; if (!box) return;
      fw.style.fontSize = '100px'; fw.style.width = 'max-content';
      const w = fw.getBoundingClientRect().width; fw.style.width = '';
      if (w) fw.style.fontSize = Math.floor(100 * box / w * .98 * 10) / 10 + 'px';
    };
    fit(); addEventListener('resize', fit); document.fonts && document.fonts.ready.then(fit); addEventListener('load', fit);
  }

  /* ---- live server status ---- */
  const pill = $('#statusPill');
  fetch('/api/health').then(r => r.json()).then(d => { const ok = d.model && d.model.trained; pill.dataset.state = ok ? 'ready' : 'untrained'; $('.t', pill).textContent = ok ? 'Model ready' : 'Model not trained'; })
    .catch(() => { pill.dataset.state = 'offline'; $('.t', pill).textContent = 'Server offline'; });

  /* ---- hero: one coloured lane per source, summed into one mixture line ---- */
  const hero = $('#heroCanvas');
  if (hero) {
    const ctx = hero.getContext('2d'), numEl = $('#heroN'), box = numEl.parentElement, lanesEl = $('#lanes');
    const cols = ['#FF5A1F', '#8B5CF6', '#C8F04A', '#FF3D8B', '#12B5A6', '#FFB800', '#F2F0EB', '#FF8A5C'];
    const L = cols.map((c, i) => ({ c, f: .9 + i * .37 + (i % 3) * .21, p: i * 1.7, s: .5 + ((i * 37) % 10) / 14, a: 0 }));
    lanesEl.innerHTML = cols.map(c => `<i style="--c:${c}"></i>`).join('');
    let w = 0, h = 0, t = 0, target = 1, running = true, k = 0; const seq = [3, 5, 2, 7, 4, 6, 1, 3];
    const fit = () => { const d = Math.min(devicePixelRatio || 1, 2), r = hero.getBoundingClientRect(); w = r.width; h = r.height; hero.width = w * d; hero.height = h * d; ctx.setTransform(d, 0, 0, d, 0, 0); };
    const set = n => { target = n; numEl.textContent = n; $$('i', lanesEl).forEach((i, x) => i.classList.toggle('on', x < n)); box.classList.remove('bump'); void box.offsetWidth; box.classList.add('bump'); };
    const val = (l, x) => Math.sin(x / w * Math.PI * 2 * l.f * 1.6 + t * l.s + l.p) * (.6 + .4 * Math.sin(x / w * 3 + l.p + t * .4));
    function frame() {
      t += .016; ctx.clearRect(0, 0, w, h);
      const top = h * .12, lane = h * .56 / 8, amp = lane * .5, sumY = h * .82;
      L.forEach((l, i) => l.a += ((i < target ? 1 : 0) - l.a) * .07);
      L.forEach((l, i) => {
        if (l.a < .01) return; const cy = top + lane * (i + .5);
        ctx.globalAlpha = l.a * .95; ctx.strokeStyle = l.c; ctx.lineWidth = 2; ctx.beginPath();
        for (let x = 0; x <= w; x += 3) { const y = cy + val(l, x) * amp * l.a; x ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
        ctx.stroke();
      });
      const act = L.reduce((s, l) => s + l.a, 0), norm = Math.max(1, Math.sqrt(act));
      ctx.globalAlpha = .25; ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.setLineDash([4, 6]); ctx.beginPath(); ctx.moveTo(0, h * .72); ctx.lineTo(w, h * .72); ctx.stroke(); ctx.setLineDash([]);
      ctx.globalAlpha = 1; ctx.strokeStyle = '#fff'; ctx.lineWidth = 3; ctx.shadowColor = '#FF5A1F'; ctx.shadowBlur = 18; ctx.beginPath();
      for (let x = 0; x <= w; x += 2) { let s = 0; L.forEach(l => { if (l.a > .01) s += val(l, x) * l.a; }); const y = sumY + s / norm * h * .085; x ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
      ctx.stroke(); ctx.shadowBlur = 0;
      if (running && !reduce) requestAnimationFrame(frame);
    }
    fit(); addEventListener('resize', () => { fit(); if (reduce) frame(); });
    if (reduce) { L.forEach((l, i) => l.a = i < 3 ? 1 : 0); set(3); frame(); }
    else {
      setTimeout(() => set(1), 900); setTimeout(() => set(2), 1500); setTimeout(() => set(3), 2100);
      setTimeout(() => setInterval(() => { if (running) set(seq[k++ % seq.length]); }, 2800), 4200);
      new IntersectionObserver(([e]) => { const was = running; running = e.isIntersecting && !document.hidden; if (running && !was) frame(); }).observe(hero);
      document.addEventListener('visibilitychange', () => { if (!document.hidden && !running) { running = true; frame(); } else if (document.hidden) running = false; });
      frame();
    }
  }
})();
