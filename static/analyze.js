/* EchoCount — Analyze page: upload, waveform player, API call, result rendering */
(() => {
  const { $, esc, fmtTime, toast, H } = window.EC;
  const EXT = ['.wav', '.flac', '.mp3', '.ogg', '.m4a'], MAX = 40 * 1024 * 1024, WIN = 8;
  const el = {
    dz: $('#dropzone'), input: $('#audioInput'), fc: $('#fileCard'), fName: $('#fName'), fMeta: $('#fMeta'), clear: $('#clearBtn'),
    player: $('#player'), cv: $('#wave'), play: $('#playBtn'), tNow: $('#tNow'), tTot: $('#tTot'), go: $('#goBtn'),
    cancel: $('#cancelBtn'), empty: $('#sEmpty'), load: $('#sLoad'), loadTxt: $('#loadTxt'), bar: $('#loadBar'), err: $('#sErr'), errTxt: $('#errTxt'), res: $('#sRes'), retry: $('#retryBtn')
  };
  let file = null, url = null, audio = null, peaks = null, dur = 0, crop = null, hl = null, raf = 0, busy = false, xhr = null, runId = 0;

  const show = k => ['empty', 'load', 'err', 'res'].forEach(n => el[n].classList.toggle('hidden', n !== k));
  const bytes = b => b < 1048576 ? `${Math.round(b / 1024)} KB` : `${(b / 1048576).toFixed(2)} MB`;

  /* ---------- file handling ---------- */
  function setFile(f) {
    if (!f) return;
    const ext = '.' + f.name.split('.').pop().toLowerCase();
    if (!EXT.includes(ext)) return toast('Unsupported file type. Use WAV, FLAC, MP3, OGG or M4A.', 'err');
    if (f.size === 0) return toast('That file is empty.', 'err');
    if (f.size > MAX) return toast('That file is larger than 40 MB.', 'err');
    reset(true);
    file = f; url = URL.createObjectURL(f);
    audio = new Audio(url); audio.preload = 'metadata';
    audio.addEventListener('ended', () => { setPlayIcon(false); draw(); });
    audio.addEventListener('loadedmetadata', () => { if (!dur && isFinite(audio.duration)) { dur = audio.duration; info(); } });
    el.fName.textContent = f.name; el.fMeta.textContent = bytes(f.size);
    el.fc.classList.remove('hidden'); el.player.classList.remove('hidden');
    el.go.disabled = false; show('empty');
    requestAnimationFrame(draw); decode(f);
  }
  function reset(keepInput) {
    runId++; if (xhr) { xhr.abort(); xhr = null; } busy = false; /* a replaced or removed file must never receive an old result */
    if (audio) { audio.pause(); audio = null; }
    if (url) URL.revokeObjectURL(url);
    cancelAnimationFrame(raf);
    file = url = peaks = crop = hl = null; dur = 0;
    el.fc.classList.add('hidden'); el.player.classList.add('hidden'); el.go.disabled = true;
    setPlayIcon(false); el.tNow.textContent = el.tTot.textContent = '0:00';
    if (!keepInput) { el.input.value = ''; show('empty'); }
  }
  function info() {
    el.tTot.textContent = fmtTime(dur);
    el.fMeta.textContent = `${bytes(file.size)} · ${fmtTime(dur)}${dur > WIN ? ' · only the middle 8 s is analysed' : dur < WIN ? ' · will be padded to 8 s' : ''}`;
    crop = dur > WIN ? { start: (dur - WIN) / 2, end: (dur + WIN) / 2 } : null;
  }
  async function decode(f) {
    try {
      const AC = window.AudioContext || window.webkitAudioContext, ac = new AC();
      const buf = await ac.decodeAudioData(await f.arrayBuffer()); ac.close && ac.close();
      if (f !== file) return;
      dur = buf.duration; const n = 900, d = buf.getChannelData(0), step = Math.max(1, Math.floor(d.length / n)), skip = Math.max(1, Math.ceil(step / 64));
      const p = new Float32Array(n); let mx = 0;
      for (let i = 0; i < n; i++) { let m = 0; for (let j = i * step; j < Math.min(d.length, (i + 1) * step); j += skip) { const v = Math.abs(d[j]); if (v > m) m = v; } p[i] = m; if (m > mx) mx = m; }
      peaks = p.map(v => v / (mx || 1));
    } catch { peaks = null; toast('Waveform preview is not available for this file. You can still analyze it.'); }
    if (f === file) { info(); draw(); }
  }

  /* ---------- waveform ---------- */
  function draw() {
    const c = el.cv, r = c.getBoundingClientRect(), d = Math.min(devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(r.width * d)), h = Math.max(1, Math.round(r.height * d));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    const g = c.getContext('2d'); g.clearRect(0, 0, w, h);
    const now = audio ? audio.currentTime : 0;
    if (!peaks || !dur) { g.fillStyle = 'rgba(242,240,235,.3)'; g.fillRect(0, h / 2 - d, w, 2 * d); return; }
    const bars = Math.floor(w / (4 * d)), bw = w / bars;
    for (let i = 0; i < bars; i++) {
      const p = peaks[Math.floor(i / bars * peaks.length)], tm = (i + .5) / bars * dur, bh = Math.max(3 * d, p * h * .92);
      const inCrop = !crop || (tm >= crop.start && tm <= crop.end), inHl = hl && tm >= hl.start && tm <= hl.end;
      g.fillStyle = inHl ? '#C8F04A' : tm <= now ? '#FF5A1F' : inCrop ? '#F2F0EB' : 'rgba(242,240,235,.22)';
      const x = i * bw + bw * .19, y = (h - bh) / 2, rr = Math.min(bw * .31, 4 * d);
      g.beginPath(); g.roundRect ? g.roundRect(x, y, bw * .62, bh, rr) : g.rect(x, y, bw * .62, bh); g.fill();
    }
    g.fillStyle = '#fff'; g.fillRect(Math.min(w - 2 * d, now / dur * w), 0, 2 * d, h);
  }
  const setPlayIcon = on => { el.play.innerHTML = `<svg class="ic"><use href="#i-${on ? 'pause' : 'play'}"/></svg><span>${on ? 'Pause' : 'Play'}</span>`; };
  function tick() { draw(); el.tNow.textContent = fmtTime(audio.currentTime); if (!audio.paused) raf = requestAnimationFrame(tick); }
  el.play.addEventListener('click', () => {
    if (!audio) return;
    if (audio.paused) { audio.play().then(() => { setPlayIcon(true); tick(); }).catch(() => toast('This browser cannot play that file.', 'err')); }
    else { audio.pause(); setPlayIcon(false); cancelAnimationFrame(raf); draw(); }
  });
  el.cv.addEventListener('pointerdown', e => {
    if (!audio || !dur) return;
    const r = el.cv.getBoundingClientRect(); audio.currentTime = Math.min(dur, Math.max(0, (e.clientX - r.left) / r.width * dur));
    el.tNow.textContent = fmtTime(audio.currentTime); draw();
  });
  new ResizeObserver(() => file && draw()).observe(el.cv);

  /* ---------- dropzone ---------- */
  el.input.addEventListener('change', () => { setFile(el.input.files[0]); el.input.value = ''; });
  el.dz.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); el.input.click(); } });
  ['dragenter', 'dragover'].forEach(n => el.dz.addEventListener(n, e => { e.preventDefault(); el.dz.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(n => el.dz.addEventListener(n, e => { e.preventDefault(); el.dz.classList.remove('over'); }));
  el.dz.addEventListener('drop', e => setFile(e.dataTransfer.files[0]));
  el.clear.addEventListener('click', () => reset(false));

  /* ---------- synthetic test mixtures (for checking the pipeline without a dataset) ---------- */
  function synth(n) {
    const sr = 16000, len = sr * 8, d = new Float32Array(len), f0 = [120, 185, 250, 320, 150, 210];
    for (let s = 0; s < n; s++) {
      const f = f0[s % 6] * (1 + .03 * s), ph = Math.random() * 6.28, rate = 2.2 + s * .7;
      for (let i = 0; i < len; i++) {
        const t = i / sr, env = Math.pow(Math.max(0, Math.sin(6.283 * rate * t + ph)), .7) * (.6 + .4 * Math.sin(t * 1.3 + s)), vib = 1 + .012 * Math.sin(t * 5.5 + s);
        let v = 0; for (let h = 1; h <= 6; h++) v += Math.sin(6.283 * f * h * vib * t) / h;
        d[i] += v * env * .22;
      }
    }
    let mx = 0; for (let i = 0; i < len; i++) { d[i] += (Math.random() - .5) * .01; mx = Math.max(mx, Math.abs(d[i])); }
    const buf = new ArrayBuffer(44 + len * 2), v = new DataView(buf), str = (o, t) => [...t].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
    str(0, 'RIFF'); v.setUint32(4, 36 + len * 2, true); str(8, 'WAVEfmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); str(36, 'data'); v.setUint32(40, len * 2, true);
    for (let i = 0; i < len; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, d[i] / (mx || 1) * .9)) * 32767, true);
    return new File([buf], `synthetic-${n}-${n === 1 ? 'source' : 'sources'}.wav`, { type: 'audio/wav' });
  }
  document.querySelectorAll('[data-synth]').forEach(b => b.addEventListener('click', () => { toast('Generating a synthetic mixture'); setTimeout(() => setFile(synth(+b.dataset.synth)), 30); }));

  /* ---------- analysis ---------- */
  function run() {
    if (!file || busy) return;
    busy = true; el.go.disabled = true; show('load'); const id = ++runId;
    el.bar.classList.remove('indet'); el.bar.style.width = '0%'; el.loadTxt.textContent = 'Uploading';
    const fd = new FormData(); fd.append('audio', file);
    const x = new XMLHttpRequest(); xhr = x; x.open('POST', '/api/predict'); x.timeout = 180000;
    x.upload.onprogress = e => { if (e.lengthComputable) { const p = e.loaded / e.total * 100; el.bar.style.width = p + '%'; el.loadTxt.textContent = p < 100 ? `Uploading ${Math.round(p)}%` : 'Analysing'; } };
    x.upload.onload = () => { el.loadTxt.textContent = 'Analysing'; el.bar.classList.add('indet'); };
    x.onload = () => {
      if (id !== runId) return;
      let data = {}; try { data = JSON.parse(x.responseText); } catch {}
      done();
      if (x.status >= 200 && x.status < 300) render(data); else fail(data.error || `The server answered with status ${x.status}.`);
    };
    x.onerror = () => { if (id !== runId) return; done(); fail('Could not reach the server. Check that python app.py is running, then try again.'); };
    x.ontimeout = () => { if (id !== runId) return; done(); fail('The analysis took too long. Try a shorter file.'); };
    x.send(fd);
  }
  const done = () => { busy = false; xhr = null; el.go.disabled = !file; };
  el.cancel.addEventListener('click', () => { runId++; if (xhr) xhr.abort(); done(); show('empty'); toast('Analysis cancelled.'); });
  function fail(msg) { el.errTxt.textContent = msg; show('err'); }
  el.go.addEventListener('click', run); el.retry.addEventListener('click', run);

  /* ---------- results ---------- */
  function render(data) {
    const a = data.audio || {}, p = data.prediction;
    if (!data.trained || !p) {
      el.res.innerHTML = `<div class="result">
        <div><h3>The model is not trained yet</h3><p>Your file loaded and its signal details were measured, but no source count is shown because no trained weights exist.</p></div>
        <div class="metrics">${metric('Original length', a.original_duration_seconds + ' s')}${metric('Input sample rate', a.original_sample_rate + ' Hz')}${metric('RMS energy', a.rms_energy)}${metric('Peak amplitude', a.peak_amplitude)}</div>
        <div class="notice info">Add labelled audio and run <code>python train.py --metadata data/metadata.csv --epochs 25 --max-sources 32</code>. The <a href="/docs#train">training guide</a> has the details.</div></div>`;
      show('res'); return;
    }
    const tl = p.timeline || [], mode = p.estimated_sources, off = a.crop_start_seconds ?? (crop ? crop.start : 0);
    const avg = tl.length ? tl.reduce((s, t) => s + t.confidence, 0) / tl.length : 0;
    const agree = tl.length ? tl.filter(t => t.count === mode).length / tl.length : 0;
    const maxC = Math.max(1, ...tl.map(t => t.count)), tally = {};
    tl.forEach(t => tally[t.count] = (tally[t.count] || 0) + 1);
    const rows = Object.entries(tally).sort((x, y) => y[1] - x[1] || x[0] - y[0]);
    const orig = a.original_duration_seconds || dur;
    let warn = '';
    if (agree < .6 || avg < .5) warn = `<div class="notice">The windows disagree or the model is unsure (${Math.round(agree * 100)}% agreement, ${Math.round(avg * 100)}% average confidence). Treat this count with care.</div>`;
    if (orig > WIN + .05) warn += `<div class="notice info">Only the middle ${WIN} seconds (${fmtTime(off)} to ${fmtTime(off + WIN)}) of this ${fmtTime(orig)} recording were analysed.</div>`;

    el.res.innerHTML = `<div class="result">
      <div class="result-top">
        <div class="ring"><svg viewBox="0 0 120 120"><defs><linearGradient id="ringGrad" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#FF5A1F"/><stop offset="1" stop-color="#FF3D8B"/></linearGradient></defs><circle class="bg-c" cx="60" cy="60" r="54"/><circle class="fg-c" id="ringFg" cx="60" cy="60" r="54"/></svg>
          <div><b id="bigN">0</b><span>${mode === 1 ? 'source' : 'sources'}</span></div></div>
        <div><h3>Estimated source count</h3><p>${tl.length} windows voted. The ring shows average confidence.</p></div>
      </div>
      <div class="metrics">${metric('Average confidence', (avg * 100).toFixed(1) + '%')}${metric('Window agreement', Math.round(agree * 100) + '%')}${metric('Recording length', fmtTime(orig))}${metric('Loudness (RMS)', a.rms_energy)}</div>
      ${warn}
      <div><div class="sub">Window by window <small>Hover a bar to mark it on the waveform</small></div>
        <div class="tl">${tl.map((t, i) => `<div class="tl-bar" tabindex="0" data-i="${i}" title="${(off + t.start).toFixed(1)} to ${(off + t.end).toFixed(1)} s: ${t.count} sources, ${(t.confidence * 100).toFixed(0)}% confident"><b>${t.count}</b><i style="--h:${Math.max(14, t.count / maxC * 82)}%;--o2:${(.45 + t.confidence * .55).toFixed(2)};--d:${i * 70}ms"></i></div>`).join('')}</div>
        <div class="tl-axis"><span>${fmtTime(off)}</span><span>${fmtTime(off + WIN)}</span></div></div>
      <div><div class="sub">Votes</div><div class="votes">${rows.map(([c, n], i) => `<div class="vote ${i === 0 ? 'top' : ''}"><span>${c} ${c == 1 ? 'source' : 'sources'}</span><div><i style="--w:${n / tl.length * 100}%"></i></div><span>${n}/${tl.length}</span></div>`).join('')}</div></div>
      <p class="fine">${esc(data.message || 'The result is an estimate.')}</p></div>`;
    show('res');

    requestAnimationFrame(() => requestAnimationFrame(() => { const fg = $('#ringFg'); if (fg) fg.style.strokeDashoffset = 339.3 * (1 - Math.min(1, avg)); }));
    const big = $('#bigN'), t0 = performance.now();
    (function step(n) { const k = Math.min(1, (n - t0) / 900); big.textContent = Math.round(mode * (1 - Math.pow(1 - k, 3))); if (k < 1) requestAnimationFrame(step); })(t0);
    el.res.querySelectorAll('.tl-bar').forEach(b => {
      const on = () => { const t = tl[b.dataset.i]; hl = { start: off + t.start, end: off + t.end }; draw(); };
      const out = () => { hl = null; draw(); };
      b.addEventListener('pointerenter', on); b.addEventListener('pointerleave', out); b.addEventListener('focus', on); b.addEventListener('blur', out);
    });
    H.add({ id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6), ts: Date.now(), filename: a.filename || file.name, count: mode,
      conf: +avg.toFixed(3), agree: +agree.toFixed(3), duration: orig, counts: tl.map(t => t.count) });
  }
  const metric = (l, v) => `<div class="metric"><small>${l}</small><b>${esc(v ?? '—')}</b></div>`;
})();
