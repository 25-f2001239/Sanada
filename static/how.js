/* EchoCount — How it works: decorative spectrogram and the sliding-window vote demo */
(() => {
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* spectrogram: harmonic streaks over noise, drawn at low resolution and scaled up by CSS */
  const spec = document.getElementById('spec'), sg = spec.getContext('2d'), W = 220, Hh = 64;
  spec.width = W; spec.height = Hh;
  const img = sg.createImageData(W, Hh); let t = 0;
  const hsv = v => { v = Math.max(0, Math.min(1, v)); const A = [12, 12, 14], B = [255, 90, 31], C = [255, 236, 190], lo = v < .55, k = lo ? v / .55 : (v - .55) / .45, [p, q] = lo ? [A, B] : [B, C]; return p.map((x, i) => x + (q[i] - x) * k); };
  function paint() {
    t += .6;
    for (let y = 0; y < Hh; y++) for (let x = 0; x < W; x++) {
      const fy = 1 - y / Hh, tx = x + t;
      let v = .06 + .05 * Math.sin(tx * .31 + y) * Math.cos(y * .7);
      [[.14, .05, .0], [.27, .03, 1.7], [.41, .04, 3.1], [.62, .05, 4.4]].forEach(([base, wid, ph]) => {
        const f = base + .06 * Math.sin(tx * .035 + ph);
        v += Math.exp(-Math.pow((fy - f) / wid, 2)) * (.5 + .5 * Math.sin(tx * .05 + ph * 2));
        v += Math.exp(-Math.pow((fy - f * 2) / wid, 2)) * .35 * (.5 + .5 * Math.sin(tx * .04 + ph));
      });
      const [r, g, b] = hsv(v), i = (y * W + x) * 4; img.data[i] = r; img.data[i + 1] = g; img.data[i + 2] = b; img.data[i + 3] = 255;
    }
    sg.putImageData(img, 0, 0);
    if (!reduce) requestAnimationFrame(paint);
  }
  paint();

  /* sliding-window demo with made-up predictions */
  const preds = [3, 3, 4, 3, 3, 3, 2], strip = document.getElementById('strip'), win = document.getElementById('win');
  const cv = document.getElementById('stripCv'), chips = document.getElementById('chips'), note = document.getElementById('demoNote');
  chips.innerHTML = preds.map((p, i) => `<span class="chip" data-i="${i}">${i}–${i + 2} s: <b>${p}</b></span>`).join('');
  function backdrop() {
    const d = Math.min(devicePixelRatio || 1, 2), r = strip.getBoundingClientRect(); cv.width = r.width * d; cv.height = r.height * d;
    const g = cv.getContext('2d'); g.scale(d, d); g.fillStyle = 'rgba(242,240,235,.5)';
    for (let x = 4; x < r.width - 4; x += 5) { const a = Math.abs(Math.sin(x * .031) * Math.sin(x * .0093 + 1)) * .8 + .12; g.fillRect(x, r.height / 2 - a * r.height * .36, 3, a * r.height * .72); }
  }
  backdrop(); addEventListener('resize', backdrop);
  const els = [...chips.children]; let i = 0, timer;
  function place(n) { const r = strip.getBoundingClientRect(); win.style.width = (r.width / 8 * 2) + 'px'; win.style.left = (r.width / 8 * n) + 'px'; }
  function step() {
    els.forEach((c, k) => { c.classList.toggle('on', k <= i); c.classList.remove('win-mode'); });
    place(i);
    note.textContent = `Window ${i + 1} of ${preds.length} predicts ${preds[i]} sources.`;
    if (i === preds.length - 1) {
      const tally = {}; preds.forEach(p => tally[p] = (tally[p] || 0) + 1);
      const mode = +Object.entries(tally).sort((a, b) => b[1] - a[1])[0][0];
      els.forEach((c, k) => c.classList.toggle('win-mode', preds[k] === mode));
      note.textContent = `${tally[mode]} of ${preds.length} windows say ${mode}, so the reported count is ${mode}.`;
    }
    i = (i + 1) % preds.length;
  }
  addEventListener('resize', () => place((i + preds.length - 1) % preds.length));
  if (reduce) { i = preds.length - 1; step(); }
  else { step(); timer = setInterval(step, 1100); }
  new IntersectionObserver(([e]) => { if (reduce) return; clearInterval(timer); if (e.isIntersecting) timer = setInterval(step, 1100); }).observe(strip);
})();
