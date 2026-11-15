/* EchoCount — How it works: decorative spectrogram and full-clip interval */
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

  /* show one analysis interval across the complete clip */
  const strip = document.getElementById('strip'), win = document.getElementById('win');
  const cv = document.getElementById('stripCv');
  function backdrop() {
    const d = Math.min(devicePixelRatio || 1, 2), r = strip.getBoundingClientRect(); cv.width = r.width * d; cv.height = r.height * d;
    const g = cv.getContext('2d'); g.scale(d, d); g.fillStyle = 'rgba(242,240,235,.5)';
    for (let x = 4; x < r.width - 4; x += 5) { const a = Math.abs(Math.sin(x * .031) * Math.sin(x * .0093 + 1)) * .8 + .12; g.fillRect(x, r.height / 2 - a * r.height * .36, 3, a * r.height * .72); }
  }
  function place() { win.style.width = `${strip.clientWidth}px`; win.style.left = '0'; }
  backdrop(); place();
  addEventListener('resize', () => { backdrop(); place(); });
})();
