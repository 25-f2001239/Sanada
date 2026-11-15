const WAVE_BINS = 64;

const vertexSource = `#version 300 es
precision highp float;
out vec2 vUv;
void main() {
  vec2 point = gl_VertexID == 0 ? vec2(-1.0, -1.0) : (gl_VertexID == 1 ? vec2(3.0, -1.0) : vec2(-1.0, 3.0));
  vUv = point * 0.5 + 0.5;
  gl_Position = vec4(point, 0.0, 1.0);
}`;

const fragmentSource = `#version 300 es
precision highp float;
uniform vec2 uResolution;
uniform float uTime;
uniform float uProgress;
uniform float uPage;
uniform float uLight;
uniform float uListening;
uniform float uCount;
uniform vec2 uPointer;
uniform float uWave[64];
uniform float uFft[64];
in vec2 vUv;
out vec4 outColor;

float sampleSignal(float values[64], float x) {
  float position = clamp(x, 0.0, 1.0) * 63.0;
  int index = int(floor(position));
  return mix(values[index], values[min(index + 1, 63)], fract(position));
}
float lineGlow(float distance, float width) { return exp(-abs(distance) * width); }
void main() {
  vec2 uv = vUv;
  vec2 p = (uv - 0.5) * vec2(uResolution.x / uResolution.y, 1.0);
  float t = uTime;
  vec3 color = uLight > 0.5 ? vec3(0.935, 0.929, 0.902) : vec3(0.027, 0.030, 0.043);
  vec3 accent = uLight > 0.5 ? vec3(0.66, 0.27, 0.14) : vec3(0.95, 0.52, 0.36);
  vec3 cyan = vec3(0.32, 0.76, 0.87);
  vec3 coral = vec3(0.91, 0.42, 0.34);
  vec3 lime = vec3(0.64, 0.82, 0.38);
  vec3 lilac = vec3(0.78, 0.55, 0.90);
  float pointerRipple = exp(-length((uv - uPointer) * vec2(uResolution.x / uResolution.y, 1.0)) * 9.0) * 0.035;

  if (uPage < 0.5 || uPage > 1.5) {
    float travel = uProgress;
    float sourceMix = 1.0 - smoothstep(0.08, 0.37, travel);
    float blendMix = smoothstep(0.15, 0.38, travel) * (1.0 - smoothstep(0.37, 0.49, travel));
    float terrainMix = smoothstep(0.38, 0.64, travel) * (1.0 - smoothstep(0.62, 0.72, travel));
    float networkMix = smoothstep(0.64, 0.82, travel) * (1.0 - smoothstep(0.81, 0.91, travel));
    float resolveMix = smoothstep(0.84, 0.98, travel);

    vec3 lanes = vec3(0.0);
    float laneCount = mix(1.0, 4.0, smoothstep(0.0, 0.22, travel));
    for (int i = 0; i < 4; i++) {
      float fi = float(i);
      if (fi >= laneCount) continue;
      float phase = uv.x * (5.0 + fi * 1.8) + t * (0.23 + fi * 0.06) + fi * 1.91;
      float amplitude = 0.028 + 0.008 * fi;
      float center = 0.25 + fi * 0.12 + sin(phase) * amplitude + sin(uv.x * 10.0 + t * 0.12 + fi) * 0.012;
      float trace = lineGlow(uv.y - center, 210.0);
      float bead = (1.0 - smoothstep(0.0, 0.018, abs(fract(uv.x * (3.0 + fi) - t * 0.035 + fi * 0.2) - 0.5))) * 0.18;
      vec3 laneColor = i == 0 ? cyan : (i == 1 ? coral : (i == 2 ? lime : lilac));
      lanes += laneColor * (trace * 0.34 + bead * trace) * sourceMix;
    }
    float mixtureY = 0.47 + sin(uv.x * 12.0 + t * 0.35) * 0.055 + sin(uv.x * 27.0 - t * 0.2) * 0.016;
    float mixtureLine = lineGlow(uv.y - mixtureY, 250.0);
    vec3 mixture = accent * mixtureLine * (0.08 + blendMix * 0.58);

    float horizon = 0.37 + sin(uv.x * 3.2 + t * 0.08) * 0.025;
    float depth = max(0.0, (horizon - uv.y) / max(0.04, horizon));
    float terrainWave = sin(uv.x * 22.0 + t * 0.18) * 0.055 + cos(uv.x * 7.0 - t * 0.11) * 0.07;
    float terrainLine = lineGlow(uv.y - (horizon - depth * 0.16 + terrainWave * depth), 115.0 + depth * 120.0);
    float scan = smoothstep(0.4, 1.0, fract(uv.x * 22.0 + terrainWave * 1.4));
    vec3 terrain = accent * terrainLine * (0.14 + scan * 0.16) * terrainMix;

    float netGrid = step(0.985, fract(uv.x * 19.0)) + step(0.986, fract(uv.y * 9.0));
    float node = 1.0 - smoothstep(0.0, 0.035, length(vec2(fract(uv.x * 7.0) - 0.5, fract(uv.y * 4.0) - 0.5)));
    float signalPass = lineGlow(uv.y - (0.5 + sin(uv.x * 17.0 + t * 0.25) * 0.09), 95.0);
    vec3 network = accent * (netGrid * 0.045 + node * 0.14) * networkMix + vec3(0.56, 0.56, 0.54) * signalPass * networkMix * 0.14;
    float resolveLine = lineGlow(uv.y - (0.5 + sin(uv.x * 8.0 + t * 0.12) * 0.025), 180.0);
    vec3 resolved = accent * resolveLine * resolveMix * 0.24;
    float guideX = step(0.998, fract(uv.x * 20.0));
    float guideY = step(0.997, fract(uv.y * 10.0));
    float guide = max(guideX, guideY);
    color += lanes + mixture + terrain + network + resolved;
    color += (uLight > 0.5 ? vec3(0.16) : vec3(0.14)) * guide * 0.18;
  } else {
    float signal = sampleSignal(uWave, uv.x);
    float waveY = 0.5 + signal * 0.18;
    float waveLine = lineGlow(uv.y - waveY, 260.0);
    float fft = sampleSignal(uFft, uv.x);
    float spectrumY = 0.12 + fft * 0.21;
    float spectrumLine = lineGlow(uv.y - spectrumY, 155.0);
    float listeningPulse = uListening * exp(-abs(length(p) - (0.12 + 0.035 * sin(t * 1.1))) * 24.0);
    color += accent * waveLine * (0.16 + min(1.0, abs(signal) * 3.0) * 0.42);
    color += vec3(0.65) * spectrumLine * fft * 0.22;
    color += accent * listeningPulse * 0.2;
    for (int i = 0; i < 32; i++) {
      float fi = float(i);
      if (fi >= uCount) break;
      float angle = fi * 2.39996 + 0.4;
      float ring = 0.16 + 0.16 * fract(fi * 0.618);
      vec2 center = vec2(0.75, 0.5) + vec2(cos(angle), sin(angle)) * ring;
      float orb = exp(-length(uv - center) * 130.0);
      color += accent * orb * 0.08;
      color += accent * (1.0 - smoothstep(0.0, 0.014, abs(length(uv - center) - 0.012))) * 0.12;
    }
    float centerMark = lineGlow(uv.y - 0.5, 480.0) * (1.0 - uListening) * (uCount < 0.5 ? 0.08 : 0.0);
    color += accent * centerMark;
  }
  color += accent * pointerRipple * (uLight > 0.5 ? 0.18 : 0.42);
  outColor = vec4(color, 1.0);
}`;

function reducedMotion() {
  return matchMedia('(prefers-reduced-motion: reduce)').matches;
}

function compress(values, out = new Float32Array(WAVE_BINS), absolute = false) {
  if (!values?.length) { out.fill(0); return out; }
  const step = values.length / WAVE_BINS;
  for (let i = 0; i < WAVE_BINS; i++) {
    const start = Math.floor(i * step);
    const end = Math.max(start + 1, Math.floor((i + 1) * step));
    let total = 0;
    for (let j = start; j < end && j < values.length; j++) {
      const value = Number(values[j]) || 0;
      total += absolute ? Math.abs(value) : value;
    }
    out[i] = total / Math.max(1, end - start);
  }
  return out;
}

function compile(gl, kind, source) {
  const shader = gl.createShader(kind);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const reason = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(reason || 'Shader compilation failed.');
  }
  return shader;
}

export function createScene() {
  const canvas = document.getElementById('worldCanvas');
  const fallback = document.getElementById('worldFallback');
  const stage = document.querySelector('[data-scene-stage]');
  if (stage && canvas && fallback) stage.append(canvas, fallback);
  const page = document.body.dataset.page;
  const quiet = document.body.dataset.sceneMode === 'quiet';
  const prefersReduced = reducedMotion();
  const constrained = (navigator.deviceMemory && navigator.deviceMemory <= 4)
    || (navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 4)
    || matchMedia('(max-width: 700px)').matches;
  const pixelRatioCap = constrained ? 1.25 : 1.75;
  const prefersLight = () => document.documentElement.dataset.theme === 'light';
  let gl = null, program = null, vao = null, active = !quiet, visible = !document.hidden, inViewport = !quiet;
  let frameHandle = 0, width = 1, height = 1, time = 0, lastTime = 0;
  let sceneObserver = null;
  let progress = 0, pointer = [0.72, 0.5], pointerTarget = [0.72, 0.5];
  let listening = 0, listeningTarget = 0, count = 0, countTarget = 0;
  const wave = new Float32Array(WAVE_BINS);
  const fft = new Float32Array(WAVE_BINS);
  const api = {
    ready: Promise.resolve(),
    fallback: false,
    isReducedMotion: prefersReduced,
    setProgress(value) { progress = Math.max(0, Math.min(1, Number(value) || 0)); requestDraw(); },
    setWaveform(values) { compress(values, wave, false); requestDraw(); },
    setSpectrum(values) {
      if (!values?.length) { fft.fill(0); requestDraw(); return; }
      const normalized = Float32Array.from(values, value => Math.max(0, Math.min(1, Number(value) / 255)));
      compress(normalized, fft, false);
      requestDraw();
    },
    setListening(on) { listeningTarget = on ? 1 : 0; requestDraw(); },
    resolveCount(value) {
      const number = Number(value);
      countTarget = Number.isInteger(number) && number >= 1 && number <= 32 ? number : 0;
      if (prefersReduced) count = countTarget;
      requestDraw();
    },
    clearCount() { count = 0; countTarget = 0; listeningTarget = 0; requestDraw(); },
    destroy() {
      active = false;
      cancelAnimationFrame(frameHandle);
      if (gl) {
        if (program) gl.deleteProgram(program);
        if (vao) gl.deleteVertexArray(vao);
      }
      sceneObserver?.disconnect();
      removeEventListeners();
    }
  };

  function setFallback(on) {
    api.fallback = true;
    canvas.hidden = true;
    fallback.hidden = !on;
    document.body.dataset.sceneFallback = 'true';
  }

  function initGL() {
    if (quiet || prefersReduced || !canvas) {
      setFallback(!quiet);
      return false;
    }
    try {
      gl = canvas.getContext('webgl2', { alpha: false, antialias: false, depth: false, stencil: false, powerPreference: constrained ? 'low-power' : 'default' });
      if (!gl) throw new Error('WebGL2 is unavailable.');
      const vertex = compile(gl, gl.VERTEX_SHADER, vertexSource);
      const fragment = compile(gl, gl.FRAGMENT_SHADER, fragmentSource);
      program = gl.createProgram();
      gl.attachShader(program, vertex);
      gl.attachShader(program, fragment);
      gl.linkProgram(program);
      gl.deleteShader(vertex);
      gl.deleteShader(fragment);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) || 'Shader program could not link.');
      vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      canvas.hidden = false;
      fallback.hidden = true;
      resize();
      return true;
    } catch (error) {
      setFallback(!quiet);
      return false;
    }
  }

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, pixelRatioCap, 2);
    const bounds = stage?.getBoundingClientRect();
    width = Math.max(1, Math.floor((bounds?.width || innerWidth) * dpr));
    height = Math.max(1, Math.floor((bounds?.height || innerHeight) * dpr));
    if (gl && !api.fallback) {
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
        gl.viewport(0, 0, width, height);
      }
    } else if (fallback && !fallback.hidden) {
      if (fallback.width !== width || fallback.height !== height) {
        fallback.width = width;
        fallback.height = height;
      }
    }
    requestDraw();
  }

  function uniform(name) { return gl.getUniformLocation(program, name); }

  function renderGL(delta) {
    if (!gl || api.fallback || !program) return;
    const alpha = Math.min(1, delta * 5);
    pointer[0] += (pointerTarget[0] - pointer[0]) * alpha;
    pointer[1] += (pointerTarget[1] - pointer[1]) * alpha;
    listening += (listeningTarget - listening) * alpha;
    count += (countTarget - count) * Math.min(1, delta * 2.8);
    gl.useProgram(program);
    gl.bindVertexArray(vao);
    gl.uniform2f(uniform('uResolution'), width, height);
    gl.uniform1f(uniform('uTime'), time);
    gl.uniform1f(uniform('uProgress'), progress);
    gl.uniform1f(uniform('uPage'), page === 'analyze' ? 1 : (page === 'how' ? 2 : 0));
    gl.uniform1f(uniform('uLight'), prefersLight() ? 1 : 0);
    gl.uniform1f(uniform('uListening'), listening);
    gl.uniform1f(uniform('uCount'), count);
    gl.uniform2f(uniform('uPointer'), pointer[0], pointer[1]);
    gl.uniform1fv(uniform('uWave[0]'), wave);
    gl.uniform1fv(uniform('uFft[0]'), fft);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  function renderFallback() {
    if (!fallback || fallback.hidden || quiet) return;
    const ctx = fallback.getContext('2d');
    if (!ctx) return;
    const dpr = Math.min(window.devicePixelRatio || 1, pixelRatioCap, 2);
    const w = fallback.width, h = fallback.height, light = prefersLight();
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = light ? '#efeee9' : '#07080b';
    ctx.fillRect(0, 0, w, h);
    const laneColors = ['#72DDF7', '#FF7C66', '#C0E878', '#E49BFF'];
    const progressMix = page === 'analyze' ? 0 : progress;
    const accent = light ? '#A94424' : '#F1845D';
    ctx.save();
    ctx.globalAlpha = light ? .38 : .56;
    if (page === 'analyze') {
      ctx.strokeStyle = accent;
      ctx.lineWidth = Math.max(1, 1.4 * dpr);
      ctx.beginPath();
      for (let x = 0; x <= 220; x++) {
        const y = .5 + wave[Math.min(63, Math.floor(x / 220 * 64))] * .18;
        const px = x / 220 * w, py = (1 - y) * h;
        x ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      }
      ctx.stroke();
      ctx.beginPath();
      for (let i = 0; i < 64; i++) {
        const x = (i / 63) * w, y = h * (.85 - fft[i] * .25);
        ctx.moveTo(x, h * .86); ctx.lineTo(x, y);
      }
      ctx.strokeStyle = light ? '#6B6A66' : '#A6A49F'; ctx.globalAlpha *= .45; ctx.stroke();
    } else {
      for (let lane = 0; lane < 4; lane++) {
        ctx.beginPath();
        for (let x = 0; x <= 220; x++) {
          const u = x / 220;
          const merge = Math.max(0, Math.min(1, (progressMix - .14) / .26));
          const base = .25 + lane * .12;
          const y = base * (1 - merge) + (.51 + lane * .007) * merge
            + Math.sin(u * (5.2 + lane * 1.6) + time * .2 + lane * 1.9) * (.026 + lane * .004) * (1 - merge * .65);
          const px = u * w, py = (1 - y) * h;
          x ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
        }
        ctx.strokeStyle = laneColors[lane]; ctx.lineWidth = Math.max(1, (1.2 + lane * .12) * dpr); ctx.globalAlpha = light ? .32 : .5; ctx.stroke();
      }
      if (progressMix > .35 && progressMix < .9) {
        const horizon = h * .65;
        for (let j = 0; j < 20; j++) {
          const depth = j / 20;
          ctx.beginPath();
          for (let x = 0; x <= w; x += Math.max(6, 9 * dpr)) {
            const y = horizon + depth * h * .27 + Math.sin(x / w * 15 + time * .08 + j) * depth * h * .025;
            x ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
          }
          ctx.strokeStyle = accent; ctx.globalAlpha = (light ? .08 : .13) * (1 - depth * .4); ctx.lineWidth = 1; ctx.stroke();
        }
      }
    }
    ctx.restore();
    ctx.globalAlpha = 1;
  }

  function frame(now) {
    frameHandle = 0;
    if (!active || !visible || !inViewport || quiet) return;
    const delta = lastTime ? Math.min(.05, (now - lastTime) / 1000) : 1 / 60;
    lastTime = now;
    if (!prefersReduced) time += delta;
    renderGL(delta);
    renderFallback();
    const needsAnimation = !prefersReduced && (page === 'home' || page === 'how' || listening > .01 || count !== countTarget);
    if (needsAnimation) frameHandle = requestAnimationFrame(frame);
  }

  function requestDraw() {
    if (quiet || !active || !visible || !inViewport) return;
    if (!frameHandle) frameHandle = requestAnimationFrame(frame);
  }

  function onPointer(event) {
    if (matchMedia('(pointer: coarse)').matches) return;
    const bounds = stage?.getBoundingClientRect();
    const localX = bounds ? (event.clientX - bounds.left) / Math.max(1, bounds.width) : event.clientX / Math.max(1, innerWidth);
    const localY = bounds ? (event.clientY - bounds.top) / Math.max(1, bounds.height) : event.clientY / Math.max(1, innerHeight);
    pointerTarget = [Math.max(0, Math.min(1, localX)), 1 - Math.max(0, Math.min(1, localY))];
    requestDraw();
  }

  function onVisibility() {
    visible = !document.hidden;
    if (visible) { lastTime = 0; requestDraw(); }
    else { cancelAnimationFrame(frameHandle); frameHandle = 0; }
  }
  function onThemeChange() { requestDraw(); }

  function onContextLost(event) {
    event.preventDefault();
    gl = null;
    setFallback(true);
    resize();
    requestDraw();
  }

  function removeEventListeners() {
    removeEventListener('resize', resize);
    removeEventListener('pointermove', onPointer);
    document.removeEventListener('visibilitychange', onVisibility);
    removeEventListener('sanada:themechange', onThemeChange);
    canvas?.removeEventListener('webglcontextlost', onContextLost);
  }

  const hasGL = initGL();
  if (!hasGL && !quiet) { setFallback(true); resize(); }
  if (quiet) {
    canvas.hidden = true;
    fallback.hidden = true;
    api.fallback = true;
  }
  addEventListener('resize', resize, { passive: true });
  addEventListener('pointermove', onPointer, { passive: true });
  document.addEventListener('visibilitychange', onVisibility);
  addEventListener('sanada:themechange', onThemeChange);
  canvas?.addEventListener('webglcontextlost', onContextLost);
  if (!quiet && 'IntersectionObserver' in window) {
    sceneObserver = new IntersectionObserver(([entry]) => {
      inViewport = Boolean(entry?.isIntersecting);
      if (inViewport) requestDraw();
      else { cancelAnimationFrame(frameHandle); frameHandle = 0; }
    }, { threshold: 0.01 });
    const observed = stage || document.querySelector('main');
    if (observed) sceneObserver.observe(observed);
  }
  if (prefersReduced) renderFallback();
  else requestDraw();
  api.ready = new Promise(resolve => requestAnimationFrame(() => {
    resolve({ fallback: api.fallback, webgl: Boolean(gl), reducedMotion: prefersReduced });
    dispatchEvent(new CustomEvent('sanada:scene-ready', { detail: { fallback: api.fallback, webgl: Boolean(gl) } }));
  }));
  return api;
}
