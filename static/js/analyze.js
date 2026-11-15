import { formatTime } from './ui.js';

const EXTENSIONS = new Set(['wav', 'flac', 'mp3', 'ogg', 'm4a']);
const MAX_BYTES = 40 * 1024 * 1024;
const CLIP_SECONDS = 8;
const $ = (selector, root = document) => root.querySelector(selector);

export function initAnalyzer({ scene, ui }) {
  const { toast, history } = ui;
  const elements = {
    dropzone: $('#dropzone'), input: $('#audioInput'), dropTitle: $('#dropTitle'),
    fileMeta: $('#fileMeta'), fileName: $('#fileName'), fileDetails: $('#fileDetails'),
    remove: $('#removeFile'), waveformCard: $('#waveformCard'), waveform: $('#waveform'),
    waveformWrap: $('#waveformWrap'), marker: $('#analysisWindow'), audio: $('#audioPlayer'),
    play: $('#playAudio'), currentTime: $('#timeNow'), totalTime: $('#timeTotal'),
    fftStatus: $('#fftStatus'), fftCanvas: $('#fftCanvas'), analyze: $('#analyzeButton'),
    progress: $('#uploadProgress'), progressbar: $('.upload-progress'), loadingTitle: $('#loadingTitle'),
    loadingMessage: $('#loadingMessage'), cancel: $('#cancelAnalysis'), retry: $('#retryAnalysis'),
    errorMessage: $('#errorMessage'), untrainedMessage: $('#untrainedMessage'),
    count: $('#resultCount'), countWord: $('#resultCountWord'), confidence: $('#resultConfidence'),
    confidenceRing: $('#confidenceTag'), countOrbs: $('#countOrbs')
  };
  if (!elements.dropzone || !elements.audio) return;

  const states = {
    empty: $('#resultEmpty'), loading: $('#resultLoading'), error: $('#resultError'),
    untrained: $('#resultUntrained'), complete: $('#resultComplete')
  };
  let file = null, objectUrl = null, duration = 0, peaks = null, analysisWindow = null;
  let activeRequest = null, requestId = 0, busy = false, playbackFrame = 0;
  let audioContext = null, analyser = null, mediaSource = null;
  let fftData = null;
  let dragDepth = 0;

  function showState(name) {
    Object.entries(states).forEach(([key, node]) => { if (node) node.hidden = key !== name; });
  }

  function formatBytes(size) {
    if (size < 1024 * 1024) return `${Math.max(1, Math.round(size / 1024))} KB`;
    return `${(size / (1024 * 1024)).toFixed(2)} MB`;
  }

  function setWindowMarker() {
    if (!duration || !elements.marker) return;
    const left = analysisWindow?.start || 0;
    const width = analysisWindow ? analysisWindow.end - analysisWindow.start : Math.min(CLIP_SECONDS, duration);
    elements.marker.style.left = `${Math.max(0, Math.min(100, left / duration * 100))}%`;
    elements.marker.style.width = `${Math.max(0, Math.min(100, width / duration * 100))}%`;
    const end = Math.min(duration, left + width);
    const label = duration <= CLIP_SECONDS
      ? `Entire ${formatTime(duration)} recording analyzed, padded to eight seconds.`
      : `Analyzed window from ${formatTime(left)} to ${formatTime(end)}.`;
    elements.marker.setAttribute('aria-label', label);
  }

  function updateFileDetails() {
    if (!file) return;
    const details = [formatBytes(file.size)];
    if (duration > 0) details.push(formatTime(duration));
    if (duration > CLIP_SECONDS) details.push('center 8 seconds analyzed');
    else if (duration > 0) details.push('padded to 8 seconds');
    elements.fileDetails.textContent = details.join(' · ');
    elements.totalTime.textContent = formatTime(duration);
    if (duration > CLIP_SECONDS) {
      const start = (duration - CLIP_SECONDS) / 2;
      analysisWindow = { start, end: start + CLIP_SECONDS };
    } else {
      analysisWindow = { start: 0, end: duration || CLIP_SECONDS };
    }
    setWindowMarker();
  }

  function releaseAudioResources() {
    cancelAnimationFrame(playbackFrame);
    playbackFrame = 0;
    if (audioContext?.state === 'running') audioContext.suspend().catch(() => {});
    scene.setSpectrum(null);
    scene.setWaveform(null);
    scene.setListening(false);
    elements.fftStatus.textContent = 'FFT idle';
  }

  function clearCurrentFile() {
    requestId += 1;
    if (activeRequest) {
      activeRequest.abort();
      activeRequest = null;
    }
    busy = false;
    elements.analyze.disabled = true;
    elements.input.value = '';
    if (elements.audio) {
      elements.audio.pause();
      elements.audio.removeAttribute('src');
      elements.audio.load();
    }
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = null;
    releaseAudioResources();
    file = null;
    duration = 0;
    peaks = null;
    analysisWindow = null;
    elements.fileMeta.hidden = true;
    elements.waveformCard.hidden = true;
    elements.play.disabled = true;
    elements.dropTitle.textContent = 'Drop a recording here';
    elements.currentTime.textContent = '0:00';
    elements.totalTime.textContent = '0:00';
    elements.waveform.setAttribute('aria-valuenow', '0');
    elements.waveform.setAttribute('aria-valuetext', 'No recording loaded');
    showState('empty');
    drawWaveform();
    drawSpectrum();
  }

  function validateFile(candidate) {
    if (!candidate) return false;
    const extension = candidate.name.split('.').pop()?.toLowerCase() || '';
    if (!EXTENSIONS.has(extension)) {
      toast('Unsupported audio type. Choose WAV, FLAC, MP3, OGG or M4A.', 'error');
      return false;
    }
    if (candidate.size <= 0) {
      toast('That audio file is empty.', 'error');
      return false;
    }
    if (candidate.size > MAX_BYTES) {
      toast('That file is larger than 40 MB. Choose a smaller recording.', 'error');
      return false;
    }
    return true;
  }

  async function setFile(candidate) {
    if (!validateFile(candidate)) return;
    clearCurrentFile();
    file = candidate;
    objectUrl = URL.createObjectURL(candidate);
    elements.audio.preload = 'metadata';
    elements.audio.src = objectUrl;
    elements.fileName.textContent = candidate.name;
    elements.fileDetails.textContent = formatBytes(candidate.size);
    elements.dropTitle.textContent = 'Recording ready';
    elements.fileMeta.hidden = false;
    elements.waveformCard.hidden = false;
    elements.play.disabled = false;
    elements.analyze.disabled = false;
    showState('empty');
    await decodeWaveform(candidate);
    if (candidate !== file) return;
    updateFileDetails();
    drawWaveform();
  }

  function onMetadata() {
    if (!file || !Number.isFinite(elements.audio.duration)) return;
    duration = elements.audio.duration;
    updateFileDetails();
    drawWaveform();
  }

  async function decodeWaveform(candidate) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) {
      toast('This browser cannot create a waveform preview. You can still send the audio for analysis.');
      return;
    }
    let decoder;
    try {
      decoder = new AudioContextClass();
      const decoded = await decoder.decodeAudioData(await candidate.arrayBuffer());
      if (candidate !== file) return;
      duration = decoded.duration;
      const channels = Array.from({ length: decoded.numberOfChannels }, (_, channel) => decoded.getChannelData(channel));
      const sampleCount = decoded.length;
      const barCount = 1600;
      const step = Math.max(1, Math.floor(sampleCount / barCount));
      const values = new Float32Array(barCount);
      let peak = 0;
      for (let bar = 0; bar < barCount; bar++) {
        let maximum = 0;
        const end = Math.min(sampleCount, (bar + 1) * step);
        for (let index = bar * step; index < end; index++) {
          let monoSample = 0;
          channels.forEach(channel => { monoSample += channel[index] || 0; });
          maximum = Math.max(maximum, Math.abs(monoSample / Math.max(1, channels.length)));
        }
        values[bar] = maximum;
        peak = Math.max(peak, maximum);
      }
      if (peak > 0) for (let index = 0; index < values.length; index++) values[index] /= peak;
      peaks = values;
      scene.setWaveform(values);
      elements.fftStatus.textContent = 'FFT ready on playback';
    } catch {
      peaks = null;
      toast('The waveform preview is unavailable for this file. You can still analyze it.');
    } finally {
      if (decoder && decoder.state !== 'closed') decoder.close().catch(() => {});
    }
  }

  function drawWaveform() {
    const canvas = elements.waveform;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(rect.width * dpr));
    const height = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#F1845D';
    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = 'rgba(220, 219, 230, .12)';
    ctx.lineWidth = Math.max(1, dpr);
    ctx.beginPath(); ctx.moveTo(0, height / 2); ctx.lineTo(width, height / 2); ctx.stroke();
    if (!peaks?.length) {
      ctx.strokeStyle = 'rgba(220, 219, 230, .38)';
      ctx.beginPath(); ctx.moveTo(0, height / 2); ctx.lineTo(width, height / 2); ctx.stroke();
      return;
    }
    const bars = Math.max(1, Math.floor(width / (4 * dpr)));
    const barWidth = width / bars;
    const now = duration ? elements.audio.currentTime / duration : 0;
    for (let index = 0; index < bars; index++) {
      const bucket = Math.min(peaks.length - 1, Math.floor(index / bars * peaks.length));
      const amplitude = peaks[bucket];
      const barHeight = Math.max(2 * dpr, amplitude * height * .72);
      const progress = (index + .5) / bars;
      ctx.fillStyle = progress <= now ? accent : 'rgba(170, 169, 164, .58)';
      const x = index * barWidth + barWidth * .2;
      const y = (height - barHeight) / 2;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(x, y, Math.max(1, barWidth * .58), barHeight, Math.min(2 * dpr, barWidth * .25));
      else ctx.rect(x, y, Math.max(1, barWidth * .58), barHeight);
      ctx.fill();
    }
  }

  function drawSpectrum(data = fftData) {
    const canvas = elements.fftCanvas;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(rect.width * dpr));
    const height = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#F1845D';
    ctx.clearRect(0, 0, width, height);
    if (!data?.length) {
      ctx.strokeStyle = 'rgba(220, 219, 230, .15)';
      ctx.beginPath(); ctx.moveTo(0, height - 1); ctx.lineTo(width, height - 1); ctx.stroke();
      return;
    }
    const bars = 72;
    const gap = Math.max(1, dpr * 2);
    const barWidth = Math.max(1, (width - gap * (bars - 1)) / bars);
    for (let index = 0; index < bars; index++) {
      const bucket = Math.min(data.length - 1, Math.floor(index / bars * data.length));
      const amplitude = data[bucket] / 255;
      const barHeight = Math.max(1, amplitude * height * .9);
      ctx.fillStyle = index < bars * .62 ? accent : 'rgba(170, 169, 164, .9)';
      ctx.globalAlpha = .35 + amplitude * .65;
      ctx.fillRect(index * (barWidth + gap), height - barHeight, barWidth, barHeight);
    }
    ctx.globalAlpha = 1;
  }

  function updatePlaybackPosition() {
    if (!file) return;
    const current = elements.audio.currentTime || 0;
    elements.currentTime.textContent = formatTime(current);
    const percent = duration ? current / duration * 100 : 0;
    elements.waveform.setAttribute('aria-valuenow', String(Math.round(percent)));
    elements.waveform.setAttribute('aria-valuetext', `${formatTime(current)} of ${formatTime(duration)}`);
    drawWaveform();
  }

  function onPlaybackEnded() {
    elements.play.innerHTML = '<svg class="button-icon"><use href="#icon-play"/></svg><span>Play</span>';
    scene.setSpectrum(null);
    elements.fftStatus.textContent = 'FFT paused';
  }

  function onPause() {
    cancelAnimationFrame(playbackFrame);
    playbackFrame = 0;
    scene.setListening(false);
  }

  function readSpectrum() {
    playbackFrame = 0;
    if (!analyser || elements.audio.paused) return;
    analyser.getByteFrequencyData(fftData);
    scene.setSpectrum(fftData);
    drawSpectrum(fftData);
    elements.fftStatus.textContent = 'Live FFT';
    playbackFrame = requestAnimationFrame(readSpectrum);
  }

  async function ensureAnalyser() {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) throw new Error('Live audio visualization is not supported in this browser.');
    if (!audioContext || audioContext.state === 'closed') {
      audioContext = new AudioContextClass();
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = .78;
      mediaSource = audioContext.createMediaElementSource(elements.audio);
      mediaSource.connect(analyser);
      analyser.connect(audioContext.destination);
      fftData = new Uint8Array(analyser.frequencyBinCount);
    }
    if (audioContext.state === 'suspended') await audioContext.resume();
  }

  async function togglePlayback() {
    if (!file) return;
    if (elements.audio.paused) {
      try {
        await ensureAnalyser();
        await elements.audio.play();
        elements.play.innerHTML = '<svg class="button-icon"><use href="#icon-pause"/></svg><span>Pause</span>';
        elements.fftStatus.textContent = 'Live FFT';
        playbackFrame = requestAnimationFrame(readSpectrum);
      } catch (error) {
        toast(error.message || 'This browser could not play the recording.', 'error');
      }
    } else {
      elements.audio.pause();
      elements.play.innerHTML = '<svg class="button-icon"><use href="#icon-play"/></svg><span>Play</span>';
      elements.fftStatus.textContent = 'FFT paused';
      scene.setSpectrum(null);
      drawSpectrum(null);
    }
  }

  function seekFromEvent(event) {
    if (!duration || !file) return;
    const rect = elements.waveform.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    elements.audio.currentTime = ratio * duration;
    updatePlaybackPosition();
  }

  function seekByKeyboard(event) {
    if (!duration || !file) return;
    const step = Math.max(1, duration * .01);
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight' || event.key === 'Home' || event.key === 'End') {
      event.preventDefault();
      if (event.key === 'Home') elements.audio.currentTime = 0;
      else if (event.key === 'End') elements.audio.currentTime = duration;
      else elements.audio.currentTime = Math.max(0, Math.min(duration, elements.audio.currentTime + (event.key === 'ArrowRight' ? step : -step)));
      updatePlaybackPosition();
    }
  }

  function generateSynthetic(sourceCount) {
    const sampleRate = 16000;
    const length = sampleRate * CLIP_SECONDS;
    const samples = new Float32Array(length);
    const baseFrequencies = [128, 190, 256, 320];
    for (let source = 0; source < sourceCount; source++) {
      const frequency = baseFrequencies[source % baseFrequencies.length];
      const phase = Math.random() * Math.PI * 2;
      const pulseRate = 1.7 + source * .47;
      for (let index = 0; index < length; index++) {
        const t = index / sampleRate;
        const envelope = Math.pow(Math.max(0, Math.sin(2 * Math.PI * pulseRate * t + phase)), .65);
        const overtone = Math.sin(2 * Math.PI * frequency * t) + .34 * Math.sin(4 * Math.PI * frequency * t);
        samples[index] += envelope * overtone * .18;
      }
    }
    const buffer = new ArrayBuffer(44 + length * 2);
    const view = new DataView(buffer);
    const writeTag = (offset, text) => [...text].forEach((char, index) => view.setUint8(offset + index, char.charCodeAt(0)));
    writeTag(0, 'RIFF'); view.setUint32(4, 36 + length * 2, true);
    writeTag(8, 'WAVEfmt '); view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true); view.setUint16(34, 16, true);
    writeTag(36, 'data'); view.setUint32(40, length * 2, true);
    for (let index = 0; index < length; index++) {
      const value = Math.max(-1, Math.min(1, samples[index]));
      view.setInt16(44 + index * 2, value * 32767, true);
    }
    return new File([buffer], `synthetic-${sourceCount}-source-mixture.wav`, { type: 'audio/wav' });
  }

  function setBusy(on) {
    busy = on;
    elements.analyze.disabled = on || !file;
    elements.cancel.disabled = !on;
  }

  function setProgress(value) {
    const rounded = Math.max(0, Math.min(100, Math.round(value)));
    elements.progress.style.width = `${rounded}%`;
    elements.progressbar.setAttribute('aria-valuenow', String(rounded));
  }

  function startAnalysis() {
    if (!file || busy) return;
    setBusy(true);
    showState('loading');
    scene.setListening(true);
    scene.clearCount();
    elements.loadingTitle.textContent = 'Sending audio…';
    elements.loadingMessage.textContent = 'Upload progress follows the browser’s transfer events.';
    elements.progressbar.classList.remove('is-indeterminate');
    setProgress(0);
    const currentRequest = ++requestId;
    const form = new FormData();
    form.append('audio', file);
    const request = new XMLHttpRequest();
    activeRequest = request;
    request.open('POST', '/api/predict');
    request.timeout = 180000;
    request.upload.addEventListener('progress', event => {
      if (currentRequest !== requestId || !event.lengthComputable) return;
      const progress = event.loaded / event.total * 100;
      setProgress(progress);
      elements.loadingTitle.textContent = `Uploading ${Math.round(progress)}%`;
    });
    request.upload.addEventListener('load', () => {
      if (currentRequest !== requestId) return;
      elements.loadingTitle.textContent = 'Analyzing the clip…';
      elements.loadingMessage.textContent = 'The server is processing the audio. It does not report model progress.';
      elements.progressbar.classList.add('is-indeterminate');
    });
    request.addEventListener('load', () => {
      if (currentRequest !== requestId) return;
      activeRequest = null;
      setBusy(false);
      let data = {};
      try { data = JSON.parse(request.responseText); } catch {}
      if (request.status < 200 || request.status >= 300) {
        showError(data.error || `The server returned HTTP ${request.status}.`);
        return;
      }
      if (data.trained !== true || !data.prediction) {
        scene.setListening(false);
        elements.untrainedMessage.textContent = data.message || 'Audio details were measured, but the server has no compatible trained checkpoint.';
        showState('untrained');
        return;
      }
      const count = Number(data.prediction.estimated_sources);
      if (!Number.isInteger(count) || count < Number(data.prediction.min_sources) || count > Number(data.prediction.max_sources)) {
        showError('The server response did not contain a valid source count.');
        return;
      }
      showResult(data);
    });
    request.addEventListener('error', () => {
      if (currentRequest !== requestId) return;
      activeRequest = null; setBusy(false);
      showError('Could not reach the local server. Check that python app.py is running.');
    });
    request.addEventListener('timeout', () => {
      if (currentRequest !== requestId) return;
      activeRequest = null; setBusy(false);
      showError('The request timed out. Try a smaller file or retry the analysis.');
    });
    request.addEventListener('abort', () => {
      if (currentRequest !== requestId) return;
      activeRequest = null; setBusy(false);
      showState('empty');
    });
    request.send(form);
  }

  function showError(message) {
    scene.setListening(false);
    $('#errorMessage').textContent = message;
    showState('error');
  }

  function showResult(data) {
    const prediction = data.prediction;
    const count = Number(prediction.estimated_sources);
    const confidence = Math.max(0, Math.min(100, Number(prediction.confidence) || 0));
    const audio = data.audio || {};
    duration = Number(audio.original_duration_seconds) || duration;
    const start = Math.max(0, Number(audio.crop_start_seconds) || 0);
    const clipLength = Math.min(CLIP_SECONDS, duration || CLIP_SECONDS);
    analysisWindow = { start, end: start + clipLength };
    updateFileDetails();
    setWindowMarker();
    elements.count.textContent = String(count);
    elements.countWord.textContent = count === 1 ? 'source' : 'sources';
    elements.confidence.textContent = `${confidence.toFixed(1)}%`;
    elements.confidenceRing.textContent = `Model confidence · ${confidence.toFixed(1)}%`;
    elements.confidenceRing.style.setProperty('--confidence', `${confidence}%`);
    elements.confidenceRing.style.setProperty('--confidence-target', `${confidence}%`);
    elements.confidenceRing.setAttribute('aria-label', `Model confidence ${confidence.toFixed(1)} percent`);
    elements.countOrbs.replaceChildren();
    for (let index = 0; index < count; index++) {
      const orb = document.createElement('span');
      orb.style.setProperty('--orb-index', String(index));
      elements.countOrbs.append(orb);
    }
    showState('complete');
    scene.setListening(false);
    scene.resolveCount(count);
    const timeline = Array.isArray(prediction.timeline) ? prediction.timeline : [];
    history.add({
      id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`,
      filename: audio.filename || file?.name || 'recording',
      ts: Date.now(),
      count,
      conf: Number((confidence / 100).toFixed(3)),
      duration: Number(audio.original_duration_seconds) || duration,
      counts: timeline.map(item => Number(item.count)).filter(Number.isSafeInteger)
    });
    dispatchEvent(new CustomEvent('sanada:history-change'));
  }

  function cancelAnalysis() {
    if (!busy) return;
    requestId += 1;
    activeRequest?.abort();
    activeRequest = null;
    setBusy(false);
    scene.setListening(false);
    showState('empty');
    toast('Analysis canceled.');
  }

  elements.audio.addEventListener('loadedmetadata', onMetadata);
  elements.audio.addEventListener('timeupdate', updatePlaybackPosition);
  elements.audio.addEventListener('ended', onPlaybackEnded);
  elements.audio.addEventListener('pause', onPause);
  elements.input.addEventListener('change', () => {
    const candidate = elements.input.files?.[0];
    elements.input.value = '';
    if (candidate) setFile(candidate);
  });
  elements.dropzone.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      elements.input.click();
    }
  });
  elements.dropzone.addEventListener('dragenter', event => {
    event.preventDefault(); dragDepth += 1; elements.dropzone.classList.add('is-over');
  });
  elements.dropzone.addEventListener('dragover', event => event.preventDefault());
  elements.dropzone.addEventListener('dragleave', event => {
    event.preventDefault(); dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) elements.dropzone.classList.remove('is-over');
  });
  elements.dropzone.addEventListener('drop', event => {
    event.preventDefault(); dragDepth = 0; elements.dropzone.classList.remove('is-over');
    setFile(event.dataTransfer?.files?.[0]);
  });
  elements.remove.addEventListener('click', clearCurrentFile);
  elements.play.addEventListener('click', togglePlayback);
  elements.waveform.addEventListener('pointerdown', seekFromEvent);
  elements.waveform.addEventListener('pointermove', event => { if (event.buttons & 1) seekFromEvent(event); });
  elements.waveform.addEventListener('keydown', seekByKeyboard);
  elements.analyze.addEventListener('click', startAnalysis);
  elements.cancel.addEventListener('click', cancelAnalysis);
  elements.retry.addEventListener('click', startAnalysis);
  document.querySelectorAll('[data-synth]').forEach(button => button.addEventListener('click', () => {
    toast('Building a synthetic audio example.');
    setFile(generateSynthetic(Number(button.dataset.synth)));
  }));
  const resizeObserver = 'ResizeObserver' in window
    ? new ResizeObserver(() => { drawWaveform(); drawSpectrum(); })
    : null;
  resizeObserver?.observe(elements.waveform);
  resizeObserver?.observe(elements.fftCanvas);
  const resizeFallback = () => { drawWaveform(); drawSpectrum(); };
  if (!resizeObserver) addEventListener('resize', resizeFallback, { passive: true });
  addEventListener('sanada:themechange', resizeFallback);
  addEventListener('pagehide', () => {
    resizeObserver?.disconnect();
    removeEventListener('resize', resizeFallback);
    removeEventListener('sanada:themechange', resizeFallback);
    releaseAudioResources();
    if (audioContext && audioContext.state !== 'closed') audioContext.close().catch(() => {});
    if (objectUrl) URL.revokeObjectURL(objectUrl);
  }, { once: true });
  drawWaveform();
  drawSpectrum();
}
