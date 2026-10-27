const form = document.getElementById("uploadForm");
const input = document.getElementById("audioInput");
const dropzone = document.getElementById("dropzone");
const title = document.getElementById("fileTitle");
const subtitle = document.getElementById("fileSubtitle");
const fileMeta = document.getElementById("fileMeta");
const button = document.getElementById("analyzeBtn");
const emptyState = document.getElementById("emptyState");
const loadingState = document.getElementById("loadingState");
const resultState = document.getElementById("resultState");
let selectedFile = null;
let waveSurfer = null;
let waveRegions = null;

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function setFile(file) {
  if (!file) return;
  selectedFile = file;
  title.textContent = file.name;
  subtitle.textContent = `${formatBytes(file.size)} · Ready to analyze`;
  fileMeta.textContent = `Selected: ${file.name} (${formatBytes(file.size)})`;
  fileMeta.classList.remove("hidden");
  button.disabled = false;
}

input.addEventListener("change", () => setFile(input.files[0]));
["dragenter", "dragover"].forEach(evt => dropzone.addEventListener(evt, e => {
  e.preventDefault(); dropzone.classList.add("dragover");
}));
["dragleave", "drop"].forEach(evt => dropzone.addEventListener(evt, e => {
  e.preventDefault(); dropzone.classList.remove("dragover");
}));
dropzone.addEventListener("drop", e => setFile(e.dataTransfer.files[0]));

function showNotice(message, error=false) {
  resultState.innerHTML = `<div class="notice ${error ? "error-notice" : ""}">${escapeHtml(message)}</div>`;
  resultState.classList.remove("hidden");
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

async function initWaveform(url) {
  if (waveSurfer) waveSurfer.destroy();

  waveSurfer = WaveSurfer.create({
    container: "#waveform",
    waveColor: "#a3b3aa",
    progressColor: "#1b765b",
    cursorColor: "#1b765b",
    height: 80,
    responsive: true,
  });

  // Initialize Regions plugin
  waveRegions = waveSurfer.registerPlugin(WaveSurfer.Regions.create());

  await waveSurfer.load(url);
  document.getElementById("wave-controls").classList.remove("hidden");

  const playPauseBtn = document.getElementById("playPauseBtn");
  playPauseBtn.addEventListener("click", () => waveSurfer.playPause());

  const curTime = document.getElementById("currentTime");
  const durTime = document.getElementById("durationTime");

  waveSurfer.on("ready", () => {
    durTime.textContent = formatTime(waveSurfer.getDuration());
  });

  waveSurfer.on("timeupdate", (currentTime) => {
    curTime.textContent = formatTime(currentTime);
  });
}

function formatTime(s) {
  const mins = Math.floor(s / 60);
  const secs = Math.floor(s % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

function showResult(data) {
  const a = data.audio || {};
  if (!data.trained || !data.prediction) {
    resultState.innerHTML = `
      <div class="result-count">
        <div class="count-ring"><strong>—</strong><span>COUNT</span></div>
        <div><h4>Model is not trained yet</h4><p>Your audio was loaded and basic signal statistics were computed. A source-count estimate will appear after model training.</p></div>
      </div>
      <div class="metric-grid">
        <div class="metric"><label>Duration</label><strong>${a.duration_seconds ?? "—"} s</strong></div>
        <div class="metric"><label>Input sample rate</label><strong>${a.original_sample_rate ?? "—"} Hz</strong></div>
        <div class="metric"><label>RMS energy</label><strong>${a.rms_energy ?? "—"}</strong></div>
        <div class="metric"><label>Peak amplitude</label><strong>${a.peak_amplitude ?? "—"}</strong></div>
      </div>
      <div class="notice">${escapeHtml(data.message || "Train the model with labelled mixture audio before interpreting source counts.")}<br><br>See README.md for the dataset format and training command.</div>`;
    resultState.classList.remove("hidden");
    return;
  }

  const p = data.prediction;
  const estimatedCount = p.estimated_sources;

  let avgConf = "—";
  if (p.timeline) {
    const sum = p.timeline.reduce((acc, cur) => acc + cur.confidence, 0);
    avgConf = ((sum / p.timeline.length) * 100).toFixed(1) + "%";
  } else if (p.confidence) {
    avgConf = (p.confidence * 100).toFixed(1) + "%";
  }

  resultState.innerHTML = `
    <div class="result-count">
      <div class="count-ring"><strong>${estimatedCount}</strong><span>SOURCES</span></div>
      <div><h4>Estimated source count</h4><p>Predicted from the uploaded mono mixture using a sliding-window CNN.</p></div>
    </div>
    <div class="metric-grid">
      <div class="metric"><label>Average Confidence</label><strong>${avgConf}</strong></div>
      <div class="metric"><label>Audio duration</label><strong>${a.duration_seconds} s</strong></div>
      <div class="metric"><label>Sample rate used</label><strong>${a.processed_sample_rate} Hz</strong></div>
      <div class="metric"><label>RMS energy</label><strong>${a.rms_energy}</strong></div>
    </div>
    <div class="notice">${escapeHtml(data.message || "Model output is an estimate.")}</div>`;
  resultState.classList.remove("hidden");
}

form.addEventListener("submit", async e => {
  e.preventDefault();
  if (!selectedFile) return;
  emptyState.classList.add("hidden");
  resultState.classList.add("hidden");
  loadingState.classList.remove("hidden");
  button.disabled = true;
  button.querySelector("span").textContent = "Analyzing…";

  const fileUrl = URL.createObjectURL(selectedFile);
  await initWaveform(fileUrl);

  const body = new FormData();
  body.append("audio", selectedFile);
  try {
    const response = await fetch("/api/predict", {method: "POST", body});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Analysis failed.");
    showResult(data);
  } catch (err) {
    showNotice(err.message || "Something went wrong. Check that the Flask server is running.", true);
  } finally {
    loadingState.classList.add("hidden");
    button.disabled = !selectedFile;
    button.querySelector("span").textContent = "Analyze mixture";
  }
});
