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
  const tops = (p.top_predictions || []).map(x => `
    <div class="prediction-row"><span>${x.sources} source${x.sources === 1 ? "" : "s"}</span>
    <div class="bar"><span style="width:${Math.max(0, Math.min(100, x.probability * 100))}%"></span></div>
    <strong>${(x.probability * 100).toFixed(1)}%</strong></div>`).join("");
  resultState.innerHTML = `
    <div class="result-count">
      <div class="count-ring"><strong>${p.estimated_sources}</strong><span>SOURCES</span></div>
      <div><h4>Estimated source count</h4><p>Predicted from the uploaded mono mixture using a CNN over log-mel spectrogram features.</p></div>
    </div>
    <div class="metric-grid">
      <div class="metric"><label>Top-class confidence</label><strong>${(p.confidence * 100).toFixed(1)}%</strong></div>
      <div class="metric"><label>Audio duration</label><strong>${a.duration_seconds} s</strong></div>
      <div class="metric"><label>Sample rate used</label><strong>${a.processed_sample_rate} Hz</strong></div>
      <div class="metric"><label>RMS energy</label><strong>${a.rms_energy}</strong></div>
    </div>
    <div class="top-predictions"><h5>Other likely counts</h5>${tops}</div>
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
