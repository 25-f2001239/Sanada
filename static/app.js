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
<<<<<<< HEAD
let waveSurfer = null;
let waveRegions = null;

const SOURCE_COLORS = {
  1: "#3b82f6", // Blue
  2: "#10b981", // Green
  3: "#f59e0b", // Orange
  4: "#ef4444", // Red
  5: "#8b5cf6", // Purple
};
=======
>>>>>>> 08a62205b52c1c4eebefeaeee8f6e36d972cc3df

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}
<<<<<<< HEAD

=======
>>>>>>> 08a62205b52c1c4eebefeaeee8f6e36d972cc3df
function setFile(file) {
  if (!file) return;
  selectedFile = file;
  title.textContent = file.name;
  subtitle.textContent = `${formatBytes(file.size)} · Ready to analyze`;
  fileMeta.textContent = `Selected: ${file.name} (${formatBytes(file.size)})`;
  fileMeta.classList.remove("hidden");
  button.disabled = false;
}
<<<<<<< HEAD

=======
>>>>>>> 08a62205b52c1c4eebefeaeee8f6e36d972cc3df
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
<<<<<<< HEAD

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

  // Correct v7 plugin registration
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

=======
>>>>>>> 08a62205b52c1c4eebefeaeee8f6e36d972cc3df
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
<<<<<<< HEAD

  const p = data.prediction;

  // Since we changed the backend to return a timeline, we handle it here
  const estimatedCount = p.estimated_sources;

  resultState.innerHTML = `
    <div class="result-count">
      <div class="count-ring"><strong>${estimatedCount}</strong><span>SOURCES</span></div>
      <div><h4>Estimated source count</h4><p>Predicted from the uploaded mono mixture using a sliding-window CNN.</p></div>
    </div>
    <div class="metric-grid">
      <div class="metric"><label>Average Confidence</label><strong>${p.confidence ? (p.confidence * 100).toFixed(1) + "%" : "—"}</strong></div>
=======
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
>>>>>>> 08a62205b52c1c4eebefeaeee8f6e36d972cc3df
      <div class="metric"><label>Audio duration</label><strong>${a.duration_seconds} s</strong></div>
      <div class="metric"><label>Sample rate used</label><strong>${a.processed_sample_rate} Hz</strong></div>
      <div class="metric"><label>RMS energy</label><strong>${a.rms_energy}</strong></div>
    </div>
<<<<<<< HEAD
    <div class="top-predictions">
       <h5 style="margin-bottom:10px">Waveform Timeline Legend</h5>
       <div style="display:flex; gap:10px; flex-wrap:wrap">
         ${Object.entries(SOURCE_COLORS).map(([count, color]) => `
           <div style="display:flex; align-items:center; gap:5px; font-size:10px">
             <div style="width:10px; height:10px; background:${color}; border-radius:2px"></div> ${count} Source${count==1?'':'s'}
           </div>
         `).join("")}
       </div>
    </div>
    <div class="notice">${escapeHtml(data.message || "Model output is an estimate.")}</div>`;
  resultState.classList.remove("hidden");

  // Paint the regions on the waveform
  if (p.timeline && waveSurfer && waveRegions) {
    p.timeline.forEach(segment => {
      const color = SOURCE_COLORS[segment.count] || "#ccc";
      waveRegions.addRegion({
        start: segment.start,
        end: segment.end,
        color: color,
        drag: false,
        resize: false
      });
    });
  }
=======
    <div class="top-predictions"><h5>Other likely counts</h5>${tops}</div>
    <div class="notice">${escapeHtml(data.message || "Model output is an estimate.")}</div>`;
  resultState.classList.remove("hidden");
>>>>>>> 08a62205b52c1c4eebefeaeee8f6e36d972cc3df
}

form.addEventListener("submit", async e => {
  e.preventDefault();
  if (!selectedFile) return;
  emptyState.classList.add("hidden");
  resultState.classList.add("hidden");
  loadingState.classList.remove("hidden");
  button.disabled = true;
  button.querySelector("span").textContent = "Analyzing…";
<<<<<<< HEAD

  // Initialize waveform with the local file
  const fileUrl = URL.createObjectURL(selectedFile);
  await initWaveform(fileUrl);

=======
>>>>>>> 08a62205b52c1c4eebefeaeee8f6e36d972cc3df
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
