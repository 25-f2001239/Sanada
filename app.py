import os
from pathlib import Path
import numpy as np
import soundfile as sf
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from model import SourceCountCNN, load_audio, predict_source_count, MODEL_PATH, SAMPLE_RATE, CLIP_SECONDS

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
MAX_CONTENT_LENGTH = 40 * 1024 * 1024

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def model_status():
    return {
        "trained": MODEL_PATH.exists(),
        "model_name": "CNN log-mel source-count classifier",
        "sample_rate": SAMPLE_RATE,
        "clip_seconds": CLIP_SECONDS,
        "max_sources": int(os.getenv("MAX_SOURCES", "32")),
    }

@app.get("/")
def index():
    return render_template("index.html", status=model_status())

@app.get("/api/health")
def health():
    return jsonify({"ok": True, "model": model_status()})

@app.post("/api/predict")
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "Choose an audio file first."}), 400
    audio_file = request.files["audio"]
    if not audio_file.filename:
        return jsonify({"error": "Choose an audio file first."}), 400
    if not allowed_file(audio_file.filename):
        return jsonify({"error": "Unsupported file type. Upload WAV, FLAC, MP3, OGG, or M4A."}), 400

    safe_name = secure_filename(audio_file.filename)
    path = UPLOAD_DIR / safe_name
    audio_file.save(path)
    try:
        print(f"Processing file: {safe_name}...")
        waveform, sample_rate = load_audio(path)
        print(f"Audio loaded. Duration: {len(waveform)/SAMPLE_RATE:.2f}s")
        duration = len(waveform) / SAMPLE_RATE
        peak = float(np.max(np.abs(waveform))) if len(waveform) else 0.0
        rms = float(np.sqrt(np.mean(np.square(waveform)))) if len(waveform) else 0.0


        if not MODEL_PATH.exists():
            return jsonify({
                "trained": False,
                "status": "model_not_trained",
                "message": "The application is running, but no trained weights exist yet. Add your dataset and run train.py before using source-count predictions.",
                "audio": {
                    "filename": safe_name,
                    "duration_seconds": round(duration, 2),
                    "original_sample_rate": sample_rate,
                    "processed_sample_rate": SAMPLE_RATE,
                    "channels_processed": 1,
                    "peak_amplitude": round(peak, 4),
                    "rms_energy": round(rms, 4),
                    "model_window_seconds": CLIP_SECONDS,
                },
                "prediction": None
            })

        print("Running prediction...")
        result = predict_source_count(waveform)
        print(f"Prediction result: {result['estimated_sources']}")
        return jsonify({
            "trained": True,
            "status": "ok",
            "message": "Prediction completed. Treat it as an estimate; evaluate on held-out test audio before relying on it.",
            "audio": {
                "filename": safe_name,
                "duration_seconds": round(duration, 2),
                "original_sample_rate": sample_rate,
                "processed_sample_rate": SAMPLE_RATE,
                "channels_processed": 1,
                "peak_amplitude": round(peak, 4),
                "rms_energy": round(rms, 4),
                "model_window_seconds": CLIP_SECONDS,
            },
            "prediction": result
        })
    except Exception as exc:
        return jsonify({"error": f"Could not process this audio: {exc}"}), 500
    finally:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

if __name__ == "__main__":
    # debug=False prevents the server from restarting when files are uploaded to the uploads folder
    app.run(debug=False, host="127.0.0.1", port=5000)
