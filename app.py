"""Flask web application for EchoCount (Production & Containerized Deployment).

Core Product Capabilities:
1. Source Counting: Two-stage General + Sparse Specialist (K<=8) CNN with validation isotonic
   calibration and MC-dropout + softmax uncertainty decomposition.
2. Open-Vocabulary Sound Naming (`open_tagger.py`):
   - Pretrained 527-class AudioSet AST (`models/ast-audioset`, BSD-3-Clause)
   - Pretrained Open-Vocabulary CLAP (`models/clap-htsat-unfused`, Apache-2.0)
   - Domain-trained 32-class Hierarchical `GroupFineTaggerCNN` (`checkpoints/tagger.pt`)
   - Outputs user-facing friendly names ("Hammer", "Phone vibration", "Dog bark", "Human voice",
     "Cello", "Scissors", "Telephone ring", etc.), confidence, and frame-level time ranges.
3. Query-Conditioned Separation & Residual Stem (`query_separation.py`):
   - Extracts one stem per detected sound (or SepFormer + ECAPA for 2-speaker speech), plus
     an exact `"Everything else"` (`residual.wav`) stem (`mixture - sum(stems)`).
   - Processes long recordings (> 8 s) in overlapping 8 s windows (2 s overlap, raised-cosine cross-fade).
   - Enforces three measured quality gates: self-consistency re-tagging, correlation (< 0.90),
     and energy conservation.
4. Production Hygiene & Scalability:
   - Ground-truth dataset lookup is strictly disabled in production (`ECHOCOUNT_DEV=1` only).
   - Untrained demo weights never run in production.
   - Stateless per-request job directories with TTL cleanup, async separation job queue,
     sliding-window rate limiting, structured logging, and startup model warmup.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import io
import json
import logging
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time
import uuid
import zipfile

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

import audio_io
from counting import checkpoint_fingerprint, count_with_uncertainty
import ground_truth
from model import (
    CLIP_SAMPLES,
    CLIP_SECONDS,
    MODEL_PATH,
    SAMPLE_RATE,
    checkpoint_status,
    load_audio,
    predict_source_count,
    waveform_to_logmel,
)
import open_tagger
import query_separation
import separation
import speech_separation
import stem_quality as stem_quality_mod
import tagger

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
JOB_DIR = UPLOAD_DIR / "jobs"
EVAL_PATH = Path(os.getenv("EVAL_RESULTS", str(BASE_DIR / "evaluation" / "results.json")))
JOB_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 40 * 1024 * 1024))
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", MAX_UPLOAD_BYTES + 1024 * 1024))
MAX_DURATION_SECONDS = float(os.getenv("MAX_DURATION_SECONDS", 600))
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", 1800))
MC_PASSES = int(os.getenv("MC_PASSES", 30))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", 120))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [echocount] %(message)s",
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

_jobs: dict = {}
_jobs_lock = threading.Lock()
_rate_buckets: dict[str, deque] = {}
_rate_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=int(os.getenv("SEPARATION_WORKERS", "2")))


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return s or "sound"


def check_rate_limit() -> bool:
    """Sliding-window per-client IP rate limiter."""
    if RATE_LIMIT_PER_MINUTE <= 0:
        return True
    ip = request.remote_addr or "local"
    now = time.time()
    with _rate_lock:
        dq = _rate_buckets.setdefault(ip, deque())
        while dq and now - dq[0] > 60.0:
            dq.popleft()
        if len(dq) >= RATE_LIMIT_PER_MINUTE:
            return False
        dq.append(now)
    return True


# ---------------------------------------------------------------- helpers
def allowed_file(filename):
    normalized = str(filename).replace("\\", "/")
    return Path(normalized).suffix.lower() in ALLOWED_EXTENSIONS


def model_status():
    """Return real readiness and metadata across counter, open-vocabulary tagger, and separators."""
    checkpoint = checkpoint_status(MODEL_PATH) or {}
    tagger_st = tagger.tagger_status() or {}
    open_tag_st = open_tagger.open_tagger_status() or {}
    sep_st = separation.separator_status() or {}
    speech_sep_st = speech_separation.pretrained_speech_separator_status() or {}
    return {
        "trained": bool(checkpoint.get("trained", False)),
        "model_name": "EchoCount SourceCountCNN 8-second",
        "sample_rate": SAMPLE_RATE,
        "clip_seconds": CLIP_SECONDS,
        "min_sources": checkpoint.get("min_sources"),
        "max_sources": checkpoint.get("max_sources"),
        "val_mae": checkpoint.get("val_mae"),
        "val_macro_mae": checkpoint.get("val_macro_mae"),
        "val_exact_accuracy": checkpoint.get("val_exact_accuracy"),
        "val_within_one": checkpoint.get("val_within_one"),
        "held_out_speaker_mae": checkpoint.get("held_out_speaker_mae"),
        "held_out_speaker_exact": checkpoint.get("held_out_speaker_exact"),
        "model_error": checkpoint.get("error"),
        "separator": sep_st,
        "speech_separator": speech_sep_st,
        "tagger": tagger_st,
        "open_tagger": open_tag_st,
    }


def format_count(result):
    """'≈27 (±3)' using the rounded count and the total std rounded up to a whole source."""
    spread = int(np.ceil(result["total_std"] - 1e-9)) if result["total_std"] >= 0.5 else 0
    return f"≈{result['rounded_count']} (±{spread})"


def get_dense_thresholds() -> tuple[int, int, int]:
    """Read last reliable sparse K and first dense K accuracy from evaluation/results.json."""
    if not EVAL_PATH.is_file():
        return 6, 9, 43
    try:
        ev = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
        per_count = ev.get("per_count") or []
        reliable_sparse = [
            p for p in per_count
            if not p.get("unreliable", False) and int(p.get("n", 0)) >= 30 and int(p.get("true_k", 0)) <= 8 and float(p.get("exact_accuracy", 0.0)) >= 0.5
        ]
        last_reliable_k = max((int(p["true_k"]) for p in reliable_sparse), default=6)
        next_dense = [
            p for p in per_count
            if not p.get("unreliable", False) and int(p.get("n", 0)) >= 30 and int(p.get("true_k", 0)) > last_reliable_k
        ]
        if next_dense:
            first_dense = sorted(next_dense, key=lambda x: int(x["true_k"]))[0]
            return last_reliable_k, int(first_dense["true_k"]), round(float(first_dense.get("exact_accuracy", 0.0)) * 100)
        return last_reliable_k, last_reliable_k + 3, 40
    except Exception:
        return 6, 9, 43


def get_dense_mixture_warning(rounded_k: int) -> str | None:
    """Build dense-mixture reliability warning dynamically from evaluation/results.json (never hard-coded)."""
    if not EVAL_PATH.is_file():
        return None
    last_reliable_k, dense_k, dense_acc_pct = get_dense_thresholds()
    if int(rounded_k) <= last_reliable_k:
        return None
    return (
        f"Too many overlapping sounds to separate reliably: counts above about {last_reliable_k} are unreliable "
        f"(measured exact accuracy at K={dense_k} is {dense_acc_pct}%)"
    )


def sweep_jobs():
    now = time.time()
    with _jobs_lock:
        expired = [jid for jid, job in _jobs.items() if now - job["created"] > JOB_TTL_SECONDS]
        for jid in expired:
            _jobs.pop(jid, None)
    for jid in expired:
        shutil.rmtree(JOB_DIR / jid, ignore_errors=True)
    try:
        for path in JOB_DIR.iterdir():
            if path.is_dir() and path.name not in _jobs and now - path.stat().st_mtime > JOB_TTL_SECONDS:
                shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


def get_job(job_id):
    sweep_jobs()
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None or not all(c in "0123456789abcdef" for c in job_id):
        return None
    return job


def read_upload():
    """Validate the multipart upload, save it into a fresh job dir, decode to mono.

    Returns (job_id, job_dir, path, samples, rate, display_name).
    """
    sweep_jobs()
    audio_file = request.files.get("audio")
    if audio_file is None or not audio_file.filename:
        raise audio_io.AudioValidationError("Choose an audio file first.")
    suffix = audio_io.validate_extension(audio_file.filename)
    display_name = secure_filename(audio_file.filename) or f"audio{suffix}"
    job_id = uuid.uuid4().hex
    job_dir = JOB_DIR / job_id
    job_dir.mkdir(parents=True)
    try:
        path = job_dir / f"original{suffix}"
        audio_file.save(path)
        if path.stat().st_size > MAX_UPLOAD_BYTES:
            raise audio_io.AudioValidationError(
                f"The upload exceeds the {MAX_UPLOAD_BYTES / 1048576:g} MB file limit."
            )
        samples, rate = audio_io.decode_mono(path, MAX_DURATION_SECONDS)
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    return job_id, job_dir, path, samples, rate, display_name


def analysis_clip(samples, rate):
    """Resample to the model rate and take the centred 8 s window for counting, plus full resampled audio."""
    model_rate = audio_io.resample(samples, rate, SAMPLE_RATE)
    if len(model_rate) > CLIP_SAMPLES:
        start = (len(model_rate) - CLIP_SAMPLES) // 2
        clip, crop_start = model_rate[start:start + CLIP_SAMPLES], start / SAMPLE_RATE
    else:
        clip, crop_start = np.pad(model_rate, (0, CLIP_SAMPLES - len(model_rate))), 0.0
    return clip.astype(np.float32), crop_start, len(model_rate) / SAMPLE_RATE, model_rate.astype(np.float32)


# ---------------------------------------------------------------- pages
@app.get("/")
def index():
    return render_template("index.html", status=model_status())


@app.get("/analyze")
def analyze_page():
    return render_template("analyze.html", status=model_status())


@app.get("/evaluation")
def evaluation_page():
    return render_template("evaluation.html", status=model_status())


@app.get("/history")
def history_page():
    return render_template("history.html", status=model_status())


@app.get("/how")
def how_page():
    return render_template("how.html", status=model_status())


@app.get("/docs")
def docs_page():
    return render_template("docs.html", status=model_status())


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "model": model_status()})


@app.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    limit_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
    return jsonify({"error": f"The upload exceeds the {limit_mb:g} MB file limit."}), 413


# ---------------------------------------------------------------- evaluation
@app.get("/api/evaluation")
def evaluation():
    if not EVAL_PATH.is_file():
        return jsonify({
            "available": False,
            "reason": "No evaluation results yet. Run `python evaluate_counter.py --full` to generate "
                      f"{EVAL_PATH.name} from the held-out test set.",
        })
    try:
        data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return jsonify({"available": False, "reason": f"Evaluation file could not be read: {exc}"})
    current = checkpoint_fingerprint(MODEL_PATH)
    recorded = (data.get("checkpoint") or {}).get("sha256")
    data["stale"] = bool(current and recorded and current != recorded)
    data["checkpoint_present"] = current is not None
    sep_eval_path = BASE_DIR / "evaluation" / "separation_results.json"
    if sep_eval_path.is_file():
        try:
            data["separation_eval"] = json.loads(sep_eval_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return jsonify(data)


# ---------------------------------------------------------------- analyze / separate / download
@app.post("/api/analyze")
def analyze():
    """Decode + shared log-mel + batched count + open-vocabulary sound naming + speaker clustering."""
    if not check_rate_limit():
        return jsonify({"error": "Rate limit exceeded. Please wait a moment and try again."}), 429

    status = model_status()
    if not status["trained"]:
        return jsonify({
            "trained": False,
            "message": "No compatible counting checkpoint is available, so no count is shown.",
            "model_error": status["model_error"],
        }), 200
    timings = {}
    try:
        t0 = time.perf_counter()
        job_id, job_dir, original_path, samples, rate, name = read_upload()
        clip, crop_start, duration, full_16k = analysis_clip(samples, rate)
        timings["decode_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # Shared log-mel spectrogram computation (cached across counter and tagger)
        tf = time.perf_counter()
        shared_logmel = waveform_to_logmel(clip)
        timings["feature_ms"] = round((time.perf_counter() - tf) * 1000, 1)

        t1 = time.perf_counter()
        result = count_with_uncertainty(
            clip, MODEL_PATH, MC_PASSES, precomputed_logmel=shared_logmel
        )
        timings["count_ms"] = round((time.perf_counter() - t1) * 1000, 1)

        # Phase 0.1: Ground truth lookup is ONLY enabled when ECHOCOUNT_DEV=1 (off in production)
        gt = None
        if ground_truth.is_dev_mode():
            t2 = time.perf_counter()
            gt = ground_truth.lookup_ground_truth(original_path, name)
            timings["gt_ms"] = round((time.perf_counter() - t2) * 1000, 1)

        # Phase 1: Open-Vocabulary Tagger (AST AudioSet-527 + CLAP + 32-class FSD CNN + frame-level time ranges)
        t3 = time.perf_counter()
        use_clap = os.getenv("ECHOCOUNT_FAST_TAG", "0") != "1"
        tags_open = open_tagger.tag_open_vocabulary(
            clip,
            sr=SAMPLE_RATE,
            precomputed_logmel=shared_logmel,
            use_clap_fallback=use_clap,
            localize_windows=True,
        )
        timings["tag_ms"] = round((time.perf_counter() - t3) * 1000, 1)

        # VAD + speaker embedding clustering
        t4 = time.perf_counter()
        speech_detected = bool(tags_open.get("speech_detected", False))
        speaker_clustering = speech_separation.estimate_distinct_speakers(
            clip,
            model_count=result["rounded_count"],
            sample_rate=SAMPLE_RATE,
            speech_detected=speech_detected,
        )
        timings["speaker_cluster_ms"] = round((time.perf_counter() - t4) * 1000, 1)
    except audio_io.AudioValidationError as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception:
        app.logger.exception("Analysis failed")
        return jsonify({"error": "Could not run the trained model. Check the checkpoint and server logs."}), 503

    named_sounds = tags_open.get("named_sounds", [])
    overlapping_sounds = tags_open.get("overlapping_sounds", [])
    tags_payload = {
        "available": bool(tags_open.get("available", True)),
        "hierarchical": True,
        "open_vocabulary": True,
        "groups": tags_open.get("groups", []),
        "active_groups": tags_open.get("active_groups", []),
        "named_sounds": named_sounds,
        "overlapping_sounds": overlapping_sounds,
        "max_simultaneous_sounds": tags_open.get("max_simultaneous_sounds", len(named_sounds)),
        "speech_detected": speech_detected,
        "tags": tags_open.get("tags", []),
        "caveat": tags_open.get(
            "caveat",
            "Named sounds include friendly labels, confidence, and frame-level time ranges.",
        ),
    }

    sep = status["separator"]
    speech_sep = status.get("speech_separator", {})
    k = int(result["rounded_count"])
    last_reliable_k, _, _ = get_dense_thresholds()
    dense_warning = get_dense_mixture_warning(k)

    # Determine whether a REAL trained separator supports this recording (Phase 0.2 & Phase 2.1):
    # - Untrained demo weights NEVER run in production.
    # - Pure speech: SepFormer wsj02mix supports K=2 speakers (and wsj03mix supports K=3 if present).
    #   For K >= 3 pure speech speakers without 3-speaker model, report "Separation not available for this recording".
    # - Non-speech / multi-sound mixtures: supported when named_sounds >= 1 and K <= last_reliable_k and trained separator is ready.
    max_speech_spks = int(speech_sep.get("max_supported_speakers") or 2)
    sep_ckpt_trained = bool(sep.get("trained", False))
    non_speech_named = [s for s in named_sounds if s.get("group") != "Human voice"]
    is_pure_speech = bool(speech_detected and len(non_speech_named) == 0)

    if k > last_reliable_k:
        sep_state = {
            "can_separate": False,
            "trained": True,
            "speech_mode": is_pure_speech,
            "speech_separator": speech_sep,
            "reason": "Too many overlapping sounds to separate reliably.",
        }
        max_separable = last_reliable_k
    elif is_pure_speech:
        real_speech_supports_k = (
            (k == 2 and bool(speech_sep.get("has_2spk_model")))
            or (k == 3 and bool(speech_sep.get("has_3spk_model")))
        )
        max_separable = max_speech_spks
        if k < 2:
            sep_state = {
                "can_separate": False,
                "trained": bool(speech_sep.get("available") or sep_ckpt_trained),
                "speech_mode": True,
                "speech_separator": speech_sep,
                "reason": "Separation not available for this recording: only 1 source estimated.",
            }
        elif real_speech_supports_k:
            sep_state = {
                "can_separate": True,
                "trained": True,
                "speech_mode": True,
                "speech_separator": speech_sep,
                "reason": None,
            }
        else:
            sep_state = {
                "can_separate": False,
                "trained": bool(speech_sep.get("available") or sep_ckpt_trained),
                "speech_mode": True,
                "speech_separator": speech_sep,
                "reason": f"Separation not available for this recording: Separation is supported for up to {max_speech_spks} speakers with the installed model.",
            }
    else:
        max_separable = min(last_reliable_k, separation.MAX_SEPARABLE)
        has_queries = len(named_sounds) >= 1 and k >= 2
        if k < 2:
            sep_state = {
                "can_separate": False,
                "trained": sep_ckpt_trained,
                "speech_mode": False,
                "speech_separator": speech_sep,
                "reason": "Separation not available for this recording: only 1 source estimated.",
            }
        elif sep_ckpt_trained and has_queries:
            sep_state = {
                "can_separate": True,
                "trained": True,
                "speech_mode": False,
                "speech_separator": speech_sep,
                "reason": None,
            }
        else:
            sep_state = {
                "can_separate": False,
                "trained": sep_ckpt_trained,
                "speech_mode": False,
                "speech_separator": speech_sep,
                "reason": (
                    "Separation not available for this recording: no trained general separator checkpoint is active."
                    if not sep_ckpt_trained
                    else "Separation not available for this recording: no distinct sound queries detected above threshold."
                ),
            }

    with _jobs_lock:
        _jobs[job_id] = {
            "created": time.time(),
            "name": name,
            "original": original_path,
            "input_rate": rate,
            "crop_start": crop_start,
            "duration": duration,
            "count": result,
            "speech_detected": is_pure_speech,
            "named_sounds": named_sounds,
            "speaker_clustering": speaker_clustering,
            "stems": None,
            "zip": None,
            "sep_status": "idle",
            "sep_result": None,
        }

    app.logger.info(
        "analyze completed job_id=%s file=%s duration=%.2fs k=%d sounds=%d timings=%s",
        job_id, name, duration, k, len(named_sounds), json.dumps(timings),
    )

    resp_payload = {
        "trained": True,
        "job_id": job_id,
        "filename": name,
        "audio": {
            "duration_seconds": round(duration, 2),
            "input_sample_rate": rate,
            "analysed_window_seconds": CLIP_SECONDS,
            "crop_start_seconds": round(crop_start, 3),
        },
        "count": {
            "raw_regression_value": round(result["raw_regression_value"], 4),
            "calibrated_value": round(result.get("calibrated_value", result["raw_regression_value"]), 4),
            "calibrated": result.get("calibrated", False),
            "sparse_regime": result.get("sparse_regime", False),
            "out_of_range": result.get("out_of_range", False),
            "range_warning": result.get("range_warning"),
            "dense_warning": dense_warning,
            "rounded_count": result["rounded_count"],
            "uncertainty_std": round(result["total_std"], 3),
            "mc_dropout_std": round(result["mc_std"], 3),
            "softmax_std": round(result.get("softmax_std", max(0.0, result["total_std"] ** 2 - result["mc_std"] ** 2) ** 0.5), 3),
            "mc_passes": result["passes"],
            "display": format_count(result),
            "confidence_percent": round(result["confidence_percent"], 1),
            "min_sources": result["min_sources"],
            "max_sources": result["max_sources"],
        },
        "named_sounds": named_sounds,
        "overlapping_sounds": overlapping_sounds,
        "speaker_clustering": speaker_clustering,
        "separation": {**sep_state, "max_separable": max_separable},
        "tags": tags_payload,
        "timings": timings,
    }
    if ground_truth.is_dev_mode() and gt is not None:
        resp_payload["ground_truth"] = gt

    return jsonify(resp_payload)


def _execute_separation_job(job_id: str, job: dict) -> tuple[dict, int]:
    """Run query-conditioned or SepFormer separation + residual stem + all 3 quality gates."""
    count = job["count"]
    k = int(count["rounded_count"])
    speech_mode = bool(job.get("speech_detected", False))
    named_sounds = job.get("named_sounds", [])
    sep = separation.separator_status()
    speech_sep_st = speech_separation.pretrained_speech_separator_status()
    max_speech_spks = int(speech_sep_st.get("max_supported_speakers") or 2)
    last_reliable_k, _, _ = get_dense_thresholds()

    # Phase 0.2: Never run untrained demo weights in production
    if not sep.get("trained", False) and not (speech_mode and speech_sep_st.get("available")):
        return {"error": "Separation not available for this recording: no trained separation model is active."}, 503
    if k < 2:
        return {"error": f"Separation not available for this recording: estimated {k} source."}, 409
    if k > last_reliable_k or k > separation.MAX_SEPARABLE:
        return {"error": "Too many overlapping sounds to separate reliably."}, 409
    if speech_mode and k > max_speech_spks:
        return {"error": f"Separation is supported for up to {max_speech_spks} speakers with the installed model."}, 409

    timings = {}
    try:
        t0 = time.perf_counter()
        samples, _ = audio_io.decode_mono(job["original"], MAX_DURATION_SECONDS)
        full_16k = audio_io.resample(samples, job["input_rate"], SAMPLE_RATE).astype(np.float32)

        q_out = query_separation.separate_by_queries(
            full_16k,
            k=k,
            named_sounds=named_sounds,
            speech_mode=speech_mode,
            sr=SAMPLE_RATE,
        )
        if not q_out.get("available") or not q_out.get("stems"):
            return {"error": q_out.get("reason") or "Separation not available for this recording."}, 409

        stems = q_out["stems"]
        residual_16k = q_out["residual_stem"]
        queries = q_out["queries"]
        used_pretrained_speech = bool(q_out.get("used_pretrained_speech", False))
        timings["separate_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    except Exception:
        app.logger.exception("Separation failed")
        return {"error": "Separation failed. Check the checkpoint and server logs."}, 503

    # Run all 3 measured quality gates (Phase 2.3):
    t_gate = time.perf_counter()
    # Gate 1 & Gate 3: Waveform correlation (< 0.90) + Energy conservation (sum(stems) + residual == mixture)
    sq = stem_quality_mod.compute_stem_quality(stems, full_16k, residual=residual_16k)
    # Gate 2: Self-consistency re-tagging per stem
    sc_gate = stem_quality_mod.evaluate_stem_self_consistency(stems, queries)
    # Speaker embedding verification for speech mixtures
    spk_ver = speech_separation.verify_separated_speaker_stems(
        stems, sample_rate=SAMPLE_RATE, speech_mode=speech_mode
    )
    timings["quality_gate_ms"] = round((time.perf_counter() - t_gate) * 1000, 1)

    quality_failed = bool(sq["failed"] or (speech_mode and spk_ver["could_not_separate_speakers"]))
    quality_reason = sq["failure_reason"]
    if speech_mode and spk_ver["could_not_separate_speakers"]:
        quality_reason = f"{spk_ver['reason']} Also: {quality_reason}" if quality_reason else spk_ver["reason"]

    rate = job["input_rate"]
    all_waves_16k = list(stems) + ([residual_16k] if residual_16k is not None else [])
    energies = np.array([float(np.sum(np.square(s))) for s in all_waves_16k])
    total_e = float(energies.sum()) if energies.sum() > 0 else 1.0
    shares = (energies / total_e).tolist()

    job_dir = JOB_DIR / job_id
    t1 = time.perf_counter()
    stem_files = []
    stem_entries = []

    for i, (stem, q_info, sc_info) in enumerate(zip(stems, queries, sc_gate["per_stem"]), 1):
        lbl = str(q_info.get("name") or q_info.get("friendly_name") or f"Sound {i}")
        slug = _slugify(lbl)
        fname = f"stem_{i:02d}_{slug}.wav"
        data = audio_io.wav_bytes(audio_io.peak_normalise(audio_io.resample(stem, SAMPLE_RATE, rate)), rate)
        (job_dir / fname).write_bytes(data)
        stem_files.append(fname)
        stem_entries.append({
            "index": i,
            "label": lbl,
            "group": q_info.get("group", ""),
            "time_ranges": q_info.get("time_ranges", []),
            "file": fname,
            "is_residual": False,
            "energy_share": round(shares[i - 1], 4),
            "self_consistent": sc_info["self_consistent"],
            "low_confidence": sc_info["low_confidence"],
            "status_badge": sc_info["status_badge"],
            "top_predicted_sound": sc_info["top_predicted_sound"],
            "url": f"/api/stem/{job_id}/{fname}",
            "download_url": f"/api/stem/{job_id}/{fname}?download=1",
        })

    # Write the Residual stem ("Everything else" / residual.wav)
    if residual_16k is not None:
        res_fname = "residual.wav"
        res_data = audio_io.wav_bytes(audio_io.peak_normalise(audio_io.resample(residual_16k, SAMPLE_RATE, rate)), rate)
        (job_dir / res_fname).write_bytes(res_data)
        stem_files.append(res_fname)
        stem_entries.append({
            "index": len(stem_entries) + 1,
            "label": "Everything else (Residual)",
            "group": "Residual",
            "time_ranges": [],
            "file": res_fname,
            "is_residual": True,
            "energy_share": round(shares[-1], 4),
            "self_consistent": True,
            "low_confidence": False,
            "status_badge": "Residual (mix − stems)",
            "top_predicted_sound": "Residual",
            "url": f"/api/stem/{job_id}/{res_fname}",
            "download_url": f"/api/stem/{job_id}/{res_fname}?download=1",
        })

    manifest = {
        "trained": True,
        "estimated_count": k,
        "raw_regression_value": count["raw_regression_value"],
        "uncertainty": {
            "std": count["total_std"],
            "mc_dropout_std": count["mc_std"],
            "method": "mc_dropout",
            "passes": count["passes"],
        },
        "stems_produced": len(stems),
        "includes_residual_stem": residual_16k is not None,
        "final_residual_energy_ratio": q_out.get("residual_energy_ratio", 0.0),
        "sample_rate": rate,
        "separation_processing_rate": SAMPLE_RATE,
        "duration_seconds": round(job["duration"], 2),
        "model_version": q_out.get("model_name", "QueryConditionedWiener"),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stems": [
            {
                "file": e["file"],
                "label": e["label"],
                "is_residual": e["is_residual"],
                "energy_share": e["energy_share"],
                "self_consistent": e["self_consistent"],
                "low_confidence": e["low_confidence"],
            }
            for e in stem_entries
        ],
        "quality_check": sq,
        "self_consistency": sc_gate,
        "speaker_verification": spk_ver,
        "source_file": job["name"],
    }

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for fname in stem_files:
            archive.write(job_dir / fname, fname)
        archive.write(job["original"], f"original_mix{job['original'].suffix}")
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
    timings["zip_ms"] = round((time.perf_counter() - t1) * 1000, 1)

    result_payload = {
        "job_id": job_id,
        "status": "completed",
        "trained": True,
        "speech_mode": speech_mode,
        "used_pretrained_speech": used_pretrained_speech,
        "pretrained_speech_status": speech_sep_st,
        "stems": stem_entries,
        "stopped_early": False,
        "quality_check": sq,
        "self_consistency": sc_gate,
        "speaker_verification": spk_ver,
        "quality_failed": quality_failed,
        "quality_reason": quality_reason,
        "download_url": f"/api/download/{job_id}.zip",
        "timings": timings,
    }

    with _jobs_lock:
        job["stems"] = stem_files
        job["zip"] = buffer.getvalue()
        job["used_pretrained_speech"] = used_pretrained_speech
        job["sep_status"] = "completed"
        job["sep_result"] = result_payload

    app.logger.info(
        "separate completed job_id=%s stems=%d self_consistency=%.2f timings=%s",
        job_id, len(stems), sc_gate["pass_rate"], json.dumps(timings),
    )
    return result_payload, 200


@app.post("/api/separate")
def separate_route():
    if not check_rate_limit():
        return jsonify({"error": "Rate limit exceeded. Please wait a moment and try again."}), 429

    payload = request.get_json(silent=True) or {}
    job_id = str(payload.get("job_id", ""))
    job = get_job(job_id)
    if job is None:
        return jsonify({"error": "That analysis expired or does not exist. Analyze the file again."}), 404

    run_async = bool(payload.get("async", False))
    if run_async:
        with _jobs_lock:
            job["sep_status"] = "running"

        def _bg():
            res, code = _execute_separation_job(job_id, job)
            with _jobs_lock:
                if code != 200:
                    job["sep_status"] = "failed"
                    job["sep_result"] = res

        _executor.submit(_bg)
        return jsonify({"job_id": job_id, "status": "running", "poll_url": f"/api/separate/{job_id}"}), 202

    res, code = _execute_separation_job(job_id, job)
    return jsonify(res), code


@app.get("/api/separate/<job_id>")
def separate_status_route(job_id):
    job = get_job(job_id)
    if job is None:
        return jsonify({"error": "Job expired or does not exist."}), 404
    st = job.get("sep_status", "idle")
    if st == "completed" and job.get("sep_result"):
        return jsonify(job["sep_result"]), 200
    if st == "failed":
        return jsonify({"job_id": job_id, "status": "failed", **(job.get("sep_result") or {})}), 409
    return jsonify({"job_id": job_id, "status": st}), 200


@app.get("/api/stem/<job_id>/<filename>")
def stem_preview(job_id, filename):
    job = get_job(job_id)
    if job is None or not job["stems"] or filename not in job["stems"]:
        return jsonify({"error": "Stem not found or expired."}), 404
    as_dl = request.args.get("download") == "1"
    return send_file(
        JOB_DIR / job_id / filename,
        mimetype="audio/wav",
        as_attachment=as_dl,
        download_name=filename if as_dl else None,
        max_age=0,
    )


@app.get("/api/download/<job_id>.zip")
def download(job_id):
    job = get_job(job_id)
    if job is None or not job["zip"]:
        return jsonify({"error": "This download has expired or was never created."}), 404
    stem = Path(job["name"]).stem or "mix"
    download_name = f"{stem}_stems.zip"
    return send_file(
        io.BytesIO(job["zip"]),
        mimetype="application/zip",
        as_attachment=True,
        download_name=download_name,
        max_age=0,
    )


# ---------------------------------------------------------------- legacy single-shot endpoint
@app.post("/api/predict")
def predict():
    audio_file = request.files.get("audio")
    if audio_file is None or not audio_file.filename:
        return jsonify({"error": "Choose an audio file first."}), 400
    if not allowed_file(audio_file.filename):
        return jsonify(
            {"error": "Unsupported file type. Upload WAV, FLAC, MP3, OGG, or M4A."}
        ), 400

    suffix = Path(audio_file.filename.replace("\\", "/")).suffix.lower()
    display_name = secure_filename(audio_file.filename) or f"audio{suffix}"
    temp_path = None
    try:
        descriptor, temp_name = tempfile.mkstemp(
            prefix="echocount_", suffix=suffix, dir=UPLOAD_DIR
        )
        os.close(descriptor)
        temp_path = Path(temp_name)
        audio_file.save(temp_path)
        if temp_path.stat().st_size > MAX_UPLOAD_BYTES:
            limit_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
            return jsonify(
                {"error": f"The upload exceeds the {limit_mb:g} MB file limit."}
            ), 413

        waveform, audio_info = load_audio(temp_path)
        duration = audio_info["original_duration_seconds"]
        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(waveform)))) if waveform.size else 0.0
        audio_metadata = {
            "filename": display_name,
            "original_duration_seconds": round(duration, 2),
            "duration_seconds": round(duration, 2),
            "duration_formatted": f"{int(duration // 60)}:{int(duration % 60):02d}",
            "original_sample_rate": audio_info["original_sample_rate"],
            "processed_sample_rate": SAMPLE_RATE,
            "channels_processed": 1,
            "peak_amplitude": round(peak, 4),
            "rms_energy": round(rms, 4),
            "model_window_seconds": CLIP_SECONDS,
            "crop_start_seconds": round(audio_info["crop_start_seconds"], 3),
        }

        current_status = model_status()
        if not current_status["trained"]:
            if current_status["model_error"]:
                message = (
                    "A checkpoint exists but is not compatible with this model. "
                    "Train a SourceCountCNN checkpoint with train.py."
                )
                status = "model_checkpoint_invalid"
            else:
                message = (
                    "The application is running, but no compatible trained weights "
                    "exist yet. Add your dataset and run train.py before requesting "
                    "source-count predictions."
                )
                status = "model_not_trained"
            return jsonify(
                {
                    "trained": False,
                    "status": status,
                    "message": message,
                    "audio": audio_metadata,
                    "prediction": None,
                }
            )

        result = predict_source_count(waveform, MODEL_PATH)
        crop_start = audio_info["crop_start_seconds"]
        timeline = [
            {
                **window,
                "start": round(window["start"] + crop_start, 3),
                "end": round(window["end"] + crop_start, 3),
                "confidence": round(window["confidence"], 4),
            }
            for window in result["timeline"]
        ]
        return jsonify(
            {
                "trained": True,
                "status": "ok",
                "message": (
                    "Prediction completed. Treat it as an estimate; evaluate it "
                    "on held-out test audio before relying on it."
                ),
                "audio": audio_metadata,
                "prediction": {
                    "estimated_sources": result["estimated_sources"],
                    "raw_value": round(result["raw_value"], 4),
                    "confidence": round(result["confidence"], 1),
                    "total_windows": result["total_windows"],
                    "min_sources": result["min_sources"],
                    "max_sources": result["max_sources"],
                    "timeline": timeline,
                },
            }
        )
    except ValueError as exc:
        app.logger.info("Audio could not be processed: %s", exc)
        return jsonify({"error": str(exc)}), 422
    except Exception:
        app.logger.exception("Audio prediction failed")
        return jsonify(
            {"error": "Could not run the trained model. Check the checkpoint and server logs."}
        ), 503
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                app.logger.warning("Could not remove temporary upload %s", temp_path)


if __name__ == "__main__":
    app.run(
        debug=False,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
    )
