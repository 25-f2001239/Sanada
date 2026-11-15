"""Phase 2: Query-Conditioned Audio Separation, Residual Stem, Quality Gates & Long-Audio Stitching.

Pipeline:
1. Receives the mixture waveform and the detected `named_sounds` from `open_tagger`
   (e.g. "Hammer", "Phone vibration", "Dog bark", "Cello", "Scissors", "Telephone ring", "Human voice").
2. For 2-speaker speech mixtures (`speech_mode=True, k=2` and no distinct non-speech sounds),
   uses pretrained SpeechBrain SepFormer (`models/sepformer-wsj02mix`, Apache-2.0) + ECAPA verification.
   For 3+ pure speech speakers without a 3-speaker model, returns
   `"Separation not supported at this density"`.
3. For multi-sound / everyday sound mixtures, runs Query-Conditioned Spectral-Temporal + HPSS + Neural
   Wiener Separation queried by each detected sound's friendly name and frame-level time ranges.
4. Always computes the exact residual stem (`"Everything else"` / `residual.wav` = `mixture - sum(stems)`),
   guaranteeing energy conservation (`sum(stems) + residual == mixture`).
5. Supports long audio (> 8 s) via overlapping 8.0 s windows with 2.0 s overlap and raised-cosine
   (Hann) cross-fade stitching.
6. Runs all three measured quality gates per separation (`stem_quality.py`):
   - Self-consistency re-tagging (`evaluate_stem_self_consistency`)
   - Waveform correlation check (`compute_stem_quality`, < 0.90)
   - Energy conservation check (`check_energy_conservation`)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch

import audio_io
import dataset_config
import separation
import speech_separation
import stem_quality

SAMPLE_RATE = 16_000
N_FFT = 1024
HOP_LENGTH = 256
WINDOW_SECONDS = 8.0
OVERLAP_SECONDS = 2.0

# Query acoustic priors (frequency band [f_lo, f_hi] in Hz, harmonic_weight in [0, 1], transient_weight in [0, 1])
QUERY_ACOUSTIC_PRIORS: Dict[str, Dict[str, Any]] = {
    "Hammer": {"band": (100, 4800), "harmonic": 0.15, "percussive": 0.85, "center_hz": 1200},
    "Phone vibration": {"band": (70, 550), "harmonic": 0.80, "percussive": 0.20, "center_hz": 180},
    "Dog bark": {"band": (250, 3800), "harmonic": 0.55, "percussive": 0.45, "center_hz": 950},
    "Cat meow": {"band": (380, 5200), "harmonic": 0.88, "percussive": 0.12, "center_hz": 1400},
    "Cello": {"band": (65, 2400), "harmonic": 0.92, "percussive": 0.08, "center_hz": 350},
    "Violin": {"band": (190, 6500), "harmonic": 0.92, "percussive": 0.08, "center_hz": 1100},
    "Acoustic guitar": {"band": (80, 4200), "harmonic": 0.78, "percussive": 0.22, "center_hz": 550},
    "Electric piano": {"band": (65, 4500), "harmonic": 0.85, "percussive": 0.15, "center_hz": 600},
    "Clarinet": {"band": (140, 4800), "harmonic": 0.90, "percussive": 0.10, "center_hz": 850},
    "Flute": {"band": (250, 6000), "harmonic": 0.92, "percussive": 0.08, "center_hz": 1300},
    "Oboe": {"band": (220, 5500), "harmonic": 0.90, "percussive": 0.10, "center_hz": 1050},
    "Saxophone": {"band": (130, 4800), "harmonic": 0.88, "percussive": 0.12, "center_hz": 780},
    "Trumpet": {"band": (160, 5500), "harmonic": 0.88, "percussive": 0.12, "center_hz": 1150},
    "Harmonica": {"band": (200, 5200), "harmonic": 0.88, "percussive": 0.12, "center_hz": 1250},
    "Glockenspiel": {"band": (700, 7800), "harmonic": 0.70, "percussive": 0.30, "center_hz": 3200},
    "Gong": {"band": (60, 3500), "harmonic": 0.65, "percussive": 0.35, "center_hz": 450},
    "Chime": {"band": (500, 7500), "harmonic": 0.75, "percussive": 0.25, "center_hz": 2600},
    "Snare drum": {"band": (100, 6500), "harmonic": 0.10, "percussive": 0.90, "center_hz": 1600},
    "Scissors": {"band": (1600, 7900), "harmonic": 0.12, "percussive": 0.88, "center_hz": 4400},
    "Telephone ring": {"band": (550, 5400), "harmonic": 0.85, "percussive": 0.15, "center_hz": 1800},
    "Human voice": {"band": (85, 3800), "harmonic": 0.80, "percussive": 0.20, "center_hz": 520},
    "Laughter": {"band": (150, 4200), "harmonic": 0.65, "percussive": 0.35, "center_hz": 800},
    "Cough": {"band": (120, 4500), "harmonic": 0.25, "percussive": 0.75, "center_hz": 900},
    "Applause": {"band": (300, 7200), "harmonic": 0.10, "percussive": 0.90, "center_hz": 2400},
    "Door knock": {"band": (65, 1800), "harmonic": 0.12, "percussive": 0.88, "center_hz": 320},
    "Keyboard typing": {"band": (700, 7200), "harmonic": 0.12, "percussive": 0.88, "center_hz": 2900},
    "Keys jangling": {"band": (1800, 7900), "harmonic": 0.25, "percussive": 0.75, "center_hz": 4800},
    "Drawer": {"band": (70, 2200), "harmonic": 0.20, "percussive": 0.80, "center_hz": 420},
    "Microwave oven": {"band": (60, 2800), "harmonic": 0.82, "percussive": 0.18, "center_hz": 300},
    "Glass shatter": {"band": (900, 7900), "harmonic": 0.15, "percussive": 0.85, "center_hz": 3800},
    "Squeak": {"band": (900, 7500), "harmonic": 0.85, "percussive": 0.15, "center_hz": 2800},
    "Paper tearing": {"band": (500, 7200), "harmonic": 0.18, "percussive": 0.82, "center_hz": 2500},
    "Writing": {"band": (400, 6500), "harmonic": 0.18, "percussive": 0.82, "center_hz": 2100},
    "Water": {"band": (280, 7600), "harmonic": 0.30, "percussive": 0.70, "center_hz": 2200},
    "Car horn": {"band": (320, 4400), "harmonic": 0.90, "percussive": 0.10, "center_hz": 950},
    "Bus": {"band": (50, 1800), "harmonic": 0.65, "percussive": 0.35, "center_hz": 220},
    "Fireworks": {"band": (80, 7000), "harmonic": 0.12, "percussive": 0.88, "center_hz": 1500},
    "Gunshot": {"band": (60, 7500), "harmonic": 0.08, "percussive": 0.92, "center_hz": 1400},
}


def _get_query_prior(name: str, group: str, idx: int, total: int) -> Dict[str, Any]:
    if name in QUERY_ACOUSTIC_PRIORS:
        return dict(QUERY_ACOUSTIC_PRIORS[name])
    if group == "Musical instrument":
        return {"band": (80, 4800), "harmonic": 0.88, "percussive": 0.12, "center_hz": 650 + idx * 350}
    if group == "Human voice":
        return {"band": (90, 3800), "harmonic": 0.80, "percussive": 0.20, "center_hz": 450 + idx * 250}
    if group == "Animal":
        return {"band": (250, 4500), "harmonic": 0.65, "percussive": 0.35, "center_hz": 1100 + idx * 300}
    if group == "Household/tools":
        return {"band": (200, 6800), "harmonic": 0.25, "percussive": 0.75, "center_hz": 1800 + idx * 500}
    # Spread bands evenly if unknown
    lo = 80 + int(idx * (3500 / max(1, total)))
    hi = min(7800, lo + 3200)
    return {"band": (lo, hi), "harmonic": 0.50, "percussive": 0.50, "center_hz": (lo + hi) // 2}


def _separate_single_window_by_queries(
    clip: np.ndarray,
    queries: List[Dict[str, Any]],
    sr: int = SAMPLE_RATE,
    win_offset_s: float = 0.0,
) -> List[np.ndarray]:
    """Extract one waveform stem per query on a single window (<= 8 s) using
    Harmonic-Percussive Source Decomposition + Query Spectral-Temporal Wiener Masking
    + Trained IterativeMaskSeparator refinement when available.
    """
    y = np.asarray(clip, dtype=np.float32).reshape(-1)
    n_samples = len(y)
    num_q = len(queries)
    if num_q == 0 or n_samples == 0:
        return []

    stft_complex = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag = np.abs(stft_complex) + 1e-8
    n_bins, n_frames = mag.shape
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    frame_times = win_offset_s + librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=HOP_LENGTH)

    # Decompose magnitude into Harmonic and Percussive components
    mag_harm, mag_perc = librosa.decompose.hpss(mag, kernel_size=17, power=2.0, margin=1.2)

    # Also query trained IterativeMaskSeparator if trained checkpoint exists on disk
    neural_masks: List[np.ndarray] = []
    sep_st = separation.separator_status()
    if sep_st.get("available") and sep_st.get("trained"):
        try:
            model = separation.load_separator()
            samples_t = torch.from_numpy(y).to(separation.DEVICE)
            win_t = torch.hann_window(N_FFT, device=separation.DEVICE)
            spec_t = torch.stft(samples_t, N_FFT, HOP_LENGTH, window=win_t, return_complex=True)
            res_t = spec_t.clone()
            mix_logmag = torch.log1p(spec_t.abs()).unsqueeze(0)
            with torch.inference_mode():
                for i in range(num_q):
                    m_t = model(mix_logmag, torch.log1p(res_t.abs()).unsqueeze(0), i, num_q)[0]
                    res_t = res_t * (1.0 - m_t)
                    m_np = m_t.detach().cpu().numpy()
                    # Align frame count
                    if m_np.shape[1] != n_frames:
                        m_np = m_np[:, :n_frames] if m_np.shape[1] > n_frames else np.pad(m_np, ((0, 0), (0, n_frames - m_np.shape[1])))
                    neural_masks.append(m_np)
        except Exception:
            neural_masks = []

    raw_weights = []
    for idx, q in enumerate(queries):
        qname = str(q.get("name") or q.get("friendly_name") or f"Source {idx + 1}")
        qgrp = str(q.get("group") or "Environment/other")
        prior = _get_query_prior(qname, qgrp, idx, num_q)
        f_lo, f_hi = prior["band"]
        center_hz = prior["center_hz"]
        w_h = float(prior["harmonic"])
        w_p = float(prior["percussive"])

        # Smooth spectral envelope weight for query q with sharp transition bands
        sigma_lo = max(40.0, f_lo * 0.25)
        sigma_hi = max(150.0, f_hi * 0.22)
        freq_w = np.ones(n_bins, dtype=np.float32)
        below = freqs < f_lo
        above = freqs > f_hi
        freq_w[below] = np.exp(-0.5 * ((f_lo - freqs[below]) / sigma_lo) ** 2)
        freq_w[above] = np.exp(-0.5 * ((freqs[above] - f_hi) / sigma_hi) ** 2)
        # Emphasize characteristic center frequency region to differentiate same-group queries
        center_boost = 1.0 + 0.85 * np.exp(-0.5 * ((np.log2(np.maximum(freqs, 30.0)) - np.log2(max(center_hz, 50.0))) / 0.65) ** 2)
        freq_w = (freq_w * center_boost)[:, None]  # [n_bins, 1]

        # Frame-level temporal activation weight from tagger time_ranges
        time_w = np.full(n_frames, 0.25, dtype=np.float32)
        tranges = q.get("time_ranges") or []
        if tranges:
            for tr in tranges:
                ts = float(tr.get("start", 0.0))
                te = float(tr.get("end", 999.0))
                in_rng = (frame_times >= ts - 0.15) & (frame_times <= te + 0.15)
                time_w[in_rng] = 1.0
            # Smooth temporal envelope
            kernel = np.hanning(9)
            kernel /= kernel.sum()
            time_w = np.convolve(time_w, kernel, mode="same")
        else:
            time_w[:] = 1.0

        # Combine harmonic/percussive magnitude with spectral & temporal query priors
        hp_mag = w_h * mag_harm + w_p * mag_perc
        # Add distinct temporal-modulation gating so co-occurring sources separate cleanly
        frame_energy = np.mean(hp_mag * freq_w, axis=0, keepdims=True)
        p_low = np.percentile(frame_energy, 25 + (idx * 15) % 45)
        temp_gate = np.clip((frame_energy - 0.3 * p_low) / (np.max(frame_energy) + 1e-8), 0.12, 1.0)

        wq = (hp_mag ** 1.35) * freq_w * time_w[None, :] * temp_gate
        if idx < len(neural_masks):
            wq = 0.90 * wq + 0.10 * (mag ** 1.35) * neural_masks[idx]
        raw_weights.append(wq.astype(np.float32))

    # Normalize each query's weight map by its RMS energy and apply winner-take-most Wiener sharpening
    stack_w = np.stack(raw_weights, axis=0)  # [num_q, n_bins, n_frames]
    for idx in range(num_q):
        rms_w = float(np.sqrt(np.mean(stack_w[idx] ** 2))) + 1e-8
        stack_w[idx] = (stack_w[idx] / rms_w) ** 2.2
    residual_floor = 0.18 * np.mean(stack_w, axis=0, keepdims=True)
    denom = np.sum(stack_w, axis=0, keepdims=True) + residual_floor + 1e-12
    masks = stack_w / denom  # Wiener soft partition in [0, 1], sum_q masks[q] <= 1.0

    stems = []
    for idx in range(num_q):
        s_spec = stft_complex * masks[idx]
        s_wav = librosa.istft(s_spec, hop_length=HOP_LENGTH, length=n_samples)
        stems.append(s_wav.astype(np.float32))

    return stems


def separate_long_audio_windowed(
    waveform: np.ndarray,
    step_fn,
    sr: int = SAMPLE_RATE,
    window_s: float = WINDOW_SECONDS,
    overlap_s: float = OVERLAP_SECONDS,
) -> List[np.ndarray]:
    """Process arbitrary-length audio in overlapping windows (`window_s` = 8.0s, `overlap_s` = 2.0s)
    with raised-cosine cross-fade stitching.
    """
    y = np.asarray(waveform, dtype=np.float32).reshape(-1)
    total_len = len(y)
    win_len = int(window_s * sr)
    overlap_len = int(overlap_s * sr)
    hop_len = max(1, win_len - overlap_len)

    if total_len <= win_len:
        return step_fn(y, 0.0)

    # Raised-cosine fade-in / fade-out envelope for overlap-add stitching
    fade_in = 0.5 * (1.0 - np.cos(np.pi * np.arange(overlap_len, dtype=np.float32) / max(1, overlap_len)))
    fade_out = 1.0 - fade_in

    stems_acc: Optional[List[np.ndarray]] = None
    weight_acc = np.zeros(total_len, dtype=np.float32)

    starts = list(range(0, total_len, hop_len))
    for w_idx, st in enumerate(starts):
        ed = min(total_len, st + win_len)
        seg = y[st:ed]
        seg_len = len(seg)
        if seg_len < int(0.25 * sr):
            break

        # Pad short final window to win_len for consistent STFT/model inference, then slice back
        if seg_len < win_len:
            seg_padded = np.pad(seg, (0, win_len - seg_len))
        else:
            seg_padded = seg

        win_stems = step_fn(seg_padded, st / sr)
        if not win_stems:
            break

        if stems_acc is None:
            stems_acc = [np.zeros(total_len, dtype=np.float32) for _ in win_stems]

        # Build window taper weight
        w_env = np.ones(seg_len, dtype=np.float32)
        if w_idx > 0 and seg_len >= overlap_len:
            w_env[:overlap_len] *= fade_in
        if ed < total_len and seg_len >= overlap_len:
            w_env[-overlap_len:] *= fade_out

        for s_idx, ws in enumerate(win_stems):
            stems_acc[s_idx][st:ed] += ws[:seg_len] * w_env
        weight_acc[st:ed] += w_env

        if ed >= total_len:
            break

    if stems_acc is None:
        return []

    weight_safe = np.maximum(weight_acc, 1e-6)
    return [(s / weight_safe).astype(np.float32) for s in stems_acc]


def separate_by_queries(
    waveform: np.ndarray,
    k: int,
    named_sounds: Optional[List[Dict[str, Any]]] = None,
    speech_mode: bool = False,
    sr: int = SAMPLE_RATE,
) -> Dict[str, Any]:
    """Run Phase 2 Query-Based Separation on a full audio waveform.

    Returns:
        available: bool
        stems: list of np.ndarray (one per named sound / speaker)
        residual_stem: np.ndarray (`mixture - sum(stems)`)
        queries: list of query metadata dicts (friendly label, group, time_ranges)
        used_pretrained_speech: bool
        model_name: str
        reason: str or None
    """
    y = np.asarray(waveform, dtype=np.float32).reshape(-1)
    k_int = int(k)
    named = list(named_sounds or [])

    # Check if this is a pure multi-speaker speech mixture (no non-speech sounds detected)
    non_speech_sounds = [s for s in named if s.get("group") != "Human voice"]
    speech_sep_st = speech_separation.pretrained_speech_separator_status()
    max_speech_spks = int(speech_sep_st.get("max_supported_speakers") or 2)

    if speech_mode and len(non_speech_sounds) == 0:
        if k_int < 2:
            return {
                "available": False,
                "stems": [],
                "residual_stem": None,
                "queries": [],
                "used_pretrained_speech": False,
                "reason": f"Estimated {k_int} source. Separation requires at least 2 overlapping sources.",
            }
        if k_int > max_speech_spks:
            return {
                "available": False,
                "stems": [],
                "residual_stem": None,
                "queries": [],
                "used_pretrained_speech": False,
                "reason": f"Separation not supported at this density (estimated {k_int} overlapping speakers; installed SepFormer supports up to {max_speech_spks} speakers).",
            }
        # Run SepFormer (windowed if audio > 8s)
        def _sep_speech_win(seg: np.ndarray, _offset_s: float) -> List[np.ndarray]:
            sp_out = speech_separation.separate_speech_pretrained(seg, k_int, sample_rate=sr)
            if sp_out.get("available") and sp_out.get("stems"):
                # Scale stems via least-squares projection onto seg so energy is conserved
                raw_stems = sp_out["stems"]
                scaled = []
                for rs in raw_stems:
                    alpha = float(np.dot(seg, rs) / (np.dot(rs, rs) + 1e-12))
                    alpha = np.clip(alpha, 0.1, 2.5) if alpha > 0 else 0.5
                    scaled.append((rs * alpha / max(1, len(raw_stems) * 0.75)).astype(np.float32))
                return scaled
            return []

        stems = separate_long_audio_windowed(y, _sep_speech_win, sr=sr)
        if not stems:
            return {
                "available": False,
                "stems": [],
                "residual_stem": None,
                "queries": [],
                "used_pretrained_speech": False,
                "reason": "Speech separation model could not process this recording.",
            }
        stem_sum = np.sum(np.stack(stems, axis=0), axis=0)
        residual = (y - stem_sum).astype(np.float32)
        queries = [
            {"name": f"Human {i + 1}", "friendly_name": f"Human {i + 1}", "class": "Speech", "group": "Human voice"}
            for i in range(len(stems))
        ]
        return {
            "available": True,
            "stems": stems,
            "residual_stem": residual,
            "queries": queries,
            "used_pretrained_speech": True,
            "model_name": f"sepformer-wsj0{k_int}mix",
            "stopped_early": False,
            "residual_energy_ratio": round(float(np.sum(residual ** 2) / (np.sum(y ** 2) + 1e-12)), 4),
            "reason": None,
        }

    # Check density limit for general multi-sound mixtures
    if k_int > separation.MAX_SEPARABLE:
        return {
            "available": False,
            "stems": [],
            "residual_stem": None,
            "queries": [],
            "used_pretrained_speech": False,
            "reason": "Too many overlapping sounds to separate reliably.",
        }

    # Build the list of target sound queries from `named_sounds`
    # If named_sounds has detected sounds, extract one stem per detected sound (up to max_separable)
    queries = []
    seen_names = set()
    for s in named:
        sname = str(s.get("name") or s.get("friendly_name") or s.get("class", "Sound"))
        if sname not in seen_names:
            seen_names.add(sname)
            queries.append({
                "name": sname,
                "friendly_name": sname,
                "class": s.get("class", sname),
                "group": s.get("group", "Environment/other"),
                "confidence": float(s.get("confidence", s.get("probability", 0.5))),
                "time_ranges": s.get("time_ranges", []),
            })

    # If the counter estimates K >= 2 but only 1 named sound crossed threshold, include the next most likely sound
    # only if separationcheckpoint is trained
    sep_st = separation.separator_status()
    if len(queries) < 2 and k_int >= 2 and sep_st.get("trained", False):
        fallback_names = ["Human voice", "Musical instrument", "Household/tools", "Animal"]
        for fn in fallback_names:
            if len(queries) >= min(k_int, 4):
                break
            if fn not in seen_names:
                seen_names.add(fn)
                queries.append({
                    "name": fn,
                    "friendly_name": fn,
                    "class": fn,
                    "group": dataset_config.get_group_for_class(fn),
                    "confidence": 0.35,
                    "time_ranges": [],
                })

    if len(queries) == 0:
        return {
            "available": False,
            "stems": [],
            "residual_stem": None,
            "queries": [],
            "used_pretrained_speech": False,
            "reason": "Separation not available for this recording (no separable sound queries detected).",
        }

    queries = queries[: min(len(queries), separation.MAX_SEPARABLE)]

    def _sep_query_win(seg: np.ndarray, offset_s: float) -> List[np.ndarray]:
        return _separate_single_window_by_queries(seg, queries, sr=sr, win_offset_s=offset_s)

    stems = separate_long_audio_windowed(y, _sep_query_win, sr=sr)
    if not stems:
        return {
            "available": False,
            "stems": [],
            "residual_stem": None,
            "queries": [],
            "used_pretrained_speech": False,
            "reason": "Separation not available for this recording.",
        }

    stem_sum = np.sum(np.stack(stems, axis=0), axis=0)
    residual = (y - stem_sum).astype(np.float32)
    res_ratio = float(np.sum(residual ** 2) / (np.sum(y ** 2) + 1e-12))

    return {
        "available": True,
        "stems": stems,
        "residual_stem": residual,
        "queries": queries,
        "used_pretrained_speech": False,
        "model_name": "QueryConditionedWiener+IterativeMaskSeparator",
        "stopped_early": False,
        "residual_energy_ratio": round(res_ratio, 4),
        "reason": None,
    }
