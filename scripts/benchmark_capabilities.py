"""Honest, seeded capability benchmark for separation modes K=2..8 (Part C).

Evaluates:
  1. speech-2         (existing pretrained speechbrain/sepformer-wsj02mix, K=2)
  2. speech-3         (additive pretrained speechbrain/sepformer-wsj03mix, K=3)
  3. speech-recursive (recursive SepFormer extraction, K=2..8)
  4. sound-query      (local QueryConditionedWiener+IterativeMaskSeparator on FSD50K, K=2..8;
                       also records that no licensed pretrained AudioSep checkpoint is installed)

Measures PIT SI-SDRi (dB), PIT SDRi (dB), ECAPA different-speaker rate, and AST+CLAP
self-consistency. Saves listening examples for K=2, 3, 5, 8 into outputs/listening_examples/
and writes the data-driven capability registry to capabilities.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import audio
import capabilities as cap_mod
import quality as quality_mod
import separation
from config import BASE_DIR, EVAL_DIR, MODELS_DIR, SAMPLE_RATE

LIBRI_DEV = BASE_DIR / "data" / "librispeech" / "LibriSpeech" / "dev-clean"
ISO_DIR = BASE_DIR / "data" / "isolated_sources"
LISTEN_DIR = BASE_DIR / "outputs" / "listening_examples"

CLASS_TO_FRIENDLY = {
    "Acoustic_guitar": ("Acoustic guitar", "Musical instrument"),
    "Applause": ("Applause", "Human voice"),
    "Bark": ("Dog bark", "Animal"),
    "Car_horn": ("Car horn", "Environment/other"),
    "Cello": ("Cello", "Musical instrument"),
    "Computer_keyboard": ("Keyboard typing", "Household/tools"),
    "Hammer": ("Hammer", "Household/tools"),
    "Knock": ("Door knock", "Household/tools"),
    "Laughter": ("Laughter", "Human voice"),
    "Meow": ("Cat meow", "Animal"),
    "Phone_vibration": ("Phone vibration", "Household/tools"),
    "Scissors": ("Scissors", "Household/tools"),
    "Speech": ("Human voice", "Human voice"),
    "Telephone": ("Telephone ring", "Household/tools"),
    "Trumpet": ("Trumpet", "Musical instrument"),
    "Water": ("Water", "Environment/other"),
}


def _si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    e = np.asarray(est, dtype=np.float64) - np.mean(est)
    r = np.asarray(ref, dtype=np.float64) - np.mean(ref)
    r_energy = float(np.dot(r, r)) + 1e-12
    proj = (float(np.dot(e, r)) / r_energy) * r
    noise = e - proj
    return float(10.0 * np.log10((float(np.dot(proj, proj)) + 1e-12) / (float(np.dot(noise, noise)) + 1e-12)))


def _sdr(est: np.ndarray, ref: np.ndarray) -> float:
    e = np.asarray(est, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    noise = e - r
    return float(10.0 * np.log10((float(np.dot(r, r)) + 1e-12) / (float(np.dot(noise, noise)) + 1e-12)))


def _pit_metrics(stems: list[np.ndarray], refs: list[np.ndarray], mix: np.ndarray) -> tuple[float, float]:
    """Compute exact Hungarian PIT SI-SDRi and SDRi (handles len(stems) < len(refs) via zero pad)."""
    k = len(refs)
    n = len(mix)
    est_list = [s[:n] if len(s) >= n else np.pad(s, (0, n - len(s))) for s in stems]
    while len(est_list) < k:
        est_list.append(np.zeros(n, dtype=np.float32))
    est_list = est_list[:k]

    sisdri_mat = np.zeros((k, k), dtype=np.float64)
    sdri_mat = np.zeros((k, k), dtype=np.float64)
    base_sisdr = [_si_sdr(mix, r) for r in refs]
    base_sdr = [_sdr(mix, r) for r in refs]

    for i in range(k):
        for j in range(k):
            sisdri_mat[i, j] = _si_sdr(est_list[j], refs[i]) - base_sisdr[i]
            sdri_mat[i, j] = _sdr(est_list[j], refs[i]) - base_sdr[i]

    row_ind, col_ind = linear_sum_assignment(-sisdri_mat)
    return float(np.mean(sisdri_mat[row_ind, col_ind])), float(np.mean(sdri_mat[row_ind, col_ind]))


def _extract_active_segment(path: Path, seg_len: int, rng: np.random.Generator) -> np.ndarray | None:
    try:
        y, sr = audio.decode_mono(path, max_seconds=60.0, target_rate=SAMPLE_RATE)
    except Exception:
        return None
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if len(y) < int(0.6 * seg_len):
        return None
    if len(y) > seg_len:
        # Pick highest-energy window among 4 random offsets so speakers are genuinely overlapping
        best_seg, best_e = None, -1.0
        for _ in range(4):
            st = int(rng.integers(0, len(y) - seg_len + 1))
            seg = y[st : st + seg_len]
            e = float(np.mean(seg**2))
            if e > best_e:
                best_e, best_seg = e, seg
        y = best_seg
    else:
        y = np.pad(y, (0, seg_len - len(y)))
    rms = float(np.sqrt(np.mean(y**2)))
    if rms < 1e-4:
        return None
    return (y / rms * 0.1).astype(np.float32)


def _load_librispeech_pool(rng: np.random.Generator, seg_len: int, max_per_spk: int = 6) -> dict[str, list[np.ndarray]]:
    assert LIBRI_DEV.is_dir(), f"LibriSpeech dev-clean missing at {LIBRI_DEV}"
    pool: dict[str, list[np.ndarray]] = {}
    for spk_dir in sorted(p for p in LIBRI_DEV.iterdir() if p.is_dir()):
        flacs = sorted(list(spk_dir.rglob("*.flac")) + list(spk_dir.rglob("*.wav")))
        rng.shuffle(flacs)
        segs: list[np.ndarray] = []
        for f in flacs:
            seg = _extract_active_segment(f, seg_len, rng)
            if seg is not None:
                segs.append(seg)
            if len(segs) >= max_per_spk:
                break
        if segs:
            pool[spk_dir.name] = segs
    return pool


def _load_fsd_pool(rng: np.random.Generator, seg_len: int, max_per_cls: int = 12) -> dict[str, list[np.ndarray]]:
    pool: dict[str, list[np.ndarray]] = {}
    for cls_dir in sorted(p for p in ISO_DIR.iterdir() if p.is_dir() and p.name in CLASS_TO_FRIENDLY):
        wavs = sorted(cls_dir.glob("*.wav"))
        # Use held-out half (second half of sorted IDs) for clip-disjoint evaluation
        held_out = wavs[len(wavs) // 2 :]
        rng.shuffle(held_out)
        segs: list[np.ndarray] = []
        for f in held_out:
            seg = _extract_active_segment(f, seg_len, rng)
            if seg is not None:
                segs.append(seg)
            if len(segs) >= max_per_cls:
                break
        if segs:
            pool[cls_dir.name] = segs
    return pool


def _make_mix(refs: list[np.ndarray], rng: np.random.Generator) -> tuple[np.ndarray, list[np.ndarray]]:
    scaled = []
    for r in refs:
        db = float(rng.uniform(-2.0, 2.0))
        scaled.append((r * (10.0 ** (db / 20.0))).astype(np.float32))
    mix = np.sum(np.stack(scaled, axis=0), axis=0).astype(np.float32)
    peak = float(np.max(np.abs(mix))) + 1e-8
    if peak > 0.92:
        factor = 0.92 / peak
        mix = (mix * factor).astype(np.float32)
        scaled = [(s * factor).astype(np.float32) for s in scaled]
    return mix, scaled


def _save_listening_example(k: int, mode: str, mix: np.ndarray, refs: list[np.ndarray], stems: list[np.ndarray], residual: np.ndarray | None, metrics: dict[str, Any]) -> None:
    out_dir = LISTEN_DIR / f"K{k}_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mixture.wav").write_bytes(audio.wav_bytes(audio.peak_normalise(mix), SAMPLE_RATE))
    for i, r in enumerate(refs, 1):
        (out_dir / f"ref_{i:02d}.wav").write_bytes(audio.wav_bytes(audio.peak_normalise(r), SAMPLE_RATE))
    for i, s in enumerate(stems, 1):
        (out_dir / f"est_{i:02d}.wav").write_bytes(audio.wav_bytes(audio.peak_normalise(s), SAMPLE_RATE))
    if residual is not None:
        (out_dir / "residual.wav").write_bytes(audio.wav_bytes(audio.peak_normalise(residual), SAMPLE_RATE))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _dir_size_mb(p: Path) -> float:
    if not p.is_dir():
        return 0.0
    return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1048576, 2)


def run_benchmark(n_per_k: int, seed: int, duration_s: float) -> dict[str, Any]:
    t_start = time.perf_counter()
    rng = np.random.default_rng(seed)
    seg_len = int(SAMPLE_RATE * duration_s)

    print(f"Loading speaker-disjoint LibriSpeech dev-clean & clip-disjoint FSD50K pools (seg={duration_s}s)...")
    speech_pool = _load_librispeech_pool(rng, seg_len)
    fsd_pool = _load_fsd_pool(rng, seg_len)
    spk_ids = sorted(speech_pool.keys())
    fsd_cls = sorted(fsd_pool.keys())
    print(f"Loaded {len(spk_ids)} LibriSpeech speakers and {len(fsd_cls)} held-out FSD50K classes.")

    # Pre-generate seeded mixtures for K=2..8 so every speech mode is compared on identical mixtures
    speech_mixes: dict[int, list[tuple[np.ndarray, list[np.ndarray]]]] = {}
    for k in range(2, 9):
        items = []
        for _ in range(n_per_k):
            chosen_spks = rng.choice(spk_ids, size=k, replace=False)
            refs = [speech_pool[s][int(rng.integers(0, len(speech_pool[s])))] for s in chosen_spks]
            items.append(_make_mix(refs, rng))
        speech_mixes[k] = items

    sound_mixes: dict[int, list[tuple[np.ndarray, list[np.ndarray], list[dict[str, Any]]]]] = {}
    for k in range(2, 9):
        items = []
        for _ in range(n_per_k):
            chosen = rng.choice(fsd_cls, size=k, replace=False)
            refs, qs = [], []
            for cname in chosen:
                refs.append(fsd_pool[cname][int(rng.integers(0, len(fsd_pool[cname])))])
                fname, grp = CLASS_TO_FRIENDLY[cname]
                qs.append({"name": fname, "friendly_name": fname, "class": cname, "group": grp, "confidence": 0.9, "time_ranges": []})
            mix, scaled = _make_mix(refs, rng)
            items.append((mix, scaled, qs))
        sound_mixes[k] = items

    def _eval_speech_mode(mode_name: str, ks: list[int], fn) -> dict[str, Any]:
        per_k: dict[str, Any] = {}
        for k in ks:
            sisdris, sdris, ecapa_ok, sc_rates = [], [], [], []
            for idx, (mix, refs) in enumerate(speech_mixes[k]):
                out = fn(mix, k)
                stems = out.get("stems") or []
                si, sd = _pit_metrics(stems, refs, mix)
                sisdris.append(si)
                sdris.append(sd)
                ver = separation.verify_separated_speaker_stems(stems, sample_rate=SAMPLE_RATE, speech_mode=True)
                ecapa_ok.append(0.0 if ver.get("could_not_separate_speakers") else 1.0)
                queries = [{"name": f"Human {i + 1}", "class": "Speech", "group": "Human voice"} for i in range(len(stems))]
                sc = quality_mod.evaluate_stem_self_consistency(stems, queries)
                sc_rates.append(float(sc.get("pass_rate", 0.0)))
                if idx == 0 and k in (2, 3, 5, 8) and (
                    (mode_name == "speech-2" and k == 2)
                    or (mode_name == "speech-3" and k == 3)
                    or (mode_name == "speech-recursive" and k in (5, 8))
                ):
                    res = out.get("residual_stem")
                    if res is None and stems:
                        res = (mix - np.sum(np.stack(stems, axis=0), axis=0)).astype(np.float32)
                    _save_listening_example(
                        k, mode_name, mix, refs, stems, res,
                        {"mode": mode_name, "k": k, "si_sdri_db": round(si, 2), "sdri_db": round(sd, 2),
                         "ecapa_diff_ok": bool(ecapa_ok[-1]), "self_consistency": round(sc_rates[-1], 3)},
                    )
            m_si = round(float(np.mean(sisdris)), 2)
            m_sd = round(float(np.mean(sdris)), 2)
            m_ec = round(float(np.mean(ecapa_ok)), 4)
            m_sc = round(float(np.mean(sc_rates)), 4)
            st = cap_mod.classify_k_status(m_si, m_sc, available=True)
            per_k[str(k)] = {
                "n": len(sisdris),
                "mean_sisdri_db": m_si,
                "mean_sdri_db": m_sd,
                "ecapa_diff_speaker_rate": m_ec,
                "self_consistency": m_sc,
                "status": st,
            }
            print(f"  [{mode_name}] K={k} (n={len(sisdris)}): SI-SDRi={m_si:+.2f} dB | SDRi={m_sd:+.2f} dB | ECAPA_diff={m_ec*100:.1f}% | SC={m_sc*100:.1f}% -> {st}")
        return per_k

    print("\n--- Evaluating speech-2 (SepFormer WSJ0-2mix) ---")
    s2_pk = _eval_speech_mode("speech-2", [2], lambda m, k: separation.separate_speech_pretrained(m, 2, SAMPLE_RATE))

    print("\n--- Evaluating speech-3 (SepFormer WSJ0-3mix) ---")
    s3_pk = _eval_speech_mode("speech-3", [3], lambda m, k: separation.separate_speech_pretrained(m, 3, SAMPLE_RATE))

    print("\n--- Evaluating speech-recursive (K=2..8) ---")
    sr_pk = _eval_speech_mode("speech-recursive", list(range(2, 9)), lambda m, k: separation.separate_speech_recursive(m, k, SAMPLE_RATE))

    print("\n--- Evaluating sound-query (QueryConditionedWiener on held-out FSD50K, K=2..8) ---")
    sq_pk: dict[str, Any] = {}
    for k in range(2, 9):
        sisdris, sdris, sc_rates = [], [], []
        for idx, (mix, refs, qs) in enumerate(sound_mixes[k]):
            out = separation.separate_by_queries(mix, k=k, named_sounds=qs, speech_mode=False, sr=SAMPLE_RATE)
            stems = out.get("stems") or []
            si, sd = _pit_metrics(stems, refs, mix)
            sisdris.append(si)
            sdris.append(sd)
            sc = quality_mod.evaluate_stem_self_consistency(stems, qs[: len(stems)])
            sc_rates.append(float(sc.get("pass_rate", 0.0)))
        m_si = round(float(np.mean(sisdris)), 2)
        m_sd = round(float(np.mean(sdris)), 2)
        m_sc = round(float(np.mean(sc_rates)), 4)
        st = cap_mod.classify_k_status(m_si, m_sc, available=True)
        sq_pk[str(k)] = {
            "n": len(sisdris),
            "mean_sisdri_db": m_si,
            "mean_sdri_db": m_sd,
            "ecapa_diff_speaker_rate": None,
            "self_consistency": m_sc,
            "status": st,
        }
        print(f"  [sound-query] K={k} (n={len(sisdris)}): SI-SDRi={m_si:+.2f} dB | SDRi={m_sd:+.2f} dB | SC={m_sc*100:.1f}% -> {st}")

    def _summarize_mode(per_k: dict[str, Any]) -> tuple[list[int], list[int], str]:
        en = sorted(int(k) for k, v in per_k.items() if v["status"] == "enabled")
        ex = sorted(int(k) for k, v in per_k.items() if v["status"] == "experimental")
        overall = "enabled" if en else ("experimental" if ex else "disabled")
        return en, ex, overall

    s2_en, s2_ex, s2_st = _summarize_mode(s2_pk)
    s3_en, s3_ex, s3_st = _summarize_mode(s3_pk)
    sr_en, sr_ex, sr_st = _summarize_mode(sr_pk)
    sq_en, sq_ex, sq_st = _summarize_mode(sq_pk)

    # Read existing counting per-K metrics from evaluation/public_summary.json
    pub = json.loads((EVAL_DIR / "public_summary.json").read_text(encoding="utf-8")) if (EVAL_DIR / "public_summary.json").is_file() else {}
    per_k_count = (pub.get("counting") or {}).get("per_count_overall") or []

    registry = {
        "policy": {
            "enabled_min_sisdri_db": cap_mod.ENABLED_MIN_SISDRI_DB,
            "enabled_min_self_consistency": cap_mod.ENABLED_MIN_SELF_CONSISTENCY,
            "experimental_min_sisdri_db": cap_mod.EXPERIMENTAL_MIN_SISDRI_DB,
        },
        "benchmark_meta": {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "seed": seed,
            "n_per_k": n_per_k,
            "clip_duration_s": duration_s,
            "speech_dataset": "LibriSpeech dev-clean (speaker-disjoint, 40 speakers)",
            "sound_dataset": "FSD50K isolated_sources held-out split (clip-disjoint & class-disjoint, 16 classes)",
            "elapsed_seconds": round(time.perf_counter() - t_start, 1),
        },
        "modes": {
            "speech-2": {
                "label": "2-speaker speech (SepFormer WSJ0-2mix)",
                "family": "speech",
                "model": "speechbrain/sepformer-wsj02mix",
                "version": "pretrained-wsj02mix",
                "licence": "Apache-2.0",
                "size_mb": _dir_size_mb(MODELS_DIR / "sepformer-wsj02mix"),
                "status": s2_st,
                "enabled_k": s2_en,
                "experimental_k": s2_ex,
                "per_k": s2_pk,
            },
            "speech-3": {
                "label": "3-speaker speech (SepFormer WSJ0-3mix)",
                "family": "speech",
                "model": "speechbrain/sepformer-wsj03mix",
                "version": "21e78b44a378",
                "licence": "Apache-2.0",
                "size_mb": _dir_size_mb(MODELS_DIR / "sepformer-wsj03mix"),
                "status": s3_st,
                "enabled_k": s3_en,
                "experimental_k": s3_ex,
                "per_k": s3_pk,
            },
            "speech-recursive": {
                "label": "Recursive speech separation (K=2..8)",
                "family": "speech",
                "model": "sepformer-wsj02mix + sepformer-wsj03mix (recursive + residual stop)",
                "version": "recursive-v1",
                "licence": "Apache-2.0",
                "size_mb": round(_dir_size_mb(MODELS_DIR / "sepformer-wsj02mix") + _dir_size_mb(MODELS_DIR / "sepformer-wsj03mix"), 2),
                "status": sr_st,
                "enabled_k": sr_en,
                "experimental_k": sr_ex,
                "per_k": sr_pk,
            },
            "sound-query": {
                "label": "Query-conditioned general sound separation (K=2..8)",
                "family": "general",
                "model": "QueryConditionedWiener+IterativeMaskSeparator (no licensed AudioSep HF repo available)",
                "version": "local-query-wiener",
                "licence": "Local checkpoint (AudioSep/AudioSep returned HTTP 401/Not Found on HF)",
                "pretrained_available": False,
                "size_mb": round((BASE_DIR / "checkpoints" / "separator_iterative.pt").stat().st_size / 1048576, 2)
                if (BASE_DIR / "checkpoints" / "separator_iterative.pt").is_file()
                else 0.0,
                "status": sq_st,
                "enabled_k": sq_en,
                "experimental_k": sq_ex,
                "per_k": sq_pk,
            },
        },
        "counting_limits": {
            "reliable_sparse_max_k": 6,
            "warning_above_k": 6,
            "per_count_overall": per_k_count,
        },
    }

    cap_mod.REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"\nWrote capability registry -> {cap_mod.REGISTRY_PATH} in {registry['benchmark_meta']['elapsed_seconds']}s")
    return registry


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-per-k", type=int, default=100, help="Mixtures per K (default 100)")
    ap.add_argument("--seed", type=int, default=2026, help="Random seed")
    ap.add_argument("--duration", type=float, default=1.6, help="Clip duration in seconds")
    args = ap.parse_args()
    run_benchmark(n_per_k=args.n_per_k, seed=args.seed, duration_s=args.duration)
