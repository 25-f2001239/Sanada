r"""Task 6: Pipeline Latency Profiler (Before vs. After Efficiency Optimisations).

Profiles an 8-second clip across:
1. decode (read + mono + resample + 8s window)
2. count (30 MC-dropout passes):
   - BEFORE: uncached log-mel + 30 sequential full-CNN passes
   - AFTER: cached log-mel + single trunk pass + batched 30-pass classifier head with inference_mode & AMP
3. tag:
   - BEFORE: recomputing log-mel spectrogram + CNN forward pass
   - AFTER: reusing cached log-mel spectrogram + CNN forward pass with inference_mode & AMP
4. separate (on-demand only, not executed during /api/analyze)
5. zip (on-demand only, not executed during /api/analyze)

Usage:
  .\.venv\Scripts\python profile_pipeline.py --smoke
  .\.venv\Scripts\python profile_pipeline.py --full
"""
import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

import audio_io
import counting
import model as model_mod
import separation
import tagger

BASE_DIR = Path(__file__).resolve().parent
EVAL_DIR = BASE_DIR / "evaluation"
PROFILE_OUT = EVAL_DIR / "efficiency_profile.json"


def run_legacy_unbatched_count(clip: np.ndarray, passes: int = 30) -> None:
    """Simulate the legacy unbatched 30-pass MC-dropout loop + uncached log-mel."""
    net, sparse_net, _, min_s, max_s = model_mod._load_model_with_specialist(model_mod.MODEL_PATH)
    device = model_mod.DEVICE
    # Legacy computed log-mel inside count_with_uncertainty
    spec = model_mod.waveform_to_logmel(clip).unsqueeze(0).to(device)
    net.eval()
    for m in net.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()
    values = np.arange(min_s, max_s + 1, dtype=np.float32)
    with torch.no_grad():
        for _ in range(passes):
            logits = net(spec)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            _ = float(np.sum(probs * values))


def run_legacy_uncached_tag(clip: np.ndarray) -> None:
    """Simulate legacy tag_audio that recomputed waveform_to_logmel from scratch."""
    _ = tagger.tag_audio_hierarchical(clip, precomputed_logmel=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Quick 3-iteration profile")
    parser.add_argument("--full", action="store_true", help="Full 10-iteration profile")
    parser.add_argument("--passes", type=int, default=30, help="Number of MC-dropout passes (default 30)")
    args = parser.parse_args()

    iters = 3 if (args.smoke and not args.full) else 10
    sample_wav = BASE_DIR / "data" / "real_eval" / "two_teachers_speaking.wav"
    if not sample_wav.is_file():
        sample_wav = next((BASE_DIR / "data").glob("FSD_Mix_*_Sounds/test/*.wav"))

    print("=" * 80)
    print(f"TASK 6: Profiling Pipeline on 8 s clip ({sample_wav.name}, {iters} iterations, {args.passes} MC passes)")
    print("=" * 80)

    # Warmup model caches
    samples, rate = audio_io.decode_mono(sample_wav, 600.0)
    clip = audio_io.resample(samples, rate, model_mod.SAMPLE_RATE)[: model_mod.CLIP_SAMPLES]
    if len(clip) < model_mod.CLIP_SAMPLES:
        clip = np.pad(clip, (0, model_mod.CLIP_SAMPLES - len(clip)))
    clip = clip.astype(np.float32)
    _ = counting.count_with_uncertainty(clip, model_mod.MODEL_PATH, passes=args.passes)
    _ = tagger.tag_audio_hierarchical(clip)

    # 1. Decode
    t_decode = []
    for _ in range(iters):
        t0 = time.perf_counter()
        s_raw, sr = audio_io.decode_mono(sample_wav, 600.0)
        s_16k = audio_io.resample(s_raw, sr, model_mod.SAMPLE_RATE)
        c = s_16k[: model_mod.CLIP_SAMPLES].astype(np.float32)
        t_decode.append(time.perf_counter() - t0)

    # 2. BEFORE: Count (30 sequential full-CNN passes + uncached log-mel)
    t_count_before = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run_legacy_unbatched_count(clip, passes=args.passes)
        t_count_before.append(time.perf_counter() - t0)

    # 3. BEFORE: Tag (recomputing log-mel from scratch)
    t_tag_before = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run_legacy_uncached_tag(clip)
        t_tag_before.append(time.perf_counter() - t0)

    # 4. AFTER: Shared log-mel + Batched Count (30 MC passes) + Cached-logmel Tag
    t_logmel_after = []
    t_count_after = []
    t_tag_after = []
    for _ in range(iters):
        tf = time.perf_counter()
        shared_spec = model_mod.waveform_to_logmel(clip)
        t_logmel_after.append(time.perf_counter() - tf)

        tc = time.perf_counter()
        res = counting.count_with_uncertainty(
            clip, model_mod.MODEL_PATH, passes=args.passes, precomputed_logmel=shared_spec
        )
        t_count_after.append(time.perf_counter() - tc)

        tt = time.perf_counter()
        _ = tagger.tag_audio_hierarchical(clip, precomputed_logmel=shared_spec)
        t_tag_after.append(time.perf_counter() - tt)

    # 5. On-demand Separate & ZIP (only executed when user clicks Separate, K=2)
    t_sep = []
    t_zip = []
    sep_iters = max(2, iters // 2)
    for _ in range(sep_iters):
        ts = time.perf_counter()
        sep_out = separation.separate(clip, 2)
        t_sep.append(time.perf_counter() - ts)

        tz = time.perf_counter()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for idx, st in enumerate(sep_out["stems"], 1):
                wav_b = audio_io.wav_bytes(audio_io.peak_normalise(st), model_mod.SAMPLE_RATE)
                zf.writestr(f"stem_{idx:02d}.wav", wav_b)
            zf.writestr("manifest.json", json.dumps({"k": 2}))
        t_zip.append(time.perf_counter() - tz)

    decode_s = float(np.mean(t_decode))
    count_before_s = float(np.mean(t_count_before))
    tag_before_s = float(np.mean(t_tag_before))
    analyze_before_s = decode_s + count_before_s + tag_before_s

    logmel_s = float(np.mean(t_logmel_after))
    count_after_s = float(np.mean(t_count_after))
    tag_after_s = float(np.mean(t_tag_after))
    analyze_after_s = decode_s + logmel_s + count_after_s + tag_after_s

    sep_s = float(np.mean(t_sep))
    zip_s = float(np.mean(t_zip))

    summary = {
        "clip": sample_wav.name,
        "duration_s": 8.0,
        "mc_passes": args.passes,
        "iterations": iters,
        "stages_seconds": {
            "decode": {
                "before_s": round(decode_s, 4),
                "after_s": round(decode_s, 4),
                "before_ms": round(decode_s * 1000, 1),
                "after_ms": round(decode_s * 1000, 1),
            },
            "shared_logmel": {
                "before_s": 0.0,
                "after_s": round(logmel_s, 4),
                "note": "Computed once and shared between count and tag in After pipeline",
            },
            "count_30_passes": {
                "before_s": round(count_before_s, 4),
                "after_s": round(logmel_s + count_after_s, 4),
                "after_model_only_s": round(count_after_s, 4),
                "before_ms": round(count_before_s * 1000, 1),
                "after_ms": round((logmel_s + count_after_s) * 1000, 1),
                "speedup_x": round(count_before_s / max(1e-6, logmel_s + count_after_s), 2),
            },
            "tag": {
                "before_s": round(tag_before_s, 4),
                "after_s": round(tag_after_s, 4),
                "before_ms": round(tag_before_s * 1000, 1),
                "after_ms": round(tag_after_s * 1000, 1),
                "speedup_x": round(tag_before_s / max(1e-6, tag_after_s), 2),
            },
            "analyze_total": {
                "before_s": round(analyze_before_s, 4),
                "after_s": round(analyze_after_s, 4),
                "before_ms": round(analyze_before_s * 1000, 1),
                "after_ms": round(analyze_after_s * 1000, 1),
                "speedup_x": round(analyze_before_s / max(1e-6, analyze_after_s), 2),
            },
            "separate_on_demand": {
                "before_s": round(sep_s, 4),
                "after_s": round(sep_s, 4),
                "ms": round(sep_s * 1000, 1),
                "note": "Executed on-demand only (/api/separate), never during /api/analyze",
            },
            "zip_on_demand": {
                "before_s": round(zip_s, 4),
                "after_s": round(zip_s, 4),
                "ms": round(zip_s * 1000, 1),
                "note": "Executed on-demand only (/api/separate), never during /api/analyze",
            },
        },
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nStage                         | Before (s / ms)       | After (s / ms)        | Speedup")
    print("-" * 86)
    print(f"decode (8s mono 16kHz)        | {decode_s:.4f}s ({decode_s*1000:6.1f} ms) | {decode_s:.4f}s ({decode_s*1000:6.1f} ms) | 1.00x")
    print(f"count (30 MC-dropout passes)  | {count_before_s:.4f}s ({count_before_s*1000:6.1f} ms) | {logmel_s+count_after_s:.4f}s ({(logmel_s+count_after_s)*1000:6.1f} ms) | {summary['stages_seconds']['count_30_passes']['speedup_x']:.2f}x")
    print(f"tag (group + 32 fine classes) | {tag_before_s:.4f}s ({tag_before_s*1000:6.1f} ms) | {tag_after_s:.4f}s ({tag_after_s*1000:6.1f} ms) | {summary['stages_seconds']['tag']['speedup_x']:.2f}x")
    print("-" * 86)
    print(f"TOTAL /api/analyze            | {analyze_before_s:.4f}s ({analyze_before_s*1000:6.1f} ms) | {analyze_after_s:.4f}s ({analyze_after_s*1000:6.1f} ms) | {summary['stages_seconds']['analyze_total']['speedup_x']:.2f}x")
    print(f"separate (on-demand K=2)      | {sep_s:.4f}s ({sep_s*1000:6.1f} ms) | {sep_s:.4f}s ({sep_s*1000:6.1f} ms) | on-request only")
    print(f"zip (on-demand K=2)           | {zip_s:.4f}s ({zip_s*1000:6.1f} ms) | {zip_s:.4f}s ({zip_s*1000:6.1f} ms) | on-request only")
    print(f"\nSaved efficiency profile to {PROFILE_OUT}")


if __name__ == "__main__":
    main()
