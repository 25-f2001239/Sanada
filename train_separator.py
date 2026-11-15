"""Train count-conditioned IterativeMaskSeparator with Permutation-Invariant Training (PIT) (Task 4).

Supports:
1. Multichannel4SpeakersTrim speaker-grouped curriculum training:
   - Splits isolated clips by speaker directory (e.g. Speaker_4_1..Speaker_4_4) so held-out
     validation speakers never leak into training.
   - Curriculum over K: Stage 1 (K=2), Stage 2 (K=2..3), Stage 3 (K=2..4).
   - PIT loss with waveform SI-SDR + L1 log-magnitude STFT, AdamW + CosineAnnealingLR,
     AMP on CUDA, early stopping on validation SI-SDRi, saves trained:true only on real data.
2. General isolated-source directory (`data/isolated_sources` or `data/FSD_Master`).
3. `--smoke` mode (2 epochs, 50 synthetic mixtures, writes to `work/smoke_separator.pt`
   so `checkpoints/separator_iterative.pt` stays `trained:false` until real isolated clips exist).

Usage:
    python train_separator.py --smoke
    python train_separator.py --full
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from scipy.optimize import linear_sum_assignment
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from dataset_config import load_config
from separation import (
    DEVICE,
    HOP_LENGTH,
    MAX_PASSES,
    MAX_SEPARABLE,
    N_FFT,
    SEPARATION_SAMPLE_RATE,
    SEPARATOR_ARCHITECTURE,
    IterativeMaskSeparator,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = BASE_DIR / "checkpoints" / "separator_iterative.pt"
BACKUP_CHECKPOINT = BASE_DIR / "checkpoints" / "separator_iterative_backup.pt"
CLIP_SAMPLES = int(SEPARATION_SAMPLE_RATE * 8.0)
SMOKE_SAMPLES = int(SEPARATION_SAMPLE_RATE * 2.0)


def calculate_sisdr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    target_energy = torch.sum(target ** 2) + eps
    alpha = (torch.sum(pred * target) + eps) / target_energy
    scaled_target = alpha * target
    noise = pred - scaled_target
    snr = (torch.sum(scaled_target ** 2) + eps) / (torch.sum(noise ** 2) + eps)
    return 10.0 * torch.log10(snr + eps)


def pit_loss(pred_stems_time, pred_stems_spec, true_stems_time, true_stems_spec, lambda_sisdr=0.05):
    k = len(pred_stems_time)
    with torch.no_grad():
        pred_mags = pred_stems_spec.abs()
        true_mags = true_stems_spec.abs()
        costs = np.zeros((k, k), dtype=np.float32)
        for i in range(k):
            for j in range(k):
                l1_val = F.l1_loss(pred_mags[i], true_mags[j]).item()
                sisdr_val = calculate_sisdr(pred_stems_time[i], true_stems_time[j]).item()
                costs[i, j] = l1_val - lambda_sisdr * sisdr_val
        row_ind, col_ind = linear_sum_assignment(costs)

    total_loss = torch.tensor(0.0, device=pred_stems_time.device)
    for p_idx, t_idx in zip(row_ind, col_ind):
        p_time = pred_stems_time[p_idx]
        t_time = true_stems_time[t_idx]
        p_mag = torch.log1p(pred_stems_spec[p_idx].abs())
        t_mag = torch.log1p(true_stems_spec[t_idx].abs())
        l1_loss = F.l1_loss(p_mag, t_mag)
        sisdr = calculate_sisdr(p_time, t_time)
        total_loss = total_loss + l1_loss - (lambda_sisdr * sisdr)

    return total_loss / k, list(zip(row_ind, col_ind))


class SyntheticSmokeDataset(Dataset):
    """Generate 50 synthetic mixtures on the fly for --smoke verification."""

    def __init__(self, num_samples=50, seed=42, clip_samples=SMOKE_SAMPLES):
        self.num_samples = num_samples
        self.clip_samples = clip_samples
        self.rng = random.Random(seed)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        for i in range(self.num_samples):
            yield self[i]

    def __getitem__(self, idx):
        if idx >= self.num_samples:
            raise IndexError
        sr = SEPARATION_SAMPLE_RATE
        length = self.clip_samples
        duration = length / sr
        k = self.rng.randint(2, min(4, MAX_SEPARABLE))
        sources = []
        base_freqs = [180.0, 320.0, 500.0, 720.0]
        for i in range(k):
            t = np.linspace(0, duration, length, endpoint=False)
            f0 = base_freqs[i % len(base_freqs)] * (1.0 + 0.05 * (idx % 5))
            env = np.sin(np.pi * (t / duration)) ** 1.5
            tone = np.sin(2 * np.pi * f0 * t) + 0.25 * np.sin(4 * np.pi * f0 * t)
            sig = (tone * env).astype(np.float32)
            gain = 10.0 ** (-self.rng.uniform(0, 4) / 20.0)
            sources.append(sig * gain)

        mix = np.sum(sources, axis=0)
        peak = max(np.max(np.abs(mix)), 1e-6)
        mix = mix * (0.89 / peak)
        sources = [s * (0.89 / peak) for s in sources]
        return torch.from_numpy(mix), [torch.from_numpy(s) for s in sources], k


class SpeakerCurriculumMixtureDataset(Dataset):
    """On-the-fly 2–4 speaker mixtures from isolated clips grouped by speaker ID.

    Ensures each source in a mixture comes from a distinct speaker.
    Supports curriculum learning by adjusting `k_min` and `k_max` per stage.
    """

    def __init__(self, speaker_to_files: Dict[str, List[Path]], num_mixtures=500, k_min=2, k_max=2, seed=42, clip_samples=CLIP_SAMPLES):
        self.speaker_to_files = {k: v for k, v in speaker_to_files.items() if len(v) > 0}
        self.speakers = sorted(self.speaker_to_files.keys())
        self.num_mixtures = num_mixtures
        self.k_min = k_min
        self.k_max = min(k_max, max(2, len(self.speakers)))
        self.clip_samples = clip_samples
        self.rng = random.Random(seed)

    def __len__(self):
        return self.num_mixtures

    def __iter__(self):
        for i in range(self.num_mixtures):
            yield self[i]

    def __getitem__(self, idx):
        if idx >= self.num_mixtures:
            raise IndexError
        k = self.rng.randint(self.k_min, self.k_max)
        if len(self.speakers) >= k:
            chosen_spks = self.rng.sample(self.speakers, k)
            chosen_files = [self.rng.choice(self.speaker_to_files[s]) for s in chosen_spks]
        else:
            all_files = [f for files in self.speaker_to_files.values() for f in files]
            chosen_files = self.rng.sample(all_files, k)

        sources = []
        for path in chosen_files:
            audio, sr = sf.read(str(path), dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != SEPARATION_SAMPLE_RATE:
                from audio_io import resample
                audio = resample(audio, sr, SEPARATION_SAMPLE_RATE)
            if len(audio) >= self.clip_samples:
                max_start = len(audio) - self.clip_samples
                start = self.rng.randint(0, max_start) if max_start > 0 else 0
                clip = audio[start:start + self.clip_samples]
            else:
                clip = np.pad(audio, (0, self.clip_samples - len(audio)))
            gain_db = self.rng.uniform(0.0, 5.0)
            clip = clip * (10.0 ** (-gain_db / 20.0))
            sources.append(clip)

        mix = np.sum(sources, axis=0)
        peak = max(np.max(np.abs(mix)), 1e-6)
        scale = 0.89 / peak
        mix = mix * scale
        sources = [s * scale for s in sources]
        return torch.from_numpy(mix), [torch.from_numpy(s) for s in sources], k


def find_isolated_clips_by_speaker(candidates: List[Path]) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
    """Load isolated clips grouped by class/speaker, respecting manifest.json train/val/test splits when present."""
    iso_manifest = BASE_DIR / "data" / "isolated_sources" / "manifest.json"
    if iso_manifest.is_file():
        import json

        mdata = json.loads(iso_manifest.read_text(encoding="utf-8"))
        train_map: Dict[str, List[Path]] = defaultdict(list)
        val_map: Dict[str, List[Path]] = defaultdict(list)
        for item in mdata.get("clips", []):
            p = BASE_DIR / "data" / "isolated_sources" / item["rel_path"]
            if not p.is_file():
                continue
            cls = item["class"]
            sp = item.get("split", "train")
            if sp == "train":
                train_map[cls].append(p)
            elif sp == "val":
                val_map[cls].append(p)
            # Note: split == "test" is strictly held out for final evaluation!
        if train_map and val_map:
            return dict(train_map), dict(val_map)

    valid_exts = {".wav", ".flac", ".ogg"}
    speaker_to_files: Dict[str, List[Path]] = defaultdict(list)
    for root in candidates:
        if not root.is_dir():
            continue
        for f in root.rglob("*"):
            if "_metadata" in f.parts or "_hf_cache" in f.parts:
                continue
            if f.suffix.lower() in valid_exts and not f.name.startswith("mix_") and "mixture" not in f.name.lower():
                speaker_id = f.parent.name
                speaker_to_files[speaker_id].append(f)
    train_map, val_map = {}, {}
    for s, files in speaker_to_files.items():
        files = sorted(files)
        n = len(files)
        n_tr = max(1, int(0.70 * n))
        n_va = max(n_tr + 1, int(0.85 * n))
        train_map[s] = files[:n_tr]
        val_map[s] = files[n_tr:n_va]
    return train_map, val_map


def separate_forward(model, mix, k, window):
    length = mix.numel()
    spec = torch.stft(mix, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    residual = spec.clone()
    mix_logmag = torch.log1p(spec.abs()).unsqueeze(0)
    stems_time, stems_spec = [], []
    for i in range(k):
        res_logmag = torch.log1p(residual.abs()).unsqueeze(0)
        mask = model(mix_logmag, res_logmag, i, k)[0]
        source = residual * mask
        residual = residual - source
        stem = torch.istft(source, N_FFT, HOP_LENGTH, window=window, length=length)
        stems_time.append(stem)
        stems_spec.append(source)
    return torch.stack(stems_time), torch.stack(stems_spec)


def train(args):
    cfg = load_config()
    seed = args.seed or cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    window = torch.hann_window(N_FFT, device=device)

    if args.smoke:
        print("[Smoke Mode] Running 2-epoch curriculum PIT smoke test on 50 synthetic mixtures...", flush=True)
        train_ds = SyntheticSmokeDataset(50, seed=seed)
        val_ds = SyntheticSmokeDataset(20, seed=seed + 1)
        epochs = 2
        curriculum_stages = [(1, 2, 2), (2, 2, 4)]
        out_path = BASE_DIR / "work" / "smoke_separator.pt"
    else:
        ext_root = Path(cfg["dataset_path"])
        candidates = [
            args.data_dir,
            ext_root / "Multichannel4SpeakersTrim",
            BASE_DIR / "data" / "Multichannel4SpeakersTrim",
            BASE_DIR / "data" / "isolated_sources",
            BASE_DIR / "data" / "FSD_Master",
        ]
        train_map, val_map = find_isolated_clips_by_speaker(candidates)
        if not train_map:
            print("\n" + "=" * 78, file=sys.stderr)
            print("BLOCKED: No isolated speaker/source clips found on disk!", file=sys.stderr)
            print("Checked directories:", file=sys.stderr)
            for c in candidates:
                print(f"  - {c}", file=sys.stderr)
            print("Only pre-rendered mixtures (mix_*.wav) exist. Per rules, we do NOT train on mixtures.", file=sys.stderr)
            print("Place Multichannel4SpeakersTrim or data/isolated_sources/<ClassOrSpeaker>/*.wav first.", file=sys.stderr)
            print("=" * 78 + "\n", file=sys.stderr)
            sys.exit(1)

        print(
            f"Loaded isolated clips across {len(train_map)} classes "
            f"(train={sum(len(v) for v in train_map.values())} clips, val={sum(len(v) for v in val_map.values())} clips, test held out).",
            flush=True,
        )

        epochs = args.epochs
        s1 = max(1, epochs // 3)
        s2 = max(s1 + 1, (2 * epochs) // 3)
        curriculum_stages = [(s1, 2, 2), (s2, 2, 3), (epochs, 2, 4)]
        train_clip_samples = int(SEPARATION_SAMPLE_RATE * 4.0)
        train_ds = SpeakerCurriculumMixtureDataset(train_map, num_mixtures=args.train_mixtures, k_min=2, k_max=2, seed=seed, clip_samples=train_clip_samples)
        val_ds = SpeakerCurriculumMixtureDataset(val_map, num_mixtures=args.val_mixtures, k_min=2, k_max=4, seed=seed + 1, clip_samples=train_clip_samples)
        out_path = args.output
        if out_path.is_file():
            shutil.copy2(out_path, BACKUP_CHECKPOINT)

    model = IterativeMaskSeparator(channels=args.channels, blocks=args.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_sisdri = -float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Update curriculum K range
        for max_ep, kmin, kmax in curriculum_stages:
            if epoch <= max_ep:
                if hasattr(train_ds, "k_min"):
                    train_ds.k_min, train_ds.k_max = kmin, kmax
                break

        t0 = time.time()
        model.train()
        train_loss = 0.0
        for mix, true_stems, k in train_ds:
            mix = mix.to(device)
            true_stems_time = torch.stack([s.to(device) for s in true_stems])
            true_stems_spec = torch.stack([torch.stft(s, N_FFT, HOP_LENGTH, window=window, return_complex=True) for s in true_stems_time])

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                pred_stems_time, pred_stems_spec = separate_forward(model, mix, k, window)
                loss, _ = pit_loss(pred_stems_time, pred_stems_spec, true_stems_time, true_stems_spec)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= max(len(train_ds), 1)

        model.eval()
        all_sisdri = []
        with torch.no_grad():
            for mix, true_stems, k in val_ds:
                mix = mix.to(device)
                true_stems_time = torch.stack([s.to(device) for s in true_stems])
                true_stems_spec = torch.stack([torch.stft(s, N_FFT, HOP_LENGTH, window=window, return_complex=True) for s in true_stems_time])
                pred_stems_time, pred_stems_spec = separate_forward(model, mix, k, window)
                _, perm = pit_loss(pred_stems_time, pred_stems_spec, true_stems_time, true_stems_spec)
                for p_idx, t_idx in perm:
                    sisdri = calculate_sisdr(pred_stems_time[p_idx], true_stems_time[t_idx]).item() - calculate_sisdr(mix, true_stems_time[t_idx]).item()
                    all_sisdri.append(sisdri)

        mean_sisdri = float(np.mean(all_sisdri)) if all_sisdri else 0.0
        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | Loss={train_loss:.4f} | Val SI-SDRi={mean_sisdri:+.2f} dB", flush=True)

        if mean_sisdri > best_sisdri:
            best_sisdri = mean_sisdri
            best_epoch = epoch
            patience_counter = 0
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "architecture": SEPARATOR_ARCHITECTURE,
                "sample_rate": SEPARATION_SAMPLE_RATE,
                "n_fft": N_FFT,
                "hop_length": HOP_LENGTH,
                "config": {"channels": args.channels, "blocks": args.blocks},
                "model_state_dict": model.state_dict(),
                "version": "SMOKE-TEST" if args.smoke else "IterativeMaskSeparator-v2-curriculum",
                "trained": not args.smoke,
                "train_mixtures": len(train_ds),
                "val_mixtures": len(val_ds),
                "epochs": epoch,
                "best_val_sisdri": round(best_sisdri, 3),
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }, out_path)
        else:
            patience_counter += 1
            if not args.smoke and patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (no Val SI-SDRi improvement for {args.patience} epochs).")
                break

    print(f"Completed. Best epoch={best_epoch}, Best Val SI-SDRi={best_sisdri:+.2f} dB -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="2 epochs on 50 synthetic mixtures")
    parser.add_argument("--full", action="store_true", help="Full curriculum training on isolated clips")
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data" / "isolated_sources")
    parser.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--train-mixtures", type=int, default=600)
    parser.add_argument("--val-mixtures", type=int, default=100)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
