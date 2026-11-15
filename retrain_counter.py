"""Retrain and calibrate the counting model on Combined Speech (Org_*) + FSD mixtures (Task 2).

Features:
- Loads both FSD_* and Org_* mixtures via dataset_config.py (from config.json dataset_path)
  with a source-family flag ('fsd' vs 'speech'). K covers 2, 3, 4, 6, 9, 15, 20, 25 and variable (2..31).
- Class-balanced sampling over (family, K) so neither sparse nor dense nor speech/FSD is starved.
- Auxiliary classification head over K bins (1..12, >12) + source-family head + main count head.
- Real-condition + spectral augmentations: SpecAugment (time/freq masking), gain & time-shift,
  simulated room reverb (exponentially decaying RIR envelope in mel domain), additive background
  noise floor, microphone gain variation, and 16 kHz band-limit roll-off.
- Trains a dedicated Sparse Specialist (K <= 8) alongside the General model (K = 2..31) and embeds
  both plus their validation-only Isotonic Calibrators into checkpoints/source_count_clip.pt.
- Also trains & evaluates a Held-Out-Speaker model (holding out Speaker_4_1, Speaker8, Speaker9)
  to report the honest unseen-speaker generalisation number.
- Never tunes or selects anything on the test split. Backs up existing checkpoint to
  checkpoints/source_count_clip_backup.pt before overwriting.

Usage:
    python retrain_counter.py --smoke
    python retrain_counter.py --full
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from sklearn.isotonic import IsotonicRegression
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from dataset_config import load_config, scan_all_mixtures
from model import (
    CLIP_SAMPLES,
    CLIP_SECONDS,
    DEVICE,
    HOP_LENGTH,
    MODEL_PATH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    SourceCountCNN,
    load_audio,
    waveform_to_logmel,
)

BASE_DIR = Path(__file__).resolve().parent
BACKUP_PATH = BASE_DIR / "checkpoints" / "source_count_clip_backup.pt"
V1_BACKUP_PATH = BASE_DIR / "checkpoints" / "source_count_clip_v1_backup.pt"
CACHE_PATH = BASE_DIR / "work" / "feature_cache_v3.pt"
MIN_SOURCES = 2
MAX_SOURCES = 31
SPARSE_MAX = 8


# ---------------------------------------------------------------- Multi-Task Counter Model
class CounterWithAuxHead(nn.Module):
    """SourceCountCNN trunk with main count head, 13-bin auxiliary head, and source-family head."""

    def __init__(self, min_sources=MIN_SOURCES, max_sources=MAX_SOURCES):
        super().__init__()
        self.min_sources = min_sources
        self.max_sources = max_sources
        class_count = max_sources - min_sources + 1

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 16)),
        )
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 16, 256)
        self.act1 = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.4)

        self.main_head = nn.Linear(256, class_count)
        self.aux_head = nn.Linear(256, 13)      # 1..12, >12
        self.family_head = nn.Linear(256, 2)    # 0=fsd, 1=speech

    def forward(self, x):
        feat = self.features(x)
        emb = self.act1(self.fc1(self.flat(feat)))
        emb_d = self.drop(emb)
        return self.main_head(emb_d), self.aux_head(emb_d), self.family_head(emb_d)

    def load_from_base(self, base_state_dict):
        own_state = self.state_dict()
        for k, v in base_state_dict.items():
            if k.startswith("features.") and k in own_state and own_state[k].shape == v.shape:
                own_state[k].copy_(v)
            elif k == "classifier.1.weight" and own_state["fc1.weight"].shape == v.shape:
                own_state["fc1.weight"].copy_(v)
            elif k == "classifier.1.bias" and own_state["fc1.bias"].shape == v.shape:
                own_state["fc1.bias"].copy_(v)
            elif k == "classifier.4.weight" and own_state["main_head.weight"].shape == v.shape:
                own_state["main_head.weight"].copy_(v)
            elif k == "classifier.4.bias" and own_state["main_head.bias"].shape == v.shape:
                own_state["main_head.bias"].copy_(v)

    def export_cnn_state(self) -> dict:
        own = self.state_dict()
        return {
            **{k: v.cpu() for k, v in own.items() if k.startswith("features.")},
            "classifier.1.weight": own["fc1.weight"].cpu(),
            "classifier.1.bias": own["fc1.bias"].cpu(),
            "classifier.4.weight": own["main_head.weight"].cpu(),
            "classifier.4.bias": own["main_head.bias"].cpu(),
        }


# ---------------------------------------------------------------- Real-Condition + Spectral Augmentation
def apply_real_and_spec_augment(mel_batch: torch.Tensor) -> torch.Tensor:
    """Apply SpecAugment + real-recording condition augmentations in normalized log-mel space [-1, 0].

    Includes:
    1. Microphone gain offset & dynamic range compression/expansion
    2. Room reverb smearing (temporal exponential decay filter across frames)
    3. Additive background noise floor
    4. High-frequency band-limit / roll-off
    5. Time shift (cyclic roll)
    6. SpecAugment frequency & time masking
    """
    aug = mel_batch.clone()
    bsz, _, n_mels, n_frames = aug.shape
    device = aug.device

    # 1. Mic gain / dynamic scale per sample
    if random.random() < 0.7:
        gain = torch.empty((bsz, 1, 1, 1), device=device).uniform_(0.88, 1.12)
        offset = torch.empty((bsz, 1, 1, 1), device=device).uniform_(-0.08, 0.08)
        aug = (aug * gain + offset).clamp(-1.0, 0.0)

    # 2. Simulated room reverb temporal smearing (causal exponential smoothing across time frames)
    if random.random() < 0.45:
        alpha = random.uniform(0.15, 0.35)
        shifted1 = torch.roll(aug, shifts=1, dims=-1)
        shifted2 = torch.roll(aug, shifts=2, dims=-1)
        aug = ((1.0 - alpha) * aug + (alpha * 0.7) * shifted1 + (alpha * 0.3) * shifted2).clamp(-1.0, 0.0)

    # 3. Additive background noise floor
    if random.random() < 0.5:
        noise_level = random.uniform(-0.92, -0.72)
        noise = noise_level + torch.randn_like(aug) * 0.04
        aug = torch.maximum(aug, noise).clamp(-1.0, 0.0)

    # 4. Band-limit / top mel bin roll-off (simulating room/mic high-freq attenuation)
    if random.random() < 0.35:
        cut_bins = random.randint(2, 6)
        atten = torch.linspace(0.0, -0.35, cut_bins, device=device).view(1, 1, cut_bins, 1)
        aug[:, :, -cut_bins:, :] = (aug[:, :, -cut_bins:, :] + atten).clamp(-1.0, 0.0)

    # 5. Per-item SpecAugment & time-shift
    for b in range(bsz):
        if random.random() < 0.5:
            shift = random.randint(-28, 28)
            aug[b] = torch.roll(aug[b], shifts=shift, dims=-1)
        if random.random() < 0.6:
            f_len = random.randint(1, 8)
            f0 = random.randint(0, n_mels - f_len)
            aug[b, :, f0:f0 + f_len, :] = -1.0
        if random.random() < 0.6:
            t_len = random.randint(1, 32)
            t0 = random.randint(0, n_frames - t_len)
            aug[b, :, :, t0:t0 + t_len] = -1.0

    return aug


# ---------------------------------------------------------------- Dataset & Sampler
class CombinedCounterDataset(Dataset):
    def __init__(self, items: List[Dict], feature_map: Dict[str, torch.Tensor]):
        self.items = items
        self.features = [feature_map[str(it["path"])] for it in items]
        self.labels = np.array([int(it["num_sources"]) for it in items], dtype=np.int64)
        self.families = np.array([1 if it["family"] == "speech" else 0 for it in items], dtype=np.int64)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.families[idx]


class FamilyKBalancedSampler(Sampler):
    """Balances sampling across both (family, K) strata so speech and FSD K values get equal weight."""

    def __init__(self, labels: np.ndarray, families: np.ndarray, per_stratum: int = 30, seed: int = 42):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.families = np.asarray(families, dtype=np.int64)
        self.per_stratum = per_stratum
        self.seed = seed
        self.epoch = 0

        self.strata = defaultdict(list)
        for idx, (k, fam) in enumerate(zip(self.labels, self.families)):
            self.strata[(int(fam), int(k))].append(idx)
        self.strata = {key: np.array(idxs, dtype=np.int64) for key, idxs in self.strata.items() if len(idxs) > 0}
        self.total_samples = len(self.strata) * self.per_stratum

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        chosen = []
        for _, idxs in self.strata.items():
            replace = len(idxs) < self.per_stratum
            sampled = rng.choice(idxs, size=self.per_stratum, replace=replace)
            chosen.extend(sampled.tolist())
        rng.shuffle(chosen)
        return iter(chosen)


def select_training_subset(all_records: List[Dict], max_per_folder_split: Dict[str, int], seed: int = 42) -> List[Dict]:
    """Select balanced subset per (rel_folder, split) to keep spectrogram caching fast and representative."""
    rng = random.Random(seed)
    by_group = defaultdict(list)
    for r in all_records:
        by_group[(r["rel_folder"], r["split"])].append(r)

    selected = []
    for (folder, split), recs in sorted(by_group.items()):
        cap = max_per_folder_split.get(split, len(recs))
        if len(recs) <= cap:
            selected.extend(recs)
        else:
            # Stratify by num_sources within folder
            by_k = defaultdict(list)
            for r in recs:
                by_k[r["num_sources"]].append(r)
            per_k = max(1, cap // len(by_k))
            picked = []
            for k, krecs in sorted(by_k.items()):
                picked.extend(rng.sample(krecs, min(len(krecs), per_k)))
            if len(picked) < cap:
                rem = [r for r in recs if r not in picked]
                picked.extend(rng.sample(rem, min(len(rem), cap - len(picked))))
            selected.extend(picked[:cap])
    return selected


def build_or_load_feature_cache(records: List[Dict], smoke: bool = False) -> Dict[str, torch.Tensor]:
    """Compute or load cached [1, 64, 501] log-mel tensors keyed by str(path)."""
    cache: Dict[str, torch.Tensor] = {}
    if not smoke and CACHE_PATH.is_file():
        try:
            cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
            print(f"Loaded {len(cache)} pre-cached spectrograms from {CACHE_PATH.name}", flush=True)
        except Exception:
            cache = {}

    missing = [r for r in records if str(r["path"]) not in cache]
    if missing:
        print(f"Computing {len(missing)} log-mel spectrograms...", flush=True)
        t0 = time.time()
        for i, r in enumerate(missing):
            wav, _ = load_audio(r["path"])
            cache[str(r["path"])] = waveform_to_logmel(wav)
            if (i + 1) % 400 == 0 or (i + 1) == len(missing):
                print(f"  [{i + 1}/{len(missing)}] ({time.time() - t0:.1f}s)", flush=True)
        if not smoke:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(cache, CACHE_PATH)
            print(f"Saved feature cache ({len(cache)} total items) to {CACHE_PATH}", flush=True)
    return cache


def train_stage_model(
    train_items: List[Dict],
    val_items: List[Dict],
    feature_map: Dict[str, torch.Tensor],
    min_k: int,
    max_k: int,
    epochs: int,
    batch_size: int,
    per_stratum: int,
    lr: float,
    seed: int,
    stage_name: str,
    init_state: dict | None = None,
) -> Tuple[dict, dict, dict]:
    """Train a CounterWithAuxHead on [min_k, max_k] and fit a validation-only Isotonic Calibrator."""
    train_sub = [it for it in train_items if min_k <= it["num_sources"] <= max_k]
    val_sub = [it for it in val_items if min_k <= it["num_sources"] <= max_k]

    train_ds = CombinedCounterDataset(train_sub, feature_map)
    val_ds = CombinedCounterDataset(val_sub, feature_map)

    sampler = FamilyKBalancedSampler(train_ds.labels, train_ds.families, per_stratum=per_stratum, seed=seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = CounterWithAuxHead(min_k, max_k).to(DEVICE)
    if init_state is not None:
        model.load_from_base(init_state)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    class_values = torch.arange(min_k, max_k + 1, dtype=torch.float32, device=DEVICE)

    best_score = -1e9
    best_state = None
    best_calibrator = None
    best_metrics = {}

    print(f"\n--- Training {stage_name} (K={min_k}..{max_k}, train={len(train_sub)}, val={len(val_sub)}, epochs={epochs}) ---", flush=True)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, seen = 0.0, 0
        for feats, labels, fams in train_loader:
            feats = apply_real_and_spec_augment(feats).to(DEVICE)
            labels = labels.to(DEVICE)
            fams = fams.to(DEVICE)

            target_idx = labels - min_k
            aux_bins = torch.clamp(labels - 1, 0, 12).long()

            optimizer.zero_grad()
            main_logits, aux_logits, fam_logits = model(feats)
            loss_ce = F.cross_entropy(main_logits, target_idx)
            loss_aux = F.cross_entropy(aux_logits, aux_bins)
            loss_fam = F.cross_entropy(fam_logits, fams)

            probs = F.softmax(main_logits, dim=1)
            pred_exp = (probs * class_values).sum(dim=1)
            loss_reg = F.smooth_l1_loss(pred_exp, labels.float())

            loss = loss_reg + 0.7 * loss_ce + 0.3 * loss_aux + 0.15 * loss_fam
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item() * len(labels)
            seen += len(labels)

        scheduler.step()
        train_loss = total_loss / max(seen, 1)

        # Validation pass
        model.eval()
        val_raw, val_true, val_fam = [], [], []
        with torch.no_grad():
            for feats, labels, fams in val_loader:
                feats = feats.to(DEVICE)
                main_logits, _, _ = model(feats)
                probs = F.softmax(main_logits, dim=1)
                exp_val = (probs * class_values).sum(dim=1).cpu().numpy()
                val_raw.extend(exp_val.tolist())
                val_true.extend(labels.numpy().tolist())
                val_fam.extend(fams.numpy().tolist())

        val_raw = np.array(val_raw, dtype=np.float64)
        val_true = np.array(val_true, dtype=np.int64)
        val_fam = np.array(val_fam, dtype=np.int64)

        iso = IsotonicRegression(y_min=min_k, y_max=max_k, out_of_bounds="clip")
        iso.fit(val_raw, val_true)
        pred_cal = np.clip(np.round(iso.predict(val_raw)).astype(int), min_k, max_k)

        exact = float(np.mean(pred_cal == val_true))
        within1 = float(np.mean(np.abs(pred_cal - val_true) <= 1))
        mae = float(np.mean(np.abs(pred_cal - val_true)))

        sp_mask = val_fam == 1
        fsd_mask = val_fam == 0
        sp_exact = float(np.mean(pred_cal[sp_mask] == val_true[sp_mask])) if np.any(sp_mask) else 0.0
        fsd_exact = float(np.mean(pred_cal[fsd_mask] == val_true[fsd_mask])) if np.any(fsd_mask) else 0.0

        # Score balances exact accuracy, within-1 accuracy, and low MAE
        score = exact + 0.5 * within1 - 0.05 * mae
        elapsed = time.time() - t0
        print(
            f"  [{stage_name}] Epoch {epoch:02d}/{epochs:02d} ({elapsed:.1f}s) | loss={train_loss:.4f} | "
            f"Val Exact={exact*100:.1f}% (Speech={sp_exact*100:.1f}%, FSD={fsd_exact*100:.1f}%) | "
            f"±1={within1*100:.1f}% | MAE={mae:.3f}",
            flush=True,
        )

        if score > best_score:
            best_score = score
            best_state = model.export_cnn_state()
            best_calibrator = {
                "x_thresholds": iso.X_thresholds_.tolist(),
                "y_thresholds": iso.y_thresholds_.tolist(),
            }
            best_metrics = {
                "epoch": epoch,
                "val_exact_accuracy": round(exact, 4),
                "val_within_one": round(within1, 4),
                "val_mae": round(mae, 4),
                "val_speech_exact": round(sp_exact, 4),
                "val_fsd_exact": round(fsd_exact, 4),
            }

    return best_state, best_calibrator, best_metrics


def evaluate_two_stage_on_items(
    items: List[Dict],
    feature_map: Dict[str, torch.Tensor],
    general_state: dict,
    general_cal: dict,
    sparse_state: dict,
    sparse_cal: dict,
) -> Dict:
    """Evaluate the combined General + Sparse Specialist (K<=8) pipeline on a set of items."""
    gen_model = SourceCountCNN(MIN_SOURCES, MAX_SOURCES).to(DEVICE)
    gen_model.load_state_dict(general_state)
    gen_model.eval()

    sp_model = SourceCountCNN(MIN_SOURCES, SPARSE_MAX).to(DEVICE)
    sp_model.load_state_dict(sparse_state)
    sp_model.eval()

    gen_vals = torch.arange(MIN_SOURCES, MAX_SOURCES + 1, dtype=torch.float32, device=DEVICE)
    sp_vals = torch.arange(MIN_SOURCES, SPARSE_MAX + 1, dtype=torch.float32, device=DEVICE)

    gx, gy = np.array(general_cal["x_thresholds"]), np.array(general_cal["y_thresholds"])
    sx, sy = np.array(sparse_cal["x_thresholds"]), np.array(sparse_cal["y_thresholds"])

    preds, trues, fams = [], [], []
    with torch.no_grad():
        for it in items:
            feat = feature_map[str(it["path"])].unsqueeze(0).to(DEVICE)
            g_probs = F.softmax(gen_model(feat), dim=1)[0]
            g_raw = float(torch.sum(g_probs * gen_vals).item())
            g_cal = float(np.interp(g_raw, gx, gy))

            if g_cal <= 8.5:
                s_probs = F.softmax(sp_model(feat), dim=1)[0]
                s_raw = float(torch.sum(s_probs * sp_vals).item())
                s_cal = float(np.interp(s_raw, sx, sy))
                pred_k = int(np.clip(round(s_cal), MIN_SOURCES, SPARSE_MAX))
            else:
                pred_k = int(np.clip(round(g_cal), MIN_SOURCES, MAX_SOURCES))

            preds.append(pred_k)
            trues.append(int(it["num_sources"]))
            fams.append(it["family"])

    preds_arr = np.array(preds, dtype=int)
    trues_arr = np.array(trues, dtype=int)
    err = preds_arr - trues_arr
    return {
        "n": len(trues_arr),
        "exact_accuracy": round(float(np.mean(err == 0)), 4),
        "within_one": round(float(np.mean(np.abs(err) <= 1)), 4),
        "mae": round(float(np.mean(np.abs(err))), 4),
        "bias": round(float(np.mean(err)), 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="2 epochs on 50 mixtures")
    parser.add_argument("--full", action="store_true", help="Full training across Speech + FSD")
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    seed = args.seed or cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("=" * 80)
    print("TASK 2: Retraining Counter on Speech (Org_*) + FSD Mixtures")
    print("=" * 80)

    # Backup existing checkpoint before overwriting
    if args.output.is_file() and not args.smoke:
        shutil.copy2(args.output, BACKUP_PATH)
        print(f"Backed up existing checkpoint to {BACKUP_PATH}")

    all_records = scan_all_mixtures(cfg, include_duplicates=False)
    if args.smoke:
        print("[Smoke Mode] Selecting ~50 train, 20 val, 20 test mixtures across FSD + Speech...", flush=True)
        caps = {"train": 5, "val": 2, "test": 2}
        subset = select_training_subset(all_records, caps, seed=seed)
        epochs = 2
        per_stratum = 4
    else:
        # Full mode: all 1,285 val and 1,285 test clips + balanced 3,250 train clips (250 per folder)
        caps = {"train": 250, "val": 1000, "test": 1000}
        subset = select_training_subset(all_records, caps, seed=seed)
        epochs = args.epochs
        per_stratum = 32

    feature_map = build_or_load_feature_cache(subset, smoke=args.smoke)

    train_items = [r for r in subset if r["split"] == "train"]
    val_items = [r for r in subset if r["split"] == "val"]

    init_state = None
    if BACKUP_PATH.is_file():
        try:
            ck = torch.load(BACKUP_PATH, map_location="cpu", weights_only=False)
            init_state = ck.get("model_state_dict")
        except Exception:
            init_state = None

    # 1. Train General Counter (K = 2..31)
    gen_state, gen_cal, gen_metrics = train_stage_model(
        train_items, val_items, feature_map,
        min_k=MIN_SOURCES, max_k=MAX_SOURCES,
        epochs=epochs, batch_size=args.batch_size, per_stratum=per_stratum,
        lr=args.lr, seed=seed, stage_name="General-Counter", init_state=init_state,
    )

    # 2. Train Sparse Specialist (K = 2..8)
    sp_state, sp_cal, sp_metrics = train_stage_model(
        train_items, val_items, feature_map,
        min_k=MIN_SOURCES, max_k=SPARSE_MAX,
        epochs=epochs, batch_size=args.batch_size, per_stratum=per_stratum * 2,
        lr=args.lr, seed=seed + 7, stage_name="Sparse-Specialist-K2..8", init_state=gen_state,
    )

    # Evaluate two-stage on validation
    val_two_stage = evaluate_two_stage_on_items(val_items, feature_map, gen_state, gen_cal, sp_state, sp_cal)
    print(
        f"\n[Validation Two-Stage Combined Result] n={val_two_stage['n']} | "
        f"Exact={val_two_stage['exact_accuracy']*100:.1f}% | "
        f"Within ±1={val_two_stage['within_one']*100:.1f}% | MAE={val_two_stage['mae']:.3f}"
    )

    # 3. Held-Out-Speaker Generalisation Experiment (Unseen Speakers: Speaker_4_1, Speaker8, Speaker9)
    print("\n--- Held-Out-Speaker Split Experiment (Unseen Speakers: Speaker_4_1, Speaker8, Speaker9) ---")
    ho_train = [r for r in train_items if not r.get("held_out_speaker", False)]
    ho_val = [r for r in val_items if not r.get("held_out_speaker", False)]
    ho_unseen_test = [r for r in subset if r["family"] == "speech" and r.get("held_out_speaker", False) and r["split"] == "test"]
    if not ho_unseen_test:
        ho_unseen_test = [r for r in subset if r["family"] == "speech" and r.get("held_out_speaker", False)]

    ho_epochs = 2 if args.smoke else max(6, epochs // 2)
    ho_gen_state, ho_gen_cal, _ = train_stage_model(
        ho_train, ho_val, feature_map,
        min_k=MIN_SOURCES, max_k=MAX_SOURCES,
        epochs=ho_epochs, batch_size=args.batch_size, per_stratum=max(4, per_stratum // 2),
        lr=args.lr, seed=seed + 19, stage_name="HeldOutSpeaker-General", init_state=init_state,
    )
    ho_sp_state, ho_sp_cal, _ = train_stage_model(
        ho_train, ho_val, feature_map,
        min_k=MIN_SOURCES, max_k=SPARSE_MAX,
        epochs=ho_epochs, batch_size=args.batch_size, per_stratum=per_stratum,
        lr=args.lr, seed=seed + 23, stage_name="HeldOutSpeaker-Sparse", init_state=ho_gen_state,
    )
    ho_eval = evaluate_two_stage_on_items(ho_unseen_test, feature_map, ho_gen_state, ho_gen_cal, ho_sp_state, ho_sp_cal)
    print(
        f"[Held-Out Unseen Speaker Generalisation] n={ho_eval['n']} mixtures | "
        f"Exact={ho_eval['exact_accuracy']*100:.1f}% | "
        f"Within ±1={ho_eval['within_one']*100:.1f}% | MAE={ho_eval['mae']:.3f} | Bias={ho_eval['bias']:+.3f}"
    )

    out_ckpt = {
        "model_state_dict": gen_state,
        "sparse_specialist_state_dict": sp_state,
        "sparse_min_sources": MIN_SOURCES,
        "sparse_max_sources": SPARSE_MAX,
        "min_sources": MIN_SOURCES,
        "max_sources": MAX_SOURCES,
        "sample_rate": SAMPLE_RATE,
        "clip_seconds": CLIP_SECONDS,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "feature_version": 2,
        "architecture": "SourceCountCNN-v3-speech-fsd-specialist",
        "decision_rule": "calibrated_isotonic",
        "trained": True,
        "epoch": gen_metrics["epoch"],
        "sparse_epoch": sp_metrics["epoch"],
        "val_exact_accuracy": val_two_stage["exact_accuracy"],
        "val_within_one": val_two_stage["within_one"],
        "val_mae": val_two_stage["mae"],
        "val_macro_mae": val_two_stage["mae"],
        "train_samples": len(train_items),
        "val_samples": len(val_items),
        "calibrator": gen_cal,
        "sparse_calibrator": sp_cal,
        "held_out_speaker_eval": {
            "held_out_speakers": cfg.get("held_out_speakers", ["Speaker_4_1", "Speaker8", "Speaker9"]),
            **ho_eval,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    out_path = BASE_DIR / "work" / "smoke_counter.pt" if args.smoke else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, out_path)
    print(f"\nSaved counter checkpoint (with Sparse Specialist + Calibrators) to {out_path}", flush=True)


if __name__ == "__main__":
    main()
