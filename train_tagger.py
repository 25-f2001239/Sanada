"""Group + Fine-class Multi-Label Tagger for SANADA/EchoCount (Task 3).

Trains a dual-head log-mel CNN (reusing SourceCountCNN trunk) on FSD_* + Org_* mixtures:
- Group head (5 coarse groups from taxonomy.json):
    1. Human voice         (Org_* speech mixtures + FSD Applause/Cough/Laughter)
    2. Musical instrument  (14 FSD instrument classes)
    3. Animal              (Bark, Meow)
    4. Household/tools     (10 FSD household/tool classes)
    5. Environment/other   (Bus, Fireworks, Gunshot_or_gunfire)
  Never trains Speaker_4_X or SpeakerN identity tags.
- Fine head (32 FSD acoustic event classes; masked out on pure speech Org_* mixtures so
  speech recordings do NOT trigger false FSD fine tags like Telephone/Scissors/Cello).
- Tunes per-group and per-class F1 thresholds strictly on VALIDATION, evaluates once on TEST.
- Backs up existing checkpoints/tagger.pt to checkpoints/tagger_backup.pt before saving.

Usage:
    python train_tagger.py --smoke
    python train_tagger.py --full
"""
from __future__ import annotations

import argparse
import copy
import json
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset

from dataset_config import load_config, load_taxonomy, scan_all_mixtures
from model import DEVICE, N_MELS, N_FFT, HOP_LENGTH, SAMPLE_RATE, CLIP_SECONDS, waveform_to_logmel
from retrain_counter import apply_real_and_spec_augment, build_or_load_feature_cache, select_training_subset

BASE_DIR = Path(__file__).resolve().parent
TAGGER_PATH = BASE_DIR / "checkpoints" / "tagger.pt"
TAGGER_BACKUP = BASE_DIR / "checkpoints" / "tagger_backup.pt"


def build_vocab_and_taxonomy() -> Tuple[List[str], List[str], Dict[str, str], Dict[str, List[str]]]:
    tax = load_taxonomy()
    groups: List[str] = tax["groups"]
    group_to_fine: Dict[str, List[str]] = tax["group_to_fine_classes"]
    fine_to_group: Dict[str, str] = {}
    fine_vocab_set = set()
    for grp, f_list in group_to_fine.items():
        for fc in f_list:
            fine_to_group[fc] = grp
            fine_vocab_set.add(fc)
    fine_vocab = sorted(fine_vocab_set)
    return groups, fine_vocab, fine_to_group, group_to_fine


class GroupFineTaggerCNN(nn.Module):
    """Shared CNN trunk with 5-group head and 32-fine-class head."""

    def __init__(self, num_groups: int, num_fine: int):
        super().__init__()
        self.num_groups = int(num_groups)
        self.num_fine = int(num_fine)
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
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.group_head = nn.Linear(256, self.num_groups)
        self.fine_head = nn.Linear(256, self.num_fine)

    def forward(self, x):
        emb = self.shared_fc(self.features(x))
        return self.group_head(emb), self.fine_head(emb)


class GroupFineDataset(Dataset):
    def __init__(
        self,
        items: List[Dict],
        feature_map: Dict[str, torch.Tensor],
        groups: List[str],
        fine_vocab: List[str],
        fine_to_group: Dict[str, str],
    ):
        self.items = items
        self.features = []
        self.group_targets = []
        self.fine_targets = []

        grp_idx = {g: i for i, g in enumerate(groups)}
        fine_idx = {c: i for i, c in enumerate(fine_vocab)}
        human_grp_i = grp_idx["Human voice"]

        for it in items:
            self.features.append(feature_map[str(it["path"])])
            g_vec = torch.zeros(len(groups), dtype=torch.float32)
            f_vec = torch.zeros(len(fine_vocab), dtype=torch.float32)

            if it["family"] == "speech":
                # Speech mixture (Org_*): positive for "Human voice" group ONLY; all 32 FSD fine classes are 0
                g_vec[human_grp_i] = 1.0
            else:
                for lbl in it["labels"]:
                    if lbl.startswith("Speaker"):
                        g_vec[human_grp_i] = 1.0
                        continue
                    if lbl in fine_idx:
                        f_vec[fine_idx[lbl]] = 1.0
                        grp_name = fine_to_group.get(lbl)
                        if grp_name in grp_idx:
                            g_vec[grp_idx[grp_name]] = 1.0

            self.group_targets.append(g_vec)
            self.fine_targets.append(f_vec)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.features[idx], self.group_targets[idx], self.fine_targets[idx]


def compute_pos_weight(targets: List[torch.Tensor], num_cols: int) -> torch.Tensor:
    pos = torch.zeros(num_cols)
    for t in targets:
        pos += t
    n = max(len(targets), 1)
    neg = n - pos
    return (neg / pos.clamp(min=1.0)).clamp(1.0, 35.0)


def eval_dual_heads(model: GroupFineTaggerCNN, loader: DataLoader, device: torch.device):
    model.eval()
    g_logits_all, g_labels_all = [], []
    f_logits_all, f_labels_all = [], []
    with torch.no_grad():
        for feats, g_lbl, f_lbl in loader:
            feats = feats.to(device)
            gl, fl = model(feats)
            g_logits_all.append(gl.cpu())
            g_labels_all.append(g_lbl)
            f_logits_all.append(fl.cpu())
            f_labels_all.append(f_lbl)

    g_logits = torch.cat(g_logits_all).numpy()
    g_labels = torch.cat(g_labels_all).numpy()
    f_logits = torch.cat(f_logits_all).numpy()
    f_labels = torch.cat(f_labels_all).numpy()

    g_probs = 1.0 / (1.0 + np.exp(-g_logits))
    f_probs = 1.0 / (1.0 + np.exp(-f_logits))

    g_aps = [
        float(average_precision_score(g_labels[:, c], g_probs[:, c]))
        for c in range(g_labels.shape[1])
        if g_labels[:, c].sum() > 0
    ]
    f_aps = [
        float(average_precision_score(f_labels[:, c], f_probs[:, c]))
        for c in range(f_labels.shape[1])
        if f_labels[:, c].sum() > 0
    ]
    return (
        float(np.mean(g_aps)) if g_aps else 0.0,
        float(np.mean(f_aps)) if f_aps else 0.0,
        g_probs,
        g_labels,
        f_probs,
        f_labels,
    )


def tune_thresholds(probs: np.ndarray, labels: np.ndarray) -> List[float]:
    thresholds = []
    for c in range(probs.shape[1]):
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.15, 0.92, 0.05):
            preds = (probs[:, c] >= t).astype(float)
            tp = (preds * labels[:, c]).sum()
            fp = preds.sum() - tp
            fn = labels[:, c].sum() - tp
            prec = tp / max(tp + fp, 1.0)
            rec = tp / max(tp + fn, 1.0)
            f1 = 2 * prec * rec / max(prec + rec, 1e-6)
            if f1 > best_f1:
                best_f1, best_t = f1, float(round(t, 2))
        thresholds.append(best_t)
    return thresholds


def compute_f1s(probs: np.ndarray, labels: np.ndarray, thresholds: List[float]) -> Tuple[float, List[float]]:
    f1s = []
    for c in range(probs.shape[1]):
        preds = (probs[:, c] >= thresholds[c]).astype(float)
        tp = (preds * labels[:, c]).sum()
        fp = preds.sum() - tp
        fn = labels[:, c].sum() - tp
        prec = tp / max(tp + fp, 1.0)
        rec = tp / max(tp + fn, 1.0)
        f1s.append(float(2 * prec * rec / max(prec + rec, 1e-6)))
    return float(np.mean(f1s)), f1s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="2 epochs on ~50 mixtures")
    parser.add_argument("--full", action="store_true", help="Full training across Speech + FSD")
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", type=Path, default=TAGGER_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    seed = args.seed or cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("=" * 80)
    print("TASK 3: Training Dual-Head Group + Fine Class Tagger")
    print("=" * 80)

    if args.output.is_file() and not args.smoke:
        shutil.copy2(args.output, TAGGER_BACKUP)
        print(f"Backed up existing tagger checkpoint to {TAGGER_BACKUP}")

    groups, fine_vocab, fine_to_group, group_to_fine = build_vocab_and_taxonomy()
    print(f"Groups ({len(groups)}): {groups}")
    print(f"Fine classes ({len(fine_vocab)}): {', '.join(fine_vocab)}")

    all_records = scan_all_mixtures(cfg, include_duplicates=False)
    if args.smoke:
        caps = {"train": 5, "val": 2, "test": 2}
        subset = select_training_subset(all_records, caps, seed=seed)
        epochs = 2
    else:
        caps = {"train": 250, "val": 1000, "test": 1000}
        subset = select_training_subset(all_records, caps, seed=seed)
        epochs = args.epochs

    feature_map = build_or_load_feature_cache(subset, smoke=args.smoke)

    train_items = [r for r in subset if r["split"] == "train"]
    val_items = [r for r in subset if r["split"] == "val"]
    test_items = [r for r in subset if r["split"] == "test"]

    train_ds = GroupFineDataset(train_items, feature_map, groups, fine_vocab, fine_to_group)
    val_ds = GroupFineDataset(val_items, feature_map, groups, fine_vocab, fine_to_group)
    test_ds = GroupFineDataset(test_items, feature_map, groups, fine_vocab, fine_to_group)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = GroupFineTaggerCNN(len(groups), len(fine_vocab)).to(DEVICE)
    g_pos_weight = compute_pos_weight(train_ds.group_targets, len(groups)).to(DEVICE)
    f_pos_weight = compute_pos_weight(train_ds.fine_targets, len(fine_vocab)).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_score = -1.0
    best_state = None
    best_epoch = 1

    print(f"\nTraining on {DEVICE}: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}, epochs={epochs}")
    print("-" * 80, flush=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, seen = 0.0, 0
        for feats, g_lbl, f_lbl in train_loader:
            feats = apply_real_and_spec_augment(feats).to(DEVICE)
            g_lbl, f_lbl = g_lbl.to(DEVICE), f_lbl.to(DEVICE)

            optimizer.zero_grad()
            g_logits, f_logits = model(feats)
            loss_g = F.binary_cross_entropy_with_logits(g_logits, g_lbl, pos_weight=g_pos_weight)
            loss_f = F.binary_cross_entropy_with_logits(f_logits, f_lbl, pos_weight=f_pos_weight)
            loss = 1.2 * loss_g + loss_f
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item() * len(g_lbl)
            seen += len(g_lbl)

        scheduler.step()
        train_loss = total_loss / max(seen, 1)
        val_g_map, val_f_map, _, _, _, _ = eval_dual_heads(model, val_loader, DEVICE)
        score = 0.6 * val_g_map + 0.4 * val_f_map
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | loss={train_loss:.4f} | "
            f"val_group_mAP={val_g_map:.4f} | val_fine_mAP={val_f_map:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    val_g_map, val_f_map, val_g_probs, val_g_labels, val_f_probs, val_f_labels = eval_dual_heads(model, val_loader, DEVICE)
    g_thresholds = tune_thresholds(val_g_probs, val_g_labels)
    f_thresholds = tune_thresholds(val_f_probs, val_f_labels)
    val_g_f1, val_g_f1_list = compute_f1s(val_g_probs, val_g_labels, g_thresholds)
    val_f_f1, val_f_f1_list = compute_f1s(val_f_probs, val_f_labels, f_thresholds)

    # One-time Test evaluation
    test_g_map, test_f_map, test_g_probs, test_g_labels, test_f_probs, test_f_labels = eval_dual_heads(model, test_loader, DEVICE)
    test_g_f1, test_g_f1_list = compute_f1s(test_g_probs, test_g_labels, g_thresholds)
    test_f_f1, test_f_f1_list = compute_f1s(test_f_probs, test_f_labels, f_thresholds)

    print("\n--- Per-Group Validation & Test Metrics ---")
    group_metrics = []
    for i, grp in enumerate(groups):
        v_ap = float(average_precision_score(val_g_labels[:, i], val_g_probs[:, i])) if val_g_labels[:, i].sum() > 0 else 0.0
        t_ap = float(average_precision_score(test_g_labels[:, i], test_g_probs[:, i])) if test_g_labels[:, i].sum() > 0 else 0.0
        group_metrics.append({
            "group": grp,
            "val_ap": round(v_ap, 4),
            "val_f1": round(val_g_f1_list[i], 4),
            "test_ap": round(t_ap, 4),
            "test_f1": round(test_g_f1_list[i], 4),
            "threshold": g_thresholds[i],
        })
        print(
            f"  {grp:<22} | Val AP={v_ap:.4f}, Val F1={val_g_f1_list[i]:.4f} | "
            f"Test AP={t_ap:.4f}, Test F1={test_g_f1_list[i]:.4f} | thresh={g_thresholds[i]:.2f}"
        )

    per_class_metrics = []
    for i, fc in enumerate(fine_vocab):
        v_ap = float(average_precision_score(val_f_labels[:, i], val_f_probs[:, i])) if val_f_labels[:, i].sum() > 0 else 0.0
        t_ap = float(average_precision_score(test_f_labels[:, i], test_f_probs[:, i])) if test_f_labels[:, i].sum() > 0 else 0.0
        per_class_metrics.append({
            "class": fc,
            "group": fine_to_group[fc],
            "val_ap": round(v_ap, 4),
            "val_f1": round(val_f_f1_list[i], 4),
            "test_ap": round(t_ap, 4),
            "test_f1": round(test_f_f1_list[i], 4),
            "threshold": f_thresholds[i],
        })

    worst_10 = sorted(per_class_metrics, key=lambda x: x["val_ap"])[:10]
    print("\n--- 10 Worst Fine Classes (by Validation AP) ---")
    for m in worst_10:
        print(f"  {m['class']:<24} ({m['group']:<18}) | Val AP={m['val_ap']:.4f} | Test AP={m['test_ap']:.4f} | Test F1={m['test_f1']:.4f}")

    checkpoint = {
        "architecture": "GroupFineTaggerCNN",
        "trained": True,
        "version": "v2-group-fine",
        "groups": groups,
        "group_thresholds": g_thresholds,
        "group_to_fine_classes": group_to_fine,
        "fine_to_group": fine_to_group,
        "num_classes": len(fine_vocab),
        "vocabulary": fine_vocab,
        "thresholds": f_thresholds,
        "model_state_dict": best_state,
        "sample_rate": SAMPLE_RATE,
        "clip_seconds": CLIP_SECONDS,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "feature_version": 2,
        "epoch": best_epoch,
        "val_group_macro_map": round(val_g_map, 4),
        "val_group_macro_f1": round(val_g_f1, 4),
        "test_group_macro_map": round(test_g_map, 4),
        "test_group_macro_f1": round(test_g_f1, 4),
        "val_macro_map": round(val_f_map, 4),
        "val_macro_f1": round(val_f_f1, 4),
        "test_macro_map": round(test_f_map, 4),
        "test_macro_f1": round(test_f_f1, 4),
        "group_metrics": group_metrics,
        "per_class_metrics": per_class_metrics,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    out_path = BASE_DIR / "work" / "smoke_tagger.pt" if args.smoke else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out_path)
    print(f"\nSaved Group + Fine Tagger checkpoint to {out_path}")
    print(f"  Group Val mAP={val_g_map:.4f}, Val F1={val_g_f1:.4f} | Test mAP={test_g_map:.4f}, Test F1={test_g_f1:.4f}")
    print(f"  Fine  Val mAP={val_f_map:.4f}, Val F1={val_f_f1:.4f} | Test mAP={test_f_map:.4f}, Test F1={test_f_f1:.4f}")


if __name__ == "__main__":
    main()
