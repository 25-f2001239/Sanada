"""Train an eight-second source-count CNN with ordinal-aware validation."""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from model import (
    CLIP_SECONDS,
    DEVICE,
    FEATURE_VERSION,
    HOP_LENGTH,
    MAX_SOURCES,
    MODEL_PATH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    SourceCountCNN,
    load_audio,
    waveform_to_logmel,
)


class MixtureDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)
        self.feature_cache = [None] * len(self.frame)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        features = self.feature_cache[index]
        if features is None:
            waveform, _ = load_audio(Path(row["_resolved_path"]))
            features = waveform_to_logmel(waveform).to(torch.float16)
            self.feature_cache[index] = features
        return features, torch.tensor(int(row["num_sources"]), dtype=torch.long)


class BalancedClassSampler(Sampler):
    """Draw up to N examples per class without repeats within one epoch."""
    def __init__(self, labels, min_sources, max_sources, per_class, seed=42):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.min_sources = int(min_sources)
        self.max_sources = int(max_sources)
        self.per_class = int(per_class)
        self.seed = int(seed)
        self.epoch = 0
        self.class_indices = {
            count: np.flatnonzero(self.labels == count)
            for count in range(self.min_sources, self.max_sources + 1)
        }
        missing = [count for count, ids in self.class_indices.items() if not len(ids)]
        if missing:
            raise ValueError(f"Training set has no examples for source count(s): {missing}")
        self.num_samples = sum(min(self.per_class, len(ids)) for ids in self.class_indices.values())

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        selected = []
        for ids in self.class_indices.values():
            selected.extend(rng.choice(ids, size=min(self.per_class, len(ids)), replace=False).tolist())
        rng.shuffle(selected)
        return iter(selected)

    def __len__(self):
        return self.num_samples


def resolve_paths(frame, metadata_path):
    base = Path(metadata_path).resolve().parent
    paths = []
    for raw in frame["mixture_path"].astype(str):
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = base / path
        paths.append(str(path.resolve()))
    frame = frame.copy()
    frame["_resolved_path"] = paths
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing audio file: {missing[0]}")
    return frame


def prepare_splits(frame, metadata_path, max_sources, min_sources=None):
    required = {"mixture_path", "num_sources"}
    if not required.issubset(frame.columns):
        missing = ", ".join(sorted(required - set(frame.columns)))
        raise ValueError(f"CSV is missing required column(s): {missing}")
    counts = pd.to_numeric(frame["num_sources"], errors="raise")
    if counts.isna().any() or not np.equal(counts.to_numpy(), np.floor(counts.to_numpy())).all():
        raise ValueError("Every num_sources value must be a whole number.")
    if ((counts < 1) | (counts > max_sources)).any():
        raise ValueError(f"Every source count must be between 1 and {max_sources}.")
    frame = frame.copy()
    frame["num_sources"] = counts.astype(int)
    frame = resolve_paths(frame, metadata_path)
    if "split" not in frame:
        raise ValueError("The metadata CSV must include train and val split labels.")
    splits = frame["split"].astype(str).str.strip().str.lower()
    train_frame = frame[splits == "train"].copy()
    val_frame = frame[splits.isin(["val", "validation"])].copy()
    test_frame = frame[splits == "test"].copy()
    if train_frame.empty or val_frame.empty:
        raise ValueError("Need non-empty train and validation sets.")
    if min_sources is None:
        min_sources = int(train_frame["num_sources"].min())
    if min_sources < 1 or max_sources < min_sources:
        raise ValueError("Invalid source-count range.")
    if ((frame["num_sources"] < min_sources) | (frame["num_sources"] > max_sources)).any():
        raise ValueError(f"Labels must be in {min_sources}..{max_sources}.")
    expected = set(range(min_sources, max_sources + 1))
    if set(train_frame["num_sources"].unique()) != expected:
        raise ValueError(f"Training counts must cover every class in {min_sources}..{max_sources}.")
    return train_frame, val_frame, test_frame, min_sources


def _ordinal_targets(labels, class_values, sigma=1.25):
    distances = labels.to(class_values.dtype).unsqueeze(1) - class_values.unsqueeze(0)
    targets = torch.exp(-0.5 * (distances / sigma).square())
    return targets / targets.sum(dim=1, keepdim=True)


def _augment(features):
    x = features.clone()
    batch, _, mel_bins, frames = x.shape
    for index in range(batch):
        if np.random.random() < 0.7:
            width = int(np.random.randint(8, 49))
            start = int(np.random.randint(0, max(1, frames - width)))
            x[index, :, :, start:start + width] = -1.0
        if np.random.random() < 0.7:
            width = int(np.random.randint(2, 11))
            start = int(np.random.randint(0, max(1, mel_bins - width)))
            x[index, :, start:start + width, :] = -1.0
    return x


def _mixup(features, labels, class_values):
    one_hot = F.one_hot(
        labels - int(class_values[0].item()), num_classes=len(class_values)
    ).to(features.dtype)
    if len(labels) < 2 or np.random.random() >= 0.5:
        return (
            features,
            _ordinal_targets(labels, class_values),
            one_hot,
            labels.to(torch.float32),
        )
    lam = float(np.random.beta(0.2, 0.2))
    permutation = torch.randperm(len(labels), device=labels.device)
    mixed_x = lam * features + (1.0 - lam) * features[permutation]
    target_a = _ordinal_targets(labels, class_values)
    target_b = _ordinal_targets(labels[permutation], class_values)
    mixed_targets = lam * target_a + (1.0 - lam) * target_b
    mixed_hard_targets = lam * one_hot + (1.0 - lam) * one_hot[permutation]
    mixed_counts = lam * labels.to(torch.float32) + (1.0 - lam) * labels[permutation].to(torch.float32)
    return mixed_x, mixed_targets, mixed_hard_targets, mixed_counts


def _metrics(predictions, labels, min_sources, max_sources):
    predictions = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    errors = predictions - labels
    class_maes = []
    class_exact = []
    for count in range(min_sources, max_sources + 1):
        mask = labels == count
        if mask.any():
            class_maes.append(float(np.mean(np.abs(errors[mask]))))
            class_exact.append(float(np.mean(predictions[mask] == labels[mask])))
    return {
        "mae": float(np.mean(np.abs(errors))),
        "macro_mae": float(np.mean(class_maes)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "exact_accuracy": float(np.mean(predictions == labels)),
        "macro_accuracy": float(np.mean(class_exact)),
        "within_one": float(np.mean(np.abs(errors) <= 1)),
    }


@torch.inference_mode()
def evaluate(model, loader, min_sources, max_sources, device):
    model.eval()
    all_logits, all_labels = [], []
    for features, labels in loader:
        all_logits.append(model(features.to(device, dtype=torch.float32)).cpu())
        all_labels.append(labels)
    logits, labels = torch.cat(all_logits), torch.cat(all_labels)
    probabilities = F.softmax(logits, dim=1)
    values = torch.arange(min_sources, max_sources + 1, dtype=probabilities.dtype)
    expected = (probabilities * values).sum(dim=1).round().clamp(min_sources, max_sources).numpy()
    argmax = values[logits.argmax(dim=1)].numpy()
    labels_np = labels.numpy()
    return {
        "expected": _metrics(expected, labels_np, min_sources, max_sources),
        "argmax": _metrics(argmax, labels_np, min_sources, max_sources),
        "labels": labels_np,
        "expected_predictions": expected,
        "argmax_predictions": argmax,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="CSV manifest containing train/val/test rows")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--samples-per-class", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-sources", type=int, default=None)
    parser.add_argument("--max-sources", type=int, default=MAX_SOURCES)
    parser.add_argument("--output", default=str(MODEL_PATH))
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.samples_per_class, args.patience) < 1 or args.lr <= 0:
        parser.error("epochs, batch size, samples per class, patience, and learning rate must be positive")
    torch.manual_seed(42)
    np.random.seed(42)

    metadata_path = Path(args.metadata).expanduser().resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
    frame = pd.read_csv(metadata_path)
    train_frame, val_frame, test_frame, min_sources = prepare_splits(
        frame, metadata_path, args.max_sources, args.min_sources
    )
    train_data = MixtureDataset(train_frame)
    val_data = MixtureDataset(val_frame)
    sampler = BalancedClassSampler(
        train_frame["num_sources"], min_sources, args.max_sources, args.samples_per_class
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SourceCountCNN(min_sources, args.max_sources).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=2e-5
    )
    class_values = torch.arange(min_sources, args.max_sources + 1, dtype=torch.float32, device=DEVICE)
    best_macro_mae = float("inf")
    best_exact = 0.0
    bad_epochs = 0
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + ".tmp")

    print(
        f"Device: {DEVICE}; train mixtures: {len(train_frame)}; validation mixtures: {len(val_frame)}; "
        f"test mixtures excluded: {len(test_frame)}"
    )
    print(
        f"One {CLIP_SECONDS:g}s clip per label; counts {min_sources}..{args.max_sources}; "
        f"{sampler.num_samples} balanced samples per epoch"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0
        for features, labels in train_loader:
            features = _augment(features.to(DEVICE, dtype=torch.float32))
            labels = labels.to(DEVICE)
            mixed_x, soft_targets, hard_targets, mixed_counts = _mixup(features, labels, class_values)
            logits = model(mixed_x)
            log_probs = F.log_softmax(logits, dim=1)
            ordinal_loss = -(soft_targets * log_probs).sum(dim=1)
            hard_loss = -(hard_targets * log_probs).sum(dim=1)
            expected = (F.softmax(logits, dim=1) * class_values).sum(dim=1)
            regression_loss = F.smooth_l1_loss(expected, mixed_counts, reduction="none", beta=1.0)
            loss = (0.65 * ordinal_loss + 0.15 * hard_loss + 0.20 * regression_loss).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            epoch_loss += loss.item() * len(labels)
            seen += len(labels)

        metrics = evaluate(model, val_loader, min_sources, args.max_sources, DEVICE)
        selected, argmax_metrics = metrics["expected"], metrics["argmax"]
        scheduler.step(selected["macro_mae"])
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_loss={epoch_loss/max(seen,1):.4f} "
            f"| val_MAE={selected['mae']:.3f} macro_MAE={selected['macro_mae']:.3f} "
            f"RMSE={selected['rmse']:.3f} exact={selected['exact_accuracy']:.3f} "
            f"within1={selected['within_one']:.3f} argmax_MAE={argmax_metrics['mae']:.3f} "
            f"lr={optimizer.param_groups[0]['lr']:.2g}"
        )

        improved = selected["macro_mae"] < best_macro_mae - 0.01
        if improved or (
            abs(selected["macro_mae"] - best_macro_mae) <= 0.01
            and selected["exact_accuracy"] > best_exact
        ):
            best_macro_mae = selected["macro_mae"]
            best_exact = selected["exact_accuracy"]
            bad_epochs = 0
            decision_rule = "expected" if selected["mae"] <= argmax_metrics["mae"] else "argmax"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "min_sources": min_sources,
                    "max_sources": args.max_sources,
                    "sample_rate": SAMPLE_RATE,
                    "clip_seconds": CLIP_SECONDS,
                    "n_mels": N_MELS,
                    "n_fft": N_FFT,
                    "hop_length": HOP_LENGTH,
                    "feature_version": FEATURE_VERSION,
                    "architecture": "SourceCountCNN8s",
                    "decision_rule": decision_rule,
                    "epoch": epoch,
                    "val_mae": selected["mae"],
                    "val_macro_mae": selected["macro_mae"],
                    "val_rmse": selected["rmse"],
                    "val_exact_accuracy": selected["exact_accuracy"],
                    "val_within_one": selected["within_one"],
                },
                temp_path,
            )
            os.replace(temp_path, output_path)
            print(f"  Saved best checkpoint ({decision_rule}): {output_path}")
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    print(
        f"Training finished. Best macro MAE={best_macro_mae:.3f}, exact accuracy={best_exact:.3f}; "
        "test split was excluded."
    )


if __name__ == "__main__":
    main()

