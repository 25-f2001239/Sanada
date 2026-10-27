"""Train the source-count CNN from a metadata CSV of labelled mixtures.

Expected CSV columns:
  mixture_path,num_sources
Optional:
  split  (values train/val/test)

Example:
  python train.py --metadata data/metadata.csv --epochs 25 --max-sources 32

The dataset should use pre-mixed audio with known source counts. Keep train/validation/test
splits disjoint at the source-file level to avoid leakage, as described in the supplied paper.
"""
import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from model import (
    SourceCountCNN, waveform_to_logmel, SAMPLE_RATE, CLIP_SECONDS,
    MAX_SOURCES, DEVICE, MODEL_PATH, load_audio
)

class MixtureDataset(Dataset):
    def __init__(self, frame, max_sources):
        self.frame = frame.reset_index(drop=True)
        self.max_sources = max_sources

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        path = Path(str(row["mixture_path"]))
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            # Also resolve paths relative to the metadata CSV's parent via the caller
            path = Path(str(row["_resolved_path"]))
        waveform, _ = load_audio(path)
        x = waveform_to_logmel(waveform).squeeze(0)  # [1, mel, frames]
        count = int(row["num_sources"])
        if count < 1 or count > self.max_sources:
            raise ValueError(f"num_sources={count} outside valid range 1..{self.max_sources}")
        return x, torch.tensor(count - 1, dtype=torch.long)


def resolve_paths(df, metadata_path):
    base = Path(metadata_path).resolve().parent
    resolved = []
    for raw in df["mixture_path"].astype(str):
        p = Path(raw)
        if not p.is_absolute():
            p = (base / p).resolve()
        resolved.append(str(p))
    df["_resolved_path"] = resolved
    missing = [p for p in resolved if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} audio paths could not be found. First missing path: {missing[0]}. "
            "Use paths relative to the metadata CSV or absolute paths."
        )
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="CSV with mixture_path and num_sources")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-sources", type=int, default=MAX_SOURCES)
    parser.add_argument("--output", default=str(MODEL_PATH))
    args = parser.parse_args()

    metadata_path = Path(args.metadata).resolve()
    df = pd.read_csv(metadata_path)
    required = {"mixture_path", "num_sources"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
    df = resolve_paths(df, metadata_path)
    df["num_sources"] = pd.to_numeric(df["num_sources"], errors="raise").astype(int)
    df = df[(df["num_sources"] >= 1) & (df["num_sources"] <= args.max_sources)].copy()

    if "split" in df.columns:
        train_df = df[df["split"].astype(str).str.lower() == "train"]
        val_df = df[df["split"].astype(str).str.lower().isin(["val", "validation"])]
    else:
        # Reproducible fallback split. Prefer a source-aware split prepared before mixing.
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        n_val = max(1, int(0.15 * len(df)))
        val_df, train_df = df.iloc[:n_val], df.iloc[n_val:]
    if train_df.empty or val_df.empty:
        raise ValueError("Need non-empty train and validation sets. Provide a 'split' column or more rows.")

    train_loader = DataLoader(MixtureDataset(train_df, args.max_sources), batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(MixtureDataset(val_df, args.max_sources), batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    model = SourceCountCNN(max_sources=args.max_sources).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_val = float("inf")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}; train rows: {len(train_df)}; validation rows: {len(val_df)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)

        model.eval()
        val_loss, correct, total, abs_error = 0.0, 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss = criterion(logits, y)
                preds = logits.argmax(dim=1)
                val_loss += loss.item() * len(y)
                correct += (preds == y).sum().item()
                abs_error += (preds - y).abs().sum().item()
                total += len(y)
        train_loss /= len(train_df)
        val_loss /= total
        accuracy = correct / total
        mae = abs_error / total
        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} "
              f"| val_loss={val_loss:.4f} | val_accuracy={accuracy:.3f} | val_MAE={mae:.3f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "max_sources": args.max_sources,
                "sample_rate": SAMPLE_RATE,
                "clip_seconds": CLIP_SECONDS,
                "architecture": "SourceCountCNN",
                "epoch": epoch,
                "val_loss": val_loss,
            }, output)
            print(f"  Saved best checkpoint: {output}")

    print("Training finished. Evaluate the saved model on a held-out test set before using it.")

if __name__ == "__main__":
    main()
