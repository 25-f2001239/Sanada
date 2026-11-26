"""Unified EchoCount training CLI (`counter`, `tagger`, `separator`, `all`) with `--smoke` and `--full`."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio import load_audio, resample, waveform_to_logmel
from config import (
    CHECKPOINTS_DIR,
    CLIP_SECONDS,
    DATA_DIR,
    DEVICE,
    FEATURE_VERSION,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    get_group_for_class,
    load_taxonomy,
)
from counting import SourceCountCNN
from data import scan_all_mixtures
from separation import MAX_SEPARABLE, SEPARATOR_ARCHITECTURE, IterativeMaskSeparator
from tagging import GroupFineTaggerCNN


def _extract_features(path: Path) -> torch.Tensor:
    wav, _ = load_audio(path)
    return waveform_to_logmel(wav)


class LogMelCountDataset(Dataset):
    """Cached log-mel dataset for counter or tagger training."""

    def __init__(
        self,
        records: list[dict],
        num_classes_or_min: int,
        mode: str = "counter",
        vocab: list[str] | None = None,
        groups: list[str] | None = None,
    ):
        self.records = records
        self.mode = mode
        self.min_k = num_classes_or_min
        self.vocab = vocab or []
        self.groups = groups or []
        self.v_idx = {c: i for i, c in enumerate(self.vocab)}
        self.g_idx = {g: i for i, g in enumerate(self.groups)}
        self.tax = load_taxonomy()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        x = _extract_features(Path(r["path"]))
        if self.mode == "counter":
            return x, int(r["num_sources"]) - self.min_k
        g_vec = torch.zeros(len(self.groups), dtype=torch.float32)
        f_vec = torch.zeros(len(self.vocab), dtype=torch.float32)
        if r["family"] == "speech":
            if "Human voice" in self.g_idx:
                g_vec[self.g_idx["Human voice"]] = 1.0
        else:
            for lbl in r.get("labels", []):
                grp = get_group_for_class(lbl, self.tax)
                if grp in self.g_idx:
                    g_vec[self.g_idx[grp]] = 1.0
                if lbl in self.v_idx:
                    f_vec[self.v_idx[lbl]] = 1.0
        return x, g_vec, f_vec


def train_counter(
    smoke: bool = False, epochs: int = 12, lr: float = 3e-4, batch_size: int = 32
) -> None:
    """Train SourceCountCNN on train/val mixtures (test split strictly excluded)."""
    recs = [
        r for r in scan_all_mixtures(include_duplicates=False) if r["split"] in ("train", "val")
    ]
    tr = [r for r in recs if r["split"] == "train"]
    va = [r for r in recs if r["split"] == "val"]
    if smoke:
        tr, va, epochs = tr[:64], va[:32], 2
    min_k, max_k = 2, 31
    tr_dl = DataLoader(
        LogMelCountDataset(tr, min_k, "counter"), batch_size=batch_size, shuffle=True
    )
    va_dl = DataLoader(LogMelCountDataset(va, min_k, "counter"), batch_size=batch_size)
    model = SourceCountCNN(min_k, max_k).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_mae = float("inf")
    out = ROOT / "work" / "smoke_counter.pt" if smoke else CHECKPOINTS_DIR / "source_count_clip.pt"
    if not smoke and out.is_file():
        shutil.copy2(out, CHECKPOINTS_DIR / "source_count_clip_backup.pt")

    for ep in range(1, epochs + 1):
        model.train()
        for x, y in tr_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
        model.eval()
        errs = []
        with torch.inference_mode():
            vals = torch.arange(min_k, max_k + 1, dtype=torch.float32, device=DEVICE)
            for x, y in va_dl:
                probs = F.softmax(model(x.to(DEVICE)), dim=1)
                pred = (probs * vals).sum(dim=1).cpu().numpy()
                errs.extend(np.abs(pred - (y.numpy() + min_k)).tolist())
        mae = float(np.mean(errs))
        print(f"[Counter] Epoch {ep}/{epochs} | Val MAE={mae:.3f}")
        if mae < best_mae:
            best_mae = mae
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "architecture": "SourceCountCNN",
                    "model_state_dict": model.state_dict(),
                    "min_sources": min_k,
                    "max_sources": max_k,
                    "sample_rate": SAMPLE_RATE,
                    "clip_seconds": CLIP_SECONDS,
                    "n_mels": N_MELS,
                    "n_fft": N_FFT,
                    "hop_length": HOP_LENGTH,
                    "feature_version": FEATURE_VERSION,
                    "trained": not smoke,
                    "val_mae": round(mae, 4),
                    "epoch": ep,
                },
                out,
            )


def train_tagger(
    smoke: bool = False, epochs: int = 12, lr: float = 3e-4, batch_size: int = 32
) -> None:
    """Train dual-head GroupFineTaggerCNN on train/val splits."""
    tax = load_taxonomy()
    groups = list(tax["groups"])
    vocab = sorted({c for fcs in tax["group_to_fine_classes"].values() for c in fcs})
    recs = [
        r for r in scan_all_mixtures(include_duplicates=False) if r["split"] in ("train", "val")
    ]
    tr = [r for r in recs if r["split"] == "train"]
    va = [r for r in recs if r["split"] == "val"]
    if smoke:
        tr, va, epochs = tr[:64], va[:32], 2
    tr_dl = DataLoader(
        LogMelCountDataset(tr, 2, "tagger", vocab, groups), batch_size=batch_size, shuffle=True
    )
    va_dl = DataLoader(LogMelCountDataset(va, 2, "tagger", vocab, groups), batch_size=batch_size)
    model = GroupFineTaggerCNN(len(groups), len(vocab)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    out = ROOT / "work" / "smoke_tagger.pt" if smoke else CHECKPOINTS_DIR / "tagger.pt"
    if not smoke and out.is_file():
        shutil.copy2(out, CHECKPOINTS_DIR / "tagger_backup.pt")

    best_loss = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        for x, gv, fv in tr_dl:
            x, gv, fv = x.to(DEVICE), gv.to(DEVICE), fv.to(DEVICE)
            opt.zero_grad()
            gl, fl = model(x)
            loss = F.binary_cross_entropy_with_logits(gl, gv) + F.binary_cross_entropy_with_logits(
                fl, fv
            )
            loss.backward()
            opt.step()
        model.eval()
        vlosses = []
        with torch.inference_mode():
            for x, gv, fv in va_dl:
                gl, fl = model(x.to(DEVICE))
                vlosses.append(
                    float(
                        (
                            F.binary_cross_entropy_with_logits(gl, gv.to(DEVICE))
                            + F.binary_cross_entropy_with_logits(fl, fv.to(DEVICE))
                        ).item()
                    )
                )
        vloss = float(np.mean(vlosses))
        print(f"[Tagger] Epoch {ep}/{epochs} | Val BCE={vloss:.4f}")
        if vloss < best_loss:
            best_loss = vloss
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "architecture": "GroupFineTaggerCNN",
                    "model_state_dict": model.state_dict(),
                    "groups": groups,
                    "vocabulary": vocab,
                    "group_thresholds": [0.45] * len(groups),
                    "thresholds": [0.35] * len(vocab),
                    "fine_to_group": {c: get_group_for_class(c, tax) for c in vocab},
                    "trained": not smoke,
                    "epoch": ep,
                },
                out,
            )


def train_separator(smoke: bool = False, epochs: int = 8, lr: float = 1e-3) -> None:
    """Train IterativeMaskSeparator with PIT SI-SDR + STFT L1 loss on isolated clips."""
    window = torch.hann_window(N_FFT, device=DEVICE)
    iso_manifest = DATA_DIR / "isolated_sources" / "manifest.json"
    if smoke or not iso_manifest.is_file():
        rng = random.Random(42)
        mix_samples = int(SAMPLE_RATE * 2.0)
        ds = []
        for idx in range(20 if smoke else 50):
            k = rng.randint(2, min(4, MAX_SEPARABLE))
            t = np.linspace(0, 2.0, mix_samples, endpoint=False)
            srcs = [
                (np.sin(2 * np.pi * (180 + i * 150) * t) * np.sin(np.pi * t / 2.0)).astype(
                    np.float32
                )
                for i in range(k)
            ]
            mix = np.sum(srcs, axis=0)
            ds.append((torch.from_numpy(mix), [torch.from_numpy(s) for s in srcs], k))
        epochs = 2
        out = ROOT / "work" / "smoke_separator.pt"
    else:
        mdata = json.loads(iso_manifest.read_text(encoding="utf-8"))
        by_cls: dict[str, list[Path]] = defaultdict(list)
        for item in mdata.get("clips", []):
            if item.get("split") == "train":
                p = DATA_DIR / "isolated_sources" / item["rel_path"]
                if p.is_file():
                    by_cls[item["class"]].append(p)
        rng, classes, ds = random.Random(42), sorted(by_cls.keys()), []
        clip_len = SAMPLE_RATE * 4
        for _ in range(160):
            k = rng.randint(2, 4)
            chosen = [rng.choice(by_cls[c]) for c in rng.sample(classes, k)]
            srcs = []
            for p in chosen:
                a, sr = sf.read(str(p), dtype="float32")
                a = np.mean(a, axis=1) if a.ndim > 1 else a
                if sr != SAMPLE_RATE:
                    a = resample(a, sr, SAMPLE_RATE)
                a = a[:clip_len] if len(a) >= clip_len else np.pad(a, (0, clip_len - len(a)))
                srcs.append((a * (10.0 ** (-rng.uniform(0, 5) / 20.0))).astype(np.float32))
            mix = np.sum(srcs, axis=0)
            scale = 0.89 / max(float(np.max(np.abs(mix))), 1e-6)
            ds.append(
                (torch.from_numpy(mix * scale), [torch.from_numpy(s * scale) for s in srcs], k)
            )
        out = CHECKPOINTS_DIR / "separator_iterative.pt"
        if out.is_file():
            shutil.copy2(out, CHECKPOINTS_DIR / "separator_iterative_backup.pt")

    model = IterativeMaskSeparator(32, 6).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for mix, srcs, k in ds:
            mix = mix.to(DEVICE)
            ts_t = torch.stack([s.to(DEVICE) for s in srcs])
            spec = torch.stft(mix, N_FFT, HOP_LENGTH, window=window, return_complex=True)
            res, mix_lm = spec.clone(), torch.log1p(spec.abs()).unsqueeze(0)
            preds_t, preds_s = [], []
            for i in range(k):
                mask = model(mix_lm, torch.log1p(res.abs()).unsqueeze(0), i, k)[0]
                src = res * mask
                res = res - src
                preds_t.append(
                    torch.istft(src, N_FFT, HOP_LENGTH, window=window, length=mix.numel())
                )
                preds_s.append(src)
            ps = torch.stack(preds_s)
            ts_s = torch.stack(
                [torch.stft(s, N_FFT, HOP_LENGTH, window=window, return_complex=True) for s in ts_t]
            )
            with torch.no_grad():
                costs = np.array(
                    [
                        [F.l1_loss(ps[i].abs(), ts_s[j].abs()).item() for j in range(k)]
                        for i in range(k)
                    ]
                )
                ri, ci = linear_sum_assignment(costs)
            loss = (
                sum(
                    F.l1_loss(torch.log1p(ps[i].abs()), torch.log1p(ts_s[j].abs()))
                    for i, j in zip(ri, ci)
                )
                / k
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
        print(f"[Separator] Epoch {ep}/{epochs} | Loss={tot / max(1, len(ds)):.4f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": SEPARATOR_ARCHITECTURE,
            "sample_rate": SAMPLE_RATE,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "config": {"channels": 32, "blocks": 6},
            "model_state_dict": model.state_dict(),
            "version": "SMOKE-TEST" if smoke else "IterativeMaskSeparator-v2-curriculum",
            "trained": not smoke,
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        out,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="target", required=True)
    for name in ("counter", "tagger", "separator", "all"):
        sp = sub.add_parser(name)
        sp.add_argument("--smoke", action="store_true")
        sp.add_argument("--full", action="store_true")
        sp.add_argument("--epochs", type=int, default=8)
    args = p.parse_args()
    smoke = bool(args.smoke and not args.full)
    if args.target in ("counter", "all"):
        train_counter(smoke=smoke, epochs=args.epochs)
    if args.target in ("tagger", "all"):
        train_tagger(smoke=smoke, epochs=args.epochs)
    if args.target in ("separator", "all"):
        train_separator(smoke=smoke, epochs=args.epochs)


if __name__ == "__main__":
    main()
