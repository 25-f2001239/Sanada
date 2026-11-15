"""Tagger inference module for SANADA/EchoCount (supports v2 GroupFineTaggerCNN and legacy TaggerCNN).

 enforces hierarchical consistency:
- Coarse groups ("Human voice", "Musical instrument", "Animal", "Household/tools", "Environment/other")
  are predicted and returned first with validation-tuned thresholds.
- Fine classes are secondary details and are NEVER shown as active/above-threshold if their
  parent group is below its threshold (prevents contradictory fine tags such as Telephone/Scissors
  when only Human voice is active).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from dataset_config import get_friendly_name, get_group_for_class, load_taxonomy
from model import DEVICE, waveform_to_logmel

BASE_DIR = Path(__file__).resolve().parent
TAGGER_PATH = Path(os.getenv("TAGGER_PATH", str(BASE_DIR / "checkpoints" / "tagger.pt")))


class TaggerCNN(nn.Module):
    """Legacy 32-class fine-only tagger architecture."""

    def __init__(self, num_classes: int):
        super().__init__()
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
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class GroupFineTaggerCNN(nn.Module):
    """Dual-head Group + Fine-class tagger architecture."""

    def __init__(self, num_groups: int, num_fine: int):
        super().__init__()
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
        self.group_head = nn.Linear(256, int(num_groups))
        self.fine_head = nn.Linear(256, int(num_fine))

    def forward(self, x):
        emb = self.shared_fc(self.features(x))
        return self.group_head(emb), self.fine_head(emb)


@lru_cache(maxsize=2)
def _load_tagger_cached(path: str, mtime_ns: int, file_size: int):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or not ckpt.get("trained", False):
        raise ValueError("Checkpoint missing or trained=false.")

    arch = ckpt.get("architecture", "")
    if arch == "GroupFineTaggerCNN":
        groups = ckpt["groups"]
        vocab = ckpt["vocabulary"]
        model = GroupFineTaggerCNN(len(groups), len(vocab)).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, ckpt
    elif arch == "TaggerCNN":
        vocab = ckpt["vocabulary"]
        model = TaggerCNN(len(vocab)).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, ckpt
    else:
        raise ValueError(f"Unsupported tagger architecture: {arch}")


def _load_tagger(path: Path = TAGGER_PATH):
    if not path.is_file():
        raise FileNotFoundError(f"No tagger checkpoint at {path}.")
    stat = path.stat()
    return _load_tagger_cached(str(path), stat.st_mtime_ns, stat.st_size)


def tagger_status(path: Path = TAGGER_PATH) -> dict:
    path = Path(path)
    if not path.is_file():
        return {"available": False, "trained": False, "num_classes": 0, "groups": [], "vocabulary": [],
                "error": f"No checkpoint at {path.name}. Run python train_tagger.py."}
    try:
        _, ckpt = _load_tagger(path)
        return {
            "available": True,
            "trained": True,
            "hierarchical": ckpt.get("architecture") == "GroupFineTaggerCNN",
            "architecture": ckpt.get("architecture"),
            "version": ckpt.get("version", "v1"),
            "num_groups": len(ckpt.get("groups", [])),
            "groups": ckpt.get("groups", []),
            "num_classes": len(ckpt.get("vocabulary", [])),
            "vocabulary": ckpt.get("vocabulary", []),
            "val_group_map": ckpt.get("val_group_macro_map"),
            "val_group_macro_map": ckpt.get("val_group_macro_map"),
            "test_group_macro_map": ckpt.get("test_group_macro_map"),
            "val_macro_map": ckpt.get("val_macro_map"),
            "test_macro_map": ckpt.get("test_macro_map"),
            "val_macro_f1": ckpt.get("val_macro_f1"),
            "test_macro_f1": ckpt.get("test_macro_f1"),
            "epoch": ckpt.get("epoch"),
            "error": None,
        }
    except Exception as exc:
        return {"available": False, "trained": False, "num_classes": 0, "groups": [], "vocabulary": [], "error": str(exc)}


def tag_audio_hierarchical(
    waveform: np.ndarray,
    path: Path = TAGGER_PATH,
    precomputed_logmel: Optional[torch.Tensor] = None,
) -> dict:
    """Run hierarchical group + fine-class inference on an 8s clip.

    Never shows a fine class that contradicts the group predictions:
    - Fine classes belonging to a group whose probability is below its threshold are suppressed
      (omitted from compatible fine classes so they never contradict group predictions).
    """
    try:
        model, ckpt = _load_tagger(path)
    except Exception as exc:
        return {"available": False, "reason": str(exc), "groups": [], "tags": []}

    if precomputed_logmel is not None:
        feat = precomputed_logmel.unsqueeze(0).to(DEVICE) if precomputed_logmel.dim() == 3 else precomputed_logmel.to(DEVICE)
    else:
        feat = waveform_to_logmel(waveform).unsqueeze(0).to(DEVICE)

    arch = ckpt.get("architecture")
    use_amp = DEVICE.type == "cuda"

    if arch == "GroupFineTaggerCNN":
        groups_list: List[str] = ckpt["groups"]
        g_thresh: List[float] = ckpt["group_thresholds"]
        vocab: List[str] = ckpt["vocabulary"]
        f_thresh: List[float] = ckpt["thresholds"]
        fine_to_group: Dict[str, str] = ckpt["fine_to_group"]

        with torch.inference_mode(), torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
            g_logits, f_logits = model(feat)
            g_probs = torch.sigmoid(g_logits[0].float()).cpu().numpy()
            f_probs = torch.sigmoid(f_logits[0].float()).cpu().numpy()

        group_results = []
        active_groups = set()
        for grp, gp, gt in zip(groups_list, g_probs, g_thresh):
            above = bool(gp >= gt)
            if above:
                active_groups.add(grp)
            group_results.append({
                "group": grp,
                "probability": float(gp),
                "threshold": float(gt),
                "above_threshold": above,
            })
        group_results.sort(key=lambda x: -x["probability"])

        # Only keep fine classes whose parent group is active (or if no group is above threshold, top group)
        allowed_groups = active_groups if active_groups else ({group_results[0]["group"]} if group_results else set())
        fine_results = []
        tax = load_taxonomy()
        for cls, fp, ft in zip(vocab, f_probs, f_thresh):
            parent_grp = get_group_for_class(cls, tax)
            if parent_grp not in allowed_groups:
                continue  # Never return a fine class that contradicts the active groups
            # If the mixture is speech-only ("Human voice" active, but speech isn't applause/cough/laughter unless above threshold),
            # keep above_threshold strict
            above = bool((fp >= ft) and (parent_grp in active_groups))
            fine_results.append({
                "class": cls,
                "friendly_name": get_friendly_name(cls, tax),
                "group": parent_grp,
                "probability": float(fp),
                "threshold": float(ft),
                "above_threshold": above,
            })
        fine_results.sort(key=lambda x: -x["probability"])

        is_speech_dominant = bool(
            group_results and group_results[0]["group"] == "Human voice" and group_results[0]["above_threshold"]
        )
        return {
            "available": True,
            "hierarchical": True,
            "groups": group_results,
            "active_groups": [g["group"] for g in group_results if g["above_threshold"]],
            "tags": fine_results,
            "speech_detected": is_speech_dominant,
            "caveat": "Tags describe the whole mix (coarse groups first; fine classes filtered by active group). Per-stem tagging requires a trained separator.",
        }

    # Fallback for legacy v1 TaggerCNN
    tax = load_taxonomy()
    fine_to_group = {fc: g for g, fcs in tax["group_to_fine_classes"].items() for fc in fcs}
    vocab = ckpt["vocabulary"]
    f_thresh = ckpt["thresholds"]
    with torch.inference_mode():
        logits = model(feat)[0].cpu()
    f_probs = torch.sigmoid(logits).numpy()

    fine_results = []
    grp_max: Dict[str, float] = {g: 0.0 for g in tax["groups"]}
    for cls, fp, ft in zip(vocab, f_probs, f_thresh):
        g = fine_to_group.get(cls, "Environment/other")
        grp_max[g] = max(grp_max.get(g, 0.0), float(fp))
        fine_results.append({
            "class": cls,
            "group": g,
            "probability": float(fp),
            "threshold": float(ft),
            "above_threshold": bool(fp >= ft),
        })
    fine_results.sort(key=lambda x: -x["probability"])
    group_results = [
        {"group": g, "probability": p, "threshold": 0.5, "above_threshold": p >= 0.5}
        for g, p in sorted(grp_max.items(), key=lambda x: -x[1])
    ]
    return {
        "available": True,
        "groups": group_results,
        "tags": fine_results,
        "speech_detected": False,
        "caveat": "Tags describe the whole mix. Per-stem tagging requires a trained separator.",
    }


def tag_audio(waveform: np.ndarray, path: Path = TAGGER_PATH, precomputed_logmel: Optional[torch.Tensor] = None) -> List[dict]:
    """Backwards-compatible list return of fine tags."""
    res = tag_audio_hierarchical(waveform, path=path, precomputed_logmel=precomputed_logmel)
    return res.get("tags", [])
