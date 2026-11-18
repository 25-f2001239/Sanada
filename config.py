"""Central paths, environment settings, and taxonomy lookup for EchoCount."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
TAXONOMY_PATH = BASE_DIR / "taxonomy.json"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
EVAL_DIR = BASE_DIR / "evaluation"

SAMPLE_RATE = 16_000
CLIP_SECONDS = 8.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
N_MELS = 64
N_FFT = 1_024
HOP_LENGTH = 256
FEATURE_VERSION = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def env_int(name: str, default: int, low: int = 1) -> int:
    """Read an integer environment variable with a lower bound."""
    try:
        return max(low, int(os.getenv(name, default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    """Read a float environment variable with fallback."""
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


MAX_SOURCES = env_int("MAX_SOURCES", 32)
MODEL_PATH = (
    Path(os.getenv("MODEL_PATH", str(CHECKPOINTS_DIR / "source_count_clip.pt")))
    .expanduser()
    .resolve()
)
TAGGER_PATH = (
    Path(os.getenv("TAGGER_PATH", str(CHECKPOINTS_DIR / "tagger.pt"))).expanduser().resolve()
)
SEPARATOR_PATH = (
    Path(os.getenv("SEPARATOR_PATH", str(CHECKPOINTS_DIR / "separator_iterative.pt")))
    .expanduser()
    .resolve()
)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load config.json or return default configuration."""
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "dataset_path": r"C:\coding\SANADA\sanada dataset",
        "local_data_path": str(DATA_DIR),
        "models_dir": str(MODELS_DIR),
        "checkpoints_dir": str(CHECKPOINTS_DIR),
        "exclude_duplicate_folders": ["Org_Fsd_Mix/Org_Mix_4_Sounds_Variable"],
        "held_out_speakers": ["Speaker_4_1", "Speaker8", "Speaker9"],
        "seed": 42,
        "sample_rate": SAMPLE_RATE,
        "clip_seconds": CLIP_SECONDS,
        "max_separable": 8,
        "quality_corr_threshold": 0.9,
        "speaker_cosine_similarity_max": 0.82,
    }


def load_taxonomy(path: Path = TAXONOMY_PATH) -> dict[str, Any]:
    """Load taxonomy.json and ensure class_to_group is populated."""
    data = json.loads(path.read_text(encoding="utf-8"))
    c2g = dict(data.get("class_to_group", {}))
    for grp, fcs in data.get("group_to_fine_classes", {}).items():
        for fc in fcs:
            c2g[fc] = grp
    data["class_to_group"] = c2g
    return data


def get_class_to_group(tax: dict[str, Any] | None = None) -> dict[str, str]:
    """Return canonical class-to-group mapping."""
    return dict((tax or load_taxonomy()).get("class_to_group", {}))


def get_group_for_class(label: str, tax: dict[str, Any] | None = None) -> str:
    """Resolve any class or friendly label to its canonical coarse group."""
    c2g = get_class_to_group(tax)
    for key in (label, label.replace(" ", "_"), label.replace("_", " ")):
        if key in c2g:
            return c2g[key]
    return "Environment/other"


def get_friendly_name(label: str, tax: dict[str, Any] | None = None) -> str:
    """Map an internal class label to its user-facing friendly name."""
    fmap = (tax or load_taxonomy()).get("friendly_names", {})
    for key in (label, label.replace(" ", "_")):
        if key in fmap:
            return fmap[key]
    return label.replace("_", " ")


def get_dataset_root(cfg: dict[str, Any] | None = None) -> Path:
    """Return the configured dataset root directory if present, else local data dir."""
    cfg = cfg or load_config()
    ext = Path(cfg["dataset_path"])
    return ext if ext.is_dir() else Path(cfg.get("local_data_path", str(DATA_DIR)))
