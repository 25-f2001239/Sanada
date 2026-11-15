"""Combine SANADA split metadata files into the trainer's CSV format."""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Root folder containing split metadata.csv files")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not dataset.is_dir():
        parser.error(f"Dataset folder does not exist: {dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = {"train": [], "val": [], "test": []}
    for metadata in sorted(dataset.rglob("metadata.csv")):
        split = metadata.parent.name.lower()
        if split == "validation":
            split = "val"
        if split not in frames:
            continue

        frame = pd.read_csv(metadata)
        if "mixture_path" not in frame.columns:
            raise ValueError(f"Missing mixture_path column in {metadata}")
        if "num_sources" in frame.columns:
            counts = frame["num_sources"]
        elif "num_speakers" in frame.columns:
            counts = frame["num_speakers"]
        else:
            raise ValueError(f"No source-count column in {metadata}")

        resolved_paths = []
        for raw_path in frame["mixture_path"].astype(str):
            raw = Path(raw_path).expanduser()
            candidate = raw if raw.is_absolute() else metadata.parent / raw
            if not candidate.is_file():
                candidate = metadata.parent / raw.name
            if not candidate.is_file():
                raise FileNotFoundError(f"Audio file from {metadata} was not found: {raw_path}")
            resolved_paths.append(str(candidate.resolve()))

        frame["mixture_path"] = resolved_paths
        frame["num_sources"] = pd.to_numeric(counts, errors="raise").astype(int)
        frame["split"] = split
        frames[split].append(frame[["mixture_path", "num_sources", "split"]])
        print(f"{metadata}: {len(frame)} rows")

    for split, parts in frames.items():
        if not parts:
            raise ValueError(f"No metadata.csv files found for split: {split}")
        combined = pd.concat(parts, ignore_index=True)
        combined.to_csv(output_dir / f"{split}.csv", index=False)
        print(f"{split}: {len(combined)} rows; counts={combined['num_sources'].value_counts().sort_index().to_dict()}")

    train_val = pd.concat(frames["train"] + frames["val"], ignore_index=True)
    train_val.to_csv(output_dir / "train_val.csv", index=False)
    all_rows = pd.concat(
        [frame for parts in frames.values() for frame in parts],
        ignore_index=True,
    )
    all_rows.to_csv(output_dir / "metadata.csv", index=False)
    print(
        f"Created {output_dir / 'train_val.csv'}, {output_dir / 'test.csv'}, "
        f"and {output_dir / 'metadata.csv'}."
    )


if __name__ == "__main__":
    main()
