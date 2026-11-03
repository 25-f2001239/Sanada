from pathlib import Path
import pandas as pd

dataset = Path(r"C:\Users\Akshith\Downloads\sanada")
output = Path("data")
output.mkdir(exist_ok=True)

frames = {"train": [], "val": [], "test": []}

for metadata in dataset.rglob("metadata.csv"):
    split = metadata.parent.name.lower()
    if split not in frames:
        continue

    df = pd.read_csv(metadata)

    if "num_sources" in df.columns:
        counts = df["num_sources"]
    elif "num_speakers" in df.columns:
        counts = df["num_speakers"]
    else:
        raise ValueError(f"No source-count column in {metadata}")

    # WAVs are beside each split's metadata; CSV paths may still be Colab paths.
    df["mixture_path"] = [
        str(metadata.parent / Path(path).name)
        for path in df["mixture_path"]
    ]
    df["num_sources"] = pd.to_numeric(counts).astype(int)
    df["split"] = split

    missing = [
        path for path in df["mixture_path"]
        if not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing WAV referenced by {metadata}: {missing[0]}")

    frames[split].append(df[["mixture_path", "num_sources", "split"]])
    print(f"{metadata}: {len(df)} rows")

for split in frames:
    if not frames[split]:
        raise ValueError(f"No metadata.csv files found for split: {split}")

    combined = pd.concat(frames[split], ignore_index=True)
    combined.to_csv(output / f"{split}.csv", index=False)
    print(f"\n{split}: {len(combined)} total rows")
    print("Counts:", combined["num_sources"].value_counts().sort_index().to_dict())

pd.concat([pd.concat(frames["train"]), pd.concat(frames["val"])],
          ignore_index=True).to_csv(output / "train_val.csv", index=False)

print("\nCreated data/train_val.csv and data/test.csv")