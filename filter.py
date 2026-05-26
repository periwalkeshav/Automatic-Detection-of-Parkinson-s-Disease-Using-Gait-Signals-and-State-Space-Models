import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
SPLITS = ROOT / "splits"

if __name__ == "__main__":
    df = pd.read_excel("./Data/demographics.xls", engine="xlrd")
    subject_ids = []
    for fold_dir in sorted(SPLITS.glob("Fold_2")):
        fold = fold_dir.name
        for split in ("training", "validation", "test"):
            for path in sorted((fold_dir / split).glob("*.txt")):
                subject_id = path.stem.rsplit("_", 1)[0]
                subject_ids.append(subject_id)

    df = df[~df["ID"].isin(subject_ids)]
    df.to_csv("filtered_demographics_2.csv", index=False)
    