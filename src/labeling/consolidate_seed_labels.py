"""Consolidate the grid_NNN_labels.json files (produced during the batch
visual seed-labeling pass) plus manifest.json into data/processed/mood_labels.csv.

Provenance is tagged 'ai': no human rater was available in this session, so
seed labels were assigned by Claude's visual judgment reviewing image grids.
This is documented as a real limitation in the README — these labels are a
weaker ground truth than true human annotation and should be read that way
in any "honest evaluation" reporting.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import load_config, resolve_path  # noqa: E402


def main():
    cfg = load_config()
    grids_dir = resolve_path(cfg["paths"]["data_processed"]) / "label_grids"
    manifest = json.loads((grids_dir / "manifest.json").read_text())

    rows = []
    for grid_name, image_paths in manifest.items():
        labels_path = grids_dir / grid_name.replace(".png", "_labels.json")
        if not labels_path.exists():
            print(f"WARNING: no labels found for {grid_name}, skipping {len(image_paths)} images")
            continue
        labels = json.loads(labels_path.read_text())
        for idx_str, img_path in enumerate(image_paths):
            label = labels.get(str(idx_str))
            if label is None:
                print(f"WARNING: missing label for index {idx_str} in {grid_name}")
                continue
            rows.append({"image_path": img_path, "label": label, "provenance": "ai"})

    df = pd.DataFrame(rows)
    valid_classes = set(cfg["mood_classes"])
    bad = df[~df["label"].isin(valid_classes)]
    if len(bad):
        raise ValueError(f"invalid labels found: {bad['label'].unique().tolist()}")

    out_path = resolve_path(cfg["paths"]["mood_labels_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} seed labels to {out_path}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
