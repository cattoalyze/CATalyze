"""Percentile stats for the 7 geometric features (Section 4), computed once
over the real training pool (data/processed/geometric_features.csv). The
frontend's "Ear Posture" panel uses these to describe a single analyzed
image's ear angle/spread relative to where it actually falls in the real
training distribution -- a real, data-derived comparison, not a fabricated
threshold or a claim about pupils/dilation the model never measures.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import load_config, resolve_path  # noqa: E402
from src.features.geometric import FEATURE_NAMES  # noqa: E402


def main():
    cfg = load_config()
    features_path = resolve_path(cfg["paths"]["features_csv"])
    df = pd.read_csv(features_path)

    stats = {"n": int(len(df)), "features": {}}
    for name in FEATURE_NAMES:
        col = df[name]
        stats["features"][name] = {
            "p10": float(col.quantile(0.10)),
            "p25": float(col.quantile(0.25)),
            "p50": float(col.quantile(0.50)),
            "p75": float(col.quantile(0.75)),
            "p90": float(col.quantile(0.90)),
        }

    reports_dir = resolve_path(cfg["paths"]["reports"])
    out_path = reports_dir / "feature_stats.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved {out_path} (n={stats['n']})")


if __name__ == "__main__":
    main()
