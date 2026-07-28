"""Run the trained keypoint model over labeled mood images and compute
geometric features for each, producing geometric_features.csv — the input
table for ensemble training (Section 6)."""
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import load_config, resolve_path  # noqa: E402
from src.features.geometric import compute_geometric_features, FEATURE_NAMES  # noqa: E402
from src.keypoints.infer import KeypointPredictor  # noqa: E402


def main():
    cfg = load_config()
    kcfg = cfg["keypoints"]

    mood_labels_path = resolve_path(cfg["paths"]["mood_labels_csv"])
    mood_df = pd.read_csv(mood_labels_path)

    predictor = KeypointPredictor(
        model_path=resolve_path(cfg["paths"]["keypoint_model"]),
        num_keypoints=kcfg["num_keypoints"],
        input_size=kcfg["input_size"],
        heatmap_size=kcfg["heatmap_size"],
    )

    rows = []
    for _, row in tqdm(mood_df.iterrows(), total=len(mood_df)):
        img_path = row["image_path"]
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        kp, conf = predictor.predict(img)
        feats = compute_geometric_features(kp)
        rec = {"image_path": img_path, "label": row["label"], "provenance": row["provenance"]}
        rec.update(dict(zip(FEATURE_NAMES, feats)))
        rec["keypoint_mean_confidence"] = float(conf.mean())
        rows.append(rec)

    out_df = pd.DataFrame(rows)
    out_path = resolve_path(cfg["paths"]["features_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")
    print(f"mean keypoint confidence: {out_df['keypoint_mean_confidence'].mean():.3f}")


if __name__ == "__main__":
    main()
