"""Supplement the scarce ANXIOUS class (Section 3, item 3 of the
continuation prompt) with real external data, now that a Kaggle API token
was supplied directly by the user (the prior session's Roboflow candidate,
and this session's own search, found no dataset downloadable without
creating an account on the user's behalf).

Source: kagglehub dataset `nguyenvunhuhuynh/cat-emotion-2` — itself a
mirror of a Roboflow export (roboflow.com/nidowarddarren/
cat-emotion-classification-j1v9r). License: Public Domain (stated in the
dataset's own README.dataset.txt). 1040 images across 4 classes
(Aggressive, Alert, Distressed, Relaxed), each source image augmented into
3 Roboflow-exported copies.

Only the "Distressed" class is used here, mapped to this project's
ANXIOUS label. This is a semantic judgment call, not an identical label —
documented explicitly, not asserted as equivalent. Labels are
community/uploader-assigned on Roboflow, not independently verified —
the same epistemic status this project's own provenance='ai' seed labels
already have, tracked the same way via a distinct provenance tag rather
than silently merged into 'ai' or 'pseudo'.

Two integrity steps before use:
1. De-duplication: only one of each 3 Roboflow-augmented copies per source
   image is kept (grouped by filename prefix before "_jpg.rf."), so our
   own train/test splits downstream don't get near-duplicate leakage
   between train and test.
2. Keypoint-confidence filtering: this project's own trained keypoint
   model is run on each candidate (the same way any new image would be
   scored at serving time) and any image whose mean keypoint confidence
   falls below ANXIOUS_KEYPOINT_CONF_FLOOR is dropped — these are mostly
   non-frontal/artistic/heavily-occluded shots the keypoint model (trained
   on CAT_DATASET's front-facing annotated photos) can't localize
   reliably, and using an unreliable keypoint set would corrupt the
   geometric features more than it would help.
"""
import sys
from pathlib import Path

import cv2
import kagglehub
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, resolve_path  # noqa: E402
from src.features.geometric import FEATURE_NAMES, compute_geometric_features  # noqa: E402
from src.keypoints.infer import KeypointPredictor  # noqa: E402

KAGGLE_DATASET = "nguyenvunhuhuynh/cat-emotion-2"
SOURCE_CLASS = "Distressed"
TARGET_LABEL = "ANXIOUS"
PROVENANCE_TAG = "external_kaggle_distressed"
ANXIOUS_KEYPOINT_CONF_FLOOR = 0.30


def dedupe_source_images(dataset_root: Path) -> list[Path]:
    seen_prefixes = set()
    kept = []
    for split in ["train", "valid", "test"]:
        class_dir = dataset_root / split / SOURCE_CLASS
        if not class_dir.exists():
            continue
        for img_path in sorted(class_dir.glob("*.jpg")):
            prefix = img_path.name.split("_jpg.rf.")[0]
            if prefix in seen_prefixes:
                continue
            seen_prefixes.add(prefix)
            kept.append(img_path)
    return kept


def main():
    cfg = load_config()
    kcfg = cfg["keypoints"]

    print(f"downloading {KAGGLE_DATASET} via kagglehub...")
    dataset_root = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print(f"  -> {dataset_root}")

    dedup_images = dedupe_source_images(dataset_root)
    print(f"{SOURCE_CLASS} class: {len(dedup_images)} unique source images after de-duplicating Roboflow's 3x augmentation")

    local_dir = resolve_path("data/raw/external/kaggle_distressed")
    local_dir.mkdir(parents=True, exist_ok=True)
    local_paths = []
    for src in dedup_images:
        dst = local_dir / src.name
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
        local_paths.append(dst)

    predictor = KeypointPredictor(
        model_path=resolve_path(cfg["paths"]["keypoint_model"]),
        num_keypoints=kcfg["num_keypoints"], input_size=kcfg["input_size"],
        heatmap_size=kcfg["heatmap_size"], device="cuda" if torch.cuda.is_available() else "cpu",
    )

    rows = []
    n_dropped_low_conf = 0
    n_dropped_unreadable = 0
    for p in local_paths:
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            n_dropped_unreadable += 1
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        kp, kp_conf = predictor.predict(img_rgb)
        mean_conf = float(kp_conf.mean())
        if mean_conf < ANXIOUS_KEYPOINT_CONF_FLOOR:
            n_dropped_low_conf += 1
            continue
        geo = compute_geometric_features(kp)
        row = {"image_path": str(p.resolve()), "label": TARGET_LABEL, "provenance": PROVENANCE_TAG,
               "keypoint_mean_confidence": mean_conf}
        row.update({name: float(v) for name, v in zip(FEATURE_NAMES, geo)})
        rows.append(row)

    print(f"kept {len(rows)} / {len(local_paths)} "
          f"(dropped {n_dropped_low_conf} below keypoint-confidence floor {ANXIOUS_KEYPOINT_CONF_FLOOR}, "
          f"{n_dropped_unreadable} unreadable)")

    new_df = pd.DataFrame(rows)

    features_path = resolve_path(cfg["paths"]["features_csv"])
    features_df = pd.read_csv(features_path)
    before_anxious = (features_df["label"] == TARGET_LABEL).sum()
    combined = pd.concat([features_df, new_df[["image_path", "label", "provenance"] + FEATURE_NAMES + ["keypoint_mean_confidence"]]], ignore_index=True)
    combined.to_csv(features_path, index=False)
    after_anxious = (combined["label"] == TARGET_LABEL).sum()
    print(f"geometric_features.csv: ANXIOUS {before_anxious} -> {after_anxious} (+{after_anxious - before_anxious})")

    labels_path = resolve_path(cfg["paths"]["mood_labels_csv"])
    labels_df = pd.read_csv(labels_path)
    new_labels = new_df[["image_path", "label", "provenance"]]
    combined_labels = pd.concat([labels_df, new_labels], ignore_index=True)
    combined_labels.to_csv(labels_path, index=False)
    print(f"mood_labels.csv: {len(labels_df)} -> {len(combined_labels)} rows")


if __name__ == "__main__":
    main()
