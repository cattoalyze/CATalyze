"""Subgroup/bias analysis (Section 3, item 1): slice ensemble accuracy by
lighting, coat-color, and image-source factors.

Neither coat color nor lighting is an annotated field anywhere in this
project's data — they were never labeled. Rather than skip the analysis for
that reason, this computes two honest pixel-derived proxies directly from
each test image (grayscale brightness for lighting, HSV mean saturation as a
coat-vividness proxy) and buckets by tercile of the test set's own
distribution. These are heuristics, not verified annotations — e.g.
saturation is affected by lighting/camera/background too, not coat color
alone — and are reported as such, not passed off as ground truth.

Reuses the already-trained, already-deployed ensemble
(artifacts/ensemble_model.joblib) and replicates ensemble/train.py's exact
train/test split (same seed, same stratify) so this evaluates the same test
set as reports/ensemble_metrics.json, without retraining.
"""
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import load_config, resolve_path  # noqa: E402
from src.ensemble.train import build_feature_matrix, compute_cnn_embeddings  # noqa: E402
from src.reports.experiment_log import log_run  # noqa: E402


def brightness_and_saturation(image_path: str) -> tuple[float, float]:
    img = cv2.imread(image_path)
    if img is None:
        return float("nan"), float("nan")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness = float(gray.mean()) / 255.0
    saturation = float(hsv[:, :, 1].mean()) / 255.0
    return brightness, saturation


def tercile_bucket(values: pd.Series) -> pd.Series:
    q1, q2 = values.quantile([1 / 3, 2 / 3])
    return pd.cut(values, bins=[-np.inf, q1, q2, np.inf], labels=["low", "mid", "high"])


def bucket_accuracy(df: pd.DataFrame, correct_col: str, bucket_col: str) -> dict:
    out = {}
    for bucket, group in df.groupby(bucket_col, observed=True):
        out[str(bucket)] = {
            "n": int(len(group)),
            "accuracy": float(group[correct_col].mean()) if len(group) else None,
        }
    return out


def main():
    cfg = load_config()
    ecfg = cfg["ensemble"]
    reports_dir = resolve_path(cfg["paths"]["reports"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features_path = resolve_path(cfg["paths"]["features_csv"])
    df = pd.read_csv(features_path)

    # Replicate ensemble/train.py's split exactly (same seed, same
    # stratify column) so this is the identical held-out test set that
    # reports/ensemble_metrics.json was scored on — without recomputing
    # embeddings for the training rows too.
    _, df_test = train_test_split(
        df, test_size=ecfg["test_split"], random_state=cfg["seed"], stratify=df["label"]
    )
    df_test = df_test.reset_index(drop=True)
    print(f"test set: n={len(df_test)} (replicated split, matches ensemble_metrics.json)")

    artifact_path = resolve_path(cfg["paths"]["ensemble_model"])
    artifact = joblib.load(artifact_path)
    calibrated = artifact["calibrated"]
    class_names = artifact["class_names"]

    embeddings = compute_cnn_embeddings(
        df_test["image_path"].tolist(),
        resolve_path(cfg["paths"]["mood_cnn_model"]),
        cfg["mood_cnn"]["input_size"],
        cfg["mood_cnn"]["embedding_dim"],
        device,
    )
    X_test = build_feature_matrix(df_test, embeddings)
    preds = calibrated.predict(X_test)
    df_test["correct"] = (preds == df_test["label"].values).astype(int)
    overall_acc = float(df_test["correct"].mean())
    print(f"overall calibrated accuracy (sanity check vs ensemble_metrics.json): {overall_acc:.4f}")

    print("computing pixel-derived lighting/coat-color proxies...")
    proxies = [brightness_and_saturation(p) for p in df_test["image_path"]]
    df_test["brightness"] = [b for b, s in proxies]
    df_test["saturation"] = [s for b, s in proxies]
    valid = df_test["brightness"].notna() & df_test["saturation"].notna()
    n_unreadable = int((~valid).sum())
    df_test = df_test[valid].reset_index(drop=True)

    df_test["lighting_bucket"] = tercile_bucket(df_test["brightness"])
    df_test["coat_bucket"] = tercile_bucket(df_test["saturation"])
    df_test["dataset_source"] = df_test["image_path"].apply(
        lambda p: "external_kaggle" if "external" in p else "crawford"
    )

    result = {
        "description": (
            "Accuracy of the deployed, already-trained ensemble "
            "(artifacts/ensemble_model.joblib), sliced by lighting and "
            "coat-color proxies plus label provenance and dataset source. "
            "Same held-out test split as reports/ensemble_metrics.json "
            "(same seed/stratify), not a fresh retrain."
        ),
        "caveat": (
            "Coat color and lighting are not annotated anywhere in this "
            "project's data. 'lighting' below is mean grayscale image "
            "brightness and 'coat_color' is mean HSV saturation, both "
            "computed directly from pixels and bucketed into terciles of "
            "this test set's own distribution -- real, measured pixel "
            "statistics, but heuristic proxies, not verified coat-color or "
            "lighting-condition labels. Saturation in particular is also "
            "affected by camera/background/lighting, not coat color alone."
        ),
        "n_test": int(len(df_test)),
        "n_unreadable_images_excluded": n_unreadable,
        "overall_calibrated_accuracy": overall_acc,
        "by_lighting_proxy_brightness_tercile": bucket_accuracy(df_test, "correct", "lighting_bucket"),
        "by_coat_color_proxy_saturation_tercile": bucket_accuracy(df_test, "correct", "coat_bucket"),
        "by_label_provenance": bucket_accuracy(df_test, "correct", "provenance"),
        "by_dataset_source": bucket_accuracy(df_test, "correct", "dataset_source"),
    }

    out_path = reports_dir / "subgroup_analysis.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved subgroup analysis to {out_path}")
    log_run("subgroup_analysis", result)
    for axis in ["by_lighting_proxy_brightness_tercile", "by_coat_color_proxy_saturation_tercile",
                 "by_label_provenance", "by_dataset_source"]:
        print(f"\n{axis}:")
        for bucket, stats in result[axis].items():
            print(f"  {bucket}: n={stats['n']} accuracy={stats['accuracy']}")


if __name__ == "__main__":
    main()
