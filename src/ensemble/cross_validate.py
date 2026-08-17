"""Stratified k-fold cross-validation for the ensemble (Section 3, item 1
of the continuation prompt). The single 80/20 split used elsewhere
(src/ensemble/train.py) puts only 45 ANXIOUS examples in its test set —
too few for a trustworthy per-class read on the scarcest class. K-fold
cross-validation tests every one of the 226 ANXIOUS examples exactly once
(across the 5 folds combined), giving a materially larger effective sample
for that specific number, plus a fold-to-fold spread that shows how
noisy the single-split numbers actually are.

Fits both the raw pipeline and the full CalibratedClassifierCV (same
SMOTE-inside-cv setup as train.py) per fold, so results reflect what's
actually deployed, not a cheaper proxy.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import load_config, resolve_path  # noqa: E402
from src.ensemble.evaluate import multiclass_brier_score  # noqa: E402
from src.ensemble.train import build_feature_matrix, compute_cnn_embeddings  # noqa: E402
from src.reports.experiment_log import log_run  # noqa: E402


def main(n_splits: int = 5):
    cfg = load_config()
    ecfg = cfg["ensemble"]
    class_names = cfg["mood_classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features_path = resolve_path(cfg["paths"]["features_csv"])
    df = pd.read_csv(features_path)
    print(f"total feature rows: {len(df)}")

    print("computing CNN embeddings (once, reused across all folds)...")
    embeddings = compute_cnn_embeddings(
        df["image_path"].tolist(),
        resolve_path(cfg["paths"]["mood_cnn_model"]),
        cfg["mood_cnn"]["input_size"],
        cfg["mood_cnn"]["embedding_dim"],
        device,
    )
    X = build_feature_matrix(df, embeddings)
    y = df["label"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg["seed"])

    fold_results = []
    all_y_true, all_y_pred_raw, all_y_pred_cal = [], [], []
    all_probs_raw, all_probs_cal = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        min_class_count = pd.Series(y_train).value_counts().min()
        k_neighbors = max(1, min(ecfg["smote_k_neighbors_max"], min_class_count - 1))

        def make_pipeline(k=k_neighbors):
            return ImbPipeline([
                ("smote", SMOTE(k_neighbors=k, random_state=cfg["seed"])),
                ("rf", RandomForestClassifier(
                    n_estimators=ecfg["rf_n_estimators"], max_depth=ecfg["rf_max_depth"],
                    random_state=cfg["seed"], n_jobs=-1,
                )),
            ])

        raw_pipeline = make_pipeline()
        raw_pipeline.fit(X_train, y_train)
        calibrated = CalibratedClassifierCV(make_pipeline(), method=ecfg["calibration_method"], cv=ecfg["calibration_cv"])
        calibrated.fit(X_train, y_train)

        sk_classes = list(calibrated.classes_)
        y_test_idx = np.array([sk_classes.index(v) for v in y_test])

        raw_probs = raw_pipeline.predict_proba(X_test)
        cal_probs = calibrated.predict_proba(X_test)
        raw_pred_idx = raw_probs.argmax(axis=1)
        cal_pred_idx = cal_probs.argmax(axis=1)

        raw_acc = accuracy_score(y_test_idx, raw_pred_idx)
        cal_acc = accuracy_score(y_test_idx, cal_pred_idx)
        raw_brier = multiclass_brier_score(y_test_idx, raw_probs, len(sk_classes))
        cal_brier = multiclass_brier_score(y_test_idx, cal_probs, len(sk_classes))

        precision, recall, f1, support = precision_recall_fscore_support(
            y_test_idx, cal_pred_idx, labels=range(len(sk_classes)), zero_division=0
        )
        per_class = {
            cls: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for cls, p, r, f, s in zip(sk_classes, precision, recall, f1, support)
        }

        print(f"fold {fold_idx}: n_test={len(y_test)} raw_acc={raw_acc:.4f} cal_acc={cal_acc:.4f} "
              f"ANXIOUS(cal): P={per_class['ANXIOUS']['precision']:.2f} R={per_class['ANXIOUS']['recall']:.2f} "
              f"F1={per_class['ANXIOUS']['f1']:.2f} n={per_class['ANXIOUS']['support']}")

        fold_results.append({
            "fold": fold_idx, "n_train": len(X_train), "n_test": len(X_test),
            "raw_accuracy": float(raw_acc), "calibrated_accuracy": float(cal_acc),
            "raw_brier_score": float(raw_brier), "calibrated_brier_score": float(cal_brier),
            "per_class_calibrated": per_class,
        })

        all_y_true.append(y_test_idx)
        all_y_pred_raw.append(raw_pred_idx)
        all_y_pred_cal.append(cal_pred_idx)
        all_probs_raw.append(raw_probs)
        all_probs_cal.append(cal_probs)

    all_y_true = np.concatenate(all_y_true)
    all_y_pred_cal = np.concatenate(all_y_pred_cal)
    all_probs_cal = np.concatenate(all_probs_cal, axis=0)

    pooled_precision, pooled_recall, pooled_f1, pooled_support = precision_recall_fscore_support(
        all_y_true, all_y_pred_cal, labels=range(len(class_names)), zero_division=0
    )
    pooled_per_class = {
        cls: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
        for cls, p, r, f, s in zip(class_names, pooled_precision, pooled_recall, pooled_f1, pooled_support)
    }

    raw_accs = [f["raw_accuracy"] for f in fold_results]
    cal_accs = [f["calibrated_accuracy"] for f in fold_results]
    anxious_f1s = [f["per_class_calibrated"]["ANXIOUS"]["f1"] for f in fold_results]

    summary = {
        "n_splits": n_splits,
        "note": (
            "Every one of the 226 ANXIOUS examples in geometric_features.csv is tested exactly "
            "once across these folds (pooled_per_class support), vs. only 45 in the single "
            "80/20 split reports/ensemble_metrics.json reports. Compare the per-fold spread "
            "below to that single number to see how much it was noise."
        ),
        "calibrated_accuracy_mean": float(np.mean(cal_accs)),
        "calibrated_accuracy_std": float(np.std(cal_accs)),
        "raw_accuracy_mean": float(np.mean(raw_accs)),
        "raw_accuracy_std": float(np.std(raw_accs)),
        "anxious_f1_mean": float(np.mean(anxious_f1s)),
        "anxious_f1_std": float(np.std(anxious_f1s)),
        "anxious_f1_per_fold": anxious_f1s,
        "pooled_per_class_calibrated": pooled_per_class,
        "fold_results": fold_results,
    }

    print("\n=== summary across folds ===")
    print(f"calibrated accuracy: {summary['calibrated_accuracy_mean']:.4f} +/- {summary['calibrated_accuracy_std']:.4f}")
    print(f"raw accuracy:        {summary['raw_accuracy_mean']:.4f} +/- {summary['raw_accuracy_std']:.4f}")
    print(f"ANXIOUS F1 per fold: {[round(f, 3) for f in anxious_f1s]}")
    print(f"ANXIOUS F1:          {summary['anxious_f1_mean']:.4f} +/- {summary['anxious_f1_std']:.4f}")
    print(f"pooled ANXIOUS (n={pooled_per_class['ANXIOUS']['support']}): "
          f"P={pooled_per_class['ANXIOUS']['precision']:.4f} R={pooled_per_class['ANXIOUS']['recall']:.4f} "
          f"F1={pooled_per_class['ANXIOUS']['f1']:.4f}")

    out_path = resolve_path(cfg["paths"]["reports"]) / "ensemble_kfold_metrics.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")
    log_run("ensemble_kfold", summary)
    return summary


if __name__ == "__main__":
    main()
