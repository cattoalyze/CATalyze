"""Honest evaluation of the ensemble: raw vs calibrated accuracy, per-class
P/R/F1, confusion matrix, reliability diagram + Brier score (raw vs
calibrated), and human/ai-labeled-only vs full-set accuracy — all computed
on the held-out test split produced by ensemble/train.py (never on data the
models were fit or calibrated on).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import load_config, resolve_path  # noqa: E402


def multiclass_brier_score(y_true_idx: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    one_hot = np.eye(n_classes)[y_true_idx]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def reliability_diagram(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_confidence, bin_accuracy, bin_counts = [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi if i < n_bins - 1 else confidences <= hi)
        if mask.sum() > 0:
            bin_confidence.append(confidences[mask].mean())
            bin_accuracy.append(correct[mask].mean())
            bin_counts.append(mask.sum())
    return np.array(bin_confidence), np.array(bin_accuracy), np.array(bin_counts)


def plot_reliability(raw_conf, raw_correct, cal_conf, cal_correct, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, conf, correct, title in [
        (axes[0], raw_conf, raw_correct, "Raw (uncalibrated)"),
        (axes[1], cal_conf, cal_correct, "Calibrated (sigmoid)"),
    ]:
        bc, ba, counts = reliability_diagram(conf, correct)
        ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
        ax.plot(bc, ba, "o-", label="model")
        ax.set_xlabel("mean predicted confidence")
        ax.set_ylabel("empirical accuracy")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved reliability diagram to {out_path}")


def evaluate_and_report(X_test, y_test, df_test, raw_pipeline, calibrated, class_names, reports_dir: Path):
    sk_classes = list(calibrated.classes_)  # sklearn-assigned class ordering
    # raw_pipeline and calibrated are fit on the same y_train, so sklearn's
    # np.unique(y)-based class ordering should match — verify rather than
    # assume, since a silent mismatch here would misalign every probability.
    assert list(raw_pipeline.classes_) == sk_classes, (
        f"class order mismatch between raw and calibrated models: "
        f"{list(raw_pipeline.classes_)} vs {sk_classes}"
    )
    y_test_idx = np.array([sk_classes.index(y) for y in y_test])

    raw_probs = raw_pipeline.predict_proba(X_test)
    cal_probs = calibrated.predict_proba(X_test)

    raw_preds_idx = raw_probs.argmax(axis=1)
    cal_preds_idx = cal_probs.argmax(axis=1)

    raw_acc = accuracy_score(y_test_idx, raw_preds_idx)
    cal_acc = accuracy_score(y_test_idx, cal_preds_idx)
    print(f"raw accuracy: {raw_acc:.4f}")
    print(f"calibrated accuracy: {cal_acc:.4f}")

    raw_brier = multiclass_brier_score(y_test_idx, raw_probs, len(sk_classes))
    cal_brier = multiclass_brier_score(y_test_idx, cal_probs, len(sk_classes))
    print(f"raw Brier score: {raw_brier:.4f}")
    print(f"calibrated Brier score: {cal_brier:.4f}")

    print("\n=== calibrated classification report (full test set) ===")
    report_full = classification_report(y_test_idx, cal_preds_idx, target_names=sk_classes, output_dict=True)
    print(classification_report(y_test_idx, cal_preds_idx, target_names=sk_classes))

    cm = confusion_matrix(y_test_idx, cal_preds_idx)
    print("confusion matrix (rows=true, cols=pred):")
    print(sk_classes)
    print(cm)

    # Honest reporting: accuracy on the original human/AI-reviewed seed
    # labels only (provenance == "ai"), *not* just "!= pseudo" — that
    # would silently also sweep in any other non-pseudo provenance (e.g.
    # external_kaggle_distressed) added later, changing what this number
    # means without anyone noticing.
    seed_mask = (df_test["provenance"] == "ai").values
    seed_only_acc = None
    if seed_mask.sum() > 0:
        seed_only_acc = accuracy_score(y_test_idx[seed_mask], cal_preds_idx[seed_mask])
        print(f"\nseed-labeled-only (provenance=ai) calibrated test_acc={seed_only_acc:.4f} (n={seed_mask.sum()})")

    external_mask = (df_test["provenance"] == "external_kaggle_distressed").values
    external_acc = None
    if external_mask.sum() > 0:
        external_acc = accuracy_score(y_test_idx[external_mask], cal_preds_idx[external_mask])
        print(f"external_kaggle_distressed-only calibrated test_acc={external_acc:.4f} (n={external_mask.sum()})")

    raw_conf = raw_probs.max(axis=1)
    raw_correct = (raw_preds_idx == y_test_idx).astype(int)
    cal_conf = cal_probs.max(axis=1)
    cal_correct = (cal_preds_idx == y_test_idx).astype(int)
    plot_reliability(raw_conf, raw_correct, cal_conf, cal_correct, reports_dir / "reliability_diagram.png")

    metrics = {
        "raw_accuracy": raw_acc,
        "calibrated_accuracy": cal_acc,
        "raw_brier_score": raw_brier,
        "calibrated_brier_score": cal_brier,
        "seed_labeled_only_calibrated_accuracy": seed_only_acc,
        "external_kaggle_distressed_only_calibrated_accuracy": external_acc,
        "class_names": sk_classes,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_full,
        "n_test": len(y_test_idx),
        "n_test_seed_labeled": int(seed_mask.sum()),
        "n_test_external_kaggle_distressed": int(external_mask.sum()),
    }
    return metrics


def main():
    from src.ensemble.train import main as train_main
    result = train_main()

    cfg = load_config()
    reports_dir = resolve_path(cfg["paths"]["reports"])
    metrics = evaluate_and_report(
        result["X_test"], result["y_test"], result["df_test"],
        result["raw_pipeline"], result["calibrated"], result["class_names"],
        reports_dir,
    )

    with open(reports_dir / "ensemble_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nSaved metrics to {reports_dir / 'ensemble_metrics.json'}")


if __name__ == "__main__":
    main()
