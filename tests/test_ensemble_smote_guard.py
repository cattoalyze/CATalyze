import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier


def _k_neighbors_guard(y_train, max_k: int) -> int:
    """Mirrors the guard logic in src/ensemble/train.py: SMOTE's k_neighbors
    must be < the minority class's sample count, or SMOTE raises."""
    min_class_count = pd.Series(y_train).value_counts().min()
    return max(1, min(max_k, min_class_count - 1))


def test_k_neighbors_guard_reduces_for_small_minority_class():
    y = np.array(["A"] * 50 + ["B"] * 3)
    k = _k_neighbors_guard(y, max_k=5)
    assert k == 2  # min_class_count(3) - 1


def test_k_neighbors_guard_caps_at_max_for_large_classes():
    y = np.array(["A"] * 50 + ["B"] * 50)
    k = _k_neighbors_guard(y, max_k=5)
    assert k == 5


def test_k_neighbors_guard_never_below_one():
    y = np.array(["A"] * 50 + ["B"] * 1)
    k = _k_neighbors_guard(y, max_k=5)
    assert k == 1


def test_smote_pipeline_runs_with_guarded_k_neighbors_on_tiny_minority():
    rng = np.random.RandomState(0)
    X_majority = rng.randn(40, 6)
    X_minority = rng.randn(3, 6)
    X = np.vstack([X_majority, X_minority])
    y = np.array(["A"] * 40 + ["B"] * 3)

    k = _k_neighbors_guard(y, max_k=5)
    pipeline = ImbPipeline([
        ("smote", SMOTE(k_neighbors=k, random_state=0)),
        ("rf", RandomForestClassifier(n_estimators=10, random_state=0)),
    ])
    pipeline.fit(X, y)  # should not raise
    preds = pipeline.predict(X)
    assert len(preds) == len(y)


def test_calibration_does_not_leak_smote_samples_across_folds():
    """CalibratedClassifierCV with cv=N and a SMOTE-containing pipeline must
    refit SMOTE fresh inside each fold (via imblearn's Pipeline), not have
    synthetic samples generated once and shared across calibration folds."""
    rng = np.random.RandomState(1)
    X = rng.randn(60, 6)
    y = np.array(["A"] * 45 + ["B"] * 15)

    k = _k_neighbors_guard(y, max_k=5)
    pipeline = ImbPipeline([
        ("smote", SMOTE(k_neighbors=k, random_state=0)),
        ("rf", RandomForestClassifier(n_estimators=10, random_state=0)),
    ])
    calibrated = CalibratedClassifierCV(pipeline, method="sigmoid", cv=3)
    calibrated.fit(X, y)  # should not raise, and each internal estimator refits its own SMOTE
    probs = calibrated.predict_proba(X)
    assert probs.shape == (60, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
