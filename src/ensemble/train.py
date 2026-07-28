"""Ensemble classifier (Section 6): geometric features + CNN embedding ->
RandomForest wrapped in CalibratedClassifierCV, with SMOTE for underrepresented
classes applied *inside* each calibration CV fold (via an imblearn Pipeline)
rather than once upfront — applying SMOTE before CalibratedClassifierCV.fit()
would leak synthetic minority samples across the internal train/calibration
folds, silently inflating the calibration quality. Also guards against
reusing a single held-out split for both raw and calibrated comparison model
fitting to keep the comparison honest.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from torchvision import transforms
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import load_config, resolve_path  # noqa: E402
from src.features.geometric import FEATURE_NAMES  # noqa: E402
from src.mood_cnn.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.mood_cnn.model import MoodCNN  # noqa: E402


@torch.no_grad()
def compute_cnn_embeddings(image_paths: list[str], model_path: Path, input_size: int, embedding_dim: int, device) -> np.ndarray:
    model = MoodCNN(num_classes=4, embedding_dim=embedding_dim, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    embeddings = np.zeros((len(image_paths), embedding_dim), dtype=np.float32)
    for i, p in enumerate(image_paths):
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (input_size, input_size))
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        tensor = normalize(tensor).unsqueeze(0).to(device)
        embeddings[i] = model.embed(tensor).cpu().numpy()[0]
    return embeddings


def build_feature_matrix(df: pd.DataFrame, cnn_embeddings: np.ndarray) -> np.ndarray:
    geo = df[FEATURE_NAMES].values.astype(np.float32)
    return np.concatenate([geo, cnn_embeddings], axis=1)


def main():
    cfg = load_config()
    ecfg = cfg["ensemble"]
    class_names = cfg["mood_classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features_path = resolve_path(cfg["paths"]["features_csv"])
    df = pd.read_csv(features_path)
    print(f"total feature rows: {len(df)}")
    print(df["label"].value_counts())

    print("computing CNN embeddings...")
    embeddings = compute_cnn_embeddings(
        df["image_path"].tolist(),
        resolve_path(cfg["paths"]["mood_cnn_model"]),
        cfg["mood_cnn"]["input_size"],
        cfg["mood_cnn"]["embedding_dim"],
        device,
    )
    X = build_feature_matrix(df, embeddings)
    y = df["label"].values
    print(f"feature matrix shape: {X.shape}")

    # No separate held-out val split: CalibratedClassifierCV already does its
    # own internal cross-validation on the training set, so a manual val
    # split here would just be unused data rather than serving a purpose.
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=ecfg["test_split"], random_state=cfg["seed"], stratify=y
    )
    print(f"train={len(X_train)} test={len(X_test)}")

    min_class_count = pd.Series(y_train).value_counts().min()
    k_neighbors = max(1, min(ecfg["smote_k_neighbors_max"], min_class_count - 1))
    print(f"SMOTE k_neighbors={k_neighbors} (min train class count={min_class_count})")

    def make_pipeline():
        return ImbPipeline([
            ("smote", SMOTE(k_neighbors=k_neighbors, random_state=cfg["seed"])),
            ("rf", RandomForestClassifier(
                n_estimators=ecfg["rf_n_estimators"], max_depth=ecfg["rf_max_depth"],
                random_state=cfg["seed"], n_jobs=-1,
            )),
        ])

    print("fitting raw (uncalibrated) pipeline...")
    raw_pipeline = make_pipeline()
    raw_pipeline.fit(X_train, y_train)

    print("fitting calibrated pipeline (SMOTE inside each CV fold)...")
    calibrated = CalibratedClassifierCV(make_pipeline(), method=ecfg["calibration_method"], cv=ecfg["calibration_cv"])
    calibrated.fit(X_train, y_train)

    artifact_path = resolve_path(cfg["paths"]["ensemble_model"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"calibrated": calibrated, "raw": raw_pipeline, "class_names": class_names}, artifact_path)
    print(f"Saved ensemble model to {artifact_path}")

    return {
        "X_test": X_test, "y_test": y_test, "df_test": df_test,
        "raw_pipeline": raw_pipeline, "calibrated": calibrated, "class_names": class_names,
    }


if __name__ == "__main__":
    main()
