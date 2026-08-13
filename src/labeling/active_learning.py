"""Active learning via uncertainty sampling (Section 3, item 2 of the
continuation prompt): ranks the still-unlabeled pool (images self-training
never crossed a pseudo-label confidence threshold for) by the current mood
CNN's predictive entropy, so a human reviewer spends their time on the
images most likely to actually move the model — the complement of
self-training's confidence-threshold approach (src/labeling/self_training.py),
which does the opposite: it only keeps the pool images the model is
*already* confident about. Reuses build_label_grids.py's grid-rendering so
the output can be reviewed the same way the original seed set was.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import load_config, resolve_path  # noqa: E402
from src.labeling.build_label_grids import build_grid  # noqa: E402
from src.mood_cnn.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.mood_cnn.model import MoodCNN  # noqa: E402


def entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy per row, nats. Max for 4 classes = ln(4) ~= 1.386."""
    eps = 1e-12
    return -np.sum(probs * np.log(probs + eps), axis=1)


@torch.no_grad()
def score_pool(model, pool_paths: list[Path], input_size: int, device, batch_size: int = 16) -> pd.DataFrame:
    model.eval()
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    class_names_local = None
    rows = []
    for start in range(0, len(pool_paths), batch_size):
        batch_paths = pool_paths[start : start + batch_size]
        tensors, valid_paths = [], []
        for p in batch_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (input_size, input_size))
            t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            tensors.append(normalize(t))
            valid_paths.append(p)
        if not tensors:
            continue
        batch = torch.stack(tensors).to(device)
        probs = torch.softmax(model(batch), dim=1).cpu().numpy()
        ent = entropy(probs)
        pred_idx = probs.argmax(axis=1)
        for p, pi, pr, e in zip(valid_paths, pred_idx, probs, ent):
            rows.append({"image_path": str(p), "predicted_idx": int(pi), "confidence": float(pr[pi]), "entropy": float(e)})
    return pd.DataFrame(rows)


def main(n_candidates: int = 100, cols: int = 4, rows: int = 5):
    cfg = load_config()
    class_names = cfg["mood_classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels_path = resolve_path(cfg["paths"]["mood_labels_csv"])
    labeled_df = pd.read_csv(labels_path)
    labeled = set(str(Path(p).resolve()) for p in labeled_df["image_path"])

    images_dir = resolve_path(cfg["paths"]["crawford_images_dir"])
    all_images = set(str(p.resolve()) for p in images_dir.glob("*.jpg"))
    pool = sorted(Path(p) for p in (all_images - labeled))
    print(f"unlabeled pool: {len(pool)} images (never crossed self-training's confidence threshold)")

    if not pool:
        print("pool is empty, nothing to score")
        return

    model = MoodCNN(num_classes=len(class_names), embedding_dim=cfg["mood_cnn"]["embedding_dim"], pretrained=False).to(device)
    model.load_state_dict(torch.load(resolve_path(cfg["paths"]["mood_cnn_model"]), map_location=device))

    scores = score_pool(model, pool, cfg["mood_cnn"]["input_size"], device)
    scores["predicted_label"] = scores["predicted_idx"].apply(lambda i: class_names[i])
    scores = scores.sort_values("entropy", ascending=False).reset_index(drop=True)
    scores["rank"] = scores.index

    print(f"\nentropy range: max={scores['entropy'].max():.3f} (ln(4)={np.log(4):.3f} = maximally uncertain), "
          f"min={scores['entropy'].min():.3f}, mean={scores['entropy'].mean():.3f}")
    print("\npredicted-label breakdown of the top", min(n_candidates, len(scores)), "most uncertain:")
    print(scores.head(n_candidates)["predicted_label"].value_counts())

    reports_dir = resolve_path(cfg["paths"]["reports"])
    candidates = scores.head(n_candidates)
    candidates_out = reports_dir / "active_learning_candidates.csv"
    candidates.to_csv(candidates_out, index=False)
    print(f"\nSaved ranked candidates to {candidates_out}")

    grid_dir = resolve_path(cfg["paths"]["data_processed"]) / "active_learning_grids"
    grid_dir.mkdir(parents=True, exist_ok=True)
    per_grid = cols * rows
    manifest = {}
    grid_idx = 0
    candidate_paths = [Path(p) for p in candidates["image_path"]]
    for start in range(0, len(candidate_paths), per_grid):
        batch = candidate_paths[start : start + per_grid]
        grid_path = grid_dir / f"al_grid_{grid_idx:03d}.png"
        build_grid(batch, grid_path, cols, rows)
        manifest[grid_path.name] = [
            {"image_path": str(p), "predicted_label": row["predicted_label"], "confidence": row["confidence"], "entropy": row["entropy"]}
            for p, (_, row) in zip(batch, candidates.iloc[start : start + per_grid].iterrows())
        ]
        grid_idx += 1
    with open(grid_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Built {grid_idx} review grids ({len(candidate_paths)} images) in {grid_dir}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n_candidates", type=int, default=100)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=5)
    args = ap.parse_args()
    main(args.n_candidates, args.cols, args.rows)
