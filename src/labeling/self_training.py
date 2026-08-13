"""Self-training / pseudo-labeling: bootstrap a mood classifier on the seed
labels, use it to label the remaining unlabeled pool at per-class confidence
thresholds (config.yaml labeling.pseudo_label_confidence_threshold), retrain
on seed+pseudo, and repeat for self_training_max_rounds.

Each round trains a *fresh* MoodCNN via train_mood_cnn — never reuses a
model/optimizer instance across rounds (see mood_cnn/train.py docstring for
why that specific bug matters here).
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import load_config, resolve_path  # noqa: E402
from src.mood_cnn.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.mood_cnn.train import train_mood_cnn  # noqa: E402


@torch.no_grad()
def pseudo_label_pool(model, pool_paths: list[Path], class_names: list[str], input_size: int, thresholds: dict, device, batch_size: int = 16) -> pd.DataFrame:
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    rows = []
    for start in tqdm(range(0, len(pool_paths), batch_size), desc="pseudo-labeling pool"):
        batch_paths = pool_paths[start : start + batch_size]
        tensors = []
        valid_paths = []
        for img_path in batch_paths:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (input_size, input_size))
                t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                tensors.append(normalize(t))
                valid_paths.append(img_path)
            except Exception:
                continue
        if not tensors:
            continue
        batch = torch.stack(tensors).to(device)
        probs = torch.softmax(model(batch), dim=1).cpu().numpy()
        pred_idx = probs.argmax(axis=1)
        for img_path, pi, p in zip(valid_paths, pred_idx, probs):
            pred_class = class_names[pi]
            if p[pi] >= thresholds[pred_class]:
                rows.append({"image_path": str(img_path), "label": pred_class, "provenance": "pseudo"})
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    lcfg = cfg["labeling"]
    class_names = cfg["mood_classes"]
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"device: {device}")

    labels_path = resolve_path(cfg["paths"]["mood_labels_csv"])
    df = pd.read_csv(labels_path)
    seed_df = df[df["provenance"] != "pseudo"].copy()
    print(f"seed labels: {len(seed_df)}")

    images_dir = resolve_path(cfg["paths"]["crawford_images_dir"])
    all_images = set(str(p) for p in images_dir.glob("*.jpg"))
    labeled_images = set(seed_df["image_path"])
    pool = sorted(Path(p) for p in (all_images - labeled_images))
    print(f"unlabeled pool: {len(pool)}")

    artifact_path = resolve_path(cfg["paths"]["mood_cnn_model"])
    current_df = seed_df.copy()

    for round_idx in range(lcfg["self_training_max_rounds"]):
        print(f"\n=== self-training round {round_idx}: training on {len(current_df)} labels ===")
        model, best_val_acc, test_acc, _, _, _ = train_mood_cnn(current_df, cfg, device, artifact_path, verbose=False)
        print(f"round {round_idx}: best_val_acc={best_val_acc:.4f} test_acc={test_acc:.4f}")

        remaining_pool = [p for p in pool if str(p) not in set(current_df["image_path"])]
        if not remaining_pool:
            print("pool exhausted, stopping")
            break

        new_pseudo = pseudo_label_pool(
            model, remaining_pool, class_names, cfg["mood_cnn"]["input_size"],
            lcfg["pseudo_label_confidence_threshold"], device,
        )
        print(f"round {round_idx}: newly pseudo-labeled {len(new_pseudo)} / {len(remaining_pool)} pool images")
        print(new_pseudo["label"].value_counts() if len(new_pseudo) else "  (none)")

        if len(new_pseudo) == 0:
            print("no new pseudo-labels found, stopping")
            break

        current_df = pd.concat([current_df, new_pseudo], ignore_index=True)

    current_df.to_csv(labels_path, index=False)
    print(f"\nFinal label set: {len(current_df)} rows")
    print(current_df.groupby(["provenance", "label"]).size())
    print(f"Saved to {labels_path}")


if __name__ == "__main__":
    main()
