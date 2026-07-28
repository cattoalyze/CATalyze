"""Train the heatmap-based keypoint model, report real validation/test NME,
and save a visualization of predicted vs ground-truth keypoints on sample
images (so a degenerate/flat model can't hide behind a plausible loss curve).
"""
import json
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import load_config, resolve_path  # noqa: E402
from src.keypoints.dataset import CatKeypointsDataset, list_samples, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.keypoints.heatmap_utils import decode_heatmaps  # noqa: E402
from src.keypoints.model import HeatmapKeypointModel  # noqa: E402


def normalized_mean_error(pred_kp: np.ndarray, gt_kp: np.ndarray) -> float:
    """Per-keypoint Euclidean error normalized by inter-ocular (eye-to-eye)
    distance, averaged over keypoints and samples — standard NME metric for
    facial landmark localization."""
    eye_dist = np.linalg.norm(gt_kp[:, 0] - gt_kp[:, 1], axis=-1, keepdims=True)  # (N, 1)
    eye_dist = np.maximum(eye_dist, 1.0)
    err = np.linalg.norm(pred_kp - gt_kp, axis=-1)  # (N, K)
    return float((err / eye_dist).mean())


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    all_pred, all_gt = [], []
    for imgs, heatmaps_gt, kp_gt in loader:
        imgs = imgs.to(device)
        pred_heatmaps = model(imgs).cpu().numpy()
        for i in range(pred_heatmaps.shape[0]):
            kp_pred, _ = decode_heatmaps(pred_heatmaps[i], cfg["keypoints"]["input_size"], cfg["keypoints"]["heatmap_size"])
            all_pred.append(kp_pred)
            all_gt.append(kp_gt[i].numpy())
    return normalized_mean_error(np.array(all_pred), np.array(all_gt))


def visualize_predictions(model, dataset, device, cfg, out_path: Path, n: int = 6):
    model.eval()
    idxs = random.sample(range(len(dataset)), min(n, len(dataset)))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    for ax, idx in zip(axes.flat, idxs):
        img_t, _, kp_gt = dataset[idx]
        with torch.no_grad():
            pred_hm = model(img_t.unsqueeze(0).to(device)).cpu().numpy()[0]
        kp_pred, conf = decode_heatmaps(pred_hm, cfg["keypoints"]["input_size"], cfg["keypoints"]["heatmap_size"])
        img = (img_t.numpy() * std + mean).transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        kp_gt = kp_gt.numpy()
        ax.scatter(kp_gt[:, 0], kp_gt[:, 1], c="lime", s=25, marker="o", label="ground truth")
        ax.scatter(kp_pred[:, 0], kp_pred[:, 1], c="red", s=25, marker="x", label="predicted")
        ax.set_title(f"mean conf={conf.mean():.2f}")
        ax.axis("off")
    axes.flat[0].legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved prediction visualization to {out_path}")


def main():
    cfg = load_config()
    kcfg = cfg["keypoints"]
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    images_dir = resolve_path(cfg["paths"]["crawford_images_dir"])
    annotations_dir = resolve_path(cfg["paths"]["crawford_annotations_dir"])
    samples = list_samples(images_dir, annotations_dir, validate=True)
    print(f"total samples: {len(samples)}")

    trainval, test = train_test_split(samples, test_size=kcfg["test_split"], random_state=cfg["seed"])
    val_frac = kcfg["val_split"] / (1 - kcfg["test_split"])
    train, val = train_test_split(trainval, test_size=val_frac, random_state=cfg["seed"])
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    ds_kwargs = dict(input_size=kcfg["input_size"], heatmap_size=kcfg["heatmap_size"], sigma=kcfg["gaussian_sigma"])
    train_ds = CatKeypointsDataset(train, **ds_kwargs)
    val_ds = CatKeypointsDataset(val, **ds_kwargs)
    test_ds = CatKeypointsDataset(test, **ds_kwargs)

    num_workers = 4
    train_loader = DataLoader(train_ds, batch_size=kcfg["batch_size"], shuffle=True, num_workers=num_workers, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=kcfg["batch_size"], shuffle=False, num_workers=num_workers, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=kcfg["batch_size"], shuffle=False, num_workers=2)

    model = HeatmapKeypointModel(num_keypoints=kcfg["num_keypoints"], pretrained=True).to(device)
    criterion = nn.MSELoss()

    best_val_nme = float("inf")
    epochs_without_improvement = 0
    artifact_path = resolve_path(cfg["paths"]["keypoint_model"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    model.freeze_backbone()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=kcfg["learning_rate"])

    for epoch in range(kcfg["epochs"]):
        if epoch == kcfg["freeze_backbone_epochs"]:
            print(f"epoch {epoch}: unfreezing last {kcfg['finetune_last_n_layers']} backbone layers")
            model.unfreeze_last_n_layers(kcfg["finetune_last_n_layers"])
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=kcfg["finetune_learning_rate"]
            )

        model.train()
        train_loss = 0.0
        epoch_start = time.time()
        for imgs, heatmaps_gt, _ in train_loader:
            imgs, heatmaps_gt = imgs.to(device), heatmaps_gt.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, heatmaps_gt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        epoch_time = time.time() - epoch_start
        val_nme = evaluate(model, val_loader, device, cfg)
        print(f"epoch {epoch}: train_loss={train_loss:.5f} val_nme={val_nme:.4f} time={epoch_time:.1f}s")

        if val_nme < best_val_nme:
            best_val_nme = val_nme
            epochs_without_improvement = 0
            torch.save(model.state_dict(), artifact_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= kcfg["early_stopping_patience"]:
                print(f"early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(artifact_path, map_location=device))
    test_nme = evaluate(model, test_loader, device, cfg)
    print(f"FINAL best_val_nme={best_val_nme:.4f} test_nme={test_nme:.4f}")

    reports_dir = resolve_path(cfg["paths"]["reports"])
    visualize_predictions(model, test_ds, device, cfg, reports_dir / "keypoint_predictions_sample.png")

    result = {"best_val_nme": best_val_nme, "test_nme": test_nme, "n_train": len(train), "n_val": len(val), "n_test": len(test)}
    with open(reports_dir / "keypoint_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    main()
