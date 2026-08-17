"""Train the mood classification CNN branch on seed (provenance='ai') +
self-trained pseudo-labeled (provenance='pseudo') images. Reports accuracy
separately on the seed-labeled subset (honest metric, since no human rater
was available in this session — see README) vs the full test set, and saves
a prediction visualization for verification.

`train_mood_cnn` is the reusable core used both here and by
src/labeling/self_training.py — each call builds a *fresh* model/optimizer,
which matters: reusing a stateful model instance across self-training rounds
is a known failure mode (see EXECUTION_STRATEGY.md) that produces a
plausible-looking but meaningless result instead of a crash.
"""
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import load_config, resolve_path  # noqa: E402
from src.mood_cnn.dataset import MoodDataset, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.mood_cnn.model import MoodCNN  # noqa: E402
from src.reports.experiment_log import log_run  # noqa: E402


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for imgs, y in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        labels.extend(y.numpy().tolist())
    return np.array(preds), np.array(labels)


def visualize_predictions(model, dataset, class_names, device, out_path: Path, n: int = 9):
    model.eval()
    idxs = random.sample(range(len(dataset)), min(n, len(dataset)))
    fig, axes = plt.subplots(3, 3, figsize=(11, 11))
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    for ax, idx in zip(axes.flat, idxs):
        img_t, label = dataset[idx]
        with torch.no_grad():
            logits = model(img_t.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = probs.argmax()
        img = (img_t.numpy() * std + mean).transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        color = "green" if pred == label else "red"
        ax.set_title(f"true={class_names[label]} pred={class_names[pred]} ({probs[pred]:.2f})", color=color, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"Saved mood prediction visualization to {out_path}")


def train_mood_cnn(df: pd.DataFrame, cfg: dict, device: torch.device, artifact_path: Path, verbose: bool = True):
    """Train a *fresh* MoodCNN on the given labeled dataframe. Returns
    (model, best_val_acc, test_acc, test_preds, test_labels, test_df)."""
    mcfg = cfg["mood_cnn"]
    class_names = cfg["mood_classes"]

    trainval, test = train_test_split(df, test_size=mcfg["test_split"], random_state=cfg["seed"], stratify=df["label"])
    val_frac = mcfg["val_split"] / (1 - mcfg["test_split"])
    train, val = train_test_split(trainval, test_size=val_frac, random_state=cfg["seed"], stratify=trainval["label"])
    if verbose:
        print(f"train={len(train)} val={len(val)} test={len(test)}")

    train_ds = MoodDataset(train, class_names, mcfg["input_size"], augment=True)
    val_ds = MoodDataset(val, class_names, mcfg["input_size"], augment=False)
    test_ds = MoodDataset(test, class_names, mcfg["input_size"], augment=False)

    train_loader = DataLoader(train_ds, batch_size=mcfg["batch_size"], shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=mcfg["batch_size"], shuffle=False, num_workers=2, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=mcfg["batch_size"], shuffle=False, num_workers=2)

    model = MoodCNN(num_classes=len(class_names), embedding_dim=mcfg["embedding_dim"], pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    epochs_without_improvement = 0
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    model.freeze_backbone()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=mcfg["learning_rate"])

    for epoch in range(mcfg["epochs"]):
        if epoch == mcfg["freeze_backbone_epochs"]:
            if verbose:
                print(f"epoch {epoch}: unfreezing last {mcfg['finetune_last_n_layers']} backbone layers")
            model.unfreeze_last_n_layers(mcfg["finetune_last_n_layers"])
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=mcfg["finetune_learning_rate"]
            )

        model.train()
        train_loss = 0.0
        t0 = time.time()
        for imgs, y in train_loader:
            imgs, y = imgs.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        val_preds, val_labels = evaluate(model, val_loader, device)
        val_acc = accuracy_score(val_labels, val_preds)
        if verbose:
            print(f"epoch {epoch}: train_loss={train_loss:.4f} val_acc={val_acc:.4f} time={time.time()-t0:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), artifact_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= mcfg["early_stopping_patience"]:
                if verbose:
                    print(f"early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(artifact_path, map_location=device))
    test_preds, test_labels = evaluate(model, test_loader, device)
    test_acc = accuracy_score(test_labels, test_preds)

    return model, best_val_acc, test_acc, test_preds, test_labels, test.reset_index(drop=True)


def main():
    cfg = load_config()
    class_names = cfg["mood_classes"]
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    labels_path = resolve_path(cfg["paths"]["mood_labels_csv"])
    df = pd.read_csv(labels_path)
    provenance_counts = df["provenance"].value_counts().to_dict()
    print(f"total labeled samples: {len(df)} provenance breakdown: {provenance_counts}")

    artifact_path = resolve_path(cfg["paths"]["mood_cnn_model"])
    model, best_val_acc, test_acc, test_preds, test_labels, test_df = train_mood_cnn(df, cfg, device, artifact_path)

    print(f"FINAL best_val_acc={best_val_acc:.4f} test_acc={test_acc:.4f}")
    print(classification_report(test_labels, test_preds, target_names=class_names))

    # Honest reporting: accuracy on seed (non-pseudo) test rows only.
    seed_mask = (test_df["provenance"] != "pseudo").values
    if seed_mask.sum() > 0:
        seed_acc = accuracy_score(test_labels[seed_mask], test_preds[seed_mask])
        print(f"seed-labeled-only (non-pseudo) test_acc={seed_acc:.4f} (n={seed_mask.sum()})")

    mcfg = cfg["mood_cnn"]
    test_ds = MoodDataset(test_df, class_names, mcfg["input_size"], augment=False)
    reports_dir = resolve_path(cfg["paths"]["reports"])
    visualize_predictions(model, test_ds, class_names, device, reports_dir / "mood_predictions_sample.png")

    result = {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "seed_labeled_only_test_acc": float(seed_acc) if seed_mask.sum() > 0 else None,
        "n_test": len(test_labels),
        "n_test_seed_labeled": int(seed_mask.sum()),
        "provenance_breakdown": provenance_counts,
    }
    import json
    with open(reports_dir / "mood_cnn_metrics.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    log_run("mood_cnn", result)
    return result


if __name__ == "__main__":
    main()
