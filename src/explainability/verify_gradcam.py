"""Visual verification of Grad-CAM on real predictions: 2 correct examples
per mood class plus 1 misclassified example, saved as a labeled grid so a
human can actually look at whether the highlighted region is plausible
(ears/face) rather than trusting the implementation blindly.
"""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import load_config, resolve_path  # noqa: E402
from src.explainability.gradcam import GradCAM, overlay_cam_on_image  # noqa: E402
from src.mood_cnn.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.mood_cnn.model import MoodCNN  # noqa: E402


def load_image_tensor(path: str, input_size: int, normalize):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (input_size, input_size))
    tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
    tensor = normalize(tensor)
    return tensor, img_resized


def main():
    cfg = load_config()
    mcfg = cfg["mood_cnn"]
    class_names = cfg["mood_classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MoodCNN(num_classes=len(class_names), embedding_dim=mcfg["embedding_dim"], pretrained=False).to(device)
    model.load_state_dict(torch.load(resolve_path(cfg["paths"]["mood_cnn_model"]), map_location=device))
    model.eval()

    cam_extractor = GradCAM(model)
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    df = pd.read_csv(resolve_path(cfg["paths"]["mood_labels_csv"]))
    seed_df = df[df["provenance"] != "pseudo"].reset_index(drop=True)  # honest ground truth only

    # Find 2 correct + look for 1 misclassified example, per class where possible.
    examples = []  # (image_path, true_label, title)
    misclassified = None
    for _, row in seed_df.sample(frac=1.0, random_state=cfg["seed"]).iterrows():
        tensor, _ = load_image_tensor(row["image_path"], mcfg["input_size"], normalize)
        with torch.no_grad():
            pred_idx = model(tensor.unsqueeze(0).to(device)).argmax(dim=1).item()
        pred_label = class_names[pred_idx]
        true_label = row["label"]

        if pred_label == true_label:
            count_for_class = sum(1 for e in examples if e[1] == true_label and e[2] == "correct")
            if count_for_class < 2:
                examples.append((row["image_path"], true_label, "correct"))
        elif misclassified is None:
            misclassified = (row["image_path"], true_label, pred_label)

        per_class_counts = {c: sum(1 for e in examples if e[1] == c) for c in class_names}
        if all(v >= 2 for v in per_class_counts.values()) and misclassified is not None:
            break

    if misclassified is not None:
        examples.append((misclassified[0], misclassified[1], f"WRONG pred={misclassified[2]}"))

    n = len(examples)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, (img_path, true_label, kind) in zip(axes, examples):
        tensor, img_resized = load_image_tensor(img_path, mcfg["input_size"], normalize)
        tensor = tensor.unsqueeze(0).to(device)
        tensor.requires_grad_(False)
        cam, pred_idx, logits = cam_extractor(tensor)
        pred_label = class_names[pred_idx]
        overlay = overlay_cam_on_image(img_resized, cam)

        ax.imshow(overlay)
        color = "red" if "WRONG" in kind else "black"
        ax.set_title(f"true={true_label} pred={pred_label}\n({kind})", fontsize=9, color=color)
        ax.axis("off")

    for ax in axes[len(examples):]:
        ax.axis("off")

    plt.tight_layout()
    reports_dir = resolve_path(cfg["paths"]["reports"])
    out_path = reports_dir / "gradcam_verification.png"
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved Grad-CAM verification grid ({n} examples) to {out_path}")


if __name__ == "__main__":
    main()
