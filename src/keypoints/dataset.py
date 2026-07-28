"""CAT_DATASET loading: parses `.cat` annotation files and resizes
image+keypoints together to a fixed square input size.

Known simplification: images are resized directly to `input_size` x
`input_size` (non-aspect-preserving "squash" resize) rather than letterboxed.
The network sees the same distortion at train and inference time so this
does not bias keypoint localization, it just means the model implicitly
learns on a squashed coordinate frame — documented here rather than silently
assumed.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.keypoints.heatmap_utils import generate_heatmaps

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_cat_file(path: Path) -> np.ndarray:
    parts = path.read_text().split()
    n = int(parts[0])
    coords = list(map(float, parts[1 : 1 + 2 * n]))
    return np.array(coords, dtype=np.float64).reshape(n, 2)


def _is_valid_annotation(img_path: Path, kp: np.ndarray, margin: float = 5.0) -> bool:
    """CAT_DATASET has a small fraction of corrupted/out-of-bounds annotations
    (confirmed by visual inspection — see reports/gt_keypoints_sanity_check.png).
    Drop any sample whose keypoints fall outside the image bounds."""
    if kp.shape != (9, 2):
        return False
    with Image.open(img_path) as im:
        w, h = im.size
    if np.any(kp[:, 0] < -margin) or np.any(kp[:, 0] > w + margin):
        return False
    if np.any(kp[:, 1] < -margin) or np.any(kp[:, 1] > h + margin):
        return False
    return True


def list_samples(images_dir: Path, annotations_dir: Path, validate: bool = True) -> list[tuple[Path, Path]]:
    images_dir, annotations_dir = Path(images_dir), Path(annotations_dir)
    samples = []
    dropped = 0
    for img_path in sorted(images_dir.glob("*.jpg")):
        cat_path = annotations_dir / (img_path.name + ".cat")
        if not cat_path.exists():
            continue
        if validate:
            try:
                kp = parse_cat_file(cat_path)
                if not _is_valid_annotation(img_path, kp):
                    dropped += 1
                    continue
            except Exception:
                dropped += 1
                continue
        samples.append((img_path, cat_path))
    if validate and dropped:
        print(f"list_samples: dropped {dropped} samples with invalid/out-of-bounds annotations")
    return samples


class CatKeypointsDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, Path]],
        input_size: int,
        heatmap_size: int,
        sigma: float,
    ):
        self.samples = samples
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, cat_path = self.samples[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]

        kp = parse_cat_file(cat_path)  # (9, 2) in original image coords
        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        scale = np.array([self.input_size / w0, self.input_size / h0])
        kp_resized = kp * scale

        heatmaps = generate_heatmaps(kp_resized, self.input_size, self.heatmap_size, self.sigma)

        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        img_tensor = self.normalize(img_tensor)

        return img_tensor, torch.from_numpy(heatmaps), torch.from_numpy(kp_resized).float()
