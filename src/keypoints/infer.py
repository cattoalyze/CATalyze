"""Single-image keypoint inference — used by geometric feature extraction
and the serving API."""
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from src.keypoints.heatmap_utils import decode_heatmaps
from src.keypoints.model import HeatmapKeypointModel

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class KeypointPredictor:
    def __init__(self, model_path: Path, num_keypoints: int, input_size: int, heatmap_size: int, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = HeatmapKeypointModel(num_keypoints=num_keypoints, pretrained=False).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def predict(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """image_rgb: (H, W, 3) uint8 RGB array, any size.

        Returns (keypoints, confidences): keypoints is (K, 2) in the
        *original* image's pixel coordinates, confidences is (K,).
        """
        h0, w0 = image_rgb.shape[:2]
        resized = cv2.resize(image_rgb, (self.input_size, self.input_size))
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
        tensor = self.normalize(tensor).unsqueeze(0).to(self.device)

        with torch.no_grad():
            heatmaps = self.model(tensor).cpu().numpy()[0]

        kp_input_space, conf = decode_heatmaps(heatmaps, self.input_size, self.heatmap_size)
        scale = np.array([w0 / self.input_size, h0 / self.input_size])
        kp_original_space = kp_input_space * scale
        return kp_original_space, conf
