"""Gaussian ground-truth heatmap generation and peak-based decoding.

Ground truth: a 2D Gaussian (sigma in heatmap-pixel units) centered at each
true keypoint, in heatmap space. Decoding: argmax of the predicted heatmap,
scaled back to input-image space; the peak value itself is used as a
genuine per-keypoint confidence score (Section 3 of the brief).
"""
import numpy as np


def generate_heatmaps(
    keypoints_input_space: np.ndarray,
    input_size: int,
    heatmap_size: int,
    sigma: float,
) -> np.ndarray:
    """Build (K, H, W) Gaussian heatmaps from (K, 2) keypoints in input-image
    pixel coordinates. A keypoint whose center lands outside the heatmap
    grid still contributes (Gaussian tails may partially overlap); no
    visibility masking is applied here since the CAT_DATASET keypoints are
    always fully labeled.
    """
    stride = input_size / heatmap_size
    kp_hm = keypoints_input_space / stride  # (K, 2), in heatmap-pixel coords

    yy, xx = np.meshgrid(np.arange(heatmap_size), np.arange(heatmap_size), indexing="ij")
    heatmaps = np.zeros((kp_hm.shape[0], heatmap_size, heatmap_size), dtype=np.float32)
    for k in range(kp_hm.shape[0]):
        cx, cy = kp_hm[k]
        heatmaps[k] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return heatmaps


def decode_heatmaps(heatmaps: np.ndarray, input_size: int, heatmap_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode (K, H, W) predicted heatmaps to keypoints in input-image space.

    Returns (keypoints, confidences) where keypoints is (K, 2) and
    confidences is (K,) — the raw heatmap peak value per keypoint.
    """
    stride = input_size / heatmap_size
    k = heatmaps.shape[0]
    flat_idx = heatmaps.reshape(k, -1).argmax(axis=1)
    ys, xs = np.unravel_index(flat_idx, (heatmap_size, heatmap_size))
    confidences = heatmaps.reshape(k, -1)[np.arange(k), flat_idx]
    keypoints = np.stack([xs, ys], axis=-1).astype(np.float64) * stride + stride / 2.0
    return keypoints, confidences
