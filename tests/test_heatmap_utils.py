import numpy as np
import pytest

from src.keypoints.heatmap_utils import generate_heatmaps, decode_heatmaps

INPUT_SIZE = 224
HEATMAP_SIZE = 56
SIGMA = 1.5


def test_heatmap_shape():
    kp = np.array([[100, 50], [150, 180]], dtype=np.float64)
    hm = generate_heatmaps(kp, INPUT_SIZE, HEATMAP_SIZE, SIGMA)
    assert hm.shape == (2, HEATMAP_SIZE, HEATMAP_SIZE)


def test_heatmap_peak_at_keypoint_location():
    kp = np.array([[112, 112]], dtype=np.float64)  # center of image
    hm = generate_heatmaps(kp, INPUT_SIZE, HEATMAP_SIZE, SIGMA)
    peak_idx = np.unravel_index(hm[0].argmax(), hm[0].shape)
    expected = int(112 / (INPUT_SIZE / HEATMAP_SIZE))
    assert abs(peak_idx[0] - expected) <= 1
    assert abs(peak_idx[1] - expected) <= 1


def test_heatmap_peak_value_is_one():
    kp = np.array([[112, 112]], dtype=np.float64)
    hm = generate_heatmaps(kp, INPUT_SIZE, HEATMAP_SIZE, SIGMA)
    assert hm.max() == pytest.approx(1.0, abs=1e-5)


def test_decode_recovers_original_keypoint_within_stride():
    kp = np.array([[80.0, 160.0], [40.0, 200.0]])
    hm = generate_heatmaps(kp, INPUT_SIZE, HEATMAP_SIZE, SIGMA)
    decoded, conf = decode_heatmaps(hm, INPUT_SIZE, HEATMAP_SIZE)
    stride = INPUT_SIZE / HEATMAP_SIZE
    assert np.all(np.abs(decoded - kp) <= stride)
    assert np.all(conf > 0.9)


def test_decode_confidence_low_for_flat_heatmap():
    flat = np.zeros((1, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
    _, conf = decode_heatmaps(flat, INPUT_SIZE, HEATMAP_SIZE)
    assert conf[0] == pytest.approx(0.0)
