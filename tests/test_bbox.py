import numpy as np
import pytest

from src.keypoints.bbox import compute_boxes


def make_keypoints_and_conf():
    kp = np.array([
        [-20, 0], [20, 0], [0, 30],                # left_eye, right_eye, mouth
        [-30, -10], [-25, -40], [-20, -10],         # left ear: base, tip, base
        [20, -10], [25, -40], [30, -10],            # right ear: base, tip, base
    ], dtype=np.float64)
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.95, 0.5, 0.4, 0.85, 0.3])
    return kp, conf


def test_returns_three_boxes():
    kp, conf = make_keypoints_and_conf()
    boxes = compute_boxes(kp, conf)
    assert set(boxes.keys()) == {"left_ear", "right_ear", "face"}


def test_box_contains_its_keypoints_before_padding():
    kp, conf = make_keypoints_and_conf()
    boxes = compute_boxes(kp, conf, margin=0.0)
    left_ear_pts = kp[[3, 4, 5]]
    b = boxes["left_ear"]
    assert b["x_min"] <= left_ear_pts[:, 0].min()
    assert b["x_max"] >= left_ear_pts[:, 0].max()
    assert b["y_min"] <= left_ear_pts[:, 1].min()
    assert b["y_max"] >= left_ear_pts[:, 1].max()


def test_margin_grows_the_box():
    kp, conf = make_keypoints_and_conf()
    tight = compute_boxes(kp, conf, margin=0.0)["face"]
    padded = compute_boxes(kp, conf, margin=0.15)["face"]
    assert padded["x_max"] - padded["x_min"] > tight["x_max"] - tight["x_min"]
    assert padded["y_max"] - padded["y_min"] > tight["y_max"] - tight["y_min"]


def test_confidence_is_honest_mean_of_constituent_keypoints():
    kp, conf = make_keypoints_and_conf()
    boxes = compute_boxes(kp, conf)
    # left_ear = indices 3, 4, 5
    assert boxes["left_ear"]["confidence"] == pytest.approx(conf[[3, 4, 5]].mean())
    assert boxes["right_ear"]["confidence"] == pytest.approx(conf[[6, 7, 8]].mean())
    assert boxes["face"]["confidence"] == pytest.approx(conf[[0, 1, 2]].mean())


def test_degenerate_collinear_points_produce_nonzero_area_box():
    kp, conf = make_keypoints_and_conf()
    # Collapse the right ear's 3 points onto a single point.
    kp[[6, 7, 8]] = [50, -50]
    boxes = compute_boxes(kp, conf)
    b = boxes["right_ear"]
    assert (b["x_max"] - b["x_min"]) > 0
    assert (b["y_max"] - b["y_min"]) > 0


def test_image_shape_clamps_boxes():
    kp, conf = make_keypoints_and_conf()
    kp = kp + [100, 100]  # shift into a small image
    boxes = compute_boxes(kp, conf, image_shape=(120, 120))
    for b in boxes.values():
        assert 0.0 <= b["x_min"] <= b["x_max"] <= 120.0
        assert 0.0 <= b["y_min"] <= b["y_max"] <= 120.0
