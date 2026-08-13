"""Bounding boxes derived algorithmically from the 9 facial keypoints —
no separate detector. Each box is the min/max rectangle of its constituent
keypoints, padded by a margin; its confidence is the mean of those same
keypoints' real heatmap-peak confidences (never a separately invented
number — see the continuation prompt's honesty requirement).

Keypoint order matches config.yaml `keypoints.names` / src.features.geometric:
    0 left_eye     1 right_eye    2 mouth
    3 left_ear_1   4 left_ear_2   5 left_ear_3   (base, tip, base)
    6 right_ear_1  7 right_ear_2  8 right_ear_3  (base, tip, base)
"""
import numpy as np

LEFT_EYE, RIGHT_EYE, MOUTH = 0, 1, 2
LEFT_EAR_IDX = [3, 4, 5]
RIGHT_EAR_IDX = [6, 7, 8]
FACE_IDX = [LEFT_EYE, RIGHT_EYE, MOUTH]

DEFAULT_MARGIN = 0.15
# Floor for degenerate boxes (e.g. near-collinear keypoints), as a fraction
# of eye-to-eye distance, so a box never collapses to a zero-area rectangle.
MIN_SIZE_FRAC_OF_EYE_DIST = 0.25


def _box_from_points(points: np.ndarray, confidences: np.ndarray, margin: float, min_size: float) -> dict:
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    width, height = x_max - x_min, y_max - y_min

    if width < min_size:
        cx = (x_min + x_max) / 2.0
        x_min, x_max = cx - min_size / 2.0, cx + min_size / 2.0
        width = min_size
    if height < min_size:
        cy = (y_min + y_max) / 2.0
        y_min, y_max = cy - min_size / 2.0, cy + min_size / 2.0
        height = min_size

    pad_x, pad_y = margin * width, margin * height
    return {
        "x_min": float(x_min - pad_x),
        "y_min": float(y_min - pad_y),
        "x_max": float(x_max + pad_x),
        "y_max": float(y_max + pad_y),
        "confidence": float(np.mean(confidences)),
    }


def compute_boxes(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    margin: float = DEFAULT_MARGIN,
    image_shape: tuple[int, int] | None = None,
) -> dict[str, dict]:
    """keypoints: (9, 2) array of (x, y) pixel coords. confidences: (9,).

    Returns {"left_ear": {...}, "right_ear": {...}, "face": {...}}, each a
    dict with x_min/y_min/x_max/y_max (pixel coords, image-space) and a
    confidence that is the mean of that box's real constituent-keypoint
    confidences. If image_shape=(H, W) is given, boxes are clamped to it.
    """
    kp = np.asarray(keypoints, dtype=np.float64)
    conf = np.asarray(confidences, dtype=np.float64)

    eye_dist = np.linalg.norm(kp[LEFT_EYE] - kp[RIGHT_EYE])
    min_size = max(eye_dist * MIN_SIZE_FRAC_OF_EYE_DIST, 1.0)

    boxes = {
        "left_ear": _box_from_points(kp[LEFT_EAR_IDX], conf[LEFT_EAR_IDX], margin, min_size),
        "right_ear": _box_from_points(kp[RIGHT_EAR_IDX], conf[RIGHT_EAR_IDX], margin, min_size),
        "face": _box_from_points(kp[FACE_IDX], conf[FACE_IDX], margin, min_size),
    }

    if image_shape is not None:
        h, w = image_shape[:2]
        for box in boxes.values():
            box["x_min"] = max(0.0, min(box["x_min"], w))
            box["x_max"] = max(0.0, min(box["x_max"], w))
            box["y_min"] = max(0.0, min(box["y_min"], h))
            box["y_max"] = max(0.0, min(box["y_max"], h))

    return boxes
