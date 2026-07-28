"""Geometric feature engineering from the 9 facial keypoints (Section 4).

Keypoint order (matches config.yaml `keypoints.names` and the CAT_DATASET
.cat annotation format):
    0 left_eye     1 right_eye    2 mouth
    3 left_ear_1   4 left_ear_2   5 left_ear_3   (base, tip, base)
    6 right_ear_1  7 right_ear_2  8 right_ear_3  (base, tip, base)

Coordinates are image-space (x right, y down). All functions accept either
a single (9, 2) array or a batch (N, 9, 2) array and return matching shapes.

Feature vector order (7-dim): left_ear_angle, right_ear_angle, ear_spread,
ear_symmetry, left_ear_height, right_ear_height, face_compactness.
"""
import numpy as np

LEFT_EYE, RIGHT_EYE, MOUTH = 0, 1, 2
LEFT_EAR_BASE_A, LEFT_EAR_TIP, LEFT_EAR_BASE_B = 3, 4, 5
RIGHT_EAR_BASE_A, RIGHT_EAR_TIP, RIGHT_EAR_BASE_B = 6, 7, 8

FEATURE_NAMES = [
    "left_ear_angle",
    "right_ear_angle",
    "ear_spread",
    "ear_symmetry",
    "left_ear_height",
    "right_ear_height",
    "face_compactness",
]

EYE_DISTANCE_FLOOR_PX = 1.0


def _as_batch(keypoints: np.ndarray) -> tuple[np.ndarray, bool]:
    kp = np.asarray(keypoints, dtype=np.float64)
    if kp.ndim == 2:
        return kp[None, ...], True
    return kp, False


def _eye_distance(kp: np.ndarray) -> np.ndarray:
    """Eye-to-eye distance per sample, used as the head-width normalizer."""
    d = np.linalg.norm(kp[:, LEFT_EYE] - kp[:, RIGHT_EYE], axis=-1)
    return np.maximum(d, EYE_DISTANCE_FLOOR_PX)


def _ear_angle(base_a: np.ndarray, tip: np.ndarray, base_b: np.ndarray) -> np.ndarray:
    """Angle (radians) of the ear axis — base midpoint to tip — from vertical.

    0 = ear points straight up, positive = tilted toward +x (image right),
    negative = tilted toward -x. Image y grows downward, so "up" is -y.
    """
    base_mid = (base_a + base_b) / 2.0
    vec = tip - base_mid
    return np.arctan2(vec[:, 0], -vec[:, 1])


def compute_geometric_features(keypoints: np.ndarray) -> np.ndarray:
    """Compute the 7-dim geometric feature vector from 9 facial keypoints.

    Parameters
    ----------
    keypoints : (9, 2) or (N, 9, 2) array of (x, y) pixel coordinates.

    Returns
    -------
    (7,) or (N, 7) array, column order per FEATURE_NAMES.
    """
    kp, squeeze = _as_batch(keypoints)
    head_width = _eye_distance(kp)

    left_angle = _ear_angle(kp[:, LEFT_EAR_BASE_A], kp[:, LEFT_EAR_TIP], kp[:, LEFT_EAR_BASE_B])
    right_angle = _ear_angle(kp[:, RIGHT_EAR_BASE_A], kp[:, RIGHT_EAR_TIP], kp[:, RIGHT_EAR_BASE_B])

    # Re-express both ears in a shared "outward tilt" frame: positive = ear
    # tilts away from the face midline (splayed), negative = tilts inward.
    left_outward = -left_angle
    right_outward = right_angle

    # Spread: average outward splay of both ears (radians).
    ear_spread = (left_outward + right_outward) / 2.0
    # Symmetry: 0 = mirrored ear posture, larger = one ear tilts differently than the other.
    ear_symmetry = np.abs(left_outward - right_outward)

    eye_line_y = (kp[:, LEFT_EYE, 1] + kp[:, RIGHT_EYE, 1]) / 2.0
    left_ear_height = (eye_line_y - kp[:, LEFT_EAR_TIP, 1]) / head_width
    right_ear_height = (eye_line_y - kp[:, RIGHT_EAR_TIP, 1]) / head_width

    eye_mid = (kp[:, LEFT_EYE] + kp[:, RIGHT_EYE]) / 2.0
    face_compactness = np.linalg.norm(kp[:, MOUTH] - eye_mid, axis=-1) / head_width

    features = np.stack(
        [left_angle, right_angle, ear_spread, ear_symmetry, left_ear_height, right_ear_height, face_compactness],
        axis=-1,
    )
    return features[0] if squeeze else features
