import numpy as np
import pytest

from src.features.geometric import compute_geometric_features, FEATURE_NAMES


def make_keypoints(
    left_eye=(-20, 0), right_eye=(20, 0), mouth=(0, 30),
    left_ear_base_a=(-30, -10), left_ear_tip=(-25, -40), left_ear_base_b=(-20, -10),
    right_ear_base_a=(20, -10), right_ear_tip=(25, -40), right_ear_base_b=(30, -10),
):
    return np.array([
        left_eye, right_eye, mouth,
        left_ear_base_a, left_ear_tip, left_ear_base_b,
        right_ear_base_a, right_ear_tip, right_ear_base_b,
    ], dtype=np.float64)


def feat_dict(kp):
    return dict(zip(FEATURE_NAMES, compute_geometric_features(kp)))


def test_output_shape_single():
    kp = make_keypoints()
    out = compute_geometric_features(kp)
    assert out.shape == (7,)


def test_output_shape_batch():
    kp = np.stack([make_keypoints(), make_keypoints()])
    out = compute_geometric_features(kp)
    assert out.shape == (2, 7)


def test_symmetric_cat_has_zero_symmetry_and_spread():
    # Both ears tilt outward by the same amount -> mirrored posture.
    kp = make_keypoints(
        left_ear_base_a=(-30, -10), left_ear_tip=(-35, -40), left_ear_base_b=(-20, -10),
        right_ear_base_a=(20, -10), right_ear_tip=(35, -40), right_ear_base_b=(30, -10),
    )
    f = feat_dict(kp)
    assert f["ear_symmetry"] == pytest.approx(0.0, abs=1e-9)
    assert f["ear_spread"] > 0  # both ears splayed outward


def test_straight_up_ears_have_zero_angle():
    kp = make_keypoints(
        left_ear_base_a=(-30, -10), left_ear_tip=(-25, -40), left_ear_base_b=(-20, -10),
        right_ear_base_a=(20, -10), right_ear_tip=(25, -40), right_ear_base_b=(30, -10),
    )
    f = feat_dict(kp)
    assert f["left_ear_angle"] == pytest.approx(0.0, abs=1e-9)
    assert f["right_ear_angle"] == pytest.approx(0.0, abs=1e-9)


def test_asymmetric_ears_produce_nonzero_symmetry():
    kp = make_keypoints(
        left_ear_base_a=(-30, -10), left_ear_tip=(-25, -40), left_ear_base_b=(-20, -10),  # straight up
        right_ear_base_a=(20, -10), right_ear_tip=(35, -40), right_ear_base_b=(30, -10),  # splayed out
    )
    f = feat_dict(kp)
    assert f["ear_symmetry"] > 0.1


def test_face_compactness_scales_with_mouth_distance():
    kp_close = make_keypoints(mouth=(0, 10))
    kp_far = make_keypoints(mouth=(0, 50))
    f_close = feat_dict(kp_close)
    f_far = feat_dict(kp_far)
    assert f_far["face_compactness"] > f_close["face_compactness"]


def test_normalization_is_scale_invariant():
    kp = make_keypoints()
    kp_scaled = kp * 3.0
    f1 = feat_dict(kp)
    f2 = feat_dict(kp_scaled)
    # Angles and normalized-distance features should be unchanged by uniform scaling.
    assert f1["face_compactness"] == pytest.approx(f2["face_compactness"], abs=1e-6)
    assert f1["left_ear_height"] == pytest.approx(f2["left_ear_height"], abs=1e-6)
    assert f1["left_ear_angle"] == pytest.approx(f2["left_ear_angle"], abs=1e-9)


def test_degenerate_eye_distance_does_not_divide_by_zero():
    kp = make_keypoints(left_eye=(0, 0), right_eye=(0, 0))
    out = compute_geometric_features(kp)
    assert np.all(np.isfinite(out))


def test_ear_height_positive_when_tip_above_eye_line():
    kp = make_keypoints()  # ear tips at y=-40, eye line at y=0 -> tip is above (smaller y)
    f = feat_dict(kp)
    assert f["left_ear_height"] > 0
    assert f["right_ear_height"] > 0
