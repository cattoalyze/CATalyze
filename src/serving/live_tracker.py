"""Live-camera tracking (Section 2 of the continuation prompt): the same
keypoint -> geometric-features + mood-CNN-embedding -> calibrated-ensemble
pipeline as /predict, run per frame with EMA smoothing across frames.

Single-instance architecture note: the keypoint model predicts one 9-point
set per frame — it was never trained to separate multiple cats (no
detector, no NMS). If more than one cat is in frame, whatever region the
model's learned attention locks onto (in practice, the largest/most
prominent face, since CAT_DATASET's training images are single-cat photos)
is what gets tracked. This is a real architectural limit inherited from
Section 3, not something faked here with an ad-hoc multi-instance detector
(the continuation prompt's optional dedicated-detector stretch was
declined — see the session report).

"No cat in frame" is detected via the mean keypoint confidence dropping
below NO_CAT_MEAN_CONFIDENCE_THRESHOLD — the same real heatmap-peak
confidence /predict already returns, just thresholded and averaged rather
than a separately invented "is there a cat" signal.
"""
import numpy as np

NO_CAT_MEAN_CONFIDENCE_THRESHOLD = 0.15


class EMASmoother:
    """Exponential moving average over per-frame keypoints and mood
    probabilities, to reduce frame-to-frame jitter in live tracking."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self._kp = None
        self._probs = None

    def smooth_keypoints(self, kp: np.ndarray) -> np.ndarray:
        kp = np.asarray(kp, dtype=np.float64)
        self._kp = kp if self._kp is None else self.alpha * kp + (1 - self.alpha) * self._kp
        return self._kp

    def smooth_probs(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float64)
        self._probs = probs if self._probs is None else self.alpha * probs + (1 - self.alpha) * self._probs
        return self._probs / self._probs.sum()

    def reset(self):
        self._kp = None
        self._probs = None


class LiveTracker:
    def __init__(self, backend, ensemble, ensemble_classes, ema_alpha: float = 0.4,
                 no_cat_threshold: float = NO_CAT_MEAN_CONFIDENCE_THRESHOLD):
        self.backend = backend
        self.ensemble = ensemble
        self.ensemble_classes = ensemble_classes
        self.smoother = EMASmoother(alpha=ema_alpha)
        self.no_cat_threshold = no_cat_threshold

    def process_frame(self, frame_rgb: np.ndarray) -> dict:
        from src.features.geometric import compute_geometric_features
        from src.keypoints.bbox import compute_boxes

        kp, kp_conf = self.backend.predict_keypoints(frame_rgb)
        mean_conf = float(kp_conf.mean())

        if mean_conf < self.no_cat_threshold:
            self.smoother.reset()
            return {"cat_detected": False, "mean_keypoint_confidence": mean_conf}

        kp_smoothed = self.smoother.smooth_keypoints(kp)
        geo_features = compute_geometric_features(kp_smoothed)
        boxes = compute_boxes(kp_smoothed, kp_conf, image_shape=frame_rgb.shape)
        embedding = self.backend.embed_mood(frame_rgb)

        feature_vec = np.concatenate([geo_features, embedding]).reshape(1, -1)
        probs = self.ensemble.predict_proba(feature_vec)[0]
        probs_smoothed = self.smoother.smooth_probs(probs)

        return {
            "cat_detected": True,
            "mean_keypoint_confidence": mean_conf,
            "predictions": {cls: float(p) for cls, p in zip(self.ensemble_classes, probs_smoothed)},
            "keypoints": [
                {"x": float(x), "y": float(y), "confidence": float(c)}
                for (x, y), c in zip(kp_smoothed, kp_conf)
            ],
            "boxes": boxes,
        }
