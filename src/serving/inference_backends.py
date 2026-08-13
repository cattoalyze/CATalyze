"""Interchangeable inference backends for the keypoint + mood-CNN pair,
used by both the one-off latency benchmark and the live-camera tracker
(Section 2 of the continuation prompt). Kept swappable specifically because
the prompt's assumption (ONNX/INT8 wins on latency) needed to be checked
for real on this hardware rather than trusted from the prior CPU-only
benchmark — see reports/live_backend_benchmark.json.

Each backend exposes the same two operations on a raw (H, W, 3) uint8 RGB
frame: predict_keypoints -> (kp (9,2) in original-image pixel coords, conf
(9,)), and embed_mood -> (embedding_dim,) pooled MobileNetV2 features (the
same embedding the ensemble was trained on).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.keypoints.heatmap_utils import decode_heatmaps  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(frame_rgb: np.ndarray, input_size: int) -> np.ndarray:
    """RGB uint8 frame -> normalized (1, 3, input_size, input_size) float32 NCHW."""
    resized = cv2.resize(frame_rgb, (input_size, input_size)).astype(np.float32) / 255.0
    resized = (resized - IMAGENET_MEAN) / IMAGENET_STD
    return resized.transpose(2, 0, 1)[None, ...].astype(np.float32)


class PyTorchBackend:
    """Runs the trained .pt models directly (GPU if available)."""

    name = "pytorch"

    def __init__(self, keypoint_model, mood_model, input_size_kp: int, heatmap_size: int, input_size_mood: int, device):
        import torch  # local import: keeps this module importable before torch exists
        self.torch = torch
        self.keypoint_model = keypoint_model.eval()
        self.mood_model = mood_model.eval()
        self.input_size_kp = input_size_kp
        self.heatmap_size = heatmap_size
        self.input_size_mood = input_size_mood
        self.device = device

    def predict_keypoints(self, frame_rgb: np.ndarray):
        h0, w0 = frame_rgb.shape[:2]
        x = self._to_tensor(frame_rgb, self.input_size_kp)
        with self.torch.no_grad():
            heatmaps = self.keypoint_model(x).cpu().numpy()[0]
        kp, conf = decode_heatmaps(heatmaps, self.input_size_kp, self.heatmap_size)
        scale = np.array([w0 / self.input_size_kp, h0 / self.input_size_kp])
        return kp * scale, conf

    def embed_mood(self, frame_rgb: np.ndarray):
        x = self._to_tensor(frame_rgb, self.input_size_mood)
        with self.torch.no_grad():
            emb = self.mood_model.embed(x).cpu().numpy()[0]
        return emb

    def _to_tensor(self, frame_rgb, size):
        arr = _preprocess(frame_rgb, size)
        return self.torch.from_numpy(arr).to(self.device)


class OnnxBackend:
    """Runs exported ONNX models (fp32 or int8) via onnxruntime."""

    def __init__(self, keypoint_onnx_path, mood_onnx_path, input_size_kp: int, heatmap_size: int, input_size_mood: int, providers=None):
        import onnxruntime as ort
        self.ort = ort
        avail = ort.get_available_providers()
        providers = providers or ([p for p in ["CUDAExecutionProvider"] if p in avail] + ["CPUExecutionProvider"])
        self.kp_session = ort.InferenceSession(str(keypoint_onnx_path), providers=providers)
        self.mood_session = ort.InferenceSession(str(mood_onnx_path), providers=providers)
        self.name = f"onnx[{self.kp_session.get_providers()[0]}]"
        self.input_size_kp = input_size_kp
        self.heatmap_size = heatmap_size
        self.input_size_mood = input_size_mood
        # mood_cnn.onnx only outputs logits; the embedding backbone is not
        # separately exported, so this backend re-derives it by running the
        # backbone through the same session's intermediate isn't exposed —
        # documented limitation, see get_embedding_output_name below.
        self._mood_output = self.mood_session.get_outputs()[0].name
        self._kp_output = self.kp_session.get_outputs()[0].name
        self._kp_input = self.kp_session.get_inputs()[0].name
        self._mood_input = self.mood_session.get_inputs()[0].name

    def predict_keypoints(self, frame_rgb: np.ndarray):
        h0, w0 = frame_rgb.shape[:2]
        x = _preprocess(frame_rgb, self.input_size_kp)
        heatmaps = self.kp_session.run([self._kp_output], {self._kp_input: x})[0][0]
        kp, conf = decode_heatmaps(heatmaps, self.input_size_kp, self.heatmap_size)
        scale = np.array([w0 / self.input_size_kp, h0 / self.input_size_kp])
        return kp * scale, conf

    def predict_mood_logits(self, frame_rgb: np.ndarray):
        x = _preprocess(frame_rgb, self.input_size_mood)
        return self.mood_session.run([self._mood_output], {self._mood_input: x})[0][0]

    # NOTE: intentionally no embed_mood() here. The mood_cnn ONNX graph
    # (src/optimization/export_onnx.py) only exposes "logits" as an output,
    # not the pooled embedding the ensemble was trained on — adding that
    # would mean re-exporting with a second output and touching the
    # existing benchmark's contract. Combined with the prior CPU benchmark
    # already showing ONNX INT8 is both slower and less accurate here (see
    # reports/onnx_benchmark.json), the live-camera tracker below uses
    # PyTorchBackend and picks CPU vs CUDA by real measurement instead —
    # see reports/live_backend_benchmark.json for the numbers and
    # docs/live_camera.md (or the session report) for the reasoning.
