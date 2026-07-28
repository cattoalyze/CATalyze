"""Grad-CAM for the mood CNN branch (Section 8 stretch).

Hooks MoodCNN.backbone[-1] — the last conv block (1280-channel, post
BN+ReLU6, the same feature map that gets global-average-pooled into the
embedding) — captures its activations and the gradient of the target
class's logit w.r.t. those activations, and combines them into a
class-discriminative localization map per the standard Grad-CAM formula
(Selvaraju et al. 2017): weights = GAP(gradients) per channel,
CAM = ReLU(sum_c weight_c * activation_c).
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else model.backbone[-1]
        self.activations = None
        self.gradients = None
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = None):
        """input_tensor: (1, 3, H, W). Returns (cam, class_idx, logits) where
        cam is a (H, W) float array in [0, 1] at the input resolution."""
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        score = logits[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # (1, 1, h, w)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam_max = cam.max()
        if cam_max > 1e-8:
            cam = cam / cam_max
        return cam, class_idx, logits.detach().cpu().numpy()[0]


def overlay_cam_on_image(image_rgb_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """image_rgb_uint8: (H, W, 3) uint8. cam: (H, W) float in [0, 1] at the
    same resolution. Returns an (H, W, 3) uint8 overlay."""
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (image_rgb_uint8.astype(np.float32) * (1 - alpha) + heatmap.astype(np.float32) * alpha)
    return np.clip(overlay, 0, 255).astype(np.uint8)
