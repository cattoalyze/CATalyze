"""Heatmap-based ear/face keypoint model: MobileNetV2 backbone + a
transposed-convolution decoder head that preserves spatial structure all the
way to the output (no global pooling in this head — see Section 3 of the
brief, which explicitly rules out direct coordinate regression).
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models


class HeatmapKeypointModel(nn.Module):
    def __init__(self, num_keypoints: int = 9, pretrained: bool = True):
        super().__init__()
        backbone = tv_models.mobilenet_v2(
            weights=tv_models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
        # `features` is the conv stack before the classifier; for 224x224
        # input it outputs (N, 1280, 7, 7).
        self.backbone = backbone.features

        # Decoder: 7x7 -> 14x14 -> 28x28 -> 56x56, preserving spatial structure.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 256, kernel_size=4, stride=2, padding=1),  # 7 -> 14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 14 -> 28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 28 -> 56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_keypoints, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        heatmaps = self.decoder(feats)
        return heatmaps

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Global-average-pooled backbone features — reused as CNN embedding
        input for the ensemble in Section 6 if the keypoint backbone is
        shared; the mood CNN branch has its own separate embedding."""
        feats = self.backbone(x)
        return feats.mean(dim=[2, 3])

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_n_layers(self, n: int) -> None:
        """Unfreeze the last n immediate children modules of the backbone
        feature stack (MobileNetV2's `features` is a flat Sequential of
        InvertedResidual/Conv blocks, so this unfreezes the last n blocks)."""
        children = list(self.backbone.children())
        for module in children[-n:]:
            for p in module.parameters():
                p.requires_grad = True
