"""Mood classification CNN: MobileNetV2 backbone (frozen then fine-tuned),
trained directly on raw images against the 4 mood labels. The pooled
backbone features (`nn.AdaptiveAvgPool2d(1)` output — the PyTorch analogue
of Keras' GlobalAveragePooling2D) double as an embedding for the ensemble
in Section 6.
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models


class MoodCNN(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 1280, pretrained: bool = True):
        super().__init__()
        backbone = tv_models.mobilenet_v2(
            weights=tv_models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
        self.backbone = backbone.features  # (N, 1280, 7, 7) for 224 input
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.pool(feats).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        return self.classifier(emb)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_n_layers(self, n: int) -> None:
        children = list(self.backbone.children())
        for module in children[-n:]:
            for p in module.parameters():
                p.requires_grad = True
