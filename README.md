# CATalyze

A computer vision system that classifies a cat's mood (ALERT / ANXIOUS / PLAYFUL / RELAXED)
from a photo, with calibrated per-class confidence output.

> Status: under active development. This section will be replaced with real
> architecture notes, dataset provenance, and measured metrics as each stage
> lands — see `reports/metrics.json` for auto-generated numbers once available.

## Environment

- Python 3.11 (managed via `uv`)
- PyTorch (not TensorFlow — native Windows has no TF GPU support since 2.11;
  see "Environment notes" below)

```bash
uv venv .venv --python 3.11
uv pip install -r requirements.txt
```

## Environment notes (deviations from the original plan)

- **Framework: PyTorch, not TensorFlow.** TensorFlow dropped native-Windows GPU
  support after v2.10, and `tensorflow[and-cuda]` is a Linux-only pip extra.
  PyTorch's Windows CUDA wheels work natively. All CNN branches (keypoint
  heatmap decoder, mood classifier) are implemented in PyTorch/torchvision;
  the "GlobalAveragePooling2D output as embedding" idea from the original
  brief maps directly to `nn.AdaptiveAvgPool2d(1)` on MobileNetV2's features.

## Repository structure

See `CATalyze_ClaudeCode_Implementation_Brief.md` for the full spec this
implementation follows.
