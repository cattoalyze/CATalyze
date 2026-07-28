# CATalyze — Real End-to-End Implementation Brief (for Claude Code)

Build a real, working, modular implementation of CATalyze: a computer vision
system that classifies a cat's mood (ALERT / ANXIOUS / PLAYFUL / RELAXED) from
a photo, with calibrated per-class confidence output. This is a from-scratch
implementation — write real code, actually run it, actually measure results.
Do not simulate results or hardcode plausible-looking metrics anywhere; every
number in the final report must come from code that was actually executed.

---

## 0. Environment setup — read this before installing anything

This project has real, known dependency landmines. Set the environment up
correctly the first time rather than discovering these by trial and error:

- Use `uv` for environment management (`uv venv`, `uv pip install`). Note that
  `uv venv` does **not** install `pip` inside the venv by default — if any
  dependency's build process needs `import pip` (some legacy packages do),
  run `uv pip install pip` first.
- **GPU (if an NVIDIA GPU is present):** installing `tensorflow[and-cuda]` via
  `uv pip` can resolve mismatched CUDA major-version packages (e.g. `cu13`
  packages when TensorFlow was built against CUDA 12.x). Verify with
  `python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"`
  and make sure installed `nvidia-*` package versions match. Setting
  `LD_LIBRARY_PATH` via `os.environ` *after* Python has already started is
  unreliable — if GPU isn't detected, preload the CUDA `.so` files directly
  via `ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)` in dependency order
  (`cuda_runtime` → `cublas` → `cudnn` → ...) before importing TensorFlow, as
  the very first thing that runs.
- **If implementing any MMPose/MMCV-based component:** this stack requires
  PyTorch, and `mmcv` only has prebuilt wheels for a limited range of PyTorch
  versions — check `https://download.openmmlab.com/mmcv/dist/cpu/` for
  supported versions before picking a torch version, don't just install
  latest. Additionally, `xtcocotools` (an mmpose dependency) has historically
  shipped NumPy-1.x-compiled wheels — **pin `numpy<2` before installing
  anything in this stack**, and do it in a **separate virtual environment**
  from any NumPy-2.x-requiring framework (e.g. modern TensorFlow) since they
  cannot coexist in one environment.
- Always verify a fix worked with a fresh process/kernel restart before
  concluding — environment variables and import state set after a library has
  already failed to load once won't retroactively fix that session.

---

## 1. Repository Structure

```
catalyze/
├── config/
│   └── config.yaml              # all hyperparameters, paths, thresholds — no magic numbers in code
├── data/
│   ├── download.py               # dataset acquisition (see Section 2)
│   └── raw/, processed/          # gitignored
├── src/
│   ├── keypoints/
│   │   ├── model.py              # heatmap-based ear keypoint model (Section 3)
│   │   ├── train.py
│   │   └── infer.py
│   ├── features/
│   │   └── geometric.py          # ear feature engineering (Section 4)
│   ├── mood_cnn/
│   │   ├── model.py              # MobileNetV2 classification branch (Section 5)
│   │   └── train.py
│   ├── labeling/
│   │   ├── seed_tool.py          # manual labeling UI (reuse/adapt from prior session if available)
│   │   └── self_training.py      # pseudo-labeling with per-class confidence thresholds
│   ├── ensemble/
│   │   ├── train.py              # RF + CalibratedClassifierCV (Section 6)
│   │   └── evaluate.py           # honest metrics, human-only vs full-set reporting
│   ├── serving/
│   │   └── api.py                # FastAPI inference endpoint (Section 7)
│   └── explainability/
│       └── gradcam.py            # Grad-CAM for the CNN branch (Section 8, stretch)
├── tests/
│   └── test_*.py                 # unit tests for feature engineering, at minimum
├── artifacts/                    # trained model files, gitignored
├── reports/
│   └── metrics.json              # auto-generated from real evaluation runs
└── README.md
```

---

## 2. Datasets — use more than one

- **Primary:** Crawford Cat Dataset (~9,997 images, 9-point facial keypoints:
  eyes, mouth, 3 pts/ear). Verify current availability before hardcoding a
  download URL — check archive.org mirror and Kaggle (`crawford/cat-dataset`)
  and use whichever is currently reliable.
- **Mood labels do not exist in this dataset** — implement a seed-labeling
  workflow (a few hundred manually labeled images across all 4 classes) plus
  self-training/pseudo-labeling on the remainder, tagging every row with
  provenance (`human`/`pseudo`) so metrics can always be reported honestly on
  the human-labeled subset separately.
- **Actively search for a second real dataset** to address ANXIOUS-class
  scarcity (expect fewer than ~10 clear examples in a random sample of
  Crawford) — look for cat behavior/welfare or feline stress-indicator
  datasets. If genuinely none exists, say so in the final report rather than
  relying solely on synthetic oversampling and calling the class "handled."
- **Optional (only if time/environment permits — see Section 0's warning):**
  a tail-keypoint dataset (e.g. Animal-Pose Dataset, cat subset) for a tail
  posture feature, via a pretrained MMPose model run in an isolated
  environment as an **offline, training-time-only enrichment step** (not live
  inference — document this constraint clearly if implemented). Ear-only is a
  fully acceptable, honest fallback if this proves too costly.

---

## 3. Ear Keypoint Detection — heatmap-based regression

Do not use direct coordinate regression (predicting raw x,y via MSE on a dense
output) — it has a real, demonstrated precision ceiling. Implement:

- MobileNetV2 backbone (frozen → fine-tune last ~30 layers).
- Decoder head outputting 9 heatmaps (one per keypoint) via upsampling/
  transposed convolutions — preserve spatial structure through to the output,
  do not use global pooling in this head.
- Ground truth: 2D Gaussian (sigma ≈ 1-2px at target resolution) centered at
  each true keypoint.
- Inference: keypoint = heatmap argmax; **confidence = heatmap peak value**
  (a genuine, meaningful per-keypoint confidence score).
- Report real validation precision (e.g. normalized mean error) after
  training — do not assume improvement without measuring it.

---

## 4. Geometric Feature Engineering

From the 9 keypoints, compute: left/right ear angle, ear spread, ear symmetry,
left/right ear height (normalized by head width via eye-to-eye distance), and
face compactness (mouth-to-eye-midpoint distance, normalized). Add a tail
feature only if Section 2's optional tail component is implemented.

---

## 5. Mood CNN Branch

MobileNetV2, frozen then fine-tuned, trained directly on raw images against
the 4 mood labels (human + pseudo). This branch's `GlobalAveragePooling2D`
output also serves as an embedding feature for the ensemble in Section 6.

---

## 6. Ensemble Classifier

- Concatenate: 7 (or 8, with tail) geometric features + CNN embedding
  features.
- Random Forest classifier, wrapped in `CalibratedClassifierCV`
  (`method='sigmoid'`) for honest probability calibration.
- SMOTE on the geometric feature vectors for underrepresented classes
  (guard `k_neighbors` against very small minority class sizes).
- **Output: calibrated confidence for all 4 classes**, not a single label.
- Report: raw vs. calibrated accuracy, per-class precision/recall/F1,
  confusion matrix, and a reliability diagram (predicted confidence vs. actual
  accuracy) with Brier score, comparing raw vs. calibrated.

---

## 7. Serving Layer

A FastAPI endpoint (`/predict`) that accepts an image and returns:
```json
{
  "predictions": {"ALERT": 0.62, "ANXIOUS": 0.03, "PLAYFUL": 0.10, "RELAXED": 0.25},
  "keypoints": [{"x": 120, "y": 84, "confidence": 0.91}, ...],
  "geometric_features": {...}
}
```
This is the contract an external frontend (e.g. a separately-built dashboard)
can consume. Include a minimal health-check endpoint and basic input
validation (reject non-image uploads, oversized files).

---

## 8. Legitimate Stretch Additions (implement if time permits, real not simulated)

- **Grad-CAM** visualization for the CNN branch — shows which image regions
  drove the mood prediction; genuinely useful for both interpretability and
  as a portfolio talking point.
- **Real ONNX export + quantization** of the CNN and/or keypoint model, with
  an actual measured before/after size and latency comparison. (This turns a
  previously-fabricated claim in an earlier UI mockup into a real, honestly
  measured feature — only include it in any report if actually done.)
- A small test suite covering feature engineering functions (deterministic,
  easy to unit test) and the confidence-calibration pipeline.
- A `reports/metrics.json` auto-generated by the evaluation script, which any
  frontend or README can read from directly — avoids ever needing to
  hand-transcribe (and potentially misstate) a result.

---

## 9. Deliverables

- Working, tested, modular codebase per the structure above.
- `reports/metrics.json` with real measured results.
- README documenting: datasets used and why, architecture, real metrics,
  and explicit known limitations (expect ANXIOUS recall to remain the weakest
  class even after mitigation — report the real number).
