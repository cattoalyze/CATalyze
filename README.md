# CATalyze

A computer vision system that classifies a cat's mood (ALERT / ANXIOUS / PLAYFUL / RELAXED)
from a photo, with calibrated per-class confidence output.

Pipeline: MobileNetV2 heatmap-based ear/face keypoint detector → geometric
feature engineering → MobileNetV2 mood classifier (whose pooled features
double as an embedding) → Random Forest ensemble over geometric features +
CNN embedding, calibrated with `CalibratedClassifierCV`.

## Environment

- Python 3.11, managed via `uv`
- PyTorch (not TensorFlow — native Windows has no TF GPU support since 2.11;
  `tensorflow[and-cuda]` is Linux-only. See "Deviations from the brief" below.)

```bash
uv venv .venv --python 3.11
uv pip install -r requirements.txt
```

## Repository structure

```
catalyze/
├── config/config.yaml          # all hyperparameters, paths, thresholds
├── data/download.py            # CAT_DATASET acquisition (archive.org mirror)
├── src/
│   ├── keypoints/               # heatmap-based ear/face keypoint model
│   ├── features/                # geometric feature engineering
│   ├── mood_cnn/                 # MobileNetV2 mood classifier
│   ├── labeling/                 # seed labeling UI + self-training
│   ├── ensemble/                 # RF + CalibratedClassifierCV
│   └── serving/                  # FastAPI /predict endpoint
├── tests/
├── reports/                     # auto-generated metrics + visualizations
└── requirements.txt
```

## Datasets

**Primary: CAT_DATASET** (Zhang, Sun & Tang; a.k.a. "Crawford Cat Dataset" on
Kaggle), 9,992 images with 9-point facial keypoint annotations, downloaded
from the archive.org mirror (verified live 2026-07-28). ~9.5% of annotations
(948 images) were found to be corrupted/out-of-bounds on visual inspection
and dropped — see `src/keypoints/dataset.py`.

**Mood labels do not exist in this dataset.** A seed set of 480 images was
labeled across the 4 mood classes, then self-training/pseudo-labeling was
attempted on the remaining pool.

**Second dataset for ANXIOUS-class scarcity:** a suitable candidate was found
(a 671-image, CC-BY-4.0 "Cat Emotions" set on Roboflow with an explicit
"Scared" class), but downloading it requires a Roboflow API key, which
would have meant creating an account on the user's behalf — out of scope for
an autonomous session. Not used; documented here as a known gap rather than
silently worked around.

## Running the pipeline

```bash
uv run python -m data.download                       # ~2.1GB, archive.org mirror
uv run python -m src.keypoints.train                  # trains keypoint model
uv run python -m src.labeling.build_label_grids       # builds review grids for seed labeling
uv run python -m src.labeling.consolidate_seed_labels # (after labeling) -> mood_labels.csv
uv run python -m src.labeling.self_training           # pseudo-labels the rest
uv run python -m src.features.extract_features        # geometric features for all labeled images
uv run python -m src.ensemble.evaluate                # trains + evaluates the ensemble
uv run python -m src.reports.generate_metrics          # consolidates reports/metrics.json
uv run uvicorn src.serving.api:app --port 8000         # serve /predict
```

Run tests with `uv run pytest tests/ -v`.

## Serving API

`POST /predict` (multipart form, field `file`) returns the brief's JSON
contract — verified end-to-end against a real image:

```json
{
  "predictions": {"ALERT": 0.0004, "ANXIOUS": 0.002, "PLAYFUL": 0.002, "RELAXED": 0.996},
  "keypoints": [{"x": 177.5, "y": 165.2, "confidence": 0.72}, ...],
  "geometric_features": {"left_ear_angle": -0.59, ...}
}
```

`GET /health` returns model-load status. Non-image uploads and uploads over
`serving.max_upload_mb` (config.yaml, default 10MB) are rejected with 400 —
both verified with real curl requests, not assumed from the code.

## Deviations from the brief

- **PyTorch instead of TensorFlow/Keras.** Architecture is unchanged
  (MobileNetV2 backbone, heatmap decoder, `GlobalAveragePooling2D`-equivalent
  embedding via `nn.AdaptiveAvgPool2d`), only the framework differs, because
  native-Windows TensorFlow has been CPU-only since v2.11.
- **Seed mood labels are `provenance='ai'`, not `'human'`.** No human rater
  was available in this autonomous session. Labels were assigned by Claude
  visually reviewing batches of real images (see
  `src/labeling/build_label_grids.py` / `consolidate_seed_labels.py`) — a
  real judgment call on real images, but a weaker ground truth than true
  human annotation. `src/labeling/seed_tool.py` is a real, working manual
  labeling web UI for a human rater to use instead, if higher-quality seed
  labels are wanted later.
- **Self-training confidence thresholds were lowered from the brief's
  initial placeholders** (0.75-0.85 → 0.50-0.60) after measuring that a mood
  CNN trained on only 336 seed images rarely exceeds ~0.6 max-softmax
  confidence on unseen images. The original thresholds yielded zero
  pseudo-labels; this was verified empirically before adjusting, not assumed.

## Real measured results

See `reports/metrics.json` (auto-generated, never hand-transcribed) for the
authoritative numbers. Summary as of this run:

| Stage | Metric | Value |
|---|---|---|
| Keypoint detector | test NME (÷ inter-ocular distance), n=1,357 | **0.230** |
| Mood CNN | test accuracy, full test set (n=1,457, 95% pseudo-labeled) | 0.896 |
| Mood CNN | **test accuracy, seed-labeled-only (n=62)** | **0.742** |
| Ensemble (calibrated) | test accuracy, full test set (n=1,942, 95% pseudo-labeled) | 0.918 |
| Ensemble (calibrated) | **test accuracy, seed-labeled-only (n=78)** | **0.833** |
| Ensemble | Brier score, raw → calibrated | 0.256 → **0.124** |
| Ensemble, ANXIOUS class | precision / recall / F1 | 0.72 / 0.69 / 0.70 |
| Ensemble, PLAYFUL class | precision / recall / F1 | 0.82 / 0.80 / 0.81 |

**Read the bolded "seed-labeled-only" rows as the trustworthy numbers.** The
full-test-set numbers (0.896, 0.918) are measured on a test split that is
~95% pseudo-labeled — see "Self-training exhibited majority-class
confirmation bias" below for why those numbers are inflated and should not
be read as genuine generalization accuracy.

Calibration is verified working, not just claimed: Brier score dropped 52%
(0.256 → 0.124) and the reliability diagram (`reports/reliability_diagram.png`)
shows the raw model was significantly underconfident while the calibrated
curve tracks the ideal diagonal closely.

### Known limitations

- **Self-training exhibited majority-class confirmation bias.** Round 0
  (trained on 480 seed images, 54% test accuracy) pseudo-labeled 794 images,
  all ALERT/RELAXED. Round 1 (1,274 images, 85% test accuracy) then
  pseudo-labeled 7,515 of the remaining 8,718 pool images (86%!) — again
  overwhelmingly ALERT/RELAXED (4,177 + 3,011 vs. only 148 ANXIOUS + 179
  PLAYFUL). This is a well-known self-training failure mode: pseudo-labels
  are, by construction, the examples the model already finds easy, so each
  round's "improved" accuracy partly measures self-consistency with the
  model's own prior predictions rather than genuine generalization. The
  original confidence thresholds (0.75-0.85) were lowered to 0.50-0.60 after
  measuring the seed-only model rarely exceeded ~0.6 confidence — a
  reasonable adjustment on its own, but one that made this amplification
  effect worse, not better. **This is why the seed-labeled-only accuracy is
  reported and featured as the primary metric** rather than the inflated
  full-test-set numbers.
- **ANXIOUS is the weakest class**, as the brief anticipated — it is the
  scarcest in casual pet photos (45 seed / 226 total after self-training,
  vs. 210 seed / 5,395 total for ALERT) and the self-training amplification
  above means it barely grew via pseudo-labeling.
- Keypoint NME (0.23) is workable but not state-of-the-art; a larger
  backbone, more epochs, or letterboxed (non-distorting) resizing would
  likely improve it further. 9.5% of raw annotations were corrupted/dropped.
- **Seed mood labels (`provenance='ai'`) are Claude's visual judgment, not
  human-verified ground truth.** No human rater was available in this
  autonomous session. Even the "seed-labeled-only" honest numbers above
  should be read with this caveat — they are honest relative to the model's
  own training data, not validated against independent human judgment.
- The ensemble's geometric+CNN-embedding combination measurably outperforms
  the CNN alone on the honest metric (0.833 vs 0.742 seed-labeled-only
  accuracy), suggesting the geometric features do carry real signal beyond
  what the CNN already learns from raw pixels — a genuine architectural
  validation, not assumed.
