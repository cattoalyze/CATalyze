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
│   ├── serving/                  # FastAPI /predict, /gradcam, /metrics
│   ├── explainability/            # Grad-CAM
│   └── optimization/              # ONNX export + INT8 quantization + benchmark
├── frontend/                    # static demo page (served by the API)
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

**Second dataset for ANXIOUS-class scarcity:** the Roboflow "Cat Emotions"
candidate from the original session, and a fresh search this session, both
turned up only account-gated sources (Roboflow, Kaggle) — no dataset
downloadable without creating an account on the user's behalf. The user
then supplied a Kaggle API token directly, which unblocked
`nguyenvunhuhuynh/cat-emotion-2` (`data/download_anxious_supplement.py`):
a Kaggle-hosted mirror of a Roboflow export, Public Domain license, whose
"Distressed" class (138 unique source images after de-duplicating
Roboflow's 3x augmentation) was mapped to ANXIOUS — a semantic judgment
call, not an identical label, and community/uploader-assigned rather than
independently verified (tracked under its own
`provenance=external_kaggle_distressed` rather than merged into `ai` or
`pseudo`). Only 47/138 survived this project's own keypoint model at a
0.30 mean-confidence floor — most candidates are close-up or artistic
shots the keypoint model (trained on CAT_DATASET's front-facing photos)
can't reliably localize, so a low-confidence result is expected, not a
bug.

**Result: not adopted.** ANXIOUS grew 226 → 273 (+21%) with the supplement
integrated, but stratified 5-fold CV before vs. after
(`reports/anxious_supplement_experiment.json`) shows pooled ANXIOUS F1
flat-to-down (0.717 → 0.695), driven by a real recall drop (0.739 → 0.652)
partly offset by a precision gain — and on the single held-out split, all
12 external-sourced test examples were misclassified (n=12, not
conclusive alone, but consistent with the k-fold recall drop). Plausible
cause: a real distribution shift between CAT_DATASET's photography
style/labeling process and this external source's, not just noise.

Same pattern this project already uses for the ONNX INT8 finding: measure
honestly, document it, don't ship a measured regression. The deployed
ensemble (`artifacts/ensemble_model.joblib`, `reports/ensemble_metrics.json`,
`reports/ensemble_kfold_metrics.json`) was reverted to the pre-supplement
226-example ANXIOUS class. `data/download_anxious_supplement.py` and the
47 filtered images are kept for reuse if a larger, more carefully vetted
source turns up later — reverting the *code* because this particular
attempt didn't help would be the file-drawer bias this project avoids
elsewhere; not deploying a measured regression is a separate decision.

## Running the pipeline

```bash
uv run python -m data.download                       # ~2.1GB, archive.org mirror
uv run python -m src.keypoints.train                  # trains keypoint model
uv run python -m src.labeling.build_label_grids       # builds review grids for seed labeling
uv run python -m src.labeling.consolidate_seed_labels # (after labeling) -> mood_labels.csv
uv run python -m src.labeling.self_training           # pseudo-labels the rest
uv run python -m src.labeling.active_learning           # ranks the remaining unlabeled pool by predictive entropy for human review
uv run python -m src.features.extract_features        # geometric features for all labeled images
uv run python -m data.download_anxious_supplement       # optional: requires a Kaggle API token; see "Datasets" — not adopted by default, see reports/anxious_supplement_experiment.json
uv run python -m src.ensemble.evaluate                # trains + evaluates the ensemble
uv run python -m src.ensemble.cross_validate            # stratified 5-fold CV, esp. for the scarce ANXIOUS class
uv run python -m src.reports.generate_metrics          # consolidates reports/metrics.json
uv run python -m src.explainability.verify_gradcam     # Grad-CAM verification grid
uv run python -m src.optimization.export_onnx          # ONNX export + INT8 quantization
uv run python -m src.optimization.benchmark            # real size/latency/accuracy comparison
uv run uvicorn src.serving.api:app --port 8000         # serve /predict + demo frontend at /
```

Run tests with `uv run pytest tests/ -v`.

## Serving API

`POST /predict` (multipart form, field `file`) returns the brief's JSON
contract — verified end-to-end against a real image:

```json
{
  "predictions": {"ALERT": 0.0004, "ANXIOUS": 0.002, "PLAYFUL": 0.002, "RELAXED": 0.996},
  "keypoints": [{"x": 177.5, "y": 165.2, "confidence": 0.72}, ...],
  "boxes": {"left_ear": {"x_min": ..., "y_min": ..., "x_max": ..., "y_max": ..., "confidence": 0.75}, "right_ear": {...}, "face": {...}},
  "geometric_features": {"left_ear_angle": -0.59, ...}
}
```

`boxes` are derived algorithmically from the keypoints (min/max rectangle
of each box's constituent keypoints, padded 15%) — no separate detector.
Each box's `confidence` is the real mean of its constituent keypoints'
actual heatmap-peak confidences, never invented. See
`src/keypoints/bbox.py`.

`POST /gradcam` (same multipart contract) returns a base64-encoded PNG
overlay plus the mood CNN's own (uncalibrated) raw view of the image.
`GET /metrics` serves `reports/metrics.json` verbatim.

`GET /health` returns model-load status. Non-image uploads and uploads over
`serving.max_upload_mb` (config.yaml, default 10MB) are rejected with 400 —
both verified with real curl requests, not assumed from the code.

`WS /ws/live` streams live-camera tracking: client sends one JPEG-encoded
frame per binary message, server replies with one JSON prediction per
frame (same shape as `/predict`, plus `cat_detected` and EMA-smoothed
values, or just `{"cat_detected": false, ...}` when no cat is in frame).
Uses the raw (uncalibrated) ensemble for speed — see
`src/serving/live_tracker.py` and `reports/live_backend_benchmark.json`
for why, and measured real-hardware FPS (~12.5-14 FPS on CPU, this
session's dev machine; no GPU could be benchmarked here — see "Deviations
from the brief").

## Deviations from the brief

- **Live-camera path runs on CPU, not GPU, in this session's environment.**
  This dev machine has an NVIDIA GPU, but the ~3GB of CUDA 12.4 runtime
  wheels (`torch==2.6.0+cu124`) couldn't be reliably downloaded over this
  sandbox's network (repeated stalls/timeouts on large files, confirmed by
  direct `curl` tests, not just pip retries) — worked around with a
  CPU-only PyTorch install for this session only.
  `requirements.txt` still pins the CUDA build, correct for a machine that
  can reach it; this was a session-local workaround, not a project change.
  The live-camera backend choice (raw ensemble over calibrated, PyTorch
  over ONNX/INT8) was decided by real CPU measurement — see
  `reports/live_backend_benchmark.json` — and should be re-benchmarked on
  GPU by whoever next runs this with working CUDA access, since the
  raw-vs-calibrated tradeoff in particular could look different once the
  neural-net latency stops dominating.
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

## Descoped: tail detection (Section 8 stretch)

Not attempted. This is the pre-approved ear-only fallback, chosen for two
reasons: (1) the brief itself flags MMPose/MMCV + an isolated NumPy<2
environment as a real, demonstrated dependency-conflict risk not worth
taking on for a stretch feature, and (2) disk/compute budget was ultimately
fine (119.8GB free after the core pipeline), so this was a deliberate scope
call, not a forced one. Ear-only geometric features are used throughout.

## Grad-CAM (`src/explainability/gradcam.py`)

Implemented against `MoodCNN.backbone[-1]` (the last conv block, 1280
channels, the same feature map that gets pooled into the embedding).
Verified visually on real predictions — 2 correct examples per class plus
1 misclassification — in `reports/gradcam_verification.png`. All correct
predictions show the heatmap focused on the cat's face/eyes/ears, not
background clutter. The misclassified example (true=RELAXED, predicted
ALERT at 0.97) is genuinely informative: Grad-CAM shows the model fixating
tightly on the cat's wide, direct-staring eyes — a classic ALERT visual
cue — which plausibly explains the error even though the ground truth
(itself AI-assigned, not human-verified) was RELAXED. Exposed live via
`POST /gradcam` on the serving API, and toggleable in the demo frontend.

## ONNX export + INT8 quantization (`src/optimization/`)

Both the mood CNN and the keypoint model were exported to ONNX and
quantized (`src/optimization/export_onnx.py`), then benchmarked
(`src/optimization/benchmark.py`) with real, measured numbers — latency
averaged over 100 runs with 10 warmup runs, accuracy measured on the exact
same seed-labeled-only test split used for the core pipeline's honest
metrics. Full numbers in `reports/onnx_benchmark.json`.

| | Mood CNN | Keypoint model |
|---|---|---|
| PyTorch CPU latency | 20.8 ms | 42.8 ms |
| ONNX fp32 CPU latency | **4.1 ms** (5.1x faster) | **10.7 ms** (4.0x faster) |
| ONNX INT8 CPU latency | 98.3 ms (4.7x **slower** than PyTorch) | 109.5 ms (2.6x **slower**) |
| fp32 → int8 file size | 9.10 → 2.44 MB (−73.1%) | 32.69 → 26.04 MB (−20.3%) |
| Accuracy, fp32 vs PyTorch | identical (74.19% both) | identical (NME 0.2337 both, 200-sample) |
| Accuracy, int8 | 69.35% (**−4.8 pts**) | NME 0.2474 (**+0.014, worse**) |

**Two honest, unglamorous findings, reported exactly as measured:**

1. **ONNX export alone (no quantization) is a genuine, free win** —
   ~4-5x lower CPU latency with *zero* accuracy change (fp32 ONNX matches
   PyTorch exactly on both models' honest test metrics).
2. **INT8 dynamic quantization made things worse on both axes here** —
   slower *and* less accurate. This was measured, not assumed, and the
   likely reason is architectural: `onnxruntime.quantization.quantize_dynamic`
   is documented as most effective for MatMul/Gemm-heavy models
   (transformers, RNNs); these are Conv-heavy MobileNetV2-based CNNs, where
   the runtime quantize/dequantize overhead around each Conv op outweighs
   any benefit, and weight-only dynamic quantization of Conv layers gives
   up real accuracy without the calibrated activation quantization that
   static quantization would provide. **The quantized models are not used
   anywhere in the serving path** — this section exists to report a real,
   measured (negative) result, not to ship a regression.

A notable environment landmine hit along the way: PyTorch 2.13's default
(dynamo-based) ONNX exporter, combined with `onnxruntime.quantization`'s
internal shape-inference pass, produced a `ShapeInferenceError` on an
otherwise valid, checker-passing, correctly-running graph — fixed by
running onnxruntime's own recommended `quant_pre_process` step before
quantization (documented in `export_onnx.py`).

## Demo frontend (`frontend/index.html`)

A single static HTML/CSS/JS page (no build step, no framework), served by
the FastAPI app itself via a static-files mount — `uv run uvicorn
src.serving.api:app --port 8000` and open `http://localhost:8000/`.
Pastel orange/white palette (`#FFF8F0` background, `#FFCBA4`/`#F5A76A`
accent, `#3A3A3A` text). Upload an image (or click one of 4 bundled sample
images, one per mood class, each verified in advance to produce a
confident, correct prediction from the real pipeline) and click Analyze:

- Calls the real `/predict` endpoint and renders the 4 calibrated
  confidence bars from its actual response — not a single deterministic
  label.
- Draws the 9 returned keypoints as dots on the uploaded image (canvas).
- Calls the real `/gradcam` endpoint and offers a toggle between the
  keypoints view and the Grad-CAM overlay.
- Footer accuracy figure ("Ensemble seed-labeled-only accuracy: 83.3%") is
  fetched live from `GET /metrics`, which serves `reports/metrics.json`
  verbatim — never hardcoded or re-typed.

Verified end-to-end in a real browser session against the real running
API: sample image → real `/predict` + `/gradcam` calls → confidence bars
render correctly (matched the independently-verified prediction for that
image) → keypoints drawn on canvas (confirmed via pixel inspection, not
just "no JS errors") → Grad-CAM toggle switches views correctly → metrics
footer populated from the live API response.

**What was deliberately left out**, per the brief: no dark mode, no stats
dashboard, no training-pipeline panel, no PDF/CSV export, no fabricated or
re-typed numbers anywhere on the page.
