# Running CATalyze end-to-end

This is the exact command sequence to reproduce the project from a fresh
checkout, in order. Run everything from the repo root. Times are what this
was actually measured taking on an RTX 3050 (4GB VRAM) laptop GPU — CPU-only
will be much slower for the training steps.

Full background/rationale is in [README.md](README.md); this file is just
the runnable sequence.

## 0. Environment setup

Requires Python 3.11 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe pip   # uv venv doesn't install pip by default
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

Every command below is written as `uv run python -m ...`, which works
without a `pyproject.toml` (`uv` just uses the local `.venv` it finds) —
verified in this repo. If `uv` isn't resolving correctly on your machine,
every command has an equivalent direct form:
`.venv/Scripts/python.exe -m <module>` (Windows) or
`.venv/bin/python -m <module>` (macOS/Linux).

**If you have an NVIDIA GPU**, verify PyTorch actually sees it before
continuing — training will silently fall back to (much slower) CPU
otherwise:

```bash
.venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available())"
```

If that prints `False` on a machine with an NVIDIA GPU, you likely need
CUDA-specific wheels — see the "Environment notes" section in README.md
(native Windows needs PyTorch, not TensorFlow, for GPU support).

## 1. Get the data

```bash
uv run python -m data.download
```

Downloads ~2.1GB from the archive.org CAT_DATASET mirror and consolidates
~9,992 image/keypoint-annotation pairs into `data/raw/crawford/`. Takes a
few minutes depending on connection speed.

## 2. Train the keypoint detector

```bash
uv run python -m src.keypoints.train
```

~10–20 minutes on GPU. Saves `artifacts/keypoint_model.pt` and
`reports/keypoint_predictions_sample.png` — **open that image and check the
predicted keypoints actually land on eyes/ears/mouth** before trusting the
printed NME number blindly.

## 3. Seed mood labels

The 480 seed labels (tagged `provenance='ai'` — assigned by visual review,
not a human rater; see README.md for why) are already included at
`data/processed/mood_labels_seed.csv`. Copy it into place as the pipeline's
starting label file:

```bash
cp data/processed/mood_labels_seed.csv data/processed/mood_labels.csv
```

**If you'd rather provide your own human-labeled seed set instead**, run
the manual labeling web UI and label a few hundred images across the 4
classes yourself before continuing:

```bash
uv run python -m src.labeling.seed_tool   # open http://localhost:8001
```

## 4. Self-training (pseudo-labeling)

```bash
uv run python -m src.labeling.self_training
```

~15–25 minutes on GPU (3 rounds: trains a mood CNN, pseudo-labels
high-confidence pool images, repeats). Overwrites
`data/processed/mood_labels.csv` with the seed + pseudo-labeled set, and
`artifacts/mood_cnn_model.pt` with the final round's model.

**Known behavior, not a bug**: later rounds tend to pseudo-label mostly the
majority classes (ALERT/RELAXED). See README.md's "Self-training exhibited
majority-class confirmation bias" section — this is why the honest
evaluation later reports seed-labeled-only accuracy separately from the
full (pseudo-inflated) test-set accuracy.

## 5. Re-evaluate the mood CNN with full reporting

Self-training's internal training calls skip visualization/report
persistence for speed; run this once more on the final label set to get
`reports/mood_cnn_metrics.json` and a prediction-sample visualization:

```bash
uv run python -m src.mood_cnn.train
```

~10–15 minutes on GPU. Check `reports/mood_predictions_sample.png` visually
before trusting the accuracy number.

## 6. Geometric feature extraction

```bash
uv run python -m src.features.extract_features
```

~3–5 minutes. Runs the trained keypoint model over every labeled image and
computes the 7 geometric features, writing
`data/processed/geometric_features.csv`.

## 7. Train + evaluate the ensemble

```bash
uv run python -m src.ensemble.evaluate
```

~5–10 minutes. Trains the RF + `CalibratedClassifierCV` ensemble, saves
`artifacts/ensemble_model.joblib`, prints raw vs. calibrated accuracy/Brier
score, and writes `reports/ensemble_metrics.json` +
`reports/reliability_diagram.png`.

## 8. Consolidate real metrics

```bash
uv run python -m src.reports.generate_metrics
```

Merges the keypoint/mood-CNN/ensemble metrics JSONs into the single
`reports/metrics.json` — the source of truth the README, frontend, and
`/metrics` endpoint all read from.

## 9. (Optional) Grad-CAM verification

```bash
uv run python -m src.explainability.verify_gradcam
```

Saves `reports/gradcam_verification.png` — visually confirm the highlighted
regions land on the cat's face/ears, not background, before trusting it.

## 10. (Optional) ONNX export + quantization benchmark

```bash
uv run python -m src.optimization.export_onnx
uv run python -m src.optimization.benchmark
```

Exports both models to ONNX, applies INT8 quantization, and measures real
latency/size/accuracy differences into `reports/onnx_benchmark.json`. Takes
a few minutes (the benchmark step re-runs inference across a real test
sample for each model variant). Note: the quantized models are **not** used
in serving — see README.md for why (quantization measurably hurt both
latency and accuracy here).

## 11. Run the serving API + demo frontend

```bash
uv run uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/` for the demo page, or call the API directly:

```bash
curl -X POST http://localhost:8000/predict -F "file=@path/to/a/cat/photo.jpg"
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

## 12. Tests

Can be run any time after step 0 (the unit tests use synthetic data, not
trained models):

```bash
uv run pytest tests/ -v
```

## Total time estimate

Steps 2, 4, 5, 6, 7 (the real training/evaluation work) took roughly
1–1.5 hours combined on the reference GPU. Step 1 (download) and step 10
(ONNX benchmark) each add a few more minutes. Step 3 is instant if you use
the included seed labels.
