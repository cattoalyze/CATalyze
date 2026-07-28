"""Real, measured before/after comparison for the ONNX export + INT8
quantization in export_onnx.py: file size, inference latency (averaged over
many runs, not a single anecdotal measurement), and accuracy impact.

Accuracy impact is measured on the *exact same* seed-labeled-only test split
used to report the honest numbers in reports/mood_cnn_metrics.json and
reports/keypoint_metrics.json — same random_state, same stratification —
so the comparison is apples-to-apples with the core pipeline's own honest
evaluation, not a fresh/different sample.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import load_config, resolve_path  # noqa: E402
from src.keypoints.dataset import list_samples, parse_cat_file  # noqa: E402
from src.keypoints.heatmap_utils import decode_heatmaps  # noqa: E402
from src.keypoints.model import HeatmapKeypointModel  # noqa: E402
from src.mood_cnn.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.mood_cnn.model import MoodCNN  # noqa: E402

NORMALIZE = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def load_mood_tensor(path: str, input_size: int) -> np.ndarray:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_size, input_size))
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    tensor = NORMALIZE(tensor)
    return tensor.unsqueeze(0).numpy()


def measure_latency(run_fn, x: np.ndarray, n_warmup: int = 10, n_runs: int = 100) -> dict:
    for _ in range(n_warmup):
        run_fn(x)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        run_fn(x)
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return {"mean_ms": float(times.mean()), "std_ms": float(times.std()), "n_runs": n_runs}


def benchmark_mood_cnn(cfg: dict) -> dict:
    mcfg = cfg["mood_cnn"]
    class_names = cfg["mood_classes"]
    onnx_dir = resolve_path(cfg["paths"]["artifacts"]) / "onnx"

    pt_path = resolve_path(cfg["paths"]["mood_cnn_model"])
    fp32_path = onnx_dir / "mood_cnn_fp32.onnx"
    int8_path = onnx_dir / "mood_cnn_int8.onnx"

    device = torch.device("cpu")
    pt_model = MoodCNN(num_classes=len(class_names), embedding_dim=mcfg["embedding_dim"], pretrained=False).to(device)
    pt_model.load_state_dict(torch.load(pt_path, map_location=device))
    pt_model.eval()

    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    input_name = sess_fp32.get_inputs()[0].name

    dummy = np.random.randn(1, 3, mcfg["input_size"], mcfg["input_size"]).astype(np.float32)

    @torch.no_grad()
    def run_pt(x):
        return pt_model(torch.from_numpy(x)).numpy()

    def run_fp32(x):
        return sess_fp32.run(None, {input_name: x})[0]

    def run_int8(x):
        return sess_int8.run(None, {input_name: x})[0]

    latency = {
        "pytorch_cpu": measure_latency(run_pt, dummy),
        "onnx_fp32_cpu": measure_latency(run_fp32, dummy),
        "onnx_int8_cpu": measure_latency(run_int8, dummy),
    }

    size = {
        "pytorch_state_dict_mb": pt_path.stat().st_size / 1e6,
        "onnx_fp32_mb": fp32_path.stat().st_size / 1e6,
        "onnx_int8_mb": int8_path.stat().st_size / 1e6,
    }
    size["fp32_to_int8_reduction_pct"] = 100 * (1 - size["onnx_int8_mb"] / size["onnx_fp32_mb"])

    # Accuracy impact: exact same seed-labeled-only test split as
    # reports/mood_cnn_metrics.json.
    df = pd.read_csv(resolve_path(cfg["paths"]["mood_labels_csv"]))
    _, test = train_test_split(df, test_size=mcfg["test_split"], random_state=cfg["seed"], stratify=df["label"])
    seed_test = test[test["provenance"] != "pseudo"].reset_index(drop=True)

    y_true = seed_test["label"].tolist()
    preds = {"pytorch": [], "onnx_fp32": [], "onnx_int8": []}
    for _, row in seed_test.iterrows():
        x = load_mood_tensor(row["image_path"], mcfg["input_size"]).astype(np.float32)
        preds["pytorch"].append(class_names[run_pt(x).argmax(axis=1)[0]])
        preds["onnx_fp32"].append(class_names[run_fp32(x).argmax(axis=1)[0]])
        preds["onnx_int8"].append(class_names[run_int8(x).argmax(axis=1)[0]])

    accuracy = {k: accuracy_score(y_true, v) for k, v in preds.items()}
    accuracy["n_test"] = len(y_true)

    return {"latency_ms": latency, "size_mb": size, "seed_labeled_only_accuracy": accuracy}


def benchmark_keypoint_model(cfg: dict) -> dict:
    kcfg = cfg["keypoints"]
    onnx_dir = resolve_path(cfg["paths"]["artifacts"]) / "onnx"

    pt_path = resolve_path(cfg["paths"]["keypoint_model"])
    fp32_path = onnx_dir / "keypoint_model_fp32.onnx"
    int8_path = onnx_dir / "keypoint_model_int8.onnx"

    device = torch.device("cpu")
    pt_model = HeatmapKeypointModel(num_keypoints=kcfg["num_keypoints"], pretrained=False).to(device)
    pt_model.load_state_dict(torch.load(pt_path, map_location=device))
    pt_model.eval()

    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    input_name = sess_fp32.get_inputs()[0].name

    dummy = np.random.randn(1, 3, kcfg["input_size"], kcfg["input_size"]).astype(np.float32)

    @torch.no_grad()
    def run_pt(x):
        return pt_model(torch.from_numpy(x)).numpy()

    def run_fp32(x):
        return sess_fp32.run(None, {input_name: x})[0]

    def run_int8(x):
        return sess_int8.run(None, {input_name: x})[0]

    latency = {
        "pytorch_cpu": measure_latency(run_pt, dummy),
        "onnx_fp32_cpu": measure_latency(run_fp32, dummy),
        "onnx_int8_cpu": measure_latency(run_int8, dummy),
    }

    size = {
        "pytorch_state_dict_mb": pt_path.stat().st_size / 1e6,
        "onnx_fp32_mb": fp32_path.stat().st_size / 1e6,
        "onnx_int8_mb": int8_path.stat().st_size / 1e6,
    }
    size["fp32_to_int8_reduction_pct"] = 100 * (1 - size["onnx_int8_mb"] / size["onnx_fp32_mb"])

    # NME accuracy impact on a real sample of the test split (same split
    # logic/seed as keypoints/train.py; a 200-image subsample keeps this
    # section's runtime reasonable while still being a real measurement,
    # not a single-image anecdote).
    images_dir = resolve_path(cfg["paths"]["crawford_images_dir"])
    annotations_dir = resolve_path(cfg["paths"]["crawford_annotations_dir"])
    samples = list_samples(images_dir, annotations_dir, validate=True)
    _, test_samples = train_test_split(samples, test_size=kcfg["test_split"], random_state=cfg["seed"])
    rng = np.random.RandomState(cfg["seed"])
    subsample = [test_samples[i] for i in rng.choice(len(test_samples), size=min(200, len(test_samples)), replace=False)]

    def nme_for(run_fn):
        errs = []
        for img_path, cat_path in subsample:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h0, w0 = img.shape[:2]
            kp_gt = parse_cat_file(cat_path)
            img_resized = cv2.resize(img, (kcfg["input_size"], kcfg["input_size"]))
            scale = np.array([kcfg["input_size"] / w0, kcfg["input_size"] / h0])
            kp_gt_resized = kp_gt * scale

            tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
            tensor = NORMALIZE(tensor).unsqueeze(0).numpy().astype(np.float32)
            heatmaps = run_fn(tensor)[0]
            kp_pred, _ = decode_heatmaps(heatmaps, kcfg["input_size"], kcfg["heatmap_size"])

            eye_dist = max(np.linalg.norm(kp_gt_resized[0] - kp_gt_resized[1]), 1.0)
            err = np.linalg.norm(kp_pred - kp_gt_resized, axis=-1).mean() / eye_dist
            errs.append(err)
        return float(np.mean(errs))

    nme = {
        "pytorch": nme_for(run_pt),
        "onnx_fp32": nme_for(run_fp32),
        "onnx_int8": nme_for(run_int8),
        "n_samples": len(subsample),
    }

    return {"latency_ms": latency, "size_mb": size, "nme_on_subsample": nme}


def main():
    cfg = load_config()
    print("benchmarking mood CNN...")
    mood_results = benchmark_mood_cnn(cfg)
    print(json.dumps(mood_results, indent=2))

    print("\nbenchmarking keypoint model...")
    keypoint_results = benchmark_keypoint_model(cfg)
    print(json.dumps(keypoint_results, indent=2))

    reports_dir = resolve_path(cfg["paths"]["reports"])
    out = {"mood_cnn": mood_results, "keypoint_model": keypoint_results}
    with open(reports_dir / "onnx_benchmark.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {reports_dir / 'onnx_benchmark.json'}")


if __name__ == "__main__":
    main()
