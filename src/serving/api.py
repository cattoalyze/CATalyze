"""FastAPI inference endpoint (Section 7). Loads the trained keypoint model,
mood CNN (for its embedding), and calibrated ensemble classifier once at
startup, then serves /predict per the brief's JSON contract, plus /gradcam
and /metrics for the demo frontend.

Run with: uv run uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
"""
import base64
import io
import json
import subprocess
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import load_config, resolve_path  # noqa: E402
from src.explainability.gradcam import GradCAM, overlay_cam_on_image  # noqa: E402
from src.features.geometric import FEATURE_NAMES, compute_geometric_features  # noqa: E402
from src.keypoints.bbox import compute_boxes  # noqa: E402
from src.keypoints.infer import KeypointPredictor  # noqa: E402
from src.mood_cnn.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.mood_cnn.model import MoodCNN  # noqa: E402
from src.serving.inference_backends import PyTorchBackend  # noqa: E402
from src.serving.live_tracker import LiveTracker  # noqa: E402

cfg = load_config()
scfg = cfg["serving"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="CATalyze", description="Cat mood classification with calibrated confidence")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_state = {}


@app.on_event("startup")
def load_models():
    kcfg = cfg["keypoints"]
    _state["keypoint_predictor"] = KeypointPredictor(
        model_path=resolve_path(cfg["paths"]["keypoint_model"]),
        num_keypoints=kcfg["num_keypoints"],
        input_size=kcfg["input_size"],
        heatmap_size=kcfg["heatmap_size"],
        device=str(device),
    )

    mcfg = cfg["mood_cnn"]
    mood_model = MoodCNN(num_classes=len(cfg["mood_classes"]), embedding_dim=mcfg["embedding_dim"], pretrained=False).to(device)
    mood_model.load_state_dict(torch.load(resolve_path(cfg["paths"]["mood_cnn_model"]), map_location=device))
    mood_model.eval()
    _state["mood_model"] = mood_model
    _state["mood_input_size"] = mcfg["input_size"]
    _state["normalize"] = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    ensemble_bundle = joblib.load(resolve_path(cfg["paths"]["ensemble_model"]))
    _state["ensemble"] = ensemble_bundle["calibrated"]
    _state["ensemble_classes"] = list(ensemble_bundle["calibrated"].classes_)

    _state["gradcam"] = GradCAM(mood_model)

    # Live-camera path (Section 2): raw (uncalibrated) ensemble, not the
    # CalibratedClassifierCV used above — measured ~3.4x faster per frame
    # (reports/live_backend_benchmark.json) since CalibratedClassifierCV's
    # 5-fold internal calibration re-runs the whole 300-tree RF 5x per
    # prediction. Accuracy cost is negligible (91.76% raw vs 91.81%
    # calibrated) and EMA smoothing across frames further compensates for
    # the less-calibrated per-frame probabilities.
    _state["live_backend"] = PyTorchBackend(
        keypoint_model=_state["keypoint_predictor"].model,
        mood_model=mood_model,
        input_size_kp=kcfg["input_size"],
        heatmap_size=kcfg["heatmap_size"],
        input_size_mood=mcfg["input_size"],
        device=device,
    )
    _state["live_ensemble"] = ensemble_bundle["raw"]


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": bool(_state)}


def _decode_and_validate(file: UploadFile, content: bytes) -> np.ndarray:
    if file.content_type not in scfg["allowed_content_types"]:
        raise HTTPException(status_code=400, detail=f"unsupported content type: {file.content_type}")
    if len(content) > scfg["max_upload_mb"] * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"file exceeds {scfg['max_upload_mb']}MB limit")
    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="could not decode image")
    return np.array(img)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    img_rgb = _decode_and_validate(file, content)

    kp, kp_conf = _state["keypoint_predictor"].predict(img_rgb)
    geo_features = compute_geometric_features(kp)
    boxes = compute_boxes(kp, kp_conf, image_shape=img_rgb.shape)

    resized = cv2.resize(img_rgb, (_state["mood_input_size"], _state["mood_input_size"]))
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
    tensor = _state["normalize"](tensor).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = _state["mood_model"].embed(tensor).cpu().numpy()[0]

    feature_vec = np.concatenate([geo_features, embedding]).reshape(1, -1)
    probs = _state["ensemble"].predict_proba(feature_vec)[0]
    classes = _state["ensemble_classes"]

    return {
        "predictions": {cls: float(p) for cls, p in zip(classes, probs)},
        "keypoints": [
            {"x": float(x), "y": float(y), "confidence": float(c)}
            for (x, y), c in zip(kp, kp_conf)
        ],
        "boxes": boxes,
        "geometric_features": {name: float(v) for name, v in zip(FEATURE_NAMES, geo_features)},
    }


@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):
    content = await file.read()
    img_rgb = _decode_and_validate(file, content)

    input_size = _state["mood_input_size"]
    resized = cv2.resize(img_rgb, (input_size, input_size))
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
    tensor = _state["normalize"](tensor).unsqueeze(0).to(device)

    cam, class_idx, logits = _state["gradcam"](tensor)
    overlay = overlay_cam_on_image(resized, cam)

    ok, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    class_names = cfg["mood_classes"]
    probs = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
    return {
        "overlay_png_base64": b64,
        "mood_cnn_predicted_class": class_names[class_idx],
        "mood_cnn_raw_probabilities": {c: float(p) for c, p in zip(class_names, probs)},
    }


@app.websocket("/ws/live")
async def live_ws(websocket: WebSocket):
    """Live-camera path: client sends one JPEG/PNG-encoded frame per binary
    message, server replies with one JSON prediction per frame — same
    keypoint -> geometric-features + mood-CNN-embedding -> ensemble
    pipeline as /predict, but with EMA smoothing across frames and the raw
    (uncalibrated) ensemble for speed (see load_models() for why).

    A fresh LiveTracker (and its EMA state) is created per connection so
    concurrent clients don't smooth across each other's frames; the
    underlying models/ensemble are shared read-only.
    """
    await websocket.accept()
    tracker = LiveTracker(
        backend=_state["live_backend"],
        ensemble=_state["live_ensemble"],
        ensemble_classes=_state["ensemble_classes"],
    )
    try:
        while True:
            data = await websocket.receive_bytes()
            arr = np.frombuffer(data, dtype=np.uint8)
            frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                await websocket.send_json({"error": "could not decode frame"})
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = tracker.process_frame(frame_rgb)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass


@app.get("/metrics")
def metrics():
    """Serves reports/metrics.json verbatim — the real source of truth from
    the core pipeline's evaluation runs. Never re-typed or estimated here."""
    metrics_path = resolve_path(cfg["paths"]["reports"]) / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="reports/metrics.json not found — run src.reports.generate_metrics first")
    with open(metrics_path) as f:
        return json.load(f)


# Every one of these is a real report already written by a real
# training/eval/analysis script (see README.md "Running the pipeline") —
# this just serves whichever one the frontend asks for, verbatim, same
# honesty contract as /metrics. Allowlisted so the frontend can't be used
# to read arbitrary files off disk.
_ALLOWED_REPORTS = {
    "metrics", "ensemble_metrics", "ensemble_kfold_metrics", "keypoint_metrics",
    "mood_cnn_metrics", "subgroup_analysis", "feature_stats", "onnx_benchmark",
    "live_backend_benchmark", "anxious_supplement_experiment",
}


@app.get("/reports/{name}")
def get_report(name: str):
    if name not in _ALLOWED_REPORTS:
        raise HTTPException(status_code=404, detail=f"unknown report: {name}")
    path = resolve_path(cfg["paths"]["reports"]) / f"{name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"reports/{name}.json not found — run the corresponding script first")
    with open(path) as f:
        return json.load(f)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent.parent.parent,
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except Exception:
        return None


@app.get("/version")
def version():
    """Real build metadata for the frontend's model/version badge — no
    fabricated model name or version number. num_keypoints/backbone are
    read from config.yaml, never hardcoded separately from it."""
    return {
        "git_commit": _git_commit(),
        "keypoint_backbone": cfg["keypoints"]["backbone"],
        "mood_cnn_backbone": cfg["mood_cnn"]["backbone"],
        "num_keypoints": cfg["keypoints"]["num_keypoints"],
        "mood_classes": cfg["mood_classes"],
    }


frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
