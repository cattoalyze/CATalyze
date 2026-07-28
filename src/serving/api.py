"""FastAPI inference endpoint (Section 7). Loads the trained keypoint model,
mood CNN (for its embedding), and calibrated ensemble classifier once at
startup, then serves /predict per the brief's JSON contract.

Run with: uv run uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
"""
import io
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import load_config, resolve_path  # noqa: E402
from src.features.geometric import FEATURE_NAMES, compute_geometric_features  # noqa: E402
from src.keypoints.infer import KeypointPredictor  # noqa: E402
from src.mood_cnn.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.mood_cnn.model import MoodCNN  # noqa: E402

cfg = load_config()
scfg = cfg["serving"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="CATalyze", description="Cat mood classification with calibrated confidence")

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
        "geometric_features": {name: float(v) for name, v in zip(FEATURE_NAMES, geo_features)},
    }
