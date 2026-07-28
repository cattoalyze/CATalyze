"""Manual mood-labeling web UI. Serves one unlabeled image at a time with
four buttons (one per mood class); each click appends a row to
mood_labels.csv with provenance='human' and serves the next image.

Run with: uv run python -m src.labeling.seed_tool
Then open http://localhost:8001 in a browser.
"""
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import load_config, resolve_path  # noqa: E402

cfg = load_config()
CLASS_NAMES = cfg["mood_classes"]
IMAGES_DIR = resolve_path(cfg["paths"]["crawford_images_dir"])
LABELS_PATH = resolve_path(cfg["paths"]["mood_labels_csv"])

app = FastAPI(title="CATalyze seed labeling tool")


def _load_labels() -> pd.DataFrame:
    if LABELS_PATH.exists():
        return pd.read_csv(LABELS_PATH)
    return pd.DataFrame(columns=["image_path", "label", "provenance"])


def _next_unlabeled_image() -> Path | None:
    df = _load_labels()
    labeled = set(df["image_path"])
    for img_path in sorted(IMAGES_DIR.glob("*.jpg")):
        if str(img_path) not in labeled:
            return img_path
    return None


@app.get("/", response_class=HTMLResponse)
def index():
    img_path = _next_unlabeled_image()
    if img_path is None:
        return "<h2>All images labeled.</h2>"
    df = _load_labels()
    buttons = "".join(
        f'<button onclick="label(\'{c}\')" style="font-size:20px;margin:8px;padding:12px 24px;">{c}</button>'
        for c in CLASS_NAMES
    )
    return f"""
    <html><body style="font-family:sans-serif;text-align:center;">
      <h3>{len(df)} labeled so far</h3>
      <img src="/image" style="max-height:70vh;max-width:90vw;" /><br/>
      {buttons}
      <script>
        function label(cls) {{
          fetch('/label?path={img_path.as_posix()}&cls=' + cls, {{method:'POST'}})
            .then(() => location.reload());
        }}
      </script>
    </body></html>
    """


@app.get("/image")
def image():
    img_path = _next_unlabeled_image()
    return FileResponse(img_path)


@app.post("/label")
def label(path: str, cls: str):
    if cls not in CLASS_NAMES:
        return {"error": "invalid class"}
    df = _load_labels()
    df = pd.concat([df, pd.DataFrame([{"image_path": path, "label": cls, "provenance": "human"}])], ignore_index=True)
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LABELS_PATH, index=False)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
