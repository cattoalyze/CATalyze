"""Aggregate the per-stage metrics JSON files (produced by each training
script) into a single reports/metrics.json — the one file a README or
frontend should ever need to read for real, measured results. Never
hand-transcribe a number; if a stage hasn't run yet, its section is omitted
rather than filled with a placeholder.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import load_config, resolve_path  # noqa: E402


def _load_if_exists(path: Path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def main():
    cfg = load_config()
    reports_dir = resolve_path(cfg["paths"]["reports"])

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "keypoints": _load_if_exists(reports_dir / "keypoint_metrics.json"),
        "mood_cnn": _load_if_exists(reports_dir / "mood_cnn_metrics.json"),
        "ensemble": _load_if_exists(reports_dir / "ensemble_metrics.json"),
    }

    out_path = reports_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Wrote {out_path}")
    for k, v in metrics.items():
        if k == "generated_at":
            continue
        print(f"  {k}: {'present' if v else 'MISSING (stage not run yet)'}")


if __name__ == "__main__":
    main()
