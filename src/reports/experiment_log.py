"""Lightweight experiment/run logging (Section 3, item 4). Every stage's
metrics JSON (keypoint_metrics.json, mood_cnn_metrics.json,
ensemble_metrics.json, ensemble_kfold_metrics.json) gets overwritten on
each run -- only the latest survives, so past runs aren't queryable. This
appends one JSON line per run to reports/experiment_log.jsonl instead, with
a timestamp and git commit, so run history stays queryable without pulling
in MLflow/W&B as a dependency for a project this size.

Run `python -m src.reports.experiment_log` to print the logged history.
"""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_PATH = REPO_ROOT / "reports" / "experiment_log.jsonl"

# Best-effort single-line summary: first of these keys found in a run's
# metrics dict is shown in the CLI listing.
_SUMMARY_KEYS = (
    "calibrated_accuracy", "overall_calibrated_accuracy", "test_acc",
    "test_nme", "anxious_f1_mean",
)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except Exception:
        return None


def log_run(stage: str, metrics: dict) -> None:
    """Append one run record. Never raises -- a logging failure must never
    fail the training/eval run it's trying to record."""
    try:
        record = {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "stage": stage,
            "metrics": metrics,
        }
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        print(f"[experiment_log] warning: failed to log run for stage={stage}: {e}")


def _summarize(metrics: dict) -> str:
    for key in _SUMMARY_KEYS:
        if key in metrics and metrics[key] is not None:
            return f"{key}={metrics[key]:.4f}"
    return ""


def main():
    if not LOG_PATH.exists():
        print(f"No experiment log yet at {LOG_PATH} -- run a training/eval stage first.")
        return
    with open(LOG_PATH) as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"{len(records)} logged run(s):\n")
    for r in records:
        commit = r.get("git_commit") or "?"
        print(f"{r['logged_at']}  commit={commit}  stage={r['stage']:<16}  {_summarize(r['metrics'])}")


if __name__ == "__main__":
    main()
