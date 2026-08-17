#!/usr/bin/env bash
# Starts the CATalyze serving API (src/serving/api.py), which also serves
# the demo frontend at "/". Backgrounded so this script returns immediately
# -- use ./stop.sh to shut it down. Port overridable via PORT (default
# 8000, matching README.md / RUN.md).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PORT="${PORT:-8000}"
RUN_DIR="$REPO_ROOT/.run"
PID_FILE="$RUN_DIR/server.pid"
LOG_FILE="$RUN_DIR/server.log"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"

mkdir -p "$RUN_DIR"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Already running (PID $(cat "$PID_FILE")). Run ./stop.sh first if you want to restart." >&2
  exit 1
fi

if [ ! -x "$VENV_PYTHON" ]; then
  echo "No .venv found at $VENV_PYTHON -- run the environment setup in README.md first:" >&2
  echo "  uv venv .venv --python 3.11 && uv pip install -r requirements.txt" >&2
  exit 1
fi

missing=0
for artifact in artifacts/keypoint_model.pt artifacts/mood_cnn_model.pt artifacts/ensemble_model.joblib; do
  if [ ! -f "$REPO_ROOT/$artifact" ]; then
    echo "Missing $artifact" >&2
    missing=1
  fi
done
if [ "$missing" -eq 1 ]; then
  echo "Pipeline hasn't been trained yet -- see RUN.md steps 1-7." >&2
  exit 1
fi

nohup "$VENV_PYTHON" -m uvicorn src.serving.api:app --host 0.0.0.0 --port "$PORT" \
  > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "Starting CATalyze server (PID $(cat "$PID_FILE"), port $PORT)..."

# Confirm it actually came up rather than crashed on startup.
for _ in $(seq 1 20); do
  if curl -s -o /dev/null "http://127.0.0.1:$PORT/health"; then
    echo "Up: http://localhost:$PORT/  (logs: $LOG_FILE)"
    exit 0
  fi
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Server process died on startup -- check $LOG_FILE" >&2
    rm -f "$PID_FILE"
    exit 1
  fi
  sleep 0.5
done

echo "Server process is running but didn't respond to /health within 10s -- check $LOG_FILE" >&2
exit 1
