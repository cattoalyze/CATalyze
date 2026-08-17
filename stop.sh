#!/usr/bin/env bash
# Stops the CATalyze server started by run.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$REPO_ROOT/.run/server.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "No PID file at $PID_FILE -- server doesn't appear to be running (or wasn't started with run.sh)." >&2
  exit 1
fi

PID="$(cat "$PID_FILE")"
if ! kill -0 "$PID" 2>/dev/null; then
  echo "Process $PID isn't running; removing stale PID file."
  rm -f "$PID_FILE"
  exit 0
fi

kill "$PID"
for _ in $(seq 1 20); do
  if ! kill -0 "$PID" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "Stopped (PID $PID)."
    exit 0
  fi
  sleep 0.5
done

echo "PID $PID didn't exit after SIGTERM, sending SIGKILL." >&2
kill -9 "$PID" 2>/dev/null || true
rm -f "$PID_FILE"
echo "Force-stopped (PID $PID)."
