#!/usr/bin/env bash
# Start OmniWatch FastAPI and Web UI server
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

echo "[OmniWatch] Starting server on http://$HOST:$PORT"
python -m uvicorn omniwatch.api.main:app --host "$HOST" --port "$PORT" --reload
