#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.nfp-venv"

if [ -f "$ROOT_DIR/network_failure_service/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/network_failure_service/.env"
  set +a
fi

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install --upgrade pip >/dev/null
"$VENV_DIR/bin/pip" install -r "$ROOT_DIR/network_failure_service/requirements.txt"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

exec "$VENV_DIR/bin/python" -m uvicorn network_failure_service.main:app --host "$HOST" --port "$PORT" --reload
