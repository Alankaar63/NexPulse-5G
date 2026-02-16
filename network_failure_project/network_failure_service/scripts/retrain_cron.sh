#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d "$ROOT_DIR/.nfp-venv" ]; then
  python3 -m venv "$ROOT_DIR/.nfp-venv"
fi

"$ROOT_DIR/.nfp-venv/bin/pip" install -r "$ROOT_DIR/network_failure_service/requirements.txt" >/dev/null
"$ROOT_DIR/.nfp-venv/bin/python" "$ROOT_DIR/network_failure_service/scripts/retrain_model.py" --promote
