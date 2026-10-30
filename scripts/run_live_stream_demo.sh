#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv-fraud/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"; fi
echo "Starting local synthetic transaction stream on http://127.0.0.1:8765"
echo "Press Ctrl-C to stop."
exec "$PYTHON_BIN" scripts/stream_demo_server.py
