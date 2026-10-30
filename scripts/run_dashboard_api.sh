#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv-fraud/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then PYTHON_BIN="python3"; fi

echo "Starting DATA 298B dashboard API on http://127.0.0.1:8766"
exec "${PYTHON_BIN}" "${ROOT_DIR}/scripts/api_server.py"
