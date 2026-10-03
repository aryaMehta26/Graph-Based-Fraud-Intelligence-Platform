#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"
PYTHON_BIN="$ROOT_DIR/.venv-fraud/bin/python"

if command -v pnpm >/dev/null 2>&1; then
  PNPM=(pnpm)
elif command -v corepack >/dev/null 2>&1; then
  PNPM=(corepack pnpm)
elif command -v npx >/dev/null 2>&1; then
  PNPM=(npx --yes pnpm)
else
  echo "Error: pnpm, corepack, and npx are unavailable. Install Node.js or pnpm first." >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

echo "Refreshing dashboard snapshot from pipeline artifacts..."
"$PYTHON_BIN" "$ROOT_DIR/scripts/export_frontend_snapshot.py"

echo "Building Figma dashboard..."
cd "$FRONTEND_DIR"
"${PNPM[@]}" build

echo "Dashboard available at http://localhost:4173/#/overview"
if curl -fsS "http://127.0.0.1:4173" >/dev/null 2>&1; then
  echo "A dashboard server is already running on port 4173; reuse it after the build."
  exit 0
fi
exec "${PNPM[@]}" preview --host 127.0.0.1 --port 4173
