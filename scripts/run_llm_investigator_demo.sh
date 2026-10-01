#!/usr/bin/env bash
# LLM Investigator — run from project root:  bash scripts/run_llm_investigator_demo.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "Warning: .env not found. Set ANTHROPIC_API_KEY in the environment." >&2
fi

# Pick one subgraph under artifacts/ (JSON). Examples ship with the repo:
SUBGRAPH="${SUBGRAPH:-sample_subgraph}"
VARIANT="${VARIANT:-v2}"

echo "Running investigator: subgraph=${SUBGRAPH} variant=${VARIANT}"
python3 src/llm/investigator.py --variant "${VARIANT}" --input "artifacts/${SUBGRAPH}.json"

echo ""
echo "Optional: recompute eval metrics (no API calls)"
python3 src/llm/evaluate.py --subgraph "${SUBGRAPH}" --input "artifacts/${SUBGRAPH}.json"

echo "Done. Outputs under artifacts/llm_outputs/ and artifacts/metrics/"
