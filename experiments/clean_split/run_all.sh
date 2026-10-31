#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
run() { echo "STAGE START $1 $(date +%H:%M:%S)"; "$PY" -u "$2" > "logs/$1.log" 2>&1 || { echo "STAGE FAILED $1 (see logs/$1.log)"; tail -5 "logs/$1.log"; exit 1; }; echo "STAGE DONE $1 $(date +%H:%M:%S)"; }
run 01_splits build_clean_splits.py
run 04b_leiden src/models/04b_louvain_communities.py
run 05_features src/models/05_build_feature_store.py
run 06_baseline src/models/06_train_xgboost_baseline.py
run tune_threshold src/models/tune_threshold.py
run 07_graph src/models/07_train_graph_enhanced_model.py
echo "ALL STAGES DONE $(date +%H:%M:%S)"
