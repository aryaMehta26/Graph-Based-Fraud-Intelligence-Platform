#!/usr/bin/env python3
"""
IBM AML end-to-end pipeline orchestrator (same stages as run_all_backend_eda.py).

Stages reference the canonical notebooks:
  1  notebooks/01_data_extraction.py   — Kaggle ingest + file validation
  2  notebooks/02_data_cleaning.py       — full clean + parquet lakehouse (31M rows, heavy)
  3  notebooks/03_eda_visualizations.py  — full EDA charts (31M rows, heavy)
  4  (optional) caller runs fraud_lakehouse_neo4j_sync.py for Neo4j QA

Environment:
  SKIP_HEAVY_STAGES=1  — skip stages 2–3 if processed artifacts already exist
                         (default for Airflow in Docker so the DAG does not OOM).
  SKIP_HEAVY_STAGES=0  — always run 02 and 03 (use on a machine with RAM + time).

Usage:
  python3 pipeline_orchestrate.py --stage 1
  python3 pipeline_orchestrate.py --stage 2
  python3 pipeline_orchestrate.py --stage 3
  python3 pipeline_orchestrate.py --all
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.getenv("PROJ_ROOT", Path(__file__).resolve().parent)).resolve()
NOTEBOOKS = ROOT / "notebooks"
PROCESSED = ROOT / "data" / "processed"
GRAPH_PQ = PROCESSED / "transactions_graph.parquet"
EDA_DIR = NOTEBOOKS / "eda_charts_live"

SCRIPTS = {
    1: NOTEBOOKS / "01_data_extraction.py",
    2: NOTEBOOKS / "02_data_cleaning.py",
    3: NOTEBOOKS / "03_eda_visualizations.py",
}


def skip_heavy() -> bool:
    return os.getenv("SKIP_HEAVY_STAGES", "").strip().lower() in ("1", "true", "yes")


def artifacts_ready_for_stage2() -> bool:
    return GRAPH_PQ.is_file() and GRAPH_PQ.stat().st_size > 0


def artifacts_ready_for_stage3() -> bool:
    if not EDA_DIR.is_dir():
        return False
    return any(EDA_DIR.glob("*.png"))


def run_stage(stage: int) -> int:
    script = SCRIPTS[stage]
    if not script.is_file():
        print(f"ERROR: missing pipeline script: {script}", file=sys.stderr)
        return 1

    if skip_heavy():
        if stage == 1:
            if (PROCESSED / "transactions_clean.parquet").is_file() or GRAPH_PQ.is_file():
                print(
                    "[SKIP_HEAVY_STAGES] Stage 1: processed parquet already present; "
                    "skipping Kaggle download step. Re-run: SKIP_HEAVY_STAGES=0 python3 pipeline_orchestrate.py --stage 1"
                )
                return 0
        if stage == 2 and artifacts_ready_for_stage2():
            print(
                f"[SKIP_HEAVY_STAGES] Stage 2: using existing lakehouse artifacts "
                f"({GRAPH_PQ.relative_to(ROOT)}). Full rebuild: SKIP_HEAVY_STAGES=0 python3 {script}"
            )
            return 0
        if stage == 3 and artifacts_ready_for_stage3():
            print(
                f"[SKIP_HEAVY_STAGES] Stage 3: using existing EDA outputs under "
                f"{EDA_DIR.relative_to(ROOT)}. Full rebuild: SKIP_HEAVY_STAGES=0 python3 {script}"
            )
            return 0
        if stage == 2 and not artifacts_ready_for_stage2():
            print(
                f"[ERROR] Stage {stage} skipped but {GRAPH_PQ} missing. "
                "Run once with SKIP_HEAVY_STAGES=0 on a capable host, or restore artifacts.",
                file=sys.stderr,
            )
            return 1
        if stage == 3 and not artifacts_ready_for_stage3():
            print(
                f"[ERROR] Stage {stage} skipped but no PNGs in {EDA_DIR}. "
                "Run notebooks/03_eda_visualizations.py with SKIP_HEAVY_STAGES=0.",
                file=sys.stderr,
            )
            return 1

    print("=" * 72)
    print(f"PIPELINE STAGE {stage}: {script.relative_to(ROOT)}")
    print("=" * 72)
    proc = subprocess.run([sys.executable, str(script)], cwd=str(ROOT))
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="IBM AML notebook pipeline orchestrator")
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), help="Run a single stage")
    parser.add_argument("--all", action="store_true", help="Run stages 1 then 2 then 3 in order")
    args = parser.parse_args()

    if not args.stage and not args.all:
        parser.error("pass --stage N or --all")

    stages = [1, 2, 3] if args.all else [args.stage]
    for s in stages:
        rc = run_stage(s)
        if rc != 0:
            return rc
    print("\nPIPELINE_ORCHESTRATE_ALL_STAGES_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
