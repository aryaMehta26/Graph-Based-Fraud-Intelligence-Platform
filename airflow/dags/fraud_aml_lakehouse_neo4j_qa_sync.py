from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/project"))
ORCHESTRATOR = PROJECT_ROOT / "pipeline_orchestrate.py"
SYNC_SCRIPT = PROJECT_ROOT / "fraud_lakehouse_neo4j_sync.py"
LOG_DIR = PROJECT_ROOT / "logs"
RAW_SAMPLE = PROJECT_ROOT / "3.2_Raw_Data_Sample.png"
TRANSFORMED_SAMPLE = PROJECT_ROOT / "3.4_Transformed_Parquet_Sample.png"
GRAPH_PARQUET = PROJECT_ROOT / "data" / "processed" / "transactions_graph.parquet"
TRAIN_SPLIT = PROJECT_ROOT / "data" / "processed" / "split_train.parquet"
VAL_SPLIT = PROJECT_ROOT / "data" / "processed" / "split_val.parquet"
TEST_SPLIT = PROJECT_ROOT / "data" / "processed" / "split_test.parquet"
GRAPH_METRICS = PROJECT_ROOT / "data" / "models" / "xgboost_graph_enhanced_metrics.json"
LLM_METRICS = PROJECT_ROOT / "artifacts" / "metrics" / "llm_eval.json"

BASH_PREFIX = (
    "cd /opt/project && export PROJ_ROOT=/opt/project && "
    "export SKIP_HEAVY_STAGES=${SKIP_HEAVY_STAGES:-1} && "
)


def require_paths(label: str, paths: list[str]) -> None:
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise RuntimeError(f"{label} missing required paths: {missing}")


DAG_DOC = """
## Full IBM AML pipeline in one DAG (ingestion → cleaning → EDA → warehouse → Neo4j QA)

This DAG wires the **same notebook scripts** as `run_all_backend_eda.py`:

| Order | Task | Runs (real code) |
|-------|------|------------------|
| 1 | `stage01_ingestion_extraction` | `notebooks/01_data_extraction.py` (Kaggle download + validation) |
| 2 | `stage02_cleaning_transformation` | `notebooks/02_data_cleaning.py` (31M clean + parquet lakehouse) |
| 3 | `stage03_eda_visualizations` | `notebooks/03_eda_visualizations.py` (EDA charts) |
| 4–6 | `validate_*` | File checks for screenshots, Parquet splits, model/LLM JSON |
| 7 | `reset_and_load_qa_neo4j_graph` | `fraud_lakehouse_neo4j_sync.py` → clears **`fraudgraph`**, loads sample |
| 8 | `emit_post_sync_airflow_metrics` | Prints **`[AIRFLOW_METRIC]`** lines: Parquet row counts, Neo4j QA counts, latest sync log stats (see task log in UI) |
| 9 | `validate_operational_sync_log` | Asserts required phrases exist in latest `logs/pipeline_run_*.log` |

### `SKIP_HEAVY_STAGES` (Airflow / Docker default = `1`)

Stages **2–3** (and **1** if processed Parquet already exists) **may skip execution** and use **committed artifacts**
so the DAG finishes in minutes inside Docker. That is still the **full pipeline definition** wired to the real
notebook files; on a workstation you set **`SKIP_HEAVY_STAGES=0`** to force a true 31M rebuild (hours + RAM).

### Full rebuild (laptop / server)

```bash
export SKIP_HEAVY_STAGES=0
python3 pipeline_orchestrate.py --all
python3 fraud_lakehouse_neo4j_sync.py --sample-size 5000
```

### Neo4j

Only **`fraudgraph`** is cleared and reloaded by `fraud_lakehouse_neo4j_sync.py`.

### Extra metrics in Airflow

Task **`emit_post_sync_airflow_metrics`** prints lines prefixed with **`[AIRFLOW_METRIC]`** to the task log
(Parquet row counts, Neo4j node/rel counts, latest pipeline log line count and phrase flags). In the Airflow UI:
open that task → **Log** → search `AIRFLOW_METRIC`.
"""


def validate_operational_sync_log() -> None:
    if not LOG_DIR.exists():
        raise RuntimeError("Log directory does not exist after lakehouse sync run.")
    log_files = sorted(LOG_DIR.glob("pipeline_run_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not log_files:
        raise RuntimeError("No lakehouse → Neo4j sync logs found under logs/.")
    latest = log_files[0]
    text = latest.read_text(encoding="utf-8", errors="ignore")
    required_phrases = [
        "Retry successful",
        "Ingested",
        "Graph injection successful",
        "Full operational logs saved to:",
    ]
    missing = [phrase for phrase in required_phrases if phrase not in text]
    if missing:
        raise RuntimeError(f"Latest log is missing required proof phrases: {missing}")


def emit_post_sync_airflow_metrics() -> None:
    """
    Emit grep-friendly metrics into the Airflow task log (no extra services).
    Search task log for prefix: [AIRFLOW_METRIC]
    """
    import pyarrow.parquet as pq

    root = Path(os.getenv("PROJECT_ROOT", str(PROJECT_ROOT)))

    def pq_rows(rel: str) -> None:
        p = root / rel
        if not p.is_file():
            key = rel.replace("/", "_").replace(".", "_")
            print(f"[AIRFLOW_METRIC] {key}_rows=missing")
            return
        n = pq.ParquetFile(p).metadata.num_rows
        key = rel.replace("/", "_").replace(".", "_")
        print(f"[AIRFLOW_METRIC] {key}_rows={int(n)}")

    for rel in (
        "data/processed/transactions_graph.parquet",
        "data/processed/split_train.parquet",
        "data/processed/split_val.parquet",
        "data/processed/split_test.parquet",
    ):
        pq_rows(rel)

    uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    db = os.getenv("NEO4J_DATABASE", "fraudgraph")
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        with driver.session(database=db) as session:
            nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            tx = session.run("MATCH (t:Transaction) RETURN count(t) AS c").single()["c"]
            fraud_tx = session.run(
                "MATCH (t:Transaction) WHERE coalesce(t.is_laundering, 0) = 1 RETURN count(t) AS c"
            ).single()["c"]
        driver.close()
        print(f"[AIRFLOW_METRIC] neo4j_database={db}")
        print(f"[AIRFLOW_METRIC] neo4j_nodes_total={int(nodes)}")
        print(f"[AIRFLOW_METRIC] neo4j_relationships_total={int(rels)}")
        print(f"[AIRFLOW_METRIC] neo4j_transaction_nodes={int(tx)}")
        print(f"[AIRFLOW_METRIC] neo4j_laundering_transaction_nodes={int(fraud_tx)}")
        print("[AIRFLOW_METRIC] neo4j_connect_ok=1")
    except Exception as exc:
        print(f"[AIRFLOW_METRIC] neo4j_connect_ok=0")
        print(f"[AIRFLOW_METRIC] neo4j_snapshot_error={exc!s}")

    log_dir = root / "logs"
    if log_dir.is_dir():
        log_files = sorted(log_dir.glob("pipeline_run_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if log_files:
            latest = log_files[0]
            text = latest.read_text(encoding="utf-8", errors="ignore")
            lines = text.splitlines()
            print(f"[AIRFLOW_METRIC] pipeline_log_latest_file={latest.name}")
            print(f"[AIRFLOW_METRIC] pipeline_log_line_count={len(lines)}")
            print(f"[AIRFLOW_METRIC] pipeline_log_size_bytes={latest.stat().st_size}")
            for phrase in (
                "Retry successful",
                "Ingested",
                "Graph injection successful",
                "Full operational logs saved to:",
            ):
                slug = "_".join(phrase.lower().split())
                print(f"[AIRFLOW_METRIC] pipeline_log_contains__{slug}={1 if phrase in text else 0}")

    print("[AIRFLOW_METRIC] emit_post_sync_airflow_metrics_done=1")


with DAG(
    dag_id="fraud_aml_lakehouse_neo4j_qa_sync",
    description="IBM AML: ingestion + cleaning + EDA (notebook scripts), validate artifacts, Neo4j fraudgraph QA sync.",
    doc_md=DAG_DOC,
    start_date=datetime.now() - timedelta(days=1),
    schedule=None,
    catchup=False,
    tags=["aml", "fraud", "neo4j", "lakehouse", "fraudgraph", "qa-sync", "demo", "ingestion", "eda"],
) as dag:
    stage01 = BashOperator(
        task_id="stage01_ingestion_extraction",
        bash_command=BASH_PREFIX + "python3 pipeline_orchestrate.py --stage 1",
    )

    stage02 = BashOperator(
        task_id="stage02_cleaning_transformation",
        bash_command=BASH_PREFIX + "python3 pipeline_orchestrate.py --stage 2",
    )

    stage03 = BashOperator(
        task_id="stage03_eda_visualizations",
        bash_command=BASH_PREFIX + "python3 pipeline_orchestrate.py --stage 3",
    )

    validate_orchestrator = PythonOperator(
        task_id="validate_pipeline_orchestrator_present",
        python_callable=require_paths,
        op_kwargs={
            "label": "Pipeline orchestrator and Neo4j sync driver",
            "paths": [str(ORCHESTRATOR), str(SYNC_SCRIPT)],
        },
    )

    validate_source = PythonOperator(
        task_id="validate_source_dataset_evidence",
        python_callable=require_paths,
        op_kwargs={
            "label": "Source dataset evidence",
            "paths": [str(RAW_SAMPLE), str(TRANSFORMED_SAMPLE)],
        },
    )

    validate_parquet = PythonOperator(
        task_id="validate_engineered_parquet_lakehouse",
        python_callable=require_paths,
        op_kwargs={
            "label": "Engineered parquet lakehouse",
            "paths": [str(GRAPH_PARQUET), str(TRAIN_SPLIT), str(VAL_SPLIT), str(TEST_SPLIT)],
        },
    )

    validate_downstream = PythonOperator(
        task_id="validate_downstream_model_and_llm_metrics",
        python_callable=require_paths,
        op_kwargs={
            "label": "Downstream model and LLM metric artifacts",
            "paths": [str(GRAPH_METRICS), str(LLM_METRICS)],
        },
    )

    reset_and_load = BashOperator(
        task_id="reset_and_load_qa_neo4j_graph",
        bash_command="cd /opt/project && NEO4J_DATABASE=fraudgraph python3 fraud_lakehouse_neo4j_sync.py --sample-size 1000",
    )

    emit_metrics = PythonOperator(
        task_id="emit_post_sync_airflow_metrics",
        python_callable=emit_post_sync_airflow_metrics,
    )

    validate_log = PythonOperator(
        task_id="validate_operational_sync_log",
        python_callable=validate_operational_sync_log,
    )

    (
        stage01
        >> stage02
        >> stage03
        >> validate_orchestrator
        >> validate_source
        >> validate_parquet
        >> validate_downstream
        >> reset_and_load
        >> emit_metrics
        >> validate_log
    )
