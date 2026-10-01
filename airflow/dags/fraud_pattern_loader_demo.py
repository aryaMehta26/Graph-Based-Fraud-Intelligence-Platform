from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator


PROJECT_ROOT = Path("/opt/project")
PATTERN_SCRIPT = PROJECT_ROOT / "src" / "graph" / "load_fraud_patterns.py"


def verify_pattern_loader_artifacts() -> None:
    missing = []
    if not PATTERN_SCRIPT.exists():
        missing.append(str(PATTERN_SCRIPT))
    if missing:
        raise RuntimeError(f"Fraud pattern loader prerequisites missing: {missing}")


DAG_DOC = """
## `fraud_pattern_loader_demo`

**Purpose:** Load **IBM AML fraud-pattern** relationship data into Neo4j (ontology used by your graph story).

**Not EDA:** This DAG does not run notebooks. It checks that `src/graph/load_fraud_patterns.py` exists, then runs it.

**Terminal command (what Airflow runs):**
```bash
cd /opt/project && python3 src/graph/load_fraud_patterns.py
```

**Code file:** `src/graph/load_fraud_patterns.py`
"""


with DAG(
    dag_id="fraud_pattern_loader_demo",
    description="Load IBM AML laundering pattern relationships into Neo4j using the project fraud-pattern script.",
    doc_md=DAG_DOC,
    start_date=datetime.now() - timedelta(days=1),
    schedule=None,
    catchup=False,
    tags=["fraud-patterns", "neo4j", "demo", "ontology"],
) as dag:
    verify_loader = PythonOperator(
        task_id="verify_pattern_loader_script",
        python_callable=verify_pattern_loader_artifacts,
    )

    load_patterns = BashOperator(
        task_id="load_fraud_patterns_into_neo4j",
        bash_command="cd /opt/project && python3 src/graph/load_fraud_patterns.py",
    )

    verify_loader >> load_patterns
