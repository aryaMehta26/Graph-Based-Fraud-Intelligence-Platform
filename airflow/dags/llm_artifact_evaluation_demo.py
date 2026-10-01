from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator


PROJECT_ROOT = Path("/opt/project")
SUBGRAPH = PROJECT_ROOT / "artifacts" / "real_fan_in_subgraph.json"
INVESTIGATOR_OUTPUT = PROJECT_ROOT / "artifacts" / "llm_outputs" / "real_fan_in_subgraph_v2.json"


def verify_llm_inputs() -> None:
    required = [SUBGRAPH, INVESTIGATOR_OUTPUT]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"LLM evaluator prerequisites missing: {missing}")


DAG_DOC = """
## `llm_artifact_evaluation_demo`

**Purpose:** Recompute **LLM evaluation metrics** from **saved** subgraph + investigator JSON — **no Anthropic API calls** during this DAG.

**Not EDA / not graph sync:** Pure metrics refresh for the demo eval table.

**Terminal command (what Airflow runs):**
```bash
cd /opt/project && python3 src/llm/evaluate.py --subgraph real_fan_in_subgraph
```

**Code file:** `src/llm/evaluate.py`
**Prereq files checked:** `artifacts/real_fan_in_subgraph.json`, `artifacts/llm_outputs/real_fan_in_subgraph_v2.json`
"""


with DAG(
    dag_id="llm_artifact_evaluation_demo",
    description="Recompute saved LLM evaluation metrics from project artifacts without making new API calls.",
    doc_md=DAG_DOC,
    start_date=datetime.now() - timedelta(days=1),
    schedule=None,
    catchup=False,
    tags=["llm", "evaluation", "demo", "artifacts"],
) as dag:
    verify_inputs = PythonOperator(
        task_id="verify_llm_saved_artifacts",
        python_callable=verify_llm_inputs,
    )

    evaluate_outputs = BashOperator(
        task_id="recompute_llm_metrics",
        bash_command="cd /opt/project && python3 src/llm/evaluate.py --subgraph real_fan_in_subgraph",
    )

    verify_inputs >> evaluate_outputs
