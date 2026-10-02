"""Real-data DATA 298B integration dry run; no model weights or Neo4j writes."""
import json, os, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.aml.dataset import build_cases_from_processed_data, to_sft_record
from src.aml.registry import load_registry
from src.aml.tools import TOOLS, execute_tool
from src.aml.agent import AMLAgent
from src.evaluation.benchmark_runner import run as run_evaluation
OUT = ROOT / "artifacts" / "evaluation_dry_run"
class MockBackend:
    def generate(self, messages, **kwargs):
        return {"text": json.dumps({"decision":"SUSPICIOUS","pattern":"FAN_IN","risk_level":"HIGH","confidence":0.5,"evidence":["dry-run mocked evidence"],"recommended_actions":["Review case"],"summary":"Mocked integration investigation"}),"prompt_tokens":10,"output_tokens":35}
def main():
    OUT.mkdir(parents=True, exist_ok=True); issues=[]; warnings=[]
    cases=build_cases_from_processed_data(ROOT/"data/processed/train_graph_enriched.parquet", ROOT/"HI-Medium_Patterns.txt", OUT/"cases.jsonl", limit=100)
    if len(cases)!=100: issues.append(f"expected 100 cases, built {len(cases)}")
    forbidden=("is_laundering", "pattern_type", '"pattern"', "ground_truth", '"label"')
    leaks={c.case_id:[x for x in forbidden if x in json.dumps(c.agent_context()).lower()] for c in cases if any(x in json.dumps(c.agent_context()).lower() for x in forbidden)}
    if leaks: issues.append(f"inference leakage: {leaks}")
    print("cases:",len(cases)); print("decision distribution:",dict(Counter(c.ground_truth["decision"] for c in cases))); print("pattern distribution:",dict(Counter(c.ground_truth["pattern"] for c in cases)))
    sft=OUT/"sft.jsonl"; sft.write_text("\n".join(json.dumps(to_sft_record(c),default=str) for c in cases)+"\n"); print("sft_jsonl:",sft)
    try:
        for name,spec in load_registry().items(): print(f"registry {name}: {spec['model_id']} @ {spec['revision']} ({spec['parameters_b']}B)")
    except Exception as exc: issues.append(f"model registry: {type(exc).__name__}: {exc}")
    for name in TOOLS:
        try: execute_tool(name,cases[0].agent_context(),{}); print(f"tool {name}: OK")
        except Exception as exc: issues.append(f"tool {name}: {type(exc).__name__}: {exc}")
    trace=AMLAgent(MockBackend(),max_tool_calls=3,artifact_dir=str(OUT/"traces")).investigate(cases[0],model_family="dry-run",variant="base")
    traces=OUT/"traces.jsonl"; traces.write_text(json.dumps(trace,default=str)+"\n"); print("mock trace:",traces); metrics=run_evaluation(traces,OUT); print("evaluation:",metrics["cases"],"cases; schema",metrics["schema"]["schema_compliance_rate"])
    try:
        from neo4j import GraphDatabase
        uri=os.getenv("NEO4J_URI","neo4j://127.0.0.1:7687"); d=GraphDatabase.driver(uri,auth=(os.getenv("NEO4J_USER","neo4j"),os.getenv("NEO4J_PASSWORD","Aryamehta@26")),connection_timeout=3); d.verify_connectivity(); d.close(); print("neo4j: reachable")
    except Exception as exc: warnings.append(f"Neo4j connectivity (no writes): {type(exc).__name__}: {exc}")
    for p in ("xgboost_baseline_metrics.json","xgboost_graph_enhanced_metrics.json","model_comparison.json"):
        try: json.loads((ROOT/"data/models"/p).read_text()); print("xgboost artifact:",p,"OK")
        except Exception as exc: issues.append(f"XGBoost artifact {p}: {exc}")
    print("ISSUES:"); [print("-",x) for x in issues] if issues else print("- none")
    print("WARNINGS:"); [print("-",x) for x in warnings] if warnings else print("- none")
    return int(bool(issues))
if __name__=="__main__": raise SystemExit(main())
