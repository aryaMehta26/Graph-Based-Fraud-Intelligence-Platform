"""Run metrics over JSONL traces; no model calls and no test-label leakage."""
import argparse, json
from pathlib import Path
from .metrics import classification_metrics, pattern_metrics, schema_metrics, faithfulness_metrics, agent_metrics, efficiency_metrics, generation_metrics

def run(trace_path: Path, output_dir: Path):
    traces=[json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    y_true=[t.get("ground_truth",{}).get("decision","LEGITIMATE") for t in traces]; y_pred=[t.get("report",{}).get("decision","LEGITIMATE") for t in traces]
    p_true=[t.get("ground_truth",{}).get("pattern","EMERGING_UNKNOWN") for t in traces]; p_pred=[t.get("report",{}).get("pattern","EMERGING_UNKNOWN") for t in traces]; reports=[t.get("report",{}) for t in traces]
    references=[t.get("reference_report",{}).get("summary","") for t in traces]; predictions=[t.get("report",{}).get("summary","") for t in traces]
    result={"binary":classification_metrics(y_true,y_pred),"pattern":pattern_metrics(p_true,p_pred),"schema":schema_metrics(reports),"faithfulness":faithfulness_metrics(traces),"agent":agent_metrics(traces),"efficiency":efficiency_metrics(traces),"generation":generation_metrics(predictions,references) if any(references) else {"status":"no reference reports"},"cases":len(traces)}
    output_dir.mkdir(parents=True,exist_ok=True); (output_dir/"benchmark_metrics.json").write_text(json.dumps(result,indent=2)); return result

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--traces",type=Path,required=True); p.add_argument("--output-dir",type=Path,default=Path("artifacts/evaluation")); a=p.parse_args(); print(json.dumps(run(a.traces,a.output_dir),indent=2))
