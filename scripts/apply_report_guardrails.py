"""Apply report guardrails to saved traces and recompute metrics locally."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.aml.report_guardrails import apply_report_guardrails
from src.evaluation.benchmark_runner import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    traces = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    guarded = []
    for trace in traces:
        updated = dict(trace)
        updated["report"] = apply_report_guardrails(trace.get("report", {}), trace.get("context", {}))
        guarded.append(updated)
    args.output.write_text("\n".join(json.dumps(trace, default=str) for trace in guarded) + "\n")
    metrics = run(args.output, args.output.parent / "metrics")
    (args.output.parent / "metrics_guarded.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps(metrics, indent=2, default=str))
    print("saved:", args.output)


if __name__ == "__main__":
    main()
