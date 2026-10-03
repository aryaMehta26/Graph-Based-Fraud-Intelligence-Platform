"""Run the controlled, no-tool four-model AML comparison."""
import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.aml.agent import SYSTEM_PROMPT, _json
from src.aml.backend import TransformersBackend
from src.aml.dataset import to_sft_record
from src.aml.schemas import InvestigationCase, InvestigationReport
from src.aml.registry import load_registry
from src.evaluation.benchmark_runner import run as evaluate_traces


def load_cases(path: Path, legitimate: int, suspicious: int):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    cases = [InvestigationCase(**record) for record in records]
    legit = [case for case in cases if case.ground_truth.get("decision") == "LEGITIMATE"][:legitimate]
    fraud = [case for case in cases if case.ground_truth.get("decision") == "SUSPICIOUS"][:suspicious]
    selected = legit + fraud
    if len(legit) != legitimate or len(fraud) != suspicious:
        raise ValueError(f"Expected {legitimate} legitimate and {suspicious} suspicious cases; found {len(legit)} and {len(fraud)}")
    forbidden = {"is_laundering", "fraud_label", "pattern", "pattern_type", "ground_truth", "label"}
    leaks = []
    for case in selected:
        context = json.dumps(case.agent_context(), default=str).lower()
        if any(token in context for token in forbidden):
            leaks.append(case.case_id)
    if leaks:
        raise ValueError(f"Inference-context leakage detected: {leaks}")
    return selected


def report_from_text(text: str):
    try:
        obj = _json(text)
        report = InvestigationReport(**{key: obj[key] for key in InvestigationReport.__dataclass_fields__ if key in obj})
        report_errors = report.validate()
        if report_errors:
            return report, report_errors
        return report, []
    except Exception as exc:
        return InvestigationReport(summary="Invalid model output"), [f"{type(exc).__name__}: {exc}"]


def run_model(model_name, spec, cases, adapter_root, output_root, max_tokens):
    model_output = output_root / model_name
    model_output.mkdir(parents=True, exist_ok=True)
    trace_path = model_output / "fast_traces.jsonl"
    completed = {}
    if trace_path.exists():
        for line in trace_path.read_text().splitlines():
            if line.strip():
                trace = json.loads(line)
                completed[trace["case_id"]] = trace

    adapter_path = adapter_root / f"{model_name}_full_grounded"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter directory missing: {adapter_path}")
    backend = TransformersBackend(
        spec["model_id"],
        revision=spec["revision"],
        adapter_path=str(adapter_path),
        model_kwargs={"load_in_4bit": True, "device_map": "auto"},
    )
    mode = "a" if trace_path.exists() else "w"
    with trace_path.open(mode) as output:
        for index, case in enumerate(cases, 1):
            if case.case_id in completed:
                print(f"[{model_name}] {index}/{len(cases)} {case.case_id}: already complete", flush=True)
                continue
            started = time.perf_counter()
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(case.agent_context(), default=str)},
            ]
            response = backend.generate(messages, max_tokens=max_tokens, temperature=0.0)
            report, errors = report_from_text(response.get("text", ""))
            reference = json.loads(to_sft_record(case)["messages"][2]["content"])
            trace = {
                "trace_id": f"fast_{model_name}_{case.case_id}",
                "case_id": case.case_id,
                "model_family": model_name,
                "variant": "finetuned_fast_no_tools",
                "seed": 42,
                "tool_budget": 0,
                "tool_calls": [],
                "inference_config": {"max_tokens": max_tokens, "temperature": 0.0},
                "context": case.agent_context(),
                "ground_truth": case.ground_truth,
                "reference_report": reference,
                "report": report.__dict__,
                "schema_errors": errors,
                "latency_seconds": time.perf_counter() - started,
                "prompt_tokens": response.get("prompt_tokens", 0),
                "output_tokens": response.get("output_tokens", 0),
                "raw_response": response.get("text", ""),
            }
            output.write(json.dumps(trace, default=str) + "\n")
            output.flush()
            print(f"[{model_name}] {index}/{len(cases)} {case.case_id}: {trace['latency_seconds']:.1f}s", flush=True)

    metrics = evaluate_traces(trace_path, model_output / "metrics")
    (model_output / "metrics_summary.json").write_text(json.dumps(metrics, indent=2, default=str))
    del backend
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return metrics


def main(args):
    cases = load_cases(args.heldout, args.legitimate, args.suspicious)
    print("selected cases:", len(cases))
    print("patterns:", json.dumps({p: sum(c.ground_truth.get('pattern') == p for c in cases) for p in sorted({c.ground_truth.get('pattern') for c in cases})}, sort_keys=True))
    registry = load_registry()
    results = {}
    for model_name in ("ministral", "phi", "qwen", "gemma"):
        print(f"\n=== {model_name} ===", flush=True)
        results[model_name] = run_model(model_name, registry[model_name], cases, args.adapter_root, args.output_root, args.max_tokens)

    rows = []
    for model_name, metrics in results.items():
        row = {"model": model_name, "cases": metrics.get("cases", 0)}
        for section, values in metrics.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    if isinstance(value, (int, float, str)) or value is None:
                        row[f"{section}_{key}"] = value
        rows.append(row)
    comparison_dir = args.output_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    (comparison_dir / "model_comparison.json").write_text(json.dumps(rows, indent=2, default=str))
    with (comparison_dir / "model_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    print("comparison CSV:", comparison_dir / "model_comparison.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--legitimate", type=int, default=10)
    parser.add_argument("--suspicious", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=700)
    main(parser.parse_args())
