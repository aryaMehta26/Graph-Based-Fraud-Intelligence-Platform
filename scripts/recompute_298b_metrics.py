"""Recompute comparison metrics from saved traces without loading model weights."""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.benchmark_runner import run as base_run

STOPWORDS = {
    "the", "and", "with", "from", "this", "that", "case", "observed", "account",
    "accounts", "transaction", "transactions", "activity", "has", "have", "are",
    "was", "were", "for", "into", "through", "over", "total", "pattern", "shows",
}


def words(value):
    return {word for word in re.findall(r"[a-z][a-z0-9_]+", str(value).lower()) if word not in STOPWORDS and len(word) > 2}


def anchors(value):
    text = str(value)
    found = set(re.findall(r"\b\d+(?:,\d{3})*\.\d+\b", text))
    found.update(re.findall(r"\b[A-Z0-9][A-Z0-9_-]{5,}\b", text))
    found.update(re.findall(r"\b20\d{2}(?:[-/]\d{2})?(?:[-/]\d{2})?\b", text))
    return {item.lower() for item in found}


def claim_status(claim, context_text):
    claim_text = json.dumps(claim, default=str) if isinstance(claim, (dict, list)) else str(claim)
    claim_lower = claim_text.lower()
    if claim_lower in context_text:
        return "SUPPORTED"
    claim_anchors = anchors(claim_text)
    context_lower = context_text.lower()
    if claim_anchors and not all(anchor in context_lower for anchor in claim_anchors):
        return "UNSUPPORTED"
    claim_words = words(claim_text)
    if not claim_words:
        return "SUPPORTED" if not claim_anchors else "UNSUPPORTED"
    overlap = len(claim_words & words(context_text)) / len(claim_words)
    if overlap >= 0.35:
        return "SUPPORTED"
    if overlap >= 0.15:
        return "PARTIALLY_SUPPORTED"
    return "UNSUPPORTED"


def improved_faithfulness(traces):
    total = supported = partial = unsupported = hallucinating_cases = 0
    for trace in traces:
        context = json.dumps(trace.get("context", {}), default=str).lower()
        statuses = [claim_status(item, context) for item in trace.get("report", {}).get("evidence", [])]
        total += len(statuses)
        supported += statuses.count("SUPPORTED")
        partial += statuses.count("PARTIALLY_SUPPORTED")
        unsupported += statuses.count("UNSUPPORTED")
        hallucinating_cases += int(unsupported > 0)
    denominator = max(total, 1)
    return {
        "evidence_faithfulness_rate": supported / denominator,
        "supported_claim_rate": supported / denominator,
        "partially_supported_claim_rate": partial / denominator,
        "unsupported_evidence_rate": unsupported / denominator,
        "hallucination_count": unsupported,
        "evidence_items": total,
        "cases_with_hallucination": hallucinating_cases,
        "case_hallucination_rate": hallucinating_cases / max(len(traces), 1),
        "method": "anchor_and_token_overlap_v1",
    }


def optional_bertscore(traces):
    try:
        from bert_score import score
    except Exception:
        return {"bertscore": None, "bertscore_status": "install bert-score for semantic scoring"}
    predictions, references = [], []
    for trace in traces:
        predictions.append(str(trace.get("report", {}).get("summary", "")))
        references.append(str(trace.get("reference_report", {}).get("summary", "")))
    if not any(references):
        return {"bertscore": None, "bertscore_status": "no reference summaries"}
    _, _, f1 = score(predictions, references, lang="en", verbose=False)
    return {"bertscore": float(f1.mean()), "bertscore_status": "computed"}


def main(args):
    results = {}
    for model in ("ministral", "phi", "qwen", "gemma"):
        trace_path = args.input_root / model / "fast_traces.jsonl"
        if not trace_path.exists():
            raise FileNotFoundError(trace_path)
        output_dir = args.output_root / model
        metrics = base_run(trace_path, output_dir / "metrics_v2")
        traces = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
        metrics["faithfulness"] = improved_faithfulness(traces)
        metrics["generation"].update(optional_bertscore(traces))
        (output_dir / "metrics_v2.json").write_text(json.dumps(metrics, indent=2, default=str))
        row = {"model": model, "cases": metrics.get("cases", 0)}
        for section, values in metrics.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    if isinstance(value, (int, float, str)) or value is None:
                        row[f"{section}_{key}"] = value
        results[model] = row
    comparison = args.output_root / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    rows = list(results.values())
    (comparison / "model_comparison_v2.json").write_text(json.dumps(rows, indent=2, default=str))
    import csv
    with (comparison / "model_comparison_v2.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    print("saved:", comparison / "model_comparison_v2.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    main(parser.parse_args())
