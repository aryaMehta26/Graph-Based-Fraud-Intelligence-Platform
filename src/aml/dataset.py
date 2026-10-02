"""Build leakage-aware SFT/benchmark cases from subgraphs or parquet rows."""
import argparse, hashlib, json
from pathlib import Path
from typing import Iterable, List, Dict, Any
from .schemas import InvestigationCase

PATTERN_MAP = {"fan_in": "FAN_IN", "fan-in": "FAN_IN", "fan_out": "FAN_OUT", "fan-out": "FAN_OUT", "cycle": "CYCLE", "stack": "STACK", "scatter_gather": "SCATTER_GATHER", "scatter-gather": "SCATTER_GATHER", "gather_scatter": "GATHER_SCATTER", "gather-scatter": "GATHER_SCATTER", "bipartite": "BIPARTITE", "random": "RANDOM"}

def case_from_subgraph(path: Path, split="benchmark") -> InvestigationCase:
    data = json.loads(path.read_text())
    case_id = data.get("case_id") or data.get("subgraph_id") or path.stem
    labels = {"decision": "SUSPICIOUS" if any(t.get("is_laundering") for t in data.get("transactions", [])) else "LEGITIMATE", "pattern": PATTERN_MAP.get(str(data.get("pattern", "")).lower(), "EMERGING_UNKNOWN")}
    return InvestigationCase(case_id=str(case_id), context=data, ground_truth=labels, split=split)

def stable_split(case_id: str) -> str:
    bucket = int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"

def build_cases(input_dir: Path, output: Path, split_mode="case_hash") -> List[InvestigationCase]:
    cases = []
    for path in sorted(input_dir.glob("*.json")):
        case = case_from_subgraph(path)
        case.split = stable_split(case.case_id) if split_mode == "case_hash" else split_mode
        cases.append(case)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for case in cases: handle.write(case.to_jsonl() + "\n")
    return cases

def _parse_pattern_attempts(patterns_file: Path):
    import re
    attempts, current, rows = [], None, []
    begin = re.compile(r"^BEGIN LAUNDERING ATTEMPT - ([A-Z-]+)")
    for line in patterns_file.read_text().splitlines():
        line = line.strip(); match = begin.match(line)
        if match: current, rows = match.group(1), []
        elif line.startswith("END LAUNDERING ATTEMPT"):
            if current and rows: attempts.append({"attempt_id": len(attempts) + 1, "pattern_type": current, "transactions": rows})
            current, rows = None, []
        elif current and line and not line.startswith("BEGIN"):
            parts = line.split(",")
            if len(parts) >= 10:
                try: rows.append({"timestamp": parts[0].strip(), "src_acct": parts[2].strip(), "dst_acct": parts[4].strip(), "amount": float(parts[5]), "payment_format": parts[9].strip()})
                except ValueError: pass
    return attempts

def build_cases_from_processed_data(parquet_path: Path, patterns_file: Path, output: Path, limit: int = 100) -> List[InvestigationCase]:
    """Build real cases from the enriched project Parquet and pattern file."""
    try: import pyarrow.dataset as ds
    except ImportError as exc: raise RuntimeError("pyarrow is required to read processed Parquet files") from exc
    attempts = _parse_pattern_attempts(patterns_file)
    columns = ["Timestamp", "src_acct", "dst_acct", "Amount Paid", "Receiving Currency", "Payment Format", "hour", "is_weekend", "src_out_degree", "src_in_degree", "src_total_degree", "src_degree_centrality", "src_community_id", "src_community_size", "src_community_fraud_rate", "dst_out_degree", "dst_in_degree", "dst_total_degree", "dst_degree_centrality", "dst_community_id", "dst_community_size", "dst_community_fraud_rate", "Is Laundering"]
    lookup = {}
    scanner = ds.dataset(str(parquet_path), format="parquet").scanner(columns=columns, filter=ds.field("Is Laundering") == 1, batch_size=65536)
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            ts = row["Timestamp"].strftime("%Y/%m/%d %H:%M") if hasattr(row["Timestamp"], "strftime") else str(row["Timestamp"])
            lookup.setdefault((ts, row["src_acct"], row["dst_acct"], round(float(row["Amount Paid"]), 2), row["Payment Format"]), row)
    cases = []
    for attempt in attempts:
        if len(cases) >= limit: break
        matched = [lookup[(tx["timestamp"], tx["src_acct"], tx["dst_acct"], round(tx["amount"], 2), tx["payment_format"])] for tx in attempt["transactions"] if (tx["timestamp"], tx["src_acct"], tx["dst_acct"], round(tx["amount"], 2), tx["payment_format"]) in lookup]
        if not matched: continue
        accounts, transactions = {}, []
        for index, row in enumerate(matched):
            for side in ("src", "dst"):
                aid = row[f"{side}_acct"]
                accounts[aid] = {"account_id": aid, "out_degree": row[f"{side}_out_degree"], "in_degree": row[f"{side}_in_degree"], "total_degree": row[f"{side}_total_degree"], "degree_centrality": row[f"{side}_degree_centrality"], "community_id": row[f"{side}_community_id"], "community_size": row[f"{side}_community_size"], "community_fraud_rate": row[f"{side}_community_fraud_rate"]}
            transactions.append({"txn_id": f"{attempt['attempt_id']}_{index}", "src_acct": row["src_acct"], "dst_acct": row["dst_acct"], "amount": float(row["Amount Paid"]), "payment_format": row["Payment Format"], "timestamp": row["Timestamp"].isoformat(), "hour": row["hour"], "is_weekend": row["is_weekend"]})
        context = {"subgraph_id": f"attempt_{attempt['attempt_id']}", "accounts": list(accounts.values()), "transactions": transactions, "graph_stats": {"transaction_count": len(transactions)}}
        gt = {"decision": "SUSPICIOUS", "pattern": PATTERN_MAP.get(attempt["pattern_type"].lower(), "EMERGING_UNKNOWN"), "pattern_source": attempt["pattern_type"]}
        cases.append(InvestigationCase(case_id=f"attempt_{attempt['attempt_id']}", context=context, ground_truth=gt, split="benchmark"))
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text("\n".join(case.to_jsonl() for case in cases) + ("\n" if cases else ""))
    return cases

def _reference_report(case: InvestigationCase) -> Dict[str, Any]:
    """Create a grounded SFT target from visible case facts plus evaluator labels.

    Labels are used only in the assistant target. Evidence and the narrative are
    generated from the label-free context so the model has useful supervision
    without putting ground truth into its inference prompt.
    """
    context = case.agent_context()
    transactions = context.get("transactions", [])
    accounts = context.get("accounts", [])
    decision = case.ground_truth.get("decision", "SUSPICIOUS")
    pattern = case.ground_truth.get("pattern", "EMERGING_UNKNOWN")
    suspicious = decision == "SUSPICIOUS"

    evidence = [
        f"The case contains {len(transactions)} observed transaction(s) involving {len(accounts)} account(s)."
    ]
    if transactions:
        amounts = [float(tx["amount"]) for tx in transactions if tx.get("amount") is not None]
        if amounts:
            evidence.append(
                f"Observed transaction amounts range from {min(amounts):.2f} to {max(amounts):.2f}, "
                f"with a total of {sum(amounts):.2f}."
            )
        formats = sorted({str(tx["payment_format"]) for tx in transactions if tx.get("payment_format")})
        if formats:
            evidence.append(f"Payment formats observed: {', '.join(formats)}.")
        timestamps = [str(tx["timestamp"]) for tx in transactions if tx.get("timestamp")]
        if timestamps:
            evidence.append(f"Observed activity spans {min(timestamps)} through {max(timestamps)}.")

    if accounts:
        hub = max(accounts, key=lambda account: float(account.get("total_degree") or 0))
        if hub.get("account_id") is not None:
            evidence.append(
                f"Account {hub['account_id']} has total degree {hub.get('total_degree', 0)} "
                f"({hub.get('out_degree', 0)} outgoing and {hub.get('in_degree', 0)} incoming connection(s))."
            )
        communities = sorted({str(a["community_id"]) for a in accounts if a.get("community_id") is not None})
        if communities:
            evidence.append(f"The observed accounts belong to community group(s): {', '.join(communities)}.")

    if suspicious:
        actions = [
            "Escalate the case for enhanced due diligence.",
            "Review the linked accounts and transaction provenance.",
            "Document the observed pattern and supporting transaction evidence before disposition.",
        ]
        summary = (
            f"The observed transaction and graph context is consistent with a {pattern} pattern; "
            "retain the case for analyst review and corroborate the linked activity."
        )
        risk_level, confidence = "HIGH", 0.85
    else:
        actions = [
            "Document the observed activity and close the alert with rationale.",
            "Continue routine monitoring for additional anomalous activity.",
        ]
        summary = "The available transaction and graph context does not support escalation beyond routine monitoring."
        risk_level, confidence = "LOW", 0.75

    return {
        "decision": decision,
        "pattern": pattern,
        "risk_level": risk_level,
        "confidence": confidence,
        "evidence": evidence,
        "recommended_actions": actions,
        "summary": summary,
    }


def to_sft_record(case: InvestigationCase) -> dict:
    report = case.reference_report or _reference_report(case)
    return {"case_id": case.case_id, "messages": [{"role": "system", "content": "You are an AML investigator. Return the required structured JSON."}, {"role": "user", "content": json.dumps(case.agent_context(), default=str)}, {"role": "assistant", "content": json.dumps(report, default=str)}]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--input-dir", type=Path, default=Path("artifacts")); parser.add_argument("--output", type=Path, default=Path("artifacts/datasets/aml_cases.jsonl")); args = parser.parse_args()
    print(f"wrote {len(build_cases(args.input_dir, args.output))} case records to {args.output}")
