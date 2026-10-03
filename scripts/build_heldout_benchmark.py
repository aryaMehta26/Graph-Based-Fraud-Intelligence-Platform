"""Build a private, leakage-checked AML benchmark from the test Parquet split."""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.aml.schemas import InvestigationCase

PATTERN_MAP = {
    "fan_in": "FAN_IN",
    "fan-out": "FAN_OUT",
    "fan_out": "FAN_OUT",
    "cycle": "CYCLE",
    "stack": "STACK",
    "scatter_gather": "SCATTER_GATHER",
    "scatter-gather": "SCATTER_GATHER",
    "gather_scatter": "GATHER_SCATTER",
    "gather-scatter": "GATHER_SCATTER",
    "bipartite": "BIPARTITE",
    "random": "RANDOM",
}

CONTEXT_COLUMNS = [
    "Timestamp", "src_acct", "dst_acct", "Amount Paid", "Payment Format",
    "Receiving Currency", "hour", "is_weekend", "src_out_degree",
    "src_in_degree", "src_total_degree", "src_degree_centrality",
    "src_community_id", "src_community_size", "src_community_fraud_rate",
    "dst_out_degree", "dst_in_degree", "dst_total_degree",
    "dst_degree_centrality", "dst_community_id", "dst_community_size",
    "dst_community_fraud_rate", "Is Laundering",
]


def parse_attempts(path: Path):
    attempts, current, rows = [], None, []
    begin = re.compile(r"^BEGIN LAUNDERING ATTEMPT - ([A-Z-]+)")
    for line in path.read_text().splitlines():
        line = line.strip()
        match = begin.match(line)
        if match:
            current, rows = match.group(1), []
        elif line.startswith("END LAUNDERING ATTEMPT"):
            if current and rows:
                attempts.append({"attempt_id": len(attempts) + 1, "pattern": current, "rows": rows})
            current, rows = None, []
        elif current and line and not line.startswith("BEGIN"):
            fields = line.split(",")
            if len(fields) >= 10:
                try:
                    rows.append({
                        "timestamp": fields[0].strip(),
                        "src_acct": fields[2].strip(),
                        "dst_acct": fields[4].strip(),
                        "amount": round(float(fields[5]), 2),
                        "payment_format": fields[9].strip(),
                    })
                except ValueError:
                    pass
    return attempts


def key_from_pattern(row):
    return (row["timestamp"], row["src_acct"], row["dst_acct"], row["amount"], row["payment_format"])


def key_from_parquet(row):
    timestamp = row["Timestamp"]
    timestamp = timestamp.strftime("%Y/%m/%d %H:%M") if hasattr(timestamp, "strftime") else str(timestamp)
    return (timestamp, row["src_acct"], row["dst_acct"], round(float(row["Amount Paid"]), 2), row["Payment Format"])


def account_and_transaction_context(rows):
    accounts, transactions = {}, []
    for index, row in enumerate(rows):
        for side in ("src", "dst"):
            account_id = row[f"{side}_acct"]
            accounts[account_id] = {
                "account_id": account_id,
                "out_degree": row.get(f"{side}_out_degree"),
                "in_degree": row.get(f"{side}_in_degree"),
                "total_degree": row.get(f"{side}_total_degree"),
                "degree_centrality": row.get(f"{side}_degree_centrality"),
                "community_id": row.get(f"{side}_community_id"),
                "community_size": row.get(f"{side}_community_size"),
                "community_fraud_rate": row.get(f"{side}_community_fraud_rate"),
            }
        transactions.append({
            "txn_id": f"tx_{index}",
            "src_acct": row["src_acct"],
            "dst_acct": row["dst_acct"],
            "amount": float(row["Amount Paid"]),
            "payment_format": row["Payment Format"],
            "timestamp": row["Timestamp"].isoformat() if hasattr(row["Timestamp"], "isoformat") else str(row["Timestamp"]),
            "hour": row.get("hour"),
            "is_weekend": row.get("is_weekend"),
        })
    return {"accounts": list(accounts.values()), "transactions": transactions, "graph_stats": {"transaction_count": len(transactions)}}


def load_training_ids(paths):
    ids = set()
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                ids.add(json.loads(line)["case_id"])
    return ids


def build(args):
    attempts = parse_attempts(args.patterns)
    target_keys = {key_from_pattern(row) for attempt in attempts for row in attempt["rows"]}
    found = {}
    legitimate_rows = []
    scanner = ds.dataset(str(args.parquet), format="parquet").scanner(columns=CONTEXT_COLUMNS, batch_size=131072)
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            key = key_from_parquet(row)
            if key in target_keys and int(row.get("Is Laundering") or 0) == 1:
                found[key] = row
            elif int(row.get("Is Laundering") or 0) == 0 and len(legitimate_rows) < args.legitimate_pool:
                legitimate_rows.append(row)

    training_ids = load_training_ids(args.training_jsonl)
    suspicious_candidates = []
    for attempt in attempts:
        case_id = f"attempt_{attempt['attempt_id']}"
        if case_id in training_ids:
            continue
        matched = [found.get(key_from_pattern(row)) for row in attempt["rows"]]
        if not matched or any(row is None for row in matched):
            continue
        context = account_and_transaction_context(matched)
        suspicious_candidates.append(InvestigationCase(
            case_id=case_id,
            context={"subgraph_id": case_id, **context},
            ground_truth={"decision": "SUSPICIOUS", "pattern": PATTERN_MAP.get(attempt["pattern"].lower(), "EMERGING_UNKNOWN")},
            split="test",
        ))
    available_patterns = sorted({case.ground_truth["pattern"] for case in suspicious_candidates})
    selected_suspicious = []
    for pattern in available_patterns:
        candidate = next((case for case in suspicious_candidates if case.ground_truth["pattern"] == pattern), None)
        if candidate is not None:
            selected_suspicious.append(candidate)
    for candidate in suspicious_candidates:
        if len(selected_suspicious) >= args.suspicious_cases:
            break
        if candidate not in selected_suspicious:
            selected_suspicious.append(candidate)
    cases = selected_suspicious[:args.suspicious_cases]

    for index in range(0, min(len(legitimate_rows), args.legitimate_cases * args.transactions_per_case), args.transactions_per_case):
        rows = legitimate_rows[index:index + args.transactions_per_case]
        if len(rows) < args.transactions_per_case:
            break
        case_id = f"legitimate_{index // args.transactions_per_case + 1:03d}"
        cases.append(InvestigationCase(
            case_id=case_id,
            context={"subgraph_id": case_id, **account_and_transaction_context(rows)},
            ground_truth={"decision": "LEGITIMATE", "pattern": "NONE"},
            split="test",
        ))
        if sum(c.ground_truth["decision"] == "LEGITIMATE" for c in cases) >= args.legitimate_cases:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(case.to_jsonl() for case in cases) + ("\n" if cases else ""))
    forbidden = {"is_laundering", "pattern_type", "ground_truth", "label"}
    leaks = []
    for case in cases:
        context = json.dumps(case.agent_context()).lower()
        if any(token in context for token in forbidden):
            leaks.append(case.case_id)
    overlap = {case.case_id for case in cases} & training_ids
    print("output:", args.output)
    print("cases:", len(cases))
    print("decisions:", dict(Counter(c.ground_truth["decision"] for c in cases)))
    print("patterns:", dict(Counter(c.ground_truth["pattern"] for c in cases)))
    print("training overlap:", sorted(overlap))
    print("context leaks:", leaks)
    if not cases or overlap or leaks:
        raise SystemExit("held-out benchmark validation failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=Path("data/processed/test_graph_enriched.parquet"))
    parser.add_argument("--patterns", type=Path, default=Path("HI-Medium_Patterns.txt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/final_benchmark/heldout_cases.jsonl"))
    parser.add_argument("--training-jsonl", type=Path, action="append", default=[Path("artifacts/evaluation_dry_run/cases.jsonl")])
    parser.add_argument("--suspicious-cases", type=int, default=50)
    parser.add_argument("--legitimate-cases", type=int, default=50)
    parser.add_argument("--transactions-per-case", type=int, default=6)
    parser.add_argument("--legitimate-pool", type=int, default=10000)
    build(parser.parse_args())
