"""
DATA 298A — Graph-Based Fraud Intelligence Platform
Team 2 | San José State University

src/llm/build_subgraph.py
--------------------------
Extracts real subgraphs from the IBM AML test set for use with
src/llm/investigator.py.

No Neo4j required — reads test_graph_enriched.parquet which already
has degree and community features joined onto every transaction row.

Selects one focal account per fraud pattern type (Fan-In, Fan-Out,
Layering), builds a subgraph JSON around it, and saves to artifacts/.

Usage
-----
    python3 src/llm/build_subgraph.py
    python3 src/llm/build_subgraph.py --max-txns 15 --out-dir artifacts

Then pipe into investigator:
    python3 src/llm/investigator.py --variant v2 \
        --input artifacts/real_fan_in_subgraph.json
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

# ── config ────────────────────────────────────────────────────────────────────
PROJ     = Path(__file__).resolve().parents[2]
DATA     = PROJ / "data" / "processed"
OUT_DIR  = PROJ / "artifacts"

MAX_ACCOUNTS = 12   # max accounts to include per subgraph
MAX_TXNS     = 20   # max transactions to include per subgraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def get_account_features(acct: str, df: pd.DataFrame) -> dict:
    """
    Pull degree + community features for one account from the enriched parquet.
    The parquet has src_ and dst_ prefixed columns — we try src_ first.
    """
    src_rows = df[df["src_acct"] == acct]
    dst_rows = df[df["dst_acct"] == acct]

    if len(src_rows):
        r = src_rows.iloc[0]
        prefix = "src"
    elif len(dst_rows):
        r = dst_rows.iloc[0]
        prefix = "dst"
    else:
        return {}

    return {
        "out_degree":           int(r[f"{prefix}_out_degree"]),
        "in_degree":            int(r[f"{prefix}_in_degree"]),
        "total_degree":         int(r[f"{prefix}_total_degree"]),
        "degree_centrality":    round(float(r[f"{prefix}_degree_centrality"]), 7),
        "community_id":         int(r[f"{prefix}_community_id"]),
        "community_size":       int(r[f"{prefix}_community_size"]),
        "community_fraud_rate": round(float(r[f"{prefix}_community_fraud_rate"]), 6),
    }


def detect_pattern(focal: str, connected: pd.DataFrame) -> str:
    """Heuristic pattern label from subgraph topology."""
    fraud = connected[connected["Is Laundering"] == 1]
    inflows  = fraud[fraud["dst_acct"] == focal]
    outflows = fraud[fraud["src_acct"] == focal]
    n_senders   = inflows["src_acct"].nunique()
    n_receivers = outflows["dst_acct"].nunique()

    if n_senders >= 3 and n_receivers <= 1:
        # many → one: classic aggregation / smurfing
        amounts = inflows["Amount Paid"]
        if (amounts < 10_000).all() and amounts.max() > 8_000:
            return "Smurfing"
        return "Fan-In"
    if n_receivers >= 3 and n_senders <= 1:
        return "Fan-Out"
    if n_senders >= 2 and n_receivers >= 2:
        return "Layering Ring"
    if n_senders >= 2:
        return "Fan-In"
    if n_receivers >= 2:
        return "Fan-Out"
    return "Unknown"


def assign_role(acct: str, focal: str, connected: pd.DataFrame) -> str:
    fraud = connected[connected["Is Laundering"] == 1]
    sends_fraud    = (fraud["src_acct"] == acct).any()
    receives_fraud = (fraud["dst_acct"] == acct).any()

    if acct == focal:
        return "focal_account"
    if sends_fraud and receives_fraud:
        return "layering_node"
    if receives_fraud:
        return "feeder"
    if sends_fraud:
        return "distributor"
    return "peripheral"


def build_subgraph(focal: str, df: pd.DataFrame, subgraph_id: str) -> dict:
    """Build the subgraph JSON for one focal account."""
    connected = df[
        (df["src_acct"] == focal) | (df["dst_acct"] == focal)
    ].copy().sort_values("Timestamp")

    fraud_rows = connected[connected["Is Laundering"] == 1]
    legit_rows = connected[connected["Is Laundering"] == 0]

    pattern = detect_pattern(focal, connected)

    # ── accounts ──────────────────────────────────────────────────────────────
    all_accts = pd.concat(
        [connected["src_acct"], connected["dst_acct"]]
    ).unique().tolist()

    # focal first, then sorted by how often they appear (most connected first)
    freq = pd.concat(
        [connected["src_acct"], connected["dst_acct"]]
    ).value_counts()
    ordered = [focal] + [
        a for a in freq.index if a != focal
    ][: MAX_ACCOUNTS - 1]

    accounts = []
    for acct in ordered:
        feats = get_account_features(acct, df)
        if not feats:
            continue
        bank_rows = connected[connected["src_acct"] == acct]
        bank_id = int(bank_rows["From Bank"].iloc[0]) if len(bank_rows) else None
        accounts.append({
            "account_id":           acct,
            "bank_id":              bank_id,
            "role":                 assign_role(acct, focal, connected),
            **feats,
        })

    # ── transactions ──────────────────────────────────────────────────────────
    # prioritise fraud transactions, pad with legit up to MAX_TXNS
    n_fraud = min(len(fraud_rows), MAX_TXNS)
    n_legit = min(len(legit_rows), MAX_TXNS - n_fraud)
    txn_df  = pd.concat(
        [fraud_rows.head(n_fraud), legit_rows.head(n_legit)]
    ).sort_values("Timestamp")

    transactions = []
    for i, (_, row) in enumerate(txn_df.iterrows(), start=1):
        transactions.append({
            "txn_id":         f"T_{i:03d}",
            "src_acct":       row["src_acct"],
            "dst_acct":       row["dst_acct"],
            "from_bank":      int(row["From Bank"]),
            "to_bank":        int(row["To Bank"]),
            "amount":         round(float(row["Amount Paid"]), 2),
            "payment_format": row["Payment Format"],
            "timestamp":      str(row["Timestamp"]),
            "hour":           int(row["hour"]),
            "is_laundering":  int(row["Is Laundering"]),
        })

    # ── stats ─────────────────────────────────────────────────────────────────
    dominant_fmt = connected["Payment Format"].mode()
    focal_community = accounts[0]["community_id"] if accounts else -1

    graph_stats = {
        "total_fraud_amount":       round(float(fraud_rows["Amount Paid"].sum()), 2),
        "total_legit_amount":       round(float(legit_rows["Amount Paid"].sum()), 2),
        "fraud_transaction_count":  int(len(fraud_rows)),
        "total_transaction_count":  int(len(connected)),
        "unique_accounts":          int(len(all_accts)),
        "dominant_payment_format":  dominant_fmt[0] if len(dominant_fmt) else "Unknown",
        "time_span_hours":          round(
            (connected["Timestamp"].max() - connected["Timestamp"].min())
            .total_seconds() / 3600, 1
        ) if len(connected) > 1 else 0.0,
        "focal_community_id":       focal_community,
    }

    return {
        "subgraph_id":  subgraph_id,
        "focal_account": focal,
        "pattern_hint": pattern,
        "flagged_by":   "ground_truth_is_laundering=1",
        "accounts":     accounts,
        "transactions": transactions,
        "graph_stats":  graph_stats,
    }


# ── candidate selection ───────────────────────────────────────────────────────

def find_candidates(df: pd.DataFrame) -> dict:
    """
    Return one best focal account per pattern type.
    Criteria: enough fraud transactions, not too many total (keeps JSON small).
    """
    fraud = df[df["Is Laundering"] == 1]

    # Fan-In: dst_acct receives from many distinct fraud senders
    fan_in = (
        fraud.groupby("dst_acct")["src_acct"]
        .nunique()
        .sort_values(ascending=False)
    )

    # Fan-Out: src_acct sends to many distinct fraud receivers
    fan_out = (
        fraud.groupby("src_acct")["dst_acct"]
        .nunique()
        .sort_values(ascending=False)
    )

    # Layering: accounts that both send AND receive fraud
    fraud_senders   = set(fraud["src_acct"])
    fraud_receivers = set(fraud["dst_acct"])
    both = fraud_senders & fraud_receivers
    layering = fraud[fraud["src_acct"].isin(both)]["src_acct"].value_counts()

    def pick(series: pd.Series, lo=3, hi=60):
        """Pick first account whose total connected transactions is in [lo, hi]."""
        for acct in series.index:
            n = len(df[(df["src_acct"] == acct) | (df["dst_acct"] == acct)])
            if lo <= n <= hi:
                return acct
        return None

    return {
        "fan_in":   pick(fan_in),
        "fan_out":  pick(fan_out),
        "layering": pick(layering),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build real subgraphs from IBM AML test set"
    )
    parser.add_argument(
        "--max-txns", type=int, default=MAX_TXNS,
        help=f"Max transactions per subgraph (default: {MAX_TXNS})"
    )
    parser.add_argument(
        "--out-dir", type=str, default=str(OUT_DIR),
        help=f"Output directory (default: {OUT_DIR})"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    parquet_path = DATA / "test_graph_enriched.parquet"
    log.info("Loading %s ...", parquet_path)
    df = pd.read_parquet(parquet_path)

    fraud = df[df["Is Laundering"] == 1]
    log.info(
        "Test set: %s rows | %s fraud | %s unique fraud accounts",
        f"{len(df):,}", f"{len(fraud):,}",
        f"{pd.concat([fraud['src_acct'], fraud['dst_acct']]).nunique():,}",
    )

    # ── find candidates ───────────────────────────────────────────────────────
    log.info("Selecting focal accounts per pattern type ...")
    candidates = find_candidates(df)

    built = []
    for pattern_name, focal in candidates.items():
        if focal is None:
            log.warning("No suitable candidate found for pattern: %s", pattern_name)
            continue

        subgraph_id = f"real_{pattern_name}_{focal}"
        log.info("Building subgraph: %s (focal: %s)", pattern_name, focal)

        sg = build_subgraph(focal, df, subgraph_id)

        out_path = out_dir / f"real_{pattern_name}_subgraph.json"
        with open(out_path, "w") as f:
            json.dump(sg, f, indent=2)

        json_str = json.dumps(sg)
        log.info(
            "  %-10s | accounts: %2d | txns: %2d (fraud: %d) | "
            "fraud $%.0f | ~%d tokens | saved: %s",
            pattern_name.upper(),
            len(sg["accounts"]),
            len(sg["transactions"]),
            sg["graph_stats"]["fraud_transaction_count"],
            sg["graph_stats"]["total_fraud_amount"],
            len(json_str) // 4,
            out_path.name,
        )
        built.append((pattern_name, out_path))

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SUBGRAPHS BUILT — ready for LLM investigator")
    print("=" * 65)
    for pattern_name, out_path in built:
        print(f"\n  {pattern_name.upper()}")
        print(f"    python3 src/llm/investigator.py --variant v2 \\")
        print(f"        --input {out_path}")
    print("\n  Or run all variants on all subgraphs:")
    for pattern_name, out_path in built:
        print(
            f"    python3 src/llm/investigator.py --variant all "
            f"--input {out_path}"
        )
    print("=" * 65)


if __name__ == "__main__":
    main()
