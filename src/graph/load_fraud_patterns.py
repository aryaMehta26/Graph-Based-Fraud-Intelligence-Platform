"""
DATA 298A — Graph-Based Fraud Intelligence Platform
Team 2 | San José State University

load_fraud_patterns.py
-----------------------
Parses the IBM AML HI-Medium_Patterns.txt file and loads all 2,756 ground-truth
laundering patterns into Neo4j.

What this does
--------------
1. Removes any existing stub FraudPattern nodes (manually created placeholders)
2. Creates 8 canonical FraudPattern nodes (one per type) with metadata
3. For each of the 2,756 laundering attempts in the patterns file:
   - Assigns a unique attempt_id
   - Matches the corresponding Transaction nodes already in Neo4j via
     (Account)-[:SENT]->(Transaction)-[:RECEIVED_BY]->(Account) + amount + payment_format
   - Creates PART_OF_PATTERN(attempt_id, pattern_type) on each Transaction
   - Creates INVOLVED_IN_PATTERN(attempt_id, pattern_type) on each Account

Graph relationships added
-------------------------
(:Transaction)-[:PART_OF_PATTERN]->(:FraudPattern)
(:Account)-[:INVOLVED_IN_PATTERN]->(:FraudPattern)

Prerequisites
-------------
- Neo4j running with full graph loaded (bulk_load_cypher.py complete)
- IBM dataset available via kagglehub (auto-downloaded by script 02)
- .env at project root: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

Usage
-----
    python3 src/graph/load_fraud_patterns.py

Note: teammates need Neo4j with the full graph loaded to run this.
      For ML pipeline only (scripts 05-07), this script is not required.
"""

import os
import re
import time
import logging
import kagglehub
from neo4j import GraphDatabase
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI",      "neo4j://127.0.0.1:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Aryamehta@26")
NEO4J_DB       = os.getenv("NEO4J_DB",       "neo4j")

BASE         = kagglehub.dataset_download("ealtman2019/ibm-transactions-for-anti-money-laundering-aml")
PATTERNS_FILE = os.path.join(BASE, "HI-Medium_Patterns.txt")

PATTERN_META = {
    "CYCLE":          {"description": "Funds cycle back to the originating account through a chain of transfers.", "risk_score": 0.95},
    "FAN-IN":         {"description": "Multiple accounts funnel funds into a single destination account.",          "risk_score": 0.85},
    "FAN-OUT":        {"description": "A single account disperses funds to many destination accounts.",            "risk_score": 0.85},
    "STACK":          {"description": "Sequential chain of transfers layering funds through multiple accounts.",   "risk_score": 0.80},
    "SCATTER-GATHER": {"description": "Funds dispersed then re-aggregated through an intermediary network.",       "risk_score": 0.90},
    "GATHER-SCATTER": {"description": "Funds gathered from multiple sources then redistributed.",                  "risk_score": 0.90},
    "BIPARTITE":      {"description": "Transactions form a bipartite structure between two account groups.",       "risk_score": 0.75},
    "RANDOM":         {"description": "Randomised layering pattern designed to obscure transaction trail.",        "risk_score": 0.70},
}

BATCH_SIZE = 200

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parse patterns file
# ---------------------------------------------------------------------------

def parse_patterns(filepath: str) -> list[dict]:
    """
    Returns a list of pattern attempts, each with:
      { attempt_id, pattern_type, transactions: [{from_acct, to_acct, amount, payment_format}] }
    """
    attempts = []
    current_type = None
    current_txns = []
    attempt_id   = 0

    begin_re = re.compile(r"^BEGIN LAUNDERING ATTEMPT - ([A-Z\-]+)")
    end_re   = re.compile(r"^END LAUNDERING ATTEMPT")

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m = begin_re.match(line)
            if m:
                current_type = m.group(1).strip()
                current_txns = []
                continue

            if end_re.match(line):
                if current_type and current_txns:
                    attempt_id += 1
                    attempts.append({
                        "attempt_id":   attempt_id,
                        "pattern_type": current_type,
                        "transactions": current_txns,
                    })
                current_type = None
                current_txns = []
                continue

            # Transaction line: timestamp,from_bank,from_acct,to_bank,to_acct,
            #                   amount_paid,pay_currency,amount_recv,recv_currency,format,is_laundering
            parts = line.split(",")
            if len(parts) >= 10 and current_type:
                try:
                    current_txns.append({
                        "from_acct":      parts[2].strip(),
                        "to_acct":        parts[4].strip(),
                        "amount":         float(parts[5].strip()),
                        "payment_format": parts[9].strip(),
                    })
                except (ValueError, IndexError):
                    pass

    log.info("Parsed %d laundering attempts from patterns file", len(attempts))
    return attempts


# ---------------------------------------------------------------------------
# Neo4j operations
# ---------------------------------------------------------------------------

def remove_stub_nodes(session):
    result = session.run(
        "MATCH (f:FraudPattern) DETACH DELETE f RETURN count(f) AS deleted"
    ).single()
    log.info("Removed %d existing FraudPattern stub nodes", result["deleted"])


def create_fraud_pattern_nodes(session):
    for ptype, meta in PATTERN_META.items():
        session.run("""
            MERGE (f:FraudPattern {type: $type})
            SET f.description = $description,
                f.risk_score  = $risk_score
        """, type=ptype, description=meta["description"], risk_score=meta["risk_score"])
    log.info("Created/updated %d FraudPattern nodes with metadata", len(PATTERN_META))


def link_batch(session, batch: list[dict]):
    """
    For each attempt in the batch, match existing Transaction nodes via
    (src:Account)-[:SENT]->(t:Transaction)-[:RECEIVED_BY]->(dst:Account)
    using from_acct, to_acct, amount, and payment_format, then create
    PART_OF_PATTERN and INVOLVED_IN_PATTERN relationships.
    """
    session.run("""
        UNWIND $attempts AS attempt
        MATCH (f:FraudPattern {type: attempt.pattern_type})
        WITH attempt, f
        UNWIND attempt.transactions AS txn
        MATCH (src:Account {account_id: txn.from_acct})-[:SENT]->(t:Transaction)-[:RECEIVED_BY]->(dst:Account {account_id: txn.to_acct})
        WHERE t.amount = txn.amount
          AND t.payment_format = txn.payment_format
        MERGE (t)-[:PART_OF_PATTERN {attempt_id: attempt.attempt_id, pattern_type: attempt.pattern_type}]->(f)
        MERGE (src)-[:INVOLVED_IN_PATTERN {attempt_id: attempt.attempt_id, pattern_type: attempt.pattern_type}]->(f)
        MERGE (dst)-[:INVOLVED_IN_PATTERN {attempt_id: attempt.attempt_id, pattern_type: attempt.pattern_type}]->(f)
    """, attempts=batch)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  FRAUD PATTERN LOADER — IBM AML HI-Medium_Patterns.txt")
    print("  DATA 298A | Team 2 | Graph-Based Fraud Intelligence Platform")
    print("=" * 70)

    if not os.path.exists(PATTERNS_FILE):
        raise FileNotFoundError(f"Patterns file not found: {PATTERNS_FILE}\nRun notebooks/02_data_cleaning.py first to download the dataset.")

    # Parse
    attempts = parse_patterns(PATTERNS_FILE)
    pattern_counts = {}
    for a in attempts:
        pattern_counts[a["pattern_type"]] = pattern_counts.get(a["pattern_type"], 0) + 1

    print(f"\n  Parsed {len(attempts)} laundering attempts:")
    for ptype, cnt in sorted(pattern_counts.items()):
        print(f"    {ptype:<20}: {cnt}")

    # Connect
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    log.info("Connected to Neo4j: %s", NEO4J_URI)

    with driver.session(database=NEO4J_DB) as session:
        # Step 1: remove stubs
        remove_stub_nodes(session)

        # Step 2: create canonical FraudPattern nodes with metadata
        create_fraud_pattern_nodes(session)

        # Step 3: link transactions in batches
        log.info("Linking transactions to patterns in batches of %d...", BATCH_SIZE)
        total_batches = (len(attempts) + BATCH_SIZE - 1) // BATCH_SIZE
        t0 = time.time()

        for i in range(0, len(attempts), BATCH_SIZE):
            batch = attempts[i: i + BATCH_SIZE]
            link_batch(session, batch)
            batch_num = i // BATCH_SIZE + 1
            log.info("  Batch %d/%d done (%.1fs elapsed)", batch_num, total_batches, time.time() - t0)

    driver.close()

    # Summary
    print("\n" + "=" * 70)
    print("  FRAUD PATTERN LOAD COMPLETE")
    print(f"  Attempts loaded : {len(attempts):,}")
    print(f"  Pattern types   : {len(PATTERN_META)}")
    print(f"  Elapsed         : {time.time() - t0:.1f}s")
    print("\n  Relationships created:")
    print("    (:Transaction)-[:PART_OF_PATTERN]->(:FraudPattern)")
    print("    (:Account)-[:INVOLVED_IN_PATTERN]->(:FraudPattern)")
    print("=" * 70)


if __name__ == "__main__":
    main()
