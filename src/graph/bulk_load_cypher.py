"""
Bulk load 29.3M records into Neo4j using LOAD CSV (Cypher).
CSVs must already be in the Neo4j import/ folder.
"""
from neo4j import GraphDatabase
import time

URI      = "neo4j://127.0.0.1:7687"
USER     = "neo4j"
PASSWORD = "Aryamehta@26"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def run(query, db="neo4j", **kwargs):
    with driver.session(database=db) as s:
        r = s.run(query, **kwargs)
        return r.consume()

print("=" * 60)
print("BULK LOAD — 29.3M RECORDS VIA LOAD CSV")
print("=" * 60)

# ── Step 0: Wipe old sample data ─────────────────────────
print("\n[0/6] Wiping old sample data...")
t0 = time.time()
# Delete in batches to avoid memory issues
while True:
    res = run("""
        MATCH (n) WITH n LIMIT 50000
        DETACH DELETE n
        RETURN count(*) AS deleted
    """)
    summary = res
    with driver.session(database="neo4j") as s:
        result = s.run("""
            MATCH (n) WITH n LIMIT 50000
            DETACH DELETE n
            RETURN count(*) AS deleted
        """)
        record = result.single()
        deleted = record["deleted"]
        print(f"      Deleted batch: {deleted:,}")
        if deleted == 0:
            break
print(f"      ✅ Old data wiped in {time.time()-t0:.1f}s")

# ── Step 1: Create constraints & indexes ─────────────────
print("\n[1/6] Creating constraints...")
try:
    run("CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.account_id IS UNIQUE")
except:
    pass
try:
    run("CREATE CONSTRAINT txn_id IF NOT EXISTS FOR (t:Transaction) REQUIRE t.txn_id IS UNIQUE")
except:
    pass
try:
    run("CREATE INDEX txn_laundering IF NOT EXISTS FOR (t:Transaction) ON (t.is_laundering)")
except:
    pass
print("      ✅ Constraints created")

# ── Step 2: Load Account nodes ───────────────────────────
print("\n[2/6] Loading Account nodes (1.7M)...")
t1 = time.time()
run("""
    LOAD CSV WITH HEADERS FROM 'file:///nodes_accounts.csv' AS row
    CALL {
        WITH row
        CREATE (a:Account {
            account_id: row.`account_id:ID(Account)`,
            bank_id: toInteger(row.`bank_id:int`)
        })
    } IN TRANSACTIONS OF 10000 ROWS
""")
print(f"      ✅ Accounts loaded in {time.time()-t1:.1f}s")

# ── Step 3: Load Transaction nodes ───────────────────────
print("\n[3/6] Loading Transaction nodes (29.3M) — THIS WILL TAKE 15-30 MIN...")
t2 = time.time()
run("""
    LOAD CSV WITH HEADERS FROM 'file:///nodes_transactions.csv' AS row
    CALL {
        WITH row
        CREATE (t:Transaction {
            txn_id: row.`txn_id:ID(Transaction)`,
            amount: toFloat(row.`amount:float`),
            timestamp: datetime(row.`timestamp:datetime`),
            payment_format: row.payment_format,
            currency: row.currency,
            is_laundering: toInteger(row.`is_laundering:int`)
        })
    } IN TRANSACTIONS OF 10000 ROWS
""")
print(f"      ✅ Transactions loaded in {time.time()-t2:.1f}s")

# ── Step 4: Create SENT relationships ────────────────────
print("\n[4/6] Creating SENT edges (29.3M) — THIS WILL TAKE 15-30 MIN...")
t3 = time.time()
run("""
    LOAD CSV WITH HEADERS FROM 'file:///edges_sent.csv' AS row
    CALL {
        WITH row
        MATCH (a:Account {account_id: row.`:START_ID(Account)`})
        MATCH (t:Transaction {txn_id: row.`:END_ID(Transaction)`})
        CREATE (a)-[:SENT]->(t)
    } IN TRANSACTIONS OF 5000 ROWS
""")
print(f"      ✅ SENT edges loaded in {time.time()-t3:.1f}s")

# ── Step 5: Create RECEIVED_BY relationships ─────────────
print("\n[5/6] Creating RECEIVED_BY edges (29.3M) — THIS WILL TAKE 15-30 MIN...")
t4 = time.time()
run("""
    LOAD CSV WITH HEADERS FROM 'file:///edges_received.csv' AS row
    CALL {
        WITH row
        MATCH (t:Transaction {txn_id: row.`:START_ID(Transaction)`})
        MATCH (a:Account {account_id: row.`:END_ID(Account)`})
        CREATE (t)-[:RECEIVED_BY]->(a)
    } IN TRANSACTIONS OF 5000 ROWS
""")
print(f"      ✅ RECEIVED_BY edges loaded in {time.time()-t4:.1f}s")

# ── Step 6: Verify ───────────────────────────────────────
print("\n[6/6] Verifying counts...")
with driver.session(database="neo4j") as s:
    acct = s.run("MATCH (a:Account) RETURN count(a) AS c").single()["c"]
    txns = s.run("MATCH (t:Transaction) RETURN count(t) AS c").single()["c"]
    sent = s.run("MATCH ()-[r:SENT]->() RETURN count(r) AS c").single()["c"]
    recv = s.run("MATCH ()-[r:RECEIVED_BY]->() RETURN count(r) AS c").single()["c"]
    fraud = s.run("MATCH (t:Transaction {is_laundering: 1}) RETURN count(t) AS c").single()["c"]

total_time = time.time() - t0
print(f"""
{'='*60}
✅ BULK LOAD COMPLETE!
{'='*60}
  Account nodes:       {acct:>12,}
  Transaction nodes:   {txns:>12,}
  SENT edges:          {sent:>12,}
  RECEIVED_BY edges:   {recv:>12,}
  Fraud transactions:  {fraud:>12,}
  Total time:          {total_time/60:.1f} minutes
{'='*60}
""")

driver.close()
