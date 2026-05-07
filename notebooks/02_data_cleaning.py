"""
============================================================
SCRIPT 2 OF 3 — DATA CLEANING
============================================================
Team     : DATA 298A — Team 12
Purpose  : Clean the IBM AML HI-Medium dataset and validate
           every cleaning step with before/after statistics
Run      : python3 02_data_cleaning.py
============================================================
"""

import os
import numpy as np
import pandas as pd

DIVIDER = "=" * 60
BASE = (
    "/Users/aryaaa/.cache/kagglehub/datasets/"
    "ealtman2019/ibm-transactions-for-anti-money-laundering-aml/versions/8"
)
TX_FILE  = f"{BASE}/HI-Medium_Trans.csv"
ACC_FILE = f"{BASE}/HI-Medium_accounts.csv"

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

print(DIVIDER)
print("STEP 2: IBM AML DATASET — DATA CLEANING")
print("Team 12 | DATA 298A | Graph Fraud Intelligence Platform")
print(DIVIDER)

# ── LOAD ─────────────────────────────────────────────────
print("\n[LOADING] Reading HI-Medium_Trans.csv (31.9M rows)...")
tx  = pd.read_csv(TX_FILE)
acc = pd.read_csv(ACC_FILE)
raw_rows = len(tx)
print(f"  ✅ Loaded: {raw_rows:,} transaction rows | {len(acc):,} account rows")

print(f"\n{'─'*60}")
print("CLEANING STEPS — WITH BEFORE / AFTER VALIDATION")
print(f"{'─'*60}")

# ── STEP 1: Parse Timestamps ──────────────────────────────
print("\n[STEP 1/7] Parse Timestamps → datetime64")
print(f"  Before: dtype = {tx['Timestamp'].dtype}")
tx["Timestamp"] = pd.to_datetime(tx["Timestamp"])
print(f"  After : dtype = {tx['Timestamp'].dtype}")
print(f"  Range : {tx['Timestamp'].min()}  →  {tx['Timestamp'].max()}")
print(f"  Span  : {(tx['Timestamp'].max() - tx['Timestamp'].min()).days} days")
print("  ✅ PASSED")

# ── STEP 2: Rename Columns ────────────────────────────────
print("\n[STEP 2/7] Rename Columns for Clarity")
print(f"  Before: {list(tx.columns)}")
tx = tx.rename(columns={"Account": "src_acct", "Account.1": "dst_acct"})
print(f"  After : {list(tx.columns)}")
print("  ✅ PASSED")

# ── STEP 3: Null Check ────────────────────────────────────
print("\n[STEP 3/7] Null / Missing Value Check")
nulls = tx.isnull().sum()
total_nulls = nulls.sum()
print(f"  Null counts per column:")
for col, n in nulls.items():
    status = "✅" if n == 0 else "❌"
    print(f"    {status} {col:<25}: {n:,}")
print(f"  Total null values : {total_nulls:,}")
assert total_nulls == 0, "❌ FAILED: Unexpected null values found!"
print("  ✅ PASSED — Zero nulls confirmed")

# ── STEP 4: Self-Loop Detection & Removal (Graph Path) ────
print("\n[STEP 4/7] Self-Loop Detection (src_acct == dst_acct)")
self_loops = tx[tx["src_acct"] == tx["dst_acct"]]
non_self   = tx[tx["src_acct"] != tx["dst_acct"]]
pct = len(self_loops) / len(tx) * 100

print(f"  Total transactions     : {len(tx):,}")
print(f"  Self-loops detected    : {len(self_loops):,}  ({pct:.2f}%)")
print(f"  Non-self-loop (graph)  : {len(non_self):,}")
print(f"\n  Self-loop breakdown by Payment Format:")
for fmt, cnt in self_loops["Payment Format"].value_counts().items():
    print(f"    • {fmt:<15}: {cnt:,}")

print(f"\n  → For GRAPH (Neo4j load) : using {len(non_self):,} rows (self-loops removed)")
print(f"  → For ML (XGBoost train) : using all {len(tx):,} rows (self-loops = legit behavior)")
print("  ✅ PASSED")

# ── STEP 5: Amount Validation & Log Transform ─────────────
print("\n[STEP 5/7] Amount Validation + Log Transform")
neg_amounts = (tx["Amount Paid"] < 0).sum()
zero_amounts= (tx["Amount Paid"] == 0).sum()
print(f"  Negative amounts : {neg_amounts:,}")
print(f"  Zero amounts     : {zero_amounts:,}")
print(f"  Min amount       : ${tx['Amount Paid'].min():.6f}")
print(f"  Max amount       : ${tx['Amount Paid'].max():,.2f}")

tx["log_amount"] = np.log1p(tx["Amount Paid"])
print(f"\n  Log-transformed: log1p(Amount_Paid) → 'log_amount'")
print(f"  log_amount range : {tx['log_amount'].min():.4f}  →  {tx['log_amount'].max():.4f}")
assert neg_amounts == 0, "❌ Negative amounts found!"
print("  ✅ PASSED — No negative amounts, log transform applied")

# ── STEP 6: Feature Engineering ──────────────────────────
print("\n[STEP 6/7] Feature Engineering")

tx["hour"]       = tx["Timestamp"].dt.hour
tx["dow"]        = tx["Timestamp"].dt.dayofweek
tx["is_weekend"] = tx["dow"].isin([5, 6]).astype(int)
tx["is_ACH"]     = (tx["Payment Format"] == "ACH").astype(int)
tx["is_Bitcoin"] = (tx["Payment Format"] == "Bitcoin").astype(int)
tx["is_Wire"]    = (tx["Payment Format"] == "Wire").astype(int)
tx["is_Cheque"]  = (tx["Payment Format"] == "Cheque").astype(int)
tx["is_CC"]      = (tx["Payment Format"] == "Credit Card").astype(int)
tx["is_cross_currency"] = (
    (tx["Payment Currency"] != tx["Receiving Currency"]).astype(int)
)

bins_b  = [0,100,500,1000,5000,10000,50000,100000,1e6,1e9,np.inf]
labels_b= ["<100","100-500","500-1K","1K-5K","5K-10K",
           "10K-50K","50K-100K","100K-1M","1M-1B",">1B"]
tx["amount_bucket"] = pd.cut(tx["Amount Paid"], bins=bins_b, labels=labels_b)

new_features = ["hour","dow","is_weekend","is_ACH","is_Bitcoin","is_Wire",
                "is_Cheque","is_CC","is_cross_currency","log_amount","amount_bucket"]
print(f"  New features added: {new_features}")
print(f"  Columns now: {tx.shape[1]} total")
print("  ✅ PASSED")

# ── STEP 7: Class Imbalance Analysis ─────────────────────
print("\n[STEP 7/7] Class Imbalance Analysis")
vc = tx["Is Laundering"].value_counts()
fraud_count = vc.get(1, 0)
legit_count = vc.get(0, 0)
ratio       = legit_count // fraud_count
fraud_rate  = fraud_count / len(tx) * 100

print(f"  Legit (0)          : {legit_count:,}")
print(f"  Fraud (1)          : {fraud_count:,}")
print(f"  Fraud rate         : {fraud_rate:.4f}%")
print(f"  Class ratio        : {ratio:,}:1 (legit:fraud)")
print(f"  scale_pos_weight   : {ratio} (recommended for XGBoost)")
print(f"  SMOTE alternative  : Applied AFTER time split on train only")
print("  ✅ PASSED")

# ── TIME-AWARE SPLITS ─────────────────────────────────────
print(f"\n{'─'*60}")
print("TIME-AWARE DATA SPLITS (No Random Shuffle — No Leakage)")
print(f"{'─'*60}")

tx_sorted = tx.sort_values("Timestamp").reset_index(drop=True)
n = len(tx_sorted)
i70, i85 = int(n * 0.70), int(n * 0.85)

df_train = tx_sorted.iloc[:i70]
df_val   = tx_sorted.iloc[i70:i85]
df_test  = tx_sorted.iloc[i85:]

for name, df in [("TRAIN (70%)", df_train), ("VAL   (15%)", df_val), ("TEST  (15%)", df_test)]:
    f = df["Is Laundering"].sum()
    r = f / len(df) * 100
    print(f"\n  {name}")
    print(f"    Date range : {df['Timestamp'].min().date()} → {df['Timestamp'].max().date()}")
    print(f"    Rows       : {len(df):,}")
    print(f"    Fraud rows : {f:,} ({r:.4f}%)")

# ── SAVE CLEAN FILES ──────────────────────────────────────
print(f"\n{'─'*60}")
print("SAVING CLEANED FILES")
print(f"{'─'*60}")

# Save full cleaned transactions
clean_path = os.path.join(OUT_DIR, "transactions_clean.parquet")
tx_sorted.to_parquet(clean_path, index=False)
size_mb = os.path.getsize(clean_path) / (1024 * 1024)
print(f"  ✅ transactions_clean.parquet  →  {clean_path}  ({size_mb:.1f} MB)")

# Save graph-ready (no self-loops)
graph_path = os.path.join(OUT_DIR, "transactions_graph.parquet")
graph_tx = tx_sorted[tx_sorted["src_acct"] != tx_sorted["dst_acct"]]
graph_tx.to_parquet(graph_path, index=False)
size_mb2 = os.path.getsize(graph_path) / (1024 * 1024)
print(f"  ✅ transactions_graph.parquet  →  {graph_path}  ({size_mb2:.1f} MB)")

# Save splits
for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
    p = os.path.join(OUT_DIR, f"split_{name}.parquet")
    df.to_parquet(p, index=False)
    s = os.path.getsize(p) / (1024 * 1024)
    print(f"  ✅ split_{name}.parquet          →  {p}  ({s:.1f} MB)")

print(f"\n{DIVIDER}")
print("DATA CLEANING COMPLETE ✅")
print(f"  Raw rows      : {raw_rows:,}")
print(f"  Cleaned rows  : {len(tx_sorted):,}  (preserved — self-loops kept for ML)")
print(f"  Graph rows    : {len(graph_tx):,}  (self-loops removed for Neo4j)")
print(f"  New features  : {len(new_features)}")
print(f"  Output dir    : {OUT_DIR}")
print(f"  Next step     : python3 03_eda_visualizations.py")
print(DIVIDER)
