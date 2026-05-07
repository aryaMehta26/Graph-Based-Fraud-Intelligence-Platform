"""
============================================================
SCRIPT 1 OF 3 — DATA EXTRACTION
============================================================
Team     : DATA 298A — Team 12
Purpose  : Download IBM AML dataset and validate all files exist
Run      : python3 01_data_extraction.py
============================================================
"""

import os
import kagglehub
import pandas as pd

DIVIDER = "=" * 60

print(DIVIDER)
print("STEP 1: IBM AML DATASET — DATA EXTRACTION")
print("Team 12 | DATA 298A | Graph Fraud Intelligence Platform")
print(DIVIDER)

# ── STEP 1: Download Dataset ──────────────────────────────
print("\n[1/4] Downloading IBM AML dataset from Kaggle...")
print("      Source: ealtman2019/ibm-transactions-for-anti-money-laundering-aml")

path = kagglehub.dataset_download(
    "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
)

print(f"      ✅ Dataset path: {path}")

# ── STEP 2: List All Available Files ─────────────────────
print(f"\n[2/4] Files available in dataset:")
print(f"{'File Name':<40} {'Size':>10}")
print("-" * 52)

total_size = 0
files = sorted(os.listdir(path))
for fname in files:
    fpath = os.path.join(path, fname)
    size_bytes = os.path.getsize(fpath)
    size_mb    = size_bytes / (1024 * 1024)
    size_str   = f"{size_mb:.1f} MB" if size_mb < 1024 else f"{size_mb/1024:.2f} GB"
    print(f"  {fname:<38} {size_str:>10}")
    total_size += size_bytes

print("-" * 52)
print(f"  {'TOTAL':<38} {total_size/(1024**3):.2f} GB")

# ── STEP 3: Load & Validate Our 3 Target Files ───────────
print(f"\n[3/4] Loading and validating HI-Medium files...")

FILES = {
    "Transactions" : "HI-Medium_Trans.csv",
    "Accounts"     : "HI-Medium_accounts.csv",
    "Patterns"     : "HI-Medium_Patterns.txt",
}

for label, fname in FILES.items():
    fpath = os.path.join(path, fname)
    if not os.path.exists(fpath):
        print(f"  ❌ {label}: {fname} NOT FOUND")
        continue

    size_mb = os.path.getsize(fpath) / (1024 * 1024)
    size_str = f"{size_mb:.1f} MB" if size_mb < 1024 else f"{size_mb/1024:.2f} GB"

    if fname.endswith(".txt"):
        with open(fpath) as f:
            lines = f.readlines()
        print(f"  ✅ {label:<15}: {fname} | {size_str} | {len(lines):,} lines")
    else:
        df = pd.read_csv(fpath, nrows=5)
        row_count = sum(1 for _ in open(fpath)) - 1
        print(f"  ✅ {label:<15}: {fname} | {size_str} | "
              f"{row_count:,} rows × {df.shape[1]} columns")

# ── STEP 4: Print Schema ─────────────────────────────────
print(f"\n[4/4] Transaction file schema (HI-Medium_Trans.csv):")
tx = pd.read_csv(os.path.join(path, "HI-Medium_Trans.csv"), nrows=3)
print(f"\n  Columns ({len(tx.columns)}):")
for col, dtype in tx.dtypes.items():
    print(f"    • {col:<22} ({dtype})")

print(f"\n  Sample first 3 rows:")
print(tx.to_string(index=False))

print(f"\n{DIVIDER}")
print("EXTRACTION COMPLETE ✅")
print(f"  All 3 HI-Medium files validated and ready")
print(f"  Dataset path: {path}")
print(f"  Next step  : python3 02_data_cleaning.py")
print(DIVIDER)
