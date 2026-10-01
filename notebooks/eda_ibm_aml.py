"""
============================================================
IBM AML Dataset — Exploratory Data Analysis (EDA)
============================================================
Project  : Palantir-Inspired Graph Fraud Intelligence Platform
Team     : DATA 298A — Team 12
Dataset  : IBM Transactions for Anti-Money Laundering (AML)
Source   : kagglehub.dataset_download("ealtman2019/ibm-transactions-for-anti-money-laundering-aml")
File     : HI-Small_Trans.csv + HI-Small_accounts.csv
Charts   : Saved to notebooks/eda_charts/
============================================================
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
from pathlib import Path

# ── 0. PATHS & OUTPUT DIR ────────────────────────────────
BASE = os.getenv("AML_DATASET_DIR", str(Path.home() / ".cache/kagglehub/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/versions/8"))
TX_FILE  = f"{BASE}/HI-Small_Trans.csv"
ACC_FILE = f"{BASE}/HI-Small_accounts.csv"
PAT_FILE = f"{BASE}/HI-Small_Patterns.txt"

OUT_DIR = os.path.join(os.path.dirname(__file__), "eda_charts")
os.makedirs(OUT_DIR, exist_ok=True)

# ── STYLE ────────────────────────────────────────────────
PALETTE   = {"fraud": "#E05C5C", "legit": "#4C9BE8", "neutral": "#6C8EBF"}
plt.rcParams.update({
    "figure.facecolor": "#0F1117",
    "axes.facecolor":   "#1A1D27",
    "axes.edgecolor":   "#3A3D4D",
    "axes.labelcolor":  "#CCCCCC",
    "text.color":       "#CCCCCC",
    "xtick.color":      "#AAAAAA",
    "ytick.color":      "#AAAAAA",
    "grid.color":       "#2E3145",
    "grid.linestyle":   "--",
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
})

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

# ── LOAD ─────────────────────────────────────────────────
print("Loading data …")
tx  = pd.read_csv(TX_FILE)
acc = pd.read_csv(ACC_FILE)
tx["Timestamp"] = pd.to_datetime(tx["Timestamp"])
fraud = tx[tx["Is Laundering"] == 1]
legit = tx[tx["Is Laundering"] == 0]
print(f"  Transactions : {len(tx):,} rows")
print(f"  Accounts     : {len(acc):,} rows")

# ════════════════════════════════════════════════════════
# SECTION 1 — BASIC SHAPE & SCHEMA
# ════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 1: BASIC SHAPE & SCHEMA")
print("=" * 60)
print(f"\nTransactions shape : {tx.shape}")
print(f"Accounts shape     : {acc.shape}")
print(f"\nTransaction columns:\n  {list(tx.columns)}")
print(f"\nAccounts columns:\n  {list(acc.columns)}")
print("\nTransaction dtypes:")
print(tx.dtypes.to_string())
print("\nFirst 3 transaction rows:")
print(tx.head(3).to_string())
print("\nFirst 3 account rows:")
print(acc.head(3).to_string())

# ════════════════════════════════════════════════════════
# SECTION 2 — NULL / MISSING VALUES
# ════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 2: NULL / MISSING VALUES")
print("=" * 60)
tx_nulls  = tx.isnull().sum()
acc_nulls = acc.isnull().sum()
print("\nTransaction nulls:")
print(tx_nulls.to_string())
print("\nAccount nulls:")
print(acc_nulls.to_string())

# ── Chart 1: Null heatmap ────────────────────────────────
print("\nGenerating Chart 1 — Null heatmap …")
fig, ax = plt.subplots(figsize=(10, 3))
null_data = tx_nulls.to_frame(name="Null Count").T
im = ax.imshow([[0] * len(tx_nulls)], aspect="auto", cmap="RdYlGn_r",
               vmin=0, vmax=1)
for j, (col, val) in enumerate(tx_nulls.items()):
    color = "#2ECC71" if val == 0 else "#E74C3C"
    ax.text(j, 0, f"{col}\n{val}", ha="center", va="center",
            fontsize=7.5, color=color, fontweight="bold")
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Missing Values per Column (Green = 0 Nulls ✅)", pad=10)
save(fig, "01_null_heatmap.png")

# ════════════════════════════════════════════════════════
# SECTION 3 — TIME RANGE
# ════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 3: TIME RANGE")
print("=" * 60)
print(f"\nEarliest : {tx['Timestamp'].min()}")
print(f"Latest   : {tx['Timestamp'].max()}")
print(f"Span     : {(tx['Timestamp'].max() - tx['Timestamp'].min()).days} days")

# ── Chart 2: Transactions over time (daily) ──────────────
print("Generating Chart 2 — Daily transaction volume …")
tx["Date"] = tx["Timestamp"].dt.date
daily = tx.groupby("Date").size()
daily_fraud = fraud.groupby(fraud["Timestamp"].dt.date).size().reindex(daily.index, fill_value=0)

fig, ax = plt.subplots(figsize=(12, 4))
ax.fill_between(range(len(daily)), daily.values, alpha=0.4,
                color=PALETTE["legit"], label="All Txns")
ax.fill_between(range(len(daily_fraud)), daily_fraud.values * 100, alpha=0.8,
                color=PALETTE["fraud"], label="Fraud ×100 (scaled)")
ax.set_xticks(range(len(daily)))
ax.set_xticklabels([str(d) for d in daily.index], rotation=45, ha="right", fontsize=7)
ax.set_ylabel("Transaction Count")
ax.set_title("Daily Transaction Volume — All vs Fraud (×100 scaled)")
ax.legend(facecolor="#1A1D27", edgecolor="#3A3D4D")
ax.grid(True)
save(fig, "02_daily_volume.png")

# ════════════════════════════════════════════════════════
# SECTION 4 — FRAUD LABEL DISTRIBUTION
# ════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 4: FRAUD LABEL DISTRIBUTION")
print("=" * 60)
vc = tx["Is Laundering"].value_counts()
fraud_rate = vc.get(1, 0) / len(tx) * 100
print(f"\nLegit (0) : {vc.get(0,0):,}")
print(f"Fraud (1) : {vc.get(1,0):,}")
print(f"Fraud rate: {fraud_rate:.4f}%")
print(f"Class ratio (legit:fraud): {vc.get(0,0)//vc.get(1,1):,}:1")

# ── Chart 3: Class imbalance donut ──────────────────────
print("Generating Chart 3 — Class imbalance donut …")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Donut
sizes  = [vc.get(0, 0), vc.get(1, 0)]
colors = [PALETTE["legit"], PALETTE["fraud"]]
wedges, texts, autotexts = ax1.pie(
    sizes, labels=["Legit", "Fraud"],
    colors=colors, autopct="%1.3f%%",
    startangle=90, pctdistance=0.75,
    wedgeprops=dict(width=0.5, edgecolor="#0F1117")
)
for t in autotexts:
    t.set_color("white"); t.set_fontsize(10)
ax1.set_title("Class Distribution (Donut)")

# Bar
bars = ax2.bar(["Legit", "Fraud"], sizes, color=colors, width=0.4, edgecolor="#0F1117")
for bar, val in zip(bars, sizes):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20000,
             f"{val:,}", ha="center", va="bottom", fontsize=10, color="white")
ax2.set_yscale("log")
ax2.set_ylabel("Count (log scale)")
ax2.set_title("Log-Scale Count (979:1 Imbalance)")
ax2.grid(True, axis="y")

fig.suptitle("Fraud vs Legit — Extreme Class Imbalance (0.10% Fraud)", y=1.02, fontsize=13)
save(fig, "03_class_imbalance.png")

# ════════════════════════════════════════════════════════
# SECTION 5 — AMOUNT STATISTICS
# ════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 5: AMOUNT STATISTICS")
print("=" * 60)
print("\nAll transactions — Amount Paid:")
print(tx["Amount Paid"].describe().to_string())
print("\nFraud transactions — Amount Paid:")
print(fraud["Amount Paid"].describe().to_string())
print("\nLegit transactions — Amount Paid:")
print(legit["Amount Paid"].describe().to_string())

# ── Chart 4: Amount distributions (log scale) ────────────
print("Generating Chart 4 — Amount distributions …")
import numpy as np
fig, ax = plt.subplots(figsize=(10, 5))
bins = np.logspace(-2, 12, 80)
ax.hist(legit["Amount Paid"].clip(upper=1e12), bins=bins, alpha=0.6,
        color=PALETTE["legit"], label=f"Legit (n={len(legit):,})")
ax.hist(fraud["Amount Paid"].clip(upper=1e12), bins=bins, alpha=0.8,
        color=PALETTE["fraud"], label=f"Fraud (n={len(fraud):,})")
ax.set_xscale("log")
ax.set_xlabel("Amount Paid (USD, log scale)")
ax.set_ylabel("Transaction Count")
ax.set_title("Amount Distribution — Fraud vs Legit (Log Scale)")
ax.legend(facecolor="#1A1D27", edgecolor="#3A3D4D")
ax.grid(True, axis="x")
ax.axvline(fraud["Amount Paid"].median(), color=PALETTE["fraud"],
           linestyle="--", linewidth=1.5, label="Fraud Median")
ax.axvline(legit["Amount Paid"].median(), color=PALETTE["legit"],
           linestyle="--", linewidth=1.5, label="Legit Median")
ax.legend(facecolor="#1A1D27", edgecolor="#3A3D4D")
save(fig, "04_amount_distribution.png")

# ── Chart 5: Box plots (log scale) ──────────────────────
print("Generating Chart 5 — Box plots fraud vs legit …")
fig, ax = plt.subplots(figsize=(7, 5))
data_to_plot = [
    legit["Amount Paid"].clip(upper=1e9).values,
    fraud["Amount Paid"].clip(upper=1e9).values
]
bp = ax.boxplot(data_to_plot, patch_artist=True, notch=True,
                medianprops=dict(color="white", linewidth=2))
bp["boxes"][0].set_facecolor(PALETTE["legit"])
bp["boxes"][1].set_facecolor(PALETTE["fraud"])
ax.set_yscale("log")
ax.set_xticklabels(["Legit", "Fraud"])
ax.set_ylabel("Amount Paid (log scale, clipped at $1B)")
ax.set_title("Transaction Amount — Fraud vs Legit (Box Plot)")
ax.grid(True, axis="y")
save(fig, "05_amount_boxplot.png")

# ════════════════════════════════════════════════════════
# SECTION 6 — PAYMENT FORMAT BREAKDOWN
# ════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 6: PAYMENT FORMAT BREAKDOWN")
print("=" * 60)
fmt_all   = tx["Payment Format"].value_counts()
fmt_fraud = fraud["Payment Format"].value_counts().reindex(fmt_all.index, fill_value=0)
print("\nAll transactions:")
print(fmt_all.to_string())
print("\nFraud only:")
print(fmt_fraud.dropna().to_string())

# ── Chart 6: Grouped bar — format vs fraud ──────────────
print("Generating Chart 6 — Payment format vs fraud …")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
x = range(len(fmt_all))
ax1.bar(x, fmt_all.values, color=PALETTE["legit"], alpha=0.7, label="All")
ax1.set_xticks(list(x))
ax1.set_xticklabels(fmt_all.index, rotation=30, ha="right")
ax1.set_ylabel("Transaction Count")
ax1.set_title("All Transactions by Payment Format")
ax1.grid(True, axis="y")

colors_fmt = [PALETTE["fraud"] if v > 100 else PALETTE["neutral"]
              for v in fmt_fraud.values]
ax2.bar(x, fmt_fraud.values, color=colors_fmt, alpha=0.85)
ax2.set_xticks(list(x))
ax2.set_xticklabels(fmt_fraud.index, rotation=30, ha="right")
ax2.set_ylabel("Fraud Count")
ax2.set_title("Fraud Transactions by Payment Format\n(ACH = 87% of all fraud!)")
ax2.grid(True, axis="y")
for i, v in enumerate(fmt_fraud.values):
    if v > 0:
        ax2.text(i, v + 30, str(int(v)), ha="center", fontsize=9, color="white")

fig.suptitle("Payment Format Analysis", fontsize=14)
save(fig, "06_payment_format.png")

# ════════════════════════════════════════════════════════
# SECTION 7 — CURRENCY BREAKDOWN
# ════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 7: CURRENCY BREAKDOWN (Receiving)")
print("=" * 60)
curr = tx["Receiving Currency"].value_counts()
print(curr.to_string())

# ── Chart 7: Currency breakdown (horizontal bar) ────────
print("Generating Chart 7 — Currency breakdown …")
fig, ax = plt.subplots(figsize=(8, 6))
colors_curr = [PALETTE["fraud"] if c == "Bitcoin" else PALETTE["legit"]
               for c in curr.index]
bars = ax.barh(curr.index[::-1], curr.values[::-1], color=colors_curr[::-1], alpha=0.8)
ax.set_xlabel("Transaction Count")
ax.set_title("Transaction Volume by Receiving Currency")
ax.grid(True, axis="x")
for bar, val in zip(bars, curr.values[::-1]):
    ax.text(bar.get_width() + 5000, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", fontsize=8)
save(fig, "07_currency.png")

# ════════════════════════════════════════════════════════
# SECTION 8 — GRAPH EDGE STRUCTURE
# ════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 8: GRAPH EDGE STRUCTURE")
print("=" * 60)
print(f"\nUnique source accounts (Account)   : {tx['Account'].nunique():,}")
print(f"Unique dest accounts   (Account.1) : {tx['Account.1'].nunique():,}")
all_accts = pd.concat([tx["Account"], tx["Account.1"]]).unique()
print(f"Total unique accounts (combined)   : {len(all_accts):,}")
print(f"Unique source banks (From Bank)    : {tx['From Bank'].nunique():,}")
print(f"Unique dest banks   (To Bank)      : {tx['To Bank'].nunique():,}")

self_loops = tx[tx["Account"] == tx["Account.1"]]
print(f"\nSelf-loop transactions (src==dst)  : {len(self_loops):,}")
print(f"Self-loop formats:")
print(self_loops["Payment Format"].value_counts().to_string())

# Degree distribution (top 50 accounts by out-degree)
out_degree = tx.groupby("Account").size().sort_values(ascending=False)
in_degree  = tx.groupby("Account.1").size().sort_values(ascending=False)

print(f"\nTop 5 senders (out-degree):")
print(out_degree.head(5).to_string())
print(f"\nTop 5 receivers (in-degree, potential funnels):")
print(in_degree.head(5).to_string())

# ── Chart 8: Degree distribution ────────────────────────
print("Generating Chart 8 — Degree distribution …")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

out_vals = out_degree[out_degree <= 500].values
in_vals  = in_degree[in_degree <= 500].values
ax1.hist(out_vals, bins=60, color=PALETTE["legit"], alpha=0.8, edgecolor="#0F1117")
ax1.set_xlabel("Out-Degree (# of transactions sent)")
ax1.set_ylabel("Number of Accounts")
ax1.set_title("Out-Degree Distribution\n(clipped at 500)")
ax1.grid(True, axis="y")

ax2.hist(in_vals, bins=60, color=PALETTE["neutral"], alpha=0.8, edgecolor="#0F1117")
ax2.set_xlabel("In-Degree (# of transactions received)")
ax2.set_ylabel("Number of Accounts")
ax2.set_title("In-Degree Distribution\n(clipped at 500 — high spikes = funnel suspects)")
ax2.grid(True, axis="y")

fig.suptitle("Account Degree Distribution (Graph Structure)", fontsize=13)
save(fig, "08_degree_distribution.png")

# ════════════════════════════════════════════════════════
# SECTION 9 — CROSS-CURRENCY
# ════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 9: CROSS-CURRENCY TRANSACTIONS")
print("=" * 60)
cross = tx[tx["Payment Currency"] != tx["Receiving Currency"]]
print(f"\nCross-currency txns : {len(cross):,} ({len(cross)/len(tx)*100:.2f}%)")
print(f"Cross-currency fraud: {cross['Is Laundering'].sum():,}")

# ════════════════════════════════════════════════════════
# SECTION 10 — PATTERNS FILE (GROUND TRUTH)
# ════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 10: PATTERNS FILE — GROUND TRUTH RING LABELS")
print("=" * 60)
with open(PAT_FILE) as f:
    pat_text = f.read()
ring_types_raw = [l for l in pat_text.splitlines() if l.startswith("BEGIN")]
# Parse ring type name (e.g. FAN-OUT, CYCLE, etc.)
import re
ring_names = []
for r in ring_types_raw:
    m = re.search(r"ATTEMPT - ([A-Z\-]+)", r)
    if m:
        ring_names.append(m.group(1))
from collections import Counter
ring_counts = Counter(ring_names)
print(f"\nTotal labeled ring patterns: {len(ring_types_raw)}")
print("Ring type breakdown:")
for k, v in ring_counts.most_common():
    print(f"  {k:20s}: {v:4d}")

# ── Chart 9: Ring type breakdown ────────────────────────
print("Generating Chart 9 — Ring type breakdown …")
fig, ax = plt.subplots(figsize=(8, 5))
rkeys   = list(ring_counts.keys())
rvals   = list(ring_counts.values())
colors9 = [PALETTE["fraud"], PALETTE["neutral"], PALETTE["legit"],
           "#F39C12", "#9B59B6", "#1ABC9C"][:len(rkeys)]
bars9 = ax.bar(rkeys, rvals, color=colors9[:len(rkeys)], alpha=0.85, edgecolor="#0F1117")
ax.set_ylabel("Count")
ax.set_title(f"Ground Truth Fraud Ring Types (Total: {len(ring_types_raw)})\nfrom HI-Small_Patterns.txt")
ax.grid(True, axis="y")
for bar, val in zip(bars9, rvals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            str(val), ha="center", fontsize=10, color="white", fontweight="bold")
save(fig, "09_ring_types.png")

# ════════════════════════════════════════════════════════
# SECTION 11 — TIME-AWARE SPLIT BOUNDARIES
# ════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SECTION 11: PROPOSED TIME-AWARE SPLIT BOUNDARIES")
print("=" * 60)
tx_sorted = tx.sort_values("Timestamp").reset_index(drop=True)
n = len(tx_sorted)
i_train = int(n * 0.70)
i_val   = int(n * 0.85)

train_end = tx_sorted.loc[i_train, "Timestamp"]
val_end   = tx_sorted.loc[i_val,   "Timestamp"]
test_end  = tx_sorted.loc[n - 1,   "Timestamp"]

df_train = tx_sorted.iloc[:i_train]
df_val   = tx_sorted.iloc[i_train:i_val]
df_test  = tx_sorted.iloc[i_val:]

print(f"\nTrain (70%): {tx_sorted.loc[0,'Timestamp']} → {train_end}  "
      f"| Fraud: {df_train['Is Laundering'].sum():,}")
print(f"Val   (15%): {train_end} → {val_end}  "
      f"| Fraud: {df_val['Is Laundering'].sum():,}")
print(f"Test  (15%): {val_end} → {test_end}  "
      f"| Fraud: {df_test['Is Laundering'].sum():,}")

# ── Chart 10: Split boundaries on daily volume ──────────
print("Generating Chart 10 — Time split visualization …")
fig, ax = plt.subplots(figsize=(12, 4))
daily2 = tx_sorted.groupby(tx_sorted["Timestamp"].dt.date).size()
xs = list(range(len(daily2)))
ax.fill_between(xs, daily2.values, alpha=0.5, color=PALETTE["legit"])
ax.set_xticks(xs)
ax.set_xticklabels([str(d) for d in daily2.index], rotation=45, ha="right", fontsize=7)

days = [str(d) for d in daily2.index]
def day_idx(ts):
    s = str(ts.date())
    return days.index(s) if s in days else 0

ax.axvline(day_idx(train_end), color="#F39C12", linewidth=2,
           linestyle="--", label=f"Train/Val split ({train_end.date()})")
ax.axvline(day_idx(val_end),   color=PALETTE["fraud"], linewidth=2,
           linestyle="--", label=f"Val/Test split ({val_end.date()})")

ax.fill_betweenx([0, daily2.max()], 0, day_idx(train_end),
                 alpha=0.08, color=PALETTE["legit"])
ax.fill_betweenx([0, daily2.max()], day_idx(train_end), day_idx(val_end),
                 alpha=0.08, color="#F39C12")
ax.fill_betweenx([0, daily2.max()], day_idx(val_end), xs[-1],
                 alpha=0.08, color=PALETTE["fraud"])

ax.text(day_idx(train_end) / 2, daily2.max() * 0.85,
        "TRAIN (70%)", ha="center", color=PALETTE["legit"], fontsize=10, fontweight="bold")
ax.text((day_idx(train_end) + day_idx(val_end)) / 2, daily2.max() * 0.85,
        "VAL (15%)", ha="center", color="#F39C12", fontsize=10, fontweight="bold")
ax.text((day_idx(val_end) + xs[-1]) / 2, daily2.max() * 0.85,
        "TEST (15%)", ha="center", color=PALETTE["fraud"], fontsize=10, fontweight="bold")

ax.set_ylabel("Transaction Count")
ax.set_title("Time-Aware Train / Val / Test Splits (No Leakage)")
ax.legend(facecolor="#1A1D27", edgecolor="#3A3D4D")
ax.grid(True)
save(fig, "10_time_splits.png")

# ════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EDA COMPLETE")
print(f"All charts saved to: {OUT_DIR}")
print("=" * 60)
