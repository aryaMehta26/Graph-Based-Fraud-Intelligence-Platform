"""
============================================================
IBM AML HI-Medium — DEEP Exploratory Data Analysis
============================================================
Project  : Palantir-Inspired Graph Fraud Intelligence
Dataset  : HI-Medium_Trans.csv (31.9M rows)
Covers   : Schema, Nulls, Time, Fraud Distribution, Amounts,
           Payment Formats, Currencies, Hourly/Daily/Weekly
           Patterns, Graph Structure (Degrees, Self-loops,
           Top Accounts), Cross-Currency, Fraud Velocity,
           Amount Buckets, Ring Types, Time Splits
Charts   : notebooks/eda_charts_deep/
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os, re
from pathlib import Path
from collections import Counter

# ── PATHS ────────────────────────────────────────────────
BASE = os.getenv("AML_DATASET_DIR", str(Path.home() / ".cache/kagglehub/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/versions/8"))
TX_FILE  = f"{BASE}/HI-Medium_Trans.csv"
ACC_FILE = f"{BASE}/HI-Medium_accounts.csv"
PAT_FILE = f"{BASE}/HI-Medium_Patterns.txt"
OUT_DIR  = str(Path(__file__).resolve().parent / "eda_charts_deep")
os.makedirs(OUT_DIR, exist_ok=True)

# ── STYLE ────────────────────────────────────────────────
FR = "#E05C5C"; LG = "#4C9BE8"; NT = "#6C8EBF"
plt.rcParams.update({
    "figure.facecolor":"#0F1117","axes.facecolor":"#1A1D27",
    "axes.edgecolor":"#3A3D4D","axes.labelcolor":"#CCCCCC",
    "text.color":"#CCCCCC","xtick.color":"#AAAAAA","ytick.color":"#AAAAAA",
    "grid.color":"#2E3145","grid.linestyle":"--",
    "axes.titlesize":13,"axes.labelsize":11,
})

def save(fig, name):
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p}")

# ════════════════════════════════════════════════════════
print("Loading HI-Medium_Trans.csv (31.9M rows) ...")
tx  = pd.read_csv(TX_FILE)
acc = pd.read_csv(ACC_FILE)
tx["Timestamp"] = pd.to_datetime(tx["Timestamp"])

# Rename for clarity
tx = tx.rename(columns={"Account":"src_acct","Account.1":"dst_acct"})

fraud = tx[tx["Is Laundering"]==1].copy()
legit = tx[tx["Is Laundering"]==0].copy()
print(f"Loaded: {len(tx):,} rows | Fraud: {len(fraud):,} | Legit: {len(legit):,}")

# ════════════════════════════════════════════════════════
# S1 — SCHEMA & BASIC STATS
# ════════════════════════════════════════════════════════
print("\n====== S1: SCHEMA & BASIC STATS ======")
print(f"Shape            : {tx.shape}")
print(f"Columns          : {list(tx.columns)}")
print(f"Accounts file    : {acc.shape}")
print("\nDtypes:\n", tx.dtypes.to_string())
print("\nFirst 5 rows:\n", tx.head(5).to_string())

# ════════════════════════════════════════════════════════
# S2 — NULL / MISSING
# ════════════════════════════════════════════════════════
print("\n====== S2: NULL / MISSING VALUES ======")
nulls = tx.isnull().sum()
print(nulls.to_string())
pct_null = (nulls / len(tx) * 100).round(4)

# Chart 01 — Null summary
fig, ax = plt.subplots(figsize=(11, 3))
colors_null = [FR if v > 0 else "#2ECC71" for v in nulls.values]
bars = ax.bar(nulls.index, nulls.values, color=colors_null, edgecolor="#0F1117")
for bar, v in zip(bars, nulls.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
            str(int(v)), ha="center", fontsize=9, color="white")
ax.set_title("Missing Values per Column (All Zero = Clean Dataset)")
ax.set_ylabel("Null Count"); ax.grid(True, axis="y")
plt.xticks(rotation=30, ha="right")
save(fig, "01_nulls.png")

# ════════════════════════════════════════════════════════
# S3 — TIME RANGE & DESCRIPTIVE
# ════════════════════════════════════════════════════════
print("\n====== S3: TIME RANGE ======")
print(f"Start   : {tx['Timestamp'].min()}")
print(f"End     : {tx['Timestamp'].max()}")
print(f"Span    : {(tx['Timestamp'].max()-tx['Timestamp'].min()).days} days")

tx["Hour"]    = tx["Timestamp"].dt.hour
tx["DayName"] = tx["Timestamp"].dt.day_name()
tx["Date"]    = tx["Timestamp"].dt.date
tx["Week"]    = tx["Timestamp"].dt.isocalendar().week.astype(int)

# Chart 02 — Daily volume ALL
daily_all   = tx.groupby("Date").size()
daily_fraud = fraud.groupby(fraud["Timestamp"].dt.date).size().reindex(daily_all.index, fill_value=0)
daily_legit = legit.groupby(legit["Timestamp"].dt.date).size().reindex(daily_all.index, fill_value=0)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,7), sharex=True)
xs = range(len(daily_all))
ax1.fill_between(xs, daily_all.values, alpha=0.6, color=LG, label="All")
ax1.fill_between(xs, daily_legit.values, alpha=0.4, color=NT, label="Legit")
ax1.set_ylabel("Transaction Count"); ax1.legend(); ax1.grid(True)
ax1.set_title("HI-Medium: Daily Transaction Volume")

ax2.bar(xs, daily_fraud.values, color=FR, alpha=0.85, label="Fraud")
ax2.set_ylabel("Fraud Count"); ax2.legend(); ax2.grid(True, axis="y")
ax2.set_xticks(list(xs))
ax2.set_xticklabels([str(d) for d in daily_all.index], rotation=45, ha="right", fontsize=7)
ax2.set_title("Daily Fraud Count")
fig.tight_layout()
save(fig, "02_daily_volume.png")

# Chart 03 — Hourly patterns
hourly_all   = tx.groupby("Hour").size()
hourly_fraud = fraud.groupby(fraud["Timestamp"].dt.hour).size().reindex(range(24), fill_value=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.bar(hourly_all.index, hourly_all.values, color=LG, alpha=0.8, edgecolor="#0F1117")
ax1.set_xlabel("Hour of Day (0-23)"); ax1.set_ylabel("Count")
ax1.set_title("All Transactions by Hour of Day"); ax1.grid(True, axis="y")

ax2.bar(hourly_fraud.index, hourly_fraud.values, color=FR, alpha=0.85, edgecolor="#0F1117")
ax2.set_xlabel("Hour of Day (0-23)"); ax2.set_ylabel("Fraud Count")
ax2.set_title("Fraud Transactions by Hour of Day"); ax2.grid(True, axis="y")
fig.suptitle("Hourly Transaction Patterns", fontsize=14)
save(fig, "03_hourly_patterns.png")

# Chart 04 — Day-of-week
dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dow_all   = tx["DayName"].value_counts().reindex(dow_order, fill_value=0)
dow_fraud = fraud["Timestamp"].dt.day_name().value_counts().reindex(dow_order, fill_value=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
colors_dow = ["#4C9BE8" if d not in ["Saturday","Sunday"] else "#6C8EBF" for d in dow_order]
ax1.bar(range(7), dow_all.values, color=colors_dow, edgecolor="#0F1117")
ax1.set_xticks(range(7)); ax1.set_xticklabels(dow_order, rotation=30, ha="right")
ax1.set_title("All Transactions by Day of Week"); ax1.grid(True, axis="y")

colors_dow2 = ["#E05C5C" if d not in ["Saturday","Sunday"] else "#F39C12" for d in dow_order]
ax2.bar(range(7), dow_fraud.values, color=colors_dow2, edgecolor="#0F1117")
ax2.set_xticks(range(7)); ax2.set_xticklabels(dow_order, rotation=30, ha="right")
ax2.set_title("Fraud by Day of Week"); ax2.grid(True, axis="y")
for i, v in enumerate(dow_fraud.values):
    ax2.text(i, v+5, str(int(v)), ha="center", fontsize=9, color="white")
fig.suptitle("Day-of-Week Transaction & Fraud Patterns", fontsize=14)
save(fig, "04_dayofweek.png")

# ════════════════════════════════════════════════════════
# S4 — FRAUD LABEL DISTRIBUTION
# ════════════════════════════════════════════════════════
print("\n====== S4: FRAUD LABEL DISTRIBUTION ======")
vc = tx["Is Laundering"].value_counts()
fr_rate = vc.get(1,0)/len(tx)*100
ratio   = vc.get(0,0)//max(vc.get(1,1),1)
print(f"Legit : {vc.get(0,0):,}")
print(f"Fraud : {vc.get(1,0):,}")
print(f"Rate  : {fr_rate:.4f}%")
print(f"Ratio : {ratio:,}:1")

# Chart 05 — Class imbalance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
wedges, texts, auto = ax1.pie(
    [vc.get(0,0), vc.get(1,0)], labels=["Legit","Fraud"],
    colors=[LG, FR], autopct="%1.3f%%", startangle=90,
    pctdistance=0.75, wedgeprops=dict(width=0.5, edgecolor="#0F1117"))
for t in auto: t.set_color("white"); t.set_fontsize(10)
ax1.set_title("Class Distribution (Donut)")
ax2.bar(["Legit","Fraud"], [vc.get(0,0), vc.get(1,0)], color=[LG,FR], edgecolor="#0F1117")
ax2.set_yscale("log"); ax2.set_ylabel("Count (log scale)"); ax2.grid(True, axis="y")
ax2.set_title(f"Log-Scale ({ratio:,}:1 imbalance)")
fig.suptitle(f"HI-Medium: {len(tx):,} transactions | Fraud = {fr_rate:.4f}%", fontsize=13)
save(fig, "05_class_imbalance.png")

# ════════════════════════════════════════════════════════
# S5 — AMOUNT ANALYSIS
# ════════════════════════════════════════════════════════
print("\n====== S5: AMOUNTS ======")
print("All:\n",  tx["Amount Paid"].describe().to_string())
print("Fraud:\n", fraud["Amount Paid"].describe().to_string())
print("Legit:\n", legit["Amount Paid"].describe().to_string())

# Chart 06 — Distribution (log scale)
fig, ax = plt.subplots(figsize=(11, 5))
bins = np.logspace(-3, 13, 100)
ax.hist(legit["Amount Paid"].clip(upper=1e13), bins=bins, alpha=0.5, color=LG,
        label=f"Legit (n={len(legit):,})")
ax.hist(fraud["Amount Paid"].clip(upper=1e13), bins=bins, alpha=0.8, color=FR,
        label=f"Fraud (n={len(fraud):,})")
ax.axvline(legit["Amount Paid"].median(), color=LG, lw=2, linestyle="--",
           label=f"Legit median ${legit['Amount Paid'].median():,.0f}")
ax.axvline(fraud["Amount Paid"].median(), color=FR, lw=2, linestyle="--",
           label=f"Fraud median ${fraud['Amount Paid'].median():,.0f}")
ax.set_xscale("log"); ax.set_xlabel("Amount Paid (USD, log scale)")
ax.set_ylabel("Count"); ax.legend(facecolor="#1A1D27", edgecolor="#3A3D4D")
ax.set_title("Amount Distribution — Fraud vs Legit (Log Scale)")
ax.grid(True, axis="x")
save(fig, "06_amount_distribution.png")

# Chart 07 — Amount buckets
print("\n=== AMOUNT BUCKETS ===")
bins_b = [0, 100, 500, 1000, 5000, 10000, 50000, 100000, 1e6, 1e9, np.inf]
labels_b = ["<$100","$100-500","$500-1K","$1K-5K","$5K-10K",
            "$10K-50K","$50K-100K","$100K-1M","$1M-1B",">$1B"]
tx["AmtBucket"]    = pd.cut(tx["Amount Paid"], bins=bins_b, labels=labels_b)
fraud["AmtBucket"] = pd.cut(fraud["Amount Paid"], bins=bins_b, labels=labels_b)
legit["AmtBucket"] = pd.cut(legit["Amount Paid"], bins=bins_b, labels=labels_b)

bkt_all   = tx["AmtBucket"].value_counts().reindex(labels_b, fill_value=0)
bkt_fraud = fraud["AmtBucket"].value_counts().reindex(labels_b, fill_value=0)
bkt_legit = legit["AmtBucket"].value_counts().reindex(labels_b, fill_value=0)

# Fraud rate per bucket
bkt_rate = (bkt_fraud / bkt_all.replace(0, np.nan) * 100).fillna(0)
print("Fraud rate per amount bucket:")
for l, r in zip(labels_b, bkt_rate.values):
    print(f"  {l:<15}: {r:.4f}%")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
x = range(len(labels_b))
ax1.bar([i-0.2 for i in x], bkt_legit.values, 0.4, color=LG, alpha=0.7, label="Legit")
ax1.bar([i+0.2 for i in x], bkt_fraud.values, 0.4, color=FR, alpha=0.85, label="Fraud")
ax1.set_yscale("log"); ax1.set_xticks(list(x))
ax1.set_xticklabels(labels_b, rotation=30, ha="right")
ax1.set_ylabel("Count (log)"); ax1.legend(); ax1.grid(True, axis="y")
ax1.set_title("Transaction Count by Amount Bucket")

ax2.bar(x, bkt_rate.values, color=[FR if v>0.5 else NT for v in bkt_rate.values],
        alpha=0.85, edgecolor="#0F1117")
ax2.set_xticks(list(x)); ax2.set_xticklabels(labels_b, rotation=30, ha="right")
ax2.set_ylabel("Fraud Rate (%)"); ax2.grid(True, axis="y")
ax2.set_title("Fraud Rate (%) per Amount Bucket")
for i, v in enumerate(bkt_rate.values):
    if v > 0: ax2.text(i, v+0.002, f"{v:.3f}%", ha="center", fontsize=8, color="white")
fig.suptitle("Amount Bucket Analysis", fontsize=14)
fig.tight_layout()
save(fig, "07_amount_buckets.png")

# ════════════════════════════════════════════════════════
# S6 — PAYMENT FORMAT DEEP DIVE
# ════════════════════════════════════════════════════════
print("\n====== S6: PAYMENT FORMAT ======")
fmt_all   = tx["Payment Format"].value_counts()
fmt_fraud = fraud["Payment Format"].value_counts().reindex(fmt_all.index, fill_value=0)
fmt_legit = legit["Payment Format"].value_counts().reindex(fmt_all.index, fill_value=0)
fmt_rate  = (fmt_fraud / fmt_all * 100).round(4)
print("Payment format fraud rates:")
for f, r in fmt_rate.items():
    print(f"  {f:<15}: {r:.4f}%  (fraud={int(fmt_fraud[f]):,} / total={int(fmt_all[f]):,})")

# Chart 08 — Payment format
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
x = range(len(fmt_all))
axes[0].bar(x, fmt_all.values, color=LG, alpha=0.75, edgecolor="#0F1117")
axes[0].set_xticks(list(x)); axes[0].set_xticklabels(fmt_all.index, rotation=30, ha="right")
axes[0].set_title("All Txns"); axes[0].grid(True, axis="y")

cc = [FR if v>500 else NT for v in fmt_fraud.values]
axes[1].bar(x, fmt_fraud.values, color=cc, alpha=0.85, edgecolor="#0F1117")
axes[1].set_xticks(list(x)); axes[1].set_xticklabels(fmt_fraud.index, rotation=30, ha="right")
axes[1].set_title("Fraud Count by Format"); axes[1].grid(True, axis="y")
for i, v in enumerate(fmt_fraud.values):
    if v > 0: axes[1].text(i, v+100, str(int(v)), ha="center", fontsize=9, color="white")

axes[2].bar(x, fmt_rate.values, color=[FR if v>0.5 else NT for v in fmt_rate.values],
            alpha=0.85, edgecolor="#0F1117")
axes[2].set_xticks(list(x)); axes[2].set_xticklabels(fmt_rate.index, rotation=30, ha="right")
axes[2].set_ylabel("Fraud Rate %"); axes[2].set_title("Fraud Rate % by Format")
axes[2].grid(True, axis="y")
for i, v in enumerate(fmt_rate.values):
    axes[2].text(i, v+0.01, f"{v:.3f}%", ha="center", fontsize=9, color="white")
fig.suptitle("Payment Format Deep Dive — HI-Medium", fontsize=14)
save(fig, "08_payment_format_deep.png")

# ════════════════════════════════════════════════════════
# S7 — CURRENCY ANALYSIS
# ════════════════════════════════════════════════════════
print("\n====== S7: CURRENCY ANALYSIS ======")
curr_all   = tx["Receiving Currency"].value_counts()
curr_fraud = fraud["Receiving Currency"].value_counts().reindex(curr_all.index, fill_value=0)
curr_rate  = (curr_fraud / curr_all * 100).fillna(0).round(4)
print("Currency fraud rates:")
for c, r in curr_rate.items():
    print(f"  {c:<20}: {r:.4f}%")

cross_curr = tx[tx["Payment Currency"] != tx["Receiving Currency"]]
print(f"\nCross-currency txns : {len(cross_curr):,} ({len(cross_curr)/len(tx)*100:.2f}%)")
print(f"  of which fraud    : {cross_curr['Is Laundering'].sum():,}")

# Chart 09 — Currency
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.barh(curr_all.index[::-1], curr_all.values[::-1], color=LG, alpha=0.75, edgecolor="#0F1117")
ax1.set_xlabel("Count"); ax1.set_title("All Txns by Currency"); ax1.grid(True, axis="x")

cc9 = [FR if curr_fraud.get(c,0)>200 else NT for c in curr_fraud.index[::-1]]
ax2.barh(curr_fraud.index[::-1], curr_fraud.values[::-1], color=cc9, alpha=0.85, edgecolor="#0F1117")
ax2.set_xlabel("Fraud Count"); ax2.set_title("Fraud Txns by Currency"); ax2.grid(True, axis="x")
for i, (c, v) in enumerate(zip(curr_fraud.index[::-1], curr_fraud.values[::-1])):
    if v > 0: ax2.text(v+10, i, str(int(v)), va="center", fontsize=9)
fig.suptitle("Currency Analysis — HI-Medium", fontsize=14)
save(fig, "09_currency.png")

# ════════════════════════════════════════════════════════
# S8 — GRAPH STRUCTURE (Degrees, Self-loops, Hubs)
# ════════════════════════════════════════════════════════
print("\n====== S8: GRAPH STRUCTURE ======")
out_deg = tx.groupby("src_acct").size().sort_values(ascending=False)
in_deg  = tx.groupby("dst_acct").size().sort_values(ascending=False)
all_accts = pd.concat([tx["src_acct"], tx["dst_acct"]]).unique()
self_loops = tx[tx["src_acct"] == tx["dst_acct"]]
non_self   = tx[tx["src_acct"] != tx["dst_acct"]]

print(f"Unique src accounts      : {tx['src_acct'].nunique():,}")
print(f"Unique dst accounts      : {tx['dst_acct'].nunique():,}")
print(f"Total unique accounts    : {len(all_accts):,}")
print(f"Self-loop transactions   : {len(self_loops):,} ({len(self_loops)/len(tx)*100:.2f}%)")
print(f"Non-self-loop txns       : {len(non_self):,}")
print(f"Unique banks (src)       : {tx['From Bank'].nunique():,}")
print(f"Unique banks (dst)       : {tx['To Bank'].nunique():,}")

print("\nSelf-loop payment formats:")
print(self_loops["Payment Format"].value_counts().to_string())
print("\nTop 10 senders (out-degree):")
print(out_deg.head(10).to_string())
print("\nTop 10 receivers (in-degree — potential funnels):")
print(in_deg.head(10).to_string())

# Fraud by account
fraud_src = fraud.groupby("src_acct").size().sort_values(ascending=False)
fraud_dst = fraud.groupby("dst_acct").size().sort_values(ascending=False)
print("\nTop 10 fraud-sending accounts:")
print(fraud_src.head(10).to_string())
print("\nTop 10 fraud-receiving accounts (funnel targets):")
print(fraud_dst.head(10).to_string())

# Chart 10 — Degree distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, data, title, color in zip(
    axes.flatten(),
    [out_deg[out_deg<=200], in_deg[in_deg<=200],
     out_deg[out_deg<=1000], in_deg[in_deg<=1000]],
    ["Out-Degree (clipped @200)", "In-Degree (clipped @200)",
     "Out-Degree (clipped @1000)", "In-Degree (clipped @1000)"],
    [LG, NT, LG, NT]
):
    ax.hist(data.values, bins=80, color=color, alpha=0.8, edgecolor="#0F1117")
    ax.set_xlabel("Degree"); ax.set_ylabel("Accounts"); ax.set_title(title)
    ax.grid(True, axis="y")
fig.suptitle("Account Degree Distributions (Out = Sender, In = Receiver)", fontsize=14)
fig.tight_layout()
save(fig, "10_degree_distributions.png")

# Chart 11 — Top 20 accounts by out-degree
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
top20_out = out_deg.head(20)
ax1.barh(top20_out.index[::-1], top20_out.values[::-1], color=LG, alpha=0.8, edgecolor="#0F1117")
ax1.set_xlabel("# Transactions Sent"); ax1.set_title("Top 20 Senders (Out-Degree)")
ax1.grid(True, axis="x")
for i, (acct, v) in enumerate(zip(top20_out.index[::-1], top20_out.values[::-1])):
    ax1.text(v+500, i, f"{v:,}", va="center", fontsize=8)

top20_in = in_deg.head(20)
ax2.barh(top20_in.index[::-1], top20_in.values[::-1], color=FR, alpha=0.8, edgecolor="#0F1117")
ax2.set_xlabel("# Transactions Received"); ax2.set_title("Top 20 Receivers (In-Degree — Funnel Suspects)")
ax2.grid(True, axis="x")
for i, (acct, v) in enumerate(zip(top20_in.index[::-1], top20_in.values[::-1])):
    ax2.text(v+20, i, f"{v:,}", va="center", fontsize=8)
fig.suptitle("Top Hub Accounts — Graph Structure Analysis", fontsize=14)
fig.tight_layout()
save(fig, "11_top_accounts.png")

# Chart 12 — Self-loop breakdown
sl_formats = self_loops["Payment Format"].value_counts()
fig, ax = plt.subplots(figsize=(9, 5))
colors_sl = [FR if f=="ACH" or f=="Bitcoin" else NT for f in sl_formats.index]
bars = ax.bar(sl_formats.index, sl_formats.values, color=colors_sl, alpha=0.85, edgecolor="#0F1117")
ax.set_ylabel("Self-Loop Count"); ax.set_title(f"Self-Loop Transactions by Format (Total: {len(self_loops):,})")
ax.grid(True, axis="y")
for bar, v in zip(bars, sl_formats.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5000,
            f"{v:,}", ha="center", fontsize=9, color="white")
save(fig, "12_self_loops.png")

# ════════════════════════════════════════════════════════
# S9 — FRAUD VELOCITY (Fraud per account — how fast they move)
# ════════════════════════════════════════════════════════
print("\n====== S9: FRAUD VELOCITY ======")
fraud_per_acct = fraud.groupby("src_acct").agg(
    fraud_txns=("Is Laundering","count"),
    total_fraud_amt=("Amount Paid","sum"),
    first_fraud=("Timestamp","min"),
    last_fraud=("Timestamp","max")
)
fraud_per_acct["duration_hrs"] = (
    (fraud_per_acct["last_fraud"] - fraud_per_acct["first_fraud"])
    .dt.total_seconds() / 3600
).clip(lower=0.01)
fraud_per_acct["fraud_velocity"] = (
    fraud_per_acct["fraud_txns"] / fraud_per_acct["duration_hrs"]
)
print(f"Accounts with >1 fraud txn: {(fraud_per_acct['fraud_txns']>1).sum():,}")
print("\nTop 10 accounts by fraud_txns:")
print(fraud_per_acct.sort_values("fraud_txns", ascending=False).head(10).to_string())
print("\nTop 10 accounts by fraud velocity (txns/hr):")
print(fraud_per_acct.sort_values("fraud_velocity", ascending=False).head(10)[
    ["fraud_txns","fraud_velocity","total_fraud_amt"]].to_string())

# Chart 13 — Fraud velocity
multi_fraud = fraud_per_acct[fraud_per_acct["fraud_txns"] > 1].copy()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.hist(multi_fraud["fraud_txns"].clip(upper=100), bins=40, color=FR, alpha=0.8, edgecolor="#0F1117")
ax1.set_xlabel("Fraud Txns per Account"); ax1.set_ylabel("Accounts")
ax1.set_title("Fraud Transactions per Account\n(clipped @100, accounts with >1 fraud)")
ax1.grid(True, axis="y")

ax2.hist(multi_fraud["fraud_velocity"].clip(upper=50), bins=40, color="#F39C12",
         alpha=0.85, edgecolor="#0F1117")
ax2.set_xlabel("Fraud Txns per Hour"); ax2.set_ylabel("Accounts")
ax2.set_title("Fraud Velocity (Txns/hr per Account)\n(clipped @50)")
ax2.grid(True, axis="y")
fig.suptitle("Fraud Velocity Analysis", fontsize=14)
save(fig, "13_fraud_velocity.png")

# ════════════════════════════════════════════════════════
# S10 — ACCOUNTS FILE ANALYSIS
# ════════════════════════════════════════════════════════
print("\n====== S10: ACCOUNTS FILE ======")
print(f"Shape    : {acc.shape}")
print(f"Columns  : {list(acc.columns)}")
print("\nFirst 5 rows:\n", acc.head(5).to_string())
print("\nNulls:\n", acc.isnull().sum().to_string())

entity_types = acc["Entity Name"].str.extract(r"^(.*?) #")[0].value_counts()
print("\nEntity types (from Entity Name):")
print(entity_types.to_string())

bank_counts = acc["Bank Name"].str.extract(r"^(.*?) #")[0].value_counts()
print("\nTop 10 Bank types:")
print(bank_counts.head(10).to_string())

# Chart 14 — Entity types
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
colors14 = [LG, NT, FR, "#F39C12", "#9B59B6"][:len(entity_types)]
bars14 = ax1.bar(entity_types.index, entity_types.values, color=colors14, alpha=0.85, edgecolor="#0F1117")
ax1.set_ylabel("Count"); ax1.set_title("Account Entity Types")
ax1.grid(True, axis="y")
for b, v in zip(bars14, entity_types.values):
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+1000,
             f"{v:,}", ha="center", fontsize=9, color="white")

colors14b = [LG, NT, FR, "#F39C12", "#9B59B6"]*5
bars14b = ax2.barh(bank_counts.index[:10][::-1], bank_counts.values[:10][::-1],
                   color=colors14b[:10], alpha=0.8, edgecolor="#0F1117")
ax2.set_xlabel("Count"); ax2.set_title("Top 10 Bank Types")
ax2.grid(True, axis="x")
fig.suptitle("Accounts File Analysis", fontsize=14)
save(fig, "14_accounts_analysis.png")

# ════════════════════════════════════════════════════════
# S11 — PATTERNS / GROUND TRUTH RINGS
# ════════════════════════════════════════════════════════
print("\n====== S11: PATTERNS FILE (GROUND TRUTH RINGS) ======")
pat = open(PAT_FILE).read()
rings_raw = [l for l in pat.splitlines() if l.startswith("BEGIN")]
names = []
for r in rings_raw:
    m = re.search(r"ATTEMPT - ([A-Z\-]+)", r)
    if m: names.append(m.group(1))
rc = Counter(names)
print(f"Total labeled rings : {len(rings_raw):,}")
print("By type:")
for k, v in rc.most_common():
    print(f"  {k:<20}: {v:4d}  ({v/len(rings_raw)*100:.1f}%)")

# Chart 15 — Ring types
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
rk, rv = zip(*rc.most_common())
cc15 = [FR, NT, LG, "#F39C12", "#9B59B6", "#1ABC9C", "#E67E22", "#3498DB"][:len(rk)]
bars15 = ax1.bar(rk, rv, color=cc15, alpha=0.85, edgecolor="#0F1117")
ax1.set_ylabel("Count"); ax1.set_title(f"Fraud Ring Types (Total: {len(rings_raw):,})")
ax1.grid(True, axis="y")
for b, v in zip(bars15, rv):
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+2,
             str(v), ha="center", fontsize=10, color="white", fontweight="bold")

wedges, texts, auto = ax2.pie(rv, labels=rk, colors=cc15, autopct="%1.1f%%",
    startangle=90, pctdistance=0.75, wedgeprops=dict(width=0.5, edgecolor="#0F1117"))
for t in auto: t.set_color("white"); t.set_fontsize(8)
ax2.set_title("Ring Type Distribution (%)")
fig.suptitle("Ground Truth Fraud Ring Analysis — HI-Medium Patterns File", fontsize=13)
save(fig, "15_ring_types.png")

# ════════════════════════════════════════════════════════
# S12 — TIME-AWARE SPLITS
# ════════════════════════════════════════════════════════
print("\n====== S12: TIME-AWARE SPLITS ======")
tx_s = tx.sort_values("Timestamp").reset_index(drop=True)
n = len(tx_s)
i70, i85 = int(n*0.70), int(n*0.85)
t_start = tx_s.loc[0,"Timestamp"]
t70     = tx_s.loc[i70,"Timestamp"]
t85     = tx_s.loc[i85,"Timestamp"]
t_end   = tx_s.loc[n-1,"Timestamp"]
f_train = tx_s.iloc[:i70]["Is Laundering"].sum()
f_val   = tx_s.iloc[i70:i85]["Is Laundering"].sum()
f_test  = tx_s.iloc[i85:]["Is Laundering"].sum()
print(f"Train 70%: {t_start} -> {t70}  | rows={i70:,} | fraud={f_train:,}")
print(f"Val   15%: {t70} -> {t85}  | rows={i85-i70:,} | fraud={f_val:,}")
print(f"Test  15%: {t85} -> {t_end}  | rows={n-i85:,} | fraud={f_test:,}")

# Chart 16 — Time splits on daily volume
tx_s2 = tx_s.copy(); tx_s2["Date"] = tx_s2["Timestamp"].dt.date
daily2 = tx_s2.groupby("Date").size()
fr_s2  = tx_s2[tx_s2["Is Laundering"]==1]; fr_s2["Date2"] = fr_s2["Timestamp"].dt.date
daily_f2 = fr_s2.groupby("Date2").size().reindex(daily2.index, fill_value=0)
xs2 = list(range(len(daily2)))
days2 = [str(d) for d in daily2.index]
def find_idx(ts):
    s = str(ts.date())
    return days2.index(s) if s in days2 else 0

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(xs2, daily2.values, alpha=0.4, color=LG, label="All Txns")
ax.fill_between(xs2, daily_f2.values*200, alpha=0.8, color=FR, label="Fraud x200 (scaled)")
i1 = find_idx(t70); i2 = find_idx(t85)
ax.axvline(i1, color="#F39C12", lw=2, linestyle="--", label=f"Train|Val @ {t70.date()}")
ax.axvline(i2, color=FR, lw=2, linestyle="--", label=f"Val|Test @ {t85.date()}")
ax.fill_betweenx([0, daily2.max()], 0, i1, alpha=0.06, color=LG)
ax.fill_betweenx([0, daily2.max()], i1, i2, alpha=0.06, color="#F39C12")
ax.fill_betweenx([0, daily2.max()], i2, xs2[-1], alpha=0.06, color=FR)
ax.text(i1//2,   daily2.max()*0.85, "TRAIN (70%)\n"+f"fraud={f_train:,}", ha="center", color=LG, fontsize=9, fontweight="bold")
ax.text((i1+i2)//2, daily2.max()*0.85, "VAL (15%)\n"+f"fraud={f_val:,}",  ha="center", color="#F39C12", fontsize=9, fontweight="bold")
ax.text((i2+xs2[-1])//2, daily2.max()*0.85, "TEST (15%)\n"+f"fraud={f_test:,}", ha="center", color=FR, fontsize=9, fontweight="bold")
ax.set_xticks(xs2); ax.set_xticklabels(days2, rotation=45, ha="right", fontsize=7)
ax.legend(facecolor="#1A1D27", edgecolor="#3A3D4D"); ax.grid(True)
ax.set_title("HI-Medium: Daily Volume + Time-Aware Train/Val/Test Splits")
save(fig, "16_time_splits.png")

# ════════════════════════════════════════════════════════
print("\n====== EDA COMPLETE ======")
print(f"All charts saved to: {OUT_DIR}")
print(f"Total charts        : 16")
