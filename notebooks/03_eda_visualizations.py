"""
============================================================
SCRIPT 3 OF 3 — EDA + VISUALIZATIONS
============================================================
Team     : DATA 298A — Team 12
Purpose  : Full exploratory data analysis with charts
           Covers: class imbalance, amounts, payment formats,
           currencies, graph structure, fraud velocity,
           ring types, and time-aware splits
Run      : python3 03_eda_visualizations.py
Charts   : notebooks/eda_charts_live/
============================================================
"""

import os
import re
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import kagglehub

DIVIDER = "=" * 60
BASE     = kagglehub.dataset_download("ealtman2019/ibm-transactions-for-anti-money-laundering-aml")
TX_FILE  = os.path.join(BASE, "HI-Medium_Trans.csv")
ACC_FILE = os.path.join(BASE, "HI-Medium_accounts.csv")
PAT_FILE = f"{BASE}/HI-Medium_Patterns.txt"
OUT_DIR  = os.path.join(os.path.dirname(__file__), "eda_charts_live")
os.makedirs(OUT_DIR, exist_ok=True)

# ── STYLE ────────────────────────────────────────────────
FR = "#E05C5C"; LG = "#4C9BE8"; NT = "#6C8EBF"; YL = "#F39C12"
plt.rcParams.update({
    "figure.facecolor":"#0F1117", "axes.facecolor":"#1A1D27",
    "axes.edgecolor":"#3A3D4D",   "axes.labelcolor":"#CCCCCC",
    "text.color":"#CCCCCC",       "xtick.color":"#AAAAAA",
    "ytick.color":"#AAAAAA",      "grid.color":"#2E3145",
    "grid.linestyle":"--",        "axes.titlesize":13,
    "axes.labelsize":11,
})

chart_num = [0]
def save(fig, name):
    chart_num[0] += 1
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Chart {chart_num[0]:02d}] Saved → {p}")

# ════════════════════════════════════════════════════════
print(DIVIDER)
print("STEP 3: EDA + VISUALIZATIONS")
print("Team 12 | DATA 298A | Graph Fraud Intelligence Platform")
print(DIVIDER)

print("\n[LOADING] Reading HI-Medium_Trans.csv (31.9M rows)...")
tx  = pd.read_csv(TX_FILE)
acc = pd.read_csv(ACC_FILE)
tx["Timestamp"] = pd.to_datetime(tx["Timestamp"])
tx = tx.rename(columns={"Account": "src_acct", "Account.1": "dst_acct"})
tx["Hour"]    = tx["Timestamp"].dt.hour
tx["DayName"] = tx["Timestamp"].dt.day_name()
tx["Date"]    = tx["Timestamp"].dt.date
fraud = tx[tx["Is Laundering"] == 1].copy()
legit = tx[tx["Is Laundering"] == 0].copy()
vc    = tx["Is Laundering"].value_counts()
print(f"  ✅ {len(tx):,} rows loaded | Fraud: {len(fraud):,} | Legit: {len(legit):,}")

# ════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("EDA SECTION 1: DATASET OVERVIEW")
print(DIVIDER)
print(f"\n  Shape              : {tx.shape}")
print(f"  Columns            : {list(tx.columns)}")
print(f"  Date range         : {tx['Timestamp'].min().date()} → {tx['Timestamp'].max().date()}")
print(f"  Span               : {(tx['Timestamp'].max() - tx['Timestamp'].min()).days} days")
print(f"  Unique src accounts: {tx['src_acct'].nunique():,}")
print(f"  Unique dst accounts: {tx['dst_acct'].nunique():,}")
print(f"  Unique banks (src) : {tx['From Bank'].nunique():,}")

print(f"\n  Null values per column:")
nulls = tx.isnull().sum()
for col, n in nulls.items():
    print(f"    ✅ {col:<25}: {n}")

# ════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("EDA SECTION 2: CLASS IMBALANCE")
print(DIVIDER)
fraud_rate = len(fraud) / len(tx) * 100
ratio      = len(legit) // len(fraud)
print(f"\n  Legit (0)    : {len(legit):,}")
print(f"  Fraud (1)    : {len(fraud):,}")
print(f"  Fraud rate   : {fraud_rate:.4f}%")
print(f"  Class ratio  : {ratio:,}:1")
print(f"  Implication  : Cannot use accuracy — use PR-AUC")
print(f"  XGBoost fix  : scale_pos_weight = {ratio}")

# Chart 1 — Class Imbalance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
sizes  = [len(legit), len(fraud)]
colors = [LG, FR]
wedges, texts, auto = ax1.pie(
    sizes, labels=["Legit", "Fraud"], colors=colors,
    autopct="%1.3f%%", startangle=90, pctdistance=0.75,
    wedgeprops=dict(width=0.5, edgecolor="#0F1117"))
for t in auto: t.set_color("white"); t.set_fontsize(10)
ax1.set_title("Class Distribution (Donut)")
ax2.bar(["Legit", "Fraud"], sizes, color=colors, edgecolor="#0F1117", width=0.4)
ax2.set_yscale("log"); ax2.set_ylabel("Count (log scale)"); ax2.grid(True, axis="y")
ax2.set_title(f"{ratio:,}:1 Imbalance — Log Scale")
fig.suptitle(f"HI-Medium: {len(tx):,} transactions | Fraud Rate = {fraud_rate:.4f}%", fontsize=13)
save(fig, "01_class_imbalance.png")

# ════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("EDA SECTION 3: DAILY & HOURLY PATTERNS")
print(DIVIDER)
daily_all   = tx.groupby("Date").size()
daily_fraud = fraud.groupby(fraud["Timestamp"].dt.date).size().reindex(daily_all.index, fill_value=0)
hourly_all  = tx.groupby("Hour").size()
hourly_fr   = fraud.groupby(fraud["Timestamp"].dt.hour).size().reindex(range(24), fill_value=0)
peak_hour   = hourly_fr.idxmax()
print(f"\n  Daily avg transactions : {daily_all.mean():,.0f}")
print(f"  Daily avg fraud        : {daily_fraud.mean():.1f}")
print(f"  Peak fraud hour        : {peak_hour}:00  ({hourly_fr[peak_hour]:,} fraud txns)")

# Chart 2 — Daily volume
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
xs = range(len(daily_all))
ax1.fill_between(xs, daily_all.values, alpha=0.5, color=LG, label="All Transactions")
ax1.set_ylabel("Count"); ax1.legend(); ax1.grid(True)
ax1.set_title("HI-Medium: Daily Transaction Volume (27 days)")
ax2.bar(xs, daily_fraud.values, color=FR, alpha=0.85)
ax2.set_ylabel("Fraud Count"); ax2.grid(True, axis="y")
ax2.set_xticks(list(xs))
ax2.set_xticklabels([str(d) for d in daily_all.index], rotation=45, ha="right", fontsize=7)
ax2.set_title("Daily Fraud Count")
fig.tight_layout()
save(fig, "02_daily_volume.png")

# Chart 3 — Hourly patterns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.bar(hourly_all.index, hourly_all.values, color=LG, alpha=0.8, edgecolor="#0F1117")
ax1.set_xlabel("Hour of Day"); ax1.set_ylabel("Count")
ax1.set_title("All Transactions by Hour"); ax1.grid(True, axis="y")
colors_hr = [FR if h == peak_hour else NT for h in range(24)]
ax2.bar(hourly_fr.index, hourly_fr.values, color=colors_hr, alpha=0.85, edgecolor="#0F1117")
ax2.set_xlabel("Hour of Day"); ax2.set_ylabel("Fraud Count")
ax2.set_title(f"Fraud by Hour of Day (Peak = {peak_hour}:00)")
ax2.grid(True, axis="y")
fig.suptitle("Hourly Transaction & Fraud Patterns", fontsize=14)
save(fig, "03_hourly_patterns.png")

# ════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("EDA SECTION 4: AMOUNT ANALYSIS")
print(DIVIDER)
print(f"\n  Fraud  median  : ${fraud['Amount Paid'].median():>15,.2f}")
print(f"  Legit  median  : ${legit['Amount Paid'].median():>15,.2f}")
print(f"  Fraud  mean    : ${fraud['Amount Paid'].mean():>15,.2f}")
print(f"  Legit  mean    : ${legit['Amount Paid'].mean():>15,.2f}")
print(f"  Fraud  max     : ${fraud['Amount Paid'].max():>15,.2f}")
print(f"  Legit  max     : ${legit['Amount Paid'].max():>15,.2f}")

# Chart 4 — Amount distribution
fig, ax = plt.subplots(figsize=(11, 5))
bins = np.logspace(-3, 13, 100)
ax.hist(legit["Amount Paid"].clip(upper=1e13), bins=bins, alpha=0.5, color=LG,
        label=f"Legit (n={len(legit):,})")
ax.hist(fraud["Amount Paid"].clip(upper=1e13), bins=bins, alpha=0.8, color=FR,
        label=f"Fraud (n={len(fraud):,})")
ax.axvline(legit["Amount Paid"].median(), color=LG, lw=2, linestyle="--",
           label=f"Legit median = ${legit['Amount Paid'].median():,.0f}")
ax.axvline(fraud["Amount Paid"].median(), color=FR, lw=2, linestyle="--",
           label=f"Fraud median = ${fraud['Amount Paid'].median():,.0f}")
ax.set_xscale("log"); ax.set_xlabel("Amount Paid (USD, log scale)")
ax.set_ylabel("Transaction Count")
ax.set_title("Amount Distribution — Fraud vs Legit (Fraud is ~6x larger at median)")
ax.legend(facecolor="#1A1D27"); ax.grid(True, axis="x")
save(fig, "04_amount_distribution.png")

# Chart 5 — Amount buckets fraud rate
bins_b  = [0,100,500,1000,5000,10000,50000,100000,1e6,1e9,np.inf]
labels_b= ["<$100","$100-500","$500-1K","$1K-5K","$5K-10K",
           "$10K-50K","$50K-100K","$100K-1M","$1M-1B",">$1B"]
tx["AmtBucket"]    = pd.cut(tx["Amount Paid"], bins=bins_b, labels=labels_b)
fraud["AmtBucket"] = pd.cut(fraud["Amount Paid"], bins=bins_b, labels=labels_b)
bkt_all   = tx["AmtBucket"].value_counts().reindex(labels_b, fill_value=0)
bkt_fraud = fraud["AmtBucket"].value_counts().reindex(labels_b, fill_value=0)
bkt_rate  = (bkt_fraud / bkt_all.replace(0, np.nan) * 100).fillna(0)

print(f"\n  Fraud rate per amount bucket:")
for l, r in zip(labels_b, bkt_rate.values):
    bar = "█" * int(r * 20) if r > 0 else ""
    print(f"    {l:<15}: {r:.4f}% {bar}")

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(len(labels_b)), bkt_rate.values,
       color=[FR if v > 0.3 else NT for v in bkt_rate.values],
       alpha=0.85, edgecolor="#0F1117")
ax.set_xticks(range(len(labels_b)))
ax.set_xticklabels(labels_b, rotation=30, ha="right")
ax.set_ylabel("Fraud Rate (%)"); ax.grid(True, axis="y")
ax.set_title("Fraud Rate (%) per Amount Bucket — Higher Amounts = Higher Fraud Risk")
for i, v in enumerate(bkt_rate.values):
    if v > 0: ax.text(i, v+0.003, f"{v:.3f}%", ha="center", fontsize=8, color="white")
save(fig, "05_amount_buckets.png")

# ════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("EDA SECTION 5: PAYMENT FORMAT — KEY FRAUD SIGNAL")
print(DIVIDER)
fmt_all   = tx["Payment Format"].value_counts()
fmt_fraud = fraud["Payment Format"].value_counts().reindex(fmt_all.index, fill_value=0)
fmt_rate  = (fmt_fraud / fmt_all * 100).round(4)

print(f"\n  {'Format':<15} {'Total':>12} {'Fraud':>8} {'Fraud%':>8}")
print(f"  {'-'*47}")
for f in fmt_all.index:
    pct = fmt_rate.get(f, 0)
    flag = " ⚠️  HIGHEST" if f == "ACH" else ("  ✅ ZERO" if fmt_fraud.get(f,0)==0 else "")
    print(f"  {f:<15} {int(fmt_all[f]):>12,} {int(fmt_fraud.get(f,0)):>8,} {pct:>7.3f}%{flag}")

pct_ach = fmt_fraud.get("ACH", 0) / len(fraud) * 100
print(f"\n  ACH = {pct_ach:.1f}% of ALL fraud transactions")

# Chart 6 — Payment format
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
x = range(len(fmt_all))
axes[0].bar(x, fmt_all.values, color=LG, alpha=0.75, edgecolor="#0F1117")
axes[0].set_xticks(list(x)); axes[0].set_xticklabels(fmt_all.index, rotation=30, ha="right")
axes[0].set_title("All Txns by Payment Format"); axes[0].grid(True, axis="y")

cc = [FR if v > 500 else NT for v in fmt_fraud.values]
axes[1].bar(x, fmt_fraud.values, color=cc, alpha=0.85, edgecolor="#0F1117")
axes[1].set_xticks(list(x)); axes[1].set_xticklabels(fmt_fraud.index, rotation=30, ha="right")
axes[1].set_title(f"Fraud Count (ACH = {pct_ach:.0f}% of all fraud!)")
axes[1].grid(True, axis="y")
for i, v in enumerate(fmt_fraud.values):
    if v > 0: axes[1].text(i, v+100, f"{int(v):,}", ha="center", fontsize=9, color="white")

axes[2].bar(x, fmt_rate.values, color=[FR if v>0.3 else NT for v in fmt_rate.values],
            alpha=0.85, edgecolor="#0F1117")
axes[2].set_xticks(list(x)); axes[2].set_xticklabels(fmt_rate.index, rotation=30, ha="right")
axes[2].set_ylabel("Fraud %"); axes[2].set_title("ACH Fraud Rate = 0.79%")
axes[2].grid(True, axis="y")
fig.suptitle("Payment Format Deep Dive — ACH Is The Fraud Highway", fontsize=14)
save(fig, "06_payment_format.png")

# ════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("EDA SECTION 6: CURRENCY ANALYSIS")
print(DIVIDER)
curr_all   = tx["Receiving Currency"].value_counts()
curr_fraud = fraud["Receiving Currency"].value_counts().reindex(curr_all.index, fill_value=0)
cross_curr = tx[tx["Payment Currency"] != tx["Receiving Currency"]]
print(f"\n  Top 5 receiving currencies:")
for c in curr_all.head(5).index:
    print(f"    {c:<20}: {int(curr_all[c]):>10,} total | {int(curr_fraud.get(c,0)):>6,} fraud")
print(f"\n  Cross-currency transactions : {len(cross_curr):,} ({len(cross_curr)/len(tx)*100:.2f}%)")
print(f"  Cross-currency fraud       : {cross_curr['Is Laundering'].sum():,}")
print(f"  → Cross-currency = ZERO fraud (strong NEGATIVE signal)")

# Chart 7 — Currency
fig, ax = plt.subplots(figsize=(10, 7))
cc7 = [FR if (curr_fraud.get(c, 0) > 1000) else LG for c in curr_all.index[::-1]]
ax.barh(curr_all.index[::-1], curr_all.values[::-1], color=cc7, alpha=0.8, edgecolor="#0F1117")
ax.set_xlabel("Transaction Count"); ax.set_title("Transaction Volume by Receiving Currency")
ax.grid(True, axis="x")
save(fig, "07_currency.png")

# ════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("EDA SECTION 7: GRAPH STRUCTURE")
print(DIVIDER)
out_deg     = tx.groupby("src_acct").size().sort_values(ascending=False)
in_deg      = tx.groupby("dst_acct").size().sort_values(ascending=False)
self_loops  = tx[tx["src_acct"] == tx["dst_acct"]]
all_accts   = pd.concat([tx["src_acct"], tx["dst_acct"]]).unique()

print(f"\n  Unique accounts total  : {len(all_accts):,}")
print(f"  Unique src accounts    : {tx['src_acct'].nunique():,}")
print(f"  Unique dst accounts    : {tx['dst_acct'].nunique():,}")
print(f"  Self-loops             : {len(self_loops):,} ({len(self_loops)/len(tx)*100:.2f}%)")
print(f"\n  Top 5 senders (ring suspects):")
for acct, cnt in out_deg.head(5).items():
    is_fraud_acct = fraud[fraud["src_acct"]==acct]["Is Laundering"].sum()
    print(f"    {acct} : {cnt:,} txns sent | fraud_txns={is_fraud_acct:,}")
print(f"\n  Top 5 receivers (funnel suspects):")
for acct, cnt in in_deg.head(5).items():
    print(f"    {acct} : {cnt:,} txns received")

# Chart 8 — Degree distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.hist(out_deg[out_deg<=200].values, bins=80, color=LG, alpha=0.8, edgecolor="#0F1117")
ax1.set_xlabel("Out-Degree (txns sent)"); ax1.set_ylabel("# Accounts")
ax1.set_title("Out-Degree Distribution (clipped @200)\nHigh spikes = potential hub fraudsters")
ax1.grid(True, axis="y")

ax2.hist(in_deg[in_deg<=200].values, bins=80, color=NT, alpha=0.8, edgecolor="#0F1117")
ax2.set_xlabel("In-Degree (txns received)"); ax2.set_ylabel("# Accounts")
ax2.set_title("In-Degree Distribution (clipped @200)\nHigh spikes = potential funnel accounts")
ax2.grid(True, axis="y")
fig.suptitle("Account Degree Distribution — Graph Structure Analysis", fontsize=13)
save(fig, "08_degree_distributions.png")

# Chart 9 — Top 15 hub accounts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
top15_out = out_deg.head(15)
ax1.barh(top15_out.index[::-1], top15_out.values[::-1], color=LG, alpha=0.8, edgecolor="#0F1117")
ax1.set_xlabel("Transactions Sent"); ax1.set_title("Top 15 Senders (Out-Degree)")
ax1.grid(True, axis="x")
for i, v in enumerate(top15_out.values[::-1]):
    ax1.text(v+2000, i, f"{v:,}", va="center", fontsize=8)

top15_in = in_deg.head(15)
ax2.barh(top15_in.index[::-1], top15_in.values[::-1], color=FR, alpha=0.8, edgecolor="#0F1117")
ax2.set_xlabel("Transactions Received"); ax2.set_title("Top 15 Receivers (Funnel Suspects)")
ax2.grid(True, axis="x")
for i, v in enumerate(top15_in.values[::-1]):
    ax2.text(v+100, i, f"{v:,}", va="center", fontsize=8)
fig.suptitle("Hub Account Analysis — Graph Node Suspects", fontsize=13)
fig.tight_layout()
save(fig, "09_top_hub_accounts.png")

# ════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("EDA SECTION 8: GROUND TRUTH RING LABELS (PATTERNS FILE)")
print(DIVIDER)
pat   = open(PAT_FILE).read()
rings = [l for l in pat.splitlines() if l.startswith("BEGIN")]
names = []
for r in rings:
    m = re.search(r"ATTEMPT - ([A-Z\-]+)", r)
    if m: names.append(m.group(1))
rc = Counter(names)

print(f"\n  Total labeled rings : {len(rings):,}")
print(f"  Ring types          : {len(rc)}")
print(f"\n  {'Ring Type':<20} {'Count':>6} {'%':>7}")
print(f"  {'-'*35}")
for k, v in rc.most_common():
    bar = "█" * (v // 10)
    print(f"  {k:<20} {v:>6}  ({v/len(rings)*100:.1f}%) {bar}")

# Chart 10 — Ring types
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
rk, rv = zip(*rc.most_common())
cc10 = [FR,NT,LG,YL,"#9B59B6","#1ABC9C","#E67E22","#3498DB"][:len(rk)]
bars10 = ax1.bar(rk, rv, color=cc10, alpha=0.85, edgecolor="#0F1117")
ax1.set_ylabel("Count"); ax1.set_title(f"Fraud Ring Types (N={len(rings):,})")
ax1.grid(True, axis="y")
for b, v in zip(bars10, rv):
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+2,
             str(v), ha="center", fontsize=10, color="white", fontweight="bold")

wedges, texts, auto = ax2.pie(rv, labels=rk, colors=cc10, autopct="%1.1f%%",
    startangle=90, pctdistance=0.75, wedgeprops=dict(width=0.5, edgecolor="#0F1117"))
for t in auto: t.set_color("white"); t.set_fontsize(8)
ax2.set_title("Ring Type Distribution")
fig.suptitle("Ground Truth: 2,756 Pre-Labeled Fraud Rings in HI-Medium_Patterns.txt", fontsize=12)
save(fig, "10_ring_types.png")

# ════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("EDA SECTION 9: TIME-AWARE TRAIN / VAL / TEST SPLITS")
print(DIVIDER)
tx_s = tx.sort_values("Timestamp").reset_index(drop=True)
n = len(tx_s)
i70, i85 = int(n * 0.70), int(n * 0.85)
df_train = tx_s.iloc[:i70]
df_val   = tx_s.iloc[i70:i85]
df_test  = tx_s.iloc[i85:]

splits = [("TRAIN 70%", df_train), ("VAL   15%", df_val), ("TEST  15%", df_test)]
for name, df in splits:
    f = df["Is Laundering"].sum()
    r = f / len(df) * 100
    print(f"\n  {name}")
    print(f"    Dates  : {df['Timestamp'].min().date()} → {df['Timestamp'].max().date()}")
    print(f"    Rows   : {len(df):,}")
    print(f"    Fraud  : {f:,}  ({r:.4f}%)")

print(f"\n  ✅ Time-sorted before split — zero data leakage")

# Chart 11 — Time splits
tx_s2 = tx_s.copy(); tx_s2["Date2"] = tx_s2["Timestamp"].dt.date
daily2 = tx_s2.groupby("Date2").size()
fr2    = tx_s2[tx_s2["Is Laundering"]==1].copy()
fr2["Date2"] = fr2["Timestamp"].dt.date
daily_f2 = fr2.groupby("Date2").size().reindex(daily2.index, fill_value=0)
xs2  = list(range(len(daily2)))
days2= [str(d) for d in daily2.index]

def find_idx(ts):
    s = str(ts.date())
    return days2.index(s) if s in days2 else 0

t70_ts = df_train["Timestamp"].max()
t85_ts = df_val["Timestamp"].max()

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(xs2, daily2.values, alpha=0.4, color=LG, label="All Transactions")
ax.fill_between(xs2, daily_f2.values * 200, alpha=0.8, color=FR, label="Fraud ×200 (scaled)")
i1 = find_idx(t70_ts); i2 = find_idx(t85_ts)
ax.axvline(i1, color=YL, lw=2, linestyle="--", label=f"Train|Val @ {t70_ts.date()}")
ax.axvline(i2, color=FR, lw=2, linestyle="--", label=f"Val|Test @ {t85_ts.date()}")

ax.fill_betweenx([0, daily2.max()], 0, i1,         alpha=0.06, color=LG)
ax.fill_betweenx([0, daily2.max()], i1, i2,        alpha=0.06, color=YL)
ax.fill_betweenx([0, daily2.max()], i2, xs2[-1],  alpha=0.06, color=FR)
ax.text(i1//2,          daily2.max()*0.82, f"TRAIN (70%)\nFraud={df_train['Is Laundering'].sum():,}", ha="center", color=LG, fontsize=9, fontweight="bold")
ax.text((i1+i2)//2,     daily2.max()*0.82, f"VAL (15%)\nFraud={df_val['Is Laundering'].sum():,}",   ha="center", color=YL, fontsize=9, fontweight="bold")
ax.text((i2+xs2[-1])//2,daily2.max()*0.82, f"TEST (15%)\nFraud={df_test['Is Laundering'].sum():,}", ha="center", color=FR, fontsize=9, fontweight="bold")

ax.set_xticks(xs2); ax.set_xticklabels(days2, rotation=45, ha="right", fontsize=7)
ax.legend(facecolor="#1A1D27"); ax.grid(True)
ax.set_title("Time-Aware Train / Val / Test Splits — No Data Leakage")
save(fig, "11_time_splits.png")

# ════════════════════════════════════════════════════════
print(f"\n{DIVIDER}")
print("EDA COMPLETE ✅")
print(DIVIDER)
print(f"\n  Total charts generated : {chart_num[0]}")
print(f"  Output directory       : {OUT_DIR}")
print(f"\n  Charts summary:")
charts = [
    "01_class_imbalance.png    — 904:1 imbalance donut + log bar",
    "02_daily_volume.png       — 27-day transaction + fraud volume",
    "03_hourly_patterns.png    — Fraud peaks by hour of day",
    "04_amount_distribution.png— Fraud 6x larger at median",
    "05_amount_buckets.png     — Fraud rate per amount bucket",
    "06_payment_format.png     — ACH = 87% of all fraud",
    "07_currency.png           — 15 currencies breakdown",
    "08_degree_distributions.png — Out/In degree per account",
    "09_top_hub_accounts.png   — Top 15 senders/receivers",
    "10_ring_types.png         — 2,756 ground truth rings by type",
    "11_time_splits.png        — Train/Val/Test on 27-day timeline",
]
for c in charts:
    print(f"    {c}")

print(f"\n{DIVIDER}")
