"""
Second batch of leak-free features (same rule: only transactions from earlier minutes).

Reads:  ../clean_split/data/processed/split_{train,val,test}.parquet   (read-only)
Writes: data/extra_{train,val,test}.parquet

  account age      minutes since sender / receiver first appeared anywhere (NaN = brand new)
  amount shape     round to 100 / 1,000, whole-currency amounts, paid vs received gap
  same-amount      prior payments by the sender / into the receiver with the identical amount (7d)
  bank             same-bank payment, sender bank and receiver bank activity in the last hour
  reach            how many distinct accounts the receiver already pays, the sender already receives from
"""

import os
import time

import numpy as np
import pandas as pd
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(HERE, "..", "clean_split", "data", "processed")
OUT = os.path.join(HERE, "data")
HOUR, WEEK = 60, 10080
INF = np.int64(1) << 40

t0 = time.time()
def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


cols = ["Timestamp", "From Bank", "src_acct", "To Bank", "dst_acct", "Amount Paid", "Amount Received"]
parts, sizes = [], {}
for name in ["train", "val", "test"]:
    df = pd.read_parquet(os.path.join(SPLITS, f"split_{name}.parquet"), columns=cols)
    sizes[name] = len(df); parts.append(df)
tx = pd.concat(parts, ignore_index=True); del parts
assert tx["Timestamp"].is_monotonic_increasing
n = len(tx)
minute = (tx["Timestamp"].astype("int64") // 60_000_000).to_numpy()
minute = (minute - minute.min()).astype(np.int64)
codes, _ = pd.factorize(pd.concat([tx["src_acct"], tx["dst_acct"]], ignore_index=True))
u, v = codes[:n].astype(np.int64), codes[n:].astype(np.int64)
N = int(codes.max()) + 1
paid = tx["Amount Paid"].to_numpy(np.float64)
recv = tx["Amount Received"].to_numpy(np.float64)
tb_codes, tb_uni = pd.factorize(pd.concat([tx["From Bank"], tx["To Bank"]], ignore_index=True))
fbank, tbank = tb_codes[:n].astype(np.int64), tb_codes[n:].astype(np.int64)
B = len(tb_uni)
log(f"loaded {n:,} rows, {N:,} accounts, {B:,} banks")


def build_csr(keys, n_keys, times):
    order = np.argsort(keys, kind="stable")
    ptr = np.zeros(n_keys + 1, np.int64)
    np.cumsum(np.bincount(keys, minlength=n_keys), out=ptr[1:])
    return ptr, times[order]


@njit(inline="always")
def lower_bound(a, lo, hi, x):
    while lo < hi:
        mid = (lo + hi) >> 1
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(parallel=True)
def count_before(ptr, ts, qkey, qt, W, out):
    for i in prange(len(qkey)):
        lo, hi = ptr[qkey[i]], ptr[qkey[i] + 1]
        e = lower_bound(ts, lo, hi, qt[i])
        out[i] = e - lower_bound(ts, lo, e, qt[i] - W)


def count(csr, qkey, W):
    out = np.empty(len(qkey), np.float32)
    count_before(csr[0], csr[1], qkey, minute, W, out)
    return out


feat = {}

# account age: first minute each account appears on either side
first = np.full(N, INF, np.int64)
np.minimum.at(first, u, minute); np.minimum.at(first, v, minute)
for side, a in [("src", u), ("dst", v)]:
    age = (minute - first[a]).astype(np.float32)
    age[age <= 0] = np.nan                          # first seen this minute: no history
    feat[f"{side}_age_min"] = age
log("account age done")

# amount shape
feat["amt_round_100"] = (np.round(paid, 2) % 100 == 0).astype(np.float32)
feat["amt_round_1000"] = (np.round(paid, 2) % 1000 == 0).astype(np.float32)
feat["amt_whole"] = (np.round(paid, 2) % 1 == 0).astype(np.float32)
with np.errstate(divide="ignore", invalid="ignore"):
    feat["amt_paid_recv_logratio"] = np.log1p(paid).astype(np.float32) - np.log1p(recv).astype(np.float32)

# same-amount repeats
cents = np.round(paid * 100).astype(np.int64)
for side, a in [("src", u), ("dst", v)]:
    key = pd.factorize(pd.Series(a) * (1 << 40) + cents)[0].astype(np.int64)
    feat[f"{side}_same_amt_7d"] = count(build_csr(key, int(key.max()) + 1, minute), key, WEEK)
log("amount features done")

# bank
feat["same_bank"] = (tx["From Bank"].to_numpy() == tx["To Bank"].to_numpy()).astype(np.float32)
feat["from_bank_out_1h"] = count(build_csr(fbank, B, minute), fbank, HOUR)
feat["to_bank_in_1h"] = count(build_csr(tbank, B, minute), tbank, HOUR)
log("bank features done")

# reach: distinct partners, keyed by first occurrence of each pair
pair = u * N + v
uniq, pid = np.unique(pair, return_inverse=True)
pfirst = np.full(len(uniq), INF, np.int64); np.minimum.at(pfirst, pid, minute)
ps, pd_ = uniq // N, uniq % N
keep = ps != pd_
o = np.argsort(pfirst[keep], kind="stable")
out_by_src = build_csr(ps[keep][o], N, pfirst[keep][o])
in_by_dst = build_csr(pd_[keep][o], N, pfirst[keep][o])
feat["dst_distinct_out_total"] = count(out_by_src, v, INF)
feat["src_distinct_in_total"] = count(in_by_dst, u, INF)
log("reach done")

df = pd.DataFrame({k: np.asarray(x, np.float32) for k, x in feat.items()})
start = 0
for name in ["train", "val", "test"]:
    df.iloc[start:start + sizes[name]].reset_index(drop=True).to_parquet(os.path.join(OUT, f"extra_{name}.parquet"), index=False)
    start += sizes[name]
log(f"saved {df.shape[1]} features: {list(df.columns)}")
