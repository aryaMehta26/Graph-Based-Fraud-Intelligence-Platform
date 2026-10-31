"""
Leak-free, time-ordered pattern features for every transaction in the clean splits.

Every feature for a transaction at minute t uses only transactions from earlier
minutes (strictly before t), so nothing from the future or from the test period
leaks into training rows, and test rows only see their own past.

Reads:  ../clean_split/data/processed/split_{train,val,test}.parquet   (read-only)
Writes: data/pattern_{train,val,test}.parquet  (feature columns only, same row order)

Feature groups (u = sender, v = receiver):
  velocity      u out-count / v in-count over 1h, 1d, 7d and all prior history
  pass-through  u's recent incoming money, minutes since u last received, amount vs u's
                recent incoming amounts (laundering forwards what it just received)
  counterparty  distinct partners so far and new partners in the last 7d
  pair          prior u->v count, prior v->u count (2-cycle)
  cycles        v->w->u (3-cycle) and v->w->x->u (4-cycle) paths in the last 7d with
                non-decreasing timestamps, i.e. this payment closes a loop
  scatter-gather  other senders into v that share a funding source with u (gather),
                  other recipients of u that pay the same account as v (scatter)
Cycle and scatter-gather counts use each account's 16 most recent in/out edges.
"""

import os
import time

import numpy as np
import pandas as pd
from numba import njit, prange

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(HERE, "..", "clean_split", "data", "processed")
OUT = os.path.join(HERE, "data")
os.makedirs(OUT, exist_ok=True)

HOUR, DAY, WEEK = 60, 1440, 10080
INF = np.int64(1) << 40
K = 16          # recent edges kept per account for path features
K_SMALL = 8     # fan-out explored at the inner levels of 4-hop searches

t0 = time.time()
def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
cols = ["Timestamp", "src_acct", "dst_acct", "log_amount"]
parts, sizes = [], {}
for name in ["train", "val", "test"]:
    df = pd.read_parquet(os.path.join(SPLITS, f"split_{name}.parquet"), columns=cols)
    sizes[name] = len(df)
    parts.append(df)
tx = pd.concat(parts, ignore_index=True)
del parts
LIMIT = int(os.getenv("PF_LIMIT", "0"))
if LIMIT:                                            # smoke test on the first rows only
    tx = tx.iloc[:LIMIT]
    sizes = {"train": LIMIT, "val": 0, "test": 0}
    OUT = os.path.join(OUT, "smoke"); os.makedirs(OUT, exist_ok=True)
assert tx["Timestamp"].is_monotonic_increasing, "splits must be time-ordered slices"
log(f"loaded {len(tx):,} rows {sizes}")

minute = ((tx["Timestamp"].astype("int64") // 60_000_000)).to_numpy()
minute = (minute - minute.min()).astype(np.int64)
codes, uniques = pd.factorize(pd.concat([tx["src_acct"], tx["dst_acct"]], ignore_index=True))
n = len(tx)
N = len(uniques)
u = codes[:n].astype(np.int64)
v = codes[n:].astype(np.int64)
amt = tx["log_amount"].to_numpy(np.float64)
del tx, codes, uniques
log(f"{N:,} accounts")


# ---------------------------------------------------------------------------
# Window queries over per-key sorted event lists
# ---------------------------------------------------------------------------
def build_csr(keys, n_keys, times, vals):
    order = np.argsort(keys, kind="stable")          # rows are time-sorted, so each key's times are too
    ptr = np.zeros(n_keys + 1, np.int64)
    np.cumsum(np.bincount(keys, minlength=n_keys), out=ptr[1:])
    csum = np.zeros(len(keys) + 1, np.float64)
    np.cumsum(vals[order], out=csum[1:])
    return ptr, times[order], csum


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
def window_query(ptr, ts, csum, qkey, qt, W, cnt, tot, gap):
    for i in prange(len(qkey)):
        k = qkey[i]
        lo, hi = ptr[k], ptr[k + 1]
        e = lower_bound(ts, lo, hi, qt[i])            # events strictly before this minute
        s = lower_bound(ts, lo, e, qt[i] - W)
        cnt[i] = e - s
        tot[i] = csum[e] - csum[s]
        gap[i] = qt[i] - ts[e - 1] if e > lo else np.nan


def query(csr, qkey, W):
    ptr, ts, csum = csr
    cnt = np.empty(len(qkey), np.float32)
    tot = np.empty(len(qkey), np.float64)
    gap = np.empty(len(qkey), np.float32)
    window_query(ptr, ts, csum, qkey, minute, W, cnt, tot, gap)
    return cnt, tot, gap


feat = {}
graph = u != v
ones = np.ones(n, np.float64)

# velocity and pass-through
out_csr = build_csr(u, N, minute, np.where(graph, amt, 0.0))
in_csr = build_csr(v, N, minute, np.where(graph, amt, 0.0))
log("event lists built")

for tag, W in [("1h", HOUR), ("1d", DAY), ("7d", WEEK)]:
    feat[f"src_out_cnt_{tag}"], s, _ = query(out_csr, u, W)
    if tag == "7d":
        with np.errstate(invalid="ignore", divide="ignore"):
            feat["src_out_amt_dev_7d"] = (amt - s / feat["src_out_cnt_7d"]).astype(np.float32)
    feat[f"dst_in_cnt_{tag}"], _, _ = query(in_csr, v, W)
feat["src_out_total"], _, feat["src_out_gap"] = query(out_csr, u, INF)
feat["dst_in_total"], _, feat["dst_in_gap"] = query(in_csr, v, INF)

feat["src_in_cnt_1d"], _, _ = query(in_csr, u, DAY)
feat["src_in_cnt_7d"], s, _ = query(in_csr, u, WEEK)
with np.errstate(invalid="ignore", divide="ignore"):
    feat["src_in_amt_dev_7d"] = (amt - s / feat["src_in_cnt_7d"]).astype(np.float32)
feat["src_in_total"], _, feat["src_in_gap"] = query(in_csr, u, INF)
feat["dst_out_cnt_1d"], _, _ = query(out_csr, v, DAY)
feat["dst_out_total"], _, _ = query(out_csr, v, INF)
del out_csr, in_csr
log("velocity / pass-through done")

# pairs and counterparties
pair_key = u * N + v
uniq_pairs, pair_id = np.unique(pair_key, return_inverse=True)
P = len(uniq_pairs)
pair_csr = build_csr(pair_id, P, minute, ones)
feat["pair_prior_cnt"], _, _ = query(pair_csr, pair_id, INF)

rev_key = v * N + u
pos = np.minimum(np.searchsorted(uniq_pairs, rev_key), P - 1)
has_rev = uniq_pairs[pos] == rev_key
rev_id = np.where(has_rev, pos, 0)
rc, _, _ = query(pair_csr, rev_id, INF)
feat["rev_pair_prior_cnt"] = np.where(has_rev, rc, 0).astype(np.float32)
del pair_csr, rev_key, pos, has_rev, rev_id

first_t = np.full(P, INF, np.int64)
np.minimum.at(first_t, pair_id, minute)
p_src = (uniq_pairs // N).astype(np.int64)
p_dst = (uniq_pairs % N).astype(np.int64)
keep = p_src != p_dst
for name, keys, q in [("src_distinct_out", p_src, u), ("dst_distinct_in", p_dst, v)]:
    order = np.argsort(first_t[keep], kind="stable")
    csr = build_csr(keys[keep][order], N, first_t[keep][order], np.ones(int(keep.sum())))
    feat[f"{name}_total"], _, _ = query(csr, q, INF)
    feat[f"{name}_new_7d"], _, _ = query(csr, q, WEEK)
del uniq_pairs, pair_id, first_t, p_src, p_dst, keep, pair_key
log("pair / counterparty done")


# ---------------------------------------------------------------------------
# Path features: cycles and scatter-gather, sequential over time
# ---------------------------------------------------------------------------
@njit
def _has(nb, tt, node, x, tmin):
    for j in range(nb.shape[1]):
        if nb[node, j] == x and tt[node, j] >= tmin:
            return True
    return False


@njit
def _shares(nb, tt, a, b, tlo):
    # do accounts a and b have a common neighbour (both edges since tlo)?
    for j in range(nb.shape[1]):
        x = nb[a, j]
        if x >= 0 and tt[a, j] >= tlo and _has(nb, tt, b, x, tlo):
            return True
    return False


@njit
def path_features(u, v, minute, N, K, KS, W, cyc3, cyc4, gather, scatter):
    out_nb = np.full((N, K), -1, np.int64); out_t = np.full((N, K), -(1 << 40), np.int64)
    in_nb = np.full((N, K), -1, np.int64);  in_t = np.full((N, K), -(1 << 40), np.int64)
    out_p = np.zeros(N, np.int64);          in_p = np.zeros(N, np.int64)
    n = len(u)
    pending = 0
    for i in range(n):
        t = minute[i]
        if i > 0 and t != minute[i - 1]:           # publish the previous minute's edges
            for r in range(pending, i):
                a, b, tr = u[r], v[r], minute[r]
                if a != b:
                    out_nb[a, out_p[a]] = b; out_t[a, out_p[a]] = tr; out_p[a] = (out_p[a] + 1) % K
                    in_nb[b, in_p[b]] = a;   in_t[b, in_p[b]] = tr;   in_p[b] = (in_p[b] + 1) % K
            pending = i
        a, b = u[i], v[i]
        if a == b:
            continue
        tlo = t - W
        c3 = 0; c4 = 0
        for j in range(K):
            w = out_nb[b, j]; t1 = out_t[b, j]
            if w < 0 or t1 < tlo or w == a:
                continue
            if _has(out_nb, out_t, w, a, t1):       # b -> w -> a
                c3 += 1
            # b -> w -> x -> a, limited to w's KS most recent out-edges
            for jj in range(KS):
                idx = (out_p[w] - 1 - jj) % K
                x = out_nb[w, idx]; t2 = out_t[w, idx]
                if x < 0 or t2 < t1 or x == a or x == b:
                    continue
                if _has(in_nb, in_t, a, x, t2):
                    c4 += 1
        cyc3[i] = c3; cyc4[i] = c4
        g = 0
        for jj in range(KS):                         # other senders into b funded by a common source with a
            idx = (in_p[b] - 1 - jj) % K
            x = in_nb[b, idx]
            if x < 0 or x == a or in_t[b, idx] < tlo:
                continue
            if _shares(in_nb, in_t, a, x, tlo):
                g += 1
        gather[i] = g
        s = 0
        for jj in range(KS):                         # other recipients of a that pay an account b also pays
            idx = (out_p[a] - 1 - jj) % K
            y = out_nb[a, idx]
            if y < 0 or y == b or out_t[a, idx] < tlo:
                continue
            if _shares(out_nb, out_t, y, b, tlo):
                s += 1
        scatter[i] = s


cyc3 = np.zeros(n, np.float32); cyc4 = np.zeros(n, np.float32)
gather = np.zeros(n, np.float32); scatter = np.zeros(n, np.float32)
path_features(u, v, minute, N, K, K_SMALL, WEEK, cyc3, cyc4, gather, scatter)
feat.update(cyc3_cnt=cyc3, cyc4_cnt=cyc4, gather_cnt=gather, scatter_cnt=scatter)
log("cycles / scatter-gather done")


# ---------------------------------------------------------------------------
# Save per split
# ---------------------------------------------------------------------------
df = pd.DataFrame({k: np.asarray(x, np.float32) for k, x in feat.items()})
start = 0
for name in ["train", "val", "test"]:
    part = df.iloc[start:start + sizes[name]].reset_index(drop=True)
    part.to_parquet(os.path.join(OUT, f"pattern_{name}.parquet"), index=False)
    start += sizes[name]
log(f"saved {df.shape[1]} features: {list(df.columns)}")
