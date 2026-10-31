"""
Graph neural network edge classifier on the clean splits (thesis comparison against XGBoost).

Model: 2-layer GraphSAGE-style message passing with PNA-style aggregation (mean + max,
separately over incoming and outgoing edges, plus log-degree), then an MLP over
[h_sender, h_receiver, edge features]. Inputs are raw transaction fields only
(amount, payment format, cross-currency, time of day/week), no hand-made pattern features.

Protocol follows Altman et al. (NeurIPS 2023): dynamic-graph snapshots.
  train graph = train edges            -> train on train edges
  val graph   = train + val edges      -> score val edges (choose thresholds, early stopping)
  test graph  = all edges              -> score test edges
Neighbours: up to S randomly sampled incoming and outgoing edges per account in each snapshot.
Note: within a snapshot an edge can see neighbours later in the same period (the paper's setup),
which is looser than the XGBoost pattern features (strictly earlier minutes only).

Reads ../clean_split/data/processed/split_*.parquet (read-only). Writes results/, scores/, model.pt.
Run with ../.venv-gnn/bin/python.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pattern_features"))
from metrics import evaluate_scores  # noqa: E402

SPLITS = os.path.join(HERE, "..", "clean_split", "data", "processed")
OUT = os.path.join(HERE, "results"); SCORES = os.path.join(HERE, "scores")
os.makedirs(OUT, exist_ok=True); os.makedirs(SCORES, exist_ok=True)

S = int(os.getenv("GNN_S", "10"))           # sampled neighbours per direction
H = int(os.getenv("GNN_H", "64"))
EPOCHS = int(os.getenv("GNN_EPOCHS", "20"))
NEG_PER_POS = int(os.getenv("GNN_NEG", "20"))
BATCH = 2048
LIMIT = int(os.getenv("GNN_LIMIT", "0"))    # smoke test on a prefix of each split
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42); rng = np.random.default_rng(42)
FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]

t0 = time.time()
def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
cols = ["Timestamp", "src_acct", "dst_acct", "log_amount", "Payment Format", "is_cross_currency",
        "hour", "dow", "is_weekend", "Is Laundering"]
parts, sizes = [], []
for name in ["train", "val", "test"]:
    df = pd.read_parquet(os.path.join(SPLITS, f"split_{name}.parquet"), columns=cols)
    if LIMIT:
        df = df.iloc[:LIMIT]
    parts.append(df); sizes.append(len(df))
tx = pd.concat(parts, ignore_index=True); del parts
n = len(tx)
b1, b2 = sizes[0], sizes[0] + sizes[1]
codes, _ = pd.factorize(pd.concat([tx["src_acct"], tx["dst_acct"]], ignore_index=True))
src, dst = codes[:n].astype(np.int64), codes[n:].astype(np.int64)
N = int(codes.max()) + 1
y = tx["Is Laundering"].to_numpy(np.int64)
ts = pd.to_datetime(tx["Timestamp"])

la = tx["log_amount"].to_numpy(np.float32)
la = (la - la[:b1].mean()) / la[:b1].std()
fmt = np.stack([(tx["Payment Format"] == f).to_numpy(np.float32) for f in FORMATS], 1)
hr = tx["hour"].to_numpy(np.float32) * (2 * np.pi / 24)
dw = tx["dow"].to_numpy(np.float32) * (2 * np.pi / 7)
xe = np.column_stack([la, fmt, tx["is_cross_currency"].to_numpy(np.float32), np.sin(hr), np.cos(hr),
                      np.sin(dw), np.cos(dw), tx["is_weekend"].to_numpy(np.float32)]).astype(np.float32)
DE = xe.shape[1]
val_ts = ts.iloc[b1:b2]
del tx
log(f"{n:,} edges, {N:,} accounts, {DE} edge features, device {DEV}")

XE = torch.from_numpy(xe).to(DEV)
SRC = torch.from_numpy(src).to(DEV); DST = torch.from_numpy(dst).to(DEV)


def snapshot(end):
    """Sampled neighbour tables and per-day log degrees over edges [0, end)."""
    days = max((ts.iloc[end - 1] - ts.iloc[0]).total_seconds() / 86400, 1.0)
    idx = np.arange(end)
    idx = idx[src[:end] != dst[:end]]
    tables = []
    for key in (dst, src):                         # incoming table keyed by receiver, outgoing by sender
        perm = idx[rng.permutation(len(idx))]
        perm = perm[np.argsort(key[perm], kind="stable")]
        k = key[perm]
        starts = np.searchsorted(k, np.arange(N))
        rank = np.arange(len(perm)) - starts[k]
        keep = rank < S
        tab = np.full((N, S), -1, np.int64)
        tab[k[keep], rank[keep]] = perm[keep]
        tables.append(torch.from_numpy(tab).to(DEV))
    deg_in = np.bincount(dst[idx], minlength=N); deg_out = np.bincount(src[idx], minlength=N)
    xn = np.column_stack([np.log1p(deg_in / days), np.log1p(deg_out / days)]).astype(np.float32)
    return tables[0], tables[1], torch.from_numpy(xn).to(DEV)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Layer(nn.Module):
    def __init__(self, d_node, d_out):
        super().__init__()
        self.msg_in = nn.Linear(DE + d_node, d_out)
        self.msg_out = nn.Linear(DE + d_node, d_out)
        self.upd = nn.Linear(d_node + 4 * d_out + 2, d_out)

    @staticmethod
    def agg(m, mask):
        mf = mask.unsqueeze(-1).float()
        cnt = mf.sum(1).clamp(min=1)
        mean = (m * mf).sum(1) / cnt
        mx = m.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values * (mask.any(1, keepdim=True)).float()
        return mean, mx

    def forward(self, h_self, xn_self, e_in, h_in, e_out, h_out):
        # e_*: [B, S] edge ids (-1 pad); h_*: [B, S, d_node] other-endpoint states
        mi, mo = e_in >= 0, e_out >= 0
        x_in = XE[e_in.clamp(min=0)]; x_out = XE[e_out.clamp(min=0)]
        a1, a2 = self.agg(F.relu(self.msg_in(torch.cat([x_in, h_in], -1))), mi)
        a3, a4 = self.agg(F.relu(self.msg_out(torch.cat([x_out, h_out], -1))), mo)
        return F.relu(self.upd(torch.cat([h_self, a1, a2, a3, a4, xn_self], -1)))


class EdgeGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = Layer(2, H)
        self.l2 = Layer(H, H)
        self.head = nn.Sequential(nn.Linear(2 * H + DE, H), nn.ReLU(), nn.Dropout(0.1), nn.Linear(H, 1))

    def layer1(self, nodes, snap):
        tin, tout, xn = snap
        e_in, e_out = tin[nodes], tout[nodes]
        return self.l1(xn[nodes], xn[nodes], e_in, xn[SRC[e_in.clamp(min=0)]], e_out, xn[DST[e_out.clamp(min=0)]])

    def layer2(self, nodes, snap, h1_all=None):
        tin, tout, xn = snap
        e_in, e_out = tin[nodes], tout[nodes]
        nb_in, nb_out = SRC[e_in.clamp(min=0)], DST[e_out.clamp(min=0)]
        if h1_all is None:                          # training: compute neighbour states on the fly
            uniq, inv = torch.unique(torch.cat([nodes, nb_in.flatten(), nb_out.flatten()]), return_inverse=True)
            h1u = self.layer1(uniq, snap)
            B = len(nodes)
            h_self = h1u[inv[:B]]
            h_in = h1u[inv[B:B + B * S]].view(B, S, -1); h_out = h1u[inv[B + B * S:]].view(B, S, -1)
        else:
            h_self, h_in, h_out = h1_all[nodes], h1_all[nb_in], h1_all[nb_out]
        return self.l2(h_self, xn[nodes], e_in, h_in, e_out, h_out)

    def score(self, edges, h_src, h_dst):
        return self.head(torch.cat([h_src, h_dst, XE[edges]], -1)).squeeze(-1)

    def forward(self, edges, snap):
        nodes, inv = torch.unique(torch.cat([SRC[edges], DST[edges]]), return_inverse=True)
        h = self.layer2(nodes, snap)
        B = len(edges)
        return self.score(edges, h[inv[:B]], h[inv[B:]])


@torch.no_grad()
def predict(model, snap, lo, hi, chunk=131072):
    model.eval()
    all_nodes = torch.arange(N, device=DEV)
    h1 = torch.cat([model.layer1(all_nodes[i:i + chunk], snap) for i in range(0, N, chunk)])
    h2 = torch.cat([model.layer2(all_nodes[i:i + chunk], snap, h1) for i in range(0, N, chunk)])
    out = []
    for i in range(lo, hi, 4 * chunk):
        e = torch.arange(i, min(i + 4 * chunk, hi), device=DEV)
        out.append(torch.sigmoid(model.score(e, h2[SRC[e]], h2[DST[e]])).cpu())
    return torch.cat(out).numpy()


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
from sklearn.metrics import average_precision_score  # noqa: E402

snap_tr, snap_va = snapshot(b1), snapshot(b2)
log("train and val snapshots built")
model = EdgeGNN().to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
pos = np.where(y[:b1] == 1)[0]; neg_all = np.where(y[:b1] == 0)[0]
best, best_state, patience, history = -1.0, None, 0, []
for epoch in range(1, EPOCHS + 1):
    model.train()
    ep = np.concatenate([pos, rng.choice(neg_all, NEG_PER_POS * len(pos), replace=False)])
    rng.shuffle(ep)
    tot = 0.0
    for i in range(0, len(ep), BATCH):
        e = torch.from_numpy(ep[i:i + BATCH]).to(DEV)
        loss = F.binary_cross_entropy_with_logits(model(e, snap_tr), torch.from_numpy(y[ep[i:i + BATCH]]).float().to(DEV))
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(e)
    p_va = predict(model, snap_va, b1, b2)
    ap = average_precision_score(y[b1:b2], p_va)
    history.append({"epoch": epoch, "loss": tot / len(ep), "val_pr_auc": float(ap)})
    log(f"epoch {epoch:2d}  loss {tot / len(ep):.4f}  val PR-AUC {ap:.4f}")
    if ap > best:
        best, patience = ap, 0
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        patience += 1
        if patience >= 3:
            break

model.load_state_dict(best_state)
torch.save(best_state, os.path.join(HERE, "model.pt"))
del snap_tr
snap_te = snapshot(n)
p_va = predict(model, snap_va, b1, b2); p_te = predict(model, snap_te, b2, n)
np.save(os.path.join(SCORES, "gnn_val.npy"), p_va); np.save(os.path.join(SCORES, "gnn_test.npy"), p_te)
r = evaluate_scores(y[b1:b2], p_va, y[b2:], p_te, val_ts)
r.update(history=history, params={"S": S, "H": H, "neg_per_pos": NEG_PER_POS, "batch": BATCH})
json.dump(r, open(os.path.join(OUT, "gnn_results.json" if not LIMIT else "gnn_smoke.json"), "w"), indent=2)
log(f"GNN test PR-AUC {r['pr_auc']:.4f}  ROC {r['roc_auc']:.4f}")
for k, x in r["test_at_thresholds"].items():
    if x:
        print(f"  {k:28} alerts {x['alerts']:>8,}  TP {x['tp']:>5,}  FP {x['fp']:>8,}  "
              f"precision {x['precision']:6.2%}  recall {x['recall']:6.2%}  F1 {x['f1']:.4f}")
print("  top-k:", ", ".join(f"{k}: {x['precision']:.1%}/{x['recall']:.1%}" for k, x in r["test_top_k"].items()))
