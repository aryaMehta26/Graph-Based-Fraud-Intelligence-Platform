# Project FAQ — Graph-Based Fraud Intelligence Platform
DATA 298A | Team 2 | San José State University

Answers to common questions about the pipeline, model decisions, and graph design.

---

## Metrics

### What does PR-AUC mean and why do we use it instead of accuracy?

Accuracy is useless on this dataset because 99.89% of transactions are legitimate. A model that always predicts "not fraud" gets 99.89% accuracy but catches zero fraud.

**PR-AUC (Precision-Recall Area Under Curve)** measures how well the model ranks fraud transactions vs. legitimate ones, regardless of class balance. Higher is better. 0.5 on PR-AUC is very good for a 1-in-904 fraud rate; 1.0 is perfect.

### What is ROC-AUC?

ROC-AUC measures the probability that the model ranks a random fraud transaction higher than a random legitimate one. 0.9 means the model gets this right 90% of the time. It's less sensitive than PR-AUC for highly imbalanced datasets.

### What is F1 score?

F1 = harmonic mean of Precision and Recall. It balances two tradeoffs:
- **Precision**: of all transactions flagged as fraud, how many actually are?
- **Recall**: of all actual fraud transactions, how many did we catch?

High recall (catching almost all fraud) with low precision means lots of false alarms. The right balance depends on business cost — for AML, high recall is usually preferred.

### Why is F1 so low (0.026) even though ROC-AUC is 0.98?

F1 depends on a threshold — the probability cutoff above which we call something fraud. At default threshold 0.30, the model casts a wide net (high recall, low precision). Running `tune_threshold.py` finds the threshold that maximises F1 on the validation set. The ROC-AUC/PR-AUC numbers are threshold-independent and are the primary metrics.

---

## Graph & Neo4j

### Why 8 FraudPattern nodes instead of 2,756?

The 8 nodes represent pattern **types** (CYCLE, FAN-IN, FAN-OUT, STACK, SCATTER-GATHER, GATHER-SCATTER, BIPARTITE, RANDOM). The 2,756 individual laundering attempts are encoded as properties on the relationships:

```cypher
(:Transaction)-[:PART_OF_PATTERN {attempt_id: 142, pattern_type: "CYCLE"}]->(:FraudPattern)
```

This is standard Neo4j design — avoid creating a node for every instance of a category. To count attempts per type:

```cypher
MATCH ()-[r:PART_OF_PATTERN]->(f:FraudPattern)
RETURN f.type, count(DISTINCT r.attempt_id) AS attempts
ORDER BY attempts DESC
```

### When are FraudPattern nodes created? By the model or manually?

They are created by running `src/graph/load_fraud_patterns.py`, which reads the IBM ground-truth file `HI-Medium_Patterns.txt`. This is a one-time data loading step, not something the model does automatically. The ML model learns from labels in the CSV — the FraudPattern nodes enrich the graph for queries and future GNN work.

### Why does the graph use Account→Transaction→Account instead of Account→Account?

The IBM dataset has transactions as first-class entities with their own properties (amount, currency, payment format, timestamp). Keeping Transaction as a node preserves all this data. For community detection we build a projected account-to-account graph in Python without needing to change the Neo4j schema.

### Can teammates without Neo4j run the ML pipeline?

Yes. Scripts 05 → 07 only need the parquet files in `data/processed/`. Neo4j is only required for:
- `04_extract_graph_features.py` (degree features extraction)
- `src/graph/load_fraud_patterns.py` (fraud pattern loading)

`04b_louvain_communities.py` does **not** require Neo4j — it runs Leiden community detection in Python using `igraph` and `leidenalg` directly on the parquet files.

If `graph_features_accounts.csv` already exists (committed or shared), teammates can skip straight to script 05.

---

## Model Decisions

### Why XGBoost?

XGBoost handles tabular data well, is fast to train, supports class imbalance natively via `scale_pos_weight`, and gives interpretable feature importances via SHAP. It's the standard baseline for fraud detection before moving to GNNs.

### What is `scale_pos_weight=904`?

The dataset has a 1:904 fraud-to-legitimate ratio. Setting `scale_pos_weight=904` tells XGBoost to weight each fraud sample as if it appeared 904 times — effectively balancing the classes during training. Without this, the model ignores fraud almost entirely.

### Why PR-AUC as the eval metric during training?

XGBoost's early stopping monitors a metric on the validation set. Using `eval_metric="aucpr"` means training stops when PR-AUC stops improving, which is exactly what we care about for imbalanced fraud detection.

### Why use degree centrality from Neo4j when we could compute it from the parquets?

We could. But having it in Neo4j means it's always available for live queries, dashboards, and future real-time inference — not just batch training. The graph is the single source of truth for structural features.

### What is the feature store?

The enriched parquets in `data/processed/*_graph_enriched.parquet`. Each row is one transaction with all tabular features + graph features (degree centrality + community features) for both the sender and receiver accounts. These are what the model trains on.

---

## Community Detection

### What is Louvain/Leiden community detection?

An algorithm that groups accounts into "communities" based on who transacts with whom. Accounts that frequently send money to each other end up in the same community — like friend groups in a social network.

We use **Leiden** (an improved version of Louvain that produces better-quality, internally-connected communities).

### What is `community_fraud_rate`?

For each community, what fraction of **training** transactions involving community members are flagged as laundering. Computed from training labels only — val/test labels are never used — so it functions as a train-derived prior that the model can generalise from. A transaction is more suspicious if both parties belong to a community where a large share of historical (training) activity was fraud.

### Why did PR-AUC improve after adding community features?

`community_fraud_rate` gives the model "guilt by association" — the ability to flag transactions based on the fraud history of the surrounding network, not just the individual account's behaviour. Degree features alone tell you how connected someone is; community features tell you who they're connected to and whether those people launder money.

---

## Git & GitHub

### What do `Closes #N`, `Fixes #N`, `Resolves #N` do in a PR?

These are GitHub keywords. When a PR containing these phrases is merged into the **default branch** (main), GitHub automatically closes the referenced issue. They're case-insensitive and can appear anywhere in the PR description.

### Why is there a `prajwal/graph-model-complete` branch history in the PRs?

That was the feature branch for the full ML pipeline and review fixes. It was merged into main via PRs #15, #17, #19 and then deleted. The branch is gone; the commits and history are preserved in main.

---

## Pipeline Flow

### What is the full order to run the pipeline?

```
notebooks/02_data_cleaning.py          # Download + clean + split into parquets
src/models/04_extract_graph_features.py    # Degree features from Neo4j → CSV
src/models/04b_louvain_communities.py      # Community features from parquets → CSV
src/models/05_build_feature_store.py       # Join graph features into enriched parquets
src/models/06_train_xgboost_baseline.py    # Train tabular-only baseline
src/models/tune_threshold.py               # Find optimal decision threshold
src/models/07_train_graph_enhanced_model.py # Train graph-enhanced model
```

### Why are there two graph feature scripts (04 and 04b)?

Script 04 requires Neo4j (computes degree via Cypher). Script 04b only needs the parquets and runs Leiden community detection in Python using `igraph` and `leidenalg`. This separation means teammates without Neo4j can still get community features — they just need `graph_features_accounts.csv` from script 04 (which can be shared as a file).
