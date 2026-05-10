# Getting Started — Graph-Based Fraud Intelligence Platform
DATA 298A | Team 2 | San José State University

This guide covers two scenarios: starting completely from scratch, and the fast path if the data already exists. It also explains the Neo4j Desktop setup, model logic, current results, and where the project is headed.

---

## Prerequisites

```bash
git clone https://github.com/aryaMehta26/Graph-Based-Fraud-Intelligence-Platform.git
cd Graph-Based-Fraud-Intelligence-Platform
pip3 install -r requirements.txt
```

Copy the environment file and fill in your Neo4j credentials:
```bash
cp .env.example .env
# Edit .env: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
```

You also need a Kaggle account with API access for the IBM AML dataset (downloaded automatically by the scripts via `kagglehub`).

---

## Scenario 1: Starting Completely From Scratch

Use this path if you have no data files locally and no graph loaded in Neo4j.

### Step 1 — Download and clean data

```bash
python3 notebooks/02_data_cleaning.py
```

Downloads the IBM AML HI-Medium dataset (~2GB) via Kaggle, cleans it, and produces:

```
data/processed/
    split_train.parquet    (22.3M rows, 70%)
    split_val.parquet      (4.8M rows, 15%)
    split_test.parquet     (4.8M rows, 15%)
```

The split is **time-ordered** — train uses the first 70% of days, val the next 15%, test the last 15%. This prevents future transactions from being placed directly into the training split at the raw-data stage. However, later pipeline stages may compute or join graph features on the full graph (**transductive**), which can still incorporate future validation/test edges; so the split alone does not guarantee end-to-end leakage isolation for all downstream features.

### Step 2 — Load the graph into Neo4j

This step requires Neo4j Desktop running with an empty database. See the **Neo4j Setup** section below.

```bash
python3 src/graph/bulk_load_cypher.py
```

Loads ~31.9M transactions as a graph:
- **1.76M Account nodes**
- **31.9M Transaction nodes**
- **63.8M relationships** (SENT + RECEIVED_BY)

Takes 2-4 hours depending on hardware. After loading, your graph will use ~8.6GB of storage.

### Step 3 — Extract degree features from Neo4j

```bash
python3 src/models/04_extract_graph_features.py
```

Connects to Neo4j, counts how many transactions each account sent and received, and writes:

```
data/processed/graph_features_accounts.csv   (70MB, 1.76M accounts)
```

Columns: `account_id, out_degree, in_degree, total_degree, degree_centrality`

Takes ~5-10 minutes.

### Step 4 — Run Leiden community detection

```bash
python3 src/models/04b_louvain_communities.py
```

Builds an account-to-account graph from the parquets (no Neo4j needed), runs Leiden community detection, and adds community columns to `graph_features_accounts.csv`:

- `community_id` — which cluster the account belongs to
- `community_size` — how many accounts are in that cluster
- `community_fraud_rate` — fraction of transactions in the cluster that are laundering

Finds **66,740 communities** across 1.76M accounts. Takes ~4 minutes.

### Step 5 — Build the feature store

```bash
python3 src/models/05_build_feature_store.py
```

Joins degree and community features onto every transaction row (once for sender, once for receiver). Produces:

```
data/processed/
    train_graph_enriched.parquet   (1.3GB)
    val_graph_enriched.parquet     (270MB)
    test_graph_enriched.parquet    (276MB)
```

Each row now has 23 features: 11 tabular + 12 graph (8 degree + 4 community).

### Step 6 — Train the baseline model

```bash
python3 src/models/06_train_xgboost_baseline.py
```

Trains XGBoost on tabular features only (no graph). This is the comparison point for the graph-enhanced model.

Results saved to `data/models/xgboost_baseline_metrics.json`.

### Step 7 — Tune decision threshold

```bash
python3 src/models/tune_threshold.py
```

Finds the probability threshold that maximises F1 on the validation set. The optimal threshold differs from the default 0.5 because of class imbalance.

### Step 8 — Train the graph-enhanced model

```bash
python3 src/models/07_train_graph_enhanced_model.py
```

Trains XGBoost on all 23 features (tabular + graph). Prints a comparison table vs. baseline and saves:

```
data/models/
    xgboost_graph_enhanced.json         (trained model)
    xgboost_graph_enhanced_metrics.json (test metrics)
    xgboost_graph_enhanced_cm.png       (confusion matrix)
    xgboost_graph_enhanced_pr_curve.png (precision-recall curve)
    model_comparison.json               (delta vs baseline)
```

---

## Scenario 2: ML Pipeline Only (Data Already Exists)

Use this if `graph_features_accounts.csv` has been shared with you (e.g. via Google Drive or a teammate), or if you just want to run the model without Neo4j.

**Case A — `graph_features_accounts.csv` has been shared with you (with community columns):**

Place it in `data/processed/` and run:

```bash
python3 notebooks/02_data_cleaning.py               # get the parquets
python3 src/models/05_build_feature_store.py        # build enriched parquets
python3 src/models/06_train_xgboost_baseline.py     # baseline
python3 src/models/tune_threshold.py                # threshold
python3 src/models/07_train_graph_enhanced_model.py # graph model
```

**Case B — `graph_features_accounts.csv` was shared but has only degree columns (no community columns):**

```bash
python3 notebooks/02_data_cleaning.py               # get the parquets
python3 src/models/04b_louvain_communities.py       # add community features (no Neo4j needed)
python3 src/models/05_build_feature_store.py        # build enriched parquets
python3 src/models/06_train_xgboost_baseline.py     # baseline
python3 src/models/tune_threshold.py                # threshold
python3 src/models/07_train_graph_enhanced_model.py # graph model
```

To check whether community columns are present: `python3 -c "import pandas as pd; print(pd.read_csv('data/processed/graph_features_accounts.csv').columns.tolist())"`

---

## Neo4j Desktop Setup

### Install and create a database

1. Download Neo4j Desktop from [neo4j.com/download](https://neo4j.com/download/)
2. Create a new Project → Add → Local DBMS
3. Set a password (put it in your `.env` as `NEO4J_PASSWORD`)
4. Start the DBMS

Default connection: `neo4j://127.0.0.1:7687`, user `neo4j`

### View nodes and edges in the browser

With the DBMS running, click **Open** → **Neo4j Browser** in Neo4j Desktop.

**See how many nodes/edges exist:**
```cypher
MATCH (n) RETURN labels(n), count(n)
```

**Sample 25 transactions:**
```cypher
MATCH (src:Account)-[:SENT]->(t:Transaction)-[:RECEIVED_BY]->(dst:Account)
RETURN src, t, dst LIMIT 25
```

**See FraudPattern nodes:**
```cypher
MATCH (f:FraudPattern) RETURN f
```

**See which transactions are linked to a pattern type:**
```cypher
MATCH (t:Transaction)-[r:PART_OF_PATTERN]->(f:FraudPattern {type: "CYCLE"})
RETURN t, r, f LIMIT 20
```

**Count transactions per pattern type:**
```cypher
MATCH ()-[r:PART_OF_PATTERN]->(f:FraudPattern)
RETURN f.type, count(DISTINCT r.attempt_id) AS attempts
ORDER BY attempts DESC
```

**See degree centrality of top accounts:**
```cypher
MATCH (a:Account)-[:SENT]->()
RETURN a.account_id, count(*) AS out_degree
ORDER BY out_degree DESC LIMIT 10
```

### Load the IBM fraud patterns into Neo4j

After the full graph is loaded:

```bash
python3 src/graph/load_fraud_patterns.py
```

This reads `HI-Medium_Patterns.txt` from the IBM dataset and creates `PART_OF_PATTERN` and `INVOLVED_IN_PATTERN` relationships linking transactions and accounts to their FraudPattern nodes (2,756 laundering attempts across 8 pattern types).

---

## Model Logic and Results

### Why XGBoost?

XGBoost handles tabular + graph features well, trains fast, and gives interpretable results via SHAP feature importances. It's the standard industry baseline for fraud detection before moving to graph neural networks.

### The class imbalance problem

Only 1 in 904 transactions in this dataset is fraud (0.11%). Standard models learn to ignore fraud entirely because predicting "legitimate" every time gives 99.89% accuracy.

We address this with `scale_pos_weight=904` — XGBoost treats each fraud sample as if it appeared 904 times, forcing the model to actually learn fraud patterns.

### Why graph features help

Tabular features (amount, payment type, time) describe a single transaction in isolation. Graph features add context:

- **Degree features**: how active is this account? High-volume senders/receivers are often either hubs in laundering networks or banks.
- **Community features**: is this account in a cluster where fraud is common? A $500 transfer looks different if the sender's community has a 40% fraud rate vs. 0.01%.

### Current results (Test Set)

| Model | PR-AUC | ROC-AUC | Recall |
|---|---|---|---|
| Tabular baseline (11 features) | 0.3043 | 0.9447 | 0.9142 |
| + Degree features (19 features) | 0.4590 | 0.9845 | 0.9840 |
| + Community features (23 features) | **0.4617** | **0.9846** | **0.9918** |

Adding graph structure produces a **52% improvement in PR-AUC** over the tabular baseline. The community features alone contributed +0.3% on top of degree features.

### Top features by SHAP importance

1. `is_ACH` — ACH payment format is highly associated with laundering
2. `src_total_degree` — high-volume senders are suspicious
3. `src_community_fraud_rate` — guilt by association (most powerful community feature)
4. `src_out_degree` — how many accounts this sender reaches
5. `log_amount` — transaction size
6. `src_in_degree` — how many accounts fund this sender
7. `dst_community_fraud_rate` — fraud history of the receiver's community

6 of the top 8 features are graph features — this is the core thesis of the project.

### How the model will be used

The trained model scores every incoming transaction with a fraud probability (0–1). Transactions above the optimal threshold are flagged for investigator review. The eventual system will:

1. Score transactions in near-real-time as they arrive
2. Surface the top flagged transactions in the Streamlit dashboard (Issues #7, #8)
3. Pass flagged transactions to an LLM investigator layer (Issues #13, #14) for human-readable explanations

---

## Future Work

### Issue #7 — Dashboard: Model Metrics Page

Add a page to the Streamlit dashboard (`dashboard_app.py`) that shows:
- Baseline vs. graph-enhanced PR-AUC comparison chart
- Confusion matrix
- SHAP feature importance bar chart

This makes model performance visible without needing to read JSON files.

### Issue #8 — Dashboard: Suspicious Accounts and Communities Page

Add a page that shows the highest-risk accounts and communities detected by the model. Pulls degree centrality and community_fraud_rate from `graph_features_accounts.csv` and displays a ranked list with links to Neo4j Browser for graph exploration.

### Issue #9 — Docs: Align README with 4-stage pipeline

Update the main README to reflect the completed pipeline:
- Stage 1: Data ingestion and cleaning
- Stage 2: Graph loading into Neo4j
- Stage 3: Feature extraction (degree + community)
- Stage 4: ML model training (baseline + graph-enhanced)

### Issue #10 — Infra: Move Neo4j credentials to .env everywhere

Currently the scripts have fallback hardcoded credentials if `.env` is missing. Remove all hardcoded fallbacks so credentials are only ever sourced from `.env`.

### Issue #11 — Testing: Smoke tests for pipeline scripts

Add lightweight tests that verify each pipeline script produces the expected output files with the right column count, without running on the full 31.9M row dataset. Use a small sample CSV.

### Issues #13 & #14 — Layer 4: LLM Investigator

This is the most novel part of the project and the direction suggested by the professor.

**What it will do:**
When the XGBoost model flags a transaction as suspicious, an LLM (Claude or GPT-4) will:
1. Receive the transaction details + graph context (degree, community fraud rate, connected accounts)
2. Look up which fraud pattern type (CYCLE, FAN-IN, etc.) the transaction is linked to in Neo4j
3. Generate a plain-English investigator report explaining *why* this transaction is suspicious
4. Compare multiple LLMs on report quality (Issue #13) and evaluate using beyond-accuracy metrics like faithfulness and specificity (Issue #14)

**Why this matters:**
XGBoost tells you a transaction has a 0.87 fraud probability. An LLM can tell you: *"This account sent $42,000 to 7 different recipients in 3 hours — a classic FAN-OUT layering pattern. The sender's community has a 34% historical fraud rate. Recommend freezing and flagging for SAR filing."*

That explanation is what a real AML investigator needs to act on the alert.

**How to build it (high level):**
1. Pick the top-N flagged transactions from model output
2. For each: query Neo4j for graph context (pattern type, connected accounts, community fraud rate)
3. Build a structured prompt with transaction + graph context
4. Call Claude/GPT-4 API to generate an investigation report
5. Evaluate reports with LLM-as-judge on faithfulness, specificity, actionability
6. Compare 2-4 LLMs on the same transactions (the Issue #13 comparison)
