# Graph-Based Fraud Intelligence Platform

**Detecting money laundering at scale using Knowledge Graphs, Graph Analytics, and LLM Investigation**

*DATA 298A · San José State University · Team 2 · Spring 2026*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-008CC1?style=flat&logo=neo4j&logoColor=white)](https://neo4j.com)
[![GDS](https://img.shields.io/badge/Graph_Data_Science-2.27.0-4CAF50?style=flat)](https://neo4j.com/docs/graph-data-science)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/aryaMehta26/Graph-Based-Fraud-Intelligence-Platform/blob/main/LICENSE)

---

## Why This Exists

Banks lose **$3.1 trillion** to money laundering every year. Traditional fraud detection systems evaluate every transaction in isolation — one row, one decision. They cannot see the coordinated patterns that sophisticated laundering operations depend on:

- **Layering rings** — A sends to B, B sends to C, C sends back to A. Clean.
- **Smurfing** — 50 accounts all draining into one mule account below reporting thresholds.
- **Fan-out distribution** — One source account rapidly distributing to dozens of recipients.

These patterns are invisible to tabular models. They are immediately obvious in a graph.

Inspired by how **Palantir Gotham** models the world as entities and relationships — not rows and columns — we built a **Knowledge Graph** on top of 31.9 million IBM AML transactions and layered four detection models on top of it.

---

## The Core Thesis

> **Fraud is relational, not independent.**

A transaction processed in isolation looks normal. The same transaction, viewed inside a network of 47 connected accounts moving $97,000 between 3 AM and 5 AM, looks like exactly what it is.

This platform proves that thesis in numbers:

| Model Layer | PR-AUC | vs. Baseline |
|---|---|---|
| Tabular XGBoost (baseline) | 0.3043 | — |
| + Degree centrality features | 0.4590 | +51% |
| + Louvain community features | 0.5599 | +84% |

---

## The Dataset

| Property | Detail |
|---|---|
| **Source** | [IBM Anti-Money Laundering Dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) on Kaggle |
| **Author** | Erik Altman, IBM Research |
| **Pattern Set** | `HI-Medium` — High Illicit ratio, Medium complexity laundering patterns |
| **Total Transactions** | **31,898,238** |
| **Total Accounts** | **1,757,942** |
| **Fraud Transactions** | **35,158** |
| **Fraud Rate** | **0.1102%** (904 legitimate transactions per 1 fraudulent) |
| **Payment Formats** | ACH, Cheque, Credit Card, Wire, Bitcoin, Reinvestment |
| **Date Range** | September 2022 (27 days) |

---

## 4-Layer Architecture

The platform implements four detection layers in sequence. Each layer feeds signal into the next.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     4-LAYER FRAUD DETECTION SYSTEM                   │
│                                                                       │
│  ┌──────────────────┐                                                 │
│  │  DATA PIPELINE   │  31.9M rows → cleaned → feature-engineered     │
│  │  notebooks/01-03 │  → time-aware 70/15/15 splits (parquet)        │
│  │                  │  → ontology-shattered → Neo4j bulk load        │
│  └────────┬─────────┘                                                 │
│           │                                                           │
│           ▼                                                           │
│  ┌──────────────────┐                                                 │
│  │   LAYER 1        │  XGBoost on tabular features                   │
│  │   XGBoost        │  scale_pos_weight=904 | PR-AUC: 0.3043         │
│  │   Baseline       │  11 features: log_amount, is_ACH, hour, etc.   │
│  └────────┬─────────┘                                                 │
│           │                                                           │
│           ▼                                                           │
│  ┌──────────────────┐                                                 │
│  │   LAYER 2        │  Neo4j GDS: Degree Centrality                  │
│  │   Graph          │  in_degree, out_degree, degree_centrality       │
│  │   Features       │  per account → joined to transaction splits    │
│  └────────┬─────────┘                                                 │
│           │                                                           │
│           ▼                                                           │
│  ┌──────────────────┐                                                 │
│  │   LAYER 3        │  Neo4j GDS: Louvain Community Detection        │
│  │   Community      │  community_id, community_size,                 │
│  │   Detection      │  community_fraud_rate → PR-AUC: 0.5599 (+84%) │
│  └────────┬─────────┘                                                 │
│           │                                                           │
│           ▼                                                           │
│  ┌──────────────────┐                                                 │
│  │   LAYER 4        │  Claude LLM Investigator                       │
│  │   LLM            │  Subgraph JSON → structured report             │
│  │   Investigator   │  pattern + evidence + risk + actions           │
│  └──────────────────┘                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Layer 1 — XGBoost Baseline (Trainable)

Trains on 11 tabular features extracted during data cleaning. Handles 904:1 class imbalance via `scale_pos_weight`. Primary metric is PR-AUC — accuracy is meaningless at 0.11% fraud rate.

**Key features by SHAP importance:** `is_ACH` > `log_amount` > `is_Wire` > `hour` > `is_cross_currency`

### Layer 2 — Graph Features (Non-Trainable)

Runs Neo4j GDS degree centrality on the full Account graph. Extracts `in_degree`, `out_degree`, `total_degree`, and `degree_centrality` per account, then joins them onto every transaction row. A hub account with 500 outgoing transactions looks completely different from a normal account with 3.

### Layer 3 — Louvain Community Detection (Non-Trainable)

Runs Louvain community detection on the Account graph. Assigns every account a `community_id` and computes `community_fraud_rate` — what fraction of that community's transactions are fraudulent. An account in a community with 42% fraud rate is not the same as one in a community with 0.1% fraud rate.

### Layer 4 — LLM Investigator (Trainable via Prompt)

Takes a pre-computed subgraph JSON (flagged by Layers 1–3) and produces a structured investigation report. Four Claude model variants compared for consistency, faithfulness, and schema compliance.

**LLM Evaluation Results:**

| Variant | Model | Schema | Faithfulness | Consistency | Meets Target |
|---|---|---|---|---|---|
| v1 | claude-haiku-4-5 | 1.00 | 0.89 | 0.95 | ✓ |
| v2 | claude-sonnet-4-6 | 1.00 | 1.00 | 0.89 | ✓ |
| v3 | claude-sonnet-4-6 + enhanced prompt | 1.00 | 1.00 | 0.87 | ✓ |
| v4 | claude-sonnet-4-6 + chain-of-thought | 1.00 | 1.00 | 0.84 | ✓ |

Consistency target: Jaccard ≥ 0.8 across 3 runs. **Recommended variant: v2** (composite score 0.967).

---

## Graph Ontology

Inspired by the entity-relationship modeling philosophy of **Palantir Gotham** — where the world is modeled as objects and connections, not tables and rows — we defined a strict ontology to represent the financial network:

```
                  [:SENT]                    [:RECEIVED_BY]
(Account) ─────────────────▶ (Transaction) ─────────────────▶ (Account)
    │                               │
    │ Properties:                   │ Properties:
    │  • account_id (unique)        │  • txn_id (unique)
    │  • bank_id                    │  • amount (raw USD)
    │  • degree_centrality          │  • log_amount (normalized)
    │  • louvain_community_id       │  • timestamp (datetime)
                                    │  • payment_format
                                    │  • is_laundering (ground truth)
                                    │  • hour, is_weekend
                                    │  • amount_bucket
```

**Ontology Shattering** — the flat CSV is split into exactly 4 files before loading:

| File | Role | Rows |
|---|---|---|
| `nodes_accounts.csv` | De-duplicated Account node definitions | 1,758,573 |
| `nodes_transactions.csv` | Transaction nodes with all feature properties | 29,336,378 |
| `edges_sent.csv` | Account → Transaction pointers | 29,336,378 |
| `edges_received.csv` | Transaction → Account pointers | 29,336,378 |

Each account exists exactly once. Loading a flat CSV directly creates duplicate nodes and corrupts cycle detection.

---

## EDA Key Findings

| Signal | Finding | Implication |
|---|---|---|
| **Class Imbalance** | 904:1 ratio (0.11% fraud) | Accuracy is useless — use PR-AUC |
| **Payment Format** | ACH = 87.3% of all fraud | `is_ACH` is the strongest single feature |
| **Amount Distribution** | Fraud transactions 6x larger on average | `log_amount` suppresses outlier noise |
| **Cross-Currency** | 0% fraud rate on cross-currency transactions | Strong negative predictor |
| **Temporal Patterns** | Fraud spikes at 3–5 AM | `hour` and `is_weekend` are key signals |

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── docker-compose.yml              # Neo4j with GDS + APOC plugins
├── .env.example                    # Environment variable template
├── .gitignore
│
├── notebooks/
│   ├── 01_data_extraction.py       # Kaggle API download + validation
│   ├── 02_data_cleaning.py         # Cleaning, feature engineering, splits
│   └── 03_eda_visualizations.py    # 16-chart EDA suite
│
├── src/
│   ├── graph/
│   │   ├── export_bulk_neo4j.py    # Shatter parquet → 4 ontology CSVs
│   │   ├── bulk_load_cypher.py     # 31M node micro-batch Cypher loader (~2 hrs)
│   │   ├── run_analytics.py        # GDS: degree centrality + Louvain
│   │   └── visualize_graph.py      # PyVis interactive graph explorer
│   │
│   ├── models/
│   │   ├── 04_extract_graph_features.py   # Pull degree + Louvain from Neo4j → CSV
│   │   ├── 05_build_feature_store.py      # Join graph features onto transaction splits
│   │   ├── 06_train_xgboost_baseline.py   # Layer 1: tabular XGBoost
│   │   ├── 07_train_graph_enhanced_model.py # Layer 1+2+3: XGBoost + graph features
│   │   └── tune_threshold.py              # PR curve threshold optimisation
│   │
│   └── llm/
│       ├── investigator.py         # Layer 4: 4-variant Claude LLM investigator
│       └── evaluate.py             # Consistency + faithfulness + schema evaluation
│
├── artifacts/
│   ├── sample_subgraph.json        # Smurfing test case
│   ├── layering_ring_subgraph.json # Layering Ring test case
│   ├── fan_out_subgraph.json       # Fan-Out test case
│   ├── llm_outputs/                # Saved LLM responses per variant
│   └── metrics/                    # llm_eval.json + comparison table
│
├── data/models/                    # Gitignored — trained model artifacts
│   ├── xgboost_baseline.json
│   ├── xgboost_baseline_metrics.json
│   ├── xgboost_graph_enhanced.json
│   └── model_comparison.json
│
├── dashboard_app.py                # Streamlit intelligence dashboard
└── data/                           # Gitignored — 5.4GB dataset lives locally
    ├── processed/                  # Parquet splits
    └── neo4j_bulk_import/          # 4 ontology CSVs
```

---

## How to Run

### Step 1 — Environment setup

```bash
cp .env.example .env
# Edit .env and set NEO4J_PASSWORD and ANTHROPIC_API_KEY
pip install -r requirements.txt
```

### Step 2 — Start Neo4j

```bash
docker-compose up -d
# Verify at http://localhost:7474 — login: neo4j / fraud2026
```

### Step 3 — Data pipeline

```bash
python3 notebooks/01_data_extraction.py    # Download 7.6GB from Kaggle
python3 notebooks/02_data_cleaning.py      # Clean + engineer features + split
python3 notebooks/03_eda_visualizations.py # Generate 16 EDA charts
```

### Step 4 — Load Knowledge Graph (~2 hours)

```bash
python3 src/graph/export_bulk_neo4j.py     # Shatter parquet → 4 CSVs
python3 src/graph/bulk_load_cypher.py      # Load 31M records into Neo4j
```

### Step 5 — Train models

```bash
# Layer 1: XGBoost baseline
python3 src/models/06_train_xgboost_baseline.py

# Layers 2+3: Extract graph features and train enhanced model
python3 src/models/04_extract_graph_features.py
python3 src/models/05_build_feature_store.py
python3 src/models/07_train_graph_enhanced_model.py
```

### Step 6 — Run LLM Investigator

```bash
# Run all 4 variants against a subgraph
python3 src/llm/investigator.py --variant all --input artifacts/sample_subgraph.json --runs 3

# Evaluate
python3 src/llm/evaluate.py --subgraph sample_subgraph
```

### Step 7 — Launch Dashboard

```bash
python3 -m streamlit run dashboard_app.py
# Open http://localhost:8501
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data Sourcing | Kaggle Hub API | Authenticated dataset download |
| Data Lake | Apache Parquet (PyArrow) | Compressed columnar storage |
| Processing | Pandas 2.0, NumPy | Vectorized transformations |
| Graph Database | Neo4j 5.x | Knowledge graph |
| Graph ML | Neo4j GDS 2.27.0 | Louvain + Degree Centrality |
| ML Layer | XGBoost 2.0 + SHAP | Fraud classification |
| LLM Layer | Anthropic Claude (claude-sonnet-4-6) | Structured investigation reports |
| Visualization | Matplotlib, PyVis | EDA + graph explorer |
| Dashboard | Streamlit 1.30 | Intelligence interface |
| Container | Docker + Compose | Reproducible Neo4j environment |

---

## Current Progress

### Phase 1 — Data Engineering ✅

- [x] Raw dataset extraction from Kaggle (31.9M rows, 5.45GB)
- [x] Schema cleaning, self-loop removal, null validation
- [x] Feature engineering (13 columns: log_amount, is_ACH, hour, dow, is_cross_currency, etc.)
- [x] Time-aware chronological splits (70/15/15) — zero temporal leakage
- [x] Ontology mapping — flat CSV → 4-component graph structure
- [x] 31M node bulk load into Neo4j via micro-batching (~2 hours)
- [x] 16-chart EDA suite

### Phase 2 — Graph Analytics ✅

- [x] Degree centrality — identify high-volume hub accounts
- [x] Louvain community detection — map fraud rings
- [x] Community fraud rate computation
- [x] Graph features joined onto transaction splits (feature store)

### Phase 3 — Model Development ✅

- [x] XGBoost baseline — PR-AUC 0.3043 on tabular features
- [x] Graph-enhanced XGBoost — PR-AUC 0.4590 with degree features (+51%)
- [x] Community-enhanced XGBoost — PR-AUC 0.5599 with Louvain features (+84%)
- [x] SHAP feature importance analysis
- [x] Threshold tuning via PR curve

### Phase 4 — LLM Investigator ✅

- [x] 4-variant Claude investigator (Haiku, Sonnet, Sonnet+prompt, Sonnet+CoT)
- [x] Structured output schema: pattern, evidence, risk_level, actions
- [x] Evaluation framework: consistency (Jaccard), faithfulness, schema compliance
- [x] Tested on 3 subgraph types: Smurfing, Layering Ring, Fan-Out
- [x] All 4 variants meet consistency target (≥ 0.8 Jaccard)

---

## Team

**Team 2 — DATA 298A Applied Data Science Capstone**
**San José State University · Spring 2026**

| Name | Contributions |
|---|---|
| Arya Mehta | Data engineering, graph architecture, pipeline orchestration, graph-enhanced model |
| Aishwarya | XGBoost baseline training, LLM investigator layer, evaluation framework |
| Prajwal Dambalkar | Graph features, Louvain community detection, model comparison |
| Keith Gonsalves | Dashboard, model metrics visualisation |
| Om Dankhara | Supporting infrastructure |

---

## Academic Context

This project satisfies the following DATA 298A capstone sections:

- **3.1–3.6** — Data pipeline: collection, cleaning, transformation, splits, EDA
- **4.1** — XGBoost baseline model with PR-AUC evaluation
- **4.2** — Graph feature extraction (degree centrality, Louvain)
- **4.3** — Graph-enhanced model — baseline vs. graph comparison
- **4.4** — LLM investigator layer with structured output and multi-variant evaluation
- **4.5** — Pipeline demo via Streamlit dashboard

---

*Built with Python, Neo4j, and a lot of coffee.*
