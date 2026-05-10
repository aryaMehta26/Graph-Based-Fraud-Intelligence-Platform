<div align="center">

# Graph-Based Fraud Intelligence Platform

**Detecting money laundering at scale using Knowledge Graphs and Deep Graph Learning**

*DATA 298A · San José State University · *

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-008CC1?style=flat&logo=neo4j&logoColor=white)](https://neo4j.com)
[![GDS](https://img.shields.io/badge/Graph_Data_Science-2.27.0-4CAF50?style=flat)](https://neo4j.com/docs/graph-data-science)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![License: Apache_2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>

---

## Why This Exists

Banks lose **$3.1 trillion** to money laundering every year. Traditional fraud detection systems look at every transaction in isolation — one row, one decision. They cannot see the patterns that sophisticated laundering operations depend on:

- **Layering rings** — A sends to B, B sends to C, C sends back to A. Clean.
- **Smurfing (account funneling)** — 50 accounts all draining into one mule account.
- **Cyclic chains** — Money travels through 8 accounts before surfacing clean.

These patterns are invisible to tabular models. They are immediately obvious in a graph.

Inspired by how **Palantir Gotham** and enterprise financial intelligence platforms model the world as entities and relationships — not rows and columns — we built a **Knowledge Graph** on top of 31.9 million IBM AML transactions. Every account is a node. Every transaction is an edge. Every pattern is a shape we can detect.

---

## The Dataset

| Property | Detail |
|---|---|
| **Source** | [IBM Anti-Money Laundering Dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) on Kaggle |
| **Author** | Erik Altman, IBM Research |
| **Pattern Set Used** | `HI-Medium` — High Illicit ratio, Medium complexity laundering patterns |
| **Available Pattern Sets** | LI-Small, LI-Medium, HI-Small, HI-Medium, HI-Large |
| **Total Transactions** | **31,898,238** |
| **Total Accounts** | **1,757,942** |
| **Fraud Transactions** | **35,158** |
| **Fraud Rings (Patterns)** | **2,756 identified laundering patterns** |
| **Fraud Rate** | **0.1102%** (906 legitimate transactions for every 1 fraudulent one) |
| **Payment Formats** | ACH, Cheque, Credit Card, Wire, Bitcoin, Reinvestment |
| **Currencies** | 8 currencies across 27 days of transaction history |
| **Raw File Size** | **5.45 GB** across 3 files |
| **Date Range** | September 2022 (27 days) |

### Raw File Inventory

| File | Size | Contents |
|---|---|---|
| `HI-Medium_Trans.csv` | 4.0 GB | All 31.9M transaction records |
| `HI-Medium_accounts.csv` | 63.5 MB | 1.76M unique account + bank mappings |
| `HI-Medium_Patterns.txt` | 0.2 MB | 2,756 ground-truth laundering ring definitions |

### Why This Dataset?

Most public fraud datasets have tens of thousands of rows. This one has **31.9 million**, with real synthetic laundering ring structures defined by IBM researchers. It mirrors real enterprise-scale financial data, which is exactly what makes the engineering problem hard and the solution valuable.

---

## Architecture Overview

The system is split into two parallel pipelines that share the same cleaned, engineered data:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        5-STAGE DATA PIPELINE                                  │
│                                                                                │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────────┐                    │
│  │  1. COLLECT  │──▶│  2. CLEAN   │──▶│  3. ENGINEER      │                    │
│  │             │   │             │   │                  │                    │
│  │ Kaggle API  │   │ Timestamps  │   │ log_amount       │                    │
│  │ 31.9M rows  │   │ Self-loops  │   │ is_ACH           │                    │
│  │ 5.45 GB     │   │ Null checks │   │ hour / weekend   │                    │
│  │             │   │ Log-norm    │   │ amount_bucket    │                    │
│  └─────────────┘   └─────────────┘   └────────┬─────────┘                    │
│                                                │                              │
│                                   ┌────────────┴────────────┐                │
│                                   │  CENTRAL PIPELINE FORK  │                │
│                                   └────────────┬────────────┘                │
│                                                │                              │
│                        ┌───────────────────────┴───────────────────────┐     │
│                        ▼                                               ▼     │
│              PATH A: TABULAR ML                           PATH B: GRAPH ML   │
│              ─────────────────                            ───────────────     │
│         4a. Time-Aware Split                         4b. Ontology Mapping    │
│              70% Train                                4 CSV components        │
│              15% Val                                  Nodes + Edges           │
│              15% Test                                                          │
│                   │                                          │                │
│                   ▼                                          ▼                │
│           XGBoost Parquets                        Neo4j 31M Bulk Load        │
│           (Fraud Classifier)                      (~2 hours, batched)         │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Graph Ontology

This is the core intellectual contribution of the data engineering phase. Inspired by the entity-relationship modeling philosophy of **Palantir Gotham** — where the world is modeled as objects and connections, not tables and rows — we defined a strict **3-component ontology** to represent the financial network:

```
                    [:SENT]                    [:RECEIVED_BY]
  (Account) ─────────────────▶ (Transaction) ─────────────────▶ (Account)
      │                               │
      │ Properties:                   │ Properties:
      │  • account_id (unique)        │  • txn_id (unique)
      │  • bank_id                    │  • amount (raw USD)
                                      │  • log_amount (normalized)
                                      │  • timestamp (datetime)
                                      │  • payment_format (ACH/Wire/etc)
                                      │  • currency
                                      │  • is_ACH (1-hot encoded)
                                      │  • is_laundering (ground truth label)
                                      │  • hour (extracted from timestamp)
                                      │  • is_weekend (temporal signal)
                                      │  • amount_bucket (categorical range)
```

### Why Not Just Load the CSV Into Neo4j?

This is the most common mistake people make with Graph Databases. If you load a flat table directly, Neo4j creates a brand new Account node for every row where that account appears. Account `10A2B3C` appears 50 times in the CSV if it made 50 transfers — so you get 50 duplicate nodes. The graph becomes corrupt and cycle detection fails entirely.

**The Solution: Ontology Shattering.** We took the flat Parquet lake and split it into exactly 4 CSV files, each serving one specific purpose in the graph structure:

| File | Role | Rows |
|---|---|---|
| `nodes_accounts.csv` | De-duplicated Account node definitions | 1,758,573 |
| `nodes_transactions.csv` | Transaction node definitions with all feature properties | 29,336,378 |
| `edges_sent.csv` | Geometric pointer: Account → Transaction | 29,336,378 |
| `edges_received.csv` | Geometric pointer: Transaction → Account | 29,336,378 |

Each account exists **exactly once**. All 11 engineered features live on the Transaction nodes. The edges contain zero data — they are pure structural connections.

---

## EDA Key Findings

Running `03_eda_visualizations.py` on the full dataset revealed 5 critical signals:

| Signal | Finding | Implication |
|---|---|---|
| **Class Imbalance** | 906:1 ratio (0.11% fraud) | Accuracy is useless. Use PR-AUC + F1-Macro |
| **Payment Format** | ACH = **87.3% of all fraud** | `is_ACH` is the single strongest tabular predictor |
| **Amount Distribution** | Fraud transactions are **6x larger** on average | `log_amount` required to suppress outlier noise |
| **Cross-Currency** | 0% fraud rate on cross-currency transactions | Powerful negative predictor for XGBoost |
| **Temporal Patterns** | Fraud spikes at 3–5 AM | `hour` and `is_weekend` are key temporal signals |

---

## The 31M Engineering Challenge

> "Just load the CSV into Neo4j." — Everyone who has never tried it with 31 million rows.

Naively loading 31.9M rows causes an **Out-Of-Memory (OOM) crash**. Mapping 58 million edges simultaneously exceeds Java heap limits. The server freezes. Data is lost.

**The Engineering Solution — Cypher Micro-Batching:**

We engineered a Cypher query pattern that forces Neo4j to **commit and flush memory every 10,000 records**. This keeps RAM usage bounded regardless of total dataset size:

```cypher
LOAD CSV WITH HEADERS FROM 'file:///nodes_transactions.csv' AS row
CALL {
    WITH row
    CREATE (t:Transaction {
        txn_id:        row.`txn_id:ID(Transaction)`,
        amount:        toFloat(row.`amount:float`),
        timestamp:     datetime(row.`timestamp:datetime`),
        is_laundering: toInteger(row.`is_laundering:int`)
    })
} IN TRANSACTIONS OF 10000 ROWS;
```

**Total ingest time: ~2 hours.** Final graph state:

| Component | Count |
|---|---|
| Account Nodes | 1,758,573 |
| Transaction Nodes | 29,336,378 |
| SENT Edges | 29,336,378 |
| RECEIVED_BY Edges | 29,336,378 |
| Fraud Transactions | 35,158 |
| **Total Graph Objects** | **~89 Million** |

---

## Pipeline Data Decay Table

Tracking exactly how the record count transforms at each stage:

| Stage | Records | Retention |
|---|---|---|
| Raw Kaggle Extract | 31,898,238 | 100% |
| After Cleaning (self-loops removed for graph) | 29,336,378 | 91.97% |
| Account Nodes (de-duplicated) | 1,758,573 | — |
| Transaction Nodes | 29,336,378 | — |
| SENT Edges | 29,336,378 | — |
| RECEIVED_BY Edges | 29,336,378 | — |
| Total Objects in Neo4j | ~89,000,000 | — |

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── docker-compose.yml              # Neo4j with GDS + APOC plugins
├── .gitignore                      # Excludes 5.4GB dataset, logs, .docx files
│
├── notebooks/
│   ├── 01_data_extraction.py       # Kaggle API download + file validation
│   ├── 02_data_cleaning.py         # Schema cleaning, self-loop removal, feature engineering, splits
│   └── 03_eda_visualizations.py    # 16-chart EDA suite (class imbalance, distributions, temporal)
│
├── src/graph/
│   ├── export_bulk_neo4j.py        # Shatter parquet into 4 ontology CSVs
│   ├── bulk_load_cypher.py         # 31M node micro-batch Cypher ingestor (~2 hrs)
│   ├── load_neo4j.py               # Live 5K-row demo loader for presentations
│   ├── run_analytics.py            # GDS: Louvain community detection + Degree centrality
│   └── visualize_graph.py          # PyVis interactive graph visualization
│
├── dashboard_app.py                # 7-phase Streamlit intelligence dashboard
├── demo_TA_pipeline.py             # Full pipeline orchestrator with live streaming logs
├── run_all_backend_eda.py          # Master runner: executes 01 → 02 → 03 in sequence
│
└── data/                           # Gitignored — 5.4GB lives on local machine only
    ├── processed/                  # Parquet outputs (transactions_clean, splits)
    └── neo4j_bulk_import/          # 4 ontology CSVs for Neo4j import
```

---

## How to Run

### Step 1 — Start Neo4j via Docker
```bash
docker-compose up -d
```
This launches Neo4j at `http://localhost:7474` with the **Graph Data Science 2.27.0** and **APOC** plugins pre-configured.

Verify GDS is active in the Neo4j Browser:
```cypher
RETURN gds.version()
-- Expected output: "2.27.0"
```

### Step 2 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the Full Data Pipeline
```bash
# Download and validate raw dataset from Kaggle
python3 notebooks/01_data_extraction.py

# Clean, engineer features, generate Parquet files
python3 notebooks/02_data_cleaning.py

# Generate all 16 EDA charts
python3 notebooks/03_eda_visualizations.py

# Or run all 3 in sequence automatically:
python3 run_all_backend_eda.py
```

### Step 4 — Prepare and Load the Knowledge Graph
```bash
# Shatter parquet into 4 ontology CSVs
python3 src/graph/export_bulk_neo4j.py

# Bulk load 31M records into Neo4j (takes ~2 hours)
python3 src/graph/bulk_load_cypher.py
```

### Step 5 — Launch the Intelligence Dashboard
```bash
python3 -m streamlit run dashboard_app.py
```
Open `http://localhost:8501` — the 7-phase dashboard walks through the complete pipeline.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data Sourcing | Kaggle Hub API | Authenticated dataset download |
| Data Lake | Apache Parquet (PyArrow) | Compressed, fast columnar storage |
| Processing | Pandas 2.0, NumPy | Vectorized data transformations |
| Graph Database | Neo4j 5.x | Entity-relationship knowledge graph |
| Graph ML Engine | Neo4j GDS 2.27.0 | Louvain, Centrality, FastRP in-database |
| Visualization | Matplotlib, Seaborn, PyVis | EDA charts + interactive graph explorer |
| Dashboard | Streamlit 1.30 | Live presentation interface |
| Container | Docker + Compose | Reproducible Neo4j environment |
| ML (Next Phase) | XGBoost, GraphSAGE | Fraud classification models |

---

## Current Progress & Roadmap

### Phase 1 — Data Engineering (COMPLETE)
- [x] Raw dataset extraction from Kaggle (31.9M rows, 5.45GB)
- [x] Schema cleaning, self-loop removal, null validation
- [x] Feature engineering (11 new columns: log_amount, is_ACH, hour, etc.)
- [x] Time-aware chronological splits (70/15/15) for tabular ML
- [x] Ontology mapping — flat CSV → 4-component graph structure
- [x] 31M node bulk load into Neo4j via micro-batching (~2 hours)
- [x] EDA suite — 16 charts covering class imbalance, distributions, temporal patterns
- [x] Streamlit intelligence dashboard

### Phase 2 — Graph Analytics (IN PROGRESS)
- [x] Degree Centrality — identify high-volume hub accounts
- [x] Louvain Community Detection — map potential fraud rings
- [ ] Community fraud scoring — rank communities by fraud concentration
- [ ] Graph visualization — interactive PyVis network explorer

### Phase 3 — Model Development (PLANNED)
- [ ] **XGBoost Classifier** — tabular fraud detection on the 70/15/15 splits, with `scale_pos_weight=906` to handle class imbalance
- [ ] **GraphSAGE** — graph neural network trained on node embeddings from the Neo4j knowledge graph
- [ ] **FastRP Embeddings** — in-database graph embeddings via GDS, fed into downstream classifiers
- [ ] **Ensemble Model** — combine tabular XGBoost (real-time speed) with GraphSAGE (structural depth) for production-grade fraud scoring

### Phase 4 — Production Intelligence Platform (PLANNED)
- [ ] Real-time transaction scoring API (FastAPI)
- [ ] Streaming pipeline (Kafka / Flink) for live transaction monitoring
- [ ] Alert dashboard with fraud probability scores and ring visualizations
- [ ] Model retraining pipeline with drift detection

---

## Team

**Team 12 — DATA 298A Applied Data Science Capstone**
**San José State University · Spring 2026**

| Name | Role |
|---|---|
| Arya Mehta | Data Engineering, Graph Architecture, Pipeline Orchestration |

---

## Academic Context

This repository covers the data engineering and knowledge graph foundation phase of a multi-phase research initiative. The work specifically satisfies:

- **3.1** Data Process — pipeline flowchart, exact steps from intake to splits
- **3.2** Data Collection — dataset sourcing, parameters, quantities, raw samples
- **3.3** Pre-Processing — schema validation, self-loop handling, null checks, log normalization
- **3.4** Transformation — CSV → Parquet conversion, 11-feature engineering pass, ontology shattering
- **3.5** Data Preparation — time-aware 70/15/15 chronological splits with zero temporal leakage
- **3.6** Data Statistics — 16-chart EDA suite, class imbalance analysis, pipeline decay table
- **4.5** Pipeline Demo — live Streamlit dashboard with real-time log streaming to Neo4j

---

*Built with Python, Neo4j, and a lot of coffee.*
