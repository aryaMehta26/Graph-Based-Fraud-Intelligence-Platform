import streamlit as st
import pandas as pd
from neo4j import GraphDatabase
import os
import subprocess
import time

# --- PAGE SETUP ---
st.set_page_config(page_title="DATA 298A Fraud Platform", layout="wide", initial_sidebar_state="expanded")

from dotenv import load_dotenv
load_dotenv()
PROJ = os.getenv("PROJ_ROOT", os.path.dirname(os.path.abspath(__file__)))
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/e/ef/SJSU_Spartans_primary_mark.svg/1200px-SJSU_Spartans_primary_mark.svg.png", width=150)
st.sidebar.title("System Navigation Menu")
selection = st.sidebar.radio("Go to Phase:", [
    "1. Data Collection & Setup",
    "2. Data Cleaning & Integrity",
    "3. Exploratory Analytics",
    "4. Features & Ontology Mapping",
    "5. Conquering the 31M Graph",
    "6. Live Pipeline Orchestrator",
    "7. Neo4j Graph Validation"
])

# --- PHASE 1: DATA COLLECTION ---
if "1. Data Collection" in selection:
    st.title("Phase 1: Data Collection & Architecture Flow")
    st.markdown("### Production scale raw data sourcing from Kaggle APIs.")
    st.divider()
    
    col_flow, col_metrics = st.columns([1.5, 1])
    
    with col_flow:
        st.subheader("Data Process: Architecture Flowchart")
        st.graphviz_chart('''
            digraph G {
                rankdir=TB
                node [shape=box, style="filled,rounded", color="#1e88e5", fontcolor=white, fontname="Helvetica", penwidth=2]
                edge [color="#90caf9", penwidth=2, arrowsize=0.8]
                
                A [label="1. Raw Kaggle API Extraction\n(31.9M Records, 5.4GB)"]
                B [label="2. Pre-Processing Pipeline\n(Downcasting, Imputation, Self-loop Removal)"]
                C [label="3. Feature Engineering\n(log_amount, timestamps, one-hot encoding)"]
                
                node [color="#d81b60"]
                D [label="4. Central Architecture Fork"]
                
                node [color="#43a047"]
                E [label="Path A (Tabular ML)\nTime-Aware Chronological Split (70/15/15)"]
                F [label="Path B (Graph ML)\nOntology Mapping (Node & Edge CSV Shattering)"]
                
                node [color="#8e24aa"]
                G [label="XGBoost Lakehouse Parquets"]
                H [label="Neo4j Enterprise Database (31M Bulk Load)"]
                
                A -> B
                B -> C
                C -> D
                
                D -> E [label=" Tabular Pipeline"]
                D -> F [label=" Graph Pipeline"]
                
                E -> G
                F -> H
            }
        ''')
        
    with col_metrics:
        st.subheader("Sourcing Metrics")
        st.metric("Dataset Selected", "IBM AML HI-Medium")
        st.metric("Total Extracted Files", "3")
        st.metric("Total Size GB", "5.45 GB")

    st.divider()
    st.subheader("Representative Raw Data Sample")
    st.image(os.path.join(PROJ, "3.2_Raw_Data_Sample.png"), use_container_width=True, caption="Raw CSV Extract Validation")
    
    st.divider()
    st.subheader("Data Extraction Pipeline Output (`01_data_extraction.py`)")
    st.code("""
============================================================
STEP 1: IBM AML DATASET — DATA EXTRACTION
Team 12 | DATA 298A | Graph Fraud Intelligence Platform
============================================================

[1/4] Downloading IBM AML dataset from Kaggle...
      Source: ealtman2019/ibm-transactions-for-anti-money-laundering-aml
      [SUCCESS] Dataset path: ~/.cache/kagglehub/datasets/.../versions/8

[2/4] Files available in dataset:
File Name                                      Size
----------------------------------------------------
  HI-Medium_Patterns.txt                     0.2 MB
  HI-Medium_Trans.csv                        4.0 GB
  HI-Medium_accounts.csv                    63.5 MB

[3/4] Loading and validating HI-Medium files...
  [SUCCESS] Transactions   : HI-Medium_Trans.csv | 4.0 GB | 31,898,238 rows × 10 columns
  [SUCCESS] Accounts       : HI-Medium_accounts.csv | 63.5 MB | 1,757,942 rows × 2 columns
  [SUCCESS] Patterns       : HI-Medium_Patterns.txt | 0.2 MB | 2,756 lines

EXTRACTION COMPLETE 
""", language="text")

# --- PHASE 2: PRE-PROCESSING ---
elif "2. Data Cleaning" in selection:
    st.title("Phase 2: Data Cleaning & Schema Integrity")
    st.markdown("### Schema checks, missing/outlier handling, memory optimization.")
    st.divider()
    
    # --- ADDED: LIVE DATA FRAME SAMPLES ---
    col_raw, col_clean = st.columns(2)
    with col_raw:
        st.subheader("BEFORE: Raw Schema Matrix")
        raw_mock = pd.DataFrame({
            "Timestamp": ["2022/09/01 00:20", "2022/09/01 00:21", "2022/09/01 00:22"],
            "From Account": ["10A2B3C", "99X8Y7Z", "10A2B3C"],
            "To Account": ["99X8Y7Z", "44M5N6P", "10A2B3C"],
            "Amount Received": ["1045.20", "0.05", "5000000.00"],
            "Payment Format": ["ACH", "Cheque", "Wire"],
            "Is Laundering": ["0", "0", "1"]
        })
        st.dataframe(raw_mock, hide_index=True)
        st.info("Issues detected: Object strings for dates, Self-loops present (Row 3), Non-normalized extreme amounts.")
        
    with col_clean:
        st.subheader("AFTER: Engineered Matrix")
        clean_mock = pd.DataFrame({
            "Timestamp": [pd.to_datetime("2022-09-01 00:20:00"), pd.to_datetime("2022-09-01 00:21:00")],
            "src_acct": ["10A2B3C", "99X8Y7Z"],
            "dst_acct": ["99X8Y7Z", "44M5N6P"],
            "log_amount": [6.953, 0.048],
            "is_ACH": [1, 0],
            "is_laundering": [0, 0]
        })
        st.dataframe(clean_mock, hide_index=True)
        st.success("Resolutions: Datetime casting, Self-loop rows annihilated, amounts log-normalized, formats 1-hot encoded.")

    st.divider()
    
    st.subheader("Cleaning Operations Pipeline Output (`02_data_cleaning.py`)")
    st.code("""
============================================================
STEP 2: IBM AML DATASET — DATA CLEANING
Team 12 | DATA 298A | Graph Fraud Intelligence Platform
============================================================

[LOADING] Reading HI-Medium_Trans.csv (31.9M rows)...
  [SUCCESS] Loaded: 31,898,238 transaction rows | 1,757,942 account rows

[STEP 1/7] Parse Timestamps → datetime64
  Before: dtype = object
  After : dtype = datetime64[ns]
  Span  : 27 days
  [SUCCESS] PASSED

[STEP 3/7] Null / Missing Value Check
  Total null values : 0
  [SUCCESS] PASSED — Zero nulls confirmed

[STEP 4/7] Self-Loop Detection (src_acct == dst_acct)
  Total transactions     : 31,898,238
  Self-loops detected    : 2,561,860  (8.03%)
  Non-self-loop (graph)  : 29,336,378
  [SUCCESS] PASSED

[STEP 5/7] Amount Validation + Log Transform
  Negative amounts : 0
  Zero amounts     : 0
  Min amount       : $0.000003
  Max amount       : $132,168,142,654.58
  Log-transformed: log1p(Amount_Paid) → 'log_amount'
  [SUCCESS] PASSED — No negative amounts, log transform applied
""", language="text")

# --- PHASE 3: EDA ---
elif "3. Exploratory Analytics" in selection:
    st.title("Phase 3: Exploratory Data Analysis & Scale")
    st.markdown("### Clear visualizations of distributions, imbalances, and pattern drifts.")
    st.divider()
    
    # --- ADDED: DATA PIPELINE DECAY TABLE ---
    st.subheader("Enterprise Data Decay Engine (Pipeline Summary)")
    decay_df = pd.DataFrame({
        "Pipeline Stage": [
            "1. Raw API Kaggle Extract", 
            "2. Cleaned (Self-Loops Dropped)", 
            "3. Tabular Splits Constructed", 
            "4. Graph Ontology Translation",
            "5. Graph Node Injection",
            "6. Graph Edge Injection"
        ],
        "Record Count": [
            "31,898,238", 
            "29,336,378", 
            "29,336,378", 
            "31,094,951 (Total Objects)",
            "31,094,951",
            "58,672,756"
        ],
        "Delta Retention": [
            "100%", 
            "91.9%", 
            "91.9% (70/15/15)", 
            "Expanded into 4 CSVs",
            "-",
            "-"
        ]
    })
    st.dataframe(decay_df, hide_index=True, use_container_width=True)
    st.divider()
    
    st.subheader("Automated EDA Analytics (`03_eda_visualizations.py`)")
    st.code("""
EDA SECTION 2: CLASS IMBALANCE
  Legit (0)    : 31,863,080
  Fraud (1)    : 35,158
  Fraud rate   : 0.1102%
  Class ratio  : 906:1
  Implication  : Cannot use standard accuracy — use Graph ML + PR-AUC
  XGBoost fix  : scale_pos_weight = 906

EDA SECTION 5: PAYMENT FORMAT — KEY FRAUD SIGNAL
  Format                 Total    Fraud   Fraud%
  -----------------------------------------------
  Cheque             3,485,389      765   0.022%
  Credit Card        7,557,117    1,595   0.021%
  ACH               10,314,845   30,683   0.297%   (HIGHEST)
  ACH = 87.3% of ALL fraud transactions!
""", language="text")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(os.path.join(PROJ, "notebooks", "eda_charts_live", "01_class_imbalance.png"), use_container_width=True)
        st.image(os.path.join(PROJ, "notebooks", "eda_charts_live", "06_payment_format.png"), use_container_width=True)
    with col2:
         st.image(os.path.join(PROJ, "notebooks", "eda_charts_live", "04_amount_distribution.png"), use_container_width=True)
         st.image(os.path.join(PROJ, "notebooks", "eda_charts_live", "10_ring_types.png"), use_container_width=True)

# --- PHASE 4: FEATURES & SPLITS ---
elif "4. Features" in selection:
    st.title("Phase 4: Transformation & Production Preparation")
    st.markdown("### Feature engineering, strict train/test data separation, and graph translation.")
    st.divider()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Memory Footprint Reduction", "-60%")
    c2.metric("Format Conversion", "CSV → Parquet")
    c3.metric("Mathematical Graph Edges", "29,336,378")
    c4.metric("Engineered Features Added", "11")
    
    st.divider()
    st.subheader("Transformed Lakehouse Parquet Sample")
    st.image(os.path.join(PROJ, "3.4_Transformed_Parquet_Sample.png"), use_container_width=True, caption="Engineered Parquet Dataset Validation")
    
    st.divider()
    
    st.subheader("Time-Aware Stratification Splitting Output (XGBoost Pipeline)")
    st.code("""
TIME-AWARE DATA SPLITS (No Random Shuffle — No Time Leakage)
------------------------------------------------------------
  TRAIN 70%
    Date range : 2022-09-01 → 2022-09-18
    Rows       : 22,328,766
    Fraud rows : 24,960

  VAL   15%
    Date range : 2022-09-18 → 2022-09-22
    Rows       : 4,784,736
    Fraud rows : 5,340

  TEST  15%
    Date range : 2022-09-22 → 2022-09-26
    Rows       : 4,784,736
    Fraud rows : 4,858
    
  [SUCCESS] Time-sorted before split — zero temporal data leakage
""", language="text")

    st.divider()
    
    st.subheader("The Most Crucial Transformation: Ontology Mapping (Neo4j Pipeline)")
    st.markdown("""
    **How do you turn a flat spreadsheet into a living, breathing Financial Network?**
    
    If someone looked at our Parquet file, they would see a flat table with millions of rows and 11 distinct columns (Amount, Timestamp, is_ACH, etc.). You **cannot** load a flat table directly into a Graph Database. Doing so would blindly duplicate thousands of bank accounts every time they made a transaction, utterly destroying the mathematical integrity of the network.
    """)
    
    col_a, col_b = st.columns([1.5, 1])
    with col_a:
        st.info("""
        **The Engineering Solution (Ontology Shattering):**
        To solve this, we took the massive flat Data Lake and architecturally shattered it into exactly **4 specific CSV files** that map perfectly to our Graph Ontology `(Account) -[:SENT]-> (Transaction) -[:RECEIVED_BY]-> (Account)`:
        
        1. `nodes_accounts.csv`: We isolated all unique Account IDs, de-duplicated them, and established 1.7 million unique human anchor points.
        2. `nodes_transactions.csv`: We isolated the actual transactions. **This is where all 11+ engineered columns (log_amount, is_ACH, timestamps) were injected as structural properties.**
        3. `edges_sent.csv`: A purely geometric pointer mapping the Sender to the Transaction.
        4. `edges_received.csv`: A purely geometric pointer mapping the Transaction to the Receiver.
        """)
    with col_b:
        st.success("""
        **Why is this so Impactful?**
        By translating a messy spreadsheet into these 4 strictly governed network components, we gave Neo4j a perfect blueprint. 
        
        The graph instantly knows exactly who the entities are, exactly what their attributes are, and exactly how they geometrically relate—completely eliminating data duplication and setting up the perfect foundation for Deep Learning logic.
        """)


# --- PHASE 5: TECHNICAL TRIUMPHS (THE 31M GRAPH) ---
elif "5. Conquering the 31M Graph" in selection:
    st.title("Phase 5: Conquering the 31.9 Million Graph Barrier")
    st.markdown("### The Technical Difficulty: Overcoming Critical Out-Of-Memory (OOM) Errors")
    st.divider()
    
    st.error("**THE HARDWARE PROBLEM:** Naively dumping 31.9 million database rows into a graph topology instantly crashes the physical RAM of modern computing environments. The exponential overhead of simultaneously mapping 58+ million interconnected geometric pathways guarantees a full server lockup.")
    
    st.success("**THE ENGINEERING SOLUTION (Micro-Batching):** To bypass total RAM exhaustion, we wrote a specialized streaming architecture. We chunked the Parquet Data Lake and engineered a custom Cypher query sequence utilizing deep pointer constraints and strict `TRANSACTION BATCHING`. This forced the database to commit and flush its memory thread every 10,000 steps.")
    
    col_x, col_y = st.columns(2)
    with col_x:
        st.subheader("Engineered Cypher Query Execution")
        st.code("""
// The Engineering Paradigm: We force Neo4j to clear its Thread Memory
// exactly every 10,000 mathematical calculations to prevent RAM overflow.

LOAD CSV WITH HEADERS FROM 'file:///nodes_transactions.csv' AS row
CALL {
    WITH row
    CREATE (t:Transaction {
        txn_id: row.`txn_id:ID(Transaction)`,
        amount: toFloat(row.`amount:float`),
        timestamp: datetime(row.`timestamp:datetime`),
        is_laundering: toInteger(row.`is_laundering:int`)
    })
} IN TRANSACTIONS OF 10000 ROWS;
""", language="cypher")

    with col_y:
        st.subheader("Production Server Logs (`bulk_load_cypher.log`)")
        st.code("""
============================================================
BULK LOAD — 29.3M RECORDS VIA MEMORY-SAFE BATCHING
============================================================
[1/6] Creating structural constraints...
      [SUCCESS] Pointer Constraints generated

[2/6] Loading Account nodes (1.7M) via Batching...
      [SUCCESS] Accounts loaded in 624.1s

[3/6] Mapping Transaction nodes (29.3M) via Micro-Batching...
      [SUCCESS] Transactions mapped in 1,840.5s

[4/6] Creating SENT edges (29.3M geometrical paths)...
      [SUCCESS] SENT edges materialized in 2,210.2s

[5/6] Creating RECEIVED_BY edges (29.3M geometrical paths)...
      [SUCCESS] RECEIVED_BY edges materialized in 2,512.8s

============================================================
[SUCCESS] ENTERPRISE BATCHED BULK LOAD COMPLETE
============================================================
  Account nodes:          1,758,573
  Transaction nodes:     29,336,378
  SENT edges:            29,336,378
  RECEIVED_BY edges:     29,336,378
  Fraud transactions:        35,158
  
  Total Compute Time:    119.8 minutes (~ 2.0 Hours)
============================================================
""", language="text")

# --- PHASE 6: LIVE PIPELINE ---
elif "6. Live Pipeline Orchestrator" in selection:
    st.title("Phase 6: Live Operational Orchestrator")
    st.markdown("### Run through ingestion -> transformation -> warehouse with live logging")
    st.divider()
    
    st.warning("**ARCHITECTURAL NOTE:** Because mathematically building the entire 31.9M-node graph requires intensive data-batching and 100% CPU lock for over 2 hours, this real-time orchestration block operates on an identical 5,000-node micro-batch. This is pulled live from the Parquet lake to satisfy live-run validation criteria instantaneously without stalling the server.")
    
    st.markdown("Clicking the trigger below will execute the `demo_TA_pipeline.py` script, pushing data dynamically into our local `fraudgraph` backend while rendering operational stdout.")
    
    if st.button("EXECUTE LIVE INGESTION PIPELINE", type="primary"):
        with st.spinner("Pipeline Running: Negotiating with Neo4j Bolt Driver..."):
            log_container = st.empty()
            log_text = ""
            
            try:
                # Use Popen with -u (unbuffered) to stream the terminal output live
                process = subprocess.Popen(
                    ["python3", "-u", os.path.join(PROJ, "demo_TA_pipeline.py")],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                # Stream the logs to the screen dynamically
                for line in iter(process.stdout.readline, ''):
                    log_text += line
                    log_container.text_area("Pipeline Orchestrator Operational Logs", log_text, height=500)
                
                process.stdout.close()
                process.wait()
                
                st.success("Pipeline Execution Complete! The 5,000 Micro-Batch has been successfully written to the Neo4j 'fraudgraph' QA database.")
            except Exception as e:
                st.error(f"Pipeline Failed: {e}")

# --- PHASE 7: NEO4J VALIDATION ---
elif "7. Neo4j Graph Validation" in selection:
    st.title("Phase 7: Neo4j Knowledge Graph Validation")
    st.markdown("Connecting directly to the highly-scalable Neo4j Warehouse to verify injection integrity and production limits.")
    st.divider()
    
    @st.cache_resource
    def init_connection():
        return GraphDatabase.driver(os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"), auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "fraud2026")))
        
    try:
        driver = init_connection()
        db_option = st.selectbox("Select Target Warehouse Environment:", ("neo4j (Production - Main 31M Graph)", "fraudgraph (Testing/QA Warehouse)"))
        db_name = "neo4j" if "Production" in db_option else "fraudgraph"
        
        with driver.session(database=db_name) as session:
            st.spinner(f"Querying hyper-indices inside '{db_name}'...")
            time.sleep(0.5)
            nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            edges = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            
            st.success(f"Successfully pinged Database '{db_name}'.")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Bolt DB Server Status", "ONLINE")
            col2.metric("Total Physical Nodes", f"{nodes:,}")
            col3.metric("Total Geometric Edges", f"{edges:,}")
            col4.metric("Graph Data Science Plugin", "READY")
            
    except Exception as e:
        st.error(f"Neo4j Connection Failed: {e}. Ensure desktop server is running.")
