const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, ImageRun,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle, LevelFormat,
  Footer, PageNumber, TableLayoutType,
} = require("docx");

const ROOT = "/Users/prajwaldambalkar/Developer/personal/Graph-Based-Fraud-Intelligence-Platform/docs/workbook1";
const FIG = (f) => path.join(ROOT, "figures", f);
const SHOT = (f) => path.join(ROOT, "screenshots", f);
const OUT = path.join(ROOT, "Team2-Workbook1.docx");

const FONT = "Calibri";
const CONTENT_W = 9360; // 6.5 in

// ---------- helpers ----------
function runs(text, base = {}) {
  // **bold** and _italic_ markup
  const out = [];
  const re = /(\*\*[^*]+\*\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text), spacing: { after: 120, line: 276 }, alignment: AlignmentType.JUSTIFIED, ...opts });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const B = (t, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, children: runs(t), spacing: { after: 60, line: 264 } });
const N = (t, ref = "num1") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(t), spacing: { after: 60, line: 264 } });

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
let figNo = { 4: 0, 5: 0 }, tabNo = { 4: 0, 5: 0 };
function figure(file, ch, caption, widthIn = 6.3) {
  figNo[ch] += 1;
  const { w, h } = pngSize(file);
  const pxW = Math.round(widthIn * 96), pxH = Math.round(pxW * h / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: pxW, height: pxH },
        altText: { title: `Figure ${ch}.${figNo[ch]}`, description: caption, name: `fig${ch}_${figNo[ch]}` } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: `Figure ${ch}.${figNo[ch]}. `, bold: true, size: 20 }), ...runs(caption, { size: 20 })] }),
  ];
}
const border = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(ch, caption, headers, rows, widths) {
  tabNo[ch] += 1;
  const total = widths.reduce((a, b) => a + b, 0);
  const scale = CONTENT_W / total;
  const ws = widths.map((w) => Math.round(w * scale));
  ws[ws.length - 1] += CONTENT_W - ws.reduce((a, b) => a + b, 0);
  const cell = (text, i, head) => new TableCell({
    borders, width: { size: ws[i], type: WidthType.DXA },
    shading: head ? { fill: "D9E2F3", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: String(text).split("\n").map((line) => new Paragraph({ spacing: { after: 0, line: 240 },
      children: runs(line, { size: 18, bold: head ? true : undefined }) })),
  });
  return [
    new Paragraph({ keepNext: true, spacing: { before: 160, after: 80 }, alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: `Table ${ch}.${tabNo[ch]}. `, bold: true, size: 20 }), ...runs(caption, { size: 20 })] }),
    new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: ws, layout: TableLayoutType.FIXED,
      rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
        ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i, false)) }))] }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}
function plainTable(rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const ws = widths.map((w) => Math.round(w * CONTENT_W / total));
  ws[ws.length - 1] += CONTENT_W - ws.reduce((a, b) => a + b, 0);
  const cell = (text, i) => new TableCell({ borders, width: { size: ws[i], type: WidthType.DXA },
    shading: i === 0 ? { fill: "D9E2F3", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({ spacing: { after: 0, line: 240 }, children: runs(text, { size: 19, bold: i === 0 ? true : undefined }) })] });
  return [new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: ws, layout: TableLayoutType.FIXED,
    rows: rows.map((r) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i)) })) }),
    new Paragraph({ spacing: { after: 120 }, children: [] })];
}
const center = (t, size, bold = true, after = 80) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after },
  children: [new TextRun({ text: t, bold, size })] });

// example report from the frozen dashboard snapshot
const dash = JSON.parse(fs.readFileSync("/Users/prajwaldambalkar/Developer/personal/Graph-Based-Fraud-Intelligence-Platform-298b/frontend/src/data/dashboard.json", "utf8"));
const rep = dash.reports[0];
const exampleJson = [
  "{",
  `  "case_id": "${rep.case}",`,
  `  "decision": "${rep.decision}",`,
  `  "pattern": "${rep.pattern}",`,
  `  "risk_level": "${rep.riskLevel}",`,
  `  "confidence": ${Number(rep.confidence).toFixed(2)},`,
  `  "evidence": ["${rep.evidence[0].slice(0, 48)}", ...],`,
  `  "recommended_actions": ["${rep.actions[0].slice(0, 60)}", ...]`,
  "}",
];
const code = (lines) => lines.map((l) => new Paragraph({ spacing: { after: 0, line: 240 }, indent: { left: 360 },
  children: [new TextRun({ text: l, font: "Courier New", size: 17 })] }));

// ---------- content ----------
const body = [];
const add = (...xs) => xs.flat().forEach((x) => body.push(x));

// Title block (template style)
add(
  center("DATA 298B", 28, true, 60),
  center("MSDA Project II Workbook 1", 28, true, 60),
  center("Team 2", 26, true, 60),
  center("AI-Driven Real-Time Fraud Detection and Trend Mitigation", 24, false, 60),
  center("Aishwarya Iyer, Keith Gonsalves, Om Dankhara, Prajwal Dambalkar", 22, true, 60),
  center("October 5, 2026", 22, true, 280),
);

add(new Paragraph({ spacing: { before: 120, after: 80 }, children: [new TextRun({ text: "Progress Summary", bold: true, size: 24 })] }));
add(plainTable([
  ["Targeted problems", "P1 transaction detection under 904:1 imbalance; P2 relational laundering patterns; P3 evidence-grounded investigation"],
  ["Final architecture", "Four layers: XGBoost detection (L1), Neo4j graph intelligence (L2), Leiden communities (L3), fine-tuned LLM investigator (L4)"],
  ["Best detection model", "Graph-enhanced XGBoost (23 features): test PR-AUC 0.462 (baseline 0.304), recall 99.2% (baseline 91.4%)"],
  ["LLM comparison", "Four QLoRA fine-tuned models on 50 held-out cases: binary F1 0.899 to 0.964, specificity 0.10 to 0.70"],
  ["Selected investigator", "Gemma-4 31B: binary F1 0.964, specificity 0.70, pattern macro-F1 0.225"],
  ["Unseen validation", "10 new cases with guardrails: recall 1.00, specificity 0.60 (2 of 5 legitimate cases flagged)"],
  ["Main limitation", "Pattern classification remains weak for all models (macro-F1 0.13 to 0.23)"],
], [1.8, 6.2]));
// ===================== 4 =====================
add(H1("4. Model Development"));
add(H2("4.1 Model Proposals"));
add(P("The project targets anti-money-laundering (AML) detection on the IBM AML HI-Medium dataset: 31,898,238 transactions between about 1.76 million accounts over 27 days in September 2022, of which 35,230 (0.11%) are laundering. This gives a class ratio of roughly 904 legitimate transactions per laundering transaction. We address three targeted problems:"));
add(
  N("**P1: Transaction-level detection under extreme imbalance.** Score every transaction for laundering risk when accuracy is meaningless and positives are rare.", "p"),
  N("**P2: Relational pattern discovery.** Laundering is organised across accounts (fan-in, fan-out, cycles, scatter-gather, bipartite and stack structures). A row-by-row model cannot see these structures; a graph can.", "p"),
  N("**P3: Explainable, evidence-grounded investigation.** A risk score alone does not help an analyst. The system must produce a structured case report whose claims trace back to transactions, graph neighbourhoods and communities.", "p"),
);
add(P("Figure 4.1 shows how the proposed models combine into a four-layer framework: transaction detection (L1), graph intelligence (L2), community intelligence (L3) and AML investigation (L4). We propose eight models (M1 to M8)."));
add(figure(FIG("fig1_system_architecture.png"), 4, "Overview of the proposed four-layer framework. Stage I prepares and chronologically splits the data; Stage II builds the knowledge graph, derives degree (L2) and Leiden community (L3) features and scores transactions with XGBoost (L1); Stage III (read right to left) turns flagged transactions into label-isolated cases that a fine-tuned LLM investigator (L4) reports on."));

add(H3("M1: Baseline XGBoost classifier (P1)"));
add(P("**Concept.** Gradient-boosted decision trees learn an additive ensemble of shallow trees, each fitted to the gradient of the logistic loss of the previous ensemble. **Features.** Eleven tabular features engineered from each transaction: log-transformed amount, amount bucket, hour, day of week, weekend flag, cross-currency flag and five payment-format indicators (ACH, cheque, credit card, wire, Bitcoin). **Algorithm and configuration.** XGBoost with the histogram tree method, 500 estimators, maximum depth 6, learning rate 0.05, row and column subsampling of 0.8, early stopping after 20 rounds on validation PR-AUC, and scale_pos_weight = 904 to re-weight the minority class instead of resampling. The decision threshold is tuned on the validation split."));
add(H3("M2: Degree-centrality graph analytics (P2)"));
add(P("**Concept.** Accounts that send to or receive from unusually many counterparties act as hubs, mules or collection points. **Graph.** A Neo4j knowledge graph models each transaction as a node: (Account)-[:SENT]->(Transaction)-[:RECEIVED_BY]->(Account), with 1,758,573 Account nodes, 29,336,378 Transaction nodes and 58.7M relationships (self-loops removed). **Algorithm.** Cypher aggregations compute in-degree, out-degree and total degree per account, and normalised degree centrality C(v) = deg(v) / (N − 1). The result is eight features per transaction (four for the sender, four for the receiver)."));
add(H3("M3: Leiden community detection (P2)"));
add(P("**Concept.** Laundering rings form densely connected groups of accounts. **Algorithm.** An undirected, weighted account-to-account graph is built from all sender-receiver pairs (edge weight = number of transactions), and the Leiden algorithm (leidenalg, RBConfigurationVertexPartition, a modularity-based quality function, fixed random seed) partitions it into 66,227 communities. Leiden refines Louvain by guaranteeing that every community is internally connected. **Features.** Community size and community fraud rate for sender and receiver (four features). The fraud rate uses training-split labels only, so validation and test labels never leak into the features."));
add(H3("M4: Graph-enhanced XGBoost (P1 + P2, improved model)"));
add(P("**Concept.** Fuse transaction attributes with relational context in one classifier. **Features.** A 23-dimensional vector per transaction: the 11 tabular features of M1, the 8 degree features of M2 and the 4 community features of M3 (Figure 4.4). **Algorithm.** The same XGBoost configuration as M1, so any gain is attributable to the graph features. This is our **improved model**: it raises test PR-AUC from 0.304 to 0.462 (+52%) and recall from 91.4% to 99.2%."));
add(H3("M5 to M8: Fine-tuned LLM investigators (P3, innovative model)"));
add(P("**Concept.** A large language model, adapted to AML investigation, turns a flagged case into a structured, evidence-grounded report. **Architectures.** Four decoder-only transformer families of different sizes: **M5** Ministral-3 14B (mistralai/Ministral-3-14B-Instruct-2512-BF16), **M6** Phi-4 14B (microsoft/phi-4), **M7** Qwen3.5 27B (Qwen/Qwen3.5-27B) and **M8** Gemma-4 31B (google/gemma-4-31B). **Algorithm.** Supervised fine-tuning with QLoRA: the base model is loaded in 4-bit NF4 with double quantisation and bf16 compute, and low-rank adapters (rank 16, alpha 32, dropout 0.05) are trained on all linear layers with gradient accumulation of 8. The SFT corpus pairs grounded case contexts with reference reports. **Inputs.** A label-isolated case context plus deterministic retrieval of transaction, graph and community evidence. **Output.** A JSON report containing a suspicious/legitimate decision, a laundering pattern (eight IBM typologies, EMERGING_UNKNOWN or NONE), risk level, confidence, evidence and recommended analyst actions."));
add(P("**Innovation.** Most AML tools either score transactions or prompt a general-purpose LLM. Our investigator differs in four ways: it is fine-tuned on grounded reports that follow a fixed schema; it uses graph and community evidence from M2 to M4; it never sees ground-truth labels, which go only to the evaluator; and its output passes deterministic guardrails that relabel or downgrade conclusions that the observed topology and XGBoost score do not support. The prompt-only Claude investigator built in DATA 298A is a legacy baseline only and is not part of the DATA 298B system."));

add(H2("4.2 Model Supports"));
add(H3("4.2.1 Platform, environment and tools"));
add(P("Development is split by computational need (Figure 4.2). Data preparation, graph analytics and XGBoost run on a local workstation (macOS, 24 GB RAM, 12 CPU cores) in a Python 3.11 virtual environment, with Neo4j and Airflow in Docker Compose. LLM fine-tuning and inference need a large GPU and run on Google Colab (40 GB GPU runtime), with datasets, adapters and traces on Google Drive and base checkpoints from Hugging Face. All code, configuration and frozen metrics are versioned on GitHub. Table 4.1 lists the environment and tools of each model."));
add(table(4, "Development and execution environment of each model.",
  ["Model", "Development environment", "Execution environment", "Key tools and libraries"],
  [
    ["M1 Baseline XGBoost", "Local Python 3.11 venv", "Local CPU (hist trees)", "xgboost 3.2, scikit-learn 1.9, SHAP 0.51, pandas 3.0, PyArrow"],
    ["M2 Degree centrality", "Neo4j 5.26 (Docker)", "Neo4j Cypher over Bolt", "Neo4j GDS 2.13, APOC, neo4j Python driver 6.3"],
    ["M3 Leiden communities", "Local Python 3.11 venv", "Local CPU (about 4 min)", "python-igraph 1.0, leidenalg 0.12, pandas"],
    ["M4 Graph-enhanced XGBoost", "Local Python 3.11 venv", "Local CPU", "xgboost, SHAP, feature store (parquet)"],
    ["M5 to M8 LLM investigators", "Google Colab notebooks", "Colab GPU (40 GB), one model at a time", "Transformers, PEFT, TRL (SFTTrainer), bitsandbytes 4-bit, Hugging Face Hub, Google Drive"],
    ["Orchestration and UI", "Docker Compose, Node.js", "Local services", "Airflow 3.2, Streamlit 1.65, React 19, Vite 8, TypeScript 5.9"],
  ], [1.6, 1.7, 1.8, 3.2]));
add(figure(FIG("wb_platform_tools.png"), 4, "Supporting platforms and tools. Detection and graph models run locally; LLMs are fine-tuned and executed on Colab GPUs, with data and adapters exchanged through Google Drive."));
add(H3("4.2.2 Machine learning data flows"));
add(P("Figure 4.3 shows the data preparation flow shared by all models. Schema cleaning and feature engineering produce a time-ordered parquet lake. Self-loops are removed only for the graph export (29,336,378 of 31,898,238 transactions), and the tabular data is split chronologically into training (22,328,766), validation (4,784,736) and test (4,784,736) sets. Figure 4.4 shows the graph schema and how M2 and M3 features join the tabular features to form the M4 input. Figure 4.5 shows the LLM workflow: QLoRA fine-tuning, label-isolated investigation and evaluation."));
add(figure(FIG("fig2_data_pipeline.png"), 4, "Data preparation pipeline from raw IBM AML files to the Neo4j graph export and the chronological 70/15/15 split."));
add(figure(FIG("fig3_graph_ontology_features.png"), 4, "(a) Knowledge-graph schema; † used only as an evaluation label and for community fraud rates. (b) The 23-dimensional feature vector of transaction t = (s → d) that feeds M4. Graph structure is computed on the full graph (transductive setting); community fraud rates use training labels only."));
add(figure(FIG("fig4_llm_investigator_evaluation.png"), 4, "LLM investigator workflow: (a) QLoRA fine-tuning of four base LLMs; (b) label-isolated cases enriched by deterministic retrieval; (c) guardrails, evaluation against ground truth that bypasses the model, and model selection."));

add(H2("4.3 Model Comparison and Justification"));
add(P("Table 4.2 compares the eight models by targeted problem, input, approach, strengths and limitations. The models work together: M2 and M3 produce features for M4, and M4 selects the cases that M5 to M8 investigate."));
add(table(4, "Comparison of proposed models.",
  ["Model", "Problem", "Input / features", "Approach", "Strengths", "Limitations"],
  [
    ["M1 Baseline XGBoost", "P1", "11 tabular features per transaction", "Supervised ML (gradient-boosted trees)", "Fast on 22M rows; handles imbalance via class weights; SHAP-interpretable", "Blind to multi-account structure; low precision at high recall"],
    ["M2 Degree centrality", "P2", "Account-transaction graph", "Statistical graph analytics", "Cheap, deterministic, interpretable hub and mule signal", "Unsupervised; banks and payroll accounts also have high degree"],
    ["M3 Leiden", "P2", "Weighted account-to-account graph", "Unsupervised graph clustering (modularity)", "Scales to 1.76M nodes; connected communities; captures ring context", "Transductive; community fraud rate depends on training labels"],
    ["M4 Graph-enhanced XGBoost", "P1, P2", "23 features (tabular + degree + community)", "Supervised ML with graph feature engineering", "Best detection (PR-AUC 0.462, recall 99.2%); same cost as M1", "Precision still low at the 0.30 threshold; graph features need refresh"],
    ["M5 Ministral-3 14B", "P3", "Case context + retrieved evidence", "Deep learning (transformer), QLoRA SFT", "Good pattern macro-F1 for its size", "Specificity 0.10; slow (83 s/case)"],
    ["M6 Phi-4 14B", "P3", "Same as M5", "Deep learning (transformer), QLoRA SFT", "Fastest (35 s/case); specificity 0.50", "Lowest pattern macro-F1 (0.13); more unsupported claims"],
    ["M7 Qwen3.5 27B", "P3", "Same as M5", "Deep learning (transformer), QLoRA SFT", "Best evidence support (0.64), fewest unsupported claims", "Specificity 0.10; slowest (173 s/case)"],
    ["M8 Gemma-4 31B", "P3", "Same as M5", "Deep learning (transformer), QLoRA SFT", "Best binary F1 (0.964), specificity (0.70) and pattern macro-F1 (0.23)", "Largest; 97 s/case; needs a 40 GB GPU"],
  ], [1.3, 0.75, 1.35, 1.45, 1.8, 1.75]));
add(H3("Justification"));
add(
  B("**M1.** Gradient-boosted trees are the industry baseline for tabular fraud data. They train on 22M rows in minutes, handle class imbalance without resampling through scale_pos_weight, and expose SHAP explanations. A strong baseline is required to measure what graph structure adds."),
  B("**M2.** Degree is the simplest structural signal. It directly encodes fan-in and fan-out behaviour, costs one aggregation query, and gives analysts an interpretable number."),
  B("**M3.** Laundering rings are group phenomena. Leiden is chosen over Louvain because it guarantees internally connected communities at the same computational cost, and over spectral or GNN methods because it scales to 1.76M accounts on a laptop in about four minutes."),
  B("**M4.** Keeping the learner fixed and adding only graph features shows how much those features contribute. PR-AUC improved by 52%, which supports our hypothesis that laundering is easier to detect when relationships between accounts are included."),
  B("**M5 to M8.** Open-weight models can be fine-tuned and hosted inside a bank's own boundary, which matters for confidential transaction data. Comparing four families from 14B to 31B parameters under one protocol shows the trade-off between accuracy, evidence faithfulness and latency. Gemma-4 31B is selected because it leads on binary F1, specificity and pattern macro-F1. Phi-4 remains the low-latency alternative, and Qwen the most evidence-faithful."),
);

add(H2("4.4 Model Evaluation Methods"));
add(P("All models are evaluated on data they did not see during training. Detection models use the chronological test split (4,784,736 transactions, 8,661 laundering), which avoids temporal leakage. LLM investigators use held-out cases whose labels and provenance are removed from the model-visible context (Table 4.3)."));
add(table(4, "Evaluation methods and metrics for each model.",
  ["Model", "Evaluation data", "Metrics", "Why these metrics"],
  [
    ["M1, M4 XGBoost", "Chronological test split; threshold tuned on validation", "PR-AUC (primary), ROC-AUC, precision, recall, F1, confusion matrix, PR curve; SHAP feature attribution", "At 0.11% prevalence accuracy is meaningless; PR-AUC measures ranking quality on the rare class; recall reflects missed laundering"],
    ["M2 Degree", "Full graph", "Degree distributions; ablation: change in M4 PR-AUC when degree features are added", "Unsupervised; judged by downstream lift and interpretability"],
    ["M3 Leiden", "Full graph; labels from training split only", "Number and size distribution of communities; community fraud-rate separation; ablation lift on M4; modularity-based quality function", "Shows whether communities concentrate laundering"],
    ["M5 to M8 LLMs", "50-case benchmark (40 suspicious, 5 per pattern, and 10 legitimate) + 10 unseen validation cases", "Schema validity; accuracy, precision, recall, binary F1, specificity; pattern macro-F1; evidence-supported rate; unsupported-claim rate; latency and tokens/s; manual audit", "Covers decision quality, false alarms, typology recognition, faithfulness to evidence and operating cost"],
  ], [1.2, 1.9, 2.6, 2.3]));
add(P("**Definitions.** With TP, FP, TN and FN the confusion-matrix counts: precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2·precision·recall/(precision+recall); specificity = TN/(TN+FP). PR-AUC is the area under the precision-recall curve as the threshold varies, and ROC-AUC the area under the true-positive versus false-positive rate curve. Pattern macro-F1 averages per-pattern F1 over all typologies, so rare patterns count equally. The evidence-supported rate is the share of report claims that match retrieved evidence, and the unsupported-claim rate the share that do not."));
add(P("**Protocol for LLMs.** Every model receives the same cases, evidence format, system prompt and output schema, and only one model is loaded at a time. Ground truth bypasses the model and reaches only the evaluator. Post-hoc guardrails are applied to saved reports and metrics recomputed without rerunning inference. A manual audit reviews five cases per model, and the selected model is finally tested on 10 unseen cases (5 suspicious, 5 legitimate)."));

add(H2("4.5 Model Validation and Evaluation Results"));
add(P("Table 4.4 and Figure 4.6 summarise detection on the test split. Adding graph features raises PR-AUC by 52% and recall from 91.4% to 99.2%. Degree features provide most of the gain (PR-AUC 0.459); community features add a smaller increment (0.462)."));
add(table(4, "Detection results on the chronological test split (4,784,736 transactions).",
  ["Model", "Features", "PR-AUC", "ROC-AUC", "Precision", "Recall", "F1"],
  [
    ["M1 Baseline XGBoost", "11", "0.304", "0.945", "0.017", "0.914", "0.034"],
    ["M4 with degree features only", "19", "0.459", "0.985", "n/a", "0.984", "n/a"],
    ["M4 Graph-enhanced XGBoost", "23", "**0.462**", "**0.985**", "0.011", "**0.992**", "0.022"],
  ], [2.4, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]));
add(figure(FIG("chart_detection_results.png"), 4, "Baseline versus graph-enhanced XGBoost on the test split."));
add(P("Figure 4.7 ranks features by mean absolute SHAP value for M4. The ACH flag is the strongest single feature (ACH accounts for 87.3% of the labelled laundering transactions in this dataset, which reflects how the IBM data was simulated rather than ACH being fraudulent in itself), but six of the top eight features are graph features, led by sender total degree, so the model relies heavily on account structure. Precision is low at the fixed 0.30 threshold because we prioritised recall; raising the threshold on validation data trades recall for precision when analysts can review fewer alerts."));
add(figure(FIG("chart_shap_importance.png"), 4, "Top 12 features of the graph-enhanced model by mean |SHAP| on the test split; dark bars are graph features."));
add(P("Table 4.5 and Figure 4.8 compare the four fine-tuned investigators on the 50-case benchmark. All four reach 100% schema validity and recall 1.00, so they differ mainly in false alarms on legitimate cases (specificity) and in typology recognition. Gemma-4 31B is selected. On the 10 unseen validation cases, Gemma with guardrails reached recall 1.00 and specificity 0.60 (2 of 5 legitimate cases flagged). Pattern macro-F1 remains low (0.13 to 0.23) for every model and is reported as the main limitation."));
add(table(4, "Fine-tuned LLM investigators on the 50-case integrated benchmark.",
  ["Model", "Accuracy", "Precision", "Recall", "Binary F1", "Specificity", "Pattern macro-F1", "Evidence supported", "Latency (s)"],
  [
    ["M5 Ministral-3 14B", "0.82", "0.816", "1.00", "0.899", "0.10", "0.206", "0.505", "83.3"],
    ["M6 Phi-4 14B", "0.90", "0.889", "1.00", "0.941", "0.50", "0.128", "0.409", "35.0"],
    ["M7 Qwen3.5 27B", "0.82", "0.816", "1.00", "0.899", "0.10", "0.191", "0.638", "172.6"],
    ["M8 Gemma-4 31B", "**0.94**", "**0.930**", "1.00", "**0.964**", "**0.70**", "**0.225**", "0.542", "97.5"],
  ], [1.7, 0.8, 0.8, 0.7, 0.8, 0.85, 0.95, 0.95, 0.8]));
add(figure(FIG("chart_llm_comparison.png"), 4, "Decision quality, typology recognition and evidence faithfulness of the four fine-tuned investigators."));

// ===================== 5 =====================
add(H1("5. Data Analytics System"));
add(H2("5.1 System Requirements Analysis"));
add(H3("5.1.1 System boundary, actors and use cases"));
add(P("Figure 5.1 shows the system boundary, actors and use cases. The boundary contains data ingestion and preprocessing, the knowledge graph, the detection and investigation models and the analyst dashboards. Outside the boundary are the IBM AML source on Kaggle, the external GPU and model hosting (Google Colab, Hugging Face) and downstream recipients of reports such as regulators. Three actors interact with the system: the **AML analyst** (primary actor) who reviews alerts and investigates cases; the **compliance officer** who consumes validated reports; and the **data/ML engineer** who operates the pipeline and models. Shaded use cases are powered by AI models."));
add(figure(FIG("wb_use_cases.png"), 5, "System boundary, actors and use cases; shaded use cases are AI-powered."));
add(
  N("**Review flagged transactions (AML analyst).** The analyst opens the transaction queue ranked by the XGBoost risk score, filters by account, time or payment format, and opens transaction details.", "uc"),
  N("**Explore account network (AML analyst).** For a sender or receiver, the system retrieves the live Neo4j neighbourhood with in-, out- and total degree, centrality and Leiden community.", "uc"),
  N("**Investigate case with an LLM report (AML analyst).** The analyst opens the investigator report for a case: decision, pattern, risk level, confidence, cited evidence and recommended actions. This use case includes network exploration, because the report cites graph evidence.", "uc"),
  N("**Compare detection and LLM models (AML analyst, ML engineer).** Side-by-side detection metrics and the four-model LLM comparison support model governance.", "uc"),
  N("**View and export reports (AML analyst, compliance officer).** Validated reports are listed, filtered by risk level and opened in the investigator for sign-off.", "uc"),
  N("**Ingest and preprocess data (ML engineer).** Download the IBM AML files, clean, engineer features and split chronologically.", "uc"),
  N("**Load and refresh the graph (ML engineer).** Export the graph ontology and bulk-load Neo4j, then recompute degree and community features.", "uc"),
  N("**Train and evaluate models (ML engineer).** Retrain XGBoost, tune thresholds, fine-tune and benchmark the LLM investigators.", "uc"),
  N("**Run the orchestrated pipeline (ML engineer).** Trigger the Airflow DAG that validates artifacts, reloads the QA graph and records metrics.", "uc"),
);
add(H3("5.1.2 Data analytics and machine learning capabilities"));
add(
  N("**Transaction risk scoring.** M1/M4 assign a laundering probability to every transaction. Output: risk score, binary flag at threshold τ and SHAP explanation.", "cap"),
  N("**Graph intelligence.** M2 and Neo4j provide account degree, centrality and neighbourhood retrieval for any account in under a second. Output: neighbourhood subgraph and degree metrics.", "cap"),
  N("**Community intelligence.** M3 assigns every account to one of 66,227 Leiden communities with size and historical fraud rate. Output: community context and ring-level risk signals.", "cap"),
  N("**Evidence-grounded investigation.** M8 (Gemma-4 31B) converts a flagged case and its retrieved evidence into a structured JSON report, checked by guardrails. Output: decision, pattern, risk, confidence, evidence and actions.", "cap"),
  N("**Model comparison and monitoring.** Frozen metrics for detection and LLM models are exposed for governance.", "cap"),
  N("**Reproducible orchestration.** Airflow DAGs rerun ingestion checks, reload the QA graph and emit operational metrics.", "cap"),
);

add(H2("5.2 System Design"));
add(H3("5.2.1 System architecture and infrastructure"));
add(P("The system follows a three-tier architecture (Figure 5.2). The **presentation tier** offers the React analyst dashboard, the Streamlit pipeline dashboard and the Neo4j Browser for ad hoc Cypher exploration. The **application and AI tier** contains the AI-powered function components: a read-only graph API that serves account neighbourhoods over HTTP/JSON, the detection service (XGBoost scoring with SHAP), the LLM investigator (fine-tuned Gemma-4 31B, run in batch on Colab GPUs) and Airflow orchestration. The **data tier** holds the Neo4j knowledge graph, the parquet data lake and feature store, the artifact store of models, metrics and report JSON, Google Drive for LLM artifacts, and a separate QA graph that the demo pipeline can reset without touching the full graph. The functional flow of a case is the one in Figure 4.1: score, flag, construct the case, retrieve evidence, investigate, apply guardrails and present."));
add(figure(FIG("wb_infrastructure.png"), 5, "Three-tier system architecture and infrastructure with AI-powered components (shaded)."));
add(H3("5.2.2 Supporting platforms, frameworks and cloud environment"));
add(P("Figure 4.2 shows how the platforms integrate. Table 5.1 summarises each integration. The cloud environment is Google Colab (GPU compute) with Google Drive (shared storage) and the Hugging Face Hub (model checkpoints); the graph, data lake and dashboards run as local Docker and Python services. Because every service is containerised or pinned in requirements files, the same stack can move to a managed cloud (for example Neo4j Aura and a cloud VM) without code changes. That move is planned and has not been carried out."));
add(table(5, "Integration with supporting platforms, frameworks and cloud services.",
  ["Platform / framework", "Role in the system", "Integration mechanism"],
  [
    ["Kaggle", "Source of the IBM AML HI-Medium dataset", "kagglehub dataset download with an access token"],
    ["Neo4j 5.26 (Docker)", "Knowledge graph, degree analytics, neighbourhood queries", "Bolt protocol via the neo4j Python driver; LOAD CSV bulk import; Cypher"],
    ["Google Colab (GPU)", "QLoRA fine-tuning and batch inference of LLMs", "Jupyter notebooks; Drive mounted at /content/drive"],
    ["Google Drive", "SFT corpus, held-out cases, LoRA adapters, traces, metrics", "Drive mount in Colab; files copied to the repo as frozen artifacts"],
    ["Hugging Face Hub", "Base model checkpoints (Ministral, Phi-4, Qwen, Gemma)", "Transformers from_pretrained with an access token"],
    ["Apache Airflow 3.2", "Pipeline orchestration and QA graph refresh", "Docker container; DAGs call the project scripts; Bolt to the QA Neo4j"],
    ["GitHub", "Version control, collaboration, frozen metrics", "Feature branches and pull requests"],
    ["React + Vite, Streamlit", "Analyst and pipeline dashboards", "React reads the frozen snapshot and calls the graph API; Streamlit reads local artifacts"],
  ], [1.8, 3.0, 3.2]));
add(H3("5.2.3 Data management solution and database"));
add(P("Data moves through five managed zones (Figure 5.3). **Raw**: the original Kaggle CSV files (2.8 GB of transactions) are kept unchanged in the download cache. **Curated**: cleaned, typed and time-sorted data is stored as columnar parquet (Snappy compression), including the clean table (31.9M rows), the graph table without self-loops (29.3M rows) and the three chronological splits. **Graph and features**: Neo4j stores the knowledge graph (9.6 GB store), and the feature store holds per-account graph features and the enriched splits. **Models**: XGBoost models, metrics and SHAP values are stored locally; LLM artifacts live on Google Drive. **Serving**: a frozen metrics manifest and dashboard snapshot (JSON) are versioned in Git and read by the dashboards. Raw and processed data are kept out of Git, and ground-truth labels are visible only to evaluation code."));
add(figure(FIG("wb_data_management.png"), 5, "Data management zones from raw files to serving data, with storage technologies and volumes."));
add(table(5, "Data stores.",
  ["Store", "Technology", "Content", "Volume", "Access"],
  [
    ["Raw zone", "CSV (Kaggle cache)", "Transactions, accounts, laundering patterns", "2.9 GB", "Read once by preprocessing"],
    ["Data lake", "Parquet (PyArrow)", "Clean, graph and split tables", "3.5 GB", "Batch read by models"],
    ["Knowledge graph", "Neo4j 5.26", "Accounts, transactions, SENT / RECEIVED_BY", "31.1M nodes, 58.7M rels", "Bolt / Cypher; read-only API"],
    ["Feature store", "CSV + parquet", "Account graph features; enriched splits", "1.8 GB", "Batch read by M4 and the case builder"],
    ["Model and artifact store", "JSON, parquet, PNG", "XGBoost models, metrics, SHAP, report snapshot", "< 50 MB", "Dashboards, reports"],
    ["LLM artifact store", "Google Drive (JSONL, safetensors)", "SFT data, cases, adapters, traces", "Several GB", "Colab notebooks"],
  ], [1.4, 1.5, 2.2, 1.4, 1.7]));
add(table(5, "Neo4j graph database schema.",
  ["Element", "Properties", "Constraints / indexes"],
  [
    ["(:Account)", "account_id, bank_id", "Unique constraint on account_id"],
    ["(:Transaction)", "txn_id, amount, timestamp, payment_format, currency, is_laundering", "Unique constraint on txn_id; index on is_laundering"],
    ["[:SENT]", "Account → Transaction (29,336,378)", "Created in micro-batches of 5,000 rows"],
    ["[:RECEIVED_BY]", "Transaction → Account (29,336,378)", "Created in micro-batches of 5,000 rows"],
    ["(:FraudPattern), [:PART_OF_PATTERN]", "type, attempt_id (8 IBM typologies, 2,756 attempts)", "Optional overlay for ground-truth analysis"],
  ], [2.0, 3.6, 2.4]));
add(H3("5.2.4 User interface and data visualization"));
add(P("The analyst dashboard (React, Vite, TypeScript, Tailwind) has six pages that follow the investigation workflow. Figures 5.4 to 5.8 are screenshots of the running system. The **Overview** shows dataset scale, the detection pipeline and the selected investigator. **Transactions** is the review queue with search and filters, where evaluator-only labels are hidden unless evaluation mode is enabled. **Network** renders the live Neo4j neighbourhood of an account with degree, centrality and community. The **LLM Investigator** shows the case header, escalation reason, cited evidence and recommended actions. **Reports** lists validated investigation reports by risk level. A separate Streamlit dashboard provides eight operational views (command center, pipeline, model performance, live predictions, graph communities, LLM investigator, operational readiness and project briefing)."));
add(figure(SHOT("react_overview.png"), 5, "Overview page: dataset scale, detection pipeline, detection reference metrics and the selected investigator model.", 5.4));
add(figure(SHOT("react_transactions.png"), 5, "Transactions page: held-out review queue with search and payment-format filters; risk scores appear where present in the saved snapshot.", 5.4));
add(figure(SHOT("react_graph.png"), 5, "Network page: live Neo4j neighbourhood of an account with degree, centrality and Leiden community.", 5.4));
add(figure(SHOT("react_llm.png"), 5, "LLM Investigator page: Gemma-4 31B report with decision, confidence, escalation reason and retrieved evidence.", 5.4));
add(figure(SHOT("react_reports.png"), 5, "Reports page: validated investigation reports filtered by risk level.", 5.4));

add(H2("5.3 Intelligent Solution"));
add(H3("5.3.1 AI and machine learning solutions for each targeted problem"));
add(P("The models are connected in one pipeline. Graph models feed the detector, and the detector feeds the investigator:"));
add(
  B("**P1 (detection):** M4 graph-enhanced XGBoost scores every transaction. M1 is retained as the baseline that quantifies the graph contribution, and the threshold τ is tuned on validation data."),
  B("**P2 (relational patterns):** M2 degree analytics and M3 Leiden communities turn the 29.3M-transaction graph into eight degree and four community features per transaction (stacked into M4), and into live neighbourhood and community context for analysts."),
  B("**P3 (investigation):** M5 to M8 are fine-tuned with QLoRA and benchmarked under one protocol. The selected M8 (Gemma-4 31B) produces the case reports, which then pass deterministic guardrails. A fan-in or fan-out label that the observed topology does not support is relabelled NONE. A single self-loop is not counted as a laundering cycle. A suspicious decision with neither a structural signal nor an XGBoost score of at least 0.30 is downgraded to legitimate with low risk and confidence capped at 0.50. A suspicious decision without evidence is flagged for analyst review. Every change is recorded as a guardrail flag on the report."),
);
add(P("Development proceeded in three stages. **Initial:** data acquisition, cleaning, feature engineering, chronological splitting and the Neo4j bulk load (about 18 minutes for 31.1M nodes). **Intermediate:** degree and Leiden features, XGBoost training and threshold tuning, construction of the grounded SFT corpus and leakage-checked held-out cases, and QLoRA fine-tuning and benchmarking of four LLMs on Colab. **Final:** model selection, guardrails, the frozen metrics manifest, and integration into the dashboards, the read-only graph API and the Airflow pipeline."));
add(H3("5.3.2 Input datasets, expected outputs and supporting systems"));
add(table(5, "Inputs and expected outputs.",
  ["Item", "Description"],
  [
    ["Input: transactions", "IBM AML HI-Medium_Trans.csv: 31,898,238 rows with timestamp, from/to bank and account, amounts and currencies, payment format and laundering label"],
    ["Input: accounts", "HI-Medium_accounts.csv: account to bank and entity mapping (1.76M accounts)"],
    ["Input: patterns", "HI-Medium_Patterns.txt: 2,756 laundering attempts across 8 typologies (ground truth for benchmarks)"],
    ["Input: LLM training data", "Grounded SFT corpus of case → report pairs; held-out benchmark (50 cases) and validation (10 cases)"],
    ["Output: risk scores", "Per-transaction laundering probability and flag at threshold τ, with SHAP attribution"],
    ["Output: graph context", "Account degree, centrality, neighbourhood edges, Leiden community ID, size and fraud rate"],
    ["Output: investigation report", "JSON: decision, pattern, risk level, confidence, evidence list, recommended actions, summary"],
    ["Output: metrics", "Frozen manifest of detection and LLM metrics; Airflow run logs"],
  ], [2.0, 6.0]));
add(P("An example investigation report produced by the selected model (abridged) is shown below:"));
add(code(exampleJson));
add(P(" "));
add(P("**Supporting systems.** A Docker host runs Neo4j (full and QA instances) and Airflow; Google Colab GPUs (40 GB) run LLM training and inference; Google Drive and the Hugging Face Hub store LLM data and checkpoints; GitHub hosts code and frozen artifacts."));
add(H3("5.3.3 Solution APIs"));
add(table(5, "Solution APIs.",
  ["API", "Purpose", "Interface"],
  [
    ["Graph API: GET /api/health", "Liveness and Neo4j connectivity check", "HTTP/JSON, read-only (port 8765)"],
    ["Graph API: GET /api/neo4j/stats", "Node and relationship counts", "HTTP/JSON"],
    ["Graph API: GET /api/graph/{account_id}?limit=n", "Account neighbourhood edges with in-, out- and total degree and centrality", "HTTP/JSON; Cypher over Bolt behind it"],
    ["Neo4j Bolt + Cypher", "Bulk load, degree extraction, neighbourhood queries", "neo4j Python driver"],
    ["kagglehub", "Authenticated dataset download", "kagglehub.dataset_download()"],
    ["Hugging Face Transformers / PEFT / TRL", "Model loading, LoRA adapters, supervised fine-tuning", "Python APIs (from_pretrained, LoraConfig, SFTTrainer)"],
    ["XGBoost / SHAP", "Training, scoring and attribution", "Python APIs (XGBClassifier, TreeExplainer)"],
    ["Airflow", "Trigger and monitor pipeline DAGs", "Airflow UI and CLI (port 8080)"],
  ], [2.6, 3.0, 2.4]));
add(P("The Anthropic Messages API was used only by the DATA 298A prompt-only investigator, which is a legacy baseline. It is not part of the DATA 298B system."));

add(H2("5.4 System Supporting Environment"));
add(P("Table 5.6 lists the technologies, platforms and frameworks of the supporting environment."));
add(table(5, "System supporting environment.",
  ["Layer", "Technology (version)", "Purpose"],
  [
    ["Hardware", "Apple-silicon MacBook, 12 cores, 24 GB RAM; Docker Desktop (8 GB); Colab GPU runtime (40 GB)", "Local data, graph and detection workloads; GPU for LLMs"],
    ["Language and runtime", "Python 3.11; Node.js 26", "Pipelines, models, API; dashboard build"],
    ["Data processing", "pandas 3.0, NumPy, PyArrow 25 (parquet)", "Cleaning, feature engineering, columnar storage"],
    ["Graph database", "Neo4j 5.26 Community, Graph Data Science 2.13, APOC; neo4j driver 6.3", "Knowledge graph, analytics, queries"],
    ["Graph ML", "python-igraph 1.0, leidenalg 0.12", "Leiden community detection"],
    ["Machine learning", "XGBoost 3.2, scikit-learn 1.9, SHAP 0.51", "Detection models and explanations"],
    ["LLM training and inference", "Transformers ≥ 4.45, PEFT ≥ 0.12, TRL ≥ 0.10, bitsandbytes ≥ 0.43, Hugging Face Hub", "QLoRA fine-tuning and inference of M5 to M8"],
    ["Orchestration and containers", "Docker Compose; Apache Airflow 3.2", "Reproducible services and pipeline runs"],
    ["User interfaces", "React 19, Vite 8, TypeScript 5.9, Tailwind CSS 4; Streamlit 1.65; matplotlib 3.11", "Analyst and pipeline dashboards, charts"],
    ["Storage and collaboration", "Google Drive; GitHub", "LLM artifacts; version control and reviews"],
  ], [1.7, 3.6, 2.7]));
add(P("**Local workstation and Docker.** The workstation runs the Python 3.11 environment for preprocessing, Leiden and XGBoost, and Docker Compose for two Neo4j instances (full graph and QA graph) and Airflow. Containerisation makes the graph and orchestration layers reproducible on any teammate's machine."));
add(P("**Google Colab and Google Drive.** The LLMs (14B to 31B parameters) exceed local hardware, so QLoRA fine-tuning and inference run on a 40 GB Colab GPU, one model at a time. Drive stores the SFT corpus, held-out cases, adapters and resumable traces, so long runs survive interruptions."));
add(P("**GitHub.** One repository holds the pipeline scripts, notebooks, Docker and Airflow configuration, frozen metrics and the dashboard snapshot. Work happens on feature branches merged through pull requests."));
add(P("**Neo4j and dashboards.** Neo4j provides native graph storage and Cypher for neighbourhood queries that would require expensive self-joins in a relational database. The React dashboard serves analysts, and the Streamlit dashboard serves pipeline monitoring."));

// ---------- document ----------
const doc = new Document({
  creator: "Team 2",
  title: "Team 2 Workbook 1",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: FONT, color: "1F4E79" }, paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: FONT, color: "000000" }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1, keepNext: true } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, italics: true, font: FONT, color: "333333" }, paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 2, keepNext: true } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
      ...["p", "uc", "cap", "num1"].map((ref) => ({ reference: ref, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] })),
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ children: [PageNumber.CURRENT], size: 20 })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT, buf.length, "bytes; figures", figNo, "tables", tabNo); });
