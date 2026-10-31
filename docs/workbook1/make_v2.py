import re, sys

SRC = "/Users/prajwaldambalkar/Developer/personal/Graph-Based-Fraud-Intelligence-Platform/docs/workbook1/build_workbook.js"
DST = "/private/tmp/claude-501/-Users-prajwaldambalkar-Developer-personal-Graph-Based-Fraud-Intelligence-Platform/14bac5cb-aa15-4790-9299-1afffd4441a1/scratchpad/wb/build_v2.js"
s = open(SRC).read()

def rep(a, b, count=1):
    global s
    n = s.count(a)
    if n != count:
        sys.exit(f"anchor found {n}x (expected {count}): {a[:90]!r}")
    s = s.replace(a, b)

# ---------- output + imports ----------
rep('const OUT = path.join(ROOT, "Team2-Workbook1.docx");', 'const OUT = path.join(ROOT, "Team2-Workbook1-v2.docx");')
rep("Footer, PageNumber, TableLayoutType,", "Footer, PageNumber, TableLayoutType, TabStopType,")

# ---------- automatic figure/table references ----------
rep("let figNo = { 4: 0, 5: 0 }, tabNo = { 4: 0, 5: 0 };", r'''let figNo = { 4: 0, 5: 0 }, tabNo = { 4: 0, 5: 0 };
const REFS_FILE = path.join(__dirname, "refs_v2.json");
let REFS = {}; try { REFS = JSON.parse(fs.readFileSync(REFS_FILE, "utf8")); } catch (e) {}
const NEWREFS = {};
const FIGKEYS = { fig1_system_architecture: "arch", wb_platform_tools: "platform", fig2_data_pipeline: "pipeline",
  fig3_graph_ontology_features: "onto", fig4_llm_investigator_evaluation: "llmflow", chart_detection_results: "detchart",
  chart_shap_importance: "shap", chart_llm_comparison: "llmchart", wb_use_cases: "usecase", wb_infrastructure: "infra",
  wb_data_management: "dm", react_overview: "ui1", react_transactions: "ui2", react_graph: "ui3", react_llm: "ui4",
  react_reports: "ui5", wb_cloud_design: "cloud" };
const TABKEYS = {
  "Development and execution environment of each model.": "env",
  "Targeted problems addressed by each model.": "pxm",
  "Comparison of proposed models.": "compare",
  "Non-functional comparison of the models.": "nfr",
  "Evaluation methods and metrics for each model.": "evalm",
  "Metrics applied to each model.": "mxm",
  "Detection results on the chronological test split (4,784,736 transactions).": "detres",
  "Fine-tuned LLM investigators on the 50-case integrated benchmark.": "llmres",
  "Hand check of report GEMMA-001 against the source data.": "sample",
  "Functional (FR) and non-functional (NFR) requirements.": "req",
  "Data analytics and machine learning capabilities.": "cap",
  "Integration with supporting platforms, frameworks and cloud services.": "integ",
  "Data stores.": "stores",
  "Neo4j graph database schema.": "schema",
  "Inputs and expected outputs.": "io",
  "Solution APIs.": "apis",
  "System supporting environment.": "env54",
};
const RESOLVE = (t) => t.replace(/\{(fig|tab|n):([a-z0-9]+)\}/g, (m, k, n) => {
  const v = REFS[n]; if (!v) return m; return k === "fig" ? `Figure ${v}` : k === "tab" ? `Table ${v}` : v; });''')
rep("function runs(text, base = {}) {", "function runs(text, base = {}) {\n  text = RESOLVE(text);")
rep("  figNo[ch] += 1;\n", "  figNo[ch] += 1;\n  { const k = FIGKEYS[path.basename(file, \".png\")]; if (k) NEWREFS[k] = `${ch}.${figNo[ch]}`; }\n")
rep("  tabNo[ch] += 1;\n", "  tabNo[ch] += 1;\n  if (TABKEYS[caption]) NEWREFS[TABKEYS[caption]] = `${ch}.${tabNo[ch]}`;\n")
rep('Packer.toBuffer(doc).then((buf) => {', 'fs.writeFileSync(REFS_FILE, JSON.stringify(NEWREFS, null, 1));\nconst unresolved = JSON.stringify(REFS) !== JSON.stringify(NEWREFS);\nPacker.toBuffer(doc).then((buf) => { if (unresolved) console.log("REFS CHANGED: run again");')

# equation helper
rep("const center = (t, size", '''const EQ = (text, n) => new Paragraph({ spacing: { before: 80, after: 120 },
  tabStops: [{ type: TabStopType.CENTER, position: Math.round(CONTENT_W / 2) }, { type: TabStopType.RIGHT, position: CONTENT_W }],
  children: [new TextRun({ text: "\\t" + text, font: "Cambria Math", italics: true, size: 22 }), new TextRun({ text: `\\t(${n})`, size: 22 })] });
const center = (t, size''')

# ---------- replace hard-coded references ----------
for a, b in [
    ("Figure 4.1 shows how the proposed models", "{fig:arch} shows how the proposed models"),
    ("tabular features of M1, the 8 degree features of M2 and the 4 community features of M3 (Figure 4.4)", "tabular features of M1, the 8 degree features of M2 and the 4 community features of M3 ({fig:onto})"),
    ("Development is split by computational need (Figure 4.2).", "Development is split by computational need ({fig:platform})."),
    ("Table 4.1 lists the environment and tools", "{tab:env} lists the environment and tools"),
    ("Figure 4.3 shows the data preparation flow", "{fig:pipeline} shows the data preparation flow"),
    ("Figure 4.4 shows the graph schema", "{fig:onto} shows the graph schema"),
    ("Figure 4.5 shows the LLM workflow", "{fig:llmflow} shows the LLM workflow"),
    ("Table 4.2 compares the eight models", "{tab:compare} compares the eight models"),
    ("model-visible context (Table 4.3).", "model-visible context ({tab:evalm})."),
    ("Table 4.4 and Figure 4.6 summarise", "{tab:detres} and {fig:detchart} summarise"),
    ("Figure 4.7 ranks features", "{fig:shap} ranks features"),
    ("Table 4.5 and Figure 4.8 compare", "{tab:llmres} and {fig:llmchart} compare"),
    ("Figure 5.1 shows the system boundary", "{fig:usecase} shows the system boundary"),
    ("three-tier architecture (Figure 5.2)", "three-tier architecture ({fig:infra})"),
    ("the one in Figure 4.1:", "the one in {fig:arch}:"),
    ("Figure 4.2 shows how the platforms integrate. Table 5.1 summarises", "{fig:platform} shows how the platforms integrate. {tab:integ} summarises"),
    ("five managed zones (Figure 5.3)", "five managed zones ({fig:dm})"),
    ("Figures 5.4 to 5.8 are screenshots", "Figures {n:ui1} to {n:ui5} are screenshots"),
    ("Table 5.6 lists the technologies", "{tab:env54} lists the technologies"),
]:
    rep(a, b)

# ---------- citations ----------
for a, b in [
    ("on the IBM AML HI-Medium dataset:", "on the IBM AML HI-Medium dataset [1]:"),
    ("Gradient-boosted decision trees learn", "Gradient-boosted decision trees [2] learn"),
    ("A Neo4j knowledge graph models", "A Neo4j [3] knowledge graph models"),
    ("the Leiden algorithm (leidenalg, RBConfigurationVertexPartition, a modularity-based quality function, fixed random seed)",
     "the Leiden algorithm [4] (leidenalg, RBConfigurationVertexPartition, a modularity-based quality function [5], fixed random seed)"),
    ("Leiden refines Louvain by", "Leiden refines Louvain [6] by"),
    ("Supervised fine-tuning with QLoRA:", "Supervised fine-tuning with QLoRA [7]:"),
    ("and low-rank adapters (rank 16", "and low-rank adapters [8] (rank 16"),
    ("(mistralai/Ministral-3-14B-Instruct-2512-BF16)", "(mistralai/Ministral-3-14B-Instruct-2512-BF16) [9]"),
    ("(microsoft/phi-4)", "(microsoft/phi-4) [10]"),
    ("(Qwen/Qwen3.5-27B)", "(Qwen/Qwen3.5-27B) [11]"),
    ("(google/gemma-4-31B)", "(google/gemma-4-31B) [12]"),
    ("Most AML tools either score transactions or prompt a general-purpose LLM.",
     "Most AML tools either score transactions, sometimes with graph neural networks [13], or prompt a general-purpose LLM."),
    ("it uses graph and community evidence from M2 to M4;", "it retrieves graph and community evidence from M2 to M4, following the retrieval-augmented generation idea [14];"),
    ("and expose SHAP explanations.", "and expose SHAP explanations [15]."),
]:
    rep(a, b)

# ---------- equations ----------
rep("The decision threshold is tuned on the validation split.\"));",
    "The decision threshold is tuned on the validation split.\"));\n"
    "add(P(\"The minority-class weight and the decision rule are:\"));\n"
    "add(EQ(\"w₊ = N₋ / N₊ ≈ 904,     ŷ = 1  if  p(fraud | x) ≥ τ\", 1));")
rep("The result is eight features per transaction (four for the sender, four for the receiver).\"));",
    "The result is eight features per transaction (four for the sender, four for the receiver).\"));\n"
    "add(EQ(\"C(v) = ( deg_in(v) + deg_out(v) ) / ( N − 1 ),     N = 1,758,573\", 2));")
rep("so validation and test labels never leak into the features.\"));",
    "so validation and test labels never leak into the features.\"));\n"
    "add(P(\"The partition maximises the RB-configuration (modularity) quality function with the default resolution γ = 1:\"));\n"
    "add(EQ(\"Q = Σᵢⱼ ( Aᵢⱼ − γ kᵢ kⱼ / 2m ) δ(cᵢ, cⱼ)\", 3));\n"
    "add(P(\"where Aᵢⱼ is the number of transactions between accounts i and j, kᵢ is the weighted degree of account i, m is the total edge weight, and δ(cᵢ, cⱼ) is 1 when both accounts are in the same community and 0 otherwise.\"));")
rep("evidence and recommended analyst actions.\"));",
    "evidence and recommended analyst actions.\"));\n"
    "add(P(\"For a frozen base weight matrix W₀, each adapted layer computes:\"));\n"
    "add(EQ(\"h = W₀x + (α / r) · B A x,     B ∈ ℝ^(d×r),  A ∈ ℝ^(r×k),  r = 16,  α = 32\", 4));")

# ---------- (a) problem x model matrix ----------
rep('add(H2("4.2 Model Supports"));', '''add(P("{tab:pxm} summarises which targeted problem each model addresses and its role in the system."));
add(table(4, "Targeted problems addressed by each model.",
  ["Model", "P1 Detection", "P2 Relational patterns", "P3 Investigation", "Role in the system"],
  [
    ["M1 Baseline XGBoost", "✓", "", "", "Baseline detector"],
    ["M2 Degree centrality", "", "✓", "", "Graph feature producer; live neighbourhoods"],
    ["M3 Leiden communities", "", "✓", "", "Community feature producer"],
    ["M4 Graph-enhanced XGBoost", "✓", "✓", "", "Production detector (improved model)"],
    ["M5 Ministral-3 14B", "", "", "✓", "Candidate investigator"],
    ["M6 Phi-4 14B", "", "", "✓", "Candidate investigator"],
    ["M7 Qwen3.5 27B", "", "", "✓", "Candidate investigator"],
    ["M8 Gemma-4 31B", "", "", "✓", "Selected investigator"],
  ], [2.0, 1.0, 1.3, 1.1, 2.6]));
add(H2("4.2 Model Supports"));''')

# ---------- (b) non-functional comparison + alternatives ----------
rep('add(H2("4.4 Model Evaluation Methods"));', '''add(P("{tab:nfr} compares the non-functional characteristics of the models. Values marked as measured come from this project; the others are relative ratings."));
add(table(4, "Non-functional comparison of the models.",
  ["Model", "Hardware", "Training cost", "Serving latency", "Interpretability", "Maintenance"],
  [
    ["M1, M4 XGBoost", "CPU only", "Low: CPU, no GPU", "About 0.1 µs per transaction in batch (measured: 500,000 rows in 0.05 s)", "High: SHAP values per prediction", "Retrain on drift; M4 also needs refreshed graph features"],
    ["M2 Degree centrality", "Neo4j host, 9.6 GB store", "One bulk load (18 min, measured) plus aggregation", "Neighbourhood query about 0.1 s (measured)", "High: plain counts", "Recompute after each graph refresh"],
    ["M3 Leiden", "CPU (igraph)", "About 4 min per run", "Precomputed lookup", "Medium: community IDs need context", "Rerun on refresh; IDs change between runs"],
    ["M5 to M8 LLMs", "GPU with at least 40 GB", "High: GPU hours per model for QLoRA", "35 to 173 s per case (measured)", "Medium: readable evidence, but only 41% to 64% of claims supported", "One adapter per model; re-benchmark after retraining"],
  ], [1.25, 1.1, 1.25, 1.6, 1.4, 1.4]));
add(H3("Alternatives considered"));
add(P("We considered and set aside several alternatives. **Logistic regression and random forests** are common tabular baselines, but gradient-boosted trees capture non-linear feature interactions well and, with the histogram method, train efficiently on 22 million rows, so XGBoost is the single tabular learner. **Graph neural networks** such as GraphSAGE [16] or graph convolutional networks [13] learn structure end to end and are a strong option for future work. We did not use them here because training on 29.3 million transaction nodes needs substantial GPU memory and their predictions are harder to explain to analysts than degree and community features. **Louvain** [6] was the community method in DATA 298A; we replaced it with Leiden because Louvain can return internally disconnected communities [4]. **A prompt-only LLM**, the DATA 298A Claude baseline, needs no training but cannot be adapted to our report schema and sends case data to an external API, so it is kept only as a legacy baseline. **Full fine-tuning** of 14B to 31B models does not fit on a 40 GB GPU, which is why QLoRA [7] is used."));
add(H2("4.4 Model Evaluation Methods"));''')

# PR-AUC citation + equation
rep("PR-AUC is the area under the precision-recall curve as the threshold varies,",
    "PR-AUC is the area under the precision-recall curve as the threshold varies [17], [18],")
rep("add(P(\"**Protocol for LLMs.**", "add(P(\"PR-AUC is computed as average precision over the ranked predictions, where Pₙ and Rₙ are precision and recall at the n-th threshold:\"));\nadd(EQ(\"AP = Σₙ ( Rₙ − Rₙ₋₁ ) Pₙ\", 5));\nadd(P(\"**Protocol for LLMs.**")

# ---------- (c) metric x model matrix ----------
rep('add(H2("4.5 Model Validation and Evaluation Results"));', '''add(P("{tab:mxm} shows which metrics apply to each model."));
add(table(4, "Metrics applied to each model.",
  ["Metric", "M1", "M2", "M3", "M4", "M5 to M8"],
  [
    ["PR-AUC, ROC-AUC", "✓", "", "", "✓", ""],
    ["Precision, recall, F1, confusion matrix", "✓", "", "", "✓", "✓"],
    ["Specificity, accuracy", "", "", "", "", "✓"],
    ["SHAP feature attribution", "✓", "", "", "✓", ""],
    ["Degree and community distributions", "", "✓", "✓", "", ""],
    ["Ablation lift on M4 PR-AUC", "", "✓", "✓", "", ""],
    ["Schema validity, pattern macro-F1", "", "", "", "", "✓"],
    ["Evidence support, unsupported claims", "", "", "", "", "✓"],
    ["Latency, tokens per second", "", "", "", "", "✓"],
  ], [3.2, 0.7, 0.7, 0.7, 0.7, 1.2]));
add(H2("4.5 Model Validation and Evaluation Results"));''')

# ---------- (d) hand-checked sample + shortfalls ----------
rep("// ===================== 5 =====================", '''add(H3("Sample validated by hand"));
add(P("To check the reported results against the data, we traced one validation case end to end. Report GEMMA-001 (case validation_attempt_2249) concerns account 82B75BF30. {tab:sample} compares each claim in the report with the knowledge graph, the feature store and the IBM ground-truth patterns file."));
add(table(4, "Hand check of report GEMMA-001 against the source data.",
  ["Report content", "Source checked", "Result"],
  [
    ["1 transaction between 2 accounts; amount 25,496.68; ACH; 2022-09-16 12:38", "Neo4j: 82B75BF30 to 82AD58260, ACH, 25,496.68 Canadian dollars, 2022-09-16 12:38, is_laundering = 1", "Correct (the report omits the currency)"],
    ["Account 82AD58260 has total degree 66 (17 outgoing, 49 incoming)", "Graph API and feature store: out-degree 17, in-degree 49, total 66", "Correct"],
    ["The accounts belong to community group 4", "Feature store: both accounts in Leiden community 4 (49,770 accounts, community fraud rate 0.07%)", "Correct"],
    ["Decision SUSPICIOUS, risk HIGH, confidence 0.87", "IBM patterns file: transaction belongs to laundering attempt 2,249", "Correct decision"],
    ["Pattern RANDOM", "IBM patterns file: attempt 2,249 is a BIPARTITE pattern", "Incorrect pattern"],
  ], [2.6, 3.6, 1.8]));
add(P("Every factual claim in the report is supported by the source data and the decision is correct, but the pattern label is wrong. This matches the aggregate results: decisions are reliable, while typology recognition is the weakest part of the investigator."));
add(H3("Where the system still falls short"));
add(
  B("**Pattern recognition.** Pattern macro-F1 is 0.13 to 0.23 for all four investigators, as the sample above shows."),
  B("**False alarms on legitimate cases.** Even with guardrails, Gemma flagged 2 of 5 legitimate validation cases, and the unseen validation set has only 10 cases."),
  B("**Low precision at the operating threshold.** At τ = 0.30 the graph-enhanced model has precision 0.011, about 90 alerts per true laundering transaction. Threshold tuning and alert ranking are needed before analyst use."),
  B("**Transductive graph features.** Degree and community structure use the full graph, including validation and test edges, so detection results may be optimistic. A time-respecting graph build is planned."),
  B("**Offline investigation.** The dashboard replays saved Gemma reports and does not run new inference. Traces exist for all four models, but only the selected model's reports are exported to the dashboard."),
  B("**Integration gaps.** The investigator view links to saved reports by case ID, so opening it from an arbitrary transaction does not yet show a matching report, and the dashboard's transaction snapshot does not yet include XGBoost scores."),
  B("**Local graph only.** The knowledge graph and graph API run on a local Docker host; the cloud deployment in Section 5.2.2 is a design."),
);

// ===================== 5 =====================''')

# ---------- (e) requirements + capabilities table ----------
pat = re.compile(r'add\(H3\("5\.1\.2 Data analytics and machine learning capabilities"\)\);\nadd\(\n(?:  N\([^\n]*"cap"\),\n)+\);')
if len(pat.findall(s)) != 1:
    sys.exit("capabilities block not found")
s = pat.sub(lambda m: '''add(H3("5.1.2 Functional and non-functional requirements"));
add(P("{tab:req} lists the requirements the system is designed to meet."));
add(table(5, "Functional (FR) and non-functional (NFR) requirements.",
  ["ID", "Requirement"],
  [
    ["FR1", "The system scores every transaction with a laundering probability and flags those at or above a configurable threshold τ."],
    ["FR2", "Each score comes with a SHAP explanation of the features that drove it."],
    ["FR3", "For any account, the system returns its graph neighbourhood, in-, out- and total degree, centrality and Leiden community."],
    ["FR4", "For a flagged case, the investigator returns a JSON report with decision, pattern, risk level, confidence, evidence and recommended actions."],
    ["FR5", "Every report passes deterministic guardrails, and every change they make is recorded as a flag."],
    ["FR6", "Analysts can search and filter transactions and reports and open a report in the investigator view."],
    ["FR7", "Engineers can rerun ingestion, graph loading, feature extraction, training and evaluation; the Airflow DAG validates the artifacts."],
    ["FR8", "The demo pipeline reloads a separate QA graph without touching the full knowledge graph."],
    ["NFR1", "Imbalance-aware evaluation: detection models are selected on PR-AUC and recall, never on accuracy."],
    ["NFR2", "Leakage control: chronological splits, community fraud rates from training labels only, and labels and provenance withheld from the LLM."],
    ["NFR3", "Traceability: report evidence must be checkable against the graph, and final metrics are frozen in a versioned manifest."],
    ["NFR4", "Data safety: the graph API is read-only, and the QA reset refuses to run on any graph with more than one million nodes."],
    ["NFR5", "Performance: neighbourhood queries return in about 0.1 s and batch scoring takes about 0.1 µs per transaction on the workstation."],
    ["NFR6", "Privacy and security: open-weight models can be hosted inside the institution's own boundary, and credentials live in environment files that are never committed."],
    ["NFR7", "Reproducibility: services run from Docker images and pinned requirement files; LLM experiments run one model at a time from versioned notebooks."],
    ["NFR8", "Responsible use: the investigator assists analysts and does not make filing decisions; every report is reviewed by a person."],
  ], [0.8, 7.2]));
add(H3("5.1.3 Data analytics and machine learning capabilities"));
add(P("{tab:cap} lists the high-level data analytics and machine learning capabilities, with the input, output and technique of each."));
add(table(5, "Data analytics and machine learning capabilities.",
  ["Capability", "Input", "Output", "Technique"],
  [
    ["Transaction risk scoring", "23-feature vector per transaction", "Probability, flag at τ, SHAP values", "XGBoost (M4), TreeSHAP"],
    ["Graph intelligence", "Account ID", "Neighbourhood edges, degree, centrality", "Neo4j Cypher (M2)"],
    ["Community intelligence", "Account-to-account graph", "Community ID, size, fraud rate", "Leiden (M3)"],
    ["Evidence retrieval", "Flagged case", "Transaction, graph and community evidence", "Deterministic lookups"],
    ["Investigation reports", "Case context and evidence", "Structured JSON report", "Fine-tuned Gemma-4 31B (M8) with guardrails"],
    ["Model comparison", "Saved predictions and traces", "Detection and LLM metrics", "PR-AUC, F1, specificity, evidence support"],
    ["Orchestration and monitoring", "Pipeline artifacts", "Validation results, run logs, metrics", "Airflow DAGs"],
  ], [1.8, 1.9, 2.2, 2.1]));''', s)

# ---------- (f) cloud design ----------
rep('add(H3("5.2.3 Data management solution and database"));', '''add(P("{fig:cloud} shows the planned cloud deployment. It keeps the same three tiers and moves each component to a managed equivalent: static hosting with a CDN for the React dashboard; a TLS load balancer with single sign-on in front of container services for the graph API and Streamlit; managed Airflow; a managed Neo4j database behind a private endpoint; object storage for the data lake and artifacts; and a GPU inference service for the selected investigator. Secrets management, least-privilege roles, encryption, central logging and budget alerts apply to every component. Because every component already runs from a Docker image, the move should need configuration changes rather than code changes. This design has not been deployed yet."));
add(figure(FIG("wb_cloud_design.png"), 5, "Planned cloud deployment design. Components keep their current roles and only the hosting changes. Not yet deployed."));
add(H3("5.2.3 Data management solution and database"));''')

# ---------- (g) API listing ----------
rep('''add(P("The Anthropic Messages API was used only by the DATA 298A prompt-only investigator, which is a legacy baseline. It is not part of the DATA 298B system."));''',
    '''add(P("The Anthropic Messages API was used only by the DATA 298A prompt-only investigator, which is a legacy baseline. It is not part of the DATA 298B system."));
add(P("A request to the graph API and the real response from the running system (abridged to two edges):"));
add(code([
  "GET /api/graph/810520F70?limit=2        ->  HTTP 200, 0.11 s",
  "{",
  "  \\"account\\": {\\"account_id\\": \\"810520F70\\", \\"bank_id\\": 1818,",
  "              \\"out_degree\\": 0, \\"in_degree\\": 58, \\"total_degree\\": 58,",
  "              \\"degree_centrality\\": 3.298e-05},",
  "  \\"edges\\": [",
  "    {\\"src_acct\\": \\"80DCF17B0\\", \\"dst_acct\\": \\"810520F70\\", \\"txn_id\\": \\"TXN_24925981\\",",
  "     \\"amount\\": 27549.65, \\"currency\\": \\"US Dollar\\", \\"payment_format\\": \\"Cheque\\",",
  "     \\"timestamp\\": \\"2022-09-15T05:08:00\\", \\"is_laundering\\": 0},",
  "    ...",
  "  ]",
  "}",
]));
add(P(" "));
add(P("**Error behaviour.** An unknown account returns HTTP 404 with {\\"error\\": \\"account_not_found\\"} and the requested ID; an unknown path returns 404 {\\"error\\": \\"not_found\\"}; and if Neo4j is unreachable the API returns 503 with {\\"status\\": \\"offline\\"} and the error type. The dashboard then falls back to a cached graph sample and labels the view as cached. The API issues read queries only."));''')

# ---------- (h) references ----------
refs = [
 'E. Altman, J. Blanuša, L. von Niederhäusern, B. Egressy, A. Anghel and K. Atasu, "Realistic synthetic financial transactions for anti-money laundering models," in Advances in Neural Information Processing Systems 36, Datasets and Benchmarks Track, 2023.',
 'T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in Proc. 22nd ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining, 2016, pp. 785-794.',
 'Neo4j, Inc., "Neo4j Operations Manual, version 5." [Online]. Available: https://neo4j.com/docs/operations-manual/5/ (accessed Oct. 5, 2026).',
 'V. A. Traag, L. Waltman and N. J. van Eck, "From Louvain to Leiden: Guaranteeing well-connected communities," Scientific Reports, vol. 9, art. 5233, 2019.',
 'J. Reichardt and S. Bornholdt, "Statistical mechanics of community detection," Physical Review E, vol. 74, no. 1, art. 016110, 2006.',
 'V. D. Blondel, J.-L. Guillaume, R. Lambiotte and E. Lefebvre, "Fast unfolding of communities in large networks," Journal of Statistical Mechanics: Theory and Experiment, art. P10008, 2008.',
 'T. Dettmers, A. Pagnoni, A. Holtzman and L. Zettlemoyer, "QLoRA: Efficient finetuning of quantized LLMs," in Advances in Neural Information Processing Systems 36, 2023.',
 'E. J. Hu et al., "LoRA: Low-rank adaptation of large language models," in Proc. Int. Conf. on Learning Representations (ICLR), 2022.',
 'Mistral AI, "Ministral-3-14B-Instruct-2512-BF16," model card, Hugging Face. [Online]. Available: https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512-BF16 (accessed Oct. 5, 2026).',
 'M. Abdin et al., "Phi-4 technical report," arXiv:2412.08905, 2024.',
 'Qwen Team, "Qwen3.5-27B," model card, Hugging Face. [Online]. Available: https://huggingface.co/Qwen/Qwen3.5-27B (accessed Oct. 5, 2026).',
 'Google, "Gemma 4 31B," model card, Hugging Face. [Online]. Available: https://huggingface.co/google/gemma-4-31B (accessed Oct. 5, 2026).',
 'M. Weber et al., "Anti-money laundering in Bitcoin: Experimenting with graph convolutional networks for financial forensics," KDD Workshop on Anomaly Detection in Finance, 2019, arXiv:1908.02591.',
 'P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," in Advances in Neural Information Processing Systems 33, 2020.',
 'S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in Advances in Neural Information Processing Systems 30, 2017.',
 'W. L. Hamilton, R. Ying and J. Leskovec, "Inductive representation learning on large graphs," in Advances in Neural Information Processing Systems 30, 2017.',
 'T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," PLoS ONE, vol. 10, no. 3, art. e0118432, 2015.',
 'J. Davis and M. Goadrich, "The relationship between precision-recall and ROC curves," in Proc. 23rd Int. Conf. on Machine Learning (ICML), 2006, pp. 233-240.',
]
ref_js = "add(H1(\"References\"));\n" + "\n".join(
    "add(new Paragraph({ spacing: { after: 80, line: 252 }, indent: { left: 540, hanging: 540 }, children: [new TextRun({ text: " +
    repr(f"[{i}]\t{r}").replace("'", '"') if '"' not in r else
    "add(new Paragraph({ spacing: { after: 80, line: 252 }, indent: { left: 540, hanging: 540 }, children: [new TextRun({ text: " + "`" + f"[{i}]\t{r}".replace("`", "") + "`"
    for i, r in enumerate(refs, 1))
ref_js = "\n".join(line + ", size: 20 })] }));" for line in ref_js.split("\n")[1:])
ref_js = "add(H1(\"References\"));\n" + ref_js
rep("// ---------- document ----------", ref_js + "\n\n// ---------- document ----------")

open(DST, "w").write(s)
print("v2 script written", len(s))
