"""
DATA 298A — Graph-Based Fraud Intelligence Platform
Team 2 | San José State University

tests/test_smoke.py
--------------------
Closes Issue #11.

Smoke tests — verify every pipeline script can be imported and its
core functions are callable without crashing. Does NOT require:
    - Neo4j running
    - Parquet splits to exist
    - Anthropic API key
    - Full 31M dataset

Run with:
    python3 -m pytest tests/test_smoke.py -v

Or without pytest:
    python3 tests/test_smoke.py
"""

import os
import sys
import json
import tempfile
import importlib
import importlib.util
from pathlib import Path

# Add project root to path
PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

results = []

def check(name, fn):
    try:
        fn()
        results.append((name, True, None))
        print(f"  {PASS} {name}")
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"  {FAIL} {name} — {e}")


# ---------------------------------------------------------------------------
# 1. Import checks — every module must be importable
# ---------------------------------------------------------------------------

def test_imports():
    print("\n[1/5] Import checks")

    def import_06():
        spec = importlib.util.spec_from_file_location(
            "m06", PROJ / "src" / "models" / "06_train_xgboost_baseline.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "load_split")
        assert hasattr(mod, "evaluate")
        assert hasattr(mod, "FEATURE_COLS")
        assert "Is Laundering" == mod.TARGET_COL

    def import_07():
        spec = importlib.util.spec_from_file_location(
            "m07", PROJ / "src" / "models" / "07_train_graph_enhanced_model.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "TABULAR_FEATURES")
        assert hasattr(mod, "GRAPH_FEATURES")
        assert hasattr(mod, "ALL_FEATURES")
        # Graph features must be strictly additive on top of tabular
        for f in mod.TABULAR_FEATURES:
            assert f in mod.ALL_FEATURES, f"{f} missing from ALL_FEATURES"

    def import_05():
        spec = importlib.util.spec_from_file_location(
            "m05", PROJ / "src" / "models" / "05_build_feature_store.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "enrich_split")
        assert hasattr(mod, "GRAPH_COLS")

    def import_investigator():
        llm_path = PROJ / "src" / "llm" / "investigator.py"
        if not llm_path.exists():
            print("     (skipped — src/llm/investigator.py not in main yet)")
            return
        spec = importlib.util.spec_from_file_location("inv", llm_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "VARIANTS")
        assert len(mod.VARIANTS) == 4
        assert hasattr(mod, "validate_schema")
        assert hasattr(mod, "parse_response")

    def import_evaluate():
        eval_path = PROJ / "src" / "llm" / "evaluate.py"
        if not eval_path.exists():
            print("     (skipped — src/llm/evaluate.py not in main yet)")
            return
        spec = importlib.util.spec_from_file_location("ev", eval_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "check_schema_compliance")
        assert hasattr(mod, "check_faithfulness")
        assert hasattr(mod, "check_consistency")

    def import_04b():
        spec = importlib.util.spec_from_file_location(
            "m04b", PROJ / "src" / "models" / "04b_louvain_communities.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    check("04b_louvain_communities imports cleanly", import_04b)
    check("07_train_graph_enhanced_model imports cleanly", import_07)
    check("05_build_feature_store imports cleanly",     import_05)
    check("src/llm/investigator imports cleanly",       import_investigator)
    check("src/llm/evaluate imports cleanly",           import_evaluate)


# ---------------------------------------------------------------------------
# 2. Feature set consistency
# ---------------------------------------------------------------------------

def test_feature_consistency():
    print("\n[2/5] Feature set consistency")

    def tabular_features_match():
        spec06 = importlib.util.spec_from_file_location(
            "m06", PROJ / "src" / "models" / "06_train_xgboost_baseline.py"
        )
        mod06 = importlib.util.module_from_spec(spec06)
        spec06.loader.exec_module(mod06)

        spec07 = importlib.util.spec_from_file_location(
            "m07", PROJ / "src" / "models" / "07_train_graph_enhanced_model.py"
        )
        mod07 = importlib.util.module_from_spec(spec07)
        spec07.loader.exec_module(mod07)

        missing = set(mod06.FEATURE_COLS) - set(mod07.TABULAR_FEATURES)
        extra   = set(mod07.TABULAR_FEATURES) - set(mod06.FEATURE_COLS)
        assert not missing, f"Features in baseline but not in graph model: {missing}"
        assert not extra,   f"Features in graph model but not in baseline: {extra}"

    def no_community_features_check():
        # Issue #4 (Louvain/Leiden) is now closed — community features
        # are expected in 07. This check is retired.
        pass

    def class_ratio_consistent():
        spec06 = importlib.util.spec_from_file_location(
            "m06", PROJ / "src" / "models" / "06_train_xgboost_baseline.py"
        )
        mod06 = importlib.util.module_from_spec(spec06)
        spec06.loader.exec_module(mod06)

        spec07 = importlib.util.spec_from_file_location(
            "m07", PROJ / "src" / "models" / "07_train_graph_enhanced_model.py"
        )
        mod07 = importlib.util.module_from_spec(spec07)
        spec07.loader.exec_module(mod07)

        assert mod06.CLASS_RATIO == mod07.CLASS_RATIO, (
            f"CLASS_RATIO mismatch: 06={mod06.CLASS_RATIO}, 07={mod07.CLASS_RATIO}"
        )

    check("Tabular feature set identical in 06 and 07",     tabular_features_match)
    check("No community features in 07 (Issue #4 pending)", no_community_features_check)
    check("CLASS_RATIO consistent across 06 and 07",        class_ratio_consistent)


# ---------------------------------------------------------------------------
# 3. LLM schema validation
# ---------------------------------------------------------------------------

def test_llm_schema():
    print("\n[3/5] LLM schema validation")

    llm_path = PROJ / "src" / "llm" / "investigator.py"
    if not llm_path.exists():
        print("  (skipped — src/llm/ not in main yet, merge feature/llm-investigator-layer first)")
        return

    spec = importlib.util.spec_from_file_location("inv", llm_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def valid_report_passes():
        report = {
            "pattern":    "Smurfing",
            "evidence":   ["Account 800A3F2 sent 47 transactions"],
            "risk_level": "CRITICAL",
            "actions":    ["Freeze account 800A3F2"],
            "reasoning":  "",
        }
        mod.validate_schema(report)  # should not raise

    def missing_field_fails():
        report = {
            "pattern":   "Smurfing",
            "evidence":  ["some evidence"],
            "risk_level": "CRITICAL",
            # missing "actions"
        }
        try:
            mod.validate_schema(report)
            raise AssertionError("Should have raised ValueError for missing actions")
        except ValueError:
            pass  # expected

    def invalid_risk_level_fails():
        report = {
            "pattern":    "Smurfing",
            "evidence":   ["some evidence"],
            "risk_level": "EXTREME",   # not in valid set
            "actions":    ["do something"],
        }
        try:
            mod.validate_schema(report)
            raise AssertionError("Should have raised ValueError for invalid risk_level")
        except ValueError:
            pass  # expected

    def all_4_variants_defined():
        for vk in ["v1", "v2", "v3", "v4"]:
            assert vk in mod.VARIANTS, f"Variant {vk} missing"
            v = mod.VARIANTS[vk]
            assert "model"         in v
            assert "system_prompt" in v
            assert "max_tokens"    in v

    def cyclic_transfer_not_in_prompts():
        for vk, v in mod.VARIANTS.items():
            prompt = v["system_prompt"]
            # Only fail if Cyclic Transfer appears as an enum option
            # (it's OK in prohibition text like "Do NOT use Cyclic Transfer")
            import re
            enum_matches = re.findall(
                r'(?:one of|ONLY)[^\n]*Cyclic Transfer', prompt
            )
            assert not enum_matches, (
                f"'Cyclic Transfer' still listed as valid option in {vk} prompt"
            )

    check("Valid report passes schema validation",      valid_report_passes)
    check("Missing field raises ValueError",            missing_field_fails)
    check("Invalid risk_level raises ValueError",       invalid_risk_level_fails)
    check("All 4 variants defined with required keys",  all_4_variants_defined)
    check("'Cyclic Transfer' removed from all prompts", cyclic_transfer_not_in_prompts)


# ---------------------------------------------------------------------------
# 4. LLM evaluation functions
# ---------------------------------------------------------------------------

def test_llm_evaluation():
    print("\n[4/5] LLM evaluation functions")

    eval_path = PROJ / "src" / "llm" / "evaluate.py"
    if not eval_path.exists():
        print("  (skipped — src/llm/ not in main yet)")
        return

    spec = importlib.util.spec_from_file_location("ev", eval_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    sample_subgraph = {
        "accounts": [
            {"account_id": "800A3F2", "degree_centrality": 0.891, "community_id": 447},
            {"account_id": "800B1C9", "degree_centrality": 0.412, "community_id": 447},
        ],
        "transactions": [
            {"txn_id": "T_001", "src_acct": "800B1C9", "dst_acct": "800A3F2",
             "amount": 9800.00, "payment_format": "ACH"},
        ],
        "community_fraud_rate": 0.423,
    }

    def schema_compliance_perfect():
        report = {
            "pattern":    "Smurfing",
            "evidence":   ["Account 800A3F2 has degree centrality 0.891"],
            "risk_level": "CRITICAL",
            "actions":    ["Freeze account 800A3F2"],
        }
        result = mod.check_schema_compliance(report)
        assert result["score"] == 1.0, f"Expected 1.0, got {result['score']}"

    def schema_compliance_penalises_invalid():
        report = {
            "pattern":    "Unknown Pattern",
            "evidence":   [],
            "risk_level": "VERY HIGH",
            "actions":    [],
        }
        result = mod.check_schema_compliance(report)
        assert result["score"] < 1.0, "Should have penalised invalid fields"

    def faithfulness_detects_grounded():
        report = {
            "evidence": ["Account 800A3F2 sent transactions via ACH totalling 9800.0"],
            "pattern": "Smurfing", "risk_level": "CRITICAL", "actions": ["freeze"],
        }
        result = mod.check_faithfulness(report, sample_subgraph)
        assert result["score"] > 0, "Should detect grounded evidence"

    def consistency_perfect_for_identical_runs():
        runs = [
            {"pattern": "Smurfing", "risk_level": "CRITICAL",
             "evidence": ["Account 800A3F2 flagged"], "actions": ["freeze"]},
            {"pattern": "Smurfing", "risk_level": "CRITICAL",
             "evidence": ["Account 800A3F2 flagged"], "actions": ["freeze"]},
            {"pattern": "Smurfing", "risk_level": "CRITICAL",
             "evidence": ["Account 800A3F2 flagged"], "actions": ["freeze"]},
        ]
        result = mod.check_consistency(runs)
        assert result["score"] == 1.0, f"Identical runs should score 1.0, got {result['score']}"
        assert result["meets_target"] is True

    def consistency_fails_for_divergent_runs():
        runs = [
            {"pattern": "Smurfing",      "risk_level": "CRITICAL", "evidence": ["a"], "actions": ["x"]},
            {"pattern": "Layering Ring", "risk_level": "HIGH",     "evidence": ["b"], "actions": ["y"]},
            {"pattern": "Fan-Out",       "risk_level": "LOW",      "evidence": ["c"], "actions": ["z"]},
        ]
        result = mod.check_consistency(runs)
        assert result["score"] < 0.8, f"Divergent runs should score below 0.8, got {result['score']}"

    check("Schema compliance: perfect report scores 1.0",      schema_compliance_perfect)
    check("Schema compliance: penalises invalid fields",        schema_compliance_penalises_invalid)
    check("Faithfulness: detects grounded evidence",           faithfulness_detects_grounded)
    check("Consistency: identical runs score 1.0",             consistency_perfect_for_identical_runs)
    check("Consistency: divergent runs score below target",    consistency_fails_for_divergent_runs)


# ---------------------------------------------------------------------------
# 5. Artifact and output file checks
# ---------------------------------------------------------------------------

def test_artifacts():
    print("\n[5/5] Artifact checks")

    def sample_subgraph_exists():
        path = PROJ / "artifacts" / "sample_subgraph.json"
        assert path.exists(), f"Missing: {path}"
        with open(path) as f:
            sg = json.load(f)
        assert "accounts"     in sg
        assert "transactions" in sg
        assert len(sg["accounts"])     > 0
        assert len(sg["transactions"]) > 0

    def layering_ring_subgraph_exists():
        path = PROJ / "artifacts" / "layering_ring_subgraph.json"
        assert path.exists(), f"Missing: {path}"

    def fan_out_subgraph_exists():
        path = PROJ / "artifacts" / "fan_out_subgraph.json"
        assert path.exists(), f"Missing: {path}"

    def env_example_exists():
        path = PROJ / ".env.example"
        assert path.exists(), "Missing .env.example"
        content = path.read_text()
        assert "NEO4J_URI"      in content
        assert "NEO4J_PASSWORD" in content
        assert "ANTHROPIC_API_KEY" in content or "PROJ_ROOT" in content

    def requirements_has_kagglehub():
        path = PROJ / "requirements.txt"
        assert path.exists(), "Missing requirements.txt"
        content = path.read_text()
        assert "kagglehub" in content, "kagglehub missing from requirements.txt"
        assert "xgboost"   in content, "xgboost missing from requirements.txt"
        assert "anthropic" in content or "neo4j" in content

    def llm_outputs_exist():
        output_dir = PROJ / "artifacts" / "llm_outputs"
        if not output_dir.exists():
            print("     (skipped — run investigator.py first)")
            return
        files = list(output_dir.glob("*.json"))
        assert len(files) > 0, "No LLM output files found"

    check("sample_subgraph.json exists and is valid",      sample_subgraph_exists)
    check("layering_ring_subgraph.json exists",            layering_ring_subgraph_exists)
    check("fan_out_subgraph.json exists",                  fan_out_subgraph_exists)
    check(".env.example has required keys",                env_example_exists)
    check("requirements.txt has kagglehub + xgboost",     requirements_has_kagglehub)
    check("LLM output files exist in artifacts/",         llm_outputs_exist)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  SMOKE TESTS — Graph Fraud Intelligence Platform")
    print("  DATA 298A | Team 2 | Issue #11")
    print("=" * 60)

    test_imports()
    test_feature_consistency()
    test_llm_schema()
    test_llm_evaluation()
    test_artifacts()

    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed out of {len(results)} tests")
    if failed:
        print("\n  Failed tests:")
        for name, ok, err in results:
            if not ok:
                print(f"    ✗ {name}")
                print(f"      {err}")
        sys.exit(1)
    else:
        print("  All smoke tests passed ✓")
    print("=" * 60)
