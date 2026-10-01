"""Compatibility entry point for the lakehouse-to-Neo4j QA sync."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("fraud_lakehouse_neo4j_sync.py")), run_name="__main__")
