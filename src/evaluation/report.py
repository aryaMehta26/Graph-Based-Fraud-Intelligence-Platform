"""Aggregate base/fine-tuned outputs into raw comparison artifacts."""
import argparse, csv, json
from pathlib import Path

def compare(input_dir: Path, output_dir: Path):
    rows=[]
    for path in sorted(input_dir.glob("*.json")):
        if path.name in {"base_vs_finetuned.json"}: continue
        data=json.loads(path.read_text()); rows.append({"artifact":path.stem, **{k:v for k,v in data.items() if isinstance(v,(int,float,str))}})
    output_dir.mkdir(parents=True,exist_ok=True); (output_dir/"model_comparison.json").write_text(json.dumps(rows,indent=2));
    keys=sorted({k for r in rows for k in r});
    with (output_dir/"model_comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
    return rows

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--input-dir",type=Path,default=Path("artifacts/evaluation")); p.add_argument("--output-dir",type=Path,default=Path("artifacts/evaluation")); a=p.parse_args(); print(json.dumps(compare(a.input_dir,a.output_dir),indent=2))
