import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keds.evaluation.statistics import compare_experiments
from keds.utils.io import read_json, write_json

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Paired, category-stratified bootstrap CI; requires real evaluation reports")
    p.add_argument("--baseline", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--repetitions", type=int, default=2000)
    p.add_argument("--output", default="reports/paired_comparison.json")
    args = p.parse_args()
    baseline, variant = read_json(args.baseline), read_json(args.variant)
    for key in ("dataset", "split", "seed", "phase", "partitions_sha256", "model"):
        # Cache runtime counters may differ, but checkpoint and backbone identity must not.
        a, b = baseline.get(key), variant.get(key)
        if key == "model":
            a = (a["model_version"], a["clip_revision"], a["knowledge_version"])
            b = (b["model_version"], b["clip_revision"], b["knowledge_version"])
        if a != b:
            raise ValueError(f"Mismatched evaluation provenance: {key}")
    result = compare_experiments(baseline["experiments"][0], variant["experiments"][0],
                                 baseline["seed"], args.repetitions)
    write_json(args.output, {"paired_bootstrap_95ci": result, "repetitions": args.repetitions,
               "note": "A positive point estimate whose CI includes zero is not a confirmed improvement"})
    print(result)

