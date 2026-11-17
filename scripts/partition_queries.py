"""Create a fixed stratified development/nominal-holdout partition using IDs only.

This cannot certify that the original checkpoint selection never saw holdout:
that requires the original v21 development manifest, which is not exported.
"""
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from keds.evaluation.datasets import fashioniq
from keds.utils.io import write_json

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--seed", type=int, default=999)
    p.add_argument("--development-fraction", type=float, default=0.2)
    p.add_argument("--output", default="configs/query_partitions.json")
    args = p.parse_args()
    if not 0 < args.development_fraction < 1:
        p.error("Development fraction must be between 0 and 1")
    assignments, sources = {}, {}
    for category in ("dress", "shirt", "toptee"):
        queries, _, provenance = fashioniq(args.data_root, category)
        ordered = sorted((q.query_id for q in queries), key=lambda q:
                         hashlib.sha256(f"{args.seed}:{q}".encode()).hexdigest())
        cut = round(len(ordered) * args.development_fraction)
        assignments.update({q: "development" if i < cut else "holdout" for i, q in enumerate(ordered)})
        sources[category] = provenance
    write_json(args.output, {"seed": args.seed, "assignments": assignments, "provenance": sources,
        "historical_tuning_overlap_verified": False,
        "note": "Nominal holdout only until original notebook/v21 development IDs are audited"})

