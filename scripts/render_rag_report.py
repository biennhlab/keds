import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from keds.evaluation.statistics import compare_experiments
from keds.utils.io import read_json

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Render real evaluation/profile results; never fill missing accuracy metrics")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="reports/rag_comparison.md")
    args = p.parse_args()
    report = read_json(args.input)
    if report.get("is_benchmark") is False:
        lines = ["# Functional smoke only — không phải benchmark accuracy", "",
                 "Các latency dưới đây đo trên ảnh thật, theo thứ tự profile, chỉ vài samples; không chứng minh speedup.",
                 "Không có ground truth/Recall/MRR cho gallery demo.", "",
                 "| Profile | Runs | P50 ms | P95 ms | R@10 / R@50 / MRR |", "|---|---:|---:|---:|---|"]
        for name, result in report["profiles"].items():
            latency = result["latency_ms"]
            lines.append(f"| {name} | {len(result['runs'])} | {latency['p50']:.2f} | {latency['p95']:.2f} | chưa chạy |")
    else:
        experiments = report["experiments"]
        baseline = next(e for e in experiments if e["rag_profile"] == "E0" and e["mode"] == "keds")
        lines = [f"# {report['dataset']} · {report['protocol']} · {report['phase']}", "",
                 "| Profile | Queries | R@10 | R@50 | MRR | ΔR@10 95% CI | P50/P95 ms |", "|---|---:|---:|---:|---:|---|---|"]
        for experiment in experiments:
            categories = experiment["categories"]
            metrics = experiment.get("macro_metrics", next(iter(categories.values()))["metrics"])
            ci = compare_experiments(baseline, experiment, seed=report["seed"])
            diagnostics = [q for c in categories.values() for q in c["diagnostics"]]
            latencies = [q["latency_ms"] for q in diagnostics]
            low, high = ci["R@10"]["ci95"]
            lines.append(f"| {experiment['rag_profile']} | {len(diagnostics)} | {metrics['R@10']:.3f} | "
                         f"{metrics['R@50']:.3f} | {metrics['MRR']:.5f} | [{low:.3f}, {high:.3f}] | "
                         f"{np.percentile(latencies,50):.2f}/{np.percentile(latencies,95):.2f} |")
        if not report.get("historical_tuning_overlap_verified"):
            lines.extend(["", "Historical tuning overlap chưa được audit; không gọi đây là xác nhận trên holdout chưa từng dùng."])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)

