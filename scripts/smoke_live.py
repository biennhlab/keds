"""Exercise mounted Gradio → HTTP FastAPI → real KEDs → actual image responses."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx
from gradio_client import Client, handle_file

from keds.utils.io import write_json

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--image", default="data/demo/dog.jpg")
    p.add_argument("--gallery", default="demo")
    p.add_argument("--output", default="reports/live_smoke.json")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--rag-profile", default="E0")
    args = p.parse_args()
    report = {}
    with httpx.Client(base_url=args.url, timeout=300, trust_env=False, follow_redirects=True) as client:
        for route in ("/health", "/ready", "/docs", "/ui/", "/openapi.json"):
            response = client.get(route)
            response.raise_for_status()
            report[route] = {"status": response.status_code}
    ui = Client(args.url + "/ui/", httpx_kwargs={"trust_env": False}, verbose=False)
    ui.predict(api_name="/refresh")  # Same gallery population event as browser page load.
    start = time.perf_counter()
    photos, details = ui.predict(handle_file(args.image), "with two dogs instead of one", args.gallery,
                                3, "keds", 0.5, 16, args.rag_profile, api_name="/run_search")
    if len(photos) != 3 or len(details["results"]) != 3:
        raise ValueError("UI did not return three actual results")
    report["ui_search"] = {"result_count": len(photos), "latency_ms": (time.perf_counter()-start)*1000,
                           "model_version": details["model_version"], "gallery_id": details["gallery_id"],
                           "knowledge_version": details["knowledge_version"],
                           "rag_profile": details["details"]["rag_profile"],
                           "result_ids": [r["image_id"] for r in details["results"]]}
    if args.compare:
        baseline, ked_photos, compared = ui.predict(handle_file(args.image), "with two dogs instead of one",
            args.gallery, 3, 0.5, 16, args.rag_profile, api_name="/compare")
        if len(baseline) != 3 or len(ked_photos) != 3:
            raise ValueError("Comparison did not return actual photos for both modes")
        report["ui_comparison"] = {"baseline_mode": "CLIP_fusion", "baseline_count": len(baseline),
                                   "keds_count": len(ked_photos), "model_version": compared["model_version"]}
    write_json(args.output, report)
    print(report["ui_search"])

