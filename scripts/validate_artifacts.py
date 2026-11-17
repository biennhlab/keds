from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from keds.knowledge.database import KnowledgeDatabase
from keds.knowledge.retriever import KnowledgeRetriever
from keds.models.bkp import build_bkp
from keds.models.checkpoint_loader import load_stream, read_checkpoint
from keds.utils.config import load_config
from keds.utils.io import read_json, sha256, write_json


def validate(config_path, build_indexes=False):
    config = load_config(config_path)
    report = {"config": config, "checks": {}, "valid": True}

    def check(name, work):
        try:
            report["checks"][name] = {"valid": True, "details": work()}
        except Exception as e:
            report["checks"][name] = {"valid": False, "error": f"{type(e).__name__}: {e}"}
            report["valid"] = False

    for phase in ("phi_m", "phi_a"):
        check(phase, lambda phase=phase: load_stream(build_bkp(config["bkp"]),
                                                    config["checkpoints"][phase], phase))
        path = Path(config["checkpoints"][phase]).parent / f"{phase}_last.pt"
        if path.exists():
            def resume_check(path=path, phase=phase):
                stream = build_bkp(config["bkp"])
                details = load_stream(stream, path, phase)
                payload = read_checkpoint(path)
                if not payload.get("optimizer"):
                    raise ValueError("Optimizer state absent; seamless resume unavailable")
                opt = torch.optim.AdamW(stream.parameters())
                opt.load_state_dict(payload["optimizer"])
                for param, state in opt.state.items():
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key not in state or state[key].shape != param.shape or not torch.isfinite(state[key]).all():
                            raise ValueError(f"Invalid optimizer tensor {key}")
                details["optimizer_compatible"] = True
                details["scheduler_present"] = "scheduler" in payload
                details["completed_epochs_if_zero_based"] = int(payload["epoch"]) + 1
                return details
            check(phase + "_resume", resume_check)

    def knowledge_check():
        db = KnowledgeDatabase(config["knowledge"]["database_path"],
                               expected_count=config["knowledge"]["expected_count"])
        details = db.validate()
        retriever = KnowledgeRetriever(db, config["knowledge"]["index_dir"], build=build_indexes)
        details["index_order_verified"] = retriever.validate_order()
        return details

    check("knowledge", knowledge_check)
    root = Path(config["config_path"]).parent.parent
    manifest = root / "reports/artifact_download_manifest.json"
    if manifest.exists():
        for record in read_json(manifest):
            if record.get("local_path") and record.get("sha256"):
                def checksum_check(record=record):
                    actual = sha256(root / record["local_path"])
                    if actual != record["sha256"]:
                        raise ValueError("Downloaded artifact differs from pinned manifest")
                    return {"sha256": actual, "revision": record["revision"]}
                check("checksum_" + record["file"], checksum_check)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strict artifact, optimizer and FAISS ordering validator")
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--build-indexes", action="store_true")
    parser.add_argument("--output", default="reports/artifact_validation_report.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = validate(args.config, args.build_indexes)
    write_json(args.output, report)
    for name, result in report["checks"].items():
        print(name, "PASS" if result["valid"] else result["error"], flush=True)
    sys.exit(0 if report["valid"] else 1)

