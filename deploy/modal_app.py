"""Modal GPU web deployment. Run commands from the repository root.

modal run deploy/modal_app.py --demo
modal deploy deploy/modal_app.py
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("keds-cir")
volume = modal.Volume.from_name("keds-artifacts", create_if_missing=True)
hf_secret = modal.Secret.from_name("keds-hf", required_keys=["HF_TOKEN"])
STORE = "/vol/keds"
CONFIG = "/app/configs/modal_runtime.yaml"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgomp1")
    .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install_from_requirements(str(ROOT / "deploy/requirements.txt"))
    .env({"PYTHONPATH": "/app/src:/app", "HF_HOME": "/vol/keds/hf-cache",
          "HF_HUB_DISABLE_XET": "1", "GRADIO_ANALYTICS_ENABLED": "False",
          "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"})
    .workdir("/app")
)
for directory in ("src", "app", "configs", "scripts"):
    # Whitelist project directories: never upload .env, .git, data or artifacts.
    image = image.add_local_dir(str(ROOT / directory), f"/app/{directory}", copy=True,
                                ignore=["**/__pycache__/**", "*.pyc"])


@app.function(image=image, volumes={"/vol": volume}, secrets=[hf_secret],
              cpu=4, memory=16384, timeout=7200, max_containers=1)
def prepare():
    """CPU-only ingestion; no GPU charge for downloads or FAISS construction."""
    import torch
    from scripts.prepare_modal_artifacts import prepare as prepare_artifacts
    torch.set_num_threads(2)
    report = prepare_artifacts(STORE)
    volume.commit()
    return {"knowledge_count": report["knowledge"]["count"],
            "index_order_verified": report["index_order_verified"]}


@app.function(image=image, volumes={"/vol": volume}, gpu="L4",
              cpu=4, memory=16384, timeout=7200, max_containers=1)
def index_gallery(gallery_id: str, demo: bool = False):
    """Encode only target gallery photos; CC3M embeddings are reused."""
    import subprocess
    import sys
    import torch
    from keds.gallery.repository import Gallery, safe_id
    from scripts.prepare_modal_artifacts import write_config

    torch.set_num_threads(2)
    safe_id(gallery_id)
    if demo and gallery_id != "demo":
        raise ValueError("The public photo demo must use gallery_id=demo")
    output = Path(STORE) / "gallery" / gallery_id
    if (output / "CURRENT.json").exists():
        return Gallery(output).validate()
    source = Path(STORE) / "uploads" / gallery_id
    if demo:
        subprocess.run([sys.executable, "scripts/download_demo_images.py", "--output", str(source)], check=True)
    if not source.is_dir():
        raise ValueError(f"Upload real gallery images to {source} first")
    write_config(STORE, CONFIG)
    subprocess.run([sys.executable, "scripts/build_gallery.py", "--config", CONFIG,
                    "--input", str(source), "--output", str(output), "--batch-size", "8"], check=True)
    report = Gallery(output).validate()
    volume.commit()
    return report


@app.function(image=image, volumes={"/vol": volume}, gpu="L4",
              cpu=4, memory=16384, timeout=14400, max_containers=1)
def index_fashioniq(category: str = "all"):
    """Build full official val galleries from an uploaded Fashion-IQ dataset."""
    import torch
    from scripts.import_fashioniq import import_fashioniq
    from scripts.prepare_modal_artifacts import write_config

    torch.set_num_threads(2)
    write_config(STORE, CONFIG)
    report = import_fashioniq(CONFIG, Path(STORE) / "datasets/fashioniq", category=category)
    volume.commit()
    return report


@app.function(image=image, volumes={"/vol": volume}, gpu="L4", cpu=4, memory=16384,
              timeout=1800, max_containers=1)
def audit_fashioniq(full_benchmark: bool = False):
    import subprocess
    import sys

    import torch

    from scripts.prepare_modal_artifacts import write_config
    from scripts.verify_demo import run

    torch.set_num_threads(2)
    write_config(STORE, CONFIG)
    data_root = Path(STORE) / "datasets/fashioniq"
    if full_benchmark:
        report = Path(STORE) / "reports/demo_fashioniq_full.json"
        subprocess.run([sys.executable, "scripts/evaluate.py", "--config", CONFIG,
                        "--dataset", "fashioniq", "--data-root", str(data_root),
                        "--gallery-root", str(Path(STORE) / "gallery"), "--batch-size", "16",
                        "--output", str(report)], check=True)
    else:
        report = Path(STORE) / "reports/demo_full_gallery_sample.json"
        run(CONFIG, data_root, report, 32, 6, 4, full_gallery=True)
    volume.commit()
    return str(report)


@app.function(image=image, volumes={"/vol": volume}, gpu=["L4", "T4"], cpu=4, memory=16384,
              timeout=3600, scaledown_window=120, max_containers=1)
@modal.concurrent(max_inputs=16)
@modal.web_server(8000, startup_timeout=600)
def web():
    import subprocess
    import sys
    import torch
    from scripts.prepare_modal_artifacts import write_config

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the Modal image and GPU allocation")
    print(f"KEDs web starting on {torch.cuda.get_device_name(0)}", flush=True)
    if not (Path(STORE) / "manifests/prepared.json").is_file():
        raise RuntimeError("Run modal run deploy/modal_app.py --demo before deploying")
    write_config(STORE, CONFIG)
    # Gradio callbacks call the REST server on localhost within this container.
    # The external listener must bind 0.0.0.0 for Modal routing.
    subprocess.Popen([sys.executable, "scripts/run_demo.py", "--config", CONFIG,
                      "--host", "0.0.0.0", "--port", "8000", "--public-scheme", "https",
                      "--examples-report", str(Path(STORE) / "reports/demo_full_gallery_sample.json")])


@app.local_entrypoint()
def main(demo: bool = False, gallery: str = "", fashioniq: bool = False, category: str = "all",
         fashioniq_audit: bool = False, fashioniq_benchmark: bool = False):
    if sum([bool(demo), bool(gallery), bool(fashioniq), fashioniq_audit, fashioniq_benchmark]) > 1:
        raise ValueError("Choose exactly one gallery or evaluation operation")
    if fashioniq_audit or fashioniq_benchmark:
        print(audit_fashioniq.remote(full_benchmark=fashioniq_benchmark))
        return
    if fashioniq:
        print(index_fashioniq.remote(category))
        print("Imported. Redeploy the web app and refresh galleries to select dress/shirt/toptee.")
        return
    print(prepare.remote())
    if demo or gallery:
        print(index_gallery.remote("demo" if demo else gallery, demo))
    print("Prepared. Deploy with: modal deploy deploy/modal_app.py ; open the URL + /ui/")
