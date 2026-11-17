from __future__ import annotations

import os
from pathlib import Path

import yaml


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path or os.environ.get("KEDS_CONFIG", "configs/model.yaml")).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.parent.parent
    for section, names in {
        "model": ("weights_path",),
        "knowledge": ("database_path", "index_dir"),
        "checkpoints": ("phi_m", "phi_a"),
        "gallery": ("root",),
    }.items():
        for name in names:
            value = Path(config[section][name])
            config[section][name] = str(value if value.is_absolute() else root / value)
    if config["model"]["embedding_dim"] != 768:
        raise ValueError("This deployment requires CLIP ViT-L/14 with 768-dimensional features")
    if config["model"]["backbone"] != "openai/clip-vit-large-patch14":
        raise ValueError("Changing CLIP backbone requires a separate compatible artifact family")
    if config["knowledge"]["retrieval"] != "independent":
        raise ValueError("author_bkp_v1 uses independent image/text neighbor retrieval")
    if config["bkp"]["input_normalization"] not in ("l2", "raw"):
        raise ValueError("BKP input_normalization must be explicitly l2 or raw")
    if not 0 <= config["inference"]["alpha"] <= 1:
        raise ValueError("alpha must be in [0,1]")
    template = config["inference"]["prompt_template"]
    if template.count("*") != 1 or template.count("{modification}") != 1:
        raise ValueError("Prompt requires one placeholder and one modification field")
    config["config_path"] = str(path)
    return config

