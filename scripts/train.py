"""Explicit Phi-M fine-tuning from compatible LAST; never auto-start from epoch 0.

This recipe is identified as image_infonce_v1, not claimed to reproduce the
missing v21 notebook. For original Phi-A continuation use Trainer with the
audited original objective/data callback once the notebook is supplied.
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from keds.inference.engine import KEDsEngine
from keds.training.losses import symmetric_infonce
from keds.training.trainer import Trainer


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/model.yaml")
    p.add_argument("--training-config", default="configs/training.yaml")
    p.add_argument("--resume", required=True)
    p.add_argument("--output", required=True, help="New run directory; existing artifacts are never overwritten")
    p.add_argument("--exact", action="store_true")
    args = p.parse_args()
    c = yaml.safe_load(Path(args.training_config).read_text())
    if c["phase"] != "phi_m" or c["recipe"] != "image_infonce_v1":
        p.error("Original Phi-A recipe is not exported; supply the audited callback to Trainer")
    random.seed(c["seed"])
    np.random.seed(c["seed"])
    torch.manual_seed(c["seed"])
    torch.set_num_threads(2)
    engine = KEDsEngine(args.config)
    model = engine.streams.phi_m.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"])
    trainer = Trainer(model, optimizer, "phi_m", args.output,
                      {"objective_id": c["recipe"], "source_db": engine.database.sha256,
                       "backbone": engine.clip.backbone, "seed": c["seed"],
                       "normalization": "l2", "training_config": c})
    trainer.restore(args.resume, exact=args.exact)
    loader = DataLoader(TensorDataset(torch.arange(engine.database.size)),
                        batch_size=c["batch_size"], shuffle=True, drop_last=True)

    def loss_fn(stream, batch):
        indices = batch[0]
        visual = engine.database.rows("image", indices).to(engine.device)
        neighbors_i, neighbors_t, _ = engine.knowledge.retrieve(visual,
            engine.config["knowledge"]["top_k"], exclude_indices=indices.numpy())
        pseudo = stream(visual, neighbors_i, neighbors_t)
        composed = engine.clip.compose([c["prompt"]] * len(indices), pseudo)
        return symmetric_infonce(visual, composed, c["temperature"])

    trainer.fit(loader, loss_fn, c["until_epoch"])

