from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from keds.training.checkpointing import save_checkpoint
from keds.training.resume import resume
from keds.utils.io import write_json


class Trainer:
    """Reusable trainer with an explicit objective and validation callback.

    Keeping the loss callback explicit avoids silently substituting a different
    Phi-A recipe when the original notebook/preprocessing is unavailable.
    """
    def __init__(self, model, optimizer, phase, output_dir, meta, scheduler=None):
        if phase not in ("phi_m", "phi_a") or not meta.get("objective_id"):
            raise ValueError("An explicit phase and objective_id are required")
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.phase, self.meta = phase, meta
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=False)
        self.next_epoch, self.global_step, self.losses = 0, 0, []
        self.best = -float("inf")
        self.resume_report = {"resume_mode": "new_run"}

    def restore(self, checkpoint_path, exact=False):
        self.resume_report = resume(self.model, self.optimizer, checkpoint_path, self.phase,
                                    self.scheduler, exact, self.meta["objective_id"])
        self.next_epoch, self.global_step = self.resume_report["next_epoch"], self.resume_report["global_step"]
        self.losses = self.resume_report["losses"]
        self.best = self.resume_report["source"]["meta"].get("best_validation_metric", -float("inf"))
        write_json(self.output / "resume_report.json", self.resume_report)

    def fit(self, loader, loss_fn, until_epoch=30, validate=None):
        if until_epoch <= self.next_epoch:
            raise ValueError("until_epoch must exceed completed epoch count")
        best = self.best
        for epoch in range(self.next_epoch, until_epoch):
            self.model.train()
            total, count = 0.0, 0
            for batch in loader:
                self.optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self.model, batch)
                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise ValueError("Training loss must be a finite scalar")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                total += float(loss.detach())
                count += 1
                self.global_step += 1
            if not count:
                raise ValueError("Training dataloader produced no batches")
            row = {"epoch": epoch + 1, "loss": total / count, "steps": count}
            self.losses.append(row)
            promote = False
            if validate:
                self.model.eval()
                metric = float(validate(self.model))
                if not math.isfinite(metric):
                    raise ValueError("Validation callback returned a non-finite metric")
                row["validation_metric"] = metric
                if metric > best:
                    best = metric
                    promote = True
            metadata = {**self.meta, "architecture": self.model.architecture,
                        "resume_mode": self.resume_report["resume_mode"]}
            if math.isfinite(best):
                metadata["best_validation_metric"] = best
            save_checkpoint(self.output / "LAST.pt", self.model, self.optimizer, epoch,
                            self.global_step, self.phase, metadata, self.scheduler, self.losses)
            if promote:
                save_checkpoint(self.output / "BEST.pt", self.model, self.optimizer, epoch,
                                self.global_step, self.phase, metadata, self.scheduler, self.losses)
            with (self.output / "training.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
        self.best = best
        return self.losses

