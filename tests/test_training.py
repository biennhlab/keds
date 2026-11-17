import pytest
import torch

from keds.models.bkp import BKP
from keds.models.checkpoint_loader import IncompatibleCheckpoint
from keds.training.checkpointing import save_checkpoint
from keds.training.losses import symmetric_infonce
from keds.training.resume import resume
from keds.training.trainer import Trainer


def test_checkpoint_optimizer_rng_resume_reproduces_next_step(tmp_path):
    model = BKP(dim=8, middle_dim=4, heads=2, dim_head=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    image, neighbors = torch.randn(2, 8), torch.randn(2, 3, 8)

    def step(m, opt):
        opt.zero_grad()
        loss = m(image, neighbors, neighbors).square().mean()
        loss.backward()
        opt.step()
        return loss.detach()

    step(model, optimizer)
    checkpoint = tmp_path / "last.pt"
    save_checkpoint(checkpoint, model, optimizer, 8, 17, "phi_m", {"objective_id": "unit_numerical"})
    expected = step(model, optimizer)
    clone = BKP(dim=8, middle_dim=4, heads=2, dim_head=4)
    clone_opt = torch.optim.AdamW(clone.parameters(), lr=0.7)
    report = resume(clone, clone_opt, checkpoint, "phi_m", exact=True, objective_id="unit_numerical")
    actual = step(clone, clone_opt)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for a, b in zip(model.parameters(), clone.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert report["next_epoch"] == 9 and report["global_step"] == 17
    with pytest.raises(IncompatibleCheckpoint, match="Exact resume"):
        resume(clone, clone_opt, checkpoint, "phi_m", exact=True, objective_id="another")


def test_infonce_requires_real_negatives():
    with pytest.raises(ValueError):
        symmetric_infonce(torch.ones(1, 8), torch.ones(1, 8))


def test_trainer_preserves_prior_validation_best_when_continuing(tmp_path):
    model = BKP(dim=8, middle_dim=4, heads=2, dim_head=4)
    optimizer = torch.optim.AdamW(model.parameters())
    trainer = Trainer(model, optimizer, "phi_m", tmp_path / "first", {"objective_id": "unit_numerical"})
    image, neighbors = torch.randn(2, 8), torch.randn(2, 3, 8)

    def loss_fn(stream, batch):
        return stream(image, neighbors, neighbors).square().mean()

    # Numerical policy fixtures; these are not dataset evaluation results.
    values = iter([0.7, 0.6])
    trainer.fit([None], loss_fn, until_epoch=2, validate=lambda _: next(values))
    first_best = torch.load(tmp_path / "first/BEST.pt", weights_only=True)
    last = torch.load(tmp_path / "first/LAST.pt", weights_only=True)
    assert first_best["epoch"] == 0 and last["epoch"] == 1
    assert last["meta"]["best_validation_metric"] == 0.7
    resumed = Trainer(model, optimizer, "phi_m", tmp_path / "continued", {"objective_id": "unit_numerical"})
    resumed.restore(tmp_path / "first/LAST.pt", exact=True)
    resumed.fit([None], loss_fn, until_epoch=3, validate=lambda _: 0.65)
    assert not (tmp_path / "continued/BEST.pt").exists()
    assert (tmp_path / "first/BEST.pt").is_file()

