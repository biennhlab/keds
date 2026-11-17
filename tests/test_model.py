import ast
import os
from pathlib import Path

import pytest
import torch
from torch import nn

from keds.models.bkp import BKP
from keds.models.checkpoint_loader import IncompatibleCheckpoint, extract_state, load_stream
from keds.models.clip_encoder import rgb_image
from keds.models.composer import inject_tokens


def test_bkp_shapes_gradient_and_eval_determinism():
    model = BKP(dim=8, middle_dim=4, heads=2, dim_head=4, cross_layers=3).eval()
    image = torch.randn(2, 8)
    neighbors = torch.randn(2, 3, 8)
    result = model(image, neighbors, neighbors)
    assert result.shape == (2, 3, 8)
    assert torch.equal(result, model(image, neighbors, neighbors))
    result.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_pseudo_insertion_eot_shift_and_gradient():
    embedding = nn.Embedding(12, 4).requires_grad_(False)
    ids = torch.tensor([[10, 2, 5, 7, 11, 0, 0], [10, 5, 3, 11, 0, 0, 0]])
    pseudo = torch.randn(2, 3, 4, requires_grad=True)
    values, ends = inject_tokens(embedding, ids, 5, 11, pseudo)
    assert ends.tolist() == [6, 5]
    torch.testing.assert_close(values[0, 2:5], pseudo[0])
    torch.testing.assert_close(values[1, 1:4], pseudo[1])
    torch.testing.assert_close(values[0, 6], embedding(ids)[0, 4])
    values.sum().backward()
    assert torch.equal(pseudo.grad, torch.ones_like(pseudo))
    with pytest.raises(ValueError, match="context"):
        inject_tokens(embedding, torch.tensor([[10, 2, 5, 7, 11]]), 5, 11, pseudo[:1])


def test_unknown_and_corrupt_checkpoint_rejected():
    with pytest.raises(IncompatibleCheckpoint):
        extract_state({"state_dict": {"x": torch.tensor(float("nan"))}})
    with pytest.raises(IncompatibleCheckpoint):
        extract_state({"psi_l.weight": torch.zeros(2, 2)})


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_rgb_conversion(mode):
    from PIL import Image
    assert rgb_image(Image.new(mode, (4, 5))).mode == "RGB"
    with pytest.raises(ValueError):
        rgb_image(None)


def test_official_module_numerical_parity():
    path = Path("references/official-keds/src/model/model.py")
    if not path.exists():
        pytest.skip("Official checkout not present")
    from einops import rearrange
    # Import only the three pure module definitions, avoiding official module's
    # unrelated data/CUDA side effects. No altered architecture definitions.
    tree = ast.parse(path.read_text())
    definitions = [n for n in tree.body if isinstance(n, ast.ClassDef) and
                   n.name in {"CrossAttention", "CrossFormer", "IM2TEXT"}]
    scope = {"nn": nn, "torch": torch, "rearrange": rearrange}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), scope)
    ours = BKP(dim=8, middle_dim=4, heads=2, dim_head=4).eval()
    mapper = scope["IM2TEXT"](8, 4, 8).eval()
    visual = scope["CrossFormer"](8, 8, 8, num_layers=3, heads=2, dim_head=4).eval()
    textual = scope["CrossFormer"](8, 8, 8, num_layers=3, heads=2, dim_head=4).eval()
    mapper.load_state_dict(ours.img2text.state_dict(), strict=True)
    visual.load_state_dict(ours.retrieval_fuse.state_dict(), strict=True)
    textual.load_state_dict(ours.text_condition.state_dict(), strict=True)
    image, i, t = torch.randn(2, 8), torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    q, ki, kt = mapper(image).unsqueeze(1), mapper(i), mapper(t)
    expected = torch.cat([visual(q, ki, ki), textual(q, kt, kt), q], 1)
    torch.testing.assert_close(ours(image, i, t), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.artifacts
def test_real_v21_best_strict_loading():
    if not os.environ.get("KEDS_TEST_ARTIFACTS"):
        pytest.skip("Set KEDS_TEST_ARTIFACTS=1 for downloaded real artifacts")
    for phase in ("phi_m", "phi_a"):
        report = load_stream(BKP(), f"artifacts/checkpoints/{phase}_best_author_modules.pt", phase)
        assert len(report["sha256"]) == 64

