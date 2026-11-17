from torch import nn

from .bkp import build_bkp
from .checkpoint_loader import IncompatibleCheckpoint, load_stream


class DualStream(nn.Module):
    def __init__(self, config, checkpoint_paths, device):
        super().__init__()
        self.phi_m = build_bkp(config)
        self.phi_a = build_bkp(config)
        self.metadata = {
            "phi_m": load_stream(self.phi_m, checkpoint_paths["phi_m"], "phi_m"),
            "phi_a": load_stream(self.phi_a, checkpoint_paths["phi_a"], "phi_a"),
        }
        if self.metadata["phi_m"]["sha256"] == self.metadata["phi_a"]["sha256"]:
            raise IncompatibleCheckpoint("Two streams cannot use the same checkpoint")
        a, b = (self.metadata[s]["meta"] for s in ("phi_m", "phi_a"))
        for field in ("model", "mode", "source_db", "middle_dim", "n_layer", "cross_layers",
                      "feature_normalization", "prompt_template"):
            if field in a and field in b and a[field] != b[field]:
                raise IncompatibleCheckpoint(f"Stream metadata differs: {field}")
        self.to(device).eval().requires_grad_(False)

    def forward(self, image, image_neighbors, text_neighbors):
        return (self.phi_m(image, image_neighbors, text_neighbors),
                self.phi_a(image, image_neighbors, text_neighbors))

