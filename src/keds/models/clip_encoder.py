from __future__ import annotations

import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F

from .composer import compose


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(device)


def rgb_image(image):
    if image is None:
        raise ValueError("Reference image is required")
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).convert("RGB")
    with Image.open(image) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


class CLIPEncoder(nn.Module):
    def __init__(self, backbone="openai/clip-vit-large-patch14", device="auto", revision="main",
                 local_files_only=False, weights_path=None, **_):
        super().__init__()
        from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
        self.device = resolve_device(device)
        options = {"revision": revision, "local_files_only": local_files_only}
        source = weights_path or backbone
        self.clip = CLIPModel.from_pretrained(source, low_cpu_mem_usage=True, **options).to(self.device).eval()
        self.processor = CLIPImageProcessor.from_pretrained(source, **options)
        self.tokenizer = CLIPTokenizer.from_pretrained(source, **options)
        self.clip.requires_grad_(False)
        if self.clip.config.projection_dim != 768 or self.clip.config.text_config.hidden_size != 768:
            raise ValueError("This BKP family requires CLIP text/projection width 768")
        marker = self.tokenizer.encode("*", add_special_tokens=False)
        if len(marker) != 1:
            raise ValueError("CLIP tokenizer must encode '*' as one token")
        self.placeholder_id = marker[0]
        self.backbone = backbone
        self.revision = getattr(self.clip.config, "_commit_hash", None) or revision

    @torch.no_grad()
    def encode_images(self, images, normalize=True):
        if not images:
            raise ValueError("Image batch cannot be empty")
        values = self.processor(images=[rgb_image(x) for x in images], return_tensors="pt")
        output = self.clip.get_image_features(pixel_values=values["pixel_values"].to(self.device)).float()
        return F.normalize(output, dim=-1) if normalize else output

    @torch.no_grad()
    def encode_texts(self, texts):
        if not texts or any(not x.strip() for x in texts):
            raise ValueError("Text must be non-empty")
        values = self.tokenizer(texts, padding="max_length", max_length=77,
                                truncation=False, return_tensors="pt").to(self.device)
        if values["input_ids"].shape[1] != 77:
            raise ValueError("Text exceeds CLIP context")
        output = self.clip.get_text_features(input_ids=values["input_ids"]).float()
        return F.normalize(output, dim=-1)

    def compose(self, prompts, pseudo):
        # Do not use no_grad here: training must backpropagate into BKP pseudo words.
        return compose(self.clip, self.tokenizer, prompts, pseudo, self.placeholder_id)

