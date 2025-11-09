"""
models/vision_encoder.py
------------------------
Wrapper for CLIP or open_clip vision encoder.
Default: open_clip ViT-B/32 — moderate speed/accuracy balance.
"""

import torch
import open_clip
from PIL import Image
from torchvision import transforms
import numpy as np
from functools import lru_cache

@lru_cache(maxsize=1)
def _load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model.eval(), preprocess, tokenizer

def embed_image(image: Image.Image) -> np.ndarray:
    """Return normalized image embedding vector."""
    model, preprocess, _ = _load_model()
    with torch.no_grad():
        image_tensor = preprocess(image).unsqueeze(0)
        image_features = model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features[0].cpu().numpy()
