"""
models/text_encoder.py
----------------------
Wrapper around a local text embedding model.
Default: sentence-transformers/all-MiniLM-L6-v2 (≈90 MB).
Swap freely for smaller or quantized variants.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from functools import lru_cache

@lru_cache(maxsize=1)
def _load_model():
    # Cached load for performance
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_text(text: str) -> np.ndarray:
    """Return a normalized embedding vector for the given text."""
    model = _load_model()
    vec = np.array(model.encode(text, normalize_embeddings=True))
    return vec
