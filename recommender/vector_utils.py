"""
recommender/vector_utils.py
---------------------------
Atomic math utilities for vector operations, similarity search, and embedding caching.
Used by dispatcher (cosine_sim), recommender, and vision pipelines.
"""

import numpy as np
from pathlib import Path
import pickle
import tempfile

# === GLOBAL CACHE ===
_cache = {}
CACHE_PATH = Path(__file__).resolve().parents[1] / "embeddings" / "cache.pkl"


# === CORE MATH ===
def normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector (safe for zero-length)."""
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a single vector `a` and matrix `b`.
    Returns shape (1, len(b)).
    """
    a = normalize(a).reshape(1, -1)
    b = np.asarray(b, dtype=np.float32)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.dot(a, b_norm.T)


def top_k_similar(query_vec: np.ndarray, embeddings: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Return indices of top-k most similar items using dot-product (approx cosine).
    Safe for empty embeddings or small catalogs.
    """
    sims = embeddings @ query_vec
    n = embeddings.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    k = int(min(k, n))
    idx = np.argpartition(sims, -k)[-k:]
    return idx[np.argsort(sims[idx])[::-1]]


# === CACHING + I/O ===
def _atomic_write_bytes(dst: Path, data: bytes):
    """Write to disk atomically."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, dir=dst.parent)
    with open(tmp.name, "wb") as f:
        f.write(data)
    Path(tmp.name).replace(dst)


def generate_text_embedding(text: str) -> np.ndarray:
    """
    Generate or fetch cached text embedding (mock until real model available).
    Deterministic hash-based vector for reproducibility.
    """
    global _cache
    if text in _cache:
        return _cache[text]
    if CACHE_PATH.exists():
        try:
            _cache.update(pickle.loads(CACHE_PATH.read_bytes()))
        except Exception:
            _cache = {}

    np.random.seed(abs(hash(text)) % (2**32))
    vec = np.random.rand(512).astype(np.float32)
    vec = normalize(vec)
    _cache[text] = vec

    # Write to disk (atomic)
    try:
        _atomic_write_bytes(CACHE_PATH, pickle.dumps(_cache))
    except Exception:
        pass

    return vec


def batch_generate_text_embeddings(texts, use_cache=True) -> np.ndarray:
    """Generate embeddings for a list of texts, with optional caching."""
    mat = np.stack([generate_text_embedding(t) for t in texts])
    return np.array([normalize(v) for v in mat], dtype=np.float32)
