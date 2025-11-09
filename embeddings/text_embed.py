"""
embeddings/text_embed.py
------------------------
Semantic embedding interface used across dispatcher, recommender, and vector utils.

Core API:
    - embed_text(text: str) -> np.ndarray
    - batch_embed_texts(texts: list[str]) -> np.ndarray

Behavior:
    • Uses local transformer encoder if available
    • Falls back to deterministic hash-based pseudo-embedding
    • Always returns L2-normalized float32 vectors
    • Logs latency and backend source to logs/embed_perf.json
"""

import os
import time
import numpy as np
from pathlib import Path

# Optional: try sentence-transformers or bge
try:
    from sentence_transformers import SentenceTransformer
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False


# ===== PATHS =====
MODEL_NAME = os.getenv("TEXT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "embed_perf.json"

_model = None


# ===== INTERNAL HELPERS =====
def _load_model():
    """Lazy-load transformer encoder once per runtime."""
    global _model
    if not MODEL_AVAILABLE:
        return None
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize and cast to float32."""
    vec = vec.astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _log_perf(source: str, latency_ms: float, n: int):
    """Append latency record to logs/embed_perf.json."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "samples": n,
        "latency_ms": round(latency_ms, 2),
    }
    try:
        import json
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


# ===== MAIN API =====
def embed_text(text: str) -> np.ndarray:
    """
    Generate a single semantic embedding for text.
    Uses model if available, otherwise deterministic hash fallback.
    """
    t0 = time.perf_counter()

    if MODEL_AVAILABLE:
        model = _load_model()
        vec = model.encode(text, normalize_embeddings=True)
        source = "model"
    else:
        np.random.seed(abs(hash(text)) % (2**32))
        vec = np.random.rand(512)
        vec = _normalize(vec)
        source = "fallback"

    _log_perf(source, (time.perf_counter() - t0) * 1000, 1)
    return vec.astype(np.float32)


def batch_embed_texts(texts: list[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts.
    Returns (N, D) float32 matrix of normalized vectors.
    """
    t0 = time.perf_counter()

    if MODEL_AVAILABLE:
        model = _load_model()
        vecs = model.encode(texts, normalize_embeddings=True)
        source = "model"
    else:
        vecs = []
        for t in texts:
            np.random.seed(abs(hash(t)) % (2**32))
            v = np.random.rand(512)
            vecs.append(_normalize(v))
        vecs = np.vstack(vecs)
        source = "fallback"

    _log_perf(source, (time.perf_counter() - t0) * 1000, len(texts))
    return vecs.astype(np.float32)
