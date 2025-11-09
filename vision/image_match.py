"""
vision/image_match.py
---------------------
Find visually similar products using CLIP embeddings.
Modes:
- exact : identical product/image
- style : semantic / outfit / shape / vibe
- color : dominant color similarity
"""

import time
import json
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from catalog.loader import load_catalog
from .clip_index import load_image_embeddings

try:
    import faiss
    USE_FAISS = True
except ImportError:
    USE_FAISS = False


EMB_PATH = Path(__file__).resolve().parents[1] / "embeddings"
LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "vision_perf.json"
FAISS_INDEX_PATH = EMB_PATH / "image_index.faiss"

_cache = {"embeddings": None, "catalog": None}


def _normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _format_result(item, score, mode):
    return {
        "id": item.get("id"),
        "name": item.get("name"),
        "description": item.get("description"),
        "price": item.get("price"),
        "score": round(float(score), 4),
        "match_mode": mode
    }


def _log_perf(latency_ms, k, mode):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "latency_ms": round(latency_ms, 2),
        "k": k,
        "mode": mode
    }
    if LOG_PATH.exists():
        data = json.loads(LOG_PATH.read_text())
    else:
        data = []
    data.append(record)
    LOG_PATH.write_text(json.dumps(data, indent=2))


def _load_cached_embeddings():
    if _cache["embeddings"] is None or _cache["catalog"] is None:
        _cache["catalog"] = load_catalog()
        _cache["embeddings"] = np.array(load_image_embeddings(), dtype=np.float32)
    return _cache["catalog"], _cache["embeddings"]


def _validate_alignment(catalog, embeddings):
    if len(catalog) != len(embeddings):
        raise ValueError(
            f"Catalog/embedding mismatch: {len(catalog)} vs {len(embeddings)}"
        )


def _match_with_numpy(query_vec, embeddings, catalog, mode, k):
    query_vec = _normalize(query_vec)
    embeddings = np.array([_normalize(e) for e in embeddings])

    if mode == "color":
        sims = cosine_similarity([query_vec[:3]], embeddings[:, :3])[0]
    else:
        sims = cosine_similarity([query_vec], embeddings)[0]

    top_idxs = np.argsort(sims)[::-1][:k]
    return [_format_result(catalog[i], sims[i], mode) for i in top_idxs]


def _build_faiss_index(embeddings):
    """Create or load persistent FAISS index."""
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    return index


def _load_faiss_index(embeddings):
    if FAISS_INDEX_PATH.exists():
        return faiss.read_index(str(FAISS_INDEX_PATH))
    else:
        return _build_faiss_index(embeddings)


def _match_with_faiss(query_vec, embeddings, catalog, mode, k):
    index = _load_faiss_index(embeddings)
    q = query_vec.reshape(1, -1).astype("float32")
    faiss.normalize_L2(q)
    sims, idxs = index.search(q, k)
    sims, idxs = sims[0], idxs[0]
    return [_format_result(catalog[i], sims[j], mode) for j, i in enumerate(idxs)]


def match_image(image_vector: np.ndarray, mode: str = "style", k: int = 5):
    """
    Main entrypoint — unified retrieval pipeline.
    Supports 'exact', 'style', and 'color' modes.
    """
    t0 = time.perf_counter()
    catalog, embeddings = _load_cached_embeddings()
    _validate_alignment(catalog, embeddings)

    if USE_FAISS:
        results = _match_with_faiss(image_vector, embeddings, catalog, mode, k)
    else:
        results = _match_with_numpy(image_vector, embeddings, catalog, mode, k)

    latency = (time.perf_counter() - t0) * 1000
    _log_perf(latency, k, mode)
    return results
