"""
recommender/recommend.py
------------------------
Generates ranked product recommendations from user text queries
using semantic embeddings and vectorized similarity search.
"""

from __future__ import annotations
import numpy as np
from catalog.loader import load_catalog
from recommender.vector_utils import (
    ensure_dirs,
    generate_text_embedding,
    TEXT_EMB_PATH,
)

# ---------------------------------------------------------------------------
# Core Recommender
# ---------------------------------------------------------------------------

def recommend_products(query: str, top_k: int = 5):
    """
    Recommend products by semantic similarity to the query.
    Uses precomputed catalog embeddings (text_embeddings.npy).
    """
    ensure_dirs()
    catalog = load_catalog()
    embeddings = np.load(TEXT_EMB_PATH).astype(np.float32, copy=False)
    qvec = generate_text_embedding(query)

    # Fast vectorized similarity (assumes normalized vectors)
    sims = embeddings @ qvec
    top_idxs = np.argsort(sims)[-top_k:][::-1]

    results = []
    for i in top_idxs:
        if i < len(catalog):
            item = catalog[i]
            results.append({
                "name": item.get("name"),
                "description": item.get("description"),
                "price": item.get("price"),
                "image": item.get("image", None),
                "score": round(float(sims[i]), 4),
            })
    return results
