"""
recommender/rebuild_text_embeddings.py
-------------------------------------
Rebuilds and verifies the text embedding matrix from catalog entries.
"""

from __future__ import annotations
import json
import argparse
from pathlib import Path
import numpy as np
import hashlib

from recommender.vector_utils import (
    TEXT_EMB_PATH,
    ensure_dirs,
    batch_generate_text_embeddings,
)

CATALOG_PATH = Path(__file__).resolve().parents[1] / "catalog" / "catalog.json"

def load_catalog():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def checksum(path: Path) -> str:
    """Compute SHA256 checksum of a binary file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def verify_embeddings(path: Path):
    data = np.load(path)
    print(f"✅ Verified shape={data.shape}, dtype={data.dtype}")
    print(f"Checksum: {checksum(path)[:16]}...")
    norms = np.linalg.norm(data, axis=1)
    mean_norm = float(norms.mean())
    print(f"Mean norm ≈ {mean_norm:.4f} (should be ~1.0)")
    assert np.isclose(mean_norm, 1.0, atol=1e-2), "Embeddings not normalized"

def rebuild_text_embeddings(verify: bool = False):
    ensure_dirs()
    catalog = load_catalog()
    print(f"🧠 Rebuilding embeddings for {len(catalog)} catalog items...")

    texts = [
        f"{item.get('name','')} {item.get('description','')}".strip()
        for item in catalog
    ]
    emb = batch_generate_text_embeddings(texts, use_cache=False, batch_size=512)
    np.save(TEXT_EMB_PATH, emb.astype(np.float32, copy=False))
    print(f"✅ Saved {emb.shape} embeddings → {TEXT_EMB_PATH}")

    if verify:
        verify_embeddings(TEXT_EMB_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild catalog text embeddings")
    parser.add_argument("--verify", action="store_true", help="Run post-build verification")
    args = parser.parse_args()
    rebuild_text_embeddings(verify=args.verify)
