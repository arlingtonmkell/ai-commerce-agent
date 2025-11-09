"""
catalog/sync_catalog_embeddings.py
----------------------------------
Rebuilds text and image embeddings whenever catalog.json changes.
"""

from pathlib import Path
import numpy as np
import json
from recommender.vector_utils import cosine_similarity
from models.text_encoder import generate_text_embedding
from models.vision_encoder import generate_image_embedding

CATALOG_PATH = Path(__file__).parent / "catalog.json"
TEXT_OUT = Path(__file__).resolve().parents[1] / "embeddings" / "text_embeddings.npy"
IMAGE_OUT = Path(__file__).resolve().parents[1] / "embeddings" / "image_embeddings.npy"

def rebuild_embeddings():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    text_embeddings = []
    image_embeddings = []

    for item in catalog:
        desc = f"{item['name']} {item['description']} {' '.join(item['tags'])}"
        text_embeddings.append(generate_text_embedding(desc))
        image_embeddings.append(generate_image_embedding(item['image']))

    np.save(TEXT_OUT, np.array(text_embeddings))
    np.save(IMAGE_OUT, np.array(image_embeddings))
    print(f"✅ Saved {len(catalog)} text and image embeddings.")

if __name__ == "__main__":
    rebuild_embeddings()
