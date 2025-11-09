"""
catalog/loader.py
-----------------
Loads the product catalog (JSON) and exposes helper functions
for retrieving items by ID, text query, or vector index.
"""

import json
from pathlib import Path

CATALOG_PATH = Path(__file__).parent / "catalog.json"

def load_catalog():
    """Return the parsed catalog as a list of dicts."""
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def get_item_by_id(item_id: str):
    """Fetch an item from the catalog by its ID."""
    catalog = load_catalog()
    return next((item for item in catalog if str(item.get("id")) == str(item_id)), None)

def list_all_items(limit: int | None = None):
    """Return all items, optionally limited to N."""
    items = load_catalog()
    return items if limit is None else items[:limit]
