"""
catalog/validate_catalog.py
---------------------------
Ensures catalog.json is well-formed and schema-consistent.
"""

import json
from pathlib import Path

CATALOG_PATH = Path(__file__).parent / "catalog.json"

def validate_catalog():
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_fields = ["id", "name", "description", "price", "image", "tags", "features"]
    for idx, item in enumerate(data):
        for field in required_fields:
            if field not in item:
                raise ValueError(f"Missing field '{field}' in item {idx}")
        if not isinstance(item["features"], dict):
            raise TypeError(f"features field must be a dict in item {item['id']}")
        if not isinstance(item["tags"], list):
            raise TypeError(f"tags field must be a list in item {item['id']}")
    print(f"✅ Catalog validated successfully: {len(data)} items")

if __name__ == "__main__":
    validate_catalog()
