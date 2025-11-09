"""
vision/clip_index.py
--------------------
Final optimized CLIP image embedding builder.
Implements:
 - Incremental rebuilds with hash skipping
 - Atomic writes (.tmp + replace)
 - Batched GPU encoding via DataLoader
 - Meta logging (image_index_meta.json)
"""

import os
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from catalog.loader import load_catalog

import torch
from torch.utils.data import Dataset, DataLoader

try:
    import open_clip
except ImportError:
    raise ImportError("Please install open_clip_torch: pip install open_clip_torch")

# -------------------------------------------------------
# Paths & Constants
# -------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
EMB_PATH = ROOT / "embeddings" / "image_embeddings.npy"
META_PATH = ROOT / "embeddings" / "image_index_meta.json"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"


# -------------------------------------------------------
# 1. Utility Functions
# -------------------------------------------------------
def hash_file(path: str) -> str:
    """Compute md5 hash of file for incremental rebuilds."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_clip_model(device=None):
    """Load CLIP model + preprocess transforms."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, preprocess, tokenizer, device


# -------------------------------------------------------
# 2. Dataset & DataLoader
# -------------------------------------------------------
class CatalogImageDataset(Dataset):
    """Dataset wrapping catalog images for batched CLIP encoding."""
    def __init__(self, items, preprocess):
        self.items = items
        self.preprocess = preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        path = item.get("image")
        try:
            img = Image.open(path).convert("RGB")
            return self.preprocess(img), path, item
        except Exception as e:
            return None, path, {"error": str(e)}


# -------------------------------------------------------
# 3. Main Builder Function
# -------------------------------------------------------
def build_image_index(save_path=EMB_PATH, force=False, batch_size=16, num_workers=4):
    """
    Generate or update image embeddings atomically.
    Supports incremental rebuilds and logs metadata.
    """
    start = time.time()
    catalog = load_catalog()
    model, preprocess, _, device = load_clip_model()

    # Load existing meta (hashes)
    existing_meta = {}
    if META_PATH.exists() and not force:
        with open(META_PATH, "r") as f:
            existing_meta = json.load(f)
    known_hashes = existing_meta.get("hashes", {})

    # Filter unprocessed / changed items
    new_items, skipped = [], 0
    for item in catalog:
        img_path = item.get("image")
        if not img_path or not os.path.exists(img_path):
            continue
        file_hash = hash_file(img_path)
        if not force and known_hashes.get(img_path) == file_hash:
            skipped += 1
            continue
        item["_hash"] = file_hash
        new_items.append(item)

    if not new_items:
        print(f"[INFO] Nothing new to encode. Skipped {skipped}.")
        return None, []

    # Dataset + loader
    ds = CatalogImageDataset(new_items, preprocess)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=lambda b: [x for x in b if x[0] is not None],
    )

    embeddings, valid_items = [], []
    errors = []

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        for batch in tqdm(loader, desc="Encoding images"):
            if not batch:
                continue
            imgs, paths, items = zip(*batch)
            imgs = torch.stack(imgs).to(device)
            try:
                vecs = model.encode_image(imgs)
                vecs /= vecs.norm(dim=-1, keepdim=True)
                embeddings.append(vecs.cpu())
                valid_items.extend(items)
                for p, it in zip(paths, items):
                    known_hashes[p] = it["_hash"]
            except Exception as e:
                errors.append({"batch": paths, "error": str(e)})

    # Combine + save atomically
    stacked = torch.cat(embeddings).numpy().astype(np.float32)
    tmp_path = save_path.with_suffix(".tmp")
    np.save(tmp_path, stacked, allow_pickle=False)
    os.replace(tmp_path, save_path)

    duration = time.time() - start
    meta = {
        "generated": datetime.now().isoformat(),
        "count": int(len(stacked)),
        "valid": int(len(valid_items)),
        "skipped": int(skipped),
        "duration_sec": round(duration, 2),
        "throughput_img_s": round(len(valid_items) / max(duration, 1), 3),
        "device": device,
        "errors": errors,
        "hashes": known_hashes,
    }

    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Saved {len(stacked)} embeddings → {save_path}")
    print(f"[OK] Coverage: {len(valid_items)}/{len(catalog)} | Skipped: {skipped}")
    print(f"[OK] Duration: {meta['duration_sec']}s | Throughput: {meta['throughput_img_s']} img/s")

    return stacked, valid_items


# -------------------------------------------------------
# 4. Integrity Checker
# -------------------------------------------------------
def verify_index(expected_count=None, path=EMB_PATH):
    """Verify embedding file integrity and shape."""
    if not path.exists():
        print("[ERROR] Embedding file not found.")
        return False
    try:
        arr = np.load(path)
        print(f"[INFO] Loaded {arr.shape[0]} embeddings of dim {arr.shape[1]}")
        if expected_count and arr.shape[0] != expected_count:
            print(f"[WARN] Count mismatch: {arr.shape[0]} vs expected {expected_count}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to read embeddings: {e}")
        return False


# -------------------------------------------------------
# 5. Script Entry Point
# -------------------------------------------------------
if __name__ == "__main__":
    print("🔧 Building optimized CLIP image index...")
    embs, valid = build_image_index()
    verify_index(expected_count=len(valid))
