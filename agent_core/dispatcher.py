"""
agent_core/dispatcher.py
------------------------
Dispatches user input to general, recommender, or vision pipelines.
"""

import time
import yaml
import numpy as np
from pathlib import Path

from agent_core.prompt_builders import *
from recommender.recommend import recommend_products
from vision.image_match import match_image
from embeddings.text_embed import embed_text
from recommender.vector_utils import cosine_similarity as cosine_sim


# ────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────
INTENT_PATH = Path(__file__).resolve().parents[1] / "config" / "intent.yaml"
USE_DEEP_ROUTING = False
CONFIDENCE_THRESHOLD = 0.6


# ────────────────────────────────────────────────────────────────
# LOAD INTENT DICTIONARIES
# ────────────────────────────────────────────────────────────────
if INTENT_PATH.exists():
    with open(INTENT_PATH, "r", encoding="utf-8") as f:
        INTENT_KEYWORDS = yaml.safe_load(f)
else:
    INTENT_KEYWORDS = {
        "general": ["hi", "hello", "help", "name", "what can you do"],
        "recommend": ["recommend", "find", "suggest", "buy", "looking for"],
        "vision": ["image", "photo", "upload", "picture", "similar to this"],
    }


# ────────────────────────────────────────────────────────────────
# FUZZY MATCHING
# ────────────────────────────────────────────────────────────────
def fuzzy_in(text: str, keywords: list[str]) -> bool:
    """
    Simple typo-tolerant match using Levenshtein distance ≤ 1.
    """
    from Levenshtein import distance as lev
    text = text.lower()
    for kw in keywords:
        if kw in text:
            return True
        if any(lev(w, kw) <= 1 for w in text.split()):
            return True
    return False


# ────────────────────────────────────────────────────────────────
# INTENT CLASSIFIERS
# ────────────────────────────────────────────────────────────────
def classify_with_embedding(text: str) -> str:
    """
    Embedding-based routing with confidence threshold and fallback.
    """
    global USE_DEEP_ROUTING

    text_vec = embed_text(text)
    intents = list(INTENT_KEYWORDS.keys())
    proto_vecs = np.stack([embed_text(i) for i in intents])
    sims = cosine_sim(text_vec, proto_vecs)[0]

    max_idx = int(np.argmax(sims))
    confidence = float(np.max(sims))
    intent = intents[max_idx]

    # Dynamically disable deep routing if signal weak
    if confidence < CONFIDENCE_THRESHOLD or len(text.split()) < 3:
        USE_DEEP_ROUTING = False
        return classify_simple(text)

    USE_DEEP_ROUTING = True
    return intent


def classify_simple(text: str) -> str:
    """
    Keyword- and fuzzy-based classification fallback.
    """
    text = text.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if fuzzy_in(text, keywords):
            return intent
    return "recommend"  # default fallback


def classify_intent(text: str) -> str:
    return classify_with_embedding(text) if USE_DEEP_ROUTING else classify_simple(text)


# ────────────────────────────────────────────────────────────────
# MAIN ROUTER
# ────────────────────────────────────────────────────────────────
def route_input(user_input):
    """
    High-level router selecting correct downstream module and returning prompt dict.
    """
    start = time.time()
    intent = classify_intent(user_input)

    if intent == "general":
        prompt = build_general_prompt(user_input)

    elif intent == "recommend":
        matches = recommend_products(user_input, top_k=5)
        prompt = build_recommend_prompt(user_input, matches)

    elif intent == "vision":
        matches = match_image_with_clip(user_input)
        prompt = build_vision_prompt(matches)

    else:
        prompt = {"error": f"Unrecognized intent: {intent}"}

    latency = round(time.time() - start, 4)
    log_perf("dispatcher", intent, latency)
    return prompt


# ────────────────────────────────────────────────────────────────
# IMAGE WRAPPER
# ────────────────────────────────────────────────────────────────
def match_image_with_clip(user_input, mode: str = "style"):
    """
    Dispatcher-level wrapper for image-based queries.
    Expects user_input to contain 'image' key with a NumPy vector or encoded image.
    """
    if isinstance(user_input, dict) and "image" in user_input:
        image_vec = np.array(user_input["image"], dtype=np.float32)
    else:
        raise ValueError("Invalid input for vision mode — missing 'image' key.")

    return match_image(image_vec, mode=mode, k=5)


# ────────────────────────────────────────────────────────────────
# PERFORMANCE LOGGER
# ────────────────────────────────────────────────────────────────
def log_perf(component: str, intent: str, latency: float):
    """
    Lightweight micro-benchmark logger.
    Appends latency entries to /logs/dispatcher_perf.json
    """
    LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "dispatcher_perf.json"
    LOG_PATH.parent.mkdir(exist_ok=True)
    entry = {"component": component, "intent": intent, "latency": latency, "ts": time.time()}

    try:
        import json
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[dispatcher] Logging failed: {e}")

from agent_core.logger import log_event
import numpy as np
from io import BytesIO
import base64
from PIL import Image

def handle_query(text=None, image=None):
    """
    High-level interface for API.
    Accepts text (str) or base64-encoded image (str).
    Returns structured prompt or model-ready response.
    """
    log_event("query_received", {"text": text, "has_image": bool(image)})

    # ── TEXT ROUTE ─────────────────────────────────────────────
    if text and not image:
        response = route_input(text)

    # ── IMAGE ROUTE ────────────────────────────────────────────
    elif image:
        try:
            # Decode base64 → image → mock vector (placeholder)
            image_bytes = base64.b64decode(image)
            img = Image.open(BytesIO(image_bytes))
            # Simplified embedding: flatten and normalize pixel mean
            image_vec = np.array(img).mean(axis=(0, 1))
            response = route_input({"image": image_vec.tolist()})
        except Exception as e:
            response = {"error": f"Image decoding failed: {str(e)}"}

    # ── INVALID ROUTE ───────────────────────────────────────────
    else:
        response = {"error": "No input provided."}

    log_event("response_generated", response)
    return response
