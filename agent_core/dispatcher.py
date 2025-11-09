"""
agent_core/dispatcher.py
------------------------
Dispatches user input to general, recommender, or vision pipelines.
"""

import time
import yaml
import numpy as np
from pathlib import Path
from prompt_builders import *
from recommender.recommend import recommend_products
from vision.image_match import match_image
from embeddings.text_embed import embed_text
from vector_utils import cosine_sim


# ===== CONFIG =====
INTENT_PATH = Path(__file__).resolve().parents[1] / "config" / "intent.yaml"
USE_DEEP_ROUTING = False
CONFIDENCE_THRESHOLD = 0.6


# ===== LOAD INTENT DICTIONARIES =====
with open(INTENT_PATH, "r", encoding="utf-8") as f:
    INTENT_KEYWORDS = yaml.safe_load(f)

def fuzzy_in(text, keywords):
    """Simple typo-tolerant match (edit distance ≤1)."""
    from Levenshtein import distance as lev
    for kw in keywords:
        if kw in text:
            return True
        if any(lev(w, kw) <= 1 for w in text.split()):
            return True
    return False


# ===== CLASSIFIERS =====
def classify_with_embedding(text):
    """Embedding-based intent routing with fallback."""
    global USE_DEEP_ROUTING

    text_vec = embed_text(text)
    intents = list(INTENT_KEYWORDS.keys())
    proto_vecs = np.stack([embed_text(i) for i in intents])
    sims = cosine_sim(text_vec, proto_vecs)[0]

    max_idx = int(np.argmax(sims))
    confidence = float(np.max(sims))
    intent = intents[max_idx]

    # Dynamically disable deep routing if confidence low or short query
    if confidence < CONFIDENCE_THRESHOLD or len(text.split()) < 3:
        USE_DEEP_ROUTING = False
        return classify_simple(text)
    USE_DEEP_ROUTING = True
    return intent


def classify_simple(text):
    text = text.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if fuzzy_in(text, keywords):
            return intent
    return "recommend"


def classify_intent(text):
    return classify_with_embedding(text) if USE_DEEP_ROUTING else classify_simple(text)


# ===== ROUTING =====
def route_input(user_input):
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
        prompt = {"error": "Unrecognized intent"}

    latency = round(time.time() - start, 4)
    log_perf("dispatcher", intent, latency)
    return prompt


# ===== HELPERS =====
def match_image_with_clip(user_input, mode="style"):
    """
    Dispatcher-level wrapper for image-based queries.
    Expects user_input to contain 'image' key with a NumPy vector or encoded image.
    """
    if isinstance(user_input, dict) and "image" in user_input:
        image_vec = np.array(user_input["image"], dtype=np.float32)
    else:
        raise ValueError("Invalid input for vision mode — missing 'image' key.")

    return match_image(image_vec, mode=mode, k=5)


def log_perf(component, intent, latency):
    """Micro-benchmark logger."""
    LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "dispatcher_perf.json"
    LOG_PATH.parent.mkdir(exist_ok=True)
    entry = {"component": component, "intent": intent, "latency": latency, "ts": time.time()}
    try:
        import json
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass
