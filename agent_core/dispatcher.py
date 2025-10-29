# agent_core/dispatcher.py

USE_DEEP_ROUTING = False

# TODO
def classify_with_embedding(text):
    return "recommend"

def classify_simple(text):
    text = text.lower()

    # Keywords for vision search
    vision_keywords = [
        "image", "photo", "upload", "picture", "this pic", "look like this",
        "see this", "does this match", "find this", "what is this", "similar to this"
    ]

    # Keywords for recommendation
    recommend_keywords = [
        "recommend", "suggest", "find", "looking for", "can i get",
        "want", "buy", "need", "shop for", "get me", "help me choose", "show me options"
    ]

    # Keywords for general small talk
    general_keywords = [
        "hi", "hello", "hey", "name", "who are you", "can you do", "help", "how does this work"
    ]

    # Vision always has highest priority
    if any(kw in text for kw in vision_keywords):
        return "vision"
    elif any(kw in text for kw in recommend_keywords):
        return "recommend"
    elif any(kw in text for kw in general_keywords):
        return "general"
    else:
        # fallback: try to assume recommendation, since it's common
        return "recommend"


def classify_intent(text):
    if USE_DEEP_ROUTING:
        return classify_with_embedding(text)
    else:
        return classify_simple(text)


def route_input(user_input):
    intent = classify_intent(user_input)
    
    if intent == "general":
        prompt = build_general_prompt(user_input)

    elif intent == "recommend":
        matches = search_catalog_with_text(user_input)
        prompt = build_recommend_prompt(user_input, matches)

    elif intent == "vision":
        # separate path: assumes image already processed
        matches = match_image_with_clip(user_input)
        prompt = build_vision_prompt(matches)

    return prompt
