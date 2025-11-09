"""
query/general_chat.py
---------------------
Handles general conversation using a small local LLM (e.g. TinyLlama or Phi).
"""

from datetime import datetime

def handle_general_query(text: str) -> str:
    """
    A minimal rule-based or model-assisted chat.
    Replace with a local model if available.
    """
    text_lower = text.lower()

    if "your name" in text_lower:
        return "I’m Palona — your AI shopping assistant!"
    if "time" in text_lower:
        return f"It's currently {datetime.now().strftime('%H:%M')}."
    if "what can you do" in text_lower:
        return (
            "I can chat, recommend products, and find items from images — "
            "all from our local catalog."
        )
    return (
        "I'm here to help you find products or answer your questions. "
        "Try asking me for a product recommendation!"
    )
