
"""
models/__init__.py
------------------
Expose unified model interface.
"""

from .text_encoder import embed_text
from .vision_encoder import embed_image
from .llm_core import generate_response

__all__ = ["embed_text", "embed_image", "generate_response"]
