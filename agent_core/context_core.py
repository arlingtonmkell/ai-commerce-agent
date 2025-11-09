from __future__ import annotations
from typing import Any, Dict
import os
import platform
import sys
from datetime import datetime

DEFAULT_AGENT_NAME = "Palona"
DEFAULT_INTENTS = ["general", "recommend", "vision", "unknown"]

def build_system_context(snapshot: Dict[str, Any]) -> str:
    """
    Deterministic, compact system preamble describing runtime + routing config.
    Keep short; downstream builders append task-specific instructions.
    """
    parts = [
        f"Agent: {snapshot.get('agent_name', DEFAULT_AGENT_NAME)}",
        f"Runtime: py{snapshot.get('python_version')} on {snapshot.get('os')} ({snapshot.get('machine')})",
        f"Dispatcher: {snapshot.get('dispatcher','n/a')}",
        f"Intents: {', '.join(snapshot.get('intents', DEFAULT_INTENTS))}",
        f"Recommender: {snapshot.get('recommender','n/a')}",
        f"Vision: {snapshot.get('vision','n/a')}",
        f"Catalog: {snapshot.get('catalog_items','?')} items",
        f"UTC: {snapshot.get('utc_now')}",
    ]
    return " | ".join(parts)

def default_system_snapshot() -> Dict[str, Any]:
    """
    Safe defaults; callers can merge/override before passing to build_system_context().
    """
    return {
        "agent_name": DEFAULT_AGENT_NAME,
        "python_version": platform.python_version(),
        "os": platform.platform(terse=True),
        "machine": platform.machine(),
        "dispatcher": os.getenv("PALONA_DISPATCHER_IMPL", "embedding-router"),
        "intents": DEFAULT_INTENTS,
        "recommender": os.getenv("PALONA_RECOMMENDER_IMPL", "semantic-kNN"),
        "vision": os.getenv("PALONA_VISION_IMPL", "clip-index"),
        "catalog_items": os.getenv("PALONA_CATALOG_SIZE", "?"),
        "utc_now": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
