# agent_core/logger.py
import json, os
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_DIR.mkdir(exist_ok=True)

def _log_path():
    date = datetime.now().strftime("%Y-%m-%d")
    return LOG_DIR / f"{date}.json"

def log_event(event_type, payload):
    """Append an event to today's JSON log."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "type": event_type,
        "payload": payload,
    }
    path = _log_path()
    existing = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except Exception:
                existing = []
    existing.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    return entry
