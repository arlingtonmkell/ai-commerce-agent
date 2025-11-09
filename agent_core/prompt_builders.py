from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError

# Local contracts
from agent_core.prompt_types import Prompt  # system, user, intent, meta
from agent_core.context_core import build_system_context, default_system_snapshot

# Domain deps
from recommender.recommend import recommend_products
from vision.image_match import match_image

# === Runtime Paths ===
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# ===============================
# Schema used inside meta
# ===============================

class Match(BaseModel):
    id: str
    name: str
    score: float = Field(..., ge=-1e9, le=1e9)
    summary: Optional[str] = ""
    extra: Dict[str, Any] = {}

def _normalize_matches(raw: List[Dict[str, Any]]) -> List[Match]:
    out: List[Match] = []
    for r in raw or []:
        out.append(
            Match(
                id=str(r.get("id") or r.get("sku") or r.get("uid") or r.get("name")),
                name=str(r.get("name", "")),
                score=float(r.get("score", r.get("similarity", 0.0))),
                summary=str(r.get("description", r.get("summary", "")) or ""),
                extra={k: v for k, v in r.items() if k not in {"id","sku","uid","name","score","similarity","description","summary"}},
            )
        )
    return out

# ===============================
# 1) Builders (return Prompt)
# ===============================

def _memory_tail(memory: Optional[List[str]], k: int = 4) -> List[str]:
    return (memory or [])[-k:]

def build_general_prompt(
    user_input: str,
    memory: Optional[List[str]] = None,
    system_snapshot: Optional[Dict[str, Any]] = None,
) -> Prompt:
    sys_ctx = build_system_context(system_snapshot or default_system_snapshot())
    system = f"{sys_ctx} || Role: You are {system_snapshot.get('agent_name','Palona') if system_snapshot else 'Palona'}, a concise, friendly commerce assistant."
    return Prompt(
        system=system,
        user=user_input,
        intent="general",
        meta={"memory": _memory_tail(memory)}
    )

def build_recommend_prompt(
    user_input: str,
    memory: Optional[List[str]] = None,
    system_snapshot: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
) -> Prompt:
    sys_ctx = build_system_context(system_snapshot or default_system_snapshot())
    system = (
        f"{sys_ctx} || Role: Product recommender. "
        "Return on-catalog items, explain rationale briefly, and ask 1 clarifying question if uncertainty is high."
    )
    try:
        raw = recommend_products(user_input, top_k=top_k)  # expects [{id/name/score/description/...}]
    except Exception as e:
        raw = []
        err = f"recommender_error: {type(e).__name__}: {e}"
    else:
        err = None

    matches = _normalize_matches(raw)
    return Prompt(
        system=system,
        user=user_input,
        intent="recommend",
        meta={"memory": _memory_tail(memory), "matches": [m.model_dump() for m in matches], "errors": [err] if err else []},
    )

def build_vision_prompt(
    image_vector: Any,
    memory: Optional[List[str]] = None,
    system_snapshot: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
) -> Prompt:
    sys_ctx = build_system_context(system_snapshot or default_system_snapshot())
    system = (
        f"{sys_ctx} || Role: Visual matcher. "
        "Identify visually similar catalog items; prefer higher score. Mention top 3 succinctly."
    )
    try:
        raw = match_image(image_vector, k=top_k)  # expects [{id/name/score/description/...}]
    except Exception as e:
        raw = []
        err = f"vision_error: {type(e).__name__}: {e}"
    else:
        err = None

    matches = _normalize_matches(raw)
    return Prompt(
        system=system,
        user="[image uploaded]",
        intent="vision",
        meta={"memory": _memory_tail(memory), "matches": [m.model_dump() for m in matches], "errors": [err] if err else []},
    )

# ===============================
# 2) Serializers
# ===============================

def format_messages(prompt: Prompt) -> List[Dict[str, str]]:
    """
    Preferred: role-tagged messages for chat models.
    - system
    - assistant (memory as summarized context lines)
    - user
    - assistant (catalog context, if any)
    """
    msgs: List[Dict[str, str]] = []
    msgs.append({"role": "system", "content": prompt.system})

    mem: List[str] = prompt.meta.get("memory", [])
    if mem:
        msgs.append({"role": "assistant", "content": "Recent conversation (last N):\n" + "\n".join(mem)})

    user_content = prompt.user if isinstance(prompt.user, str) else json.dumps(prompt.user)
    msgs.append({"role": "user", "content": user_content})

    matches = prompt.meta.get("matches", [])
    if matches:
        # Compact tabular context
        lines = []
        for m in matches:
            lines.append(f"- {m.get('name','?')}  [score={m.get('score',0):.4f}] — {m.get('summary','')}")
        msgs.append({"role": "assistant", "content": "Catalog context:\n" + "\n".join(lines)})

    return msgs

def format_prompt_for_model(prompt: Prompt) -> str:
    """
    Legacy flat-string serializer retained for compatibility.
    """
    msgs = format_messages(prompt)
    return "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in msgs])

# ===============================
# 3) Logging (json lines per day)
# ===============================

def _log_path_for_today() -> Path:
    return LOGS_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"

def log_prompt(prompt: Prompt, messages: List[Dict[str, str]]) -> None:
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "intent": prompt.intent,
        "prompt": prompt.model_dump(),
        "messages": messages,
    }
    p = _log_path_for_today()
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ===============================
# 4) Master builder
# ===============================

def build_prompt(
    intent: str,
    user_input: Any = None,
    image_vector: Any = None,
    memory: Optional[List[str]] = None,
    system_snapshot: Optional[Dict[str, Any]] = None,
    as_messages: bool = True,
) -> Tuple[Prompt, List[Dict[str, str]] | str]:
    """
    Returns (validated Prompt, messages|flat_string), logs both.
    """
    if intent == "general":
        prompt = build_general_prompt(str(user_input), memory, system_snapshot)
    elif intent == "recommend":
        prompt = build_recommend_prompt(str(user_input), memory, system_snapshot)
    elif intent == "vision":
        prompt = build_vision_prompt(image_vector, memory, system_snapshot)
    else:
        # Unknown intent: still construct a guarded Prompt
        sys_ctx = build_system_context(system_snapshot or default_system_snapshot())
        prompt = Prompt(
            system=f"{sys_ctx} || Role: Fallback handler.",
            user=str(user_input),
            intent="unknown",
            meta={"memory": _memory_tail(memory), "errors": [f"unknown_intent:{intent}"]},
        )

    try:
        # Ensure contract consistency
        prompt = Prompt(**prompt.model_dump())
    except ValidationError as ve:
        # Convert to 'unknown' but preserve original payload for forensics
        prompt = Prompt(
            system="Contract validation failed; entering safe fallback.",
            user=str(user_input),
            intent="unknown",
            meta={"memory": _memory_tail(memory), "errors": ["validation_error", ve.errors()]},
        )

    if as_messages:
        messages = format_messages(prompt)
        log_prompt(prompt, messages)
        return prompt, messages
    else:
        flat = format_prompt_for_model(prompt)
        log_prompt(prompt, [{"role": "flat", "content": flat}])
        return prompt, flat
