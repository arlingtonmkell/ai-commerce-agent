from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

class Prompt(BaseModel):
    system: str
    user: Any
    intent: str = Field(..., pattern="^(general|recommend|vision|unknown)$")
    meta: Dict[str, Any] = {}

class LLMOutput(BaseModel):
    message: str
    prompt: Optional[Prompt] = None
    meta: Dict[str, Any] = {}
