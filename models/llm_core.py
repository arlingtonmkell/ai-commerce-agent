"""
models/llm_core.py
------------------
Lightweight local LLM interface with backend configurability.
Compatible with HuggingFace, llama.cpp, and vLLM runtimes.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# ──────────────────────────────────────────────────────────────────────────────
# ⚙️ Configuration Dataclass
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LLMConfig:
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    backend: str = "transformers"  # options: transformers | llama.cpp | vllm
    device_map: str = "auto"
    torch_dtype: str = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.7
    do_sample: bool = True
    memory_turns: int = 3   # short-term memory length


# ──────────────────────────────────────────────────────────────────────────────
# 🧠 Runtime Pipeline Loader
# ──────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=2)
def get_pipeline(config: LLMConfig):
    """Return a text-generation pipeline based on backend + config."""
    if config.backend == "transformers":
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map=config.device_map,
            torch_dtype=config.torch_dtype,
        )
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            do_sample=config.do_sample,
        )

    elif config.backend == "llama.cpp":
        raise NotImplementedError("llama.cpp backend not yet implemented.")
    elif config.backend == "vllm":
        raise NotImplementedError("vLLM backend not yet implemented.")
    else:
        raise ValueError(f"Unknown backend: {config.backend}")


# ──────────────────────────────────────────────────────────────────────────────
# 🧩 LLM Runtime Class
# ──────────────────────────────────────────────────────────────────────────────
class PalonaLLM:
    """
    Encapsulates conversation memory, prompt serialization, and generation.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.pipe = get_pipeline(self.config)
        self.memory: list[Dict[str, str]] = []

    # ─────────────────────────────
    # Prompt Serialization
    # ─────────────────────────────
    def format_prompt(self, prompt_dict: Dict) -> str:
        """Convert a prompt dictionary into a plain text string."""
        sys = prompt_dict.get("system", "")
        usr = prompt_dict.get("user", "")
        ctx = prompt_dict.get("catalog_context") or prompt_dict.get("products") or ""
        sections = [s for s in [sys, ctx, f"User: {usr}"] if s]
        base_prompt = "\n\n".join(sections)

        if self.memory:
            mem = "\n".join(
                [f"User: {t['user']}\nPalona: {t['reply']}" for t in self.memory[-self.config.memory_turns:]]
            )
            base_prompt = f"{mem}\n\n{base_prompt}"

        return f"{base_prompt}\n\nAssistant:"

    # ─────────────────────────────
    # Inference
    # ─────────────────────────────
    def generate(self, prompt_dict: Dict) -> str:
        """Generate text output and update memory."""
        prompt = self.format_prompt(prompt_dict)
        raw = self.pipe(prompt)[0]["generated_text"]
        reply = raw.split("Assistant:")[-1].strip()

        self.memory.append({"user": prompt_dict.get("user", ""), "reply": reply})
        return reply

    # ─────────────────────────────
    # Utilities
    # ─────────────────────────────
    def clear_memory(self):
        self.memory.clear()

    def summarize_memory(self) -> str:
        """Return a readable summary of recent conversation."""
        return "\n".join(
            [f"U: {t['user']} | A: {t['reply']}" for t in self.memory[-self.config.memory_turns:]]
        )


# ──────────────────────────────────────────────────────────────────────────────
# 🧠 Functional Wrapper (for direct use)
# ──────────────────────────────────────────────────────────────────────────────
def generate_response(prompt_dict: dict, config: LLMConfig = LLMConfig()) -> str:
    """Stateless generation shortcut (no memory persistence)."""
    pipe = get_pipeline(config)
    sys = prompt_dict.get("system", "")
    usr = prompt_dict.get("user", "")
    ctx = prompt_dict.get("catalog_context", prompt_dict.get("products", ""))

    prompt = f"{sys}\n\nUser: {usr}\n{ctx}\n\nAssistant:"
    out = pipe(prompt)[0]["generated_text"]
    return out.split("Assistant:")[-1].strip()
