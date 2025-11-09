"""
api/main.py
-----------
Palona AI Agent — REST API Layer (Optimized Phase 1)
----------------------------------------------------
Production controller: dynamically initializes LLM, embeddings,
and vision pipelines through environment-configurable backend control.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, BaseSettings, Field
from typing import Optional, Any
from contextlib import asynccontextmanager
import traceback

# ─────────────────────────────────────────────────────────────────────────────
# ⚙️ 1. Global Configuration Layer
# ─────────────────────────────────────────────────────────────────────────────

class LLMConfig(BaseSettings):
    model_name: str = "local-llm"                 # e.g. "ggml-mistral", "llama3-8b-q4"
    device: str = "cuda"                          # or "cpu"
    max_tokens: int = 512
    temperature: float = 0.7
    embeddings_model: str = "bge-small-en"
    cache_dir: str = "data/cache"

    class Config:
        env_prefix = "PALONA_"                   # allows env vars like PALONA_MODEL_NAME

config = LLMConfig()

# ─────────────────────────────────────────────────────────────────────────────
# 🧠 2. Pipeline Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_pipeline(cfg: LLMConfig):
    """
    Returns callable objects for model inference & embedding access
    according to provided configuration.
    """
    from models.llm_core import load_local_llm
    from recommender.vector_utils import load_embeddings
    from vision.clip_index import load_image_embeddings

    llm = load_local_llm(
        model_name=cfg.model_name,
        device=cfg.device,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )
    text_index = load_embeddings(model_name=cfg.embeddings_model)
    image_index = load_image_embeddings()
    return {"llm": llm, "text_index": text_index, "image_index": image_index}

# Global reference
pipeline = None

# ─────────────────────────────────────────────────────────────────────────────
# 🔁 3. Lifespan Context (Replaces @app.on_event)
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern startup/shutdown lifecycle for FastAPI.
    Initializes and disposes runtime pipeline cleanly.
    """
    global pipeline
    pipeline = get_pipeline(config)
    print(f"🚀 Palona AI Agent initialized with model {config.model_name} on {config.device}")
    yield  # startup done — application now running
    # Optional teardown logic (release GPU, clear caches, etc.)
    print("🧩 Palona AI Agent shutting down...")

# ─────────────────────────────────────────────────────────────────────────────
# ⚙️ 4. Global App Instance (with lifespan)
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Palona AI Agent",
    version="1.2.0",
    description="Dynamic runtime controller for Palona AI Agent.",
    lifespan=lifespan,
)

# ─────────────────────────────────────────────────────────────────────────────
# 📥 5. Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    text: Optional[str] = Field(None, description="User text input")
    image: Optional[str] = Field(None, description="Base64-encoded image string (optional)")

class QueryResponse(BaseModel):
    type: str
    result: Any
    status: str = "ok"
    message: Optional[str] = None

# ─────────────────────────────────────────────────────────────────────────────
# 🔍 6. Main Endpoint: /query
# ─────────────────────────────────────────────────────────────────────────────

from agent_core.dispatcher import handle_query

@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    """
    Unified query endpoint with pipeline + config injection.
    """
    try:
        if not req.text and not req.image:
            raise HTTPException(status_code=400, detail="Either 'text' or 'image' must be provided.")

        result = handle_query(
            text=req.text,
            image=req.image,
            pipeline=pipeline,
            config=config,
        )
        return QueryResponse(type=result.get("type", "unknown"), result=result, status="ok")

    except HTTPException as e:
        raise e
    except Exception as e:
        print("⚠️ Internal Error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# ─────────────────────────────────────────────────────────────────────────────
# 🩺 7. Health & Config Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": config.model_name, "device": config.device}

@app.get("/config")
def get_current_config():
    return config.dict()

@app.post("/rebuild_embeddings")
def rebuild_embeddings():
    try:
        # placeholder
        return {"status": "ok", "message": "Embedding rebuild triggered (stub)."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rebuild embeddings: {str(e)}")

@app.get("/")
def root():
    return {
        "message": "Palona AI Agent running.",
        "model": config.model_name,
        "device": config.device,
        "endpoints": ["/query", "/health", "/config", "/rebuild_embeddings"],
    }
