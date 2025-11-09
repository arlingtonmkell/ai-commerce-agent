# ai-commerce-agent
An AI-powered multimodal agent for commerce applications, supporting chat, recommendations, and image-based search.


embeddings/ is ignored for size and privacy. Use vector_utils.py to regenerate from catalog.json.

##  Features

-  **General Chat** — Handles open-ended dialogue via LLM
-  **Text-Based Recommendations** — Suggests relevant products from a catalog
-  **Image-Based Search** — Uses CLIP to find visually similar items
-  **Intent Routing** — Classifies input to route to the appropriate module
-  **FastAPI Backend** — Unified endpoint for handling all user input

---

##  Architecture

```plaintext
+------------------+
|   /api/main.py   | ← FastAPI Entry Point
+------------------+
         ↓
+---------------------+
| agent_core/dispatcher.py |
+---------------------+
    ↓         ↓         ↓
 [query]   [recommender]   [vision]
```

 ##  Setup

1. Clone the Repo
git clone https://github.com/arlingtonmkell/ai-commerce-agent.git
cd ai-commerce-agent

2. Install Dependencies
pip install -r requirements.txt

3. Set Environment Variables
cp .env.example .env
# If needed, edit .env to include your API keys (e.g., OPENAI_API_KEY)

4. Run the Server
uvicorn api.main:app --reload

##  Project Structure

ai-commerce-agent/
├── api/                 # FastAPI endpoint
├── agent_core/          # Dispatcher logic
├── query/               # General query + intent routing
├── recommender/         # Text embedding and search
├── vision/              # CLIP-based image matching
├── catalog/             # Product data loader
├── embeddings/          # Ignored: text/image embedding cache
└── models/              # Model Integration Layer
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
├── SPECS.md             # Project specs & FSM
└── README.md

## License

MIT — see LICENSE for details.

## Author

Arlington Kell 

Hi!
