# Biodiversity AI & Ecological Restoration Engine

A production-grade, RAG-enabled AI engine designed to process environmental metrics, synthesize ecological pressures, and generate scientifically grounded restoration recommendations. Built with a conversational agent interface and multi-stage vector retrieval, this system acts as an expert intelligence layer for ecological restoration.

## System Architecture

The application implements a hybrid LLM architecture combining deterministic evaluation gates with vector-based Retrieval-Augmented Generation (RAG):

* **API & Core:** FastAPI, SQLAlchemy (PostgreSQL).
* **Conversational AI Workflow:** LangGraph (State Machine).
* **Vector Store & Retrieval:** `pgvector` for PostgreSQL.
* **Embeddings:** `all-MiniLM-L6-v2` (SentenceTransformers) generating 384-dimensional vectors.
* **Data Ingestion:** Automated pipeline for `.pdf`, `.txt`, and `.md` unstructured parsing + `.csv` batch structured data.
* **Memory:** Persistent PostgreSQL conversational memory.

## Key Features

1. **Intelligent Gatekeeper:** The LangGraph agent evaluates user input. If critical metrics (e.g., Soil Organic Carbon, pH, Rainfall) are missing, it asks clarifying questions before triggering heavy LLM synthesis.
2. **Scientific RAG Pipeline:** Ingests complex ecological reports, splits them into context-aware chunks with overlap, and generates embeddings for precise cross-encoder reranked retrieval.
3. **Multi-Turn Chat Memory:** Full contextual memory stored directly in PostgreSQL to support continuous conversation with the AI.
4. **Batch Data Ingestion:** Fast batch-processing of raw site data via CSV uploads.

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | System health check. |
| `POST` | `/observations` | Create a new structured environmental observation. |
| `POST` | `/observations/{id}/recommendations` | Trigger the LangGraph RAG synthesis for a specific observation. |
| `POST` | `/observations/upload-csv` | Batch ingest site metrics via CSV file. |
| `POST` | `/chat` | Multi-turn conversational endpoint with persistent memory and automatic metric gatekeeping. |

### Example cURL: Chat Endpoint
```bash
curl -X 'POST' \
  '[https://biodiversity-ai-backend.onrender.com/chat](https://biodiversity-ai-backend.onrender.com/chat)' \
  -H 'Content-Type: application/json' \
  -d '{
  "session_id": "demo-session-01",
  "message": "Biodiversity is declining on my land. Soil carbon is 0.4% and rainfall is 850mm."
}'
```

## Local Setup

### Clone & Environment:

```bash
git clone <repo_url>
cd biodiversity-ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Start Infrastructure (Docker):

```Bash
docker compose up -d
```
### Run Migrations & Ingestion:

```Bash
python -c "from app.database import engine, Base; Base.metadata.create_all(bind=engine)"
python -c "from app.database import SessionLocal; from app.knowledge.ingestion import run_ingestion_pipeline; db = SessionLocal(); run_ingestion_pipeline(db); db.close()"
```

### Run Server:

```Bash
uvicorn app.main:app --reload
```