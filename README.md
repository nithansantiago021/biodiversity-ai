# Biodiversity AI & Ecological Restoration Engine

![CI Workflow](https://github.com/<nithansantiago021>/<biodiversity-ai>/actions/workflows/ci.yml/badge.svg)

A production-grade, RAG-enabled AI engine designed to process environmental metrics, synthesize ecological pressures, and generate scientifically grounded restoration recommendations. Built with a conversational agent interface and multi-stage vector retrieval, this system acts as an expert intelligence layer for ecological restoration.

* **Live Demo (Streamlit):** [biodiversity-ai-de.streamlit.app](https://biodiversity-ai-de.streamlit.app/)
* **Backend Health Check:** [biodiversity-backend.onrender.com/health](https://biodiversity-backend.onrender.com/health)  
  *(Note: Hosted on Render's free tier. Opening the health link wakes up the instance if it is sleeping).*

## System Architecture

The application implements a hybrid LLM architecture combining deterministic evaluation gates with two-stage Retrieval-Augmented Generation (RAG):

* **API & Core:** FastAPI, SQLAlchemy, Pydantic.
* **Conversational Agent Workflow:** LangGraph (State Machine with multi-turn memory).
* **Vector Store & Database:** PostgreSQL with `pgvector` hosted on Supabase.
* **Serverless Vector Pipeline:**
  * **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` via Hugging Face Serverless Inference API (384-dimensional vectors).
  * **Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2` via serverless joint-attention inference.
* **Data Ingestion:** Automated pipeline for unstructured document parsing (`.pdf`, `.txt`, `.md`) and structured batch processing (`.csv`).
* **Memory & Persistence:** Persistent session memory stored directly in PostgreSQL (`chat_messages`).

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
  'https://biodiversity-backend.onrender.com/chat' \
  -H 'Content-Type: application/json' \
  -d '{
  "session_id": "demo-session-01",
  "message": "Biodiversity is declining on my land. Soil carbon is 0.4% and rainfall is 850mm."
}'
```

## CI/CD & Deployment Pipeline

This project uses an automated continuous integration and continuous deployment workflow to ensure code quality and seamless hosting.

### 1. Continuous Integration (GitHub Actions)
* **Automated Testing:** On every `push` or `pull_request` to the `main` branch, a GitHub Actions workflow executes:
  * **Code Linting & Formatting:** Checks code standards using `flake8` / `black`.
  * **Automated Test Suite:** Runs `pytest` to execute unit and integration tests across backend modules.
  * **Dependency Audit:** Verifies that required packages build cleanly in a isolated Python 3.12 environment.

### 2. Continuous Deployment (Render & Webhooks)
* **Backend Deployment:** The FastAPI application is connected to **Render** via automated Git webhooks.
  * Pushing changes to `main` automatically triggers a cloud build.
  * Render installs dependencies, runs container health checks (`/health`), and deploys the live service without downtime.
* **Environment Configuration:** Production credentials (`DATABASE_URL`, `HF_TOKEN`, `GROQ_API_KEY`) are injected securely through Render's managed Environment Settings.

### 3. Pipeline Architecture Flow
`Developer Commit` ➔ `GitHub Push` ➔ `GitHub Actions (pytest & linting)` ➔ `Render Auto-Deploy Webhook` ➔ `Live Production API`


## Local Setup

### Clone & Environment:

```bash
git clone https://github.com/nithansantiago021/biodiversity-ai.git
cd biodiversity-ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Create and Activate Virtual Environment:
```Bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies:
```Bash
pip install -r requirements.txt
```

### Configure Environment Variables (.env):
```bash
# Code snippet
DATABASE_URL=postgresql://user:password@localhost:5432/biodiversity_db
GROQ_API_KEY=gsk_your_groq_api_key_here
HF_TOKEN=hf_your_hf_access_code_here (read-only)
```

### Run Database Migrations & Ingest Documents:
```bash
python -m app.database.init_db
python -m app.knowledge.ingest
```

### Execute Unit Test Suite:
```Bash
pytest -q
```

### Launch Interactive CLI:
```Bash
python cli.py
```

### Lanch Streamlit UI:
```bash
streamlit run frontend.py
```

## Recommended Test Queries for Review

Direct Parameter Query:

        "For agricultural land with a soil pH of 5.2, soil organic carbon at 0.8%, and seasonal heavy rainfall, what are the primary indicators for soil biodiversity degradation and how can we mitigate it?"

Standard Water Query:

        "What is the safe TDS level for drinking water according to BIS 10500 standards?"

Clarification Trigger Query:

        "how to mitigate land detoriation?"