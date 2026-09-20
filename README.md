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

## System Limitations & Future Scalability

### Current Limitations (v1.0)

1. **Serverless Inference Latency & Rate Limits:**
   * **Constraint:** Offloading dense vector generation (`all-MiniLM-L6-v2`) and cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) to Hugging Face’s Serverless Inference API eliminates local memory overhead (<50MB RAM), but introduces cold-start latency (1–3 seconds) on un-cached requests and relies on free-tier rate limits.
   
2. **Dynamic Heuristics vs. Rigid Deterministic Constraints:**
  * **Architectural Trade-off:** To support direct, zero-shot ingestion of unstructured research PDFs, environmental reports, and regional restoration handbooks, the system shifted away from rigid, hard-coded numerical boundary rules. Because raw scientific literature exhibits diverse reporting units, variable metric ranges, and context-dependent ecological thresholds, relying on static deterministic rules proved brittle and prone to unanswerable state loops. 

3. **Geospatial & Vector Ingestion Scope:**
   * **Constraint:** The ingestion pipeline currently processes structured numerical datasets (`.csv`) and unstructured scientific literature (`.pdf`, `.md`, `.txt`). It does not natively parse raw geospatial vector formats (e.g., GeoJSON, shapefiles, raster satellite imagery).

---

### Future Scalability & Roadmap (v2.0)

#### 1. Hybrid GraphRAG for Complex Ecological Causality
* **Upgrade:** Transition from purely dense vector similarity search to a hybrid **Graph-RAG** model using `pg_graphql` or Neo4j.
* **Impact:** Explicitly maps causal relationship networks between soil chemistry deltas, microclimate shifts, and native plant survival rates, guaranteeing deeper multi-variable reasoning.

#### Deterministic Ecological Boundary Model & Standardized Dataset
* **Evolution:** To bridge the gap between flexible LLM heuristic synthesis and strict biological limits, future iterations will introduce a dedicated, curated **Environmental Metric & Species Threshold Dataset**.
* **Impact:**
  * **Standardized Knowledge Mapping:** Standardizes disparate unstructured PDF inputs into a structured schema of verified numerical boundary bounds (e.g., explicit pH, soil organic carbon, salinity, and rainfall tolerance ranges per native species).
  * **Hybrid Deterministic Validation Node:** Integrates a pre-retrieval validation node inside the `LangGraph` state machine. This node evaluates candidate species against hard environmental limits before triggering RAG retrieval—combining the strict accuracy of deterministic rules with the generative depth of LLM synthesis.

#### 3. PostGIS Spatial Layer Integration
* **Upgrade:** Expand the database layer from `pgvector` to include **PostGIS 3.4** and GeoAlchemy2.
* **Impact:** Allows users to upload regional GeoJSON polygon boundaries or draw land parcels on an interactive Mapbox map, triggering automatic spatial joins against historical soil and climate raster grids.

#### 4. Dedicated Inference Infrastructure
* **Upgrade:** Migrate from public serverless API endpoints to dedicated **Text Embeddings Inference (TEI)** containers on GPU-backed instances (e.g., AWS ECS or RunPod).
* **Impact:** Sub-50ms vector search latency and 100% control over throughput and request concurrency.