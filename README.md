# ⚔️ Fate RAG Chatbot

> *"I am the bone of my sword..."*

A **Retrieval-Augmented Generation (RAG)** chatbot for the **Fate Series** universe — powered by AWS Bedrock (Claude + Titan Embeddings), OpenSearch Serverless, and FastAPI.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![OpenSearch](https://img.shields.io/badge/OpenSearch-2.13-005EB8?logo=opensearch&logoColor=white)](https://opensearch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ⚠️ AWS Backend Shut Down

The live AWS backend (API Gateway + Lambda + OpenSearch Serverless + S3) has been **decommissioned** to avoid ongoing infrastructure costs. OpenSearch Serverless alone runs a minimum of ~$700/month even at zero usage.

The GitHub Pages frontend is still up but the API is no longer active. To run the chatbot yourself, follow the **Local Setup** section below — Docker Compose spins up a full OpenSearch stack locally at no cost.

The `infra/` CDK code and the AWS deploy workflow have been removed from the repo. The data pipeline and backend code remain intact for local use.

---

## What is RAG?

**Retrieval-Augmented Generation (RAG)** improves LLM responses by grounding them in a knowledge base:

1. **Index**: Documents are split into chunks, converted to embedding vectors, and stored in a vector database (OpenSearch).
2. **Retrieve**: When a user asks a question, the question is embedded and the most similar document chunks are retrieved via k-NN search.
3. **Generate**: The retrieved chunks are injected into the prompt as context, and the LLM generates an answer grounded in that specific content.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Generation** | AWS Bedrock — `claude-sonnet-4-20250514` |
| **Embeddings** | AWS Bedrock — `amazon.titan-embed-text-v1` (1536-dim) |
| **Vector Store** | OpenSearch (local via Docker / was AWS OpenSearch Serverless) |
| **Backend** | FastAPI + Mangum (Lambda adapter) |
| **Data Pipeline** | BeautifulSoup4, tiktoken, boto3 |
| **Frontend** | Vanilla HTML/CSS/JS (hosted on GitHub Pages) |
| **Local Dev** | Docker Compose (OpenSearch + Dashboards) |

---

## Project Structure

```
fate-rag/
├── data_pipeline/
│   ├── scraper.py              # Type-Moon Wiki scraper (BeautifulSoup)
│   ├── chunker.py              # Token-aware chunking with overlap
│   └── embedder.py             # Bedrock Titan embedding + OpenSearch upload
│
├── backend/
│   ├── app.py                  # FastAPI app + Mangum Lambda handler
│   ├── retriever.py            # OpenSearch k-NN vector search
│   └── prompt.py               # System prompt + RAG template
│
├── frontend/
│   └── index.html              # Single-page chat UI
│
├── notebooks/
│   └── rag_exploration.ipynb   # EDA: chunking, embedding, retrieval experiments
│
├── tests/
│   ├── test_pipeline.py        # Unit tests: chunker
│   └── test_retriever.py       # Unit tests: retriever (mocked OpenSearch/Bedrock)
│
├── sample_data/                # Pre-written Fate lore for immediate use
│   ├── servants.json
│   ├── noble_phantasms.json
│   ├── masters.json
│   ├── lore.json
│   └── fate_zero_servants.json
│
├── .github/workflows/
│   └── deploy-pages.yml        # GitHub Pages deployment (frontend only)
│
├── requirements.txt
├── .env.example
├── docker-compose.yml          # Local OpenSearch for development
├── Dockerfile
└── README.md
```

---

## Local Setup

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- AWS account with Bedrock model access enabled (Claude Sonnet + Titan Embeddings)
- AWS CLI configured

### 1. Clone and install

```bash
git clone https://github.com/dingjamma/fate_rag.git
cd fate_rag
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your values
```

### 3. Start local OpenSearch

```bash
docker-compose up -d
# OpenSearch: http://localhost:9200
# Dashboards: http://localhost:5601
```

Wait ~60 seconds for OpenSearch to initialize, then verify:

```bash
curl http://localhost:9200/_cluster/health
```

### 4. Run the data pipeline

#### Option A: Use sample data (no scraping required)

```bash
# Chunk sample data
python -m data_pipeline.chunker --sample

# Embed and upload to local OpenSearch
python -m data_pipeline.embedder
```

#### Option B: Scrape Type-Moon Wiki

```bash
# Scrape (rate limiting is built in)
python -m data_pipeline.scraper

# Chunk
python data_pipeline/chunker.py --input data/raw/all_documents.json

# Embed and upload
python data_pipeline/embedder.py
```

### 5. Start the API

```bash
uvicorn backend.app:app --reload --port 8000
```

### 6. Open the frontend

```bash
python -m http.server 3000 --directory frontend
# Then open http://localhost:3000
```

---

## API Usage

### Health check

```bash
curl http://localhost:8000/health
# {"status": "ok", "service": "fate-rag-chatbot"}
```

### Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is Unlimited Blade Works?",
    "conversation_history": []
  }'
```

**Response:**
```json
{
  "answer": "Unlimited Blade Works is the Reality Marble Noble Phantasm of EMIYA (Archer)...",
  "retrieved_docs": [
    {
      "text": "Unlimited Blade Works is a Reality Marble...",
      "title": "Unlimited Blade Works: Infinite Creation of Swords",
      "source_url": "https://typemoon.fandom.com/wiki/Unlimited_Blade_Works",
      "category": "noble_phantasm",
      "score": 0.94
    }
  ],
  "model_id": "claude-sonnet-4-20250514"
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

The test suite covers:
- **Chunker**: Token size limits, overlap correctness, metadata preservation, edge cases
- **Retriever**: k-NN query construction, result parsing, category filtering, error propagation

All tests use mocked AWS/OpenSearch clients — no live infrastructure required.

---

## License

MIT © 2024 — Built with ⚔️ for the Type-Moon community.

*This project is a fan work and is not affiliated with TYPE-MOON, Aniplex, or any official Fate Series rights holders.*
