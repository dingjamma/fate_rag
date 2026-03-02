# ⚔️ Fate RAG Chatbot

> *"I am the bone of my sword..."*

A production-grade **Retrieval-Augmented Generation (RAG)** chatbot for the **Fate Series** universe — powered by AWS Bedrock (Claude + Titan Embeddings), OpenSearch Serverless, and FastAPI.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock-FF9900?logo=amazonaws&logoColor=white)](https://aws.amazon.com/bedrock/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![OpenSearch](https://img.shields.io/badge/OpenSearch-2.13-005EB8?logo=opensearch&logoColor=white)](https://opensearch.org)
[![CDK](https://img.shields.io/badge/AWS_CDK-Python-FF9900?logo=amazonaws&logoColor=white)](https://docs.aws.amazon.com/cdk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Fate RAG Architecture                          │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──────────┐    HTTP     ┌─────────────────┐    Lambda    ┌───────────┐
  │ Browser  │ ──────────▶ │   API Gateway   │ ───────────▶ │  FastAPI  │
  │ (HTML UI)│ ◀────────── │   (HTTP API)    │ ◀─────────── │ (Mangum)  │
  └──────────┘             └─────────────────┘              └─────┬─────┘
                                                                   │
                           ┌───────────────────────────────────────┤
                           │                                       │
                    ┌──────▼──────┐                   ┌───────────▼──────────┐
                    │  Retriever  │                   │  AWS Bedrock Claude   │
                    │  (k-NN)     │                   │  (Generation)         │
                    └──────┬──────┘                   └──────────────────────┘
                           │
              ┌────────────▼───────────┐
              │   OpenSearch Serverless │
              │   (fate-lore index)     │
              │   k-NN vector search    │
              └────────────┬───────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
  ┌─────▼──────┐   ┌───────▼──────┐   ┌──────▼──────┐
  │  Scraper   │   │   Chunker    │   │  Embedder   │
  │(TypeMoon   │──▶│ (tiktoken    │──▶│ (Titan v1   │
  │  Wiki)     │   │  500 tok)    │   │  1536-dim)  │
  └────────────┘   └──────────────┘   └─────────────┘
        │
  ┌─────▼──────┐
  │    S3      │
  │(raw docs)  │
  └────────────┘

Data Pipeline:  scraper.py → chunker.py → embedder.py
Infrastructure: AWS CDK (Python) → infra/fate_rag_stack.py
```

---

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that improves LLM responses by grounding them in a knowledge base:

1. **Index**: Documents are split into chunks, converted to embedding vectors, and stored in a vector database (OpenSearch).
2. **Retrieve**: When a user asks a question, the question is embedded and the most similar document chunks are retrieved via k-NN search.
3. **Generate**: The retrieved chunks are injected into the prompt as context, and the LLM generates an answer grounded in that specific content.

This approach combines the fluency of large language models with the factual accuracy of a dedicated knowledge base — ideal for domain-specific assistants like this Fate Series lore bot.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Generation** | AWS Bedrock — `claude-sonnet-4-20250514` |
| **Embeddings** | AWS Bedrock — `amazon.titan-embed-text-v1` (1536-dim) |
| **Vector Store** | AWS OpenSearch Serverless (k-NN, HNSW) |
| **Backend** | FastAPI + Mangum (Lambda adapter) |
| **Compute** | AWS Lambda (Python 3.11) |
| **API** | AWS API Gateway (HTTP API) |
| **Infrastructure** | AWS CDK (Python) |
| **Data Pipeline** | BeautifulSoup4, tiktoken, boto3 |
| **Frontend** | Vanilla HTML/CSS/JS |
| **Local Dev** | Docker Compose (OpenSearch + Dashboards) |

---

## Project Structure

```
fate-rag/
├── infra/                      # AWS CDK infrastructure
│   ├── app.py                  # CDK app entry point
│   ├── fate_rag_stack.py       # Full AWS stack definition
│   ├── cdk.json                # CDK configuration
│   └── requirements.txt        # CDK-specific dependencies
│
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
│   ├── test_pipeline.py        # Unit tests: chunker (size, overlap, metadata)
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
│   └── deploy.yml              # GitHub Actions CI/CD (test + cdk deploy)
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
git clone https://github.com/your-username/fate-rag.git
cd fate-rag
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
# Scrape (be polite — rate limiting is built in)
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

Open `frontend/index.html` in a browser, or serve it:

```bash
python -m http.server 3000 --directory frontend
# Then open http://localhost:3000
```

---

## AWS Deployment

### Prerequisites
- AWS CDK CLI: `npm install -g aws-cdk`
- AWS credentials with permissions for Lambda, API Gateway, OpenSearch Serverless, S3, Bedrock, IAM

### 1. Install CDK dependencies

```bash
pip install -r infra/requirements.txt
```

### 2. Bootstrap CDK (once per account/region)

```bash
cdk bootstrap --app "python3 infra/app.py" \
  --context account=YOUR_ACCOUNT_ID \
  --context region=us-east-1
```

### 3. Deploy

```bash
# Deploy to dev environment
cdk deploy --app "python3 infra/app.py" \
  --context env=dev \
  --context account=YOUR_ACCOUNT_ID \
  --context region=us-east-1

# Deploy to prod
cdk deploy --app "python3 infra/app.py" \
  --context env=prod \
  --context account=YOUR_ACCOUNT_ID \
  --context region=us-east-1
```

### 4. Run the data pipeline against AWS OpenSearch

After deployment, CDK will output your OpenSearch endpoint. Update your `.env`:

```bash
OPENSEARCH_ENDPOINT=https://your-collection.us-east-1.aoss.amazonaws.com
USE_AWS_AUTH=true
```

Then run the embedder:

```bash
python data_pipeline/embedder.py
```

---

## API Usage

### Health check

```bash
curl https://your-api-gateway-url/health
# {"status": "ok", "service": "fate-rag-chatbot"}
```

### Chat

```bash
curl -X POST https://your-api-gateway-url/chat \
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

### Multi-turn conversation

```bash
curl -X POST https://your-api-gateway-url/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Who taught him this technique?",
    "conversation_history": [
      {"role": "user", "content": "What is Unlimited Blade Works?"},
      {"role": "assistant", "content": "Unlimited Blade Works is..."}
    ]
  }'
```

### Category-filtered search

```bash
curl -X POST https://your-api-gateway-url/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about Noble Phantasms",
    "category_filter": "noble_phantasm"
  }'
```

---

## Running Tests

```bash
pytest tests/ -v
```

The test suite covers:
- **Chunker**: Token size limits, overlap correctness, metadata preservation, edge cases (empty, short, whitespace-only documents)
- **Retriever**: k-NN query construction, result parsing, category filtering, error propagation, multi-query deduplication

All tests use mocked AWS/OpenSearch clients — no live infrastructure required.

---

## GitHub Actions CI/CD

The workflow at `.github/workflows/deploy.yml`:

1. **On every push / PR to `main`**: Runs the full test suite.
2. **On push to `main`**: Deploys to the `dev` AWS environment using CDK.

### Required GitHub Secrets

| Secret | Description |
|---|---|
| `AWS_DEPLOY_ROLE_ARN` | IAM role ARN with deployment permissions (OIDC) |
| `AWS_ACCOUNT_ID` | Your AWS account ID |
| `AWS_REGION` | Target region (e.g., `us-east-1`) |

---

## Future Improvements

- [ ] **Hybrid search**: Combine k-NN vector search with BM25 lexical search (OpenSearch hybrid query) for better recall
- [ ] **Re-ranking**: Add a cross-encoder re-ranker to improve precision of retrieved chunks
- [ ] **Streaming UI**: Full SSE streaming support in the frontend (the backend already supports it)
- [ ] **FGO Integration**: Expand the data pipeline to cover Fate/Grand Order servant profiles (5000+ entries)
- [ ] **Citation links**: Render source citations as clickable links in the chat UI
- [ ] **Authentication**: Add Cognito or API key authentication to the API Gateway
- [ ] **Evaluation**: Add a RAGAS or TruLens evaluation pipeline to measure answer quality
- [ ] **Caching**: Use ElastiCache or Lambda response caching for frequent queries
- [ ] **Multi-language**: Support Japanese-language queries using Amazon Titan Text Embeddings v2 multilingual model
- [ ] **Image support**: Index Noble Phantasm images using Amazon Titan Multimodal Embeddings

---

## License

MIT © 2024 — Built with ⚔️ for the Type-Moon community.

*This project is a fan work and is not affiliated with TYPE-MOON, Aniplex, or any official Fate Series rights holders.*
