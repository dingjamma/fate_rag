"""
Embedder for the Fate RAG Pipeline
Reads chunks from /data/chunks/, generates embeddings via AWS Bedrock Titan,
and uploads vectors + metadata to an OpenSearch index.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError, EndpointResolutionError
from dotenv import load_dotenv
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection, helpers

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration (from environment) ────────────────────────────────────────
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_REGION = os.getenv("BEDROCK_REGION", AWS_REGION)
BEDROCK_EMBEDDING_MODEL = os.getenv(
    "BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v1"
)
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT", "http://localhost:9200")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "fate-lore")
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "admin")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD", "admin")
USE_AWS_AUTH = os.getenv("USE_AWS_AUTH", "false").lower() == "true"

VECTOR_DIMS = 1536          # Titan Embeddings v1 output dimensions
BATCH_SIZE = 25             # chunks per Bedrock batch call
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 2.0

CHUNKS_DATA_DIR = Path("data/chunks")


# ── Index mapping ────────────────────────────────────────────────────────────
INDEX_MAPPING = {
    "settings": {
        "index": {
            "knn": True,
            "knn.algo_param.ef_search": 512,
        }
    },
    "mappings": {
        "properties": {
            "vector": {
                "type": "knn_vector",
                "dimension": VECTOR_DIMS,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",
                },
            },
            "text": {"type": "text", "analyzer": "english"},
            "title": {"type": "keyword"},
            "source_url": {"type": "keyword"},
            "slug": {"type": "keyword"},
            "category": {"type": "keyword"},
            "chunk_id": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "total_chunks": {"type": "integer"},
            "token_count": {"type": "integer"},
            "scraped_at": {"type": "date"},
        }
    },
}


# ── Bedrock client ───────────────────────────────────────────────────────────
def _get_bedrock_client() -> Any:
    return boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)


def embed_text(text: str, client: Any) -> Optional[list[float]]:
    """
    Call Bedrock Titan Embeddings to embed a single text string.
    Returns a list of floats or None on failure.
    """
    body = json.dumps({"inputText": text})
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.invoke_model(
                modelId=BEDROCK_EMBEDDING_MODEL,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response["body"].read())
            return result["embedding"]
        except (ClientError, EndpointResolutionError) as e:
            wait = RETRY_BACKOFF ** attempt
            logger.warning(f"Bedrock embedding failed (attempt {attempt + 1}): {e}. Retry in {wait:.1f}s")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(wait)
    return None


# ── OpenSearch client ────────────────────────────────────────────────────────
def _get_opensearch_client() -> OpenSearch:
    """
    Build an OpenSearch client.
    Uses AWS SigV4 auth for AWS OpenSearch Serverless;
    falls back to basic auth for local docker-compose instance.
    """
    if USE_AWS_AUTH:
        credentials = boto3.Session().get_credentials()
        aws_auth = AWSV4SignerAuth(credentials, AWS_REGION, "aoss")
        return OpenSearch(
            hosts=[{"host": OPENSEARCH_ENDPOINT.replace("https://", ""), "port": 443}],
            http_auth=aws_auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )
    else:
        # Local docker-compose OpenSearch
        host, port = _parse_local_endpoint(OPENSEARCH_ENDPOINT)
        return OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
            use_ssl=False,
            verify_certs=False,
        )


def _parse_local_endpoint(endpoint: str) -> tuple[str, int]:
    endpoint = endpoint.replace("http://", "").replace("https://", "")
    if ":" in endpoint:
        host, port_str = endpoint.rsplit(":", 1)
        return host, int(port_str)
    return endpoint, 9200


# ── Index management ─────────────────────────────────────────────────────────
def ensure_index(client: OpenSearch, index: str = OPENSEARCH_INDEX) -> None:
    """Create the index with k-NN mapping if it doesn't exist."""
    try:
        client.indices.create(index=index, body=INDEX_MAPPING)
        logger.info(f"Created index '{index}' with k-NN mapping.")
    except Exception as e:
        if "resource_already_exists_exception" in str(e).lower():
            logger.info(f"Index '{index}' already exists.")
        else:
            raise


# ── Bulk upload ──────────────────────────────────────────────────────────────
def _build_actions(chunks_with_vectors: list[dict[str, Any]], index: str) -> list[dict]:
    """Build OpenSearch bulk action dicts from embedded chunks."""
    actions = []
    for chunk in chunks_with_vectors:
        doc = {
            "_index": index,
            "_source": {
                "vector": chunk["vector"],
                "text": chunk["text"],
                "title": chunk["title"],
                "source_url": chunk["source_url"],
                "slug": chunk.get("slug", ""),
                "category": chunk["category"],
                "chunk_id": chunk["chunk_id"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"],
                "token_count": chunk["token_count"],
                "scraped_at": chunk.get("scraped_at") or None,
            },
        }
        actions.append(doc)
    return actions


def embed_and_upload(
    chunks: list[dict[str, Any]],
    os_client: OpenSearch,
    bedrock_client: Any,
    index: str = OPENSEARCH_INDEX,
) -> int:
    """
    Embed chunks in batches and bulk-upload to OpenSearch.
    Returns the number of successfully uploaded documents.
    """
    ensure_index(os_client, index)
    total = len(chunks)
    uploaded = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        logger.info(f"Embedding batch {batch_start // BATCH_SIZE + 1} / {-(-total // BATCH_SIZE)} ({len(batch)} chunks)...")

        embedded_batch: list[dict[str, Any]] = []
        for chunk in batch:
            vector = embed_text(chunk["text"], bedrock_client)
            if vector is None:
                logger.error(f"Failed to embed chunk {chunk['chunk_id']}; skipping.")
                continue
            embedded_batch.append({**chunk, "vector": vector})

        if not embedded_batch:
            continue

        actions = _build_actions(embedded_batch, index)
        successes, errors = helpers.bulk(os_client, actions, raise_on_error=False)
        if errors:
            logger.warning(f"{len(errors)} bulk errors in this batch.")
            for err in errors[:3]:
                logger.warning(f"  {err}")
        uploaded += successes
        logger.info(f"  Uploaded {successes} documents (cumulative: {uploaded}/{total})")

    return uploaded


# ── Entry point ──────────────────────────────────────────────────────────────
def run_embedder(
    chunks_path: Path = CHUNKS_DATA_DIR / "all_chunks.json",
    index: str = OPENSEARCH_INDEX,
) -> None:
    logger.info(f"Loading chunks from {chunks_path}")
    with open(chunks_path, encoding="utf-8") as f:
        chunks: list[dict[str, Any]] = json.load(f)

    logger.info(f"Loaded {len(chunks)} chunks. Starting embedding pipeline...")

    bedrock_client = _get_bedrock_client()
    os_client = _get_opensearch_client()

    uploaded = embed_and_upload(chunks, os_client, bedrock_client, index)
    logger.info(f"\nEmbedding pipeline complete. {uploaded}/{len(chunks)} chunks uploaded to '{index}'.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embed and upload Fate lore chunks to OpenSearch")
    parser.add_argument("--chunks", default=str(CHUNKS_DATA_DIR / "all_chunks.json"))
    parser.add_argument("--index", default=OPENSEARCH_INDEX)
    args = parser.parse_args()

    run_embedder(chunks_path=Path(args.chunks), index=args.index)
