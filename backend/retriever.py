"""
OpenSearch Retriever for the Fate RAG Pipeline
Performs k-NN vector search against the fate-lore index
and returns the top-k most semantically similar chunks.
"""

import json
import logging
import os
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
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

DEFAULT_TOP_K = 5


# ── Client helpers ───────────────────────────────────────────────────────────
def _get_bedrock_client() -> Any:
    return boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)


def _get_opensearch_client() -> OpenSearch:
    if USE_AWS_AUTH:
        credentials = boto3.Session().get_credentials()
        aws_auth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            AWS_REGION,
            "es",
            session_token=credentials.token,
        )
        endpoint = OPENSEARCH_ENDPOINT.replace("https://", "")
        return OpenSearch(
            hosts=[{"host": endpoint, "port": 443}],
            http_auth=aws_auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )
    else:
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


# ── Embedding helper ─────────────────────────────────────────────────────────
def embed_query(query: str, bedrock_client: Any) -> list[float]:
    """Embed a query string using Bedrock Titan Embeddings."""
    body = json.dumps({"inputText": query})
    try:
        response = bedrock_client.invoke_model(
            modelId=BEDROCK_EMBEDDING_MODEL,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        return result["embedding"]
    except ClientError as e:
        logger.error(f"Bedrock embedding error: {e}")
        raise


# ── Retriever ────────────────────────────────────────────────────────────────
class FateRetriever:
    """
    Wraps OpenSearch k-NN search for retrieving relevant Fate lore chunks.
    Clients can optionally filter results by category.
    """

    def __init__(
        self,
        os_client: Optional[OpenSearch] = None,
        bedrock_client: Optional[Any] = None,
        index: str = OPENSEARCH_INDEX,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.os_client = os_client or _get_opensearch_client()
        self.bedrock_client = bedrock_client or _get_bedrock_client()
        self.index = index
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        category_filter: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Perform a k-NN vector search for the given query.

        Args:
            query: Natural language question.
            top_k: Number of results to return (overrides instance default).
            category_filter: Optional category string to filter by
                             (e.g. "servant", "noble_phantasm", "lore").

        Returns:
            List of dicts with keys: text, title, source_url, category, score.
        """
        k = top_k or self.top_k
        query_vector = embed_query(query, self.bedrock_client)

        knn_query: dict[str, Any] = {
            "size": k,
            "query": {
                "knn": {
                    "vector": {
                        "vector": query_vector,
                        "k": k,
                    }
                }
            },
            "_source": ["text", "title", "source_url", "category", "chunk_id", "chunk_index"],
        }

        # Optional metadata filter via post_filter
        if category_filter:
            knn_query["post_filter"] = {
                "term": {"category": category_filter}
            }

        try:
            response = self.os_client.search(index=self.index, body=knn_query)
        except Exception as e:
            logger.error(f"OpenSearch query failed: {e}")
            raise

        hits = response.get("hits", {}).get("hits", [])
        results: list[dict[str, Any]] = []

        for hit in hits:
            source = hit.get("_source", {})
            results.append(
                {
                    "text": source.get("text", ""),
                    "title": source.get("title", ""),
                    "source_url": source.get("source_url", ""),
                    "category": source.get("category", ""),
                    "chunk_id": source.get("chunk_id", ""),
                    "score": hit.get("_score", 0.0),
                }
            )

        logger.info(
            f"Retrieved {len(results)} chunks for query '{query[:60]}...' "
            f"(category_filter={category_filter!r})"
        )
        return results

    def retrieve_multi_query(
        self,
        queries: list[str],
        top_k: Optional[int] = None,
        category_filter: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve results for multiple query strings and merge/deduplicate by chunk_id,
        keeping the highest score for each chunk.
        """
        seen: dict[str, dict[str, Any]] = {}
        for q in queries:
            for doc in self.retrieve(q, top_k=top_k, category_filter=category_filter):
                cid = doc["chunk_id"]
                if cid not in seen or doc["score"] > seen[cid]["score"]:
                    seen[cid] = doc

        merged = sorted(seen.values(), key=lambda d: d["score"], reverse=True)
        return merged[: top_k or self.top_k]
