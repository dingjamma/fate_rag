"""
Unit tests for the retriever module.
Uses mocked OpenSearch and Bedrock clients to test retrieval logic
without requiring live AWS or OpenSearch infrastructure.
"""

from unittest.mock import MagicMock, patch

import pytest

from backend.retriever import FateRetriever


# ── Mock helpers ──────────────────────────────────────────────────────────────
def _make_mock_bedrock_client(embedding_vector: list[float] | None = None) -> MagicMock:
    """Return a mock Bedrock client that returns a fixed embedding vector."""
    if embedding_vector is None:
        embedding_vector = [0.1] * 1536

    mock_client = MagicMock()
    import json

    class FakeBody:
        def read(self_inner):
            return json.dumps({"embedding": embedding_vector}).encode()

    mock_client.invoke_model.return_value = {"body": FakeBody()}
    return mock_client


def _make_mock_os_hit(
    chunk_id: str = "saber__chunk_0001",
    title: str = "Artoria Pendragon",
    text: str = "Saber is the Servant of Shirou Emiya.",
    source_url: str = "https://typemoon.fandom.com/wiki/Saber_(Fate/stay_night)",
    category: str = "servant",
    score: float = 0.92,
) -> dict:
    return {
        "_id": chunk_id,
        "_score": score,
        "_source": {
            "chunk_id": chunk_id,
            "text": text,
            "title": title,
            "source_url": source_url,
            "category": category,
            "chunk_index": 1,
        },
    }


def _make_mock_os_client(hits: list[dict] | None = None) -> MagicMock:
    """Return a mock OpenSearch client with a preset search response."""
    if hits is None:
        hits = [_make_mock_os_hit()]

    mock_client = MagicMock()
    mock_client.search.return_value = {
        "hits": {
            "total": {"value": len(hits)},
            "hits": hits,
        }
    }
    return mock_client


# ── FateRetriever tests ───────────────────────────────────────────────────────
class TestFateRetriever:
    def _make_retriever(self, hits=None, vector=None) -> FateRetriever:
        return FateRetriever(
            os_client=_make_mock_os_client(hits),
            bedrock_client=_make_mock_bedrock_client(vector),
            top_k=5,
        )

    def test_retrieve_returns_results(self):
        retriever = self._make_retriever()
        results = retriever.retrieve("Who is Saber?")
        assert len(results) == 1

    def test_retrieve_result_fields(self):
        retriever = self._make_retriever()
        results = retriever.retrieve("Who is Saber?")
        r = results[0]
        assert "text" in r
        assert "title" in r
        assert "source_url" in r
        assert "category" in r
        assert "score" in r
        assert "chunk_id" in r

    def test_retrieve_score_value(self):
        hit = _make_mock_os_hit(score=0.87)
        retriever = self._make_retriever(hits=[hit])
        results = retriever.retrieve("Noble Phantasm")
        assert results[0]["score"] == pytest.approx(0.87)

    def test_retrieve_correct_title(self):
        hit = _make_mock_os_hit(title="Gilgamesh")
        retriever = self._make_retriever(hits=[hit])
        results = retriever.retrieve("Who is Gilgamesh?")
        assert results[0]["title"] == "Gilgamesh"

    def test_retrieve_empty_results(self):
        retriever = self._make_retriever(hits=[])
        results = retriever.retrieve("Something very obscure")
        assert results == []

    def test_retrieve_multiple_results(self):
        hits = [
            _make_mock_os_hit(chunk_id=f"doc__chunk_{i:04d}", score=0.9 - i * 0.1)
            for i in range(5)
        ]
        retriever = self._make_retriever(hits=hits)
        results = retriever.retrieve("Holy Grail War")
        assert len(results) == 5

    def test_retrieve_calls_embed_query(self):
        mock_bedrock = _make_mock_bedrock_client()
        mock_os = _make_mock_os_client()
        retriever = FateRetriever(os_client=mock_os, bedrock_client=mock_bedrock)
        retriever.retrieve("Who is Rin Tohsaka?")
        mock_bedrock.invoke_model.assert_called_once()

    def test_retrieve_calls_opensearch_search(self):
        mock_bedrock = _make_mock_bedrock_client()
        mock_os = _make_mock_os_client()
        retriever = FateRetriever(os_client=mock_os, bedrock_client=mock_bedrock)
        retriever.retrieve("Tell me about Noble Phantasms")
        mock_os.search.assert_called_once()

    def test_category_filter_applied(self):
        """When category_filter is provided, post_filter should be in the query."""
        mock_bedrock = _make_mock_bedrock_client()
        mock_os = _make_mock_os_client()
        retriever = FateRetriever(os_client=mock_os, bedrock_client=mock_bedrock)
        retriever.retrieve("Saber Noble Phantasm", category_filter="servant")

        call_args = mock_os.search.call_args
        query_body = call_args[1]["body"] if "body" in call_args[1] else call_args[0][1]
        assert "post_filter" in query_body
        assert query_body["post_filter"]["term"]["category"] == "servant"

    def test_no_category_filter_no_post_filter(self):
        mock_bedrock = _make_mock_bedrock_client()
        mock_os = _make_mock_os_client()
        retriever = FateRetriever(os_client=mock_os, bedrock_client=mock_bedrock)
        retriever.retrieve("Tell me about the Fate series")

        call_args = mock_os.search.call_args
        query_body = call_args[1]["body"] if "body" in call_args[1] else call_args[0][1]
        assert "post_filter" not in query_body

    def test_top_k_override(self):
        """top_k passed to retrieve() overrides the instance default."""
        mock_bedrock = _make_mock_bedrock_client()
        mock_os = _make_mock_os_client()
        retriever = FateRetriever(os_client=mock_os, bedrock_client=mock_bedrock, top_k=5)
        retriever.retrieve("test query", top_k=3)

        call_args = mock_os.search.call_args
        query_body = call_args[1]["body"] if "body" in call_args[1] else call_args[0][1]
        assert query_body["size"] == 3
        assert query_body["query"]["knn"]["vector"]["k"] == 3

    def test_opensearch_error_raises(self):
        mock_bedrock = _make_mock_bedrock_client()
        mock_os = MagicMock()
        mock_os.search.side_effect = Exception("Connection refused")
        retriever = FateRetriever(os_client=mock_os, bedrock_client=mock_bedrock)
        with pytest.raises(Exception, match="Connection refused"):
            retriever.retrieve("test")

    def test_bedrock_error_raises(self):
        from botocore.exceptions import ClientError

        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}},
            "InvokeModel",
        )
        mock_os = _make_mock_os_client()
        retriever = FateRetriever(os_client=mock_os, bedrock_client=mock_bedrock)
        with pytest.raises(ClientError):
            retriever.retrieve("test")


# ── retrieve_multi_query tests ────────────────────────────────────────────────
class TestRetrieveMultiQuery:
    def test_deduplicates_by_chunk_id(self):
        """Same chunk returned from two queries should appear only once."""
        shared_hit = _make_mock_os_hit(chunk_id="shared__chunk_0000", score=0.9)
        mock_os = _make_mock_os_client([shared_hit])
        mock_bedrock = _make_mock_bedrock_client()
        retriever = FateRetriever(os_client=mock_os, bedrock_client=mock_bedrock)

        results = retriever.retrieve_multi_query(
            ["Who is Saber?", "Tell me about Artoria."], top_k=5
        )
        ids = [r["chunk_id"] for r in results]
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids found in multi-query results"

    def test_keeps_highest_score(self):
        """When same chunk appears twice with different scores, keep the higher one."""
        low_hit = _make_mock_os_hit(chunk_id="doc__chunk_0000", score=0.6)
        high_hit = _make_mock_os_hit(chunk_id="doc__chunk_0000", score=0.95)

        call_count = {"n": 0}
        mock_os = MagicMock()

        def side_effect(index, body):
            hit = high_hit if call_count["n"] == 0 else low_hit
            call_count["n"] += 1
            return {"hits": {"total": {"value": 1}, "hits": [hit]}}

        mock_os.search.side_effect = side_effect
        mock_bedrock = _make_mock_bedrock_client()
        retriever = FateRetriever(os_client=mock_os, bedrock_client=mock_bedrock)

        results = retriever.retrieve_multi_query(["query1", "query2"], top_k=5)
        assert results[0]["score"] == pytest.approx(0.95)

    def test_respects_top_k(self):
        hits = [
            _make_mock_os_hit(chunk_id=f"doc__chunk_{i:04d}", score=0.9 - i * 0.05)
            for i in range(10)
        ]
        mock_os = _make_mock_os_client(hits)
        mock_bedrock = _make_mock_bedrock_client()
        retriever = FateRetriever(os_client=mock_os, bedrock_client=mock_bedrock)

        results = retriever.retrieve_multi_query(["q1"], top_k=3)
        assert len(results) <= 3
