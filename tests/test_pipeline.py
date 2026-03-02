"""
Unit tests for the data pipeline (chunker).
Tests chunk size, overlap, metadata preservation, and edge cases.
"""

import json
import tempfile
from pathlib import Path

import pytest
import tiktoken

from data_pipeline.chunker import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    _get_encoder,
    _split_into_chunks,
    chunk_document,
    chunk_documents,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture
def encoder() -> tiktoken.Encoding:
    return _get_encoder()


@pytest.fixture
def sample_doc() -> dict:
    return {
        "title": "Artoria Pendragon",
        "url": "https://typemoon.fandom.com/wiki/Saber_(Fate/stay_night)",
        "slug": "Saber_(Fate/stay_night)",
        "category": "servant",
        "content": (
            "Artoria Pendragon, also known as Saber, is the Servant of Shirou Emiya "
            "in the Fifth Holy Grail War. She is the legendary King of Knights who "
            "pulled the sword Caliburn from the stone and was later given Excalibur "
            "by the Lady of the Lake. As a Heroic Spirit, she has been summoned "
            "multiple times across different Holy Grail Wars. Her Noble Phantasm, "
            "Excalibur, is a sword of promised victory that releases stored energy "
            "as a devastating beam of light. She is widely regarded as one of the "
            "most powerful Servants in the Fate universe. "
        ) * 20,  # Repeat to ensure multiple chunks
        "scraped_at": "2024-01-01T00:00:00+00:00",
    }


@pytest.fixture
def minimal_doc() -> dict:
    return {
        "title": "Test Servant",
        "url": "https://example.com/test",
        "slug": "test-servant",
        "category": "servant",
        "content": "Short content.",
        "scraped_at": "2024-01-01T00:00:00+00:00",
    }


# ── _split_into_chunks tests ──────────────────────────────────────────────────
class TestSplitIntoChunks:
    def test_short_text_single_chunk(self, encoder):
        text = "Hello, world."
        chunks = _split_into_chunks(text, encoder, chunk_size=500, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_produces_multiple_chunks(self, encoder):
        # Create text with ~1200 tokens
        word = "fate " * 300  # ~300 tokens (rough)
        long_text = word * 4
        chunks = _split_into_chunks(long_text, encoder, chunk_size=500, overlap=50)
        assert len(chunks) > 1

    def test_chunk_size_respected(self, encoder):
        word = "fate " * 300
        long_text = word * 4
        chunk_size = 200
        chunks = _split_into_chunks(long_text, encoder, chunk_size=chunk_size, overlap=20)
        for chunk in chunks[:-1]:  # All but last should be at or near chunk_size
            token_count = len(encoder.encode(chunk))
            assert token_count <= chunk_size

    def test_overlap_creates_shared_content(self, encoder):
        """Adjacent chunks should share tokens due to overlap."""
        # Generate a predictable sequence of words
        words = [f"word{i}" for i in range(600)]
        text = " ".join(words)
        chunks = _split_into_chunks(text, encoder, chunk_size=100, overlap=20)

        assert len(chunks) >= 2, "Need at least 2 chunks to test overlap"

        # Encode both chunks and verify shared suffix/prefix tokens
        tokens_0 = encoder.encode(chunks[0])
        tokens_1 = encoder.encode(chunks[1])

        # The tail of chunk 0 should appear in chunk 1's head
        tail = tokens_0[-20:]
        head = tokens_1[:30]
        shared = set(tail) & set(head)
        assert len(shared) > 0, "Expected overlapping tokens between adjacent chunks"

    def test_empty_text(self, encoder):
        chunks = _split_into_chunks("", encoder, chunk_size=500, overlap=50)
        assert chunks == []

    def test_single_token_text(self, encoder):
        chunks = _split_into_chunks("Fate", encoder, chunk_size=500, overlap=50)
        assert len(chunks) == 1
        assert "Fate" in chunks[0]

    def test_no_trailing_empty_chunk(self, encoder):
        text = "word " * 100
        chunks = _split_into_chunks(text, encoder, chunk_size=50, overlap=5)
        for chunk in chunks:
            assert chunk.strip() != ""


# ── chunk_document tests ──────────────────────────────────────────────────────
class TestChunkDocument:
    def test_returns_chunks_for_normal_doc(self, sample_doc, encoder):
        chunks = chunk_document(sample_doc, encoder)
        assert len(chunks) > 0

    def test_metadata_preserved(self, sample_doc, encoder):
        chunks = chunk_document(sample_doc, encoder)
        for chunk in chunks:
            assert chunk["source_url"] == sample_doc["url"]
            assert chunk["title"] == sample_doc["title"]
            assert chunk["category"] == sample_doc["category"]
            assert chunk["slug"] == sample_doc["slug"]
            assert chunk["scraped_at"] == sample_doc["scraped_at"]

    def test_chunk_id_unique(self, sample_doc, encoder):
        chunks = chunk_document(sample_doc, encoder)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunk_id_format(self, sample_doc, encoder):
        chunks = chunk_document(sample_doc, encoder)
        for i, chunk in enumerate(chunks):
            expected_id = f"{sample_doc['slug']}__chunk_{i:04d}"
            assert chunk["chunk_id"] == expected_id

    def test_chunk_index_sequential(self, sample_doc, encoder):
        chunks = chunk_document(sample_doc, encoder)
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i

    def test_total_chunks_consistent(self, sample_doc, encoder):
        chunks = chunk_document(sample_doc, encoder)
        total = len(chunks)
        for chunk in chunks:
            assert chunk["total_chunks"] == total

    def test_token_count_field(self, sample_doc, encoder):
        chunks = chunk_document(sample_doc, encoder)
        for chunk in chunks:
            actual = len(encoder.encode(chunk["text"]))
            assert chunk["token_count"] == actual

    def test_empty_content_returns_empty(self, encoder):
        doc = {"title": "Empty", "url": "", "slug": "empty", "category": "lore", "content": ""}
        chunks = chunk_document(doc, encoder)
        assert chunks == []

    def test_whitespace_only_content_returns_empty(self, encoder):
        doc = {"title": "Whitespace", "url": "", "slug": "ws", "category": "lore", "content": "   \n\t  "}
        chunks = chunk_document(doc, encoder)
        assert chunks == []

    def test_minimal_doc(self, minimal_doc, encoder):
        chunks = chunk_document(minimal_doc, encoder)
        assert len(chunks) == 1
        assert chunks[0]["text"] == minimal_doc["content"]


# ── chunk_documents integration test ──────────────────────────────────────────
class TestChunkDocuments:
    def test_reads_json_and_produces_output(self, sample_doc, minimal_doc):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_file = tmp / "docs.json"
            output_dir = tmp / "chunks"

            with open(input_file, "w") as f:
                json.dump([sample_doc, minimal_doc], f)

            chunks = chunk_documents(input_path=input_file, output_dir=output_dir)

            assert len(chunks) > 0
            assert (output_dir / "all_chunks.json").exists()

            # Verify per-category files created
            assert (output_dir / "servant_chunks.json").exists()

    def test_combined_output_contains_all_chunks(self, sample_doc):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_file = tmp / "docs.json"
            output_dir = tmp / "chunks"

            with open(input_file, "w") as f:
                json.dump([sample_doc], f)

            chunks = chunk_documents(input_path=input_file, output_dir=output_dir)

            with open(output_dir / "all_chunks.json") as f:
                saved = json.load(f)

            assert len(saved) == len(chunks)

    def test_default_chunk_size_used(self, sample_doc):
        """Verify that default CHUNK_SIZE constant is applied."""
        encoder = _get_encoder()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_file = tmp / "docs.json"
            output_dir = tmp / "chunks"

            with open(input_file, "w") as f:
                json.dump([sample_doc], f)

            chunks = chunk_documents(input_path=input_file, output_dir=output_dir)

            # No chunk should exceed the configured chunk size
            for chunk in chunks[:-1]:  # last chunk can be smaller
                assert chunk["token_count"] <= CHUNK_SIZE
