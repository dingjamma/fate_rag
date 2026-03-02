"""
Text Chunker for the Fate RAG Pipeline
Splits documents into ~500 token chunks with 50 token overlap,
preserving metadata for downstream embedding and retrieval.
"""

import json
import logging
from pathlib import Path
from typing import Any

import tiktoken

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")
CHUNKS_DATA_DIR = Path("data/chunks")

CHUNK_SIZE = 500        # target tokens per chunk
CHUNK_OVERLAP = 50      # overlap tokens between adjacent chunks
ENCODING_MODEL = "cl100k_base"  # matches GPT-4 / Titan tokenizer closely


def _get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING_MODEL)


def _split_into_chunks(
    text: str,
    encoder: tiktoken.Encoding,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Tokenize text, then split into overlapping chunks.
    Each chunk is re-decoded back to a string.
    """
    tokens = encoder.encode(text)
    chunks: list[str] = []

    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoder.decode(chunk_tokens)
        chunks.append(chunk_text)

        if end == len(tokens):
            break  # last chunk reached

        start += chunk_size - overlap  # slide window with overlap

    return chunks


def chunk_document(doc: dict[str, Any], encoder: tiktoken.Encoding) -> list[dict[str, Any]]:
    """
    Chunk a single document and attach metadata to each chunk.

    Returns a list of chunk dicts, each containing:
      - chunk_id: unique identifier (slug + index)
      - text: chunk content
      - token_count: number of tokens in the chunk
      - chunk_index: position within the parent document
      - total_chunks: total number of chunks from this doc
      - source_url, title, category, scraped_at: inherited from doc
    """
    content = doc.get("content", "")
    if not content.strip():
        logger.warning(f"Skipping empty document: {doc.get('title', 'unknown')}")
        return []

    raw_chunks = _split_into_chunks(content, encoder)
    total = len(raw_chunks)

    result: list[dict[str, Any]] = []
    for idx, chunk_text in enumerate(raw_chunks):
        token_count = len(encoder.encode(chunk_text))
        result.append(
            {
                "chunk_id": f"{doc['slug']}__chunk_{idx:04d}",
                "chunk_index": idx,
                "total_chunks": total,
                "text": chunk_text,
                "token_count": token_count,
                "source_url": doc.get("url", ""),
                "title": doc.get("title", ""),
                "slug": doc.get("slug", ""),
                "category": doc.get("category", "unknown"),
                "scraped_at": doc.get("scraped_at", ""),
            }
        )

    logger.debug(f"  '{doc['title']}' → {total} chunks (avg {sum(len(encoder.encode(c)) for c in raw_chunks) // max(total, 1)} tokens)")
    return result


def chunk_documents(
    input_path: Path = RAW_DATA_DIR / "all_documents.json",
    output_dir: Path = CHUNKS_DATA_DIR,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Load raw documents, chunk them, and save results.

    Args:
        input_path: Path to the combined raw JSON (or a single-category file).
        output_dir: Directory to write chunk JSON files.
        chunk_size: Max tokens per chunk.
        overlap: Token overlap between consecutive chunks.

    Returns:
        All chunks as a flat list.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = _get_encoder()

    logger.info(f"Loading documents from {input_path}")
    with open(input_path, encoding="utf-8") as f:
        documents: list[dict] = json.load(f)

    logger.info(f"Chunking {len(documents)} documents (chunk_size={chunk_size}, overlap={overlap})")

    all_chunks: list[dict[str, Any]] = []
    category_chunks: dict[str, list[dict[str, Any]]] = {}

    for doc in documents:
        chunks = chunk_document(doc, encoder)
        all_chunks.extend(chunks)
        cat = doc.get("category", "unknown")
        category_chunks.setdefault(cat, []).extend(chunks)

    # Write per-category chunk files
    for category, chunks in category_chunks.items():
        cat_file = output_dir / f"{category}_chunks.json"
        with open(cat_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"  [{category}] {len(chunks)} chunks → {cat_file}")

    # Write combined chunk file
    combined_file = output_dir / "all_chunks.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"Total chunks: {len(all_chunks)} → {combined_file}")
    return all_chunks


def chunk_from_sample_data(
    sample_dir: Path = Path("sample_data"),
    output_dir: Path = CHUNKS_DATA_DIR,
) -> list[dict[str, Any]]:
    """
    Convenience function to chunk sample JSON files for local testing
    without running the full scraper.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = _get_encoder()

    all_chunks: list[dict[str, Any]] = []
    sample_files = list(sample_dir.glob("*.json"))

    if not sample_files:
        logger.warning(f"No JSON files found in {sample_dir}")
        return []

    for sample_file in sample_files:
        logger.info(f"Processing sample file: {sample_file}")
        with open(sample_file, encoding="utf-8") as f:
            docs = json.load(f)
        if isinstance(docs, dict):
            docs = [docs]
        for doc in docs:
            all_chunks.extend(chunk_document(doc, encoder))

    combined_file = output_dir / "all_chunks.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"Sample chunking complete: {len(all_chunks)} chunks → {combined_file}")
    return all_chunks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk Fate lore documents")
    parser.add_argument("--input", default=str(RAW_DATA_DIR / "all_documents.json"))
    parser.add_argument("--output", default=str(CHUNKS_DATA_DIR))
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample_data/ instead of scraped data",
    )
    args = parser.parse_args()

    if args.sample:
        chunks = chunk_from_sample_data(output_dir=Path(args.output))
    else:
        chunks = chunk_documents(
            input_path=Path(args.input),
            output_dir=Path(args.output),
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )

    print(f"\nChunking complete. {len(chunks)} chunks created.")
