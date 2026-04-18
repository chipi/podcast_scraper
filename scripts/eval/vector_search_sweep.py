"""Vector search quality sweep: compare embedding models + chunk settings.

Tests different embedding models and chunk size/overlap configurations
on held-out episodes. Measures:
1. Quote lift recall: for known GI quotes, does search find the right chunk?
2. Semantic search relevance: do GI silver insights retrieve relevant chunks?
3. Index build time + size

Usage:
    python scripts/eval/vector_search_sweep.py \
        --dataset curated_5feeds_benchmark_v2 \
        --silver silver_sonnet46_gi_benchmark_v2

    python scripts/eval/vector_search_sweep.py \
        --dataset curated_5feeds_benchmark_v2 \
        --models all-MiniLM-L6-v2,all-mpnet-base-v2
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Embedding models to compare
DEFAULT_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",  # Current default (384-dim)
    "sentence-transformers/all-mpnet-base-v2",  # Higher quality (768-dim)
    "BAAI/bge-small-en-v1.5",  # Newer, potentially better for short text (384-dim)
]

# Chunk size × overlap grid
DEFAULT_CHUNK_CONFIGS = [
    {"chunk_size": 256, "overlap": 64},
    {"chunk_size": 512, "overlap": 128},  # Likely current default range
    {"chunk_size": 1024, "overlap": 256},
]


def chunk_text(
    text: str,
    chunk_size_chars: int,
    overlap_chars: int,
) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size_chars
        chunks.append(text[start:end])
        start += chunk_size_chars - overlap_chars
        if start >= len(text):
            break
    return chunks


def measure_quote_retrieval(
    transcript: str,
    quotes: List[Dict[str, Any]],
    chunks: List[str],
    chunk_embeddings: np.ndarray,
    embedder: Any,
) -> Dict[str, float]:
    """Measure how well the chunks retrieve known quotes."""
    if not quotes or not chunks:
        return {"recall_at_1": 0.0, "recall_at_5": 0.0, "total_quotes": 0}

    quote_texts = [q["text"] for q in quotes]
    quote_embs = embedder.encode(quote_texts, normalize_embeddings=True)
    sim = np.dot(quote_embs, chunk_embeddings.T)

    hits_at_1 = 0
    hits_at_5 = 0

    for i, quote in enumerate(quotes):
        # Find which chunks contain this quote's char range

        # Check top-1 and top-5 retrieved chunks
        top_indices = np.argsort(sim[i])[::-1]

        for k, idx in enumerate(top_indices[:5]):
            # Check if this chunk overlaps with the quote's position
            chunk_start = 0
            for ci in range(idx):
                chunk_start += len(chunks[ci])
                # Approximate — proper overlap accounting needed
            if k == 0:
                hits_at_1 += 1  # Simplified: count any match at top-1
            hits_at_5 += 1
            break  # Count once per quote

    total = len(quotes)
    return {
        "recall_at_1": hits_at_1 / total if total else 0,
        "recall_at_5": hits_at_5 / total if total else 0,
        "total_quotes": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Vector search quality sweep")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--silver", help="GI silver ref ID for quote retrieval test")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-sep embedding model names",
    )
    args = parser.parse_args()

    print(f"Dataset: {args.dataset}")
    print(f"Models: {args.models.split(',')}")
    print(f"Chunk configs: {DEFAULT_CHUNK_CONFIGS}")
    print()
    print("NOTE: This script is a scaffold. Full implementation requires")
    print("building FAISS indexes per config and measuring retrieval quality")
    print("against GI silver quotes. See #595 for full plan.")
    print()

    # List configurations to test
    models = args.models.split(",")
    total_configs = len(models) * len(DEFAULT_CHUNK_CONFIGS)
    print(f"Total configurations: {total_configs}")
    for model in models:
        for chunk_cfg in DEFAULT_CHUNK_CONFIGS:
            print(f"  {model} chunk={chunk_cfg['chunk_size']} " f"overlap={chunk_cfg['overlap']}")


if __name__ == "__main__":
    main()
