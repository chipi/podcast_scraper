"""Semantic corpus search: vector index layer (PRD-021 / RFC-061 / GitHub #484)."""

from podcast_scraper.search.chunker import chunk_transcript, TranscriptChunk
from podcast_scraper.search.faiss_store import FaissVectorStore
from podcast_scraper.search.indexer import index_corpus, IndexRunStats, maybe_index_corpus
from podcast_scraper.search.protocol import IndexStats, SearchResult, VectorStore

__all__ = [
    "FaissVectorStore",
    "IndexRunStats",
    "IndexStats",
    "SearchResult",
    "TranscriptChunk",
    "VectorStore",
    "chunk_transcript",
    "index_corpus",
    "maybe_index_corpus",
    # Insight clustering (#599, #601) — import from submodules directly:
    #   from podcast_scraper.search.insight_clusters import build_insight_clusters_for_corpus
    #   from podcast_scraper.search.insight_cluster_context import expand_with_cluster_context
]
