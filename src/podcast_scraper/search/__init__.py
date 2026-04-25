"""Semantic corpus search: vector index layer.

Re-exports were removed in favour of direct submodule imports (e.g.
``from podcast_scraper.search.faiss_store import FaissVectorStore``)
so importing ``podcast_scraper.search`` does not transitively pull in
numpy / faiss / sentence-transformers. Callers running with the
``[llm]`` extras only (``cloud_thin`` profile, ``vector_search: false``)
can still import this package as part of the ``gi`` chain without
``ModuleNotFoundError`` for numpy.

Direct-import paths (still public API):

  - ``from podcast_scraper.search.chunker import chunk_transcript, TranscriptChunk``
  - ``from podcast_scraper.search.faiss_store import FaissVectorStore``
  - ``from podcast_scraper.search.indexer import index_corpus, IndexRunStats, maybe_index_corpus``
  - ``from podcast_scraper.search.protocol import IndexStats, SearchResult, VectorStore``
  - ``from podcast_scraper.search.insight_clusters import build_insight_clusters_for_corpus``
  - ``from podcast_scraper.search.insight_cluster_context import expand_with_cluster_context``
"""
