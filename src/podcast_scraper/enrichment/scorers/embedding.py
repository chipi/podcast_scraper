"""Concrete ``EmbeddingProvider`` implementations.

Two shipped:

* :class:`TopicEmbeddingProvider` — a thin in-memory cache around a
  user-supplied ``embed_text(text) -> list[float]`` callable. Used by
  :mod:`podcast_scraper.enrichment.enrichers.topic_similarity` in
  production; the operator wires in a real
  ``sentence-transformers.SentenceTransformer.encode`` or any other
  text → vector function.
* :class:`HashEmbedder` — a deterministic, dependency-free fallback
  embedder built on top of ``hashlib``. Used by tests + CI smoke runs
  so the resilience pipeline can exercise the real provider without
  downloading model weights ([[feedback_no_llm_in_ci]]).

The mock :class:`MockEmbeddingProvider` from
``tests/fixtures/enrichment/mock_scorers.py`` covers failure
scenarios (retries, timeout, missing topic_id) — these production
providers focus on success-path semantics.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Mapping


def _safe_topic_label(topic_id: str, labels: Mapping[str, str]) -> str:
    """Look up the human label for a topic_id; fall back to the id stem."""
    label = labels.get(topic_id)
    if label:
        return label
    if ":" in topic_id:
        return topic_id.split(":", 1)[-1]
    return topic_id


@dataclass
class TopicEmbeddingProvider:
    """Production-shape ``EmbeddingProvider``.

    Accepts an injected ``embed_text(text) -> list[float]`` callable so
    the operator can plug in sentence-transformers, an Ollama
    embedding endpoint, or any other backend. Caches per-topic_id
    vectors in-memory across a single run (the executor constructs one
    instance per ``enrich()`` call).

    Missing topic_ids return ``None`` — the chunk-1 ``EmbeddingProvider``
    protocol contract.
    """

    embed_text: Callable[[str], list[float]]
    labels: Mapping[str, str] = field(default_factory=dict)
    # RFC-118: identity of the embedding backend (e.g. the sentence-transformers model
    # id). The topic_similarity persistent vector cache reuses a vector only when the
    # marker matches AND is non-empty — an unmarked provider fail-safes to re-embed.
    model_marker: str = ""
    # #1818: optional batch encoder (texts -> vectors, one call). When set,
    # ``topic_vectors`` embeds all missing topics in one backend call instead of
    # one call per topic — the cold-baseline dominator at corpus scale.
    embed_texts: Callable[[list[str]], list[list[float]]] | None = None
    _cache: dict[str, list[float] | None] = field(default_factory=dict, init=False, repr=False)

    async def topic_vector(self, topic_id: str) -> list[float] | None:
        """EmbeddingProvider.topic_vector impl — sync embed_text via to_thread."""
        if topic_id in self._cache:
            return self._cache[topic_id]
        label = _safe_topic_label(topic_id, self.labels)
        if not label:
            self._cache[topic_id] = None
            return None
        vector = await asyncio.to_thread(self.embed_text, label)
        if not vector:
            self._cache[topic_id] = None
            return None
        self._cache[topic_id] = list(vector)
        return self._cache[topic_id]

    async def topic_vectors(self, topic_ids: list[str]) -> dict[str, list[float] | None]:
        """Batch variant (#1818): one ``embed_texts`` call for every uncached topic.

        Falls back to per-topic ``topic_vector`` calls when no batch encoder is
        wired, so failure-injection test providers and API backends keep their
        per-call semantics.
        """
        out: dict[str, list[float] | None] = {}
        missing: list[str] = []
        for tid in topic_ids:
            if tid in self._cache:
                out[tid] = self._cache[tid]
            else:
                missing.append(tid)
        if not missing:
            return out
        if self.embed_texts is None:
            for tid in missing:
                out[tid] = await self.topic_vector(tid)
            return out
        labels = [_safe_topic_label(tid, self.labels) for tid in missing]
        embeddable = [(tid, lbl) for tid, lbl in zip(missing, labels) if lbl]
        for tid, lbl in zip(missing, labels):
            if not lbl:
                self._cache[tid] = None
                out[tid] = None
        if embeddable:
            vectors = await asyncio.to_thread(self.embed_texts, [lbl for _, lbl in embeddable])
            for (tid, _), vec in zip(embeddable, vectors):
                self._cache[tid] = list(vec) if vec else None
                out[tid] = self._cache[tid]
        return out


@dataclass
class AsyncTopicEmbeddingProvider:
    """Same shape but for backends that are already async (e.g. HTTP)."""

    embed_text: Callable[[str], Awaitable[list[float]]]
    labels: Mapping[str, str] = field(default_factory=dict)
    _cache: dict[str, list[float] | None] = field(default_factory=dict, init=False, repr=False)

    async def topic_vector(self, topic_id: str) -> list[float] | None:
        """EmbeddingProvider.topic_vector impl — awaitable embed_text directly."""
        if topic_id in self._cache:
            return self._cache[topic_id]
        label = _safe_topic_label(topic_id, self.labels)
        if not label:
            self._cache[topic_id] = None
            return None
        vector = await self.embed_text(label)
        if not vector:
            self._cache[topic_id] = None
            return None
        self._cache[topic_id] = list(vector)
        return self._cache[topic_id]


class HashEmbedder:
    """Deterministic, dependency-free embedder for tests + CI smoke.

    Maps each input text to a fixed-dim vector by hashing the text with
    SHA-256 and projecting bytes into ``[-1, 1]`` floats. Stable across
    runs (no randomness), so similarity tests are deterministic. Texts
    that share a prefix produce dissimilar vectors — there's no
    semantic signal, but the resilience-and-shape path is fully
    exercised.
    """

    def __init__(self, *, dim: int = 32) -> None:
        if dim < 4 or dim > 1024:
            raise ValueError("HashEmbedder dim must be in [4, 1024]")
        self.dim = dim

    def __call__(self, text: str) -> list[float]:
        if not text:
            return [0.0] * self.dim
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Repeat the digest until we have enough bytes for the requested dim.
        out: list[float] = []
        i = 0
        while len(out) < self.dim:
            byte = digest[i % len(digest)]
            out.append((byte - 127.5) / 127.5)  # map [0, 255] → [-1, 1]
            i += 1
        return out


__all__ = [
    "AsyncTopicEmbeddingProvider",
    "HashEmbedder",
    "TopicEmbeddingProvider",
]
