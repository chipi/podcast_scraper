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
    _cache: dict[str, list[float] | None] = field(default_factory=dict, init=False, repr=False)

    async def topic_vector(self, topic_id: str) -> list[float] | None:
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


@dataclass
class AsyncTopicEmbeddingProvider:
    """Same shape but for backends that are already async (e.g. HTTP)."""

    embed_text: Callable[[str], Awaitable[list[float]]]
    labels: Mapping[str, str] = field(default_factory=dict)
    _cache: dict[str, list[float] | None] = field(default_factory=dict, init=False, repr=False)

    async def topic_vector(self, topic_id: str) -> list[float] | None:
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
