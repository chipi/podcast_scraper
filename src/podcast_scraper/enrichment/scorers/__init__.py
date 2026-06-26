"""Scorer protocols — injectable backends for embedding / ml / llm enrichers.

These are the seam between the enrichment-layer framework and the
backend implementations:

* ``NliScorer`` — local DeBERTa (chunk 4) + scenario-driven mock (chunk 1 fixtures).
* ``EmbeddingProvider`` — LanceDB-backed (chunk 3) + scenario mock (chunk 1).
* ``LLMScorer`` — for future LLM-tier query enrichers (follow-on RFC).

Scenario-driven mocks under ``tests/fixtures/enrichment/`` exercise
the resilience pipeline (retry, circuit-breaker, auto-disable, cancel,
heartbeat) without real models in CI.
"""

from __future__ import annotations
