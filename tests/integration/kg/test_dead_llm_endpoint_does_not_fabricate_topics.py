"""A dead LLM endpoint must degrade honestly — never into fabricated topics.

WHY THIS LAYER EXISTS. There is already a chaos target for a DGX outage,
``make preprod-chaos-dgx-down``, and it would have passed throughout the incident this test is
written against. It only reroutes ``--dgx-whisper-port`` and asserts cloud Whisper takes over: ASR
failover is covered and the **LLM stages are never exercised under failure at all**. So "DGX down"
was considered tested while the KG path silently fabricated topics. It is also a make target,
which means operator-run and never blocking a merge.

THE INCIDENT. The autoresearch vLLM on ``dgx-llm-1:8003`` is GPU-mode-gated (DGX_SERVING.md: "when
the mode is free/idle, :8003 serves nothing — that is idle, not gone"). With it idle:

  * ``OpenAICompatibleProvider.extract_kg_graph`` swallowed the connection error and returned
    ``None`` — a dead endpoint made indistinguishable from "the model found no topics";
  * ``FallbackAwareSummarizationProvider._wrap_call`` walks the chain only on an EXCEPTION, so the
    ``None`` sailed through and a healthy ollama tier was never tried;
  * ``build_artifact`` then substituted the episode's summary BULLETS as Topic nodes.

Result: 8 fabricated sentence-topics per episode, 0 Insight / Person / Organization nodes, and
metrics reading ``llm_kg_calls=0, kg_failures=0`` — a total failure recording itself as a clean run.

These tests drive the real ``build_artifact`` with a provider that fails the way the real one does,
so the whole chain is covered rather than each half separately.
"""

from __future__ import annotations

from typing import Any

import pytest

from podcast_scraper.kg.pipeline import build_artifact

pytestmark = pytest.mark.integration

#: Real bullets from the incident — sentences, which is what makes substitution so damaging.
_BULLETS = [
    "Product development in frontier AI requires building for model capabilities two to three "
    "months ahead rather than current ones",
    "Empirical iteration replaces academic theorizing as the dominant mode of progress",
]


class _DeadEndpointProvider:
    """Behaves exactly like OpenAICompatibleProvider against a refused connection.

    The catch-all is the point: it does NOT raise, it returns None. A test that raised here would
    pass against the buggy code, because the failover wrapper handles exceptions correctly — the
    entire defect is the silent return.
    """

    summary_model = "vllm-primary"

    def __init__(self) -> None:
        self.primary_calls = 0

    def extract_kg_graph(self, *_a: Any, **_kw: Any) -> None:
        self.primary_calls += 1
        try:
            raise ConnectionRefusedError("dgx-llm-1:8003 refused")
        except Exception:
            return None  # mirrors the real `except Exception: return None`


class _DeadEndpointWithHealthyFallback(_DeadEndpointProvider):
    """…and a fallback chain that CAN answer, as the ollama tier could have."""

    def __init__(self) -> None:
        super().__init__()
        self.fallback_calls = 0

    def call_via_fallback(self, method_name: str, *_a: Any, **_kw: Any) -> Any:
        assert method_name == "extract_kg_graph"
        self.fallback_calls += 1
        return {"topics": [{"label": "ai regulation"}], "entities": []}


def _artifact(provider: Any, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from podcast_scraper.kg import pipeline

    monkeypatch.setattr(pipeline, "_resolve_source", lambda _cfg: "provider")
    return build_artifact(
        "ep:x",
        "transcript text",
        podcast_id="podcast:p1",
        episode_title="AI's third era",
        topic_labels=list(_BULLETS),
        kg_extraction_provider=provider,
    )


def _topics(art: dict[str, Any]) -> list[str]:
    return [n["properties"]["label"] for n in art["nodes"] if n["type"] == "Topic"]


def test_a_dead_endpoint_reaches_the_fallback_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """The chain must be tried even though the primary never raised."""
    provider = _DeadEndpointWithHealthyFallback()
    art = _artifact(provider, monkeypatch)
    assert provider.primary_calls == 1
    assert provider.fallback_calls == 1, (
        "a healthy fallback tier sat unused while the primary returned a silent None — this is "
        "the incident exactly"
    )
    assert _topics(art) == ["ai regulation"]


def test_a_dead_endpoint_with_no_chain_fabricates_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE regression. No chain available → an empty KG, never bullets-as-topics."""
    art = _artifact(_DeadEndpointProvider(), monkeypatch)
    topics = _topics(art)
    assert topics == [], (
        f"summary bullets were fabricated into Topic nodes: {topics}. Each is a sentence unique "
        "to its episode, so it can never cluster, and downstream it poisons co-occurrence, "
        "trending and the storyline surfaces."
    )


def test_the_empty_artifact_says_why(monkeypatch: pytest.MonkeyPatch) -> None:
    """"no topics" and "extraction failed" must not look identical to an operator."""
    art = _artifact(_DeadEndpointProvider(), monkeypatch)
    provenance = str((art.get("extraction") or {}).get("model_version") or "")
    assert "extraction_failed" in provenance, f"provenance hides the failure: {provenance!r}"


def test_nothing_else_is_invented_either(monkeypatch: pytest.MonkeyPatch) -> None:
    """The incident produced 0 entities; the fix must not paper over that with placeholders."""
    art = _artifact(_DeadEndpointProvider(), monkeypatch)
    types = {n["type"] for n in art["nodes"]}
    assert "Topic" not in types
    assert "Organization" not in types
    assert "Episode" in types  # structural, not extracted — legitimately remains
