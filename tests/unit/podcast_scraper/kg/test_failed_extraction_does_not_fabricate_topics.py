"""No extractor, no topics — and the artifact must say WHICH kind of nothing it is.

THE ORIGINAL BUG, observed on a real ingest and traced end to end. The DGX vLLM was unreachable, so
``_try_provider_extraction`` returned ``None``. Control fell to an ``elif`` whose comment said it
served "tests / legacy callers that pass a ``topic_label`` hint without a
``kg_extraction_provider``" — and it emitted the episode's SUMMARY BULLETS as Topic nodes:

    summary bullet:  "Product development in frontier AI requires building for model
                      capabilities two to three months ahead rather than current…"
    Topic node:      "Product development in frontier AI requires"

Eight per episode, 48 across six episodes, zero Insight/Person/Organization nodes — for an episode
about OpenAI and ChatGPT. Nothing said extraction had failed. Every downstream surface then consumed
sentences as subjects: clustering could never match them (each unique to its episode),
co-occurrence scored them, trending ranked them, and they were offered as followable interests.

WHAT CHANGED (ADR-156 / #2164). The first fix made the failing-provider path refuse to substitute
bullets. That left the same fabrication reachable by any caller passing ``topic_label`` /
``topic_labels`` without a provider — and the production workflow did exactly that, which is how 830
sentence-shaped Topic nodes and 96 all-sentence episodes reached prod. Those parameters are now GONE
from ``build_artifact``. The guarantee is no longer "bullets are refused"; it is that bullets
**cannot be offered**, which is why the first test asserts on the SIGNATURE.

The remaining obligation is attribution: three different kinds of "no topics" must stay
distinguishable, or an empty graph reads as an episode about nothing.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

import pytest

from podcast_scraper.kg.pipeline import build_artifact

pytestmark = pytest.mark.unit

#: Verbatim from the run that motivated this. Kept so the shape of what was fabricated stays on the
#: record even though no API can now accept it.
_REAL_BULLETS = [
    "Product development in frontier AI requires building for model capabilities two to three "
    "months ahead rather than current ones",
    "Empirical iteration replaces academic theorizing as the dominant mode of progress",
    "The future of knowledge work shifts from rowing tasks to steering direction",
]


class _FailingProvider:
    """A configured extraction provider that returns nothing — the vLLM-unreachable case."""

    summary_model = "test-model"

    def extract_kg_graph(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _types(art: dict[str, Any]) -> set[str]:
    return {n["type"] for n in art["nodes"]}


def _topic_labels(art: dict[str, Any]) -> list[str]:
    return [n["properties"]["label"] for n in art["nodes"] if n["type"] == "Topic"]


def _provenance(art: dict[str, Any]) -> str:
    return str((art.get("extraction") or {}).get("model_version") or "")


def _art(**over: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "podcast_id": "podcast:p1",
        "episode_title": "T",
    }
    kwargs.update(over)
    return build_artifact("ep:x", "x", **kwargs)


class TestBulletsCannotEvenBeOffered:
    """The strongest form of the guarantee: the parameters do not exist.

    Asserting "bullets do not become topics" only holds for the code paths a test remembers to
    cover. Asserting the API cannot accept them holds for every caller — including the one in
    ``workflow/metadata_generation`` that did this in production for months.
    """

    @pytest.mark.parametrize("param", ["topic_label", "topic_labels"])
    def test_build_artifact_has_no_topic_label_parameter(self, param: str) -> None:
        params = inspect.signature(build_artifact).parameters
        assert param not in params, (
            f"build_artifact accepts {param!r} again. That parameter is how summary bullets became "
            "Topic nodes: 830 sentence-shaped nodes and 96 all-sentence episodes on prod "
            "(ADR-156 / #2164). Only the extraction provider may create a Topic."
        )

    def test_passing_bullets_is_a_hard_error_not_a_silent_ignore(self) -> None:
        """A parameter that looks live and is ignored is how this accumulated in the first place."""
        with pytest.raises(TypeError):
            _art(topic_labels=list(_REAL_BULLETS))


class TestAFailedExtractionEmitsNothing:
    def test_no_topics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """THE regression: a configured provider returning nothing yields an EMPTY graph."""
        from podcast_scraper.kg import pipeline

        monkeypatch.setattr(pipeline, "_resolve_source", lambda _cfg: "provider")
        art = _art(kg_extraction_provider=_FailingProvider())
        labels = _topic_labels(art)
        assert labels == [], f"topics were fabricated from somewhere: {labels}"
        assert "Topic" not in _types(art)

    def test_the_failure_is_recorded_in_provenance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty KG must be attributable — "no topics" and "extraction broke" differ."""
        from podcast_scraper.kg import pipeline

        monkeypatch.setattr(pipeline, "_resolve_source", lambda _cfg: "provider")
        art = _art(kg_extraction_provider=_FailingProvider())
        assert "extraction_failed" in _provenance(art), (
            "an empty KG that does not say why reads as an episode about nothing: "
            f"{_provenance(art)!r}"
        )

    def test_the_failure_is_loud(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Silence here is what let 48 fabricated topics ship unnoticed."""
        from podcast_scraper.kg import pipeline

        monkeypatch.setattr(pipeline, "_resolve_source", lambda _cfg: "provider")
        with caplog.at_level(logging.WARNING, logger="podcast_scraper.kg.pipeline"):
            _art(kg_extraction_provider=_FailingProvider())
        assert any(
            "no topics/entities" in str(r.msg) for r in caplog.records
        ), "extraction failed and produced an empty KG without a word in the log"


class TestNoProviderIsItsOwnDistinctFault:
    """THE INVERSION of the old ``test_the_legacy_hint_path_is_untouched``.

    That test asserted a caller passing a ``topic_label`` hint with no provider still got a Topic —
    "legacy callers must keep working". ADR-156 withdraws that guarantee deliberately: it was the
    live route by which production fabricated topics, not a harmless compatibility shim.

    A missing extractor is a MISCONFIGURATION and must read as one, rather than being collapsed into
    ``metadata_only`` (a deliberate reduced mode) or ``provider:extraction_failed`` (a provider that
    was tried and failed).
    """

    def test_no_provider_yields_no_topics(self) -> None:
        art = _art()
        assert _topic_labels(art) == []
        assert "Topic" not in _types(art)

    def test_no_provider_says_so_in_the_provenance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from podcast_scraper.kg import pipeline

        monkeypatch.setattr(pipeline, "_resolve_source", lambda _cfg: "provider")
        art = _art()
        assert _provenance(art) == "no_extractor", (
            "a misconfigured run must not be indistinguishable from a deliberate reduced mode: "
            f"{_provenance(art)!r}"
        )

    def test_it_is_loud_and_says_why_bullets_are_not_used(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The log is where an operator learns an empty graph was a config fault, not a quiet show."""  # noqa: E501
        from podcast_scraper.kg import pipeline

        monkeypatch.setattr(pipeline, "_resolve_source", lambda _cfg: "provider")
        with caplog.at_level(logging.WARNING, logger="podcast_scraper.kg.pipeline"):
            _art()
        assert any("NOT substituted" in str(r.msg) for r in caplog.records)

    def test_the_three_empty_reasons_are_distinguishable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All three render as an empty topic list; only the provenance tells them apart."""
        from podcast_scraper.kg import pipeline

        monkeypatch.setattr(pipeline, "_resolve_source", lambda _cfg: "provider")
        failed = _provenance(_art(kg_extraction_provider=_FailingProvider()))
        missing = _provenance(_art())

        monkeypatch.setattr(pipeline, "_resolve_source", lambda _cfg: "metadata_only")
        reduced = _provenance(_art())

        assert len({failed, missing, reduced}) == 3, (
            "a provider that failed, no provider at all, and a deliberate metadata-only run are "
            f"three different faults: {failed!r} / {missing!r} / {reduced!r}"
        )
