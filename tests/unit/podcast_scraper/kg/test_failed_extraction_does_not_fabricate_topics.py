"""When KG extraction FAILS, the artifact must be empty — never filled with summary bullets.

THE BUG, observed on a real ingest and traced end to end. The DGX vLLM was unreachable, so
``_try_provider_extraction`` returned ``None``. Control fell to an ``elif`` whose own comment says
it serves "tests / legacy callers that pass a ``topic_label`` hint without a
``kg_extraction_provider``" — and it emitted the episode's SUMMARY BULLETS as Topic nodes.

The result was not a degraded knowledge graph. It was a fabricated one:

    summary bullet:  "Product development in frontier AI requires building for model
                      capabilities two to three months ahead rather than current…"
    Topic node:      "Product development in frontier AI requires"

Eight of those per episode, 48 across six episodes, zero Insight nodes, zero Person nodes, zero
Organization nodes — for an episode about OpenAI and ChatGPT. Nothing anywhere said extraction had
failed. Every downstream surface then consumed sentences as subjects: clustering could never match
them (each is unique to its episode), co-occurrence scored them, trending ranked them, and they
were offered to listeners as followable interests.

An empty topic set is the honest outcome. The artifact records ``provider:extraction_failed`` as
its provenance, and every consumer correctly sees nothing rather than being poisoned.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from podcast_scraper.kg.pipeline import build_artifact

pytestmark = pytest.mark.unit

#: Verbatim from the run that motivated this.
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


def test_a_failed_extraction_emits_no_topics(monkeypatch: pytest.MonkeyPatch) -> None:
    """THE regression. Bullets must not become topics when extraction was attempted and failed."""
    from podcast_scraper.kg import pipeline

    monkeypatch.setattr(pipeline, "_resolve_source", lambda _cfg: "provider")
    art = build_artifact(
        "ep:x",
        "x",
        podcast_id="podcast:p1",
        episode_title="T",
        topic_labels=list(_REAL_BULLETS),
        kg_extraction_provider=_FailingProvider(),
    )
    labels = _topic_labels(art)
    assert labels == [], (
        f"summary bullets were fabricated into Topic nodes: {labels}. Every one is a sentence "
        "unique to this episode; downstream they poison clustering, co-occurrence and trending."
    )
    assert "Topic" not in _types(art)


def test_the_failure_is_recorded_in_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty KG must be attributable — "no topics" and "extraction broke" differ."""
    from podcast_scraper.kg import pipeline

    monkeypatch.setattr(pipeline, "_resolve_source", lambda _cfg: "provider")
    art = build_artifact(
        "ep:x",
        "x",
        podcast_id="podcast:p1",
        episode_title="T",
        topic_labels=list(_REAL_BULLETS),
        kg_extraction_provider=_FailingProvider(),
    )
    provenance = str((art.get("extraction") or {}).get("model_version") or "")
    assert (
        "extraction_failed" in provenance
    ), f"an empty KG that does not say why reads as an episode about nothing: {provenance!r}"


def test_the_failure_is_loud(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Silence here is what let 48 fabricated topics ship unnoticed."""
    from podcast_scraper.kg import pipeline

    monkeypatch.setattr(pipeline, "_resolve_source", lambda _cfg: "provider")
    with caplog.at_level(logging.WARNING, logger="podcast_scraper.kg.pipeline"):
        build_artifact(
            "ep:x",
            "x",
            podcast_id="podcast:p1",
            episode_title="T",
            topic_labels=list(_REAL_BULLETS),
            kg_extraction_provider=_FailingProvider(),
        )
    assert any(
        "NOT substituting" in str(r.msg) for r in caplog.records
    ), "extraction failed and produced an empty KG without a word in the log"


def test_the_legacy_hint_path_is_untouched() -> None:
    """The mirror, and the reason the fix keys on the PROVIDER rather than on the source.

    A caller that passes a ``topic_label`` hint and never wires an extraction provider has not
    failed at anything — nothing was attempted. That path (tests, legacy callers) must keep
    working, or the fix trades one silent breakage for another.
    """
    art = build_artifact(
        "ep:x",
        "x",
        podcast_id="podcast:p1",
        episode_title="T",
        topic_label="Inflation outlook",
        detected_hosts=["Alice"],
    )
    assert "Topic" in _types(art)
    assert _topic_labels(art) == ["Inflation outlook"]


class TestEmptyTopicsAreAttributable:
    """A KG with no topics must say WHY — silence is indistinguishable from "nothing was said".

    Refusing to fabricate (above) creates a second obligation. With no extraction provider the
    only candidate labels are the summary bullets, and a summariser emits sentences, so every
    label is correctly rejected and the KG ends with zero Topic nodes. That is the right
    outcome, but ``model_version: "topic_labels"`` would then claim labels were *used*.

    A reader — or the #598 e2e acceptance test — cannot tell that apart from an episode that was
    never given labels at all. Both render as an empty topic list. The provenance string is the
    only place the difference can survive to the artifact, so it carries it (#1208).
    """

    def test_labels_all_rejected_is_marked_all_propositions(self) -> None:
        art = build_artifact(
            "ep:x",
            "x",
            podcast_id="podcast:p1",
            episode_title="T",
            topic_labels=list(_REAL_BULLETS),
            cfg=None,
        )
        assert _topic_labels(art) == []
        provenance = art["extraction"]["model_version"]
        assert provenance.endswith(":all_propositions"), (
            "every supplied label was a proposition and none became a Topic, but provenance "
            f"does not record it: {provenance!r}"
        )

    def test_surviving_labels_keep_the_plain_provenance(self) -> None:
        """The suffix must mean something — it may not appear when labels DID survive."""
        art = build_artifact(
            "ep:x",
            "x",
            podcast_id="podcast:p1",
            episode_title="T",
            topic_labels=["ai regulation", "open source ai models"],
            cfg=None,
        )
        assert sorted(_topic_labels(art)) == ["ai regulation", "open source ai models"]
        provenance = art["extraction"]["model_version"]
        assert not provenance.endswith(
            ":all_propositions"
        ), f"labels survived, so the all-rejected marker is a lie: {provenance!r}"
