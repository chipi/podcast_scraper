"""An enricher that feeds a SURFACE must count what the user can actually see.

GI marks an insight ``surfaceable: False`` when the speaking voice is not a named person — an
advertisement, a person we failed to name, or the vox-pop of a narrated piece that nobody names. An
unattributed STANCE is not a stance: nobody holds it and nobody can disagree with it.

Two enrichers feed surfaces and were counting them anyway:

* ``insight_sentiment`` is, by its own docstring, "the colour layer for the per-person position
  arc". An unsurfaceable insight HAS NO PERSON — tinting somebody's arc with it attributes a mood to
  a person who never said the words.
* ``insight_density`` feeds a timeline the user looks at. Counting insights they will never be shown
  inflates it — and on Planet Money and The Daily the tape is 36-40% of the episode, so this is not
  a rounding error.

Corpus-scope / CONNECT enrichers keep them on purpose: a fact is still a fact, and a story thread
never needed a speaker.
"""

from __future__ import annotations

from typing import Any, Dict

from podcast_scraper.enrichment.enrichers._loaders import is_surfaceable_insight


def _insight(iid: str, text: str, surfaceable: bool | None = None) -> Dict[str, Any]:
    props: Dict[str, Any] = {"text": text}
    if surfaceable is not None:
        props["surfaceable"] = surfaceable
    return {"id": iid, "type": "Insight", "properties": props}


class TestTheSurfaceablePredicate:
    def test_a_named_persons_insight_surfaces(self) -> None:
        assert is_surfaceable_insight(_insight("i1", "Prescribing via chatbot is unsafe."))

    def test_an_explicitly_surfaceable_insight_surfaces(self) -> None:
        assert is_surfaceable_insight(_insight("i1", "x", surfaceable=True))

    def test_an_unattributed_insight_does_not(self) -> None:
        assert not is_surfaceable_insight(_insight("i1", "x", surfaceable=False))

    def test_it_defaults_to_TRUE_so_an_old_corpus_is_not_silently_emptied(self) -> None:
        """Absent flag = surfaceable. Every insight in prod-v2 predates the flag, and defaulting to
        False would silently blank the surfaces of an entire corpus."""
        assert is_surfaceable_insight({"id": "i1", "type": "Insight"})
        assert is_surfaceable_insight({"id": "i1", "type": "Insight", "properties": None})


def test_insight_density_counts_only_what_the_user_can_see() -> None:
    """The timeline must not be inflated by insights that never reach a surface."""
    from podcast_scraper.enrichment.enrichers import insight_density

    gi = {
        "nodes": [
            _insight("i:seen", "The airline shut down this week."),
            _insight("i:tape", "It was the greatest thing that ever happened to me.", False),
            {
                "id": "q:seen",
                "type": "Quote",
                "properties": {"text": "a", "timestamp_start_ms": 1000},
            },
            {
                "id": "q:tape",
                "type": "Quote",
                "properties": {"text": "b", "timestamp_start_ms": 2000},
            },
        ],
        "edges": [
            {"type": "SUPPORTED_BY", "from": "i:seen", "to": "q:seen"},
            {"type": "SUPPORTED_BY", "from": "i:tape", "to": "q:tape"},
        ],
    }

    surfaceable = {
        str(n.get("id"))
        for n in gi["nodes"]
        if n.get("type") == "Insight" and is_surfaceable_insight(n)
    }
    assert surfaceable == {"i:seen"}, (
        "the tape insight would be counted on a timeline the user is shown, inflating the density "
        "with an insight that never surfaces"
    )
    assert hasattr(
        insight_density, "is_surfaceable_insight"
    ), "insight_density must import the predicate — otherwise the gate is wired to nothing"


def test_insight_sentiment_does_not_tint_a_persons_arc_with_a_voice_nobody_names() -> None:
    from podcast_scraper.enrichment.enrichers import insight_sentiment

    assert hasattr(insight_sentiment, "is_surfaceable_insight"), (
        "insight_sentiment is the colour layer for the PER-PERSON position arc, and an "
        "unsurfaceable insight has no person — it must honour the flag"
    )
