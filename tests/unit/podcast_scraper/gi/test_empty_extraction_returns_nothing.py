"""If the provider gives nothing back, we return nothing (#1657).

WHAT THIS REPLACES
The GI pipeline was written before any provider was wired up, so it needed *something* to emit
and invented a placeholder insight: the literal string "Summary insight (stub)." plus a Quote
sliced out of the transcript head by byte offset and a SUPPORTED_BY edge joining them. None of
it came from the transcript's meaning. The placeholder long outlived its reason and reached
production episodes, where nothing downstream could tell it from a finding — it was written with
``grounded: True``, ``tier: 3`` (CORE) and ``routing_tag: "surface"``, i.e. the exact profile of
the best insight an episode can produce.

An earlier pass (#1657 item 9) made the placeholder *honest* — ungrounded, FILLER tier, routed to
``drop`` — and made every fallback path warn instead of failing silently. That was damage control
on a design that should not exist. This is the deletion.

THE RULE NOW
A provider that returns nothing means the episode has no insights. The artifact says exactly
that: an Episode node and no Insight nodes. Nothing is fabricated to fill the gap.

The file is still written, deliberately. Writing nothing would collapse "GI ran and found
nothing" into "GI never ran" — the ambiguity #1647 exists to remove, and the one that let #1646
hide across most of the corpus.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import pytest

from podcast_scraper.gi import build_artifact, pipeline as gi_pipeline

pytestmark = [pytest.mark.unit]


class _Cfg:
    def __init__(self, **kw: Any) -> None:
        self.gi_insight_chunk_chars = 0
        self.gi_insight_dedupe_threshold = 0.75
        for k, v in kw.items():
            setattr(self, k, v)


class _Provider:
    """Fails the way an over-quota or misbehaving provider does."""

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def generate_insights(self, **kw: Any) -> Any:
        if self.mode == "raise":
            raise RuntimeError("no budget/credit left on this key")
        if self.mode == "non_list":
            return {"insights": []}
        if self.mode == "unparseable":
            return [None, {}, ""]
        if self.mode == "empty":
            return []
        if self.mode == "good":
            return [{"text": "A real insight about the topic.", "type": "claim"}]
        raise AssertionError(self.mode)


def _resolve(provider: Any, cfg: Any, metrics: Any = None) -> List[Tuple[str, str]]:
    return gi_pipeline._resolve_insight_specs(
        transcript_text="Some transcript body that is long enough to be worth reading.",
        cfg=cfg,
        insight_provider=provider,
        episode_title="An episode",
        pipeline_metrics=metrics,
    )


class TestNothingInNothingOut:
    """The core rule, one case per way a provider can come back empty-handed."""

    @pytest.mark.parametrize(
        "provider,reason",
        [
            pytest.param(None, "no_insight_provider", id="no_provider"),
            pytest.param(object(), "provider_has_no_generate_insights", id="no_method"),
            pytest.param(_Provider("raise"), "generate_insights_raised", id="raised"),
            pytest.param(_Provider("non_list"), "no_parseable_insights", id="dict_return"),
            pytest.param(_Provider("unparseable"), "no_parseable_insights", id="unparseable"),
            pytest.param(_Provider("empty"), "no_parseable_insights", id="empty_list"),
        ],
    )
    def test_it_returns_an_empty_list(self, provider: Any, reason: str) -> None:
        """No placeholder, no invented text — an empty list."""
        assert _resolve(provider, _Cfg()) == []

    @pytest.mark.parametrize(
        "provider",
        [
            pytest.param(None, id="no_provider"),
            pytest.param(_Provider("raise"), id="raised"),
            pytest.param(_Provider("unparseable"), id="unparseable"),
        ],
    )
    def test_it_says_so_at_warning_level(
        self, caplog: pytest.LogCaptureFixture, provider: Any
    ) -> None:
        """Empty must be VISIBLE. Before #1657 five of these six paths logged nothing at all and
        the sixth logged at debug, which is how the placeholders accumulated unnoticed."""
        with caplog.at_level(logging.WARNING, logger=gi_pipeline.logger.name):
            _resolve(provider, _Cfg())
        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert "NO insights" in joined
        assert "reason=" in joined, "the log must say WHICH path produced nothing"

    def test_no_placeholder_text_is_ever_returned(self) -> None:
        """The specific string that used to be manufactured, asserted dead."""
        for mode in ("raise", "non_list", "unparseable", "empty"):
            for text, _kind in _resolve(_Provider(mode), _Cfg()):
                assert "stub" not in text.lower()

    def test_a_working_provider_is_unaffected(self) -> None:
        out = _resolve(_Provider("good"), _Cfg())
        assert [t for t, _ in out] == ["A real insight about the topic."]


class TestTheArtifactIsHonestlyEmpty:
    """What lands on disk when there is nothing to say."""

    def _empty(self) -> Dict[str, Any]:
        return build_artifact("ep:1", "Some transcript body.", prompt_version="v1")

    def test_it_has_no_insight_nodes(self) -> None:
        assert [n for n in self._empty()["nodes"] if n["type"] == "Insight"] == []

    def test_it_has_no_quote_nodes(self) -> None:
        """The old Quote was a transcript slice chosen by offset, supporting a claim that did
        not exist. Quotes come from grounding or not at all."""
        assert [n for n in self._empty()["nodes"] if n["type"] == "Quote"] == []

    def test_it_has_no_edges(self) -> None:
        assert self._empty()["edges"] == []

    def test_it_still_carries_the_episode_and_its_provenance(self) -> None:
        """The file IS written: "ran and found nothing" must stay distinguishable from "never
        ran", and a re-derivation needs to know which model looked."""
        art = self._empty()
        assert [n["type"] for n in art["nodes"]] == ["Episode"]
        assert art["episode_id"] == "ep:1"
        assert art["prompt_version"] == "v1"
        assert "model_version" in art

    def test_it_validates_strictly(self, tmp_path: Any) -> None:
        """A zero-insight artifact is schema-legal — verified, not assumed."""
        from podcast_scraper.gi import write_artifact

        write_artifact(tmp_path / "ep.gi.json", self._empty(), validate=True)

    def test_it_passes_the_invariants(self) -> None:
        from podcast_scraper.gi.invariants import check_artifact_invariants

        assert (
            check_artifact_invariants(self._empty(), transcript_text="Some transcript body.") == []
        )


class TestTheDeletionIsComplete:
    """Structural: the placeholder cannot come back by accident."""

    def _pipeline_src(self) -> str:
        import inspect

        return inspect.getsource(gi_pipeline)

    def test_the_placeholder_constant_is_gone(self) -> None:
        assert not hasattr(gi_pipeline, "_STUB_INSIGHT_TEXT")

    def test_the_placeholder_builder_is_gone(self) -> None:
        assert not hasattr(gi_pipeline, "_build_stub_artifact")
        assert hasattr(gi_pipeline, "_build_empty_artifact")

    def test_the_pipeline_never_emits_that_string(self) -> None:
        """Allowing the docstring that explains the history, and nothing else."""
        code = [
            ln
            for ln in self._pipeline_src().splitlines()
            if "Summary insight" in ln and not ln.lstrip().startswith("#")
        ]
        # The only permitted mention is inside a docstring describing what was removed.
        assert all("used to" in ln or "invented" in ln for ln in code), code

    def test_there_is_no_insight_source_switch_any_more(self) -> None:
        """``gi_insight_source`` selected between "provider" and "stub" and DEFAULTED to stub, so
        any run whose config missed the field emitted placeholders for the whole run.

        Comments are excluded: the code that removed the switch explains what it removed.
        """
        code = [
            ln
            for ln in self._pipeline_src().splitlines()
            if "gi_insight_source" in ln and not ln.lstrip().startswith("#")
        ]
        assert not code, code


class TestADictReturnIsNotTurnedIntoInsights:
    """Kept from the earlier pass — a different way to fabricate content, still guarded.

    ``generate_chunked`` did ``list(got or [])``, and ``list()`` on a mapping yields its KEYS, so
    a provider answering ``{"insights": [...]}`` became the single insight ``"insights"`` and
    flowed through classification and grounding as if a model had said it.
    """

    def test_a_dict_return_yields_no_insights(self) -> None:
        from podcast_scraper.gi import chunked_extraction as ce

        assert ce._as_insight_list({"insights": [{"text": "x"}]}) == []

    def test_a_list_passes_through_untouched(self) -> None:
        from podcast_scraper.gi import chunked_extraction as ce

        items = [{"text": "a"}, {"text": "b"}]
        assert ce._as_insight_list(items) == items

    def test_the_end_to_end_path_produces_nothing_not_a_fake_insight(self) -> None:
        assert _resolve(_Provider("non_list"), _Cfg()) == []
