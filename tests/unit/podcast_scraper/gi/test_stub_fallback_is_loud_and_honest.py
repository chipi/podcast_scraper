"""A stub insight is a failed episode, and must look like one (#1657 acceptance item 9).

112 of 678 production episodes (16.5 %) hold exactly one insight, and that insight is the
literal string ``"Summary insight (stub)."``. They are not thin episodes. They are episodes
where insight generation failed, and two separate defects kept that invisible:

SILENCE. Six branches of ``_resolve_insight_specs`` fall through to the stub. Exactly one of
them logged anything — ``logger.debug`` on the exception path — and none touched a metric. No
provider, a provider without ``generate_insights``, a non-list return, nothing parseable in the
return, and dedup collapsing to zero all simply ran off the end of the function into the stub
return. #701 had already diagnosed this exact anti-pattern one path over ("cloud_thin produced
1-stub gi.json across 9 real-feed episodes for weeks") and fixed it with a WARNING plus
``gi_artifact_stub_fallback_count``; the fix went to the evidence-stack path only, not to the
path that decides whether an episode has any insights at all.

DISGUISE. The stub Insight node claimed ``grounded=True``, ``tier=3`` (CORE),
``routing_tag="surface"`` and ``salience=1.0`` — the exact profile of the best insight an
episode can produce. Its "evidence" is a slice of the transcript head chosen by offset, not
because it supports the claim; there is no claim. So the placeholder was ranked first and shown
to readers as a grounded, core-tier finding.

Both halves matter. Making it loud without making it honest leaves a fake insight on the page;
making it honest without making it loud leaves 112 episodes quietly empty.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import pytest

from podcast_scraper.gi import build_artifact
from podcast_scraper.gi import pipeline as gi_pipeline

pytestmark = [pytest.mark.unit]

STUB = "Summary insight (stub)."


class _Metrics:
    def __init__(self) -> None:
        self.gi_artifact_stub_fallback_count = 0


class _Cfg:
    def __init__(self, **kw: Any) -> None:
        self.gi_insight_source = "provider"
        self.gi_insight_chunk_chars = 0
        self.gi_insight_dedupe_threshold = 0.75
        for k, v in kw.items():
            setattr(self, k, v)


def _resolve(provider: Any, cfg: Any, metrics: Any) -> List[Tuple[str, str]]:
    """Call with the REAL signature — ``max_insights`` is derived inside, not passed."""
    return gi_pipeline._resolve_insight_specs(
        transcript_text="Some transcript body that is long enough to be worth reading.",
        cfg=cfg,
        insight_provider=provider,
        episode_title="An episode",
        pipeline_metrics=metrics,
    )


class _Provider:
    """Fails the way a real provider fails, at the boundary under test."""

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def generate_insights(self, **kw: Any) -> Any:
        if self.mode == "raise":
            raise RuntimeError("provider exploded")
        if self.mode == "non_list":
            return {"insights": []}
        if self.mode == "unparseable":
            return [None, {}, ""]
        if self.mode == "good":
            return [{"text": "A real insight about the topic.", "type": "claim"}]
        raise AssertionError(self.mode)


class TestEveryStubPathIsLoud:
    """One case per branch that can reach the stub. Each used to be silent."""

    @pytest.mark.parametrize(
        "provider,cfg_kw,expect_reason",
        [
            pytest.param(None, {}, "no_insight_provider", id="no_provider"),
            pytest.param(object(), {}, "provider_has_no_generate_insights", id="no_method"),
            pytest.param(_Provider("raise"), {}, "generate_insights_raised", id="raised"),
            # A dict return is rejected one layer down, in ``_as_insight_list``, which names the
            # type it refused; by the time the stub decision is made the list is simply empty.
            pytest.param(
                _Provider("non_list"),
                {},
                "no_parseable_insights_from_provider",
                id="non_list",
            ),
            pytest.param(
                _Provider("unparseable"),
                {},
                "no_parseable_insights_from_provider",
                id="unparseable",
            ),
        ],
    )
    def test_it_warns_and_names_the_reason(
        self,
        caplog: pytest.LogCaptureFixture,
        provider: Any,
        cfg_kw: Dict[str, Any],
        expect_reason: str,
    ) -> None:
        m = _Metrics()
        with caplog.at_level(logging.WARNING, logger=gi_pipeline.logger.name):
            out = _resolve(provider, _Cfg(**cfg_kw), m)
        assert out == [(STUB, "unknown")]
        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert "STUB" in joined, "the fallback must be visible at WARNING, not debug"
        assert expect_reason in joined, "the log must say WHICH path produced the stub"

    @pytest.mark.parametrize(
        "provider",
        [
            pytest.param(None, id="no_provider"),
            pytest.param(_Provider("raise"), id="raised"),
            pytest.param(_Provider("non_list"), id="non_list"),
            pytest.param(_Provider("unparseable"), id="unparseable"),
        ],
    )
    def test_the_metric_counter_moves(self, provider: Any) -> None:
        """A dashboard has to be able to see this without reading logs — that is what #701
        established, and it is why the counter already exists."""
        m = _Metrics()
        _resolve(provider, _Cfg(), m)
        assert m.gi_artifact_stub_fallback_count == 1

    def test_a_working_provider_neither_warns_nor_counts(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The guard against crying wolf: a healthy episode must stay silent, or the signal is
        worth nothing."""
        m = _Metrics()
        with caplog.at_level(logging.WARNING, logger=gi_pipeline.logger.name):
            out = _resolve(_Provider("good"), _Cfg(), m)
        assert out and out[0][0] != STUB
        assert m.gi_artifact_stub_fallback_count == 0
        assert "STUB" not in "\n".join(r.getMessage() for r in caplog.records)

    def test_the_configured_stub_source_is_not_reported_as_a_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``gi_insight_source: stub`` is a deliberate configuration, not a broken run. Warning
        on it would train operators to ignore the warning that matters."""
        m = _Metrics()
        with caplog.at_level(logging.WARNING, logger=gi_pipeline.logger.name):
            out = _resolve(None, _Cfg(gi_insight_source="stub"), m)
        assert out == [(STUB, "unknown")]
        assert m.gi_artifact_stub_fallback_count == 0
        assert "STUB" not in "\n".join(r.getMessage() for r in caplog.records)

    def test_the_debug_only_call_is_gone(self) -> None:
        """Structural: the exact line that hid 112 episodes. Matches the CALL, not the name —
        the handler's comment quotes ``logger.debug`` while explaining what it replaced."""
        import inspect

        src = inspect.getsource(gi_pipeline._resolve_insight_specs)
        assert "logger.debug(" not in src


class TestADictReturnIsNotTurnedIntoInsights:
    """Found while testing the stub paths, and worse than the bug being fixed.

    ``generate_chunked`` did ``list(got or [])``, and ``list()`` on a mapping yields its KEYS.
    A provider answering ``{"insights": [...]}`` — a shape several of them use — was silently
    converted into the single insight ``"insights"``, which then went through type
    classification and grounding as though a model had said it. Not a visible failure: a
    plausible-looking artifact assembled from a dict's key names.

    This is the opposite failure mode to the stub. The stub produces one obviously-fake insight
    that nobody was told about; this produced one insight that looks real and is not.
    """

    def test_a_dict_return_yields_no_insights(self) -> None:
        from podcast_scraper.gi import chunked_extraction as ce

        assert ce._as_insight_list({"insights": [{"text": "x"}]}) == []

    def test_the_key_names_never_become_insights(self) -> None:
        from podcast_scraper.gi import chunked_extraction as ce

        assert "insights" not in ce._as_insight_list({"insights": []})

    def test_a_list_passes_through_untouched(self) -> None:
        from podcast_scraper.gi import chunked_extraction as ce

        items = [{"text": "a"}, {"text": "b"}]
        assert ce._as_insight_list(items) == items

    def test_a_tuple_is_accepted(self) -> None:
        from podcast_scraper.gi import chunked_extraction as ce

        assert ce._as_insight_list(({"text": "a"},)) == [{"text": "a"}]

    def test_none_and_empty_are_quietly_empty(self) -> None:
        """No warning for these — an empty answer is a legitimate one; only a wrong TYPE is
        worth a line."""
        from podcast_scraper.gi import chunked_extraction as ce

        assert ce._as_insight_list(None) == []
        assert ce._as_insight_list([]) == []

    def test_a_dict_return_is_reported(self, caplog: pytest.LogCaptureFixture) -> None:
        from podcast_scraper.gi import chunked_extraction as ce

        with caplog.at_level(logging.WARNING, logger=ce.logger.name):
            ce._as_insight_list({"insights": [{"text": "x"}]})
        assert "dict" in "\n".join(r.getMessage() for r in caplog.records)

    def test_the_end_to_end_path_produces_a_stub_not_a_fake_insight(self) -> None:
        """The consequence at the boundary that matters: a dict-returning provider must end up
        as a declared stub, never as an insight called "insights"."""
        out = _resolve(_Provider("non_list"), _Cfg(), _Metrics())
        assert out == [(STUB, "unknown")]
        assert all(t != "insights" for t, _ in out)

    def test_both_call_sites_are_guarded(self) -> None:
        """The chunked loop extended ``merged`` with the same unguarded expression.

        Comment and docstring lines are excluded: the fix's own explanation quotes the old
        expression verbatim, and a naive substring search matches the prose describing the bug
        rather than the bug.
        """
        import inspect

        from podcast_scraper.gi import chunked_extraction as ce

        src = inspect.getsource(ce)
        code_lines = [
            ln for ln in src.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
        ]
        offending = [ln for ln in code_lines if "list(got or [])" in ln and "``" not in ln]
        assert not offending, f"an unguarded coercion remains: {offending}"
        assert src.count("_as_insight_list(") >= 3  # def + both call sites


class TestTheStubArtifactDoesNotPoseAsAFinding:
    """What a reader of the corpus actually sees."""

    def _stub_insight(self) -> Dict[str, Any]:
        art = build_artifact("ep:1", "Some transcript body.", prompt_version="v1")
        ins = [n for n in art["nodes"] if n.get("type") == "Insight"]
        assert len(ins) == 1 and ins[0]["properties"]["text"] == STUB
        return dict(ins[0]["properties"])

    def test_it_is_not_grounded(self) -> None:
        assert self._stub_insight()["grounded"] is False

    def test_it_is_filler_tier_not_core(self) -> None:
        from podcast_scraper.gi.value_gate import TIER_CORE, TIER_FILLER

        tier = self._stub_insight()["tier"]
        assert tier == TIER_FILLER
        assert tier != TIER_CORE

    def test_it_routes_to_drop_not_surface(self) -> None:
        """The one that decides whether a reader sees the placeholder presented as this
        episode's headline finding."""
        assert self._stub_insight()["routing_tag"] == "drop"

    def test_its_salience_is_zero(self) -> None:
        assert self._stub_insight()["salience"] == 0.0

    def test_the_tier_and_routing_agree_with_the_shared_rule(self) -> None:
        """``_apply_route_and_tag`` maps tier <= FILLER to "drop". The stub sets both by hand,
        so this pins them to the same rule rather than to a coincidence."""
        props: Dict[str, Any] = {"grounded": False, "surfaceable": False}
        gi_pipeline._apply_route_and_tag(props, self._stub_insight()["tier"])
        assert props["routing_tag"] == "drop"


class TestTheRealInsightPathIsUnaffected:
    """The fix must not quieten or downgrade a genuine episode."""

    def test_a_real_insight_keeps_its_text(self) -> None:
        out = _resolve(_Provider("good"), _Cfg(), _Metrics())
        assert out[0][0] == "A real insight about the topic."

    def test_a_real_run_produces_no_stub_spec(self) -> None:
        out = _resolve(_Provider("good"), _Cfg(), _Metrics())
        assert all(t != STUB for t, _ in out)
