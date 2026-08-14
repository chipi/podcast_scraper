"""Stale enrichment artifacts must be distinguishable from fresh ones (#1650).

``/api/corpus/enrichments`` listed whatever ``.json`` files were on disk. A DISABLED
enricher's output stays there forever, so ``topic_cooccurrence_corpus.json`` kept appearing —
1.5 MB, no ``computed_at`` in the response, nothing to indicate it had not run in days. A
consumer reading that list had no way to tell a live artifact from an abandoned one.

That is the same shape as the defect this epic exists to fix: an artifact whose PRESENCE is
reported while its VALIDITY is not.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from podcast_scraper.server.routes.corpus_enrichments import _enricher_ids_from_summary

pytestmark = [pytest.mark.unit]


def _summary(per_enricher: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "ok", "per_enricher": per_enricher}


class TestEnricherIdsFromSummary:
    def test_reads_the_ids_the_last_run_produced(self) -> None:
        ids = _enricher_ids_from_summary(
            _summary({"insight_density": {}, "guest_coappearance": {}})
        )
        assert ids == {"insight_density", "guest_coappearance"}

    def test_a_summary_without_per_enricher_is_unknown_not_empty(self) -> None:
        """None means "we cannot say"; an empty set would claim the last run produced nothing.

        Rendering "unknown" as "stale" would badge every artifact on a corpus that predates run
        summaries — noise that trains an operator to ignore the badge, which is worse than not
        having one.
        """
        assert _enricher_ids_from_summary({"status": "ok"}) is None

    def test_a_malformed_per_enricher_is_unknown(self) -> None:
        assert _enricher_ids_from_summary({"per_enricher": ["not", "a", "dict"]}) is None
        assert _enricher_ids_from_summary({"per_enricher": None}) is None

    def test_an_empty_per_enricher_is_a_real_empty_set(self) -> None:
        """Distinct from None: the run reported, and it produced nothing."""
        assert _enricher_ids_from_summary(_summary({})) == set()

    def test_an_enricher_that_did_not_run_is_absent_from_the_set(self) -> None:
        """The real case: topic_cooccurrence_corpus was disabled but its artifact remained."""
        ids = _enricher_ids_from_summary(_summary({"insight_density": {}, "insight_sentiment": {}}))
        assert ids is not None
        assert "topic_cooccurrence_corpus" not in ids

    def test_it_takes_a_payload_and_never_a_path(self) -> None:
        """Regression guard for the CodeQL finding.

        The first version of this helper took ``enrichments_dir`` and joined
        ``/ "run_summary.json"`` onto it. ``enrichments_dir`` derives from the ``path`` query
        parameter, making that join a path-traversal sink — CodeQL flagged it high severity and
        was right. The caller now sources the payload from the directory glob it is already
        walking, so no path is constructed here at all.

        Keeping this signature payload-only is what prevents the sink from coming back.
        """
        import inspect

        params = inspect.signature(_enricher_ids_from_summary).parameters
        assert list(params) == ["summary"]
        assert params["summary"].annotation in ("dict[str, Any]", Dict[str, Any])
