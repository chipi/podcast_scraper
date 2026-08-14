"""Stale enrichment artifacts must be distinguishable from fresh ones (#1650).

``/api/corpus/enrichments`` listed whatever ``.json`` files were on disk. A DISABLED
enricher's output stays there forever, so ``topic_cooccurrence_corpus.json`` kept appearing —
1.5 MB, no ``computed_at`` in the response, nothing to indicate it had not run in days. A
consumer reading that list had no way to tell a live artifact from an abandoned one.

That is the same shape as the defect this epic exists to fix: an artifact whose PRESENCE is
reported while its VALIDITY is not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from podcast_scraper.server.routes.corpus_enrichments import _latest_run_enricher_ids

pytestmark = [pytest.mark.unit]


def _write_summary(enrichments: Path, per_enricher: Dict[str, Any]) -> None:
    enrichments.mkdir(parents=True, exist_ok=True)
    (enrichments / "run_summary.json").write_text(
        json.dumps({"status": "ok", "per_enricher": per_enricher}), encoding="utf-8"
    )


class TestLatestRunEnricherIds:
    def test_reads_the_ids_the_last_run_produced(self, tmp_path: Path) -> None:
        _write_summary(tmp_path, {"insight_density": {}, "guest_coappearance": {}})
        assert _latest_run_enricher_ids(tmp_path) == {"insight_density", "guest_coappearance"}

    def test_missing_summary_is_unknown_not_empty(self, tmp_path: Path) -> None:
        """None means "we cannot say"; an empty set would claim the last run produced nothing.

        Rendering "unknown" as "stale" would flag every artifact on a corpus that predates run
        summaries — noise that trains an operator to ignore the badge, which is worse than not
        having one.
        """
        tmp_path.mkdir(parents=True, exist_ok=True)
        assert _latest_run_enricher_ids(tmp_path) is None

    def test_malformed_summary_is_unknown(self, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "run_summary.json").write_text("{not json", encoding="utf-8")
        assert _latest_run_enricher_ids(tmp_path) is None

    def test_summary_without_per_enricher_is_unknown(self, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "run_summary.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
        assert _latest_run_enricher_ids(tmp_path) is None

    def test_an_enricher_that_did_not_run_is_absent_from_the_set(self, tmp_path: Path) -> None:
        """The real case: topic_cooccurrence_corpus was disabled but its artifact remained."""
        _write_summary(tmp_path, {"insight_density": {}, "insight_sentiment": {}})
        ids = _latest_run_enricher_ids(tmp_path)
        assert ids is not None
        assert "topic_cooccurrence_corpus" not in ids

    def test_it_reads_only_the_summary_directly_under_the_given_root(self, tmp_path: Path) -> None:
        """``enrichments_dir`` derives from the ``path`` query parameter, so joining onto it is
        a path-traversal sink — CodeQL flagged the original bare
        ``enrichments_dir / "run_summary.json"`` as high severity, correctly.

        The read now goes through ``safe_fixed_file_under_root``. A run summary sitting beside
        the root must never be picked up.
        """
        _write_summary(tmp_path / "neighbour", {"leaked_enricher": {}})

        corpus = tmp_path / "corpus" / "enrichments"
        corpus.mkdir(parents=True, exist_ok=True)
        assert _latest_run_enricher_ids(corpus) is None

        # ...and once a real one exists under the root, that is what is read.
        _write_summary(corpus, {"insight_density": {}})
        assert _latest_run_enricher_ids(corpus) == {"insight_density"}
