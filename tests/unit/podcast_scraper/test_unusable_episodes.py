"""D-44 point 2: an episode the pipeline cannot complete appears NOWHERE (#2254).

The operator's instruction, 2026-10-02: "if there is no english file after translation that means
pipeline cannot work, therefore we stop and somehow flag this episode is not good and it does not
show up anywhere."

There was no mechanism for this. `corpus_incidents.jsonl` is an append-only triage log, and
`build_catalog_rows` included any episode whose metadata parsed — which is why a half-processed
episode showed up EMPTY rather than not at all, the shape of #2198.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from podcast_scraper.languages import (
    episode_is_unusable,
    UNUSABLE_FIELD,
    UNUSABLE_REASON_FIELD,
)

pytestmark = pytest.mark.unit


class TestThePredicate:
    """One place decides. Every surface asks it rather than re-deriving from disk."""

    def test_an_unmarked_episode_is_usable(self) -> None:
        assert episode_is_unusable({"episode": {"episode_id": "ep1"}}) is None

    def test_a_marked_episode_returns_its_REASON(self) -> None:
        """The reason travels with the marker: an operator reading the artifact must be able to see
        WHY without consulting a log that may have rotated."""
        doc = {
            "episode": {
                UNUSABLE_FIELD: True,
                UNUSABLE_REASON_FIELD: "episode language is 'es' and the swap did not happen",
            }
        }
        assert episode_is_unusable(doc) == "episode language is 'es' and the swap did not happen"

    def test_a_marker_with_no_reason_still_counts(self) -> None:
        """Truthiness of the flag is what decides; a missing reason is a reporting gap, not a
        licence to serve the episode."""
        assert episode_is_unusable({"episode": {UNUSABLE_FIELD: True}}) == "unspecified"

    @pytest.mark.parametrize("doc", [None, "", 42, [], {}, {"episode": None}, {"episode": "x"}])
    def test_garbage_is_not_an_error(self, doc: Any) -> None:
        """Called on every episode in the catalog scan, so a malformed artifact must read as
        usable-but-odd rather than taking the scan down."""
        assert episode_is_unusable(doc) is None

    def test_a_FALSE_marker_is_usable(self) -> None:
        """An explicit `false` is how a repaired episode comes back, so it must not read as set."""
        assert episode_is_unusable({"episode": {UNUSABLE_FIELD: False}}) is None


class TestTheCatalogHonoursIt:
    @staticmethod
    def _corpus(root: Path, *, unusable: bool) -> None:
        meta = root / "feeds" / "p10" / "run_20260101-000000" / "metadata"
        meta.mkdir(parents=True, exist_ok=True)
        (root / "feeds" / "p10" / "run_20260101-000000" / "transcripts").mkdir(
            parents=True, exist_ok=True
        )
        episode: Dict[str, Any] = {
            "episode_id": "ep-bad",
            "title": "Un episodio",
            "published_date": "2026-01-01",
            "language": "es",
            "language_source": "rss",
        }
        if unusable:
            episode[UNUSABLE_FIELD] = True
            episode[UNUSABLE_REASON_FIELD] = "the swap did not happen"
        doc = {
            "feed": {"feed_id": "p10", "title": "Sesiones", "language": "es"},
            "episode": episode,
            "content": {"transcript_file_path": "transcripts/ep.txt"},
        }
        (meta / "ep.metadata.json").write_text(json.dumps(doc), encoding="utf-8")

    def test_a_usable_episode_is_in_the_catalog(self, tmp_path: Path) -> None:
        from podcast_scraper.server.corpus_catalog import build_catalog_rows

        self._corpus(tmp_path, unusable=False)
        assert len(build_catalog_rows(tmp_path)) == 1

    def test_an_unusable_episode_is_NOT(self, tmp_path: Path) -> None:
        """The one check, in the one function that feeds the app, the digest and topic clusters —
        so there is no surface left where it can appear empty instead of not at all."""
        from podcast_scraper.server.corpus_catalog import build_catalog_rows

        self._corpus(tmp_path, unusable=True)
        assert build_catalog_rows(tmp_path) == []


class TestTheStageWritesIt:
    def test_the_marker_lands_on_the_metadata_with_its_reason(self, tmp_path: Path) -> None:
        """Written by the stage that DISCOVERED the problem, beside the skip records, because the
        two statements have to agree: the episode whose analysis was skipped for an incomplete
        translation IS the unservable one."""
        from podcast_scraper.workflow.metadata_generation import _mark_episode_unusable

        meta = tmp_path / "metadata"
        meta.mkdir()
        (meta / "01 - ep.metadata.json").write_text(
            json.dumps({"episode": {"episode_id": "ep1"}, "feed": {"feed_id": "p10"}}),
            encoding="utf-8",
        )

        _mark_episode_unusable(str(tmp_path), "transcripts/01 - ep.txt", "no complete translation")

        doc = json.loads((meta / "01 - ep.metadata.json").read_text(encoding="utf-8"))
        assert episode_is_unusable(doc) == "no complete translation"
        # and nothing else was lost
        assert doc["episode"]["episode_id"] == "ep1"
        assert doc["feed"]["feed_id"] == "p10"

    def test_it_writes_ATOMICALLY(self, tmp_path: Path) -> None:
        """Read-modify-write on an artifact the rest of the pipeline also reads, so a torn file
        would be worse than no marker. No temp file survives."""
        from podcast_scraper.workflow.metadata_generation import _mark_episode_unusable

        meta = tmp_path / "metadata"
        meta.mkdir()
        (meta / "01 - ep.metadata.json").write_text(json.dumps({"episode": {}}), encoding="utf-8")
        _mark_episode_unusable(str(tmp_path), "transcripts/01 - ep.txt", "reason")
        assert not [p for p in meta.iterdir() if ".tmp" in p.name]

    def test_a_missing_artifact_is_not_an_error(self, tmp_path: Path) -> None:
        """Best-effort by design: failing to write the marker must not fail the episode. The
        artifact is still on disk and the manifest still records the skip, so the worst case is an
        episode showing up empty — today's behaviour, not a regression."""
        from podcast_scraper.workflow.metadata_generation import _mark_episode_unusable

        _mark_episode_unusable(str(tmp_path), "transcripts/absent.txt", "reason")  # no raise

    def test_no_transcript_relpath_is_a_no_op(self, tmp_path: Path) -> None:
        from podcast_scraper.workflow.metadata_generation import _mark_episode_unusable

        _mark_episode_unusable(str(tmp_path), None, "reason")
        assert not (tmp_path / "metadata").exists()
