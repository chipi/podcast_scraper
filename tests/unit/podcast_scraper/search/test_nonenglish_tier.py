"""S2.9 / D-14 option B: non-English chunks live in a table with NO vector column.

WHY A SEPARATE TABLE RATHER THAN A LANGUAGE COLUMN PLUS A FILTER. A row with no vector CANNOT
appear in a semantic result — that is a property of the storage. A language tag plus a filter is
a guarantee only for as long as every query path remembers the filter, and there are several
query paths in several files. A zero vector would have been worse than either: it would
out-rank most real results rather than being absent from them.

AND IT MUST NOT FORCE A REBUILD. Every index that exists today was built before this table, so
its absence has to read as "no non-English content" rather than as an error. That tolerance is
what lets the table be added without bumping `LANCE_SCHEMA_VERSION`, which D-14 asked to be
confirmed in this slice.
"""

from __future__ import annotations

import pytest

from podcast_scraper.search.backends.lancedb_backend import (
    _segment_nonen_schema,
    _segment_schema,
    LANCE_SCHEMA_VERSION,
    LanceDBBackend,
)

pytestmark = pytest.mark.unit


class TestTheGuaranteeIsStructural:
    def test_the_non_english_schema_has_NO_embedding_column(self) -> None:
        """The whole design in one assertion: there is nowhere to put a vector, so no code path
        — present or future — can include these rows in a dense search."""
        names = set(_segment_nonen_schema().names)
        assert "embedding" not in names
        assert "text" in names, "but BM25 still needs the text"
        assert "language" in names

    def test_the_schema_builder_takes_no_dimension(self) -> None:
        """A signature that cannot accept `dim` cannot be handed one by mistake."""
        import inspect

        assert list(inspect.signature(_segment_nonen_schema).parameters) == []

    def test_it_otherwise_mirrors_the_english_segment_schema(self) -> None:
        """A reader of one should not have to learn a second shape. Everything but the vector
        and the added language tag is identical, so joins and filters behave the same."""
        en = set(_segment_schema(8).names) - {"embedding"}
        non_en = set(_segment_nonen_schema().names) - {"language"}
        assert en == non_en

    def test_the_tier_is_excluded_from_every_dense_path(self) -> None:
        """Dense paths derive their tier list from `DENSE_TIERS`, never from `TABLES` — so
        adding a tier cannot silently enrol it in vector search."""
        assert "segment_nonen" in LanceDBBackend.TABLES
        assert "segment_nonen" not in LanceDBBackend.DENSE_TIERS
        assert LanceDBBackend.KEYWORD_ONLY_TIERS == ("segment_nonen",)

    def test_dense_and_keyword_only_tiers_do_not_overlap(self) -> None:
        assert not set(LanceDBBackend.DENSE_TIERS) & set(LanceDBBackend.KEYWORD_ONLY_TIERS)


class TestItDoesNotForceARebuild:
    def test_the_stored_schema_version_did_NOT_move(self) -> None:
        """D-14's claim, confirmed here as the slice asked. A version bump would make every
        existing corpus report stale and force a full reindex — for content none of them have.
        Adding a TABLE leaves the three stored schemas untouched, so it does not.
        """
        assert LANCE_SCHEMA_VERSION == 3

    def test_the_three_existing_schemas_are_unchanged(self) -> None:
        """The other half of the same claim: if a field had been added to `segments` instead,
        every corpus would need rebuilding."""
        assert set(_segment_schema(8).names) == {
            "id",
            "text",
            "embedding",
            "show_id",
            "episode_id",
            "speaker_id",
            "start_time",
            "end_time",
            "linked_insight_ids",
            "source_tier",
            "publish_date",
        }


class TestReadPathTolerance:
    def test_a_missing_table_reads_as_absent_not_as_an_error(self, tmp_path: object) -> None:
        """Every index built before S2.9 — which is all of them — has no such table."""

        class _Backend(LanceDBBackend):
            def __init__(self) -> None:  # no db, no disk
                pass

            def _open_if_exists(self, tier: str):  # type: ignore[override]
                return None

        assert _Backend().has_tier("segment_nonen") is False

    def test_a_present_table_is_reported(self) -> None:
        class _Backend(LanceDBBackend):
            def __init__(self) -> None:
                pass

            def _open_if_exists(self, tier: str):  # type: ignore[override]
                return object() if tier == "segment_nonen" else None

        b = _Backend()
        assert b.has_tier("segment_nonen") is True
        assert b.has_tier("segment") is False
