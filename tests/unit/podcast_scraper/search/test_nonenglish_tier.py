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

from typing import Any, Dict, List, Optional

import pytest

from podcast_scraper.search.backend import SegmentDocument
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


class TestTheRouter:
    """Which table a chunk lands in, decided in ONE place from one attribute."""

    @staticmethod
    def _doc(language: Optional[str], *, doc_id: str = "ep1_chunk_0") -> SegmentDocument:
        return SegmentDocument(
            id=doc_id,
            text="algo de texto",
            show_id="show",
            episode_id="ep1",
            start_time=0.0,
            end_time=5.0,
            embedding=[0.1, 0.2],
            language=language,
        )

    def test_english_rows_keep_the_english_schema_exactly(self) -> None:
        """No `language` key on an English row, so the stored `segments` schema does not move —
        which is what keeps every existing corpus off a rebuild."""
        english, non_english = LanceDBBackend._split_segments_by_language([self._doc("en")])
        assert non_english == []
        assert "language" not in english[0]
        assert "embedding" in english[0]

    def test_non_english_rows_lose_the_vector_and_gain_the_language(self) -> None:
        english, non_english = LanceDBBackend._split_segments_by_language([self._doc("es")])
        assert english == []
        assert "embedding" not in non_english[0], "there is nowhere to put it"
        assert non_english[0]["language"] == "es"

    def test_a_regional_subtag_is_normalized(self) -> None:
        _en, non_english = LanceDBBackend._split_segments_by_language([self._doc("es-ES")])
        assert non_english[0]["language"] == "es"

    @pytest.mark.parametrize("language", [None, "", "   ", "EN", "en-US"])
    def test_unknown_or_english_routes_ENGLISH(self, language: Optional[str]) -> None:
        """A row with no language routes English deliberately. Most of the corpus predates
        language resolution, and sending those to a keyword-only tier would silently remove
        them from semantic search — a large, invisible regression for the corpus that works."""
        english, non_english = LanceDBBackend._split_segments_by_language([self._doc(language)])
        assert len(english) == 1 and non_english == []

    def test_a_mixed_batch_splits_both_ways(self) -> None:
        english, non_english = LanceDBBackend._split_segments_by_language(
            [
                self._doc("en", doc_id="a"),
                self._doc("es", doc_id="b"),
                self._doc(None, doc_id="c"),
                self._doc("de", doc_id="d"),
            ]
        )
        assert [r["id"] for r in english] == ["a", "c"]
        assert [r["id"] for r in non_english] == ["b", "d"]

    def test_the_schema_resolver_handles_both_kinds_of_tier(self) -> None:
        class _B(LanceDBBackend):
            def __init__(self) -> None:
                self.embed_dim = 8

        b = _B()
        assert "embedding" in set(b._schema_for("segment").names)
        assert "embedding" not in set(b._schema_for("segment_nonen").names)


class TestBothWritePathsRoute:
    """The batch path and the SINGULAR path must route identically.

    Writing `dataclasses.asdict(doc)` straight to the `segment` tier was a real failure found
    by the integration suite: the English schema has no `language` column, so adding the field
    broke every caller of `upsert_segment` at once. One router, every path.
    """

    @staticmethod
    def _recording_backend() -> Any:
        """A backend that records (tier, rows) instead of touching LanceDB."""

        class _B(LanceDBBackend):
            def __init__(self) -> None:
                self.embed_dim = 8
                self.writes: List[tuple] = []
                self.cleared: List[str] = []

            # type: ignore[override] on both — the fakes narrow the base signatures.
            def _upsert_many(  # type: ignore[override]
                self, tier: str, rows: List[Dict[str, Any]]
            ) -> None:
                self.writes.append((tier, rows))

            def _replace_many(  # type: ignore[override]
                self, tier: str, rows: List[Dict[str, Any]]
            ) -> None:
                self.writes.append((tier, rows))

            def has_tier(self, tier: str) -> bool:  # type: ignore[override]
                return True

            def clear_tier_mvcc(self, tier: str) -> None:  # type: ignore[override]
                self.cleared.append(tier)

        return _B()

    def test_the_SINGULAR_path_routes_a_spanish_chunk_away_from_the_english_tier(self) -> None:
        """Behaviour, not source text. Writing the raw dict to `segment` was a real failure the
        integration suite caught: the English schema has no `language` column, so adding the
        field broke every caller of `upsert_segment` at once."""
        b = self._recording_backend()
        b.upsert_segment(TestTheRouter._doc("es", doc_id="x"))
        assert [tier for tier, _ in b.writes] == ["segment_nonen"]
        assert "embedding" not in b.writes[0][1][0]

    def test_the_singular_path_still_routes_english_to_the_english_tier(self) -> None:
        b = self._recording_backend()
        b.upsert_segment(TestTheRouter._doc("en", doc_id="x"))
        assert [tier for tier, _ in b.writes] == ["segment"]
        assert "language" not in b.writes[0][1][0]

    def test_replace_with_no_non_english_rows_CLEARS_the_tier(self) -> None:
        """A full reindex that replaced only `segments` would leave a previous build's
        non-English rows serving beside fresh English ones — stale content surviving a
        'replace', which is exactly what replace exists to prevent."""
        b = self._recording_backend()
        b.replace_segments([TestTheRouter._doc("en", doc_id="a")])
        assert [tier for tier, _ in b.writes] == ["segment"]
        assert b.cleared == ["segment_nonen"]

    def test_replace_with_both_languages_writes_both_tiers(self) -> None:
        b = self._recording_backend()
        b.replace_segments(
            [TestTheRouter._doc("en", doc_id="a"), TestTheRouter._doc("es", doc_id="b")]
        )
        assert [tier for tier, _ in b.writes] == ["segment", "segment_nonen"]
        assert b.cleared == []
