"""The non-English tier's REAL schema and its survival across a reindex (S2.9 / D-14 option B).

MOVED OUT OF `tests/unit/podcast_scraper/search/test_nonenglish_tier.py` on 2026-10-03, and the
reason is the one the Unit Testing Guide gives: unit tests must run with no ML packages installed,
and the policy checker's rule U1 forbids `pytest.importorskip()` in `tests/unit/` outright. These
particular assertions cannot satisfy that by mocking — they are about what a real `pyarrow` schema
contains and what a real LanceDB table holds after a reindex. Against a `MagicMock` they would pass
while asserting nothing, which is worse than not having them.

HOW THIS WAS FOUND, because it is a gap worth naming: CI's `test-unit` job installs neither
`lancedb` nor `pyarrow`, both are installed on the development machine, and `lancedb_backend`
imports them lazily. So `make ci-fast` reported 13,353 passing locally while CI's unit job failed
these eight — and PR #2260 had been CONFLICTING until that morning, so no CI job had ever run on
the branch to say so. The sibling tests that only inspect a function SIGNATURE need neither wheel
and stay in the unit file; only the ones that build a schema or open a table moved here.

WHAT THE TIER IS FOR. A row with no vector column CANNOT appear in a semantic result — that is a
property of the storage, not of a filter every query path has to remember. A zero vector would have
been worse than either, out-ranking most real results rather than being absent from them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("pyarrow")
pytest.importorskip("lancedb")

from podcast_scraper.search.backend import SegmentDocument  # noqa: E402
from podcast_scraper.search.backends.lancedb_backend import (  # noqa: E402
    _segment_nonen_schema,
    _segment_schema,
    DEFAULT_EMBED_DIM as _EMBED_DIM,
    LanceDBBackend,
)


class TestTheGuaranteeIsStructural:
    """The design in assertions about the real schema, not about the splitter's intent."""

    def test_the_non_english_schema_has_NO_embedding_column(self) -> None:
        """The whole design in one assertion: there is nowhere to put a vector, so no code path
        — present or future — can include these rows in a dense search."""
        names = set(_segment_nonen_schema().names)
        assert "embedding" not in names
        assert "text" in names, "but BM25 still needs the text"
        assert "language" in names

    def test_it_otherwise_mirrors_the_english_segment_schema(self) -> None:
        """A reader of one should not have to learn a second shape. Everything but the vector
        and the added language tag is identical, so joins and filters behave the same."""
        en = set(_segment_schema(8).names) - {"embedding"}
        non_en = set(_segment_nonen_schema().names) - {"language"}
        assert en == non_en


class TestItDoesNotForceARebuild:
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


class TestTheRouter:
    def test_the_schema_resolver_handles_both_kinds_of_tier(self) -> None:
        class _B(LanceDBBackend):
            def __init__(self) -> None:
                self.embed_dim = 8

        b = _B()
        assert "embedding" in set(b._schema_for("segment").names)
        assert "embedding" not in set(b._schema_for("segment_nonen").names)


class TestAReindexDoesNotDELETETheNonEnglishTier:
    """An episode must stay findable in the language it was spoken in (Goal 6), which is why this
    is asserted against a REAL index on disk rather than against the splitter."""

    @staticmethod
    def _docs() -> List[SegmentDocument]:
        def doc(doc_id: str, language: Optional[str]) -> SegmentDocument:
            return SegmentDocument(
                id=doc_id,
                text="drenaje y estructura del suelo" if language else "drainage and soil",
                show_id="p10",
                episode_id="ep1",
                start_time=0.0,
                end_time=5.0,
                embedding=[0.1] * _EMBED_DIM,
                language=language,
            )

        return [doc("ep1_chunk_0", "en"), doc("ep1_chunk_0:src", "es")]

    def _rows(self, path: str, tier: str) -> Optional[int]:
        backend = LanceDBBackend(path, embed_dim=_EMBED_DIM)
        table = backend._open_if_exists(tier)
        return None if table is None else int(table.count_rows())

    def test_the_source_rows_survive_a_reindex_over_an_existing_index(self, tmp_path: Any) -> None:
        from podcast_scraper.search.two_tier_indexer import _finalize_reindex_clear

        path = str(tmp_path / "lance_index")
        backend = LanceDBBackend(path, embed_dim=_EMBED_DIM)
        backend.replace_segments(self._docs())
        assert self._rows(path, "segment_nonen") == 1, "setup: the source row was not written"

        # Exactly what the build does at the end of a full reindex: every tier that existed
        # before and was not recorded as overwritten gets MVCC-emptied.
        pre_existing = set(backend.existing_tier_tables())
        assert "segment_nonen" in pre_existing
        _finalize_reindex_clear(Path(path), backend, pre_existing, overwritten_tiers={"segment"})

        assert self._rows(path, "segment_nonen") == 1, (
            "the reindex emptied the non-English tier — the rows were written by "
            "replace_segments and then deleted by the finalize clear, so no non-English "
            "content is searchable on any corpus that already had an index"
        )
