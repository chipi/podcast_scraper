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

from podcast_scraper.search.backend import (  # noqa: E402
    SEGMENT_FIELDS,
    SEGMENT_NONEN_FIELDS,
    SegmentDocument,
)
from podcast_scraper.search.backends.lancedb_backend import (  # noqa: E402
    _segment_nonen_schema,
    _segment_schema,
    DEFAULT_EMBED_DIM as _EMBED_DIM,
    LanceDBBackend,
)


class TestTheAdapterRendersTheDeclarationFaithfully:
    """ONE test where four used to be, and this is the only thing here that needs pyarrow.

    The field lists are declared in `search/backend.py` (stdlib-only, so unit tests read them
    freely) and `lancedb_backend` derives both schemas from them. What a real library is still
    required for is exactly this: proving the derivation is faithful — same fields, same order, and
    a vector column that exists in one tier and not the other.

    Re-reading the field list four times through pyarrow, which is what this file did before
    2026-10-03, proved nothing the declaration could not state by itself.
    """

    def test_the_english_schema_is_the_declared_field_list_in_order(self) -> None:
        assert list(_segment_schema(8).names) == list(SEGMENT_FIELDS)

    def test_the_non_english_schema_is_the_declared_field_list_in_order(self) -> None:
        assert list(_segment_nonen_schema().names) == list(SEGMENT_NONEN_FIELDS)

    def test_the_vector_column_is_real_in_one_tier_and_absent_in_the_other(self) -> None:
        """The guarantee as the STORAGE sees it. The declaration says `embedding` is absent; this
        confirms pyarrow agrees, and that the English one is a fixed-size vector of the right
        width rather than, say, a string that happens to be named `embedding`."""
        en = _segment_schema(8)
        assert en.field("embedding").type.list_size == 8
        assert "embedding" not in set(_segment_nonen_schema().names)

    def test_the_resolver_hands_back_the_right_schema_per_tier(self) -> None:
        class _B(LanceDBBackend):
            def __init__(self) -> None:
                self.embed_dim = 8

        b = _B()
        assert list(b._schema_for("segment").names) == list(SEGMENT_FIELDS)
        assert list(b._schema_for("segment_nonen").names) == list(SEGMENT_NONEN_FIELDS)


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
