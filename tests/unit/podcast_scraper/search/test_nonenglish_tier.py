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

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from podcast_scraper.search.backend import SegmentDocument
from podcast_scraper.search.backends.lancedb_backend import (
    _segment_nonen_schema,
    _segment_schema,
    DEFAULT_EMBED_DIM as _EMBED_DIM,
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

            # The fakes narrow the base signatures, hence the ignores.
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


class TestTheQueryPath:
    """The keyword leg reads the tier; the dense leg cannot.

    Not because a filter excludes it — because the table has no vector column, so a dense query
    has nothing to match. The read path just declines to open a table that cannot answer the
    question asked.
    """

    def test_a_keyword_query_over_all_tiers_includes_the_non_english_one(self) -> None:
        class _B(LanceDBBackend):
            def __init__(self) -> None:
                pass

        assert _B()._tables_for_tier("all", keyword=True) == [
            "segment",
            "insight",
            "aux",
            "segment_nonen",
        ]

    def test_a_DENSE_query_over_all_tiers_excludes_it(self) -> None:
        class _B(LanceDBBackend):
            def __init__(self) -> None:
                pass

        assert _B()._tables_for_tier("all", keyword=False) == ["segment", "insight", "aux"]

    def test_the_default_is_dense_safe(self) -> None:
        """A caller that forgets the flag must not accidentally enrol the tier in a vector
        search — the safe default is the one that omits it."""

        class _B(LanceDBBackend):
            def __init__(self) -> None:
                pass

        assert "segment_nonen" not in _B()._tables_for_tier("all")

    def test_an_explicit_tier_request_pulls_in_NO_UNRELATED_tier(self) -> None:
        """An explicit scope stays a scope — it must not quietly widen to the whole index.

        This asserted `["segment"]` exactly, which read as "honour the request unchanged" and was
        really "a scoped search cannot see non-English content". The principle was right and its
        application conflated a PHYSICAL table with a LOGICAL tier: `segment_nonen` is the segment
        tier's other half, split from it by storage alone, so serving it IS honouring a request
        for segments. What the scope must still exclude is `insight` and `aux`, and that is what
        this now says.
        """

        class _B(LanceDBBackend):
            def __init__(self) -> None:
                pass

        tables = _B()._tables_for_tier("segment", keyword=True)
        assert set(tables) == {"segment", "segment_nonen"}
        assert "insight" not in tables and "aux" not in tables

    def test_bm25_asks_for_the_keyword_tiers_and_vector_does_not(self) -> None:
        """Driven through `_run`, so the wiring between query type and tier list is what is
        tested rather than the helper in isolation."""
        from podcast_scraper.search.backend import SearchQuery

        asked: List[str] = []

        class _B(LanceDBBackend):
            def __init__(self) -> None:
                pass

            def _fresh_read(self, tier: str, run: Any) -> Any:  # type: ignore[override]
                asked.append(tier)
                return None

        b = _B()
        b.search_bm25(SearchQuery(text="hola", embedding=[], tier="all", k=5))
        assert "segment_nonen" in asked

        asked.clear()
        b.search_vector(SearchQuery(text="hola", embedding=[0.1], tier="all", k=5))
        assert "segment_nonen" not in asked

    def test_a_failing_keyword_tier_read_returns_NOTHING_rather_than_unfiltered_rows(
        self,
    ) -> None:
        """If a filter names a column the tier lacks, the read is skipped. Retrying without the
        filter would LEAK rows past it, which is worse than returning none."""
        from podcast_scraper.search.backend import SearchQuery

        class _Table:
            def search(self, *_a: Any, **_k: Any) -> Any:
                raise RuntimeError("no such column")

        class _B(LanceDBBackend):
            def __init__(self) -> None:
                pass

            def _fresh_read(self, tier: str, run: Any) -> Any:  # type: ignore[override]
                return run(_Table()) if tier == "segment_nonen" else None

        got = _B().search_bm25(SearchQuery(text="hola", embedding=[], tier="all", k=5))
        assert got == []


class TestWhatFEEDSTheRouter:
    """What language the ANALYSIS chunks are labelled with — which D-44 makes a constant, not a
    measurement.

    THIS CLASS USED TO INFER IT. `_indexed_text_language` resolved the transcript and inspected the
    resulting filename for an `.en` suffix, to work out whether it had read a translation. The
    inference went wrong twice: first by building `english_transcript_relpath(resolved)` on an
    already-resolved path and testing a name that can never exist (`ep1.en.adfree.en.txt`), then by
    reading the suffix stack. Measured 2026-09-30: a successfully translated Spanish episode's
    chunks
    came from the English render and were labelled `es`, so the router dropped their embeddings into
    the vector-less tier and the episode was findable by neither semantic search nor its own
    language — invisible, because the chunking and the upsert both "succeeded".

    Under D-44 the ANALYSIS body is the canonical `<base>.txt` or its ad-free derivative, and both
    hold the analysis language whatever the episode was spoken in. So there is nothing to infer, and
    the half-translated states these tests enumerated cannot exist: the swap is indivisible.
    """

    @staticmethod
    def _episode(tmp_path: Any, files: List[str], language: Optional[str]) -> Any:
        """One episode on disk, with exactly the bodies named. Kept from the class this replaced."""
        from pathlib import Path

        root = Path(str(tmp_path)).resolve()
        (root / "transcripts").mkdir(parents=True, exist_ok=True)
        for name in files:
            (root / "transcripts" / name).write_text("body", encoding="utf-8")
        doc = {
            "episode": {"episode_id": "ep1", "language": language},
            "feed": {"feed_id": "f1", "language": language},
            "content": {"transcript_file_path": "transcripts/ep1.txt"},
        }
        return root, doc

    def test_a_SWAPPED_episode_indexes_as_the_target_language(self, tmp_path: Any) -> None:
        """The canonical body holds the translation, proven by the tagged source existing."""
        from podcast_scraper.languages import TARGET_LANGUAGE
        from podcast_scraper.search.indexer import _indexed_text_language

        for language in ("es", "it", "de"):
            root, doc = self._episode(
                tmp_path / language, ["ep1.txt", f"ep1.{language}.txt"], language
            )
            meta = root / "metadata" / "ep1.metadata.json"
            assert _indexed_text_language(root, meta, doc) == TARGET_LANGUAGE, language

    def test_an_UNSWAPPED_episode_indexes_as_its_OWN_language(self, tmp_path: Any) -> None:
        """The correction that matters. A pending or failed translation leaves the canonical body in
        the SOURCE language, so labelling it English would file Spanish text in the English vector
        tier — the exact bug this function exists to prevent, from the other direction.

        My first version asserted the target language unconditionally, on the strength of a
        docstring claiming unusable episodes never reach the indexer. That marker does not exist
        yet, and the indexer can establish the fact itself with one `isfile`, so it does.
        """
        from podcast_scraper.search.indexer import _indexed_text_language

        root, doc = self._episode(tmp_path, ["ep1.txt", "ep1.adfree.txt"], "es")
        meta = root / "metadata" / "ep1.metadata.json"
        assert _indexed_text_language(root, meta, doc) == "es"

    def test_an_english_episode_needs_no_swap(self, tmp_path: Any) -> None:
        from podcast_scraper.languages import TARGET_LANGUAGE
        from podcast_scraper.search.indexer import _indexed_text_language

        root, doc = self._episode(tmp_path, ["ep1.txt"], "en")
        meta = root / "metadata" / "ep1.metadata.json"
        assert _indexed_text_language(root, meta, doc) == TARGET_LANGUAGE

    def test_the_SOURCE_layer_is_labelled_by_the_caller_that_resolved_it(
        self, tmp_path: Any
    ) -> None:
        """The other half, and where the episode's own language legitimately appears: the caller
        deliberately resolves the tagged source body, so it knows the language without asking a
        filename."""
        from podcast_scraper.search.indexer import _source_layer_body

        root, doc = self._episode(tmp_path, ["ep1.txt", "ep1.es.txt"], "es")
        got = _source_layer_body(root, doc, "en")
        assert got is not None
        path, language = got
        assert path.name == "ep1.es.txt"
        assert language == "es"


# `TestTheEnglishRenderPredicate` lived here and is gone (D-44, #2254). It tested
# `is_english_render_relpath`, which inspected a path's suffix STACK to decide whether it
# was a translation — `.en` could sit under `.adfree` or be outermost — so the predicate had
# to strip repeatedly. English is now the UNSUFFIXED canonical file, so no path needs
# classifying and the predicate has no question to answer. What it was protecting — the
# router labelling English chunks `es` and dropping their embeddings — is now impossible by
# construction, and asserted as such in `TestWhatFEEDSTheRouter` above.


class TestAReindexDoesNotDELETETheNonEnglishTier:
    """A full reindex over an EXISTING index must not empty `segments_nonen` (S2.9 regression).

    THE BUG, MEASURED. The rows were written and then deleted, in the same build:

    * `_flush_tier` records the LOGICAL tier it was called with — `segment` — in
      `overwritten_tiers`. `replace_segments`/`upsert_segments` then write to TWO physical
      tables, `segments` and `segments_nonen`, because the language split lives inside the
      backend. Nothing ever put `segment_nonen` in that set.
    * `_finalize_reindex_clear` computes `pre_existing_tiers - overwritten_tiers` and
      MVCC-empties the difference, so `segment_nonen` was always in the difference and always
      cleared — after its rows had been written.

    WHY IT WAS INVISIBLE. A first build (no index on disk yet) takes neither path:
    `_plan_reindex_clear` returns no reason, so `_finalize_reindex_clear` never runs and the
    rows survive. The failure needs an index to already exist, which is every reindex in
    production and no test here. The read path then fails silently too — `segments_nonen` with
    zero rows gets no FTS index (`create_indices` skips empty tables), so BM25 raises
    "Cannot perform full text search unless an INVERTED index has been created", and
    `_run_keyword_only_tier` swallows that at `logger.debug`. Net effect: a verbatim sentence
    from an episode's own Spanish transcript returned 612 rows, none of them that episode's
    source layer.

    This is what "an episode findable in the language it was spoken in" (Goal 6) rests on, so
    it is asserted against a REAL index on disk rather than against the splitter.
    """

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

    def test_replacing_segments_counts_as_overwriting_BOTH_physical_tables(self) -> None:
        """The fix stated as a property, so it cannot regress by someone re-deriving the set.

        `segment` and `segment_nonen` are written by one call and must be bookkept as one unit.
        """
        assert "segment_nonen" in LanceDBBackend.physical_tiers_written_with("segment")
        assert "segment" in LanceDBBackend.physical_tiers_written_with("segment")
        # A tier with no split is just itself — no special-casing leaks to the other tiers.
        assert LanceDBBackend.physical_tiers_written_with("insight") == ("insight",)


class TestAScopedTranscriptSearchStillReachesTheSourceLayer:
    """`doc_types=["transcript"]` must not exclude the non-English tier.

    `_tables_for_tier` added the keyword-only tiers ONLY for `tier == "all"`; every scoped
    request returned `[tier]`. So the episode-level transcript search — the one surface whose
    entire job is "find this in the transcript" — could never reach non-English content, and a
    verbatim sentence from a Spanish episode's own body matched nothing in it.

    `segment_nonen` IS a segment tier; that it has no vector column decides which SIGNAL can
    read it (BM25 only), not which SCOPE it belongs to. Those are different questions and the
    code was answering the second with the first.
    """

    def test_a_segment_scoped_keyword_read_includes_the_non_english_tier(self) -> None:
        tables = LanceDBBackend._tables_for_tier(
            LanceDBBackend.__new__(LanceDBBackend), "segment", keyword=True
        )
        assert "segment_nonen" in tables, (
            "a transcript-scoped keyword search cannot see non-English content: " f"{tables}"
        )

    def test_a_segment_scoped_DENSE_read_still_excludes_it(self) -> None:
        """The vector-less tier has nothing for a dense query to match, so it is not opened."""
        tables = LanceDBBackend._tables_for_tier(
            LanceDBBackend.__new__(LanceDBBackend), "segment", keyword=False
        )
        assert "segment_nonen" not in tables

    def test_an_insight_scope_is_unaffected(self) -> None:
        """Only the segment scope has a non-English counterpart; nothing else gains a table."""
        for keyword in (True, False):
            tables = LanceDBBackend._tables_for_tier(
                LanceDBBackend.__new__(LanceDBBackend), "insight", keyword=keyword
            )
            assert tables == ["insight"]


class TestASearchHitSaysWhichLANGUAGEAndLAYERItCameFrom:
    """A non-English hit must be identifiable as one, or no client can label it.

    The two layers are now both searchable, which creates a question the response could not
    answer: a result set mixes an episode's English analysis chunks with its source-language
    ones, and `metadata` carried neither `language` nor `index_layer`. The transcript language
    control has to know which it is holding — "search in the original" is not a feature if the
    answer comes back unlabelled.

    Both are ADDITIVE and absent-by-default, exactly as the index rows are: absent `index_layer`
    means analysis (right for every row written before the field existed), and absent `language`
    means the English tier, which stores no language column by design.

    `language` IS A ROW COLUMN; `index_layer` IS NOT, YET. `_segment_nonen_schema` carries
    `language`, so a real source-layer hit arrives labelled — that is what the language control
    needs and it is verified end to end on the fixture corpus. `index_layer` currently lives only
    in the `metadata.json` sidecar: putting it on the row means a `LANCE_SCHEMA_VERSION` bump and
    a forced rebuild of every corpus, which is not worth doing for a field no surface reads yet.
    The projection below handles it the moment the column exists, and the test passes it
    explicitly rather than pretending a real row supplies it.
    """

    @staticmethod
    def _row(payload: Dict[str, Any], tier: str = "segment") -> Dict[str, Any]:
        from podcast_scraper.search.backend import ScoredResult
        from podcast_scraper.search.hybrid_search import _to_search_result

        base = {"text": "t", "episode_id": "ep1", "show_id": "p10"}
        result = ScoredResult(
            doc_id="chunk:x:0",
            score=1.0,
            rank=1,
            payload={**base, **payload},
            signal="bm25",
            source_tier=tier,
        )
        return _to_search_result(result).metadata

    def test_a_source_layer_hit_carries_its_language_and_layer(self) -> None:
        """`index_layer` is supplied explicitly here — see the class docstring: it is not a row
        column yet, so this pins the projection, not the index."""
        md = self._row({"language": "es", "index_layer": "source"}, tier="segment_nonen")
        assert md["language"] == "es"
        assert md["index_layer"] == "source"

    def test_an_english_hit_carries_neither(self) -> None:
        """Absent, not `"en"`/`"analysis"` — the English tier has no such columns, and inventing
        values here would claim a provenance the index never recorded."""
        md = self._row({})
        assert "language" not in md
        assert "index_layer" not in md
