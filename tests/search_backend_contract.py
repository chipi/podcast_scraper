"""One behaviour contract, run against EVERY `SearchBackend` implementation.

WHY THIS EXISTS. Seven unit test files hand-roll their own fake backend (`_FakeBackend`,
`_FakeHybridBackend`, `_B`, …), each implementing whatever subset of the port that file happens to
need. Those fakes are legitimate — what is under test at unit level is OUR logic consuming a
backend's results, not the vendor's storage, and a small real implementation of the port preserves
that assertion where a `MagicMock` would not. What was missing is any check that a fake behaves
like the real thing. A fake can drift from LanceDB indefinitely and nothing notices: the unit tests
keep passing, because they are agreeing with the fake rather than checking it.

THIS SUITE IS THE MISSING HALF. It is written strictly against
`podcast_scraper.search.backend.SearchBackend` and knows nothing about pyarrow, lance or tables.
Two files run it:

* `tests/unit/search/test_fake_backend_contract.py` — against the in-memory fake, with no
  `[search]` extra. Proves our stand-in honours the contract the real one is held to.
* `tests/integration/search/test_lancedb_backend_contract.py` — against the real `LanceDBBackend`
  in a temp directory. Proves the contract describes reality.

A behaviour belongs here when our code depends on it AND a fake could plausibly get it wrong. It
does NOT belong here if it is about the storage *format* — field names and column types are the
adapter's business, covered by the declaration in `backend.py` plus one conformance test.

=== SHAPE: NAMED CHECKS, NOT AN INHERITED MIXIN ===

Each behaviour is a function taking ``(make_backend, prepare)``, collected in `CONTRACT_CHECKS`,
and each runner parameterizes one real test function over them. The first draft was a mixin of
`test_*` methods that runners inherited, which broke two repo rules at once:

* `check_test_policy.py` rule G1 counts `def test_*` **in the file**, so a runner that inherited
  everything and defined nothing read as an empty test file;
* narrowing `prepare_for_search(self, backend: object)` to `LanceDBBackend` in a subclass is an
  unsound override, and mypy said so.

Plain functions have neither problem, and the parameterized ids name the behaviour in failure
output instead of an index.

=== IT EARNED ITS PLACE IMMEDIATELY ===

Run against both implementations on 2026-10-03 it found two real bugs in the real backend — the
same blind spot in two places, neither reproducible by a hand-written fake:

* `delete(tier="all")` resolved to `DENSE_TIERS` — `segment`, `insight`, `aux` — omitting
  `segment_nonen`, so a source-language row survived a delete-all while the method's own docstring
  promised "removes from every table";
* `health()` reported those same three tiers by name, so a corpus whose only indexed content was
  non-English looked empty.

It also caught the suite encoding ITS OWN FAKE: without `prepare`, ten checks failed against the
real backend with "Cannot perform full text search unless an INVERTED index has been created".
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

from podcast_scraper.search.backend import SearchQuery, SegmentDocument

#: Embedding width used throughout. Small on purpose — nothing here tests vector quality, only
#: whether a row with a vector can be reached by a dense signal and a row without one cannot.
DIM = 8

#: Builds a fresh backend.
MakeBackend = Callable[[], Any]

#: Makes written rows answerable.
#:
#: PART OF THE CONTRACT, not a test fixup. LanceDB answers no full-text query until an INVERTED
#: index exists (`create_indices()`, which the production indexer runs at the end of a build),
#: while the in-memory fake has no index concept and needs nothing. Each implementation supplies
#: its own; omitting the step is how the first draft came to describe the fake instead of the
#: contract.
Prepare = Callable[[Any], None]


def segment(
    doc_id: str,
    text: str = "drainage and soil structure",
    language: Optional[str] = None,
    *,
    embedding: Optional[List[float]] = None,
) -> SegmentDocument:
    """A segment document. ``language=None``/``"en"`` routes English; anything else is source."""
    return SegmentDocument(
        id=doc_id,
        text=text,
        show_id="p10",
        episode_id="ep1",
        start_time=0.0,
        end_time=5.0,
        embedding=[0.1] * DIM if embedding is None else embedding,
        language=language,
    )


def _written(make_backend: MakeBackend, prepare: Prepare, *docs: SegmentDocument) -> Any:
    """A backend holding *docs*, made answerable — the order every check needs."""
    backend = make_backend()
    for doc in docs:
        backend.upsert_segment(doc)
    prepare(backend)
    return backend


def _bm25(backend: Any, text: str) -> List[str]:
    return [h.doc_id for h in backend.search_bm25(SearchQuery(text=text, embedding=[0.0] * DIM))]


def _vector(backend: Any) -> List[str]:
    return [h.doc_id for h in backend.search_vector(SearchQuery(text="", embedding=[0.1] * DIM))]


# --- what a row is for ---------------------------------------------------------------------


def check_an_upserted_segment_is_findable_by_keyword(m: MakeBackend, p: Prepare) -> None:
    """The baseline. A fake that stored nothing would pass several checks below without it."""
    backend = _written(m, p, segment("ep1_chunk_0", text="drainage decides everything"))
    assert _bm25(backend, "drainage") == ["ep1_chunk_0"]


def check_a_keyword_miss_returns_nothing_rather_than_everything(m: MakeBackend, p: Prepare) -> None:
    """A fake that ignored the query text would return the row anyway, and the check above would
    still pass — which is why both directions are here."""
    backend = _written(m, p, segment("ep1_chunk_0", text="drainage decides everything"))
    assert _bm25(backend, "xylophone") == []


# --- the non-English tier's whole guarantee ------------------------------------------------


def check_a_source_language_row_is_reachable_by_keyword(m: MakeBackend, p: Prepare) -> None:
    """It must still be findable — that is the point of indexing it at all (Goal 6)."""
    backend = _written(m, p, segment("ep1_chunk_0:src", text="drenaje del suelo", language="es"))
    assert _bm25(backend, "drenaje") == ["ep1_chunk_0:src"]


def check_a_source_row_can_never_be_reached_by_a_dense_signal(m: MakeBackend, p: Prepare) -> None:
    """The guarantee that justified a separate tier instead of a language column plus a filter: a
    row with no vector CANNOT appear in a semantic result. A filter is a promise every query path
    has to remember; this is a property of the storage.

    The fake must enforce it STRUCTURALLY — by having no vector to match — not by filtering on
    language, or it is not modelling the thing that makes the design safe.
    """
    backend = _written(
        m,
        p,
        segment("ep1_chunk_0", language="en"),
        segment("ep1_chunk_0:src", text="drenaje", language="es"),
    )
    assert "ep1_chunk_0:src" not in set(_vector(backend))


def check_english_and_absent_language_both_reach_the_dense_signal(
    m: MakeBackend, p: Prepare
) -> None:
    """`None` is the majority of the corpus — every episode indexed before language resolution
    existed — so it must behave exactly like an explicit `"en"`."""
    backend = _written(
        m, p, segment("explicit_en", language="en"), segment("absent_lang", language=None)
    )
    assert {"explicit_en", "absent_lang"} <= set(_vector(backend))


# --- deletion ------------------------------------------------------------------------------


def check_delete_removes_the_row_from_its_own_tier(m: MakeBackend, p: Prepare) -> None:
    backend = _written(m, p, segment("ep1_chunk_0", text="drainage"))
    backend.delete("ep1_chunk_0", "segment")
    assert _bm25(backend, "drainage") == []


def check_delete_all_removes_the_row_from_every_tier(m: MakeBackend, p: Prepare) -> None:
    """``delete(tier="all")`` must mean ALL, keyword-only tier included.

    THIS IS THE CHECK THAT FOUND A REAL BUG (2026-10-03). `LanceDBBackend.delete` resolved `"all"`
    to `DENSE_TIERS`, omitting `segment_nonen` — so a Spanish row survived a delete-all while the
    docstring promised otherwise. Withdrawal, reindex and episode removal all inherited it.
    """
    backend = _written(
        m,
        p,
        segment("ep1_chunk_0", text="drainage", language="en"),
        segment("ep1_chunk_0:src", text="drenaje", language="es"),
    )
    backend.delete("ep1_chunk_0", "all")
    backend.delete("ep1_chunk_0:src", "all")
    assert _bm25(backend, "drainage") == []
    assert _bm25(backend, "drenaje") == [], (
        "a source-language row survived delete(tier='all') — the tier list for 'all' is missing "
        "the keyword-only tier"
    )


def check_deleting_an_absent_id_is_a_no_op(m: MakeBackend, p: Prepare) -> None:
    """Called on every reprocess, so it must neither raise nor take anything else out."""
    backend = _written(m, p, segment("ep1_chunk_0", text="drainage"))
    backend.delete("never_indexed", "all")
    assert _bm25(backend, "drainage") == ["ep1_chunk_0"]


# --- upsert semantics ----------------------------------------------------------------------


def check_upserting_the_same_id_twice_leaves_one_row(m: MakeBackend, p: Prepare) -> None:
    """ "Upsert", not "append". A duplicate row would double a chunk's weight in every downstream
    count and show the episode twice in results."""
    backend = _written(
        m,
        p,
        segment("ep1_chunk_0", text="drainage first"),
        segment("ep1_chunk_0", text="drainage second"),
    )
    assert _bm25(backend, "drainage") == ["ep1_chunk_0"]


def check_the_second_upsert_wins(m: MakeBackend, p: Prepare) -> None:
    backend = _written(
        m, p, segment("ep1_chunk_0", text="drainage"), segment("ep1_chunk_0", text="benching")
    )
    assert _bm25(backend, "drainage") == []
    assert _bm25(backend, "benching") == ["ep1_chunk_0"]


# --- health --------------------------------------------------------------------------------


def check_health_accounts_for_every_tier_that_has_rows(m: MakeBackend, p: Prepare) -> None:
    """An operator reading `health` must not be told a populated tier is empty.

    `health()` is on the port, so it is contract surface. THE SECOND BUG THIS FOUND: the real
    backend reported segments / insights / aux by name and never `segment_nonen`, so a corpus whose
    only indexed content was non-English looked like a corpus with nothing indexed.
    """
    backend = _written(
        m,
        p,
        segment("en_row", text="drainage", language="en"),
        segment("src_row", text="drenaje", language="es"),
    )
    health = backend.health()
    assert health.get("status") == "ok"
    reported = sum(v for v in health.values() if isinstance(v, int))
    assert reported >= 2, (
        f"health reports {reported} rows across all tiers but two were written — a populated "
        f"tier is missing from the report: {health}"
    )


# --- results carry what callers read -------------------------------------------------------


def check_a_result_reports_the_tier_it_came_from(m: MakeBackend, p: Prepare) -> None:
    """Callers branch on `source_tier` — the two-layer reader uses it to decide whether a hit is
    analysis-language or source-language content."""
    backend = _written(
        m,
        p,
        segment("en_row", text="drainage", language="en"),
        segment("src_row", text="drenaje", language="es"),
    )
    by_id = {
        h.doc_id: h
        for q in ("drainage", "drenaje")
        for h in backend.search_bm25(SearchQuery(text=q, embedding=[0.0] * DIM))
    }
    assert by_id["en_row"].source_tier == "segment"
    assert by_id["src_row"].source_tier == "segment_nonen"


def check_every_result_declares_its_signal(m: MakeBackend, p: Prepare) -> None:
    backend = _written(m, p, segment("ep1_chunk_0", text="drainage"))
    assert all(
        h.signal == "bm25"
        for h in backend.search_bm25(SearchQuery(text="drainage", embedding=[0.0] * DIM))
    )
    assert all(
        h.signal == "vector"
        for h in backend.search_vector(SearchQuery(text="", embedding=[0.1] * DIM))
    )


#: The contract. Every `SearchBackend` implementation must satisfy all of it.
CONTRACT_CHECKS: Tuple[Tuple[str, Callable[[MakeBackend, Prepare], None]], ...] = tuple(
    (fn.__name__[len("check_") :], fn)
    for fn in (
        check_an_upserted_segment_is_findable_by_keyword,
        check_a_keyword_miss_returns_nothing_rather_than_everything,
        check_a_source_language_row_is_reachable_by_keyword,
        check_a_source_row_can_never_be_reached_by_a_dense_signal,
        check_english_and_absent_language_both_reach_the_dense_signal,
        check_delete_removes_the_row_from_its_own_tier,
        check_delete_all_removes_the_row_from_every_tier,
        check_deleting_an_absent_id_is_a_no_op,
        check_upserting_the_same_id_twice_leaves_one_row,
        check_the_second_upsert_wins,
        check_health_accounts_for_every_tier_that_has_rows,
        check_a_result_reports_the_tier_it_came_from,
        check_every_result_declares_its_signal,
    )
)

#: Parameter ids, so a failure names the behaviour rather than an index.
CONTRACT_IDS: Tuple[str, ...] = tuple(name for name, _ in CONTRACT_CHECKS)
