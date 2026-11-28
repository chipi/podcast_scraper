"""A source of a test-only text type runs end to end through the text-in entry point (ADR-177).

The type and its transport exist only here: a plain text type over a fixture transport that
lists items already carrying their text. The run goes through the one dispatch point
(``content_types.run_source``), with no network, no audio and no model, and must:

- write each item's body under ``transcripts/`` and a metadata document that records
  ``content_type`` (decision 6), with the item id as its guid;
- rewrite the run summary at the corpus root, so the ``corpus_mtime`` cache token moves;
- keep the transport's cursor beside the runs, so the next run lists only newer items;
- never reach transcription, diarization or ad cleaning (decision 4).
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pytest

from podcast_scraper import config, content_types as ct
from podcast_scraper.extensions import Extension, use_extensions
from podcast_scraper.rss.feeds_spec import merge_feed_entry_into_config, RssFeedEntry

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]

TEXT = ct.ContentType(
    name="test_text_item",
    id_scheme="item",
    stage_profile=ct.StageProfile(name="test_text_item", stages=frozenset({ct.SUMMARY, ct.KG})),
    default_transport="fixture_feed",
    source_kind="test",
    card_kind="article",
)

ITEMS = [
    ct.SourceItem(
        item_id="one",
        metadata={"title": "First item", "published_at": "2026-11-01T09:00:00Z"},
        text="The first item's body. It is already text.",
        url="https://example.com/one",
    ),
    ct.SourceItem(
        item_id="two",
        metadata={"title": "Second item"},
        text="The second item's body.",
    ),
    ct.SourceItem(item_id="three", url="https://example.com/three"),  # no text
]


def _transport(listed: List[Optional[str]]) -> ct.Transport:
    def list_items(source: Any, cursor: Optional[str]) -> Tuple[List[ct.SourceItem], str]:
        listed.append(cursor)
        start = int(cursor or 0)
        return ITEMS[start:], str(len(ITEMS))

    return ct.Transport(name="fixture_feed", list_items=list_items)


def _ext(transport: ct.Transport) -> Extension:
    return Extension(
        name="fixture-text-type",
        content_types=lambda: [TEXT],
        transports=lambda: [transport],
    )


def _cfg(corpus: Path, entry: RssFeedEntry, **overrides: Any) -> config.Config:
    fields: dict = {
        "rss": entry.url,
        "output_dir": str(corpus),
        "generate_metadata": True,
        "generate_summaries": False,
    }
    base = config.Config(**{**fields, **overrides})
    return merge_feed_entry_into_config(base, entry)


def _metadata_docs(corpus: Path) -> List[dict]:
    return [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted(corpus.rglob("*.metadata.json"))
    ]


def test_a_text_source_runs_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = tmp_path / "corpus"
    entry = RssFeedEntry(url="https://example.com/fixture-source", content_type="test_text_item")
    listed: List[Optional[str]] = []

    def never(what: str) -> Any:
        def boom(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError(f"a text run must never reach {what}")

        return boom

    from podcast_scraper import builtin_stage_hooks
    from podcast_scraper.workflow import orchestration

    monkeypatch.setattr(
        builtin_stage_hooks,
        "BUILTIN",
        dataclasses.replace(
            builtin_stage_hooks.BUILTIN,
            adfree_base=never("ad cleaning"),
            excise_ads=never("ad cleaning"),
            remove_commercials=never("ad cleaning"),
            excise_crosspromo=never("ad cleaning"),
            episode_insights=never("GI"),
        ),
    )
    monkeypatch.setattr(orchestration, "_create_transcription_provider", never("transcription"))
    monkeypatch.setattr(orchestration, "_fetch_and_prepare_episodes", never("the feed fetch"))

    with use_extensions([_ext(_transport(listed))]):
        count, summary = ct.run_source(_cfg(corpus, entry), entry)

    assert listed == [None]
    assert count == 2, summary

    docs = _metadata_docs(corpus)
    assert len(docs) == 2
    assert {d["content_type"] for d in docs} == {"test_text_item"}
    assert {d["content"]["transcript_source"] for d in docs} == {"text_body"}
    assert sorted(d["episode"]["guid"] for d in docs) == [
        "item:test_text_item:one",
        "item:test_text_item:two",
    ]
    by_guid = {d["episode"]["guid"]: d for d in docs}
    assert by_guid["item:test_text_item:one"]["episode"]["published_date"].startswith("2026-11-01")

    bodies = sorted(p.read_text(encoding="utf-8") for p in corpus.rglob("transcripts/*.txt"))
    assert bodies == sorted(i.text for i in ITEMS if i.text)

    cursor = json.loads((corpus / "source_cursor.json").read_text(encoding="utf-8"))
    assert cursor == {"transport": "fixture_feed", "cursor": "3"}


def test_a_batch_of_a_text_source_moves_the_cache_token_and_resumes_from_its_cursor(
    tmp_path: Path,
) -> None:
    """Through the multi-feed loop, as a feeds spec names it: the batch rewrites the corpus run
    summary (the ``corpus_mtime`` token moves), and a second batch lists only what is new."""
    from podcast_scraper import cli
    from podcast_scraper.perf_cache import corpus_mtime

    corpus = tmp_path / "corpus"
    spec = tmp_path / "feeds.spec.yaml"
    spec.write_text(
        "feeds:\n  - url: https://example.com/fixture-source\n    content_type: test_text_item\n",
        encoding="utf-8",
    )
    listed: List[Optional[str]] = []

    def podcast(cfg: Any) -> Tuple[int, str]:
        raise AssertionError("the podcast pipeline must not run for a text source")

    argv = ["--feeds-spec", str(spec), "--output-dir", str(corpus), "--no-transcript-cache"]
    with use_extensions([_ext(_transport(listed))]):
        assert cli.main(argv, run_pipeline_fn=podcast) == 0
        summary = corpus / "corpus_run_summary.json"
        assert summary.is_file()
        first = corpus_mtime(corpus)
        assert first == summary.stat().st_mtime
        assert len(_metadata_docs(corpus)) == 2

        os.utime(summary, (first - 60, first - 60))
        assert cli.main(argv, run_pipeline_fn=podcast) == 0
    assert corpus_mtime(corpus) > first - 60
    assert listed == [None, "3"]
    assert len(_metadata_docs(corpus)) == 2  # the second batch listed nothing new
    # A non-podcast source lives under sources/<type>/, never feeds/ (podcast-only walkers skip
    # sources/); the batch's run summary stays at the corpus root.
    assert not (corpus / "feeds").exists()
    metas = list(corpus.rglob("*.metadata.json"))
    assert metas and all(
        p.relative_to(corpus).parts[:2] == ("sources", "test_text_item") for p in metas
    )
    from podcast_scraper.utils import filesystem

    workspace = Path(
        filesystem.corpus_source_output_dir(
            str(corpus), "https://example.com/fixture-source", "test_text_item"
        )
    )
    assert all(workspace in p.parents for p in metas)
    assert (workspace / "source_cursor.json").is_file()
    # The corpus walkers leave sources/ out unless asked, so a podcast reader sees none of it.
    from podcast_scraper.utils.corpus_walk import corpus_rglob

    assert list(corpus_rglob(corpus, "*.metadata.json")) == []
    assert len(list(corpus_rglob(corpus, "*.metadata.json", include_sources=True))) == 2


def test_a_text_items_kg_is_written_in_kg_3(tmp_path: Path) -> None:
    """The type's profile runs KG; its kg.json is KG 3.0 for the item (ontology v3), valid against
    kg.v3.schema.json, with the ids the type minted."""
    from podcast_scraper.kg.schema import load_v3_schema, validate_artifact

    assert load_v3_schema() is not None  # strict validation below is real, not a no-op
    corpus = tmp_path / "corpus"
    url = "https://example.com/fixture-source"
    entry = RssFeedEntry(url=url, content_type="test_text_item")
    cfg = _cfg(corpus, entry, generate_kg=True, kg_extraction_source="metadata_only")
    with use_extensions([_ext(_transport([]))]):
        count, summary = ct.run_source(cfg, entry)
    assert count == 2, summary

    kgs = sorted(corpus.rglob("*.kg.json"))
    assert len(kgs) == 2
    source = ct.source_id(TEXT, url)
    by_item = {}
    for path in kgs:
        art = json.loads(path.read_text(encoding="utf-8"))
        validate_artifact(art, strict=True)
        by_item[art["item_id"]] = art
    assert set(by_item) == {"item:test_text_item:one", "item:test_text_item:two"}
    one = by_item["item:test_text_item:one"]
    assert one["schema_version"] == "3.0" and one["content_type"] == "test_text_item"
    nodes = {n["id"]: n for n in one["nodes"]}
    assert nodes[source]["properties"] == {
        "title": url,
        "source_kind": "test",
        "transport": "fixture_feed",
        "url": url,
    }
    item = nodes["item:test_text_item:one"]["properties"]
    assert item["source_id"] == source
    assert item["published_at"] == "2026-11-01T09:00:00Z"
    assert item["url"] == "https://example.com/one"
    assert one["extraction"]["document_ref"].startswith("transcripts/")
    assert one["extraction"]["variant"] == "raw"

    docs = {d["episode"]["guid"]: d for d in _metadata_docs(corpus)}
    assert docs["item:test_text_item:one"]["knowledge_graph"]["schema_version"] == "3.0"


def test_a_text_sources_workspace_resolves_to_its_corpus_root(tmp_path: Path) -> None:
    """The corpus-root resolvers know sources/TYPE/SLUG as well as feeds/SLUG: a batch child of
    another type is a batch child, and its manifest stamp and skip-existing scope are the corpus."""
    from podcast_scraper.utils import filesystem
    from podcast_scraper.workflow import corpus_operations, run_index

    corpus = tmp_path / "corpus"
    url = "https://example.com/fixture-source"
    workspace = filesystem.corpus_source_output_dir(str(corpus), url, "test_text_item")
    cfg = config.Config(rss=url, output_dir=workspace)
    root = str(Path(filesystem.validate_and_normalize_output_dir(str(corpus))))
    assert corpus_operations.is_multi_feed_batch_child(cfg)
    assert corpus_operations.corpus_parent_for_manifest_stamp_from_cfg(cfg) == root
    assert run_index.corpus_root_from_cfg(cfg) == root


def test_a_service_run_text_source_moves_the_cache_token(tmp_path: Path) -> None:
    """A text source given through a service config stamps the corpus manifest at the corpus
    root, so the corpus_mtime token the caches key on moves, as for a podcast feed."""
    from podcast_scraper import service
    from podcast_scraper.perf_cache import corpus_mtime

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "feeds").mkdir()  # an existing corpus: the parent holds feeds/
    url = "https://example.com/fixture-source"
    cfg = config.Config.model_validate(
        {
            "rss_urls": [{"url": url, "content_type": "test_text_item"}],
            "output_dir": str(corpus),
            "single_feed_uses_corpus_layout": True,
            "generate_metadata": True,
            "generate_summaries": False,
        }
    )
    before = corpus_mtime(corpus)
    with use_extensions([_ext(_transport([]))]):
        result = service.run(cfg)
    assert result.success, result.error
    assert (corpus / "corpus_manifest.json").is_file()
    assert corpus_mtime(corpus) > before
    assert not (corpus / "sources" / "test_text_item" / "corpus_manifest.json").exists()


DATED = [
    ct.SourceItem(
        item_id=f"n{n}", text=f"Item {n}'s body.", cursor=str(n), metadata={"title": f"Item {n}"}
    )
    for n in (11, 10, 9, 8)  # newest first; cursor values compare as numbers (9 < 10)
]


def _dated_transport(listed: List[Optional[str]], cap: Optional[int] = None) -> ct.Transport:
    def list_items(source: Any, cursor: Optional[str]) -> Tuple[List[ct.SourceItem], str]:
        listed.append(cursor)
        after = int(cursor) if cursor else 0
        return [i for i in DATED if int(i.cursor or 0) > after], "11"

    return ct.Transport(
        name="fixture_feed",
        list_items=list_items,
        cursor_key=int,
        max_items=None if cap is None else (lambda source: cap),
    )


def _fail_for(monkeypatch: pytest.MonkeyPatch, titles: List[str]) -> None:
    """The metadata step raises for these items, as a summary does when its provider is down."""
    from podcast_scraper.workflow.stages import processing

    real = processing.metadata_stage.call_generate_metadata

    def flaky(*args: Any, **kwargs: Any) -> Any:
        if kwargs["episode"].title in titles:
            raise RuntimeError("Failed to generate summary using provider: connection refused")
        return real(*args, **kwargs)

    monkeypatch.setattr(processing.metadata_stage, "call_generate_metadata", flaky)


def _cursor(corpus: Path) -> Optional[str]:
    path = corpus / "source_cursor.json"
    return json.loads(path.read_text(encoding="utf-8"))["cursor"] if path.is_file() else None


def _guids(corpus: Path) -> List[str]:
    return sorted(d["episode"]["guid"].rsplit(":", 1)[-1] for d in _metadata_docs(corpus))


def test_an_item_that_failed_mid_run_is_listed_again_and_written_next_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2026-11-28: summaries failed while the LLM endpoint was down, and the cursor passed the
    failed articles. Here item 10 fails between items that write, older (8, 9) and newer (11):
    the cursor stops at 9, the run index counts 10 failed, and the next run writes it."""
    corpus = tmp_path / "corpus"
    entry = RssFeedEntry(url="https://example.com/fixture-source", content_type="test_text_item")
    listed: List[Optional[str]] = []
    with use_extensions([_ext(_dated_transport(listed))]):
        with monkeypatch.context() as m:
            _fail_for(m, ["Item 10"])
            ct.run_source(_cfg(corpus, entry), entry)
        assert _guids(corpus) == ["n11", "n8", "n9"]
        assert _cursor(corpus) == "9"
        index = json.loads(next(corpus.rglob("index.json")).read_text(encoding="utf-8"))
        assert (index["episodes_processed"], index["episodes_failed"]) == (3, 1)

        ct.run_source(_cfg(corpus, entry), entry)
    assert listed == [None, "9"]
    assert _guids(corpus) == ["n10", "n11", "n8", "n9"]  # 11 was skipped as already written
    assert _cursor(corpus) == "11"


def test_when_every_item_fails_the_cursor_stays_exactly_where_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "source_cursor.json").write_text(
        json.dumps({"transport": "fixture_feed", "cursor": "7"}), encoding="utf-8"
    )
    entry = RssFeedEntry(url="https://example.com/fixture-source", content_type="test_text_item")
    _fail_for(monkeypatch, [str(i.metadata["title"]) for i in DATED])
    with use_extensions([_ext(_dated_transport([]))]):
        ct.run_source(_cfg(corpus, entry), entry)
    assert _guids(corpus) == []
    assert _cursor(corpus) == "7"


def test_items_past_the_cap_wait_for_a_later_run_and_written_ones_do_not_count(
    tmp_path: Path,
) -> None:
    """A cap of 2 over four new items: two runs write all four, newest first, because the items
    the first run wrote do not count toward the second run's cap."""
    corpus = tmp_path / "corpus"
    entry = RssFeedEntry(url="https://example.com/fixture-source", content_type="test_text_item")
    listed: List[Optional[str]] = []
    with use_extensions([_ext(_dated_transport(listed, cap=2))]):
        ct.run_source(_cfg(corpus, entry), entry)
        assert _guids(corpus) == ["n10", "n11"]
        assert _cursor(corpus) is None  # 8 and 9 wait, and nothing older wrote
        ct.run_source(_cfg(corpus, entry), entry)
    assert _guids(corpus) == ["n10", "n11", "n8", "n9"]
    assert _cursor(corpus) == "11"
    assert listed == [None, None]


def test_a_capped_item_and_a_failed_one_in_one_run_hold_the_cursor_below_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cap 2 over 11, 10, 9, 8: 11 fails, 10 writes, 9 and 8 wait past the cap. The oldest
    unwritten item is 8 and nothing older wrote, so the cursor stays; the next run writes the
    other three (11 again, 9 and 8)."""
    corpus = tmp_path / "corpus"
    entry = RssFeedEntry(url="https://example.com/fixture-source", content_type="test_text_item")
    with use_extensions([_ext(_dated_transport([], cap=2))]):
        with monkeypatch.context() as m:
            _fail_for(m, ["Item 11"])
            ct.run_source(_cfg(corpus, entry), entry)
        assert _guids(corpus) == ["n10"]
        assert _cursor(corpus) is None
        ct.run_source(_cfg(corpus, entry), entry)
        ct.run_source(_cfg(corpus, entry), entry)
    assert _guids(corpus) == ["n10", "n11", "n8", "n9"]
    assert _cursor(corpus) == "11"


# --- the body's fetch, the park and the fuse: the podcast path's own mechanisms ---


def _fetching_type(outcomes: dict) -> ct.ContentType:
    """The fixture type with a ``fetch_body``: ``outcomes`` maps an item id to what its next
    fetches return, in turn (a NotFetched, else the item gets a body); unlisted ids get one."""

    def fetch_body(item: ct.SourceItem, source: Any) -> ct.FetchedBody:
        queue = outcomes.get(item.item_id)
        got = queue.pop(0) if queue else None
        if isinstance(got, ct.NotFetched):
            return got
        return dataclasses.replace(item, text=f"{item.item_id}'s body.")

    return dataclasses.replace(TEXT, fetch_body=fetch_body)


def _url_item(n: int) -> ct.SourceItem:
    return ct.SourceItem(
        item_id=f"u{n}",
        url=f"https://example.com/u{n}",
        cursor=str(n),
        metadata={"title": f"Url {n}"},
    )


def _feed_transport(feed: List[ct.SourceItem]) -> ct.Transport:
    def list_items(source: Any, cursor: Optional[str]) -> Tuple[List[ct.SourceItem], str]:
        after = int(cursor) if cursor else 0
        listed = [i for i in feed if int(i.cursor or 0) > after]
        return sorted(listed, key=lambda i: -int(i.cursor or 0)), str(
            max(int(i.cursor or 0) for i in feed)
        )

    return ct.Transport(name="fixture_feed", list_items=list_items, cursor_key=int)


def _with(ctype: ct.ContentType, transport: ct.Transport) -> Extension:
    return Extension(
        name="fixture-fetching-type",
        content_types=lambda: [ctype],
        transports=lambda: [transport],
    )


def _incidents(corpus: Path) -> List[dict]:
    path = corpus / "corpus_incidents.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _batch_cfg(corpus: Path, entry: RssFeedEntry) -> config.Config:
    """As a multi-feed batch has it: the incident log, and so the park's state, at the corpus."""
    return _cfg(corpus, entry, incident_log_path=str(corpus / "corpus_incidents.jsonl"))


def test_a_transient_fetch_failure_waits_and_a_permanent_one_is_passed(tmp_path: Path) -> None:
    """u12's page answers 503 (transient: the downloader's retry rule), u11's 404 (permanent).
    The cursor stops at 11, below u12 and past u11; the soft and policy incidents are recorded;
    the next run fetches u12 and writes it."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    entry = RssFeedEntry(url="https://example.com/fixture-source", content_type="test_text_item")
    outcomes = {
        "u12": [ct.NotFetched("http_503", transient=True)],
        "u11": [ct.NotFetched("http_404", transient=False)],
    }
    feed = [_url_item(13), _url_item(12), _url_item(11)]
    with use_extensions([_with(_fetching_type(outcomes), _feed_transport(feed))]):
        ct.run_source(_batch_cfg(corpus, entry), entry)
        assert _guids(corpus) == ["u13"]
        assert _cursor(corpus) == "11"
        rows = {(r["category"], r["stage"], r["message"]) for r in _incidents(corpus)}
        assert ("soft", "fetch", "not fetched: http_503") in rows
        assert ("policy", "fetch", "not fetched: http_404") in rows
        ct.run_source(_batch_cfg(corpus, entry), entry)
    assert _guids(corpus) == ["u12", "u13"]
    assert _cursor(corpus) == "13"


def test_an_item_whose_fetch_keeps_failing_is_parked_as_a_podcast_episode_is(
    tmp_path: Path,
) -> None:
    """episode_park, under stage ``fetch``: u12 fails in two batches in which another item's
    fetch passed, so the batch's commit parks it. The third batch skips it before its fetch, and
    a parked item no longer holds the cursor."""
    from podcast_scraper.workflow import episode_park

    episode_park._reset_for_tests()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    log = str(corpus / "corpus_incidents.jsonl")
    entry = RssFeedEntry(url="https://example.com/fixture-source", content_type="test_text_item")
    down = ct.NotFetched("http_503", transient=True)
    outcomes = {"u12": [down, down, down]}
    feed = [_url_item(13), _url_item(12)]
    with use_extensions([_with(_fetching_type(outcomes), _feed_transport(feed))]):
        ct.run_source(_batch_cfg(corpus, entry), entry)
        assert episode_park.commit_batch(log)["counted"] is True
        feed.insert(0, _url_item(14))
        ct.run_source(_batch_cfg(corpus, entry), entry)
        parked = episode_park.commit_batch(log)["parked"]
        assert parked == [ct.item_id(TEXT, _url_item(12))]
        feed.insert(0, _url_item(15))
        ct.run_source(_batch_cfg(corpus, entry), entry)
    assert outcomes["u12"] == [down]  # the third batch never fetched it
    assert _guids(corpus) == ["u13", "u14", "u15"]
    assert _cursor(corpus) == "15"
    assert any(r["exception_type"] == "EpisodeParked" for r in _incidents(corpus))


def test_a_batch_where_no_fetch_passed_counts_no_fetch_failure(tmp_path: Path) -> None:
    """The podcast park's outage rule: the source's host was down for every item, so nothing
    is counted and nothing is parked."""
    from podcast_scraper.workflow import episode_park

    episode_park._reset_for_tests()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    entry = RssFeedEntry(url="https://example.com/fixture-source", content_type="test_text_item")
    down = ct.NotFetched("http_0", transient=True)
    outcomes = {"u13": [down], "u12": [down]}
    with use_extensions(
        [_with(_fetching_type(outcomes), _feed_transport([_url_item(13), _url_item(12)]))]
    ):
        ct.run_source(_batch_cfg(corpus, entry), entry)
    result = episode_park.commit_batch(str(corpus / "corpus_incidents.jsonl"))
    assert result == {"counted": False, "parked": [], "cleared": []}
    assert _cursor(corpus) is None


def test_an_item_whose_summary_keeps_failing_is_parked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under stage ``metadata``, cap 2: item 10's metadata fails in a batch where 11 wrote, then
    in one where 9 wrote; the second commit parks it. The third batch skips it, writes 8, and
    the parked item no longer holds the cursor."""
    from podcast_scraper.workflow import episode_park

    episode_park._reset_for_tests()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    log = str(corpus / "corpus_incidents.jsonl")
    entry = RssFeedEntry(url="https://example.com/fixture-source", content_type="test_text_item")
    _fail_for(monkeypatch, ["Item 10"])
    with use_extensions([_ext(_dated_transport([], cap=2))]):
        ct.run_source(_batch_cfg(corpus, entry), entry)  # 11 writes, 10 fails; 9, 8 wait
        assert episode_park.commit_batch(log)["parked"] == []
        ct.run_source(_batch_cfg(corpus, entry), entry)  # 10 fails, 9 writes; 8 waits
        assert episode_park.commit_batch(log)["parked"] == ["item:test_text_item:n10"]
        ct.run_source(_batch_cfg(corpus, entry), entry)  # 10 skipped as parked, 8 writes
    assert _guids(corpus) == ["n11", "n8", "n9"]
    assert _cursor(corpus) == "11"


def test_an_open_resilience_fuse_halts_the_run_and_keeps_the_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR-122, through the real summary path: the provider's fuse opens on the first item.
    The run halts (as a podcast batch does) instead of failing every item, and the cursor file
    is never written, so the next run lists all of them again."""
    from podcast_scraper.providers.resilience import ResilienceFuseOpenError
    from podcast_scraper.workflow import orchestration

    class Held:
        """A provider whose endpoint stayed down past its hold budget (only ``summarize``)."""

        def summarize(self, *args: Any, **kwargs: Any) -> Any:
            raise ResilienceFuseOpenError("llm:vllm: overload persisted 905s (> 900s hold budget)")

    monkeypatch.setattr(orchestration, "_create_summarization_provider", lambda cfg: Held())
    corpus = tmp_path / "corpus"
    entry = RssFeedEntry(url="https://example.com/fixture-source", content_type="test_text_item")
    long = [
        dataclasses.replace(i, text=f"{i.text} " + "A sentence long enough to summarise. " * 80)
        for i in DATED
    ]

    def list_items(source: Any, cursor: Optional[str]) -> Tuple[List[ct.SourceItem], str]:
        return long, "11"

    transport = ct.Transport(name="fixture_feed", list_items=list_items, cursor_key=int)
    with use_extensions([_ext(transport)]):
        with pytest.raises(ResilienceFuseOpenError):
            ct.run_source(_cfg(corpus, entry, generate_summaries=True), entry)
    assert _cursor(corpus) is None
    assert _guids(corpus) == []
