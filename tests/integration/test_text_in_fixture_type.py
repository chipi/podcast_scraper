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

pytestmark = [pytest.mark.integration]

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
    base = config.Config(
        rss=entry.url,
        output_dir=str(corpus),
        generate_metadata=True,
        generate_summaries=False,
        **overrides,
    )
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
