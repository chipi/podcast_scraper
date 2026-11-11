"""Content types and transports as registered seams, and the one dispatch point (ADR-177)."""

from __future__ import annotations

import hashlib
from typing import Any, List, Tuple

import pytest

from podcast_scraper import config, content_types as ct
from podcast_scraper.extensions import Extension, use_extensions
from podcast_scraper.rss.feeds_spec import merge_feed_entry_into_config, RssFeedEntry

pytestmark = pytest.mark.unit

TEXT = ct.ContentType(
    name="test_text_item",
    id_scheme="item",
    stage_profile=ct.StageProfile(name="text", stages=frozenset({"summary", "kg"})),
    default_transport="fixture_feed",
    source_kind="test",
)
FEED = ct.Transport(name="fixture_feed", list_items=lambda source, cursor: ([], cursor))


def _ext(types: Tuple[Any, ...] = (), transports: Tuple[Any, ...] = ()) -> Extension:
    return Extension(
        name="types-test", content_types=lambda: list(types), transports=lambda: list(transports)
    )


def test_podcast_is_the_built_in_type_over_rss_enclosures() -> None:
    with use_extensions([]):
        types, transports = ct.registry()
    assert list(types) == [ct.PODCAST_EPISODE]
    assert list(transports) == [ct.RSS_ENCLOSURE]
    assert ct.PODCAST.stage_profile.runs("diarization")  # the full pipeline runs every stage


def test_a_source_naming_no_type_is_a_podcast() -> None:
    with use_extensions([]):
        assert ct.content_type(None) is ct.PODCAST


def test_an_unknown_type_is_refused() -> None:
    with use_extensions([]):
        with pytest.raises(ct.UnknownContentType):
            ct.content_type("nothing_registers_this")


def test_an_extension_adds_a_type_and_its_transport() -> None:
    with use_extensions([_ext((TEXT,), (FEED,))]):
        assert ct.content_type("test_text_item") is TEXT
        _, transports = ct.registry()
    assert transports["fixture_feed"] is FEED
    assert TEXT.stage_profile.runs("summary") and not TEXT.stage_profile.runs("diarization")


def test_a_type_registered_twice_is_refused() -> None:
    with use_extensions([_ext((TEXT, TEXT), (FEED,))]):
        with pytest.raises(ct.RegistryError, match="twice"):
            ct.registry()


def test_a_built_in_name_cannot_be_registered_again() -> None:
    with use_extensions([_ext((ct.PODCAST,))]):
        with pytest.raises(ct.RegistryError, match="twice"):
            ct.registry()


def test_a_type_naming_an_unregistered_transport_is_refused() -> None:
    with use_extensions([_ext((TEXT,))]):
        with pytest.raises(ct.RegistryError, match="fixture_feed"):
            ct.registry()


def test_a_podcast_source_runs_the_pipeline_unchanged() -> None:
    calls: List[Any] = []

    def run(cfg: Any) -> Tuple[int, str]:
        calls.append(cfg)
        return 3, "ok"

    cfg = object()
    with use_extensions([]):
        assert ct.run_source(cfg, RssFeedEntry(url="https://x/f.xml"), run_pipeline_fn=run) == (
            3,
            "ok",
        )
        assert ct.run_source(cfg, None, run_pipeline_fn=run) == (3, "ok")
    assert calls == [cfg, cfg]


def test_a_text_source_runs_the_text_in_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    from podcast_scraper.workflow import text_in

    def run(cfg: Any) -> Tuple[int, str]:
        raise AssertionError("the podcast pipeline must not run for a text type")

    seen: List[Any] = []

    def text_run(cfg: Any, entry: Any, ctype: Any, transport: Any) -> Tuple[int, str]:
        seen.append((cfg, entry, ctype, transport))
        return 1, "text"

    monkeypatch.setattr(text_in, "run_text_source", text_run)
    cfg = object()
    entry = RssFeedEntry(url="https://x/f.xml", content_type="test_text_item")
    with use_extensions([_ext((TEXT,), (FEED,))]):
        assert ct.run_source(cfg, entry, run_pipeline_fn=run) == (1, "text")
    assert seen == [(cfg, entry, TEXT, FEED)]


def test_a_source_naming_an_unregistered_transport_is_refused() -> None:
    entry = RssFeedEntry(url="https://x/f.xml", content_type="test_text_item", transport="nope")
    with use_extensions([_ext((TEXT,), (FEED,))]):
        with pytest.raises(ct.RegistryError, match="nope"):
            ct.run_source(object(), entry)


def test_a_transport_that_lists_nothing_itself_cannot_feed_text_in() -> None:
    from podcast_scraper.workflow.text_in import run_text_source

    with pytest.raises(ct.NoTextRunner):
        run_text_source(config.Config(), None, TEXT, ct.Transport(name="bare"))


def test_a_profile_naming_an_unknown_stage_is_refused() -> None:
    with pytest.raises(ct.RegistryError, match="diarisation"):
        ct.StageProfile(name="typo", stages=frozenset({"diarisation"}))


def test_the_current_items_profile_decides_which_stages_run() -> None:
    assert all(ct.stage_runs(s) for s in ct.STAGES)  # no job: a podcast runs every stage
    with ct.processing_item(ct.ItemContext(content_type=TEXT)):
        assert ct.stage_runs(ct.SUMMARY) and ct.stage_runs(ct.KG)
        assert not ct.stage_runs(ct.GI)
        assert not ct.stage_runs(ct.AD_CLEANING)
        assert not ct.stage_runs(ct.TRANSLATION)
    assert ct.stage_runs(ct.GI)


def test_ad_cleaning_and_gi_hooks_do_not_run_for_a_type_without_them() -> None:
    from podcast_scraper import stage_hooks

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("this hook must not run for the text type")

    hooks = stage_hooks.PipelineHooks(
        name="gate-test",
        adfree_base=boom,
        excise_ads=boom,
        remove_commercials=boom,
        excise_crosspromo=boom,
        episode_insights=boom,
    )
    ext = Extension(name="gate-test", pipeline_hooks=hooks)
    with use_extensions([ext]), ct.processing_item(ct.ItemContext(content_type=TEXT)):
        assert stage_hooks.adfree_base("t", None, "r", "o") is None
        text, segs, excision = stage_hooks.excise_ads("t")
        assert (text, segs, excision.source_length) == ("t", None, 1)
        assert stage_hooks.remove_commercials("t") == "t"
        assert stage_hooks.excise_crosspromo("t") == "t"
        assert stage_hooks.episode_insights(cfg=None) is None


def test_the_sources_type_and_transport_never_reach_config() -> None:
    entry = RssFeedEntry(
        url="https://x/f.xml", content_type="test_text_item", transport="fixture_feed", timeout=7
    )
    assert entry.override_update_dict() == {"timeout": 7}
    cfg = merge_feed_entry_into_config(config.Config(rss="https://x/other.xml"), entry)
    assert cfg.timeout == 7
    assert not hasattr(cfg, "content_type") and not hasattr(cfg, "transport")


def test_the_type_mints_the_item_id_from_the_transports_id() -> None:
    assert ct.item_id(TEXT, ct.SourceItem(item_id="abc")) == "item:test_text_item:abc"


def test_a_url_identified_type_mints_from_its_canonical_url() -> None:
    by_url = ct.ContentType(
        name="by_url",
        id_scheme="item",
        stage_profile=ct.StageProfile(name="u"),
        default_transport="fixture_feed",
        source_kind="test",
        identity=lambda item: ct.url_identity(item.url or ""),
    )
    over_rss = ct.SourceItem(item_id="guid-1", url="https://example.com/a")
    over_api = ct.SourceItem(item_id="api-77", url="https://example.com/a")
    minted = ct.item_id(by_url, over_rss)
    assert minted == ct.item_id(by_url, over_api)  # one article over two transports: one item

    assert minted == "item:by_url:" + hashlib.sha256(b"https://example.com/a").hexdigest()[:20]
    assert len(minted.rsplit(":", 1)[1]) == 20


def test_an_item_without_an_identity_is_refused() -> None:
    with pytest.raises(ValueError, match="no identity"):
        ct.item_id(TEXT, ct.SourceItem(item_id=""))


def test_a_document_without_a_type_reads_as_a_podcast_episode() -> None:
    assert ct.item_content_type({"episode": {}}) == ct.PODCAST_EPISODE
    assert ct.item_content_type({"content_type": " test_text_item "}) == "test_text_item"
    assert ct.item_content_type(None) == ct.PODCAST_EPISODE
    assert ct.is_wanted({}, ct.PODCAST_ONLY)
    assert not ct.is_wanted({"content_type": "test_text_item"}, ct.PODCAST_ONLY)


def test_a_source_kind_outside_the_vocabulary_is_refused() -> None:
    with pytest.raises(ct.RegistryError, match="source_kind"):
        ct.ContentType(
            name="t",
            id_scheme="item",
            stage_profile=ct.StageProfile(name="t"),
            default_transport="fixture_feed",
            source_kind="test_text_item",
        )
    assert ct.PODCAST.source_kind == "podcast"


def test_the_type_mints_the_source_id_from_its_url() -> None:
    url = "https://example.com/fixture-source"
    assert ct.source_id(TEXT, url) == "source:" + hashlib.sha256(url.encode()).hexdigest()[:20]
    with pytest.raises(ValueError, match="no URL"):
        ct.source_id(TEXT, "")
