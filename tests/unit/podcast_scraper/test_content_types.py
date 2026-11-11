"""Content types and transports as registered seams, and the one dispatch point (ADR-177)."""

from __future__ import annotations

from typing import Any, List, Tuple

import pytest

from podcast_scraper import config, content_types as ct
from podcast_scraper.extensions import Extension, use_extensions
from podcast_scraper.rss.feeds_spec import merge_feed_entry_into_config, RssFeedEntry

pytestmark = pytest.mark.unit

TEXT = ct.ContentType(
    name="fixture_text",
    id_scheme="item",
    stage_profile=ct.StageProfile(name="text", stages=frozenset({"summary", "kg"})),
    default_transport="fixture_feed",
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
        assert ct.content_type("fixture_text") is TEXT
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


def test_a_text_source_does_not_reach_the_podcast_pipeline() -> None:
    def run(cfg: Any) -> Tuple[int, str]:
        raise AssertionError("the podcast pipeline must not run for a text type")

    entry = RssFeedEntry(url="https://x/f.xml", content_type="fixture_text")
    with use_extensions([_ext((TEXT,), (FEED,))]):
        with pytest.raises(ct.NoTextRunner):
            ct.run_source(object(), entry, run_pipeline_fn=run)


def test_the_sources_type_and_transport_never_reach_config() -> None:
    entry = RssFeedEntry(
        url="https://x/f.xml", content_type="fixture_text", transport="fixture_feed", timeout=7
    )
    assert entry.override_update_dict() == {"timeout": 7}
    cfg = merge_feed_entry_into_config(config.Config(rss="https://x/other.xml"), entry)
    assert cfg.timeout == 7
    assert not hasattr(cfg, "content_type") and not hasattr(cfg, "transport")
