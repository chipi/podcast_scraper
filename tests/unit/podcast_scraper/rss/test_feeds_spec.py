"""Unit tests for structured feeds spec (#626)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from podcast_scraper import config as cfg_mod
from podcast_scraper.rss.feeds_spec import (
    FeedsSpecDocument,
    load_feeds_spec_file,
    merge_feed_entry_into_config,
    RssFeedEntry,
)


def test_load_feeds_spec_yaml_and_json_equivalent(tmp_path: Path) -> None:
    y = tmp_path / "f.yaml"
    j = tmp_path / "f.json"
    data = {
        "feeds": [
            "https://a.example/feed.xml",
            {"url": "https://b.example/feed.xml", "timeout": 99},
        ]
    }
    y.write_text(yaml.safe_dump(data), encoding="utf-8")
    j.write_text(json.dumps(data), encoding="utf-8")
    dy = load_feeds_spec_file(y)
    dj = load_feeds_spec_file(j)
    assert dy.model_dump() == dj.model_dump()
    assert dy.feeds[1].timeout == 99


def test_unknown_top_level_key_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        yaml.safe_dump({"feeds": ["https://a.example/x"], "extra_root": 1}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown top-level"):
        load_feeds_spec_file(p)


def test_feed_entry_unknown_key_forbidden() -> None:
    with pytest.raises(ValidationError):
        RssFeedEntry.model_validate({"url": "https://a.example/x", "not_allowed": 1})


def test_merge_feed_entry_overrides_global_timeout() -> None:
    base = cfg_mod.Config(
        rss="https://ignored.example/x",
        timeout=30,
        user_agent="global-ua",
    )
    ent = RssFeedEntry(url="https://feed.example/rss", timeout=77, user_agent=None)
    merged = merge_feed_entry_into_config(base, ent)
    assert merged.rss_url == "https://feed.example/rss"
    assert merged.timeout == 77
    assert merged.user_agent == "global-ua"
    assert merged.rss_urls is None


def test_feeds_spec_document_accepts_comment_keys() -> None:
    doc = FeedsSpecDocument.model_validate(
        {"_comment": "x", "_comment_resilience": "y", "feeds": ["https://a.example/z"]}
    )
    assert len(doc.feeds) == 1


@pytest.mark.unit
def test_repo_example_feeds_specs_load() -> None:
    """Tracked ``config/examples/feeds.spec.example.*`` stay valid feeds-spec documents."""
    root = Path(__file__).resolve().parents[4]
    examples_dir = root / "config" / "examples"
    for name in ("feeds.spec.example.yaml", "feeds.spec.example.json"):
        p = examples_dir / name
        if not p.is_file():
            pytest.skip(f"{p} not present")
        doc = load_feeds_spec_file(p)
        assert doc.feeds, f"{p.name}: expected at least one feed"
