"""Unit tests for diarization factory token resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper import config
from podcast_scraper.providers.ml.diarization import factory

pytestmark = pytest.mark.unit


def _cfg() -> config.Config:
    return config.Config(rss="https://example.com/feed.xml", transcription_provider="whisper")


def test_resolve_hf_token_prefers_config(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cfg = config.Config(
        rss="https://example.com/feed.xml", transcription_provider="whisper", hf_token="hf_cfg"
    )
    assert factory.resolve_hf_token(cfg) == "hf_cfg"


def test_resolve_hf_token_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert factory.resolve_hf_token(_cfg()) == "hf_env"


def test_resolve_hf_token_reads_modern_cache_path(monkeypatch, tmp_path: Path) -> None:
    """The token must be found at ~/.cache/huggingface/token (modern HF CLI location)."""
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    cache_token = tmp_path / ".cache" / "huggingface" / "token"
    cache_token.parent.mkdir(parents=True)
    cache_token.write_text("hf_cache\n", encoding="utf-8")
    monkeypatch.setattr(factory.Path, "home", classmethod(lambda cls: tmp_path))

    assert factory.resolve_hf_token(_cfg()) == "hf_cache"


def test_resolve_hf_token_none_when_absent(monkeypatch, tmp_path: Path) -> None:
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(factory.Path, "home", classmethod(lambda cls: tmp_path))
    assert factory.resolve_hf_token(_cfg()) is None
