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


def test_warns_when_pyannote_tuning_knobs_set_against_dgx(caplog) -> None:
    # D1 (#1295): the DGX/cloud providers do not apply the pyannote clustering/squelch knobs, so a
    # set-but-ignored knob must not be a silent no-op.
    cfg = config.Config(
        rss="https://example.com/feed.xml",
        transcription_provider="whisper",
        diarization_provider="tailnet_dgx",
        diarization_min_segment_ms=500,
    )
    with caplog.at_level("WARNING"):
        factory._warn_if_tuning_knobs_ignored(cfg, "tailnet_dgx")
    assert "diarization_min_segment_ms" in caplog.text
    assert "#1295" in caplog.text


def test_no_tuning_knob_warning_for_local_backend(caplog) -> None:
    cfg = config.Config(
        rss="https://example.com/feed.xml",
        transcription_provider="whisper",
        diarization_provider="local",
        diarization_min_segment_ms=500,
    )
    with caplog.at_level("WARNING"):
        factory._warn_if_tuning_knobs_ignored(cfg, "local")
    assert "1295" not in caplog.text
