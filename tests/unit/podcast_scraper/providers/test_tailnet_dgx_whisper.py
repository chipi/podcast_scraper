"""Unit tests for TailnetDgxWhisperTranscriptionProvider (ADR-096)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper import Config
from podcast_scraper.providers.tailnet_dgx.whisper_provider import (
    TailnetDgxWhisperTranscriptionProvider,
)


def _dgx_cfg() -> Config:
    return Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "transcription_provider": "tailnet_dgx_whisper",
            "transcription_fallback_provider": "openai",
            "dgx_tailnet_host": "dgx-llm-1.tail-test.ts.net",
            "openai_api_key": "sk-test",
        }
    )


def test_config_rejects_dgx_without_fallback() -> None:
    with pytest.raises(ValueError, match="transcription_fallback_provider"):
        Config.model_validate(
            {
                "rss_url": "https://example.com/feed.xml",
                "transcription_provider": "tailnet_dgx_whisper",
                "dgx_tailnet_host": "dgx-llm-1.tail-test.ts.net",
            }
        )


def test_gemini_fallback_requires_api_key() -> None:
    with pytest.raises(ValueError, match="transcription_fallback"):
        Config.model_validate(
            {
                "rss_url": "https://example.com/feed.xml",
                "transcription_provider": "tailnet_dgx_whisper",
                "transcription_fallback_provider": "gemini",
                "dgx_tailnet_host": "dgx-llm-1.tail-test.ts.net",
            }
        )


def test_mistral_fallback_requires_api_key() -> None:
    with pytest.raises(ValueError, match="transcription_fallback"):
        Config.model_validate(
            {
                "rss_url": "https://example.com/feed.xml",
                "transcription_provider": "tailnet_dgx_whisper",
                "transcription_fallback_provider": "mistral",
                "dgx_tailnet_host": "dgx-llm-1.tail-test.ts.net",
            }
        )


def test_nested_transcription_yaml_flattens() -> None:
    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "transcription": {
                "primary": "tailnet_dgx_whisper",
                "fallback": "openai",
            },
            "dgx_tailnet_host": "dgx-llm-1.tail-test.ts.net",
            "openai_api_key": "sk-test",
        }
    )
    assert cfg.transcription_provider == "tailnet_dgx_whisper"
    assert cfg.transcription_fallback_provider == "openai"


def test_initialize_constructs_real_cloud_fallback() -> None:
    # The other tests mock provider._fallback and force _initialized; this is the only
    # test that drives the REAL initialize() round-trip: model_dump() -> swap
    # transcription_provider to the fallback name -> model_validate() -> the real
    # create_transcription_provider() factory. It guards two regressions the mocked
    # tests can't see: (1) the re-validated fallback config failing to construct
    # (validation drift), and (2) the fallback resolving back to tailnet_dgx_whisper
    # and recursing. OpenAIProvider.initialize() is a no-op (no network), so this is
    # offline.
    from podcast_scraper.providers.openai.openai_provider import OpenAIProvider

    provider = TailnetDgxWhisperTranscriptionProvider(_dgx_cfg())
    provider.initialize()

    assert provider._initialized is True
    assert isinstance(provider._fallback, OpenAIProvider)
    assert provider._fallback.cfg.transcription_provider == "openai"


@patch("podcast_scraper.providers.tailnet_dgx.whisper_provider.emit_dgx_fallback_breadcrumb")
@patch(
    "podcast_scraper.providers.tailnet_dgx.whisper_provider.check_faster_whisper_health",
    return_value=False,
)
@patch("podcast_scraper.providers.tailnet_dgx.whisper_provider.time.sleep")
def test_falls_back_when_dgx_unhealthy(
    mock_sleep: MagicMock,
    _health: MagicMock,
    _breadcrumb: MagicMock,
    tmp_path,
) -> None:
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"\x00\x01")

    cfg = _dgx_cfg()
    provider = TailnetDgxWhisperTranscriptionProvider(cfg)
    fallback = MagicMock()
    fallback.transcribe_with_segments.return_value = (
        {"text": "cloud text", "segments": [], "language": "en"},
        1.0,
    )
    provider._fallback = fallback
    provider._initialized = True

    text = provider.transcribe(str(audio))
    assert text == "cloud text"
    fallback.transcribe_with_segments.assert_called_once()
    _breadcrumb.assert_called_once()
    mock_sleep.assert_called_once()


@patch("podcast_scraper.providers.tailnet_dgx.whisper_provider.emit_dgx_fallback_breadcrumb")
@patch(
    "podcast_scraper.providers.tailnet_dgx.whisper_provider.check_faster_whisper_health",
    side_effect=[False, True],
)
@patch("podcast_scraper.providers.tailnet_dgx.whisper_provider.time.sleep")
@patch.object(TailnetDgxWhisperTranscriptionProvider, "_transcribe_dgx")
def test_retries_once_before_success(
    mock_ollama: MagicMock,
    mock_sleep: MagicMock,
    _health: MagicMock,
    _breadcrumb: MagicMock,
    tmp_path,
) -> None:
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"\x00\x01")
    mock_ollama.return_value = ("dgx text", [], 0.5)

    provider = TailnetDgxWhisperTranscriptionProvider(_dgx_cfg())
    provider._fallback = MagicMock()
    provider._initialized = True

    assert provider.transcribe(str(audio)) == "dgx text"
    mock_sleep.assert_called_once()
    mock_ollama.assert_called_once()
    _breadcrumb.assert_not_called()


@patch(
    "podcast_scraper.providers.tailnet_dgx.whisper_provider.check_faster_whisper_health",
    return_value=True,
)
@patch.object(TailnetDgxWhisperTranscriptionProvider, "_transcribe_dgx")
def test_healthy_dgx_path(
    mock_ollama: MagicMock,
    _health: MagicMock,
    tmp_path,
) -> None:
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"\x00\x01")
    mock_ollama.return_value = ("from dgx", [{"start": 0}], 1.2)

    provider = TailnetDgxWhisperTranscriptionProvider(_dgx_cfg())
    provider._fallback = MagicMock()
    provider._initialized = True

    assert provider.transcribe(str(audio)) == "from dgx"
    provider._fallback.transcribe_with_segments.assert_not_called()


@patch("podcast_scraper.providers.tailnet_dgx.whisper_provider.emit_dgx_fallback_breadcrumb")
@patch(
    "podcast_scraper.providers.tailnet_dgx.whisper_provider.check_faster_whisper_health",
    return_value=True,
)
@patch.object(TailnetDgxWhisperTranscriptionProvider, "_transcribe_dgx")
def test_falls_back_when_ollama_returns_empty(
    mock_ollama: MagicMock,
    _health: MagicMock,
    _breadcrumb: MagicMock,
    tmp_path,
) -> None:
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"\x00\x01")
    mock_ollama.side_effect = ValueError("empty transcription from DGX faster-whisper-server")

    provider = TailnetDgxWhisperTranscriptionProvider(_dgx_cfg())
    fallback = MagicMock()
    fallback.transcribe_with_segments.return_value = (
        {"text": "fallback", "segments": [], "language": "en"},
        2.0,
    )
    provider._fallback = fallback
    provider._initialized = True

    assert provider.transcribe(str(audio)) == "fallback"
    _breadcrumb.assert_called_once()


@patch("httpx.Client")
def test_transcribe_dgx_parses_response(mock_client_cls: MagicMock, tmp_path) -> None:
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"abc")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "text": " hello ",
        "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
    }
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post.return_value = mock_resp
    mock_client_cls.return_value = mock_client

    provider = TailnetDgxWhisperTranscriptionProvider(_dgx_cfg())
    text, segments, _dur = provider._transcribe_dgx(str(audio), "en")
    assert text == "hello"
    assert len(segments) == 1


def test_whisper_provider_cleanup_calls_fallback() -> None:
    provider = TailnetDgxWhisperTranscriptionProvider(_dgx_cfg())
    fallback = MagicMock()
    provider._fallback = fallback
    provider.cleanup()
    fallback.cleanup.assert_called_once()


def test_transcribe_dgx_missing_file() -> None:
    provider = TailnetDgxWhisperTranscriptionProvider(_dgx_cfg())
    with pytest.raises(FileNotFoundError):
        provider._transcribe_dgx("/no/such/audio.mp3", None)


def test_transcribe_dgx_requires_httpx(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"abc")
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "httpx":
            raise ImportError("no httpx")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    provider = TailnetDgxWhisperTranscriptionProvider(_dgx_cfg())
    with pytest.raises(RuntimeError, match="httpx required"):
        provider._transcribe_dgx(str(audio), None)
