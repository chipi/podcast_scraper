"""Unit tests for TailnetDgxWhisperTranscriptionProvider (ADR-096)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper import Config
from podcast_scraper.providers.tailnet_dgx import whisper_provider as wp
from podcast_scraper.providers.tailnet_dgx.whisper_provider import (
    TailnetDgxWhisperTranscriptionProvider,
)


@pytest.fixture(autouse=True)
def _reset_whisper_breaker():
    """The DGX Whisper circuit breaker is process-wide; isolate every test."""
    wp._whisper_breaker.reset()
    yield
    wp._whisper_breaker.reset()


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


def test_initialize_is_pure_dgx_no_self_fallback() -> None:
    """RFC-106 (#1198): the provider no longer builds or owns a cloud fallback — the FallbackChain
    that wraps it does. initialize() just validates the host is present and marks it ready; it must
    not construct any provider (that would recreate the double-fallback we retired)."""
    provider = TailnetDgxWhisperTranscriptionProvider(_dgx_cfg())
    provider.initialize()
    assert provider._initialized is True
    assert not hasattr(provider, "_fallback") or provider.__dict__.get("_fallback") is None


@patch("podcast_scraper.providers.tailnet_dgx.whisper_provider.emit_dgx_fallback_breadcrumb")
@patch(
    "podcast_scraper.providers.tailnet_dgx.whisper_provider.check_faster_whisper_health",
    return_value=False,
)
@patch("podcast_scraper.providers.tailnet_dgx.whisper_provider.time.sleep")
def test_raises_when_dgx_unhealthy(
    mock_sleep: MagicMock,
    _health: MagicMock,
    _breadcrumb: MagicMock,
    tmp_path,
) -> None:
    """RFC-106: a persistently unhealthy DGX exhausts this tier and RAISES (the chain, not this
    provider, decides whether to fall back). The DGX-try resilience is unchanged: it still rides
    out transient health blips with 2 backoff sleeps before giving up."""
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"\x00\x01")

    provider = TailnetDgxWhisperTranscriptionProvider(_dgx_cfg())
    provider._initialized = True

    with pytest.raises(RuntimeError, match="DGX Whisper unavailable"):
        provider.transcribe(str(audio))
    _breadcrumb.assert_called_once()
    # #876 resilience: default dgx_max_attempts=3 → 2 backoff sleeps (5s, 10s) before giving up.
    assert mock_sleep.call_count == 2
    assert [c.args[0] for c in mock_sleep.call_args_list] == [5.0, 10.0]


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
@patch("podcast_scraper.providers.tailnet_dgx.whisper_provider.time.sleep")
@patch.object(TailnetDgxWhisperTranscriptionProvider, "_transcribe_dgx")
def test_raises_when_dgx_returns_empty(
    mock_ollama: MagicMock,
    _mock_sleep: MagicMock,
    _health: MagicMock,
    _breadcrumb: MagicMock,
    tmp_path,
) -> None:
    """A DGX-side error (empty/garbage) exhausts this tier and re-raises the underlying error so
    the chain can classify it (is_infra_failure -> cascade) and try the next tier."""
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"\x00\x01")
    err = ValueError("empty transcription from DGX faster-whisper-server")
    mock_ollama.side_effect = err

    provider = TailnetDgxWhisperTranscriptionProvider(_dgx_cfg())
    provider._initialized = True

    with pytest.raises(ValueError, match="empty transcription"):
        provider.transcribe(str(audio))
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


def test_whisper_provider_cleanup_is_noop() -> None:
    """RFC-106: the provider owns no fallback, so cleanup() has nothing to release and must not
    raise (the chain cleans up its own tiers)."""
    provider = TailnetDgxWhisperTranscriptionProvider(_dgx_cfg())
    provider.cleanup()  # must not raise (no owned resources)


# ---------------------------------------------------------------------------
# Response-shape guardrail integration — #999 / ADR-099                       #
# ---------------------------------------------------------------------------


@patch("httpx.Client")
def test_guardrail_fires_on_empty_dgx_response(mock_client_cls: MagicMock, tmp_path) -> None:
    """An empty ``text`` from DGX whisper-server trips the guardrail (ADR-099 §
    Initial thresholds), raising GuardrailViolation rather than the older
    ValueError. The exception class carries service + reason for downstream
    logging + Sentry capture.
    """
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"abc")

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"text": "", "segments": []}
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post.return_value = mock_resp
    mock_client_cls.return_value = mock_client

    provider = TailnetDgxWhisperTranscriptionProvider(_dgx_cfg())
    with pytest.raises(wp.guardrails.GuardrailViolation) as exc_info:
        provider._transcribe_dgx(str(audio), "en")
    assert exc_info.value.service == "whisper"
    assert exc_info.value.reason == "empty_response"


@patch("httpx.Client")
def test_guardrail_passes_on_normal_response(mock_client_cls: MagicMock, tmp_path) -> None:
    """A response whose text matches expected length does NOT raise."""
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"\x00" * 100)  # small fake audio (probe will likely return None)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "text": " ".join(["word"] * 200),
        "segments": [],
    }
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post.return_value = mock_resp
    mock_client_cls.return_value = mock_client

    provider = TailnetDgxWhisperTranscriptionProvider(_dgx_cfg())
    text, _segments, _dur = provider._transcribe_dgx(str(audio), "en")
    assert text.count("word") == 200


def test_transcribe_dgx_missing_file() -> None:
    provider = TailnetDgxWhisperTranscriptionProvider(_dgx_cfg())
    with pytest.raises(FileNotFoundError):
        provider._transcribe_dgx("/no/such/audio.mp3", None)


# ---------------------------------------------------------------------------
# #1046 — Per-call model override (sniff-pass plumbing).
# ---------------------------------------------------------------------------


@patch("httpx.Client")
def test_transcribe_dgx_uses_model_override_when_provided(
    mock_client_cls: MagicMock, tmp_path
) -> None:
    """When ``model_override`` is passed, the multipart form's ``model``
    field carries the override (NOT ``cfg.dgx_whisper_model``). This is the
    foundational wire for the #1046 sniff-pass — the sniff orchestrator
    invokes the provider with ``small.en`` for the first pass and
    ``large-v3`` (the default) only if the gate fires.
    """
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
    provider._transcribe_dgx(str(audio), "en", model_override="Systran/faster-whisper-small.en")

    # The multipart form data submitted to the server carries the override.
    _args, kwargs = mock_client.post.call_args
    assert kwargs["data"]["model"] == "Systran/faster-whisper-small.en", (
        "model override didn't propagate to the multipart form — the "
        "sniff-pass would silently re-call the default model. Wire from "
        "_transcribe_dgx must use ``model_override or self._model``."
    )


@patch("httpx.Client")
def test_transcribe_dgx_falls_back_to_default_when_override_none(
    mock_client_cls: MagicMock, tmp_path
) -> None:
    """When ``model_override`` is None (the default path), the provider's
    configured ``cfg.dgx_whisper_model`` flows to the server. Regression
    guard against accidentally inverting the override semantics.
    """
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
    provider._transcribe_dgx(str(audio), "en")  # no override

    _args, kwargs = mock_client.post.call_args
    assert kwargs["data"]["model"] == "Systran/faster-whisper-large-v3", (
        f"default-model path regression: expected the cfg default "
        f"'Systran/faster-whisper-large-v3', got "
        f"{kwargs['data']['model']!r}"
    )


@patch("httpx.Client")
def test_transcribe_dgx_empty_override_string_falls_back_to_default(
    mock_client_cls: MagicMock, tmp_path
) -> None:
    """An empty-string ``model_override`` (the disabled-sniff-pass default
    that the Config field carries) is treated as None — the provider uses
    its configured default. This guards the orchestrator's likely
    code path of unconditionally passing ``cfg.dgx_whisper_sniff_model``
    (which is ``""`` when the operator hasn't opted into the gate).
    """
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
    provider._transcribe_dgx(str(audio), "en", model_override="")

    _args, kwargs = mock_client.post.call_args
    assert kwargs["data"]["model"] == "Systran/faster-whisper-large-v3"


@patch.object(TailnetDgxWhisperTranscriptionProvider, "_transcribe_dgx")
@patch.object(wp, "check_faster_whisper_health", return_value=True)
def test_transcribe_with_segments_threads_model_override_through(
    mock_health: MagicMock,
    mock_transcribe_dgx: MagicMock,
    tmp_path,
) -> None:
    """Public ``transcribe_with_segments`` propagates ``model_override`` to
    the lower-level ``_transcribe_dgx`` call AND records the effective
    model in the returned dict's ``model_requested`` / ``model_used``
    provenance fields. On the DGX-success path both equal the override.
    """
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"abc")
    mock_transcribe_dgx.return_value = ("text", [], 1.0)

    provider = TailnetDgxWhisperTranscriptionProvider(_dgx_cfg())
    result, _dur = provider.transcribe_with_segments(
        str(audio),
        language="en",
        model_override="Systran/faster-whisper-small.en",
    )

    # The override threaded all the way down to _transcribe_dgx.
    call_kwargs = mock_transcribe_dgx.call_args.kwargs
    assert call_kwargs.get("model_override") == "Systran/faster-whisper-small.en"
    # Provenance on DGX-success: requested == used == the override.
    assert result["model_requested"] == "Systran/faster-whisper-small.en"
    assert result["model_used"] == "Systran/faster-whisper-small.en"


@patch.object(TailnetDgxWhisperTranscriptionProvider, "_transcribe_dgx")
@patch.object(wp, "check_faster_whisper_health", return_value=True)
def test_transcribe_with_segments_default_path_records_default_model(
    mock_health: MagicMock,
    mock_transcribe_dgx: MagicMock,
    tmp_path,
) -> None:
    """No override → both provenance fields are the configured default."""
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"abc")
    mock_transcribe_dgx.return_value = ("text", [], 1.0)

    provider = TailnetDgxWhisperTranscriptionProvider(_dgx_cfg())
    result, _dur = provider.transcribe_with_segments(str(audio), language="en")
    assert result["model_requested"] == "Systran/faster-whisper-large-v3"
    assert result["model_used"] == "Systran/faster-whisper-large-v3"


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


# --- #876 resilience: duration-scaled timeout + retry classification + single-flight ---


def test_effective_timeout_scales_with_audio_duration() -> None:
    provider = TailnetDgxWhisperTranscriptionProvider(_dgx_cfg())
    base = provider._timeout_sec
    assert provider._effective_timeout_sec(None) == base  # unknown duration -> base
    # base + (minutes * per-minute budget)
    assert provider._effective_timeout_sec(3600) == base + 60 * provider._timeout_per_audio_min
    assert provider._effective_timeout_sec(5400) > provider._effective_timeout_sec(1800)


@patch("podcast_scraper.providers.tailnet_dgx.whisper_provider.emit_dgx_fallback_breadcrumb")
@patch(
    "podcast_scraper.providers.tailnet_dgx.whisper_provider.check_faster_whisper_health",
    return_value=True,
)
@patch("podcast_scraper.providers.tailnet_dgx.whisper_provider.time.sleep")
@patch.object(TailnetDgxWhisperTranscriptionProvider, "_transcribe_dgx")
def test_timeout_raises_without_repiling(
    mock_dgx: MagicMock,
    mock_sleep: MagicMock,
    _health: MagicMock,
    _breadcrumb: MagicMock,
    tmp_path,
) -> None:
    """A DGX timeout (busy GPU) must NOT retry the POST — that would pile a duplicate
    request onto the overloaded server. It raises after a single DGX attempt; the timeout is
    cascade-worthy (is_infra_failure), so the chain moves on."""
    import httpx

    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"\x00\x01")
    mock_dgx.side_effect = httpx.ReadTimeout("timed out")

    provider = TailnetDgxWhisperTranscriptionProvider(_dgx_cfg())
    provider._initialized = True

    with pytest.raises(httpx.ReadTimeout):
        provider.transcribe(str(audio))
    mock_dgx.assert_called_once()  # no re-queue on timeout
    mock_sleep.assert_not_called()  # broke immediately, no backoff sleep
    _breadcrumb.assert_called_once()


@patch("podcast_scraper.providers.tailnet_dgx.whisper_provider.emit_dgx_fallback_breadcrumb")
@patch(
    "podcast_scraper.providers.tailnet_dgx.whisper_provider.check_faster_whisper_health",
    return_value=True,
)
@patch("podcast_scraper.providers.tailnet_dgx.whisper_provider.time.sleep")
@patch.object(TailnetDgxWhisperTranscriptionProvider, "_transcribe_dgx")
def test_connection_error_retries_with_backoff_then_raises(
    mock_dgx: MagicMock,
    mock_sleep: MagicMock,
    _health: MagicMock,
    _breadcrumb: MagicMock,
    tmp_path,
) -> None:
    """A connection blip (no duplicate work in flight) is retried up to
    dgx_max_attempts with exponential backoff before raising for the chain."""
    import httpx

    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"\x00\x01")
    mock_dgx.side_effect = httpx.ConnectError("connection refused")

    provider = TailnetDgxWhisperTranscriptionProvider(_dgx_cfg())
    provider._initialized = True

    with pytest.raises(httpx.ConnectError):
        provider.transcribe(str(audio))
    assert mock_dgx.call_count == provider._max_attempts  # retried each attempt
    # exponential backoff between attempts (max_attempts-1 sleeps)
    assert mock_sleep.call_count == provider._max_attempts - 1
    _breadcrumb.assert_called_once()


@patch("podcast_scraper.providers.tailnet_dgx.whisper_provider.emit_dgx_fallback_breadcrumb")
@patch.object(TailnetDgxWhisperTranscriptionProvider, "_transcribe_dgx")
@patch("podcast_scraper.providers.tailnet_dgx.whisper_provider.check_faster_whisper_health")
def test_open_breaker_skips_dgx_entirely(
    mock_health: MagicMock,
    mock_dgx: MagicMock,
    _breadcrumb: MagicMock,
    tmp_path,
) -> None:
    """While the breaker is open, transcription must not probe health or hit DGX —
    it raises immediately (circuit-open) so the chain fails over without a wasted probe (#954)."""
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"\x00\x01")
    wp._whisper_breaker.record_failure(hard=True)  # force open

    provider = TailnetDgxWhisperTranscriptionProvider(_dgx_cfg())
    provider._initialized = True

    with pytest.raises(RuntimeError, match="dgx_whisper_circuit_open"):
        provider.transcribe(str(audio))
    mock_health.assert_not_called()
    mock_dgx.assert_not_called()
    assert _breadcrumb.call_args.kwargs["failure_reason"] == "dgx_whisper_circuit_open"


@patch("podcast_scraper.providers.tailnet_dgx.whisper_provider.emit_dgx_fallback_breadcrumb")
@patch(
    "podcast_scraper.providers.tailnet_dgx.whisper_provider.check_faster_whisper_health",
    return_value=True,
)
def test_watchdog_hard_deadline_raises(
    _health: MagicMock,
    _breadcrumb: MagicMock,
    tmp_path,
) -> None:
    """A request that hangs past the hard deadline (httpx's own timeout never fires
    under a co-tenant GPU stall) is abandoned by the watchdog, which raises a TimeoutLike error
    for the chain to fail over on, and trips the breaker (#954)."""
    import time as _time

    from podcast_scraper.providers.resilience import TimeoutLike

    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"\x00\x01")

    provider = TailnetDgxWhisperTranscriptionProvider(_dgx_cfg())
    provider._initialized = True

    def _hang(*_a, **_k):
        _time.sleep(1.0)  # daemon worker abandoned; doesn't block the test
        return ("never", [], 1.0)

    started = _time.monotonic()
    with (
        patch.object(provider, "_transcribe_dgx", side_effect=_hang),
        patch.object(provider, "_effective_timeout_sec", return_value=0.05),
        patch.object(wp.resilience, "WATCHDOG_GRACE_SEC", 0.1),
        pytest.raises(TimeoutLike),
    ):
        provider.transcribe(str(audio))
    assert _time.monotonic() - started < 0.8  # bailed at ~0.15s, not the 1s hang
    assert wp._whisper_breaker.state == "open"  # watchdog timeout trips the breaker
