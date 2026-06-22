"""Unit tests for #1046 sniff-pass orchestrator.

Mocks the spaCy NER + the transcription provider so these run without DGX
and without the en_core_web_sm model download — the production path is
covered by the matching integration test against the real speaches
container (see scripts/measure_1046.py + the design doc § 12).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper import config as _config_mod
from podcast_scraper.workflow import sniff_gate


def _cfg(**overrides):
    base = {
        "transcription_provider": "tailnet_dgx_whisper",
        "transcription_fallback_provider": "openai",
        "openai_api_key": "sk-test",
        "language": "en",
        "dgx_tailnet_host": "dgx.example.ts.net",
        "dgx_whisper_model": "Systran/faster-whisper-large-v3",
        "dgx_whisper_sniff_model": "Systran/faster-whisper-small.en",
        "dgx_whisper_sniff_gate_min_entities": 5,
    }
    base.update(overrides)
    return _config_mod.Config.model_validate(base)


def _provider(*, sniff_text="sniff transcript text", deep_text="deep transcript"):
    """Provider mock whose transcribe_with_segments returns sniff/deep based
    on whether model_override was passed."""
    provider = MagicMock()

    def _call(media_path, language=None, model_override=None, **_kwargs):
        if model_override:
            return (
                {
                    "text": sniff_text,
                    "segments": [],
                    "language": "en",
                    "model_requested": model_override,
                    "model_used": model_override,
                },
                1.2,
            )
        return (
            {
                "text": deep_text,
                "segments": [{"start": 0.0, "end": 1.0, "text": deep_text}],
                "language": "en",
                "model_requested": "Systran/faster-whisper-large-v3",
                "model_used": "Systran/faster-whisper-large-v3",
            },
            3.4,
        )

    provider.transcribe_with_segments.side_effect = _call
    return provider


# ---------------------------------------------------------------------------
# is_enabled
# ---------------------------------------------------------------------------


def test_is_enabled_when_provider_and_model_configured():
    assert sniff_gate.is_enabled(_cfg()) is True


def test_is_enabled_false_when_sniff_model_empty():
    assert sniff_gate.is_enabled(_cfg(dgx_whisper_sniff_model="")) is False


def test_is_enabled_false_for_non_dgx_provider():
    cfg = _cfg(transcription_provider="whisper", transcription_fallback_provider="whisper")
    assert sniff_gate.is_enabled(cfg) is False


def test_is_enabled_false_when_sniff_model_whitespace_only():
    """Whitespace-only override is treated as disabled — operators sometimes
    leave a stray space when toggling envs.
    """
    assert sniff_gate.is_enabled(_cfg(dgx_whisper_sniff_model="   ")) is False


# ---------------------------------------------------------------------------
# transcribe_with_sniff_gate — happy paths
# ---------------------------------------------------------------------------


@patch.object(sniff_gate, "count_gate_entities", return_value=2)
def test_gate_below_threshold_keeps_sniff_and_skips_deep(mock_ner):
    """Entity count under threshold → return sniff transcript; deep NOT called."""
    cfg = _cfg(dgx_whisper_sniff_gate_min_entities=5)
    provider = _provider(sniff_text="short chat with 2 names")
    result, elapsed = sniff_gate.transcribe_with_sniff_gate(
        media_path="/tmp/audio.mp3",
        cfg=cfg,
        provider=provider,
    )
    # Exactly ONE provider call — the sniff. Deep was skipped.
    assert provider.transcribe_with_segments.call_count == 1
    call = provider.transcribe_with_segments.call_args
    assert call.kwargs["model_override"] == "Systran/faster-whisper-small.en"
    # Sniff transcript survives.
    assert result["text"] == "short chat with 2 names"
    # Gate provenance is honest.
    assert result["sniff_gate"]["decision"] == sniff_gate.GATE_DECISION_KEPT_SNIFF
    assert result["sniff_gate"]["entity_count"] == 2
    assert result["sniff_gate"]["threshold"] == 5
    assert result["sniff_gate"]["deep_model_skipped"] == "Systran/faster-whisper-large-v3"
    # Elapsed is sniff-only.
    assert elapsed == pytest.approx(1.2)


@patch.object(sniff_gate, "count_gate_entities", return_value=12)
def test_gate_above_threshold_runs_deep_and_returns_deep_transcript(mock_ner):
    """Entity count over threshold → deep called; result is deep transcript."""
    cfg = _cfg(dgx_whisper_sniff_gate_min_entities=5)
    provider = _provider(sniff_text="rich content", deep_text="full deep transcript")

    result, elapsed = sniff_gate.transcribe_with_sniff_gate(
        media_path="/tmp/audio.mp3",
        cfg=cfg,
        provider=provider,
    )

    # TWO provider calls — sniff (override) then deep (no override).
    assert provider.transcribe_with_segments.call_count == 2
    first_call_kwargs = provider.transcribe_with_segments.call_args_list[0].kwargs
    second_call_kwargs = provider.transcribe_with_segments.call_args_list[1].kwargs
    assert first_call_kwargs["model_override"] == "Systran/faster-whisper-small.en"
    assert (
        "model_override" not in second_call_kwargs
        or second_call_kwargs.get("model_override") is None
    )
    # Deep transcript is what's returned (NOT the sniff one).
    assert result["text"] == "full deep transcript"
    # Gate provenance.
    assert result["sniff_gate"]["decision"] == sniff_gate.GATE_DECISION_RAN_DEEP
    assert result["sniff_gate"]["entity_count"] == 12
    assert result["sniff_gate"]["threshold"] == 5
    assert result["sniff_gate"]["sniff_elapsed_s"] == pytest.approx(1.2)
    assert result["sniff_gate"]["deep_elapsed_s"] == pytest.approx(3.4)
    # Elapsed is sniff + deep.
    assert elapsed == pytest.approx(1.2 + 3.4)


@patch.object(sniff_gate, "count_gate_entities", return_value=5)
def test_gate_at_threshold_runs_deep(mock_ner):
    """Boundary: entity count EXACTLY at threshold → gate fires (>= not >)."""
    cfg = _cfg(dgx_whisper_sniff_gate_min_entities=5)
    provider = _provider()
    result, _ = sniff_gate.transcribe_with_sniff_gate(
        media_path="/tmp/audio.mp3",
        cfg=cfg,
        provider=provider,
    )
    assert result["sniff_gate"]["decision"] == sniff_gate.GATE_DECISION_RAN_DEEP


# ---------------------------------------------------------------------------
# transcribe_with_sniff_gate — failure modes
# ---------------------------------------------------------------------------


@patch.object(sniff_gate, "count_gate_entities", return_value=None)
def test_ner_unavailable_falls_open_to_deep(mock_ner):
    """When spaCy is missing, the gate falls OPEN (deep runs) — never silently
    degrades to sniff-only.
    """
    cfg = _cfg()
    provider = _provider()

    result, elapsed = sniff_gate.transcribe_with_sniff_gate(
        media_path="/tmp/audio.mp3",
        cfg=cfg,
        provider=provider,
    )

    # Both sniff and deep ran.
    assert provider.transcribe_with_segments.call_count == 2
    assert result["text"] == "deep transcript"
    assert result["sniff_gate"]["decision"] == sniff_gate.GATE_DECISION_NER_UNAVAILABLE
    assert result["sniff_gate"]["sniff_model"] == "Systran/faster-whisper-small.en"
    assert result["sniff_gate"]["deep_model"] == "Systran/faster-whisper-large-v3"
    assert elapsed == pytest.approx(1.2 + 3.4)


def test_disabled_path_runs_deep_only_and_tags_decision():
    """When called with sniff_model empty, the orchestrator still runs and
    returns deep-only with decision='disabled' for symmetry — so downstream
    artifact writers can always read result['sniff_gate'].
    """
    cfg = _cfg(dgx_whisper_sniff_model="")
    provider = _provider()
    result, elapsed = sniff_gate.transcribe_with_sniff_gate(
        media_path="/tmp/audio.mp3",
        cfg=cfg,
        provider=provider,
    )
    assert provider.transcribe_with_segments.call_count == 1
    # The single call had no override.
    assert provider.transcribe_with_segments.call_args.kwargs.get("model_override") is None
    assert result["text"] == "deep transcript"
    assert result["sniff_gate"]["decision"] == sniff_gate.GATE_DECISION_DISABLED
    assert result["sniff_gate"]["deep_model"] == "Systran/faster-whisper-large-v3"
    assert elapsed == pytest.approx(3.4)


# ---------------------------------------------------------------------------
# count_gate_entities — NER label filter
# ---------------------------------------------------------------------------


def test_count_gate_entities_returns_none_when_spacy_missing(monkeypatch):
    """When spaCy can't be imported the helper returns None — caller must
    handle (currently: fall open to deep). This guards the production fail
    mode where the container ships without spaCy."""
    sniff_gate._load_nlp.cache_clear()
    import builtins

    real_import = builtins.__import__

    def _fail_spacy(name, *args, **kwargs):
        if name == "spacy":
            raise ImportError("no spacy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_spacy)
    try:
        assert sniff_gate.count_gate_entities("Hello Maya, this is Liam.") is None
    finally:
        sniff_gate._load_nlp.cache_clear()


def test_count_gate_entities_filters_to_person_org(monkeypatch):
    """Only PERSON + ORG labels count toward the gate. GPE/LOC/DATE are
    excluded — they're noisy on podcast small.en transcripts (place names
    and dates inflate without correlating to content density).
    """
    sniff_gate._load_nlp.cache_clear()

    class _Ent:
        def __init__(self, label):
            self.label_ = label

    class _Doc:
        ents = [
            _Ent("PERSON"),
            _Ent("PERSON"),
            _Ent("ORG"),
            _Ent("GPE"),  # excluded
            _Ent("DATE"),  # excluded
            _Ent("LOC"),  # excluded
            _Ent("ORG"),
        ]

    class _NLP:
        def __call__(self, text):
            assert text == "x"
            return _Doc()

    fake_nlp = _NLP()
    # Patch the lazy loader so we don't depend on spaCy at all. Use a callable
    # (not lru_cache) — monkeypatch restores the original lru_cache after.
    monkeypatch.setattr(sniff_gate, "_load_nlp", lambda: fake_nlp)
    assert sniff_gate.count_gate_entities("x") == 4  # 2 PERSON + 2 ORG
