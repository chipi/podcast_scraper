"""ADR-124 model governance: only registry-sanctioned models may run (opt-in)."""

from __future__ import annotations

import pytest

from podcast_scraper.config import Config
from podcast_scraper.providers.ml.model_governance import (
    active_models,
    assert_models_sanctioned,
    MODEL_NOT_SANCTIONED,
    sanctioned_models,
    UnsanctionedModelError,
)

TURBO = "deepdml/faster-whisper-large-v3-turbo-ct2"
LV3 = "Systran/faster-whisper-large-v3"

_BASE = {
    "rss_url": "https://example.com/feed.xml",
    "transcription_provider": "tailnet_dgx_whisper",
    "transcription_fallback_provider": "whisper",
    "dgx_tailnet_host": "dgx-llm-1.tail-test.ts.net",
}


@pytest.mark.unit
def test_sanctioned_models_include_turbo_and_lv3() -> None:
    tx = sanctioned_models("transcription")
    assert TURBO in tx  # added as a StageOption when turbo was adopted
    assert LV3 in tx
    assert "pyannote/speaker-diarization-community-1" in sanctioned_models("diarization")
    assert sanctioned_models("nonsense-stage") == frozenset()


@pytest.mark.unit
def test_active_models_resolves_only_the_configured_provider() -> None:
    cfg = Config.model_validate(
        {
            **_BASE,
            "dgx_whisper_model": TURBO,
            "transcription_coverage_min": 0.85,
            "transcription_coverage_failover_model": LV3,
        }
    )
    got = {(stage, model) for stage, _field, model in active_models(cfg)}
    assert ("transcription", TURBO) in got  # the DGX provider's model
    assert ("transcription", LV3) in got  # the coverage failover model
    # a default for an UNUSED provider (openai_transcription_model) is NOT gated
    assert not any(field == "openai_transcription_model" for _s, field, _m in active_models(cfg))


@pytest.mark.unit
def test_governance_passes_when_all_models_sanctioned() -> None:
    Config.model_validate(
        {
            **_BASE,
            "dgx_whisper_model": TURBO,
            "transcription_coverage_min": 0.85,
            "transcription_coverage_failover_model": LV3,
            "enforce_model_governance": True,
        }
    )  # must NOT raise


@pytest.mark.unit
def test_governance_rejects_unsanctioned_model_with_code() -> None:
    with pytest.raises(UnsanctionedModelError) as ei:
        Config.model_validate(
            {**_BASE, "dgx_whisper_model": "acme/made-up-model", "enforce_model_governance": True}
        )
    err = ei.value
    assert err.code == MODEL_NOT_SANCTIONED
    assert err.model == "acme/made-up-model"
    assert err.stage == "transcription"
    assert err.field == "dgx_whisper_model"
    assert LV3 in err.sanctioned


@pytest.mark.unit
def test_unsanctioned_error_is_not_wrapped_by_pydantic() -> None:
    """It must propagate as UnsanctionedModelError (a RuntimeError), NOT a pydantic ValidationError,
    so callers can catch the specific type + code."""
    from pydantic import ValidationError

    with pytest.raises(UnsanctionedModelError):
        Config.model_validate(
            {**_BASE, "dgx_whisper_model": "acme/made-up-model", "enforce_model_governance": True}
        )
    # and it is NOT a ValidationError (would be, if the error were a ValueError)
    try:
        Config.model_validate(
            {**_BASE, "dgx_whisper_model": "acme/made-up-model", "enforce_model_governance": True}
        )
    except ValidationError:  # pragma: no cover - must not happen
        pytest.fail("UnsanctionedModelError was wrapped into ValidationError")
    except UnsanctionedModelError:
        pass


@pytest.mark.unit
def test_governance_off_is_a_noop_opt_in() -> None:
    # bogus model, flag OFF (default) -> builds fine; experiment/test models never gated
    Config.model_validate({**_BASE, "dgx_whisper_model": "acme/made-up-model"})
    # explicit assert helper is also a no-op when off
    cfg = Config.model_validate({**_BASE, "dgx_whisper_model": "acme/made-up-model"})
    assert_models_sanctioned(cfg)  # no raise
