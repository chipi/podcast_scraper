"""Every event must say WHICH profile ran it, and what that profile routed to.

2026-08-28: no event carried the profile at all. Answering "which profile produced this
cost?" meant reading the run's argv out of the jobs registry and assuming the corpus config
had not changed since — and with three override layers (corpus YAML, the feed's own pin, a
per-request override) that inference is not even sound. The routing is therefore stamped,
not reconstructed.
"""

from __future__ import annotations

import json

import pytest

from podcast_scraper.obs.events import (
    clear_run_context,
    emit_event,
    get_run_context,
    run_context_from_config,
    set_run_context,
)

pytestmark = [pytest.mark.unit]


@pytest.fixture(autouse=True)
def _clean_context():
    clear_run_context()
    yield
    clear_run_context()


@pytest.fixture()
def dgx_cfg(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dummy-for-validation")
    from podcast_scraper import config as config_mod

    return config_mod.Config(rss_url="https://x.example/f", profile="cloud_with_dgx_primary")


class TestRoutingIsDerivedFromResolvedConfig:
    def test_records_profile_and_every_stage_route(self, dgx_cfg):
        rc = run_context_from_config(dgx_cfg)
        assert rc["profile"] == "cloud_with_dgx_primary"
        assert rc["asr_provider"] == "tailnet_dgx_whisper"
        assert rc["asr_fallback_provider"] == "deepgram"
        assert rc["diarization_provider"] == "tailnet_dgx"
        assert rc["summary_provider"] == "litellm"

    def test_routing_not_inferred_from_the_name(self, dgx_cfg):
        """An explicit field moves routing under a profile — the RESOLVED value must win."""
        moved = dgx_cfg.model_copy(update={"transcription_provider": "deepgram"})
        rc = run_context_from_config(moved)
        assert rc["profile"] == "cloud_with_dgx_primary"
        assert (
            rc["asr_provider"] == "deepgram"
        ), "recorded the profile's nominal ASR instead of what the run will actually use"

    def test_none_config_is_harmless(self):
        assert run_context_from_config(None) == {}


class TestEveryEventCarriesIt:
    def test_event_is_stamped(self, dgx_cfg):
        set_run_context(**run_context_from_config(dgx_cfg))
        record = json.loads(emit_event("llm_cost", provider="litellm", estimated_cost_usd=0.01))
        assert record["profile"] == "cloud_with_dgx_primary"
        assert record["asr_provider"] == "tailnet_dgx_whisper"
        assert record["diarization_provider"] == "tailnet_dgx"
        assert record["estimated_cost_usd"] == 0.01

    def test_an_events_own_field_wins_over_the_run_default(self, dgx_cfg):
        """A stage that fell back must be able to say so, not inherit the run's nominal route."""
        set_run_context(**run_context_from_config(dgx_cfg))
        record = json.loads(emit_event("llm_cost", asr_provider="deepgram"))
        assert record["asr_provider"] == "deepgram"
        assert record["profile"] == "cloud_with_dgx_primary"

    def test_no_context_means_no_extra_fields(self):
        record = json.loads(emit_event("llm_cost", provider="litellm"))
        assert "profile" not in record

    def test_clear_is_effective(self, dgx_cfg):
        set_run_context(**run_context_from_config(dgx_cfg))
        assert get_run_context()
        clear_run_context()
        assert json.loads(emit_event("x")).get("profile") is None
