"""Unit tests for cloud-LLM output-token floor helper (#645)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

from podcast_scraper.providers.common.output_tokens import (
    cloud_structured_max_output_tokens,
    warn_if_output_truncated,
)


class TestCloudStructuredMaxOutputTokens:
    def test_clamps_up_to_floor(self):
        cfg = MagicMock()
        cfg.cloud_llm_structured_min_output_tokens = 4096
        assert cloud_structured_max_output_tokens(cfg, 650) == 4096

    def test_respects_larger_than_floor(self):
        cfg = MagicMock()
        cfg.cloud_llm_structured_min_output_tokens = 4096
        assert cloud_structured_max_output_tokens(cfg, 8192) == 8192

    def test_none_requested_returns_floor(self):
        cfg = MagicMock()
        cfg.cloud_llm_structured_min_output_tokens = 4096
        assert cloud_structured_max_output_tokens(cfg, None) == 4096

    def test_zero_requested_returns_floor(self):
        cfg = MagicMock()
        cfg.cloud_llm_structured_min_output_tokens = 4096
        assert cloud_structured_max_output_tokens(cfg, 0) == 4096

    def test_equal_to_floor_returns_as_is(self):
        cfg = MagicMock()
        cfg.cloud_llm_structured_min_output_tokens = 4096
        assert cloud_structured_max_output_tokens(cfg, 4096) == 4096

    def test_falls_back_to_default_when_attr_missing(self):
        cfg = object()  # no cloud_llm_structured_min_output_tokens attribute
        # Default floor is 4096 per the helper's module-level constant.
        assert cloud_structured_max_output_tokens(cfg, 650) == 4096

    def test_falls_back_to_default_when_attr_none(self):
        cfg = MagicMock()
        cfg.cloud_llm_structured_min_output_tokens = None
        assert cloud_structured_max_output_tokens(cfg, 650) == 4096

    def test_custom_floor_from_cfg_is_respected(self):
        cfg = MagicMock()
        cfg.cloud_llm_structured_min_output_tokens = 8192
        assert cloud_structured_max_output_tokens(cfg, 650) == 8192
        assert cloud_structured_max_output_tokens(cfg, 16384) == 16384


class TestWarnIfOutputTruncated:
    def test_warns_on_max_tokens_finish_reason(self, caplog):
        caplog.set_level(logging.WARNING, logger="podcast_scraper.providers.common.output_tokens")
        warn_if_output_truncated(
            provider_name="gemini",
            finish_reason="MAX_TOKENS",
            output_tokens=4090,
            max_output_tokens=4096,
        )
        assert any("possible output truncation" in rec.getMessage() for rec in caplog.records)

    def test_warns_on_near_budget_even_without_finish_reason(self, caplog):
        caplog.set_level(logging.WARNING, logger="podcast_scraper.providers.common.output_tokens")
        # 4000/4096 = 97.6% — above the 95% near-budget threshold.
        warn_if_output_truncated(
            provider_name="openai",
            finish_reason="STOP",
            output_tokens=4000,
            max_output_tokens=4096,
        )
        assert any("possible output truncation" in rec.getMessage() for rec in caplog.records)

    def test_no_warn_on_normal_finish(self, caplog):
        caplog.set_level(logging.WARNING, logger="podcast_scraper.providers.common.output_tokens")
        warn_if_output_truncated(
            provider_name="anthropic",
            finish_reason="STOP",
            output_tokens=1000,
            max_output_tokens=4096,
        )
        assert not any("possible output truncation" in rec.getMessage() for rec in caplog.records)

    def test_no_warn_when_no_signals(self, caplog):
        caplog.set_level(logging.WARNING, logger="podcast_scraper.providers.common.output_tokens")
        warn_if_output_truncated(
            provider_name="openai",
            finish_reason=None,
            output_tokens=None,
            max_output_tokens=4096,
        )
        assert not any("possible output truncation" in rec.getMessage() for rec in caplog.records)

    def test_safety_finish_reason_warns(self, caplog):
        caplog.set_level(logging.WARNING, logger="podcast_scraper.providers.common.output_tokens")
        warn_if_output_truncated(
            provider_name="gemini",
            finish_reason="SAFETY",
            output_tokens=100,
            max_output_tokens=4096,
        )
        assert any("possible output truncation" in rec.getMessage() for rec in caplog.records)

    def test_episode_id_included_in_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger="podcast_scraper.providers.common.output_tokens")
        warn_if_output_truncated(
            provider_name="gemini",
            finish_reason="MAX_TOKENS",
            output_tokens=4090,
            max_output_tokens=4096,
            episode_id="p04_e03",
        )
        assert any("ep=p04_e03" in rec.getMessage() for rec in caplog.records)
