#!/usr/bin/env python3
"""Tests for provider metrics and retry utilities.

These tests verify retry behavior, jitter, and metrics tracking.
"""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from podcast_scraper.utils.provider_metrics import (
    anthropic_message_usage_tokens,
    apply_gil_evidence_llm_call_metrics,
    gemini_generate_usage_tokens,
    merge_gil_evidence_call_metrics_on_failure,
    openai_compatible_chat_usage_tokens,
    ProviderCallMetrics,
    retry_with_metrics,
)


class TestProviderCallMetrics(unittest.TestCase):
    """Test ProviderCallMetrics dataclass."""

    def test_initial_state(self):
        """Test that metrics start with correct initial values."""
        metrics = ProviderCallMetrics()
        self.assertEqual(metrics.retries, 0)
        self.assertEqual(metrics.rate_limit_sleep_sec, 0.0)
        self.assertIsNone(metrics.prompt_tokens)
        self.assertIsNone(metrics.completion_tokens)
        self.assertIsNone(metrics.estimated_cost)

    def test_record_retry(self):
        """Test recording retry attempts."""
        metrics = ProviderCallMetrics()
        metrics.set_provider_name("test_provider")
        metrics.record_retry(sleep_seconds=2.5, reason="429")
        metrics.finalize()
        self.assertEqual(metrics.retries, 1)
        self.assertEqual(metrics.rate_limit_sleep_sec, 2.5)

    def test_record_multiple_retries(self):
        """Test recording multiple retry attempts."""
        metrics = ProviderCallMetrics()
        metrics.record_retry(sleep_seconds=1.0, reason="500")
        metrics.record_retry(sleep_seconds=2.0, reason="429")
        metrics.record_retry(sleep_seconds=4.0, reason="429")
        metrics.finalize()
        self.assertEqual(metrics.retries, 3)
        self.assertEqual(metrics.rate_limit_sleep_sec, 7.0)

    def test_set_tokens(self):
        """Test setting token counts."""
        metrics = ProviderCallMetrics()
        metrics.set_tokens(prompt_tokens=100, completion_tokens=50)
        self.assertEqual(metrics.prompt_tokens, 100)
        self.assertEqual(metrics.completion_tokens, 50)

    def test_set_cost(self):
        """Test setting estimated cost."""
        metrics = ProviderCallMetrics()
        metrics.set_cost(0.05)
        self.assertEqual(metrics.estimated_cost, 0.05)


class TestRetryWithMetrics(unittest.TestCase):
    """Test retry_with_metrics function."""

    def test_successful_call_no_retry(self):
        """Test that successful calls don't retry."""
        func = Mock(return_value="success")
        result = retry_with_metrics(func, max_retries=3)
        self.assertEqual(result, "success")
        self.assertEqual(func.call_count, 1)

    def test_retry_on_exception(self):
        """Test that retries occur on retryable exceptions."""
        func = Mock(side_effect=[Exception("error"), "success"])
        with patch("time.sleep"):
            result = retry_with_metrics(func, max_retries=3, initial_delay=0.1)
        self.assertEqual(result, "success")
        self.assertEqual(func.call_count, 2)

    def test_max_retries_exhausted(self):
        """Test that exception is raised when max retries exhausted."""
        func = Mock(side_effect=Exception("error"))
        with patch("time.sleep"):
            with self.assertRaises(Exception) as context:
                retry_with_metrics(func, max_retries=2, initial_delay=0.1)
        self.assertEqual(str(context.exception), "error")
        self.assertEqual(func.call_count, 3)  # Initial + 2 retries

    def test_exhausted_logs_provider_retries_exhausted_with_context(self):
        """Terminal failure emits structured provider_retries_exhausted (GitHub 741)."""
        func = Mock(side_effect=Exception("503 UNAVAILABLE"))
        metrics = ProviderCallMetrics()
        metrics.set_provider_name("gemini")
        with patch("time.sleep"):
            with patch("podcast_scraper.utils.provider_metrics.logger") as mock_logger:
                with self.assertRaises(Exception):
                    retry_with_metrics(
                        func,
                        max_retries=1,
                        initial_delay=0.05,
                        max_delay=1.0,
                        jitter=False,
                        metrics=metrics,
                        retry_context={"stage": "test_stage", "episode_title": "Ep"},
                    )
        err_calls = [c for c in mock_logger.error.call_args_list if c.args]
        self.assertTrue(any("provider_retries_exhausted" in str(c) for c in err_calls))
        blob = str(err_calls[-1])
        self.assertIn("gemini", blob)
        self.assertIn("test_stage", blob)

    def test_jitter_adds_variation(self):
        """Test that jitter adds random variation to delays."""
        func = Mock(side_effect=[Exception("error"), "success"])
        metrics = ProviderCallMetrics()
        with patch("time.sleep") as mock_sleep:
            with patch("random.uniform", return_value=1.05):  # 5% increase
                retry_with_metrics(
                    func, max_retries=3, initial_delay=1.0, jitter=True, metrics=metrics
                )
        # Verify sleep was called with jittered delay
        mock_sleep.assert_called_once()
        # With jitter factor 1.05, delay should be 1.0 * 1.05 = 1.05
        self.assertAlmostEqual(mock_sleep.call_args[0][0], 1.05, places=2)

    def test_jitter_disabled(self):
        """Test that jitter can be disabled."""
        func = Mock(side_effect=[Exception("error"), "success"])
        with patch("time.sleep") as mock_sleep:
            retry_with_metrics(func, max_retries=3, initial_delay=1.0, jitter=False)
        # Verify sleep was called with exact delay (no jitter)
        mock_sleep.assert_called_once()
        self.assertEqual(mock_sleep.call_args[0][0], 1.0)

    def test_jitter_respects_max_delay(self):
        """Test that jitter doesn't exceed max_delay."""
        func = Mock(side_effect=[Exception("error"), "success"])
        with patch("time.sleep") as mock_sleep:
            with patch("random.uniform", return_value=2.0):  # Would exceed max
                retry_with_metrics(
                    func,
                    max_retries=3,
                    initial_delay=20.0,
                    max_delay=30.0,
                    jitter=True,
                )
        # Verify sleep was capped at max_delay
        mock_sleep.assert_called_once()
        self.assertLessEqual(mock_sleep.call_args[0][0], 30.0)

    def test_metrics_tracking(self):
        """Test that metrics are tracked correctly."""
        func = Mock(side_effect=[Exception("error"), "success"])
        metrics = ProviderCallMetrics()
        metrics.set_provider_name("test_provider")
        with patch("time.sleep"):
            retry_with_metrics(func, max_retries=3, initial_delay=0.1, metrics=metrics)
        metrics.finalize()
        self.assertEqual(metrics.retries, 1)
        self.assertGreater(metrics.rate_limit_sleep_sec, 0)

    def test_rate_limit_detection(self):
        """Test that rate limit errors are detected correctly."""
        func = Mock(side_effect=[Exception("429 Rate limit exceeded"), "success"])
        metrics = ProviderCallMetrics()
        metrics.set_provider_name("test_provider")
        with patch("time.sleep"):
            with patch("podcast_scraper.utils.provider_metrics.logger") as mock_logger:
                retry_with_metrics(func, max_retries=3, initial_delay=0.1, metrics=metrics)
        # Verify rate limit was logged
        mock_logger.info.assert_called()
        log_call = str(mock_logger.info.call_args)
        self.assertIn("429", log_call)

    def test_exponential_backoff(self):
        """Test that delays increase exponentially."""
        func = Mock(side_effect=[Exception("error"), Exception("error"), "success"])
        delays = []

        def capture_sleep(delay):
            delays.append(delay)

        with patch("time.sleep", side_effect=capture_sleep):
            with patch("random.uniform", return_value=1.0):  # No jitter for test
                retry_with_metrics(func, max_retries=3, initial_delay=1.0, jitter=False)
        # First retry: 1.0, second retry: 2.0 (doubled)
        self.assertEqual(len(delays), 2)
        self.assertAlmostEqual(delays[0], 1.0, places=1)
        self.assertAlmostEqual(delays[1], 2.0, places=1)

    def test_max_delay_cap(self):
        """Test that delays are capped at max_delay."""
        func = Mock(side_effect=[Exception("error"), Exception("error"), "success"])
        delays = []

        def capture_sleep(delay):
            delays.append(delay)

        with patch("time.sleep", side_effect=capture_sleep):
            with patch("random.uniform", return_value=1.0):  # No jitter for test
                retry_with_metrics(
                    func,
                    max_retries=3,
                    initial_delay=20.0,
                    max_delay=30.0,
                    jitter=False,
                )
        # First retry: 20.0, second retry: 30.0 (capped, not 40.0)
        self.assertEqual(len(delays), 2)
        self.assertAlmostEqual(delays[0], 20.0, places=1)
        self.assertAlmostEqual(delays[1], 30.0, places=1)

    def test_non_retryable_error_raises_immediately(self):
        """401-style errors are not retried (is_retryable_error False)."""
        func = Mock(side_effect=Exception("401 Unauthorized"))
        with patch("time.sleep"):
            with self.assertRaises(Exception) as ctx:
                retry_with_metrics(func, max_retries=3, initial_delay=0.1)
        self.assertIn("401", str(ctx.exception))
        self.assertEqual(func.call_count, 1)

    def test_exception_not_in_retryable_tuple_reraises(self):
        """Exceptions outside retryable_exceptions bypass retry loop."""
        func = Mock(side_effect=ValueError("nope"))
        with self.assertRaises(ValueError):
            retry_with_metrics(
                func, max_retries=3, retryable_exceptions=(ConnectionError,), initial_delay=0.1
            )
        func.assert_called_once()

    def test_rate_limit_retry_after_invalid_uses_backoff_delay(self):
        """Invalid retry_after on 429 falls back to exponential delay."""

        class RateLimitExc(Exception):
            def __init__(self) -> None:
                super().__init__("429 rate limit")
                self.retry_after = "not-a-number"

        func = Mock(side_effect=[RateLimitExc(), "ok"])
        with patch("time.sleep") as mock_sleep:
            with patch("random.uniform", return_value=1.0):
                retry_with_metrics(func, max_retries=3, initial_delay=2.0, jitter=False)
        mock_sleep.assert_called_once_with(2.0)


class TestGeminiRetryableExceptions(unittest.TestCase):
    """Regression: _safe_gemini_retryable must cover the new google-genai
    SDK's ServerError (wraps 5xx including 503 UNAVAILABLE), not only the
    legacy google.api_core exceptions.

    Missing this meant mega_bundled fell back to staged on the first 503
    instead of retrying with backoff. Discovered during the 10-feed
    cloud_balanced production run (2026-04-21).

    Unit CI installs ``.[dev]`` only — no ``google.*`` packages. We inject
    minimal fake modules via ``sys.modules`` so imports inside
    :func:`_safe_gemini_retryable` resolve without skips."""

    def _fake_google_modules_genai_only(self) -> dict[str, ModuleType]:
        genai_errors = ModuleType("google.genai.errors")

        class ServerError(Exception):
            pass

        genai_errors.ServerError = ServerError
        genai_pkg = ModuleType("google.genai")
        genai_pkg.__path__ = []
        google_pkg = ModuleType("google")
        google_pkg.__path__ = []
        return {
            "google": google_pkg,
            "google.genai": genai_pkg,
            "google.genai.errors": genai_errors,
        }

    def _fake_google_modules_api_core_only(self) -> dict[str, ModuleType]:
        api_core_exc = ModuleType("google.api_core.exceptions")

        class ResourceExhausted(Exception):
            pass

        class ServiceUnavailable(Exception):
            pass

        api_core_exc.ResourceExhausted = ResourceExhausted
        api_core_exc.ServiceUnavailable = ServiceUnavailable
        api_core_pkg = ModuleType("google.api_core")
        api_core_pkg.__path__ = []
        google_pkg = ModuleType("google")
        google_pkg.__path__ = []
        return {
            "google": google_pkg,
            "google.api_core": api_core_pkg,
            "google.api_core.exceptions": api_core_exc,
        }

    def test_new_genai_sdk_server_error_is_retryable(self):
        from podcast_scraper.utils.provider_metrics import _safe_gemini_retryable

        mods = self._fake_google_modules_genai_only()
        with patch.dict(sys.modules, mods, clear=False):
            retryable = _safe_gemini_retryable()
            server_err = mods["google.genai.errors"].ServerError
        self.assertIn(server_err, retryable)

    def test_legacy_api_core_service_unavailable_is_retryable(self):
        """Legacy SDK coverage must remain."""
        from podcast_scraper.utils.provider_metrics import _safe_gemini_retryable

        mods = self._fake_google_modules_api_core_only()
        with patch.dict(sys.modules, mods, clear=False):
            retryable = _safe_gemini_retryable()
            su = mods["google.api_core.exceptions"].ServiceUnavailable
        self.assertIn(su, retryable)

    def test_resource_exhausted_still_retryable(self):
        from podcast_scraper.utils.provider_metrics import _safe_gemini_retryable

        mods = self._fake_google_modules_api_core_only()
        with patch.dict(sys.modules, mods, clear=False):
            retryable = _safe_gemini_retryable()
            rexc = mods["google.api_core.exceptions"].ResourceExhausted
        self.assertIn(rexc, retryable)


class TestOpenAiCompatibleChatUsageTokens(unittest.TestCase):
    """openai_compatible_chat_usage_tokens."""

    def test_no_usage_attribute(self):
        self.assertEqual(openai_compatible_chat_usage_tokens(object()), (None, None))

    def test_usage_empty(self):
        r = SimpleNamespace(usage=None)
        self.assertEqual(openai_compatible_chat_usage_tokens(r), (None, None))

    def test_int_tokens(self):
        u = SimpleNamespace(prompt_tokens=10, completion_tokens=20)
        r = SimpleNamespace(usage=u)
        self.assertEqual(openai_compatible_chat_usage_tokens(r), (10, 20))

    def test_float_tokens_coerced(self):
        u = SimpleNamespace(prompt_tokens=3.0, completion_tokens=4.0)
        r = SimpleNamespace(usage=u)
        self.assertEqual(openai_compatible_chat_usage_tokens(r), (3, 4))

    def test_non_numeric_tokens_return_none(self):
        u = SimpleNamespace(prompt_tokens="x", completion_tokens=5)
        r = SimpleNamespace(usage=u)
        self.assertEqual(openai_compatible_chat_usage_tokens(r), (None, 5))


class TestAnthropicMessageUsageTokens(unittest.TestCase):
    """anthropic_message_usage_tokens."""

    def test_no_usage(self):
        self.assertEqual(anthropic_message_usage_tokens(SimpleNamespace(usage=None)), (None, None))

    def test_int_tokens(self):
        u = SimpleNamespace(input_tokens=7, output_tokens=8)
        self.assertEqual(anthropic_message_usage_tokens(SimpleNamespace(usage=u)), (7, 8))


class TestGeminiGenerateUsageTokens(unittest.TestCase):
    """gemini_generate_usage_tokens."""

    def test_no_usage_metadata(self):
        self.assertEqual(
            gemini_generate_usage_tokens(SimpleNamespace(usage_metadata=None)), (None, None)
        )

    def test_valid_counts(self):
        m = SimpleNamespace(prompt_token_count=1, candidates_token_count=2)
        self.assertEqual(gemini_generate_usage_tokens(SimpleNamespace(usage_metadata=m)), (1, 2))

    def test_bad_prompt_count_still_parses_completion(self):
        class Bad:
            def __int__(self) -> int:
                raise TypeError("no int")

        m = SimpleNamespace(prompt_token_count=Bad(), candidates_token_count=3)
        self.assertEqual(gemini_generate_usage_tokens(SimpleNamespace(usage_metadata=m)), (None, 3))


class TestGilEvidenceCallMetrics(unittest.TestCase):
    """apply_gil_evidence_llm_call_metrics / merge on failure."""

    def test_apply_sets_tokens_and_records_pipeline(self):
        m = ProviderCallMetrics()
        m.set_provider_name("gil")
        pipe = Mock()
        apply_gil_evidence_llm_call_metrics(m, pipe, 10, 20)
        self.assertEqual(m.prompt_tokens, 10)
        self.assertEqual(m.completion_tokens, 20)
        pipe.record_llm_gi_evidence_call_metrics.assert_called_once()
        # cost_usd threads through from call_metrics.estimated_cost
        # (#650 Finding 17) — None here because test ProviderCallMetrics never
        # called set_cost.
        pipe.record_llm_gi_call.assert_called_once_with(10, 20, cost_usd=None)

    def test_apply_pipeline_none_no_crash(self):
        m = ProviderCallMetrics()
        apply_gil_evidence_llm_call_metrics(m, None, 1, 2)
        m.finalize()
        self.assertEqual(m.retries, 0)

    def test_apply_partial_tokens_skips_record_llm_gi_call(self):
        m = ProviderCallMetrics()

        class Pipe:
            def __init__(self) -> None:
                self.calls = 0

            def record_llm_gi_evidence_call_metrics(self, _cm: ProviderCallMetrics) -> None:
                self.calls += 1

        pipe = Pipe()
        apply_gil_evidence_llm_call_metrics(m, pipe, None, 20)
        self.assertEqual(pipe.calls, 1)

    def test_merge_failure_records_evidence_only(self):
        m = ProviderCallMetrics()
        pipe = Mock()
        merge_gil_evidence_call_metrics_on_failure(m, pipe)
        pipe.record_llm_gi_evidence_call_metrics.assert_called_once()

    def test_apply_computes_cost_when_cfg_provider_model_passed(self):
        """#650 Finding 17 — helper fills cost_usd when call_metrics lacks it.

        Before this path existed, GIL evidence LLM calls via
        extract_quotes / score_entailment on non-OpenAI providers (gemini,
        deepseek, grok, anthropic, ollama) reported $0 because the shared
        helper never invoked calculate_provider_cost. This test locks in
        the cost-computation branch.
        """
        from podcast_scraper import config as cfg_mod

        cfg = cfg_mod.Config(
            transcription_provider="whisper",
            speaker_detector_provider="anthropic",
            summary_provider="anthropic",
            anthropic_api_key="test-api-key-123",
            transcribe_missing=False,
            auto_speakers=True,
            generate_summaries=True,
        )
        m = ProviderCallMetrics()
        pipe = Mock()
        apply_gil_evidence_llm_call_metrics(
            m,
            pipe,
            prompt_tokens=5000,
            completion_tokens=200,
            cfg=cfg,
            provider_type="anthropic",
            model="claude-haiku-4-5",
        )
        # Cost computed + pushed into both call_metrics and pipeline_metrics.
        self.assertIsNotNone(m.estimated_cost)
        self.assertGreater(m.estimated_cost, 0.0)
        pipe.record_llm_gi_call.assert_called_once()
        call = pipe.record_llm_gi_call.call_args
        self.assertEqual(call.args, (5000, 200))
        self.assertIsNotNone(call.kwargs.get("cost_usd"))
        self.assertEqual(call.kwargs["cost_usd"], m.estimated_cost)

    def test_apply_preserves_existing_call_metrics_cost(self):
        """If call_metrics.estimated_cost is already set, helper must not
        overwrite it — provider's own cost calculation wins.
        """
        m = ProviderCallMetrics()
        m.set_cost(0.0042)
        pipe = Mock()
        apply_gil_evidence_llm_call_metrics(
            m,
            pipe,
            prompt_tokens=10,
            completion_tokens=20,
            cfg=Mock(),
            provider_type="anthropic",
            model="claude-haiku-4-5",
        )
        self.assertEqual(m.estimated_cost, 0.0042)
        pipe.record_llm_gi_call.assert_called_once_with(10, 20, cost_usd=0.0042)

    def test_merge_failure_pipeline_none(self):
        m = ProviderCallMetrics()
        merge_gil_evidence_call_metrics_on_failure(m, None)
        m.finalize()
        self.assertEqual(m.retries, 0)


if __name__ == "__main__":
    unittest.main()
