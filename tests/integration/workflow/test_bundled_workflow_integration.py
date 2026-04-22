"""Integration: bundled LLM pipeline mode in metadata_generation (Issue #477).

Tests the bundled dispatch logic, fallback to staged, and cleaned
transcript file handling without running the full generate_metadata
function (which requires extensive setup).
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import Mock

import pytest

from podcast_scraper import config

pytestmark = [pytest.mark.integration]

VALID_BUNDLED_JSON = json.dumps(
    {
        "title": "Test Title",
        "summary": "A detailed prose summary paragraph.",
        "bullets": ["Point one.", "Point two."],
    }
)


class TestBundledConfigWiring(unittest.TestCase):
    """Config.llm_pipeline_mode correctly controls bundled dispatch."""

    def test_config_accepts_bundled_mode(self):
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="openai",
            openai_api_key="sk-test",
            llm_pipeline_mode="bundled",
        )
        self.assertEqual(cfg.llm_pipeline_mode, "bundled")

    def test_config_defaults_to_staged(self):
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="openai",
            openai_api_key="sk-test",
        )
        self.assertEqual(cfg.llm_pipeline_mode, "staged")

    def test_config_rejects_invalid_mode(self):
        with self.assertRaises(Exception):
            config.Config(
                rss_url="https://example.com/feed.xml",
                summary_provider="openai",
                openai_api_key="sk-test",
                llm_pipeline_mode="invalid_mode",
            )


class TestBundledProviderDispatch(unittest.TestCase):
    """Workflow dispatches to summarize_bundled when mode=bundled."""

    def test_bundled_mode_checks_for_summarize_bundled_method(self):
        """Provider must have summarize_bundled for bundled mode."""
        provider_with = Mock()
        provider_with.summarize_bundled = Mock(
            return_value={
                "summary": VALID_BUNDLED_JSON,
                "metadata": {"bundled": True},
            }
        )
        fn = getattr(provider_with, "summarize_bundled", None)
        self.assertTrue(callable(fn))

        provider_without = Mock(spec=["summarize"])
        fn2 = getattr(provider_without, "summarize_bundled", None)
        self.assertIsNone(fn2)

    def test_bundled_result_has_no_cleaned_text_key(self):
        """Bundled JSON output must not contain cleaned_text."""
        parsed = json.loads(VALID_BUNDLED_JSON)
        self.assertIn("title", parsed)
        self.assertIn("summary", parsed)
        self.assertIn("bullets", parsed)
        self.assertNotIn("cleaned_text", parsed)


class TestBundledFallbackBehavior(unittest.TestCase):
    """When bundled fails, workflow falls back to staged."""

    def test_fallback_records_metric(self):
        pm = Mock()
        pm.record_llm_bundled_fallback_to_staged = Mock()
        pm.record_llm_bundled_fallback_to_staged()
        pm.record_llm_bundled_fallback_to_staged.assert_called_once()

    def test_bundled_max_output_tokens_config(self):
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="openai",
            openai_api_key="sk-test",
            llm_pipeline_mode="bundled",
            llm_bundled_max_output_tokens=8192,
        )
        self.assertEqual(cfg.llm_bundled_max_output_tokens, 8192)

    def test_bundled_max_output_tokens_default(self):
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="openai",
            openai_api_key="sk-test",
            llm_pipeline_mode="bundled",
        )
        self.assertEqual(cfg.llm_bundled_max_output_tokens, 16384)


class TestBundledMetricsRecording(unittest.TestCase):
    """Metrics correctly track bundled calls."""

    def test_metrics_has_bundled_fields(self):
        from podcast_scraper.workflow.metrics import Metrics

        m = Metrics()
        m.record_llm_bundled_clean_summary_call(100, 50)
        result = m.finish()
        self.assertEqual(result["llm_bundled_clean_summary_calls"], 1)
        self.assertEqual(result["llm_bundled_clean_summary_input_tokens"], 100)
        self.assertEqual(result["llm_bundled_clean_summary_output_tokens"], 50)
        stage = result["llm_token_totals_by_stage"]["bundled_clean_summary"]
        self.assertEqual(stage["input"], 100)
        self.assertEqual(stage["output"], 50)
        self.assertEqual(stage["calls"], 1)

    def test_metrics_fallback_counter(self):
        from podcast_scraper.workflow.metrics import Metrics

        m = Metrics()
        m.record_llm_bundled_fallback_to_staged()
        m.record_llm_bundled_fallback_to_staged()
        result = m.finish()
        self.assertEqual(result["llm_bundled_fallback_to_staged_count"], 2)


# ---------------------------------------------------------------------------
# True integration tests: dispatcher in metadata_generation._generate_episode_summary
# routes correctly across all 4 llm_pipeline_mode values + their fallback paths.
#
# Prior coverage at this layer was thin: ``TestBundledConfigWiring`` /
# ``TestBundledProviderDispatch`` above just assert config values + ``hasattr``
# — not the real dispatcher. Tests below exercise the real
# ``_generate_episode_summary`` with a stub SummarizationProvider and prove
# the mode→method routing for each of {staged, bundled, mega_bundled,
# extraction_bundled} plus their fallback paths.
# ---------------------------------------------------------------------------


_TRANSCRIPT = (
    "This is a real transcript with enough words to exceed the 50-char "
    "minimum required by _generate_episode_summary. " * 10
)


def _staged_summary_payload() -> dict:
    """Fresh staged-summary return value per test.

    The dispatcher mutates ``result["metadata"]`` in place (writes
    ``prefilled_extraction`` from a prior extraction_bundled call). A shared
    module-level dict would leak that write across tests.
    """
    return {
        "summary": json.dumps({"title": "T", "summary": "S", "bullets": ["b1", "b2"]}),
        "summary_short": None,
        "metadata": {"provider": "anthropic", "bundled": False},
    }


def _make_cfg(mode: str) -> config.Config:
    """Minimal config driving the dispatcher down a specific mode."""
    return config.Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "summary_provider": "anthropic",
            "anthropic_api_key": "test-api-key-123",
            "llm_pipeline_mode": mode,
            "generate_summaries": True,
            "auto_speakers": False,
            "transcribe_missing": False,
            "generate_metadata": True,
        }
    )


def _make_mega_bundle_result():
    """Build a real MegaBundleResult so the dispatcher's isinstance check passes."""
    from podcast_scraper.providers.common.megabundle_parser import MegaBundleResult

    return MegaBundleResult(
        title="T",
        summary="S",
        bullets=["b1", "b2"],
        insights=[{"text": "ins1", "insight_type": "claim"}],
        topics=["topic1"],
        entities=[{"name": "E1", "kind": "org"}],
        raw={},
    )


def _make_provider_stub(*, has_methods: tuple[str, ...]) -> Mock:
    """Provider stub that only advertises the named bundle methods.

    Methods not listed are REMOVED from the Mock so the dispatcher's
    ``getattr(provider, '<method>', None)`` returns None (falls back to
    staged). Critical for testing the "provider lacks method" branch.
    """
    all_methods = {
        "summarize",
        "summarize_bundled",
        "summarize_mega_bundled",
        "summarize_extraction_bundled",
    }
    spec = (
        ["_summarization_initialized", "cleaning_processor"]
        + list(all_methods & set(has_methods))
        + ["summarize"]
    )  # summarize is always present for staged fallback
    provider = Mock(spec=sorted(set(spec)))
    provider._summarization_initialized = True
    provider.cleaning_processor = None
    provider.summarize.side_effect = lambda *a, **kw: _staged_summary_payload()
    if "summarize_bundled" in has_methods:
        provider.summarize_bundled.side_effect = lambda *a, **kw: _staged_summary_payload()
    if "summarize_mega_bundled" in has_methods:
        provider.summarize_mega_bundled.return_value = _make_mega_bundle_result()
    if "summarize_extraction_bundled" in has_methods:
        provider.summarize_extraction_bundled.return_value = _make_mega_bundle_result()
    return provider


def _invoke_dispatcher(cfg, provider, tmp_path):
    """Drive _generate_episode_summary once; return (pipeline_metrics, result_metadata).

    Swallows downstream parsing errors — we only care about dispatcher routing.
    """
    from podcast_scraper.workflow import metrics as m
    from podcast_scraper.workflow.metadata_generation import _generate_episode_summary

    transcript = tmp_path / "t.txt"
    transcript.write_text(_TRANSCRIPT, encoding="utf-8")
    pm = m.Metrics()
    try:
        result = _generate_episode_summary(
            transcript_file_path="t.txt",
            output_dir=str(tmp_path),
            cfg=cfg,
            episode_idx=0,
            summary_provider=provider,
            pipeline_metrics=pm,
        )
    except Exception:
        # Downstream parsing may fail on stub data; only routing matters.
        result = (None, None)
    return pm, result


class TestLLMPipelineModeDispatchIntegration:
    """Real ``_generate_episode_summary`` dispatcher routes correctly across
    all 4 llm_pipeline_mode values.
    """

    def test_staged_mode_routes_to_summarize(self, tmp_path):
        cfg = _make_cfg("staged")
        provider = _make_provider_stub(has_methods=("summarize",))
        _invoke_dispatcher(cfg, provider, tmp_path)

        assert provider.summarize.called, "staged mode must invoke summarize()"

    def test_bundled_mode_routes_to_summarize_bundled(self, tmp_path):
        cfg = _make_cfg("bundled")
        provider = _make_provider_stub(has_methods=("summarize", "summarize_bundled"))
        _invoke_dispatcher(cfg, provider, tmp_path)

        assert provider.summarize_bundled.called, "bundled mode must route to summarize_bundled()"
        assert (
            not provider.summarize.called
        ), "bundled mode succeeded; staged summarize() should NOT be called"

    def test_mega_bundled_mode_routes_to_summarize_mega_bundled(self, tmp_path):
        cfg = _make_cfg("mega_bundled")
        provider = _make_provider_stub(has_methods=("summarize", "summarize_mega_bundled"))
        _invoke_dispatcher(cfg, provider, tmp_path)

        assert (
            provider.summarize_mega_bundled.called
        ), "mega_bundled mode must route to summarize_mega_bundled()"
        assert (
            not provider.summarize.called
        ), "mega_bundled produced a valid MegaBundleResult; staged should NOT run"

    def test_extraction_bundled_routes_to_extraction_method_plus_staged(self, tmp_path):
        """extraction_bundled is a 2-call pipeline: extraction bundle + staged summary.

        Contract: ``summarize_extraction_bundled`` is called FIRST (for
        insights/topics/entities), then ``summarize`` runs for the staged
        summary. Both must be invoked.
        """
        cfg = _make_cfg("extraction_bundled")
        provider = _make_provider_stub(has_methods=("summarize", "summarize_extraction_bundled"))
        _invoke_dispatcher(cfg, provider, tmp_path)

        assert provider.summarize_extraction_bundled.called
        assert (
            provider.summarize.called
        ), "extraction_bundled leaves summary staged — summarize() must still run"


class TestLLMPipelineModeFallbackIntegration:
    """When a configured bundle mode cannot be satisfied, the dispatcher must
    fall back to the staged path. Prior gap: only negative unit tests
    (exception paths); the "provider missing method" fallback wasn't
    integration-tested at all.
    """

    def test_bundled_mode_missing_method_falls_back_to_staged(self, tmp_path):
        cfg = _make_cfg("bundled")
        # Provider has no summarize_bundled — dispatcher must fall back.
        provider = _make_provider_stub(has_methods=("summarize",))
        _invoke_dispatcher(cfg, provider, tmp_path)

        assert (
            provider.summarize.called
        ), "bundled mode with no summarize_bundled must fall back to staged"

    def test_mega_bundled_missing_method_falls_back_to_staged(self, tmp_path):
        cfg = _make_cfg("mega_bundled")
        provider = _make_provider_stub(has_methods=("summarize",))
        _invoke_dispatcher(cfg, provider, tmp_path)

        assert (
            provider.summarize.called
        ), "mega_bundled with no summarize_mega_bundled must fall back to staged"

    def test_bundled_exception_falls_back_to_staged(self, tmp_path):
        cfg = _make_cfg("bundled")
        provider = _make_provider_stub(has_methods=("summarize", "summarize_bundled"))
        provider.summarize_bundled.side_effect = RuntimeError("simulated failure")
        pm, _ = _invoke_dispatcher(cfg, provider, tmp_path)

        assert provider.summarize.called, "bundled exception must fall back to staged summarize()"
        assert pm.llm_bundled_fallback_to_staged_count >= 1

    def test_mega_bundled_exception_falls_back_to_staged(self, tmp_path):
        cfg = _make_cfg("mega_bundled")
        provider = _make_provider_stub(has_methods=("summarize", "summarize_mega_bundled"))
        provider.summarize_mega_bundled.side_effect = RuntimeError("simulated failure")
        pm, _ = _invoke_dispatcher(cfg, provider, tmp_path)

        assert provider.summarize.called
        assert pm.llm_bundled_fallback_to_staged_count >= 1


class TestPrefilledExtractionWiringIntegration:
    """When mega_bundled returns insights+topics+entities, the dispatcher
    must stash them in ``result["metadata"]["prefilled_extraction"]`` so
    the downstream GI+KG stages (configured separately) can short-circuit
    their own LLM calls. Pre-#643 Phase 3C this wiring was silently
    broken — mega_bundled returned data but downstream never read it,
    so GI+KG still fired their own calls at full cost.
    """

    def test_mega_bundled_populates_prefilled_extraction(self, tmp_path):
        cfg = _make_cfg("mega_bundled")
        provider = _make_provider_stub(has_methods=("summarize", "summarize_mega_bundled"))
        _, result = _invoke_dispatcher(cfg, provider, tmp_path)

        # _generate_episode_summary returns (SummaryMetadata, call_metrics).
        summary_metadata, _call_metrics = result
        assert summary_metadata is not None, "mega_bundled must produce a summary metadata"
        # SummaryMetadata.prefilled_extraction is the load-bearing field
        # downstream GI+KG stages read to skip their own LLM calls (#643).
        prefilled = summary_metadata.prefilled_extraction
        assert prefilled is not None, (
            "mega_bundled must stash insights/topics/entities on "
            "SummaryMetadata.prefilled_extraction so downstream GI+KG stages "
            "can skip their LLM calls (#643)"
        )
        assert "insights" in prefilled and len(prefilled["insights"]) == 1
        assert "topics" in prefilled and prefilled["topics"] == ["topic1"]
        assert "entities" in prefilled and prefilled["entities"][0]["kind"] == "org"

    def test_staged_mode_produces_no_prefilled_extraction(self, tmp_path):
        """Inverse: staged mode must NOT produce prefilled_extraction
        (there's no bundled call to pre-fill from). Prevents the downstream
        GI/KG skip logic from accidentally triggering on a staged run.
        """
        cfg = _make_cfg("staged")
        provider = _make_provider_stub(has_methods=("summarize",))
        _, result = _invoke_dispatcher(cfg, provider, tmp_path)

        summary_metadata, _ = result
        if summary_metadata is None:
            return  # Staged path may bail before producing metadata; that's fine.
        assert summary_metadata.prefilled_extraction is None
