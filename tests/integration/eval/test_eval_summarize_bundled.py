"""Integration: bundled dispatch logic for eval summarization (Issue #477).

Tests the dispatch pattern used by run_experiment._eval_summarize
without importing the full run_experiment module (which pulls in
heavy ML dependencies).
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Dict
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.integration

VALID_BUNDLED_JSON = json.dumps(
    {
        "title": "Test Title",
        "summary": "A detailed prose summary paragraph.",
        "bullets": ["Point one.", "Point two."],
    }
)


def _eval_summarize(
    provider: Any,
    cfg_obj: Any,
    text: str,
    summary_params: Dict[str, Any],
    pipeline_metrics: Any,
) -> Any:
    """Replicate the dispatch logic from run_experiment._eval_summarize."""
    mode = getattr(cfg_obj, "llm_pipeline_mode", "staged")
    if mode == "bundled":
        bundled = getattr(provider, "summarize_bundled", None)
        if not callable(bundled):
            raise ValueError(
                "Experiment llm_pipeline_mode=bundled requires a "
                "provider with summarize_bundled "
                f"(got {type(provider).__name__})."
            )
        return bundled(
            text,
            episode_title=None,
            episode_description=None,
            params=summary_params,
            pipeline_metrics=pipeline_metrics,
            call_metrics=None,
        )
    return provider.summarize(
        text,
        episode_title=None,
        episode_description=None,
        params=summary_params,
        pipeline_metrics=pipeline_metrics,
        call_metrics=None,
    )


class TestEvalSummarizeBundledDispatch(unittest.TestCase):
    """Dispatch routes to summarize_bundled when mode=bundled."""

    def _make_cfg(self, mode: str = "staged") -> Mock:
        cfg = Mock()
        cfg.llm_pipeline_mode = mode
        return cfg

    def _make_provider(
        self,
        staged_return: str = "staged result",
        has_bundled: bool = True,
    ) -> Mock:
        provider = Mock()
        provider.summarize.return_value = {
            "summary": staged_return,
            "metadata": {"provider": "mock"},
        }
        if has_bundled:
            provider.summarize_bundled.return_value = {
                "summary": VALID_BUNDLED_JSON,
                "metadata": {"provider": "mock", "bundled": True},
            }
        else:
            del provider.summarize_bundled
        return provider

    def test_staged_mode_calls_summarize(self):
        cfg = self._make_cfg("staged")
        provider = self._make_provider()
        pm = Mock()

        result = _eval_summarize(provider, cfg, "text", {}, pm)

        provider.summarize.assert_called_once()
        provider.summarize_bundled.assert_not_called()
        self.assertEqual(result["summary"], "staged result")

    def test_bundled_mode_calls_summarize_bundled(self):
        cfg = self._make_cfg("bundled")
        provider = self._make_provider()
        pm = Mock()

        result = _eval_summarize(provider, cfg, "text", {}, pm)

        provider.summarize_bundled.assert_called_once()
        provider.summarize.assert_not_called()
        parsed = json.loads(result["summary"])
        self.assertIn("title", parsed)
        self.assertIn("summary", parsed)
        self.assertIn("bullets", parsed)
        self.assertNotIn("cleaned_text", parsed)

    def test_bundled_mode_without_method_raises(self):
        cfg = self._make_cfg("bundled")
        provider = self._make_provider(has_bundled=False)
        pm = Mock()

        with self.assertRaises(ValueError) as ctx:
            _eval_summarize(provider, cfg, "text", {}, pm)
        self.assertIn("summarize_bundled", str(ctx.exception))

    def test_default_mode_is_staged(self):
        cfg = Mock(spec=[])
        provider = self._make_provider()
        pm = Mock()

        result = _eval_summarize(provider, cfg, "text", {}, pm)

        provider.summarize.assert_called_once()
        self.assertEqual(result["summary"], "staged result")

    def test_bundled_passes_params_and_metrics(self):
        cfg = self._make_cfg("bundled")
        provider = self._make_provider()
        pm = Mock()
        params = {"max_length": 800, "temperature": 0.0}

        _eval_summarize(provider, cfg, "transcript", params, pm)

        call_kwargs = provider.summarize_bundled.call_args
        self.assertEqual(call_kwargs.kwargs["params"], params)
        self.assertEqual(call_kwargs.kwargs["pipeline_metrics"], pm)
        self.assertEqual(call_kwargs.args[0], "transcript")
