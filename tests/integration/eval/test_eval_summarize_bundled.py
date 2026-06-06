"""Integration: bundled dispatch logic for eval summarization (Issue #477).

Tests the *real* ``run_experiment._eval_summarize`` (imported as a namespace
package from the repo root), not a copy. The previous version pasted a
replica of the dispatch logic into the test, so a divergence in the real
function (a renamed kwarg, a flipped mode check) left these green — a hollow
test. The real import costs ~2.5s but actually guards the shipped code.
"""

from __future__ import annotations

import importlib
import json
import unittest
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.integration

# Import the REAL function at runtime via importlib (not a static ``from scripts...``):
# ``scripts/`` is excluded from mypy (sys.path tricks), but a static import is still
# *followed* by mypy and would surface scripts/ type-debt as errors here. A dynamic
# import keeps mypy out of scripts/ while the test exercises the shipped function.
_eval_summarize = importlib.import_module("scripts.eval.experiment.run_experiment")._eval_summarize

VALID_BUNDLED_JSON = json.dumps(
    {
        "title": "Test Title",
        "summary": "A detailed prose summary paragraph.",
        "bullets": ["Point one.", "Point two."],
    }
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
