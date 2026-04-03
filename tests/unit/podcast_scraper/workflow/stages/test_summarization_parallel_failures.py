"""High-value tests for parallel summarization failure handling (Issue: hardening).

``_execute_parallel_summarization`` stops *consuming* futures after N consecutive
errors to avoid unbounded error logging on systemic failure. ThreadPoolExecutor
still waits for submitted work on exit; this test locks in the guardrail log and
error formatting path, not full job cancellation.
"""

from __future__ import annotations

import os
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

parent_tests_dir = Path(__file__).parent.parent.parent.parent
if str(parent_tests_dir) not in sys.path:
    sys.path.insert(0, str(parent_tests_dir))

import pytest

from podcast_scraper import models
from podcast_scraper.workflow.stages import summarization as summ_stage
from podcast_scraper.workflow.types import FeedMetadata, HostDetectionResult
from tests.conftest import create_test_config, create_test_episode  # noqa: E402


@pytest.mark.unit
class TestExecuteParallelSummarizationConsecutiveFailures(unittest.TestCase):
    """Regression: consecutive failure cap logs stop condition (workflow hardening)."""

    @patch.object(summ_stage, "summarize_single_episode", side_effect=RuntimeError("synthetic"))
    @patch.object(summ_stage, "logger")
    def test_logs_stop_after_max_consecutive_failures(self, mock_logger, _mock_summarize):
        cfg = create_test_config()
        feed = models.RssFeed(
            title="T",
            authors=[],
            items=[],
            base_url="https://example.com",
        )
        feed_meta = FeedMetadata(None, None, None)
        host_result = HostDetectionResult(set(), None, None)
        episodes_data = []
        for i in range(5):
            ep = create_test_episode(idx=i, title=f"Ep{i}", title_safe=f"Ep{i}")
            episodes_data.append((ep, f"/tmp/tr_{i}.txt", f"/tmp/md_{i}.json", []))

        thread_local = threading.local()
        worker_providers = [Mock()]

        summ_stage._execute_parallel_summarization(
            episodes_data,
            feed,
            cfg,
            "/tmp/out",
            None,
            feed_meta,
            host_result,
            None,
            worker_providers,
            thread_local,
            max_workers=2,
        )

        error_text = " ".join(str(c.args) for c in mock_logger.error.call_args_list if c.args)
        self.assertIn("Stopping parallel summarization", error_text)
        self.assertIn("3", error_text)


if __name__ == "__main__":
    unittest.main()
