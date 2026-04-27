#!/usr/bin/env python3
"""Tests for ``init_sentry`` Sentry SDK bootstrap helper (#681).

The helper is consumed by the api startup hook and by the pipeline
subprocess entrypoint; both surfaces should:

* No-op cleanly when the relevant DSN env var is unset.
* Initialise the SDK with the right environment + release tags when set.
* Tag events with ``component=<api|pipeline>`` so the two streams stay
  separable in the Sentry UI.

These tests stub the ``sentry_sdk`` module via ``sys.modules`` patching
because ``sentry-sdk`` is in the base install (no need for [llm]/[ml]),
but the unit tier still mocks it to avoid a real ``sentry_sdk.init``
side-effect on the Sentry hub during test collection.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

import pytest

from podcast_scraper.utils.sentry_init import init_sentry


@pytest.mark.unit
class TestInitSentryNoOp(unittest.TestCase):
    """When DSN env vars are unset, init returns False without side effects."""

    def setUp(self) -> None:
        # Snapshot env so test doesn't leak.
        self._prior_env = {
            k: os.environ.get(k)
            for k in (
                "PODCAST_SENTRY_DSN_API",
                "PODCAST_SENTRY_DSN_PIPELINE",
                "PODCAST_ENV",
                "PODCAST_RELEASE",
            )
        }
        for k in self._prior_env:
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        for k, v in self._prior_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_api_dsn_unset_returns_false(self) -> None:
        self.assertFalse(init_sentry("api"))

    def test_pipeline_dsn_unset_returns_false(self) -> None:
        self.assertFalse(init_sentry("pipeline"))

    def test_api_dsn_blank_string_returns_false(self) -> None:
        os.environ["PODCAST_SENTRY_DSN_API"] = "   "
        self.assertFalse(init_sentry("api"))


@pytest.mark.unit
class TestInitSentryInitPath(unittest.TestCase):
    """When DSN is set, init_sentry calls sentry_sdk.init with right args."""

    def setUp(self) -> None:
        self._prior_env = {
            k: os.environ.get(k)
            for k in (
                "PODCAST_SENTRY_DSN_API",
                "PODCAST_SENTRY_DSN_PIPELINE",
                "PODCAST_ENV",
                "PODCAST_RELEASE",
            )
        }
        for k in self._prior_env:
            os.environ.pop(k, None)
        # Stub sentry_sdk so init() doesn't actually mutate the global hub.
        self._mock_sentry_sdk = MagicMock()
        self._mock_sentry_sdk.init = Mock()
        self._mock_sentry_sdk.set_tag = Mock()
        self._patch_sentry = patch.dict(sys.modules, {"sentry_sdk": self._mock_sentry_sdk})
        self._patch_sentry.start()

    def tearDown(self) -> None:
        self._patch_sentry.stop()
        for k, v in self._prior_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_api_dsn_set_initialises_with_traces_rate_0_1(self) -> None:
        os.environ["PODCAST_SENTRY_DSN_API"] = "https://x@sentry.io/1"
        os.environ["PODCAST_ENV"] = "preprod"
        os.environ["PODCAST_RELEASE"] = "podcast-scraper@2.6.0"

        self.assertTrue(init_sentry("api"))

        self._mock_sentry_sdk.init.assert_called_once()
        kwargs = self._mock_sentry_sdk.init.call_args.kwargs
        self.assertEqual(kwargs["dsn"], "https://x@sentry.io/1")
        self.assertEqual(kwargs["environment"], "preprod")
        self.assertEqual(kwargs["release"], "podcast-scraper@2.6.0")
        self.assertEqual(kwargs["traces_sample_rate"], 0.1)
        self.assertFalse(kwargs["send_default_pii"])

        self._mock_sentry_sdk.set_tag.assert_called_with("component", "api")

    def test_pipeline_dsn_set_initialises_with_traces_rate_0(self) -> None:
        os.environ["PODCAST_SENTRY_DSN_PIPELINE"] = "https://y@sentry.io/2"
        os.environ["PODCAST_ENV"] = "preprod"

        self.assertTrue(init_sentry("pipeline"))

        kwargs = self._mock_sentry_sdk.init.call_args.kwargs
        self.assertEqual(kwargs["dsn"], "https://y@sentry.io/2")
        self.assertEqual(kwargs["environment"], "preprod")
        # Pipeline runs short bursty jobs; default traces rate is 0 to
        # protect the Sentry free tier transaction quota.
        self.assertEqual(kwargs["traces_sample_rate"], 0.0)

        self._mock_sentry_sdk.set_tag.assert_called_with("component", "pipeline")

    def test_release_falls_back_to_package_version_when_unset(self) -> None:
        os.environ["PODCAST_SENTRY_DSN_API"] = "https://x@sentry.io/1"
        # PODCAST_RELEASE deliberately unset.
        self.assertTrue(init_sentry("api"))
        kwargs = self._mock_sentry_sdk.init.call_args.kwargs
        # The fallback wraps the package __version__ with the
        # ``podcast-scraper@`` prefix so events at least group.
        self.assertIsNotNone(kwargs["release"])
        self.assertTrue(kwargs["release"].startswith("podcast-scraper@"))

    def test_environment_defaults_to_dev_when_unset(self) -> None:
        os.environ["PODCAST_SENTRY_DSN_API"] = "https://x@sentry.io/1"
        # PODCAST_ENV deliberately unset.
        self.assertTrue(init_sentry("api"))
        kwargs = self._mock_sentry_sdk.init.call_args.kwargs
        self.assertEqual(kwargs["environment"], "dev")


@pytest.mark.unit
class TestInitSentryImportError(unittest.TestCase):
    """When sentry_sdk import fails despite DSN being set, return False."""

    def setUp(self) -> None:
        self._prior_dsn = os.environ.get("PODCAST_SENTRY_DSN_API")
        os.environ["PODCAST_SENTRY_DSN_API"] = "https://x@sentry.io/1"

    def tearDown(self) -> None:
        if self._prior_dsn is None:
            os.environ.pop("PODCAST_SENTRY_DSN_API", None)
        else:
            os.environ["PODCAST_SENTRY_DSN_API"] = self._prior_dsn

    def test_returns_false_when_sentry_sdk_unimportable(self) -> None:
        # Replace ``sentry_sdk`` with a None entry so ``import sentry_sdk``
        # raises ImportError without uninstalling the real package.
        with patch.dict(sys.modules, {"sentry_sdk": None}):
            self.assertFalse(init_sentry("api"))


if __name__ == "__main__":
    unittest.main()
