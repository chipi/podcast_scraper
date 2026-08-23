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

from podcast_scraper.utils.sentry_init import _DEV_DSN, init_sentry, set_run_tag


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

    def test_api_dsn_malformed_returns_false_without_raising(self) -> None:
        # Regression (#24, 2026-07-22): a non-empty but INVALID DSN (no scheme)
        # made sentry_sdk.init raise BadDsn ("Unsupported scheme ''") and crash
        # the whole app at startup. init_sentry must DISABLE Sentry and return
        # False instead — telemetry must never break the app. Exercises the real
        # sentry_sdk.init (not mocked) so it hits the BadDsn path.
        os.environ["PODCAST_SENTRY_DSN_API"] = "SENTINEL_not_a_valid_dsn"
        try:
            result = init_sentry("api")
        except Exception as exc:  # noqa: BLE001 — the whole point is it must not raise
            self.fail(f"init_sentry raised on a malformed DSN (must be caught): {exc!r}")
        self.assertFalse(result)


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
class TestInitSentryDevDefault(unittest.TestCase):
    """Dev rung of the env ladder: with no DSN set, GENUINE local dev falls back
    to the ``operator-dev`` homelab DSN — but NEVER under pytest, CI, or prod.
    The suppression markers (``PYTEST_CURRENT_TEST`` / ``CI``) are popped here to
    simulate a laptop; every other test keeps them set, so the unset-DSN no-op
    contract (TestInitSentryNoOp) is unaffected."""

    def setUp(self) -> None:
        self._prior_env = {
            k: os.environ.get(k)
            for k in (
                "PODCAST_SENTRY_DSN_API",
                "PODCAST_SENTRY_DSN_PIPELINE",
                "PODCAST_ENV",
                "PODCAST_RELEASE",
                "PYTEST_CURRENT_TEST",
                "CI",
            )
        }
        for k in self._prior_env:
            os.environ.pop(k, None)
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

    def test_local_dev_no_dsn_uses_operator_dev_default(self) -> None:
        os.environ["PODCAST_ENV"] = "dev"
        self.assertTrue(init_sentry("api"))
        kwargs = self._mock_sentry_sdk.init.call_args.kwargs
        self.assertEqual(kwargs["dsn"], _DEV_DSN)
        self.assertEqual(kwargs["environment"], "dev")

    def test_ci_marker_suppresses_dev_default(self) -> None:
        os.environ["PODCAST_ENV"] = "dev"
        os.environ["CI"] = "true"
        self.assertFalse(init_sentry("api"))
        self._mock_sentry_sdk.init.assert_not_called()

    def test_pytest_marker_suppresses_dev_default(self) -> None:
        os.environ["PODCAST_ENV"] = "dev"
        os.environ["PYTEST_CURRENT_TEST"] = "something::test"
        self.assertFalse(init_sentry("api"))
        self._mock_sentry_sdk.init.assert_not_called()

    def test_prod_env_no_dsn_stays_silent(self) -> None:
        os.environ["PODCAST_ENV"] = "production"
        self.assertFalse(init_sentry("api"))
        self._mock_sentry_sdk.init.assert_not_called()


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


class TestSetRunTag(unittest.TestCase):
    """set_run_tag stamps the correlation join key onto the Sentry scope (#1053)."""

    def test_noop_without_run_id(self) -> None:
        fake = MagicMock()
        with patch.dict(sys.modules, {"sentry_sdk": fake}):
            set_run_tag(None)
        fake.set_tag.assert_not_called()

    def test_noop_when_sentry_sdk_unimportable(self) -> None:
        # absent SDK -> no raise, no effect (Sentry is an optional o11y extension).
        with patch.dict(sys.modules, {"sentry_sdk": None}):
            set_run_tag("run-1", "ep:1")  # must not raise

    def test_sets_run_and_episode_tags(self) -> None:
        fake = MagicMock()
        with patch.dict(sys.modules, {"sentry_sdk": fake}):
            set_run_tag("run-1", "ep:1")
        fake.set_tag.assert_any_call("run_id", "run-1")
        fake.set_tag.assert_any_call("episode_id", "ep:1")

    def test_episode_tag_optional(self) -> None:
        fake = MagicMock()
        with patch.dict(sys.modules, {"sentry_sdk": fake}):
            set_run_tag("run-1")
        fake.set_tag.assert_called_once_with("run_id", "run-1")


class TestEnrichmentHelpers(unittest.TestCase):
    """Enrichment-layer Sentry helpers (RFC-088 / Epic #1101 chunk 1).

    All three helpers gracefully degrade to no-op when ``sentry-sdk``
    isn't installed — Sentry is an optional o11y extension; enrichment
    runs must succeed without it.
    """

    def test_set_correlation_tags_stamps_each_string_tag(self) -> None:
        from podcast_scraper.utils.sentry_init import set_correlation_tags

        fake = MagicMock()
        with patch.dict(sys.modules, {"sentry_sdk": fake}):
            set_correlation_tags(
                {
                    "run_id": "run-1",
                    "enricher_id": "topic_similarity",
                    "tier": "embedding",
                    "attempt": "2",
                }
            )
        fake.set_tag.assert_any_call("run_id", "run-1")
        fake.set_tag.assert_any_call("enricher_id", "topic_similarity")
        fake.set_tag.assert_any_call("tier", "embedding")
        fake.set_tag.assert_any_call("attempt", "2")

    def test_set_correlation_tags_skips_non_string_values(self) -> None:
        from podcast_scraper.utils.sentry_init import set_correlation_tags

        fake = MagicMock()
        with patch.dict(sys.modules, {"sentry_sdk": fake}):
            set_correlation_tags({"run_id": "r", "attempt": 2})  # int!
        # ``attempt: 2`` (int) is silently skipped; only string tags land.
        for call in fake.set_tag.call_args_list:
            args = call.args
            self.assertNotEqual(args[0], "attempt")

    def test_set_correlation_tags_noop_when_empty(self) -> None:
        from podcast_scraper.utils.sentry_init import set_correlation_tags

        fake = MagicMock()
        with patch.dict(sys.modules, {"sentry_sdk": fake}):
            set_correlation_tags({})
        fake.set_tag.assert_not_called()

    def test_set_correlation_tags_noop_when_sdk_unimportable(self) -> None:
        from podcast_scraper.utils.sentry_init import set_correlation_tags

        with patch.dict(sys.modules, {"sentry_sdk": None}):
            set_correlation_tags({"run_id": "r"})  # must not raise

    def test_emit_enrichment_breadcrumb_calls_add_breadcrumb(self) -> None:
        from podcast_scraper.utils.sentry_init import emit_enrichment_breadcrumb

        fake = MagicMock()
        with patch.dict(sys.modules, {"sentry_sdk": fake}):
            emit_enrichment_breadcrumb(
                "enrichment.circuit_opened",
                "circuit opened for topic_consensus",
                level="warning",
                data={"enricher_id": "topic_consensus"},
            )
        fake.add_breadcrumb.assert_called_once()
        kwargs = fake.add_breadcrumb.call_args.kwargs
        self.assertEqual(kwargs["category"], "enrichment.circuit_opened")
        self.assertEqual(kwargs["message"], "circuit opened for topic_consensus")
        self.assertEqual(kwargs["level"], "warning")

    def test_emit_enrichment_breadcrumb_noop_when_sdk_unimportable(self) -> None:
        from podcast_scraper.utils.sentry_init import emit_enrichment_breadcrumb

        with patch.dict(sys.modules, {"sentry_sdk": None}):
            emit_enrichment_breadcrumb("enrichment.auto_disabled", "x")  # must not raise

    def test_capture_enrichment_message_calls_capture_with_tags(self) -> None:
        from podcast_scraper.utils.sentry_init import capture_enrichment_message

        fake = MagicMock()
        with patch.dict(sys.modules, {"sentry_sdk": fake}):
            capture_enrichment_message(
                "topic_consensus auto-disabled after 2 failed runs",
                level="warning",
                tags={"enricher_id": "topic_consensus", "run_id": "job-1"},
            )
        fake.capture_message.assert_called_once()
        self.assertEqual(
            fake.capture_message.call_args.args[0],
            "topic_consensus auto-disabled after 2 failed runs",
        )

    def test_capture_enrichment_message_noop_when_sdk_unimportable(self) -> None:
        from podcast_scraper.utils.sentry_init import capture_enrichment_message

        with patch.dict(sys.modules, {"sentry_sdk": None}):
            capture_enrichment_message("x")  # must not raise


class TestBeforeSendRedaction(unittest.TestCase):
    """`_before_send` scrubs secrets from anything the SDK would send."""

    def test_redacts_extra_contexts_headers_and_frame_vars(self) -> None:
        from podcast_scraper.utils.sentry_init import _before_send

        event = {
            "extra": {"OPENAI_API_KEY": "sk-live", "note": "keep"},
            "contexts": {"auth_token": "abc", "os": "linux"},
            "request": {"headers": {"Authorization": "Bearer x", "Accept": "json"}},
            "exception": {
                "values": [
                    {"stacktrace": {"frames": [{"vars": {"password": "hunter2", "n": "3"}}]}}
                ]
            },
        }
        out = _before_send(event, None)
        self.assertEqual(out["extra"]["OPENAI_API_KEY"], "[redacted]")
        self.assertEqual(out["extra"]["note"], "keep")
        self.assertEqual(out["contexts"]["auth_token"], "[redacted]")
        self.assertEqual(out["contexts"]["os"], "linux")
        self.assertEqual(out["request"]["headers"]["Authorization"], "[redacted]")
        self.assertEqual(out["request"]["headers"]["Accept"], "json")
        frame_vars = out["exception"]["values"][0]["stacktrace"]["frames"][0]["vars"]
        self.assertEqual(frame_vars["password"], "[redacted]")
        self.assertEqual(frame_vars["n"], "3")

    def test_never_raises_on_malformed_event(self) -> None:
        from podcast_scraper.utils.sentry_init import _before_send

        # Missing / oddly-typed sections must not raise — the error event is still useful.
        self.assertIs(_before_send({}, None).__class__, dict)
        _before_send({"extra": None, "exception": "nope"}, None)  # must not raise


class TestCaptureStageException(unittest.TestCase):
    """o11y P1: swallowed stage failures reach GlitchTip, tagged stage + run/episode ids."""

    def setUp(self) -> None:
        # Import correlation BEFORE patch.dict starts. patch.dict(sys.modules) restores a
        # snapshot on exit, EVICTING anything imported inside its window — and
        # capture_stage_exception imports correlation lazily, inside that window. Evicted, the
        # next test's `correlation.set_run_id("run-x")` lands on one module instance while the
        # function under test re-imports a fresh one whose `_RUN_ID` global is still None, so the
        # run_id tag silently disappears. The tests only passed because the fragile one happened
        # to sort first; adding a test starting with "a" was enough to break it.
        from podcast_scraper.utils import correlation

        self._mock_sdk = MagicMock()
        self._scope = MagicMock()
        self._mock_sdk.push_scope.return_value.__enter__.return_value = self._scope
        self._patch = patch.dict(sys.modules, {"sentry_sdk": self._mock_sdk})
        self._patch.start()

        correlation.set_run_id("run-x")
        correlation.set_episode_id("ep-x")

    def tearDown(self) -> None:
        self._patch.stop()
        from podcast_scraper.utils import correlation

        correlation.set_run_id(None)
        correlation.set_episode_id(None)

    def test_captures_with_stage_and_correlation_tags(self) -> None:
        from podcast_scraper.utils.sentry_init import capture_stage_exception

        exc = ValueError("boom")
        capture_stage_exception(exc, stage="transcription")
        self._mock_sdk.capture_exception.assert_called_once_with(exc)
        self._scope.set_tag.assert_any_call("stage", "transcription")
        self._scope.set_tag.assert_any_call("run_id", "run-x")
        self._scope.set_tag.assert_any_call("episode_id", "ep-x")

    def test_severity_defaults_to_the_sdk_error_level(self) -> None:
        """#1632: unset level must not touch the scope — GI/KG keep reporting as errors."""
        from podcast_scraper.utils.sentry_init import capture_stage_exception

        self._scope.level = "__never-assigned__"
        capture_stage_exception(ValueError("boom"), stage="gi")
        # Untouched => Sentry applies its own default, which is "error".
        assert self._scope.level == "__never-assigned__"

    def test_a_by_design_degradation_is_reported_as_a_warning(self) -> None:
        """#1632: a summary that degraded exactly as intended must not look like a crash.

        The recoverable-summary branch (#1496) keeps the episode's transcript, GI and KG, logs at
        WARNING and records the outcome in the stage ledger. It still reached GlitchTip at the
        SDK's default `error` severity, so an automated triage pass could not tell it apart from
        a broken run and filed it as a bug. The event is still captured — only its loudness
        changes.
        """
        from podcast_scraper.utils.sentry_init import capture_stage_exception

        exc = ValueError("Summary schema parsing failed after the ADR-148 re-roll")
        capture_stage_exception(exc, stage="summary", level="warning")

        self._mock_sdk.capture_exception.assert_called_once_with(exc)
        assert self._scope.level == "warning"
        # Correlation tags must survive the severity change — they are what makes it triageable.
        self._scope.set_tag.assert_any_call("stage", "summary")
        self._scope.set_tag.assert_any_call("episode_id", "ep-x")

    def test_no_sentry_sdk_is_noop(self) -> None:
        with patch.dict(sys.modules, {"sentry_sdk": None}):
            from podcast_scraper.utils.sentry_init import capture_stage_exception

            capture_stage_exception(ValueError("x"), stage="gi")  # importable → no raise


if __name__ == "__main__":
    unittest.main()
