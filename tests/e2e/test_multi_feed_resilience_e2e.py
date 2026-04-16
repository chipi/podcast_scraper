#!/usr/bin/env python3
"""Multi-feed resilience E2E (GitHub #560).

Offline-only: local ``e2e_server`` / ``E2EHTTPRequestHandler``. Covers corpus
artifacts (``corpus_run_summary.json``), soft ``failure_kind`` (#559), optional
default lenient vs ``multi_feed_strict`` / ``--multi-feed-strict`` and CLI exit codes,
unknown RSS slug (404), wrong ``feed.xml`` path under a known feed (404),
transient RSS 503 on one feed (retry then success), advisory corpus lock
(``service.run`` fails fast when ``.podcast_scraper.lock`` is already held), and
multi-feed + multi-episode partial transcript failure (multi_episode mode only;
pairs ``podcast1_episode_selection`` with ``podcast1_with_transcript``).
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
from filelock import FileLock

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import cli, Config, service
from podcast_scraper.utils.corpus_lock import LOCK_BASENAME

try:
    from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler
except ImportError:
    E2EHTTPRequestHandler = None  # type: ignore[assignment,misc]


def _cleanup_errors() -> None:
    if E2EHTTPRequestHandler is not None:
        E2EHTTPRequestHandler.clear_all_error_behaviors()


@pytest.fixture(autouse=True)
def _reset_multi_feed_resilience_errors():
    _cleanup_errors()
    yield
    _cleanup_errors()


def _read_corpus_run_summary(corpus_parent: Path) -> Dict[str, Any]:
    path = corpus_parent / "corpus_run_summary.json"
    assert path.is_file(), f"expected corpus_run_summary.json at {path}"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _multi_feed_base_kwargs() -> Dict[str, Any]:
    return {
        "max_episodes": 1,
        "transcribe_missing": False,
        "http_retry_total": 2,
        "http_backoff_factor": 0.0,
        "rss_retry_total": 2,
        "rss_backoff_factor": 0.0,
    }


def _run_json_summaries_under_feeds(corpus_parent: Path) -> list[Dict[str, Any]]:
    """Load ``run.json`` dicts that live under ``feeds/`` (per-feed pipeline runs)."""
    out: list[Dict[str, Any]] = []
    for path in corpus_parent.rglob("run.json"):
        if "feeds" not in path.parts:
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(raw, dict):
            out.append(raw)
    return out


def _run_json_skipped_total(run_doc: Dict[str, Any]) -> int:
    metrics = run_doc.get("metrics")
    if not isinstance(metrics, dict):
        return 0
    return int(metrics.get("episodes_skipped_total") or 0)


@pytest.mark.e2e
@pytest.mark.critical_path
class TestMultiFeedCorpusSummaryAndSoftKind:
    """corpus_run_summary + failure_kind; default soft exit vs strict (#559)."""

    def test_rss_500_on_one_feed_default_soft_exit_success(self, e2e_server):
        """Permanent RSS HTTP error: soft failure_kind; default exit-zero policy."""
        assert E2EHTTPRequestHandler is not None
        E2EHTTPRequestHandler.set_error_behavior("/feeds/podcast2/feed.xml", status=500)

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            cfg = Config(
                rss_url=e2e_server.urls.feed("podcast1_episode_selection"),
                rss_urls=[
                    e2e_server.urls.feed("podcast1_episode_selection"),
                    e2e_server.urls.feed("podcast2"),
                ],
                output_dir=str(tmpdir),
                **_multi_feed_base_kwargs(),
            )
            result = service.run(cfg)

            assert result.episodes_processed >= 1
            assert result.success is True
            assert result.error is None
            assert result.soft_failures is not None

            doc = _read_corpus_run_summary(tmpdir)
            assert doc.get("overall_ok") is False
            rows = doc.get("feeds") or []
            assert len(rows) == 2
            failed = [r for r in rows if r.get("ok") is False]
            assert len(failed) == 1
            assert failed[0].get("failure_kind") == "soft"
            err = failed[0].get("error")
            assert isinstance(err, str) and err.strip()
            assert "podcast2" in (failed[0].get("feed_url") or "")

            ok_rows = [r for r in rows if r.get("ok") is True]
            assert len(ok_rows) == 1
            assert ok_rows[0].get("failure_kind") is None

    def test_rss_500_strict_mode_service_fails(self, e2e_server):
        """multi_feed_strict=True: strict success False on soft RSS failure."""
        assert E2EHTTPRequestHandler is not None
        E2EHTTPRequestHandler.set_error_behavior("/feeds/podcast2/feed.xml", status=500)

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            cfg = Config(
                rss_url=e2e_server.urls.feed("podcast1_episode_selection"),
                rss_urls=[
                    e2e_server.urls.feed("podcast1_episode_selection"),
                    e2e_server.urls.feed("podcast2"),
                ],
                output_dir=str(tmpdir),
                multi_feed_strict=True,
                **_multi_feed_base_kwargs(),
            )
            result = service.run(cfg)

            assert result.success is False
            assert result.error is not None
            assert result.soft_failures is None

            doc = _read_corpus_run_summary(tmpdir)
            assert doc.get("overall_ok") is False
            rows = doc.get("feeds") or []
            failed = [r for r in rows if r.get("ok") is False]
            assert len(failed) == 1
            assert failed[0].get("failure_kind") == "soft"

    def test_unknown_rss_slug_404_is_soft_classified(self, e2e_server):
        """Valid /feeds/<slug>/feed.xml path with unknown slug -> 404 -> soft."""
        assert E2EHTTPRequestHandler is not None
        bad_url = f"{e2e_server.urls.base()}/feeds/zzznonexistentfeedslug/feed.xml"

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            cfg = Config(
                rss_url=e2e_server.urls.feed("podcast1_episode_selection"),
                rss_urls=[
                    e2e_server.urls.feed("podcast1_episode_selection"),
                    bad_url,
                ],
                output_dir=str(tmpdir),
                **_multi_feed_base_kwargs(),
            )
            result = service.run(cfg)

            assert result.success is True
            assert result.error is None
            assert result.soft_failures is not None
            doc = _read_corpus_run_summary(tmpdir)
            failed = [r for r in (doc.get("feeds") or []) if r.get("ok") is False]
            assert len(failed) == 1
            assert failed[0].get("failure_kind") == "soft"

    def test_second_feed_wrong_xml_path_404_is_soft_classified(self, e2e_server):
        """Known feed slug but not ``.../feed.xml`` -> 404 on e2e_server -> soft row."""
        assert E2EHTTPRequestHandler is not None
        bad_url = f"{e2e_server.urls.base()}/feeds/podcast2/not_feed.xml"

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            cfg = Config(
                rss_url=e2e_server.urls.feed("podcast1_episode_selection"),
                rss_urls=[
                    e2e_server.urls.feed("podcast1_episode_selection"),
                    bad_url,
                ],
                output_dir=str(tmpdir),
                **_multi_feed_base_kwargs(),
            )
            result = service.run(cfg)

            assert result.success is True
            assert result.error is None
            assert result.soft_failures is not None

            doc = _read_corpus_run_summary(tmpdir)
            assert doc.get("overall_ok") is False
            failed = [r for r in (doc.get("feeds") or []) if r.get("ok") is False]
            assert len(failed) == 1
            assert failed[0].get("failure_kind") == "soft"
            assert bad_url in (failed[0].get("feed_url") or "")

    def test_cli_multi_feed_default_exits_zero_on_soft_only(self, e2e_server):
        """CLI default: exit 0 when every failed feed is soft-classified."""
        assert E2EHTTPRequestHandler is not None
        E2EHTTPRequestHandler.set_error_behavior("/feeds/podcast2/feed.xml", status=500)

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            code = cli.main(
                [
                    e2e_server.urls.feed("podcast1_episode_selection"),
                    "--rss",
                    e2e_server.urls.feed("podcast2"),
                    "--output-dir",
                    str(tmpdir),
                    "--max-episodes",
                    "1",
                    "--no-transcribe-missing",
                    "--http-retry-total",
                    "2",
                    "--http-backoff-factor",
                    "0",
                    "--rss-retry-total",
                    "2",
                    "--rss-backoff-factor",
                    "0",
                ]
            )
            assert code == 0, f"expected exit 0, got {code}"
            doc = _read_corpus_run_summary(tmpdir)
            assert doc.get("overall_ok") is False

    def test_cli_multi_feed_strict_exits_1_on_soft_only(self, e2e_server):
        """CLI --multi-feed-strict: exit 1 on soft-only feed failure."""
        assert E2EHTTPRequestHandler is not None
        E2EHTTPRequestHandler.set_error_behavior("/feeds/podcast2/feed.xml", status=500)

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            code = cli.main(
                [
                    e2e_server.urls.feed("podcast1_episode_selection"),
                    "--rss",
                    e2e_server.urls.feed("podcast2"),
                    "--output-dir",
                    str(tmpdir),
                    "--max-episodes",
                    "1",
                    "--no-transcribe-missing",
                    "--http-retry-total",
                    "2",
                    "--http-backoff-factor",
                    "0",
                    "--rss-retry-total",
                    "2",
                    "--rss-backoff-factor",
                    "0",
                    "--multi-feed-strict",
                ]
            )
            assert code == 1, f"expected exit 1, got {code}"
            doc = _read_corpus_run_summary(tmpdir)
            assert doc.get("overall_ok") is False

    def test_cli_strict_unknown_slug_exits_1_on_soft_only(self, e2e_server):
        """CLI ``--multi-feed-strict`` with unknown slug RSS URL -> exit 1 (soft-only batch)."""
        assert E2EHTTPRequestHandler is not None
        bad_url = f"{e2e_server.urls.base()}/feeds/zzznonexistentfeedslug/feed.xml"

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            code = cli.main(
                [
                    e2e_server.urls.feed("podcast1_episode_selection"),
                    "--rss",
                    bad_url,
                    "--output-dir",
                    str(tmpdir),
                    "--max-episodes",
                    "1",
                    "--no-transcribe-missing",
                    "--http-retry-total",
                    "2",
                    "--http-backoff-factor",
                    "0",
                    "--rss-retry-total",
                    "2",
                    "--rss-backoff-factor",
                    "0",
                    "--multi-feed-strict",
                ]
            )
            assert code == 1, f"expected exit 1, got {code}"
            doc = _read_corpus_run_summary(tmpdir)
            assert doc.get("overall_ok") is False


@pytest.mark.e2e
class TestMultiFeedMultiEpisodePartialTranscriptFailure:
    """Requires E2E_TEST_MODE=multi_episode (fast mode caps episodes to 1)."""

    def test_partial_transcript_404_one_feed_other_feed_ok(self, e2e_server):
        if os.environ.get("E2E_TEST_MODE", "multi_episode").lower() == "fast":
            pytest.skip("Needs multi_episode mode so max_episodes>1 applies per feed")

        assert E2EHTTPRequestHandler is not None
        E2EHTTPRequestHandler.set_error_behavior("/transcripts/p01_multi_e03.txt", status=404)

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            cfg = Config(
                rss_url=e2e_server.urls.feed("podcast1_episode_selection"),
                rss_urls=[
                    e2e_server.urls.feed("podcast1_episode_selection"),
                    e2e_server.urls.feed("podcast1_with_transcript"),
                ],
                output_dir=str(tmpdir),
                max_episodes=3,
                transcribe_missing=False,
                http_retry_total=2,
                http_backoff_factor=0.0,
                rss_retry_total=2,
                rss_backoff_factor=0.0,
            )
            result = service.run(cfg)

            assert result.success is True
            assert result.error is None
            assert result.soft_failures is None

            doc = _read_corpus_run_summary(tmpdir)
            assert doc.get("overall_ok") is True

            feeds_root = tmpdir / "feeds"
            assert feeds_root.is_dir()
            metrics_paths = [
                p
                for p in feeds_root.rglob("metrics.json")
                if any(str(part).startswith("run_") for part in p.parts)
            ]
            assert len(metrics_paths) >= 2

            skipped_totals: list[int] = []
            for mp in metrics_paths:
                data = json.loads(mp.read_text(encoding="utf-8"))
                skipped_totals.append(int(data.get("episodes_skipped_total") or 0))

            assert max(skipped_totals) >= 1, (
                "transcript 404 with transcribe_missing=False should increment "
                f"episodes_skipped_total in one feed's metrics (got {skipped_totals=})"
            )

            run_summaries = _run_json_summaries_under_feeds(tmpdir)
            assert (
                len(run_summaries) >= 2
            ), f"expected per-feed run.json files, got {run_summaries=}"
            with_skips = [s for s in run_summaries if _run_json_skipped_total(s) > 0]
            without_skips = [s for s in run_summaries if _run_json_skipped_total(s) == 0]
            assert with_skips, (
                "transcript 404 should surface as non-zero episodes_skipped_total "
                "in that feed's run.json metrics"
            )
            assert without_skips, (
                "the other feed's run.json should have no skipped episodes "
                f"(got skipped totals {[ _run_json_skipped_total(s) for s in run_summaries ]})"
            )


@pytest.mark.e2e
@pytest.mark.critical_path
class TestMultiFeedTransientRSSRecovery:
    """RSS transient errors on one feed; retries succeed; other feed unaffected."""

    def test_transient_503_on_second_feed_then_both_ok(self, e2e_server):
        """Two RSS 503 responses on podcast2 feed.xml then 200; batch succeeds."""
        assert E2EHTTPRequestHandler is not None
        E2EHTTPRequestHandler.set_transient_error(
            "/feeds/podcast2/feed.xml",
            status=503,
            fail_count=2,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            cfg = Config(
                rss_url=e2e_server.urls.feed("podcast1_episode_selection"),
                rss_urls=[
                    e2e_server.urls.feed("podcast1_episode_selection"),
                    e2e_server.urls.feed("podcast2"),
                ],
                output_dir=str(tmpdir),
                max_episodes=1,
                transcribe_missing=False,
                http_retry_total=3,
                http_backoff_factor=0.0,
                rss_retry_total=3,
                rss_backoff_factor=0.0,
            )
            result = service.run(cfg)

            assert result.success is True
            assert result.error is None
            assert result.soft_failures is None
            assert result.episodes_processed >= 1

            doc = _read_corpus_run_summary(tmpdir)
            assert doc.get("overall_ok") is True
            rows = doc.get("feeds") or []
            assert len(rows) == 2
            assert all(r.get("ok") is True for r in rows)


@pytest.mark.e2e
@pytest.mark.critical_path
class TestMultiFeedCorpusLockContention:
    """Advisory corpus lock: concurrent holder blocks second ``service.run``."""

    def test_service_fails_fast_when_lock_held_elsewhere(self, e2e_server, monkeypatch):
        """Simulate another writer by pre-acquiring the same ``.podcast_scraper.lock``."""
        monkeypatch.setenv("PODCAST_SCRAPER_CORPUS_LOCK", "1")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            lock_path = root / LOCK_BASENAME
            outer = FileLock(str(lock_path), timeout=0)
            outer.acquire()
            try:
                cfg = Config(
                    rss_url=e2e_server.urls.feed("podcast1_episode_selection"),
                    rss_urls=[
                        e2e_server.urls.feed("podcast1_episode_selection"),
                        e2e_server.urls.feed("podcast2"),
                    ],
                    output_dir=str(root),
                    **_multi_feed_base_kwargs(),
                )
                blocked = service.run(cfg)
                assert blocked.success is False
                assert blocked.episodes_processed == 0
                assert blocked.multi_feed_summary is None
                err = (blocked.error or "").lower()
                assert "locked" in err or "lock" in err
            finally:
                outer.release()

            cfg2 = Config(
                rss_url=e2e_server.urls.feed("podcast1_episode_selection"),
                rss_urls=[
                    e2e_server.urls.feed("podcast1_episode_selection"),
                    e2e_server.urls.feed("podcast2"),
                ],
                output_dir=str(root),
                **_multi_feed_base_kwargs(),
            )
            ok = service.run(cfg2)
            assert ok.success is True
            assert _read_corpus_run_summary(root).get("overall_ok") is True
