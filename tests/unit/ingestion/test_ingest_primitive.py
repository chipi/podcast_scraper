"""Unit tests for the corpus ingestion primitive (#1069).

Exercise the orchestration ``ingest_feed`` performs — authorize → dedup → run →
result mapping — with an **injected stub runner**, so no real ML pipeline runs.
This is the phase-1 spine both PRD-037 phases share; the policy seam here is
where phase-2 guardrails will slot in.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast, TYPE_CHECKING

import pytest

from podcast_scraper.ingestion import (
    AllowAllPolicy,
    ingest_feed,
    IngestNotAuthorized,
    IngestRequest,
)
from podcast_scraper.ingestion.primitive import (
    STATUS_ALREADY_PRESENT,
    STATUS_FAILED,
    STATUS_INGESTED,
)
from podcast_scraper.utils.filesystem import feed_workspace_dirname

if TYPE_CHECKING:
    from podcast_scraper import config

pytestmark = [pytest.mark.unit]

_URL = "https://example.com/feed.xml"


def _corpus_cfg(corpus: Path) -> "config.Config":
    """A single-feed corpus-layout config stub: output_dir = <corpus>/feeds/<stable>.

    A ``SimpleNamespace`` is enough — ``ingest_feed`` only reads ``rss_url`` /
    ``output_dir`` / ``single_feed_uses_corpus_layout`` off ``cfg`` and passes it to
    the injected runner — so we cast it to ``Config`` for the type checker.
    """
    output_dir = corpus / "feeds" / feed_workspace_dirname(_URL)
    return cast(
        "config.Config",
        SimpleNamespace(
            rss_url=_URL,
            output_dir=str(output_dir),
            single_feed_uses_corpus_layout=True,
        ),
    )


class _RecordingRunner:
    """Stub for ``service.run`` — records calls and returns a canned ServiceResult."""

    def __init__(
        self, *, success: bool = True, episodes: int = 0, error: str | None = None
    ) -> None:
        self.calls: list[object] = []
        self._result = SimpleNamespace(
            success=success, episodes_processed=episodes, summary="stub-summary", error=error
        )

    def __call__(self, cfg: object) -> object:
        self.calls.append(cfg)
        return self._result


def test_fresh_feed_runs_and_reports_added(tmp_path: Path) -> None:
    cfg = _corpus_cfg(tmp_path / "corpus")
    runner = _RecordingRunner(success=True, episodes=3)

    result = ingest_feed(cfg, run=runner)

    assert result.status == STATUS_INGESTED
    assert result.episodes_added == 3
    assert result.feed_url == _URL
    assert result.feed_dir == feed_workspace_dirname(_URL)
    assert len(runner.calls) == 1  # the pipeline ran once


def test_already_present_is_a_no_op(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    cfg = _corpus_cfg(corpus)
    Path(str(cfg.output_dir)).mkdir(parents=True)  # feed dir already in the corpus
    runner = _RecordingRunner(success=True, episodes=5)

    result = ingest_feed(cfg, run=runner)

    assert result.status == STATUS_ALREADY_PRESENT
    assert result.episodes_added == 0
    assert runner.calls == []  # dedup short-circuits before any pipeline work


def test_force_reruns_when_already_present(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    cfg = _corpus_cfg(corpus)
    Path(str(cfg.output_dir)).mkdir(parents=True)
    runner = _RecordingRunner(success=True, episodes=1)

    result = ingest_feed(cfg, run=runner, force=True)

    assert result.status == STATUS_INGESTED
    assert result.episodes_added == 1
    assert len(runner.calls) == 1  # force overrides the dedup short-circuit


def test_failed_run_maps_to_failed(tmp_path: Path) -> None:
    cfg = _corpus_cfg(tmp_path / "corpus")
    runner = _RecordingRunner(success=False, error="feed 404")

    result = ingest_feed(cfg, run=runner)

    assert result.status == STATUS_FAILED
    assert result.error == "feed 404"


def test_missing_rss_url_fails_without_running(tmp_path: Path) -> None:
    cfg = cast(
        "config.Config",
        SimpleNamespace(
            rss_url=None, output_dir=str(tmp_path), single_feed_uses_corpus_layout=True
        ),
    )
    runner = _RecordingRunner()

    result = ingest_feed(cfg, run=runner)

    assert result.status == STATUS_FAILED
    assert "rss_url" in (result.error or "")
    assert runner.calls == []


def test_policy_rejection_short_circuits_before_run(tmp_path: Path) -> None:
    cfg = _corpus_cfg(tmp_path / "corpus")
    runner = _RecordingRunner(success=True, episodes=9)

    class _DenyPolicy:
        def authorize(self, request: IngestRequest) -> None:
            raise IngestNotAuthorized(f"quota exceeded for {request.actor}")

    with pytest.raises(IngestNotAuthorized, match="quota exceeded"):
        ingest_feed(cfg, run=runner, policy=_DenyPolicy(), actor="user:alice")

    assert runner.calls == []  # rejected before any pipeline work


def test_policy_receives_actor_and_feed(tmp_path: Path) -> None:
    cfg = _corpus_cfg(tmp_path / "corpus")
    seen: list[IngestRequest] = []

    class _CapturePolicy:
        def authorize(self, request: IngestRequest) -> None:
            seen.append(request)

    ingest_feed(cfg, run=_RecordingRunner(), policy=_CapturePolicy(), actor="user:bob")

    assert len(seen) == 1
    assert seen[0].feed_url == _URL
    assert seen[0].actor == "user:bob"


def test_allow_all_policy_is_the_default() -> None:
    # The operator-path default gates nothing.
    AllowAllPolicy().authorize(IngestRequest(feed_url=_URL))
