"""W4 (#1874): attribution enforced by a test, not by remembering.

Every event, span and error is supposed to name its run, feed, episode and profile (#1873).
That held on the day it was written. The question this file answers is whether it still holds
after the NEXT person adds an event — because "remember to attribute it" is not a mechanism,
and the reason #1873 existed at all is that four surfaces each drifted their own way.

Two kinds of check:
  * a PROPERTY check that exercises the real emit path from a worker thread, because that is
    where most stage events are raised and where the ContextVar silently did not reach; and
  * a STRUCTURAL check over the source, so a new ``emit_event`` call site cannot quietly opt
    out of the run context.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]


def _src_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "src" / "podcast_scraper"
        if candidate.is_dir():
            return candidate
    raise AssertionError("src/podcast_scraper not found from the test file")


class TestAttributionSurvivesAWorkerThread:
    """The pipeline raises most events inside per-stage ThreadPoolExecutors."""

    @pytest.fixture()
    def wired(self, monkeypatch):
        monkeypatch.setenv("DEEPGRAM_API_KEY", "dummy-for-validation")
        monkeypatch.setenv("DGX_TAILNET_HOST", "dgx.test.ts.net")
        from podcast_scraper import config as config_mod
        from podcast_scraper.workflow.orchestration import stamp_run_identity

        cfg = config_mod.Config(
            rss_url="https://feed.example/f.xml", profile="cloud_with_dgx_primary"
        )
        stamp_run_identity(cfg)
        from podcast_scraper.utils import correlation

        correlation.set_episode_id("ep-in-a-worker")
        yield cfg

    def test_run_scoped_attribution_survives_a_worker(self, wired) -> None:
        from podcast_scraper.obs.events import emit_event

        with ThreadPoolExecutor(max_workers=1) as pool:
            line = pool.submit(emit_event, "llm_cost", provider="litellm").result()

        assert line is not None, "emit_event returned None — the event was never serialised"
        record = json.loads(line)
        for field in ("profile", "asr_provider", "diarization_provider", "feed_id"):
            assert record.get(field), (
                f"event raised in a worker thread is missing {field!r} — most stage events are "
                f"raised exactly this way, so an unattributed worker event is the common case, "
                f"not the edge case. got: {sorted(record)}"
            )

    def test_episode_id_does_NOT_survive_a_worker_and_that_is_deliberate(self, wired) -> None:
        """Pins a KNOWN limitation so it stays known (#1874 W4 follow-up).

        ``episode_id`` is a ContextVar with no thread-visible mirror, unlike feed/profile.
        That asymmetry is deliberate, not an oversight to paper over: a batch walks feeds
        SEQUENTIALLY, so a module-global feed is always the right one, whereas episodes are
        processed CONCURRENTLY inside stage pools — a global episode id would attribute one
        episode's events to another. Wrong attribution is worse than missing attribution,
        because the entire value of these fields is being trustworthy without a cross-check.

        The correct fix is propagating the context at submit time (``contextvars.copy_context``
        / the existing ``wrap_with_current_context``) into the three stage pools, which is a
        change to the executors rather than to telemetry. Until that lands, this test documents
        the gap and will fail — deliberately — the moment someone "fixes" it with a global.
        """
        from podcast_scraper.obs.events import emit_event

        with ThreadPoolExecutor(max_workers=1) as pool:
            line = pool.submit(emit_event, "llm_cost").result()
        assert line is not None
        record = json.loads(line)

        assert record.get("episode_id") is None, (
            "episode_id now survives a worker thread. If that came from context PROPAGATION, "
            "delete this test — the gap is closed. If it came from a module global, revert it: "
            "concurrent episodes would cross-attribute."
        )

    def test_the_log_record_is_attributed_in_a_worker_too(self, wired) -> None:
        import logging

        from podcast_scraper.utils.correlation import CorrelationFormatter

        formatter = CorrelationFormatter("%(run_id)s|%(feed_id)s|%(profile)s|%(message)s")

        def _format() -> str:
            record = logging.LogRecord("t", logging.INFO, __file__, 1, "stage failed", None, None)
            return str(formatter.format(record))

        with ThreadPoolExecutor(max_workers=1) as pool:
            rendered = pool.submit(_format).result()

        assert "feed.example" in rendered, f"log line lost the feed in a worker: {rendered}"
        assert "cloud_with_dgx_primary" in rendered, f"log line lost the profile: {rendered}"


class TestNoEventEscapesTheRunContext:
    """Structural: emit_event must merge the run context for EVERY caller.

    A per-call-site opt-in is stale the first time someone forgets. This asserts the merge
    happens inside emit_event itself, so the guarantee is a property of the function rather
    than of every author who ever calls it.
    """

    def test_emit_event_merges_the_run_context_centrally(self) -> None:
        source = (_src_root() / "obs" / "events.py").read_text(encoding="utf-8")
        body = source[source.index("def emit_event(") :]
        assert "_RUN_CONTEXT" in body, (
            "emit_event no longer merges the run context — every event would carry only what "
            "its caller remembered to pass, which is the state #1873 was opened to end"
        )
        assert (
            "get_episode_id" in body and "get_feed_id" in body
        ), "emit_event no longer falls back to the correlation context for episode/feed"

    def test_the_context_is_established_in_one_place(self) -> None:
        """Exactly one production entry point stamps run identity.

        Two places doing it is how the four surfaces drifted apart in the first place.
        """
        orchestration = (_src_root() / "workflow" / "orchestration.py").read_text(encoding="utf-8")
        assert "def stamp_run_identity(" in orchestration

        callers = []
        for path in _src_root().rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            # Catch aliased and attribute forms too — ``events.set_run_context(...)`` or an
            # ``as`` import evaded a bare-call regex, which would let a second stamping site
            # appear without failing this test.
            # Calls only — the definition in obs/events.py is not a caller. Aliased and
            # attribute forms count (``events.set_run_context(...)``), because a bare-call
            # regex would let a second stamping site appear without failing this test.
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith(("def ", "async def ", "#")):
                    continue
                if re.search(r"(?<![\w.])set_run_context\s*\(|\.set_run_context\s*\(", line):
                    callers.append(path.name)
                    break
        assert callers == ["orchestration.py"] or callers == [], (
            "set_run_context is called outside the single stamping entry point "
            f"({callers}) — run identity must be established in one place so it cannot be "
            "established two different ways"
        )
