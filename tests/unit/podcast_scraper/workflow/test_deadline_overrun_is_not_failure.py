"""Blowing a deadline you already met is not a failure (#1657 acceptance item 4).

Two feeds of the acceptance run reported ``ok: true`` with ``episodes_processed: 0`` and wrote
no incident anywhere — two fully-processed episodes reading as a silent no-op. The artifacts on
disk said otherwise:

    Dwarkesh      summary 470.0s  gi 1529.3s  kg  9.1s   -> status failed @ summarization
    Latent Space  summary 153.4s  gi 1208.6s  kg  8.1s   -> status failed @ summarization
    (12 other episodes, longest gi 970.3s)               -> status ok @ metadata_written

Both "failed" episodes had complete, valid output: summary ``schema_status: valid`` with real
bullets, 114 and 54 grounded insights, 26 KG nodes each. The only thing they had in common was
a GI stage past the 1200s ``summarization_timeout``.

The mechanism is in ``utils/timeout.py`` and is not in dispute — its own docstring and
``test_timeout_contract.py`` both state it: ``timeout_context`` OBSERVES, it cannot interrupt.
It raises from ``__exit__``, strictly after the wrapped block has returned normally. So arriving
in ``except SummarizationTimeoutError`` is positive proof that the work COMPLETED; work that
actually raised lands in the generic handler instead. (``with_timeout``, the only other producer
of that exception class, has no callers.)

Recording that as ``failed`` cost the episode: ``_pipeline_return_episode_count`` counts ok
statuses, so 1 saved transcript became ``episodes_processed: 0``, while the feed stayed
``ok: true`` because nothing escaped to feed level.

This is the #1647 shape again — a signal meaning "finished, but notable" routed into the word
reserved for "did not finish". These tests pin the distinction, and pin that the overrun stays
VISIBLE: it is not being swept under the rug, it is being filed correctly.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import pytest

from podcast_scraper.utils import timeout as timeout_mod

pytestmark = [pytest.mark.unit]


def _handler_source() -> str:
    """The body of the ``except SummarizationTimeoutError`` block, and nothing else.

    Delimited on the fully-indented ``except`` clauses rather than the bare words, because the
    handler's own comment quotes ``except Exception`` when explaining why a genuine failure
    lands there instead — a naive split cuts the slice off inside the comment.
    """
    import inspect

    from podcast_scraper.workflow.stages import processing

    src = inspect.getsource(processing.process_processing_jobs_concurrent)
    start = src.index("except SummarizationTimeoutError")
    end = src.index("\n        except Exception as exc:", start)
    return src[start:end]


class TestTheMechanismThisRestsOn:
    """If these ever fail, the reasoning above is void and the handler must be revisited."""

    def test_the_block_completes_before_the_exception(self) -> None:
        """The load-bearing fact: reaching the handler proves the work finished."""
        completed: List[str] = []
        with pytest.raises(timeout_mod.TimeoutError):
            with timeout_mod.timeout_context(1, "op"):
                time.sleep(1.3)
                completed.append("finished")
        assert completed == ["finished"]

    def test_a_raising_block_propagates_its_own_error_not_a_timeout(self) -> None:
        """So a genuine failure is still distinguishable — it never reaches the timeout
        handler, however long it took to fail."""
        with pytest.raises(ValueError):
            with timeout_mod.timeout_context(1, "op"):
                time.sleep(1.2)
                raise ValueError("the work itself failed")

    def test_a_return_inside_the_block_still_triggers_the_deadline_raise(self) -> None:
        """Why the transcription path lost finished transcripts: the value is evaluated, then
        __exit__ raises and discards it."""

        def f() -> str:
            with timeout_mod.timeout_context(1, "op"):
                time.sleep(1.2)
                return "computed"

        with pytest.raises(timeout_mod.TimeoutError):
            f()

    def test_stashing_before_leaving_the_block_preserves_the_result(self) -> None:
        """The shape the fix uses."""
        done: Dict[str, str] = {}

        def f() -> str:
            try:
                with timeout_mod.timeout_context(1, "op"):
                    time.sleep(1.2)
                    done["r"] = "computed"
            except timeout_mod.TimeoutError:
                if "r" not in done:
                    raise
            return done["r"]

        assert f() == "computed"


class TestTheSummarizationHandlerKeepsTheEpisode:
    """The behaviour that turned two good episodes into zeroes."""

    def test_the_status_written_is_ok_not_failed(self) -> None:
        """The episode succeeded, so its status must say so. ``failed`` here is exactly what
        produced ``episodes_processed: 0``."""
        handler = _handler_source()
        assert 'status="ok"' in handler, "a completed episode must not be recorded as failed"
        assert 'status="failed"' not in handler

    def test_the_handler_returns_success(self) -> None:
        handler = _handler_source()
        assert "return True" in handler
        assert "return False" not in handler

    def test_the_overrun_is_still_recorded_as_a_degradation(self) -> None:
        """Not swept away: the episode counts, AND the overrun is filed. Both, or this is
        just a different lie."""
        handler = _handler_source()
        assert '"degraded"' in handler
        assert "deadline_exceeded_but_completed" in handler

    def test_an_incident_row_is_written(self) -> None:
        """The acceptance run's two slowest episodes left NO row in corpus_incidents.jsonl, so
        the batch rollup reported zero incidents for feeds 20+ minutes over budget."""
        handler = _handler_source()
        assert "_record_summarization_overrun_incident(" in handler

    def test_the_overrun_counter_is_separate_from_errors_total(self) -> None:
        """Folding a success into the error count is how the run looked broken."""
        handler = _handler_source()
        assert "summarization_deadline_overruns" in handler
        assert "errors_total" not in handler


class TestTheIncidentItself:
    def test_it_is_written_to_the_configured_log(self, tmp_path: Any) -> None:
        import json

        from podcast_scraper.workflow.stages import processing

        class _Cfg:
            incident_log_path = str(tmp_path / "corpus_incidents.jsonl")
            rss_url = "https://example.com/feed.xml"

        class _Ep:
            idx = 1
            title = "t"
            item = object()

        class _Job:
            episode = _Ep()

        processing._record_summarization_overrun_incident(
            _Cfg(),  # type: ignore[arg-type]
            _Job(),
            str(tmp_path),
            timeout_mod.TimeoutError("overran"),
        )
        rows = [
            json.loads(line)
            for line in open(_Cfg.incident_log_path, encoding="utf-8")
            if line.strip()
        ]
        assert len(rows) == 1
        assert rows[0]["stage"] == "summarization"
        assert rows[0]["exception_type"] == "DeadlineExceededButCompleted"
        assert rows[0]["scope"] == "episode"

    def test_it_is_soft_not_policy(self, tmp_path: Any) -> None:
        """``policy`` means a documented by-design skip (an API audio limit). An overrun is an
        anomaly worth chasing, so it must land in the bucket operators actually look at."""
        import json

        from podcast_scraper.workflow.stages import processing

        class _Cfg:
            incident_log_path = str(tmp_path / "i.jsonl")
            rss_url = "https://example.com/f.xml"

        class _Ep:
            idx = 2
            title = "t"
            item = object()

        class _Job:
            episode = _Ep()

        processing._record_summarization_overrun_incident(
            _Cfg(), _Job(), str(tmp_path), timeout_mod.TimeoutError("x")  # type: ignore[arg-type]
        )
        row = json.loads(open(_Cfg.incident_log_path, encoding="utf-8").readline())
        assert row["category"] == "soft"

    def test_it_never_raises_even_with_an_unwritable_path(self) -> None:
        """Observability must not be able to fail the episode it is describing."""
        from podcast_scraper.workflow.stages import processing

        class _Cfg:
            incident_log_path = "/nonexistent-dir-xyz/nope/i.jsonl"
            rss_url = None

        class _Job:
            episode = None

        processing._record_summarization_overrun_incident(
            _Cfg(),  # type: ignore[arg-type]
            _Job(),
            "/nonexistent-dir-xyz/nope",
            timeout_mod.TimeoutError("x"),
        )


class TestAGenuineFailureIsAlsoNoLongerSilent:
    """The other route to the same complaint.

    Fixing the deadline lie makes the two acceptance episodes count, but it does not by itself
    make a REAL metadata failure visible. That path recorded ``status=failed`` inside the run's
    own metrics and nothing else — no incident row — so a genuinely failed episode still
    produced ``ok: true``, ``episodes_processed: 0``, and a batch rollup reporting zero
    incidents. Same silence, different cause; both had to go.
    """

    def test_the_failure_handler_writes_an_incident(self) -> None:
        import inspect

        from podcast_scraper.workflow.stages import processing

        src = inspect.getsource(processing.process_processing_jobs_concurrent)
        handler = src[src.index("\n        except Exception as exc:") :]
        assert "_record_metadata_failure_incident(" in handler

    def test_it_is_hard_not_soft(self, tmp_path: Any) -> None:
        """Nothing about an unexpected exception is by design, so it must not land in the
        bucket reserved for documented skips or recoverable anomalies."""
        import json

        from podcast_scraper.workflow.stages import processing

        class _Cfg:
            incident_log_path = str(tmp_path / "i.jsonl")
            rss_url = "https://example.com/f.xml"

        class _Ep:
            idx = 3
            title = "t"
            item = object()

        class _Job:
            episode = _Ep()

        processing._record_metadata_failure_incident(
            _Cfg(),  # type: ignore[arg-type]
            _Job(),
            str(tmp_path),
            ValueError("kg builder exploded"),
        )
        row = json.loads(open(_Cfg.incident_log_path, encoding="utf-8").readline())
        assert row["category"] == "hard"
        assert row["stage"] == "metadata"
        assert row["exception_type"] == "ValueError"

    def test_the_two_outcomes_are_distinguishable_in_the_log(self, tmp_path: Any) -> None:
        """An operator grouping the incident log must be able to separate "succeeded, slowly"
        from "did not succeed" — collapsing them would just relocate the original ambiguity."""
        import json

        from podcast_scraper.workflow.stages import processing

        class _Cfg:
            incident_log_path = str(tmp_path / "i.jsonl")
            rss_url = "https://example.com/f.xml"

        class _Ep:
            idx = 4
            title = "t"
            item = object()

        class _Job:
            episode = _Ep()

        cfg: Any = _Cfg()
        processing._record_summarization_overrun_incident(
            cfg, _Job(), str(tmp_path), timeout_mod.TimeoutError("slow")
        )
        processing._record_metadata_failure_incident(
            cfg, _Job(), str(tmp_path), ValueError("broken")
        )
        rows = [
            json.loads(line)
            for line in open(_Cfg.incident_log_path, encoding="utf-8")
            if line.strip()
        ]
        assert [r["category"] for r in rows] == ["soft", "hard"]
        assert rows[0]["exception_type"] != rows[1]["exception_type"]


class TestTheCountFollows:
    """The reported symptom, at the layer that reported it."""

    def test_an_ok_status_is_counted(self) -> None:
        from podcast_scraper.workflow.helpers import _pipeline_return_episode_count

        class _S:
            status = "ok"

        class _M:
            episode_statuses = [_S()]

        assert _pipeline_return_episode_count(1, _M()) == 1  # type: ignore[arg-type]

    def test_a_failed_status_yields_zero_even_with_a_saved_transcript(self) -> None:
        """Documents the amplifier, unchanged by this fix: the count trusts the status
        completely, so a wrong status is worth exactly one whole episode. That is why the
        status had to be fixed at the source rather than compensated for here."""
        from podcast_scraper.workflow.helpers import _pipeline_return_episode_count

        class _S:
            status = "failed"

        class _M:
            episode_statuses = [_S()]

        assert _pipeline_return_episode_count(1, _M()) == 0  # type: ignore[arg-type]


class TestTheTranscriptionPathKeepsFinishedWork:
    def test_the_result_is_stashed_before_leaving_the_with_block(self) -> None:
        import inspect

        from podcast_scraper.workflow import episode_processor

        src = inspect.getsource(episode_processor)
        start = src.index("def _transcribe_one(")
        body = src[start : start + 2200]
        assert 'done["r"]' in body, "the completed transcript is still discarded on overrun"
        assert "return done[" in body

    def test_it_still_raises_when_nothing_was_produced(self) -> None:
        """The fix must not swallow a real failure — no result means the caller's failure
        handling is correct."""
        import inspect

        from podcast_scraper.workflow import episode_processor

        src = inspect.getsource(episode_processor)
        start = src.index("def _transcribe_one(")
        body = src[start : start + 2200]
        assert 'if "r" not in done:' in body
        assert "raise" in body
