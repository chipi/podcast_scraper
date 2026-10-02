"""Episode spans: a failure keeps its own exception, and every stage of an episode is traced.

From the 2026-10-02 observability review of the nightly:
  * ``episode.process`` lasted at most 11s, because it wraps only the download; transcription and
    metadata (up to 16 and 45 minutes) ran outside any episode span.
  * ERROR lines were logged and no span was an error: stages report failure by returning it.
  * Found while fixing those: the span helper turned the wrapped block's exception into
    ``RuntimeError: generator didn't stop after throw()``.
"""

from __future__ import annotations

import importlib
import time
from typing import Any, List

import pytest

pytestmark = pytest.mark.unit


class _Boom(Exception):
    pass


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://backend:10428/insert/opentelemetry/v1/traces"
    )
    import podcast_scraper.utils.otel_init as m

    importlib.reload(m)
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        trace.set_tracer_provider(TracerProvider())
    exporter = InMemorySpanExporter()
    trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))  # type: ignore
    yield m, exporter
    exporter.clear()


def _finished(exporter: Any, name: str) -> List[Any]:
    return [s for s in exporter.get_finished_spans() if s.name == name]


def test_a_failure_inside_the_episode_span_reaches_the_caller_unchanged(spans: Any) -> None:
    m, exporter = spans
    with pytest.raises(_Boom, match="the real failure"):
        with m.episode_span(run_id="r1", episode_id="e1"):
            raise _Boom("the real failure")
    (span,) = _finished(exporter, "episode.process")
    assert span.status.status_code.name == "ERROR"


def test_a_failure_inside_the_enrichment_span_reaches_the_caller_unchanged(spans: Any) -> None:
    m, exporter = spans
    with pytest.raises(_Boom):
        with m.enrichment_span(run_id="r1", enricher_id="org_web"):
            raise _Boom("enricher failed")
    (span,) = _finished(exporter, "enrichment.enricher")
    assert span.status.status_code.name == "ERROR"


def test_a_successful_block_is_not_an_error(spans: Any) -> None:
    m, exporter = spans
    with m.episode_span(run_id="r1", episode_id="e1", name="episode.metadata"):
        pass
    (span,) = _finished(exporter, "episode.metadata")
    assert span.status.status_code.name != "ERROR"
    assert span.attributes["episode_id"] == "e1"


def test_a_handled_failure_can_mark_its_span(spans: Any) -> None:
    m, exporter = spans
    with m.episode_span(run_id="r1", episode_id="e1", name="episode.transcribe") as span:
        m.mark_span_failed(span, "transcription returned success=False")
    (finished,) = _finished(exporter, "episode.transcribe")
    assert finished.status.status_code.name == "ERROR"


def test_marking_no_span_is_a_no_op() -> None:
    import podcast_scraper.utils.otel_init as m

    m.mark_span_failed(None, "nothing to mark")


def test_transcription_returning_failure_is_an_error_span(
    spans: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _m, exporter = spans
    from podcast_scraper.workflow.stages import transcription

    monkeypatch.setattr(
        transcription, "factory_transcribe_media_to_text", lambda *a, **k: (False, None, 0)
    )
    assert transcription.transcribe_media_to_text(object(), object()) == (False, None, 0)
    (span,) = _finished(exporter, "episode.transcribe")
    assert span.status.status_code.name == "ERROR"


def test_transcription_that_succeeds_is_traced_and_not_an_error(
    spans: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _m, exporter = spans
    from podcast_scraper.workflow.stages import transcription

    monkeypatch.setattr(
        transcription, "factory_transcribe_media_to_text", lambda *a, **k: (True, "t.txt", 10)
    )
    transcription.transcribe_media_to_text(object(), object())
    (span,) = _finished(exporter, "episode.transcribe")
    assert span.status.status_code.name != "ERROR"


def test_an_overrun_that_completed_does_not_mark_the_metadata_span(spans: Any) -> None:
    # The order processing.py uses: deadline observer OUTSIDE, span INSIDE. The observer raises
    # its TimeoutError after the block returns, so the finished work is not reported as failed.
    m, exporter = spans
    from podcast_scraper.utils.timeout import timeout_context, TimeoutError as Overrun

    with pytest.raises(Overrun):
        with timeout_context(1, "metadata"), m.episode_span(name="episode.metadata"):
            time.sleep(1.1)
    (span,) = _finished(exporter, "episode.metadata")
    assert span.status.status_code.name != "ERROR"
