"""Stage duration, failure status, failed-episode event and stage spans (pipeline observability).

Measured on prod: ``pipeline_stage`` events for turns, diarization and naming carried no
``duration_s``, a failed stage was visible only as a missing event, and the stages that ran on
worker threads were absent from the episode's trace.
"""

from __future__ import annotations

import importlib
import json
import os
import threading
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest import mock

import pytest

from podcast_scraper.utils import correlation
from podcast_scraper.workflow import episode_processor, processing_manifest as pm
from podcast_scraper.workflow.metrics import Metrics

pytestmark = pytest.mark.unit

REL = "transcripts/0006 - X.txt"
DIAR_RESULT: Dict[str, Any] = {
    "diarization_num_speakers": 2,
    "diarization_speech_seconds": 100.0,
    "speaker_diagnostics": {"summary": {"num_speakers": 2, "named": 1}, "voices": []},
}


@pytest.fixture(autouse=True)
def _no_run_id():
    saved = correlation.get_run_id()
    correlation.set_run_id(None)
    yield
    correlation.set_run_id(saved)


@pytest.fixture
def out_dir(tmp_path):
    (tmp_path / "transcripts").mkdir()
    (tmp_path / REL).write_text("")
    return str(tmp_path)


def _cfg() -> Any:
    return SimpleNamespace(
        transcription_provider="whisper",
        diarization_provider="pyannote",
        rss_url="https://example.com/feed.xml",
        run_id="run-1",
        transcription_speech_coverage_min=0.85,
    )


def _job() -> Any:
    from podcast_scraper.models.entities import TranscriptionJob

    return TranscriptionJob(idx=6, ep_title="X", ep_title_safe="X", temp_media="", episode=None)


def _stage_events(emit: Any) -> Dict[str, Dict[str, Any]]:
    return {
        c.kwargs["stage"]: c.kwargs for c in emit.call_args_list if c.args[0] == "pipeline_stage"
    }


def test_diarization_and_naming_events_carry_duration(out_dir):
    with mock.patch("podcast_scraper.obs.events.emit_event") as emit:
        episode_processor._write_processing_manifest(
            dict(DIAR_RESULT),
            _cfg(),
            _job(),
            REL,
            out_dir,
            diarization_elapsed=40.0,
            naming_elapsed=2.5,
        )
    events = _stage_events(emit)
    assert events["diarization"]["duration_s"] == 40.0
    assert events["naming"]["duration_s"] == 2.5
    assert events["diarization"]["status"] == "ok"


def test_one_diarization_call_splits_into_diarize_and_naming_time():
    assert episode_processor._split_diarize_naming(42.5, {"diarize_s": 40.0}) == (40.0, 2.5)
    # The call failed before diarizing: no naming time, and diarization keeps the whole call.
    assert episode_processor._split_diarize_naming(3.0, {}) == (3.0, None)


def test_relabel_naming_duration_is_its_own(out_dir):
    only_naming = {"speaker_diagnostics": DIAR_RESULT["speaker_diagnostics"]}
    with mock.patch("podcast_scraper.obs.events.emit_event") as emit:
        episode_processor._write_processing_manifest(
            only_naming, _cfg(), _job(), REL, out_dir, naming_elapsed=3.25
        )
    events = _stage_events(emit)
    assert events["naming"]["duration_s"] == 3.25
    assert "diarization" not in events


def test_failed_diarization_is_recorded_with_status_and_error_class(out_dir):
    with mock.patch("podcast_scraper.obs.events.emit_event") as emit:
        episode_processor._write_processing_manifest(
            {"text": "x"},
            _cfg(),
            _job(),
            REL,
            out_dir,
            diarization_elapsed=7.0,
            diarization_error="OSError",
        )
    event = _stage_events(emit)["diarization"]
    assert event["status"] == "failed"
    assert event["error_class"] == "OSError"
    assert event["ran"] is False
    assert event["duration_s"] == 7.0
    block = json.load(open(pm.manifest_path(out_dir, REL)))["stages"]["diarization"]
    assert block["status"] == "failed" and block["error_class"] == "OSError"


def test_stage_event_status_ok_and_skipped(out_dir):
    with mock.patch("podcast_scraper.obs.events.emit_event") as emit:
        pm.update_stage(out_dir, REL, "asr", pm.stage_block(ran=True), episode_id="e")
        pm.update_stage(out_dir, REL, "translation", pm.stage_block(ran=False), episode_id="e")
    events = _stage_events(emit)
    assert events["asr"]["status"] == "ok" and events["asr"]["error_class"] is None
    assert events["translation"]["status"] == "skipped"


def test_episode_failed_event_is_emitted_once():
    metrics = Metrics()
    metrics.episode_statuses.append(
        SimpleNamespace(episode_id="ep1", status="ok", stage="queued", error_type=None)
    )
    with mock.patch("podcast_scraper.obs.events.emit_event") as emit:
        metrics.update_episode_status("ep1", status="failed", stage="metadata", error_type="Boom")
        metrics.update_episode_status("ep1", status="failed", stage="metadata", error_type="Boom")
        metrics.update_episode_status("ep2", stage="transcribed")
    assert emit.call_count == 1
    assert emit.call_args.args[0] == "pipeline_episode.failed"
    assert emit.call_args.kwargs == {
        "episode_id": "ep1",
        "stage": "metadata",
        "error_class": "Boom",
    }


def test_episode_failed_emit_error_does_not_break_status_update():
    metrics = Metrics()
    with mock.patch("podcast_scraper.obs.events.emit_event", side_effect=RuntimeError("x")):
        metrics.update_episode_status("ep9", status="failed", stage="metadata")
    assert metrics.episode_statuses[-1].status == "failed"


@pytest.fixture
def spans(monkeypatch):
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


def _by_name(exporter: Any) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {}
    for s in exporter.get_finished_spans():
        out.setdefault(s.name, []).append(s)
    return out


def test_stage_spans_are_children_of_the_episode_span_across_threads(spans, out_dir):
    m, exporter = spans
    with m.episode_span(run_id="r", episode_id="ep-span", name="episode.process"):
        pass

    def worker() -> None:
        # No current span on this thread: the stage must find the episode's root on its own.
        with m.episode_span(run_id="r", episode_id="ep-span", name="episode.metadata"):
            for stage in ("summary", "gi", "kg"):
                pm.update_stage(
                    out_dir,
                    REL,
                    stage,
                    pm.stage_block(ran=True, duration_s=2.0),
                    episode_id="ep-span",
                )

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    # A stage written on a bare thread (no episode.metadata span at all) is still in the trace.
    t2 = threading.Thread(
        target=lambda: pm.update_stage(
            out_dir, REL, "turns", pm.stage_block(ran=True, duration_s=1.0), episode_id="ep-span"
        )
    )
    t2.start()
    t2.join()

    named = _by_name(exporter)
    (root,) = named["episode.process"]
    (meta,) = named["episode.metadata"]
    assert meta.parent.span_id == root.context.span_id
    for stage in ("summary", "gi", "kg"):
        (span,) = named[f"stage.{stage}"]
        assert span.context.trace_id == root.context.trace_id
        assert span.parent.span_id == meta.context.span_id
        assert span.attributes["episode_id"] == "ep-span"
        assert span.attributes["duration_s"] == 2.0
        assert span.end_time - span.start_time >= 1.9e9
    (turns,) = named["stage.turns"]
    assert turns.parent.span_id == root.context.span_id


def test_failed_stage_span_is_an_error_span(spans, out_dir):
    m, exporter = spans
    with m.episode_span(run_id="r", episode_id="ep-fail"):
        pm.update_stage(
            out_dir,
            REL,
            "diarization",
            pm.stage_block(ran=False, error_class="OSError"),
            episode_id="ep-fail",
        )
    (span,) = _by_name(exporter)["stage.diarization"]
    assert span.status.status_code.name == "ERROR"
    assert span.attributes["error_class"] == "OSError"


def test_stage_event_carries_the_stage_spans_trace_id(spans, out_dir):
    m, exporter = spans
    seen: List[Any] = []
    real = pm._emit_stage_event_inner

    def spy(*args: Any, **kwargs: Any) -> None:
        from opentelemetry import trace

        seen.append(trace.get_current_span().get_span_context().trace_id)
        real(*args, **kwargs)

    with mock.patch.object(pm, "_emit_stage_event_inner", spy):
        with m.episode_span(run_id="r", episode_id="ep-ev") as root:
            pm.update_stage(out_dir, REL, "asr", pm.stage_block(ran=True), episode_id="ep-ev")
    assert seen == [root.get_span_context().trace_id]


def test_stage_span_is_a_noop_without_tracing(monkeypatch):
    monkeypatch.delenv("OTEL_TRACES_EXPORTER", raising=False)
    import podcast_scraper.utils.otel_init as m

    importlib.reload(m)
    with m.stage_span("asr", episode_id="e", duration_s=1.0) as span:
        assert span is None
    assert os.environ.get("OTEL_TRACES_EXPORTER") is None


def test_a_failed_diarization_block_carries_no_method_version(out_dir):
    """A reprocess selects on method_version; a failed run must not read as done."""
    with mock.patch("podcast_scraper.obs.events.emit_event"):
        episode_processor._write_processing_manifest(
            {"text": "x"},
            _cfg(),
            _job(),
            REL,
            out_dir,
            diarization_elapsed=7.0,
            diarization_error="OSError",
        )
    block = json.load(open(pm.manifest_path(out_dir, REL)))["stages"]["diarization"]
    assert block.get("method_version") is None
