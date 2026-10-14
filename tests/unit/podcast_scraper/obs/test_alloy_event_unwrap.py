"""The Alloy stage that ships an emitted event as bare JSON (infra/observability/operator.alloy).

``emit_event(sink="log")`` reaches VictoriaLogs behind the pipeline's text log prefix, which kept
VictoriaLogs from unpacking it: on production, ``event_type:llm_cost`` matched nothing while the
lines were there (2026-10-14). The stage captures the event's JSON object as the whole line. This
test reads the stage's regex from the Alloy config and runs it on a line formatted exactly as the
pipeline formats it, so a change to either side that breaks the unwrap fails here.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pytest

from podcast_scraper.obs.events import emit_event
from podcast_scraper.utils.correlation import CorrelationFormatter

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]
_PIPELINE_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s "
    "[run=%(run_id)s feed=%(feed_id)s profile=%(profile)s trace=%(trace_id)s]: %(message)s"
)


def _stage_regex() -> re.Pattern[str]:
    alloy = (ROOT / "infra" / "observability" / "operator.alloy").read_text()
    block = alloy[alloy.index("stage.match {") :]
    raw = re.search(r'expression = "((?:[^"\\]|\\.)*)"', block)
    assert raw, "the unwrap stage's regex"
    return re.compile(json.loads(f'"{raw.group(1)}"'))  # Alloy strings unescape like JSON


def test_the_pipeline_formatted_event_line_unwraps_to_the_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="podcast_scraper.events"):
        emit_event("llm_cost", stage="gi", estimated_cost_usd=0.0, prompt_tokens=12)
    record = caplog.records[-1]
    line = CorrelationFormatter(_PIPELINE_FORMAT).format(record)
    assert not line.startswith("{"), "the pipeline line carries the text prefix"

    m = _stage_regex().search(line)
    assert m, "the Alloy stage captures nothing from an event line"
    event = json.loads(m.group("event_json"))
    assert (event["event_type"], event["stage"], event["prompt_tokens"]) == ("llm_cost", "gi", 12)


def test_a_line_that_is_already_bare_json_passes_whole() -> None:
    bare = json.dumps({"ts": "2026-10-14T00:00:00+00:00", "event_type": "listen", "n": 1})
    m = _stage_regex().search(bare)
    assert m and m.group("event_json") == bare


def test_the_stage_only_runs_on_event_lines() -> None:
    alloy = (ROOT / "infra" / "observability" / "operator.alloy").read_text()
    block = alloy[alloy.index("stage.match {") :]
    found = re.search(r'selector = ("(?:[^"\\]|\\.)*")', block)
    assert found
    selector = json.loads(found.group(1))
    assert selector == '{app="podcast"} |= "\\"event_type\\": "'
