"""The translated shows' English renders run on the episode's own timeline (operator 2026-10-12).

A short listen cut its clip from the English render at the quote's time, but the captured render
carried another run's timings, so "Start at strava.com/podcast" played for a point about drainage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

import build_app_validation_corpus as builder  # noqa: E402

pytestmark = [pytest.mark.unit]

TRANSCRIPTS = (
    ROOT / "tests/fixtures/app-validation-corpus/v3/feeds/p10/run_20260101-000000/transcripts"
)


def _segs(name: str) -> list[dict]:
    segs: list[dict] = builder._segments_of(json.loads((TRANSCRIPTS / name).read_text("utf-8")))
    return segs


def test_every_english_sentence_sits_inside_a_source_segment() -> None:
    source = _segs("p10_e01.es.segments.json")
    for seg in _segs("p10_e01.segments.json"):
        assert any(
            s["start"] <= seg["start"] and seg["end"] <= s["end"] + 0.01 for s in source
        ), seg["text"]


def test_the_point_about_drainage_plays_where_its_quote_is() -> None:
    drainage = next(s for s in _segs("p10_e01.segments.json") if s["text"].startswith("Honestly"))
    assert drainage["start"] == 24.0  # the GI quote's 24-30 s, not the ad's 23-25 s


def test_the_committed_renders_need_no_retiming() -> None:
    assert builder.retime_english_render(TRANSCRIPTS, "p10_e01", apply=False) == 0


def test_an_episode_without_a_translation_has_no_render_to_retime(tmp_path: Path) -> None:
    assert builder.english_render_times(tmp_path, "p01_e01") == {}


@pytest.mark.parametrize(
    "line",
    [
        "We'll be right back after a quick word from our sponsors.",
        "Volvemos enseguida, después de una breve pausa publicitaria.",
        "Torniamo subito, dopo una breve pausa pubblicitaria.",
        "Nous revenons tout de suite, après une courte pause publicitaire.",
        "Wir sind gleich zurück, nach einer kurzen Werbepause.",
        "Voltamos já a seguir, depois de uma breve pausa publicitária.",
    ],
)
def test_the_break_lead_in_is_ad_copy_in_every_language(line: str) -> None:
    assert builder.is_ad_copy(line)


def test_a_point_is_not_ad_copy() -> None:
    assert not builder.is_ad_copy("Honestly, drainage is the design decision with the most impact.")
