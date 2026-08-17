"""The build refuses to succeed on the defect shapes v3 actually shipped with.

v3 committed two defects that a file-existence check cannot see, and both survived a long time:

  * every one of the 36 episode summaries was the transcript's opening greeting;
  * every duration was the same hardcoded 1800s.

The first was fixed in ``metadata.json`` and left untouched in the GI Insight layer, where it stayed
at 36/36. The second was fixed in ``metadata.json`` and left untouched in BOTH the GI Episode node
and the insight-density sidecar. In each case the build printed a success line, because nothing
asserted anything about content — only about files.

These tests pin the audit that now runs at the end of every build. They are written against the
AUDIT rather than against the committed fixture, so they state the rule instead of re-deriving it
from whatever the corpus happens to contain today.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from build_app_validation_corpus import (  # noqa: E402
    _audit_built_corpus,
    is_greeting_or_filler,
)

pytestmark = pytest.mark.unit

_REAL_SUMMARY = "Position sizing is the only risk control that survives an uncertain edge."
_REAL_INSIGHT = "Correlation between supposedly uncorrelated bets is the thing that ruins you."


# --- the shared rule --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Welcome back to the show, everyone, and thanks for tuning in again this week.",
        "welcome to another episode of the podcast where we get into the weeds",
        '  "Welcome back to the show" — with a leading quote and spaces',
        "You're listening to the show that takes risk seriously.",
        "I'm your host and today we have a fantastic guest lined up for you.",
        "Yeah, exactly. And it ties into what we're covering today.",
    ],
)
def test_openings_and_filler_are_recognised(text: str) -> None:
    assert is_greeting_or_filler(text) is True


@pytest.mark.parametrize(
    "text",
    [
        _REAL_SUMMARY,
        "The welcome mat problem in API design is that defaults become permanent.",
        "A great question to ask is whether the correlation survives a regime change.",
    ],
)
def test_real_content_is_not_mistaken_for_a_greeting(text: str) -> None:
    # "welcome" and "great question" MID-sentence must not trip the filter. A rule that eats real
    # content is a rule someone will disable, which is how the fixture ends up unchecked again.
    assert is_greeting_or_filler(text) is False


# --- the audit --------------------------------------------------------------------------------


def _corpus(
    tmp_path: Path,
    *,
    episodes: int = 24,
    summary: str = _REAL_SUMMARY,
    insight: str = _REAL_INSIGHT,
    duration_of: Callable[[int], float] | None = None,
    gi_ms_of: Callable[[int, float], float] | None = None,
) -> Path:
    """Write a small structurally-real corpus; every knob defaults to a HEALTHY value."""
    root = tmp_path / "v9"
    meta_dir = root / "feeds" / "p01" / "run_20260101-000000" / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for i in range(episodes):
        stem = f"p01_e{i:02d}"
        seconds = duration_of(i) if duration_of else 300.0 + i * 17
        (meta_dir / f"{stem}.metadata.json").write_text(
            json.dumps(
                {
                    "episode": {"title": f"Episode {i}", "duration_seconds": seconds},
                    "summary": {"raw_text": summary},
                }
            ),
            encoding="utf-8",
        )
        (meta_dir / f"{stem}.gi.json").write_text(
            json.dumps(
                {
                    "nodes": [
                        {
                            "type": "Episode",
                            "properties": {
                                "duration_ms": (
                                    gi_ms_of(i, seconds) if gi_ms_of else seconds * 1000
                                )
                            },
                        },
                        {"type": "Insight", "properties": {"text": insight}},
                    ]
                }
            ),
            encoding="utf-8",
        )
    return root


def test_a_healthy_corpus_passes(tmp_path: Path) -> None:
    assert _audit_built_corpus(_corpus(tmp_path)) == []


def test_one_duration_for_every_episode_is_caught(tmp_path: Path) -> None:
    """The v3 shape exactly: 1 distinct duration across the whole corpus."""
    problems = _audit_built_corpus(_corpus(tmp_path, duration_of=lambda _i: 1800.0))
    assert any("distinct value" in p for p in problems), problems


def test_a_duration_fixed_in_only_one_layer_is_caught(tmp_path: Path) -> None:
    """The specific way the v3 fix went wrong — metadata measured, GI still on the old constant.

    This is the check that matters most: the corpus reported itself fixed because the ONE layer
    anybody looked at was fixed.
    """
    problems = _audit_built_corpus(_corpus(tmp_path, gi_ms_of=lambda _i, _s: 1800.0 * 1000))
    assert any("gi Episode duration_ms" in p for p in problems), problems


def test_a_greeting_summary_is_caught(tmp_path: Path) -> None:
    problems = _audit_built_corpus(
        _corpus(tmp_path, summary="Welcome back to the show, everyone. Great to have you here.")
    )
    assert any("summary is a greeting" in p for p in problems), problems


def test_a_summary_that_only_restates_the_title_is_caught(tmp_path: Path) -> None:
    root = _corpus(tmp_path, episodes=1)
    path = next(root.rglob("*.metadata.json"))
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["summary"]["raw_text"] = doc["episode"]["title"]
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert any("restates the episode title" in p for p in _audit_built_corpus(root)), root


def test_a_greeting_INSIGHT_is_caught(tmp_path: Path) -> None:
    """The half the v3 summary fix never touched — 36/36 episodes led with the host's welcome.

    These are indexed and surface everywhere insights do, so a corpus like this cannot exercise an
    insight surface with anything an insight surface is for.
    """
    problems = _audit_built_corpus(
        _corpus(tmp_path, insight="Welcome back to the show — let's get into it.")
    )
    assert any("Insight is a greeting/filler" in p for p in problems), problems


def test_an_empty_corpus_is_a_failure_not_a_pass(tmp_path: Path) -> None:
    """A build that wrote nothing must not audit clean — that is the loudest possible false green."""
    assert _audit_built_corpus(tmp_path / "nothing-here") != []
