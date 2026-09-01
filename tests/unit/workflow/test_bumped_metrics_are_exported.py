"""Every counter the code bumps must survive the trip to ``metrics.json``.

``Metrics`` is a dataclass, but ``finish`` does NOT use ``asdict`` — it builds an explicit
dict literal, field by field. So a counter has to clear TWO independent bars to be observable:

1. be a declared dataclass field, and
2. be named in that dict literal.

Miss either and the bump still "works": ``setattr`` on a non-slotted dataclass happily creates
the attribute, the call site reads as instrumented, and the number is discarded on the way out.
Nothing fails. Nothing warns.

That is not hypothetical — it is how ``gi_insight_overgeneration_events`` and its two siblings
shipped on 2026-08-31, under a source comment asserting "these counters make the ratio
aggregatable". They made nothing aggregatable; they were never written anywhere. The counters
were added specifically because the DGX runs at $0, so waste has no cost alarm and has to be
counted or it is invisible — and the counting was itself invisible.

This test scans the source for bump call sites and checks both bars, so the next counter cannot
repeat it. It is deliberately source-scanning rather than a list of known names: a hand-kept
list has the same failure mode as the thing it is guarding.
"""

from __future__ import annotations

import ast
import re
from dataclasses import fields
from pathlib import Path

import pytest

from podcast_scraper.workflow.metrics import Metrics

_SRC = Path(__file__).resolve().parents[3] / "src" / "podcast_scraper"

# Functions whose FIRST arg is a metrics sink and second is the counter name.
_BUMP_FUNCS = {"_bump_metric", "update_metric_safely", "_bump"}

# Names bumped on objects that are not the run-level ``Metrics`` dataclass. Each entry needs a
# reason; this is the pressure valve, and an unexplained entry is how the guard gets hollowed out.
_NOT_RUN_METRICS: dict[str, str] = {}


# Objects that ARE the run-level Metrics dataclass, by local variable name. Direct attribute
# writes (`pipeline_metrics.foo += 1`) are how most of the package bumps counters — the helper
# functions are the minority. A first version of this scan only saw the helpers and therefore
# passed while EIGHT counters were dead, including `gi_empty_extraction_count` (episodes that
# produced zero insights) and `edges_enriched`, whose source comment claims it is "Surfaced,
# not fire-and-forget".
_METRICS_VARS = {"pipeline_metrics", "metrics", "_metrics", "run_metrics", "self.metrics"}

# ``workflow/metrics.py`` is where the dataclasses are DEFINED and where per-episode records are
# populated. Attribute writes there target ``EpisodeMetrics`` / ``EpisodeStatus`` — different
# dataclasses with their own serialization (``asdict(status)`` under "episode_statuses") — so
# scanning it produces only false positives (audio_sec, transcribe_sec, prompt_tokens, ...).
# Excluded by path, not by name, so a NEW run-level counter added elsewhere is still caught.
_SKIP_FILES = {"workflow/metrics.py"}


def _bumped_metric_names() -> dict[str, set[str]]:
    """{counter_name: {files that bump it}} across the whole package.

    Collects three shapes, because a counter is equally dead in all of them:
      * ``_bump_metric(m, "name")`` / ``update_metric_safely(m, "name", n)``
      * ``m.name += 1`` (AugAssign on a metrics-looking base)
      * ``m.name = ...`` (plain Assign on a metrics-looking base)
    """
    found: dict[str, set[str]] = {}

    def _base_is_metrics(value: ast.expr) -> bool:
        if isinstance(value, ast.Name):
            return value.id in _METRICS_VARS
        if isinstance(value, ast.Attribute):
            return value.attr in _METRICS_VARS
        return False

    for path in _SRC.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - the package must parse
            continue
        rel = str(path.relative_to(_SRC))
        if rel in _SKIP_FILES:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and len(node.args) >= 2:
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
                if name in _BUMP_FUNCS:
                    arg = node.args[1]
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        found.setdefault(arg.value, set()).add(rel)
            elif isinstance(node, ast.AugAssign):
                t = node.target
                if isinstance(t, ast.Attribute) and _base_is_metrics(t.value):
                    found.setdefault(t.attr, set()).add(rel)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and _base_is_metrics(target.value):
                        found.setdefault(target.attr, set()).add(rel)
    return found


@pytest.fixture(scope="module")
def bumped() -> dict[str, set[str]]:
    names = _bumped_metric_names()
    assert names, "found no bump call sites at all — the AST scan is broken, not the code"
    return names


def test_the_scan_finds_the_known_call_sites(bumped):
    """Guards the guard: a scan that silently matches nothing would pass everything below."""
    assert "gi_insight_overgeneration_events" in bumped
    assert "gi_insight_salvage_events" in bumped


def test_every_bumped_counter_is_a_declared_field(bumped):
    declared = {f.name for f in fields(Metrics)}
    missing = {
        n: sorted(f) for n, f in bumped.items() if n not in declared and n not in _NOT_RUN_METRICS
    }
    assert not missing, (
        "these counters are bumped but are not declared fields on Metrics, so they exist only "
        f"as stray attributes: {missing}"
    )


def test_every_bumped_counter_reaches_finish(bumped):
    """The second bar — and the one that actually dropped the 2026-08-31 counters."""
    summary = Metrics().finish()
    missing = {
        n: sorted(f) for n, f in bumped.items() if n not in summary and n not in _NOT_RUN_METRICS
    }
    assert not missing, (
        "these counters are bumped and declared but never named in finish's dict literal, "
        f"so they are dropped before metrics.json: {missing}"
    )


def test_finish_is_still_a_literal_not_asdict():
    """If this ever flips to ``asdict``, the export bar disappears and so should this test.

    Written as an assertion rather than a comment because the two tests above would keep
    passing under ``asdict`` while silently guarding nothing.
    """
    src = (_SRC / "workflow" / "metrics.py").read_text(encoding="utf-8")
    body = src[src.index("def finish") :]
    body = body[: body.index("\n    def ", 10)]
    assert not re.search(r"\basdict\(self\)", body), (
        "finish now uses asdict(self) — declaration alone would be sufficient and "
        "test_every_bumped_counter_reaches_finish is redundant. Delete it deliberately."
    )


def test_the_gi_waste_counters_round_trip():
    """End to end for the specific counters that were dead, with real values."""
    m = Metrics()
    m.gi_insight_overgeneration_events = 3
    m.gi_insight_overgenerated_total = 47
    m.gi_insight_overgeneration_severe_events = 1
    m.gi_insight_salvage_events = 12
    m.gi_insight_salvage_lines_recovered = 455
    m.gi_insight_salvage_failed_events = 2
    m.selected_episodes_produced_no_jobs = 1

    s = m.finish()
    assert s["gi_insight_overgeneration_events"] == 3
    assert s["gi_insight_overgenerated_total"] == 47
    assert s["gi_insight_overgeneration_severe_events"] == 1
    assert s["gi_insight_salvage_events"] == 12
    assert s["gi_insight_salvage_lines_recovered"] == 455
    assert s["gi_insight_salvage_failed_events"] == 2
    assert s["selected_episodes_produced_no_jobs"] == 1


def test_defaults_are_zero_not_absent():
    """A run with no waste must report 0, not omit the key.

    An absent key and a zero read the same in a log but not in a query: `sum(...)` over a
    missing field silently returns nothing, which looks like "no waste" rather than "no data".
    """
    s = Metrics().finish()
    for n in (
        "gi_insight_overgeneration_events",
        "gi_insight_salvage_events",
        "gi_insight_salvage_failed_events",
        "selected_episodes_produced_no_jobs",
    ):
        assert s[n] == 0
