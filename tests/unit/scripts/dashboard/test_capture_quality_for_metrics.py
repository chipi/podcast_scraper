"""Tests for capture_quality_for_metrics parsers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_DASH = Path(__file__).resolve().parents[4] / "scripts" / "dashboard"


def _load():
    spec = importlib.util.spec_from_file_location(
        "capture_quality_for_metrics",
        _DASH / "capture_quality_for_metrics.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_parse_interrogate_from_result_line() -> None:
    mod = _load()
    text = "--------------- RESULT: FAILED (minimum: 80.0%, actual: 62.3%) ---------------"
    assert mod.parse_interrogate_coverage_percent(text) == 62.3


def test_parse_interrogate_from_total_row() -> None:
    mod = _load()
    text = "| TOTAL                                        |   785 |   10 |   775 |  98.7% |"
    assert mod.parse_interrogate_coverage_percent(text) == 98.7


def test_vulture_stdout_to_json_list_counts_lines() -> None:
    mod = _load()
    text = """src/podcast_scraper/foo.py:10: unused import 'os' (90% confidence)
some noise
src/podcast_scraper/bar.py:2: unused function 'x' (100% confidence)
"""
    items = mod.vulture_stdout_to_json_list(text)
    assert len(items) == 2
