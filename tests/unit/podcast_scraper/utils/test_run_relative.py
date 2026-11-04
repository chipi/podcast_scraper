"""utils.run_relative: an absolute path stored at ingest is read inside this corpus's run, and the
copies of the rule agree with it."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.upgrade.migrations.m0025_one_person_one_entry import _run_relative as m25_rule
from podcast_scraper.utils.run_relative import run_relative

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[4]
RUN = Path("/corpus/feeds/f/run_x")
CASES = [
    ("transcripts/ep.txt", "transcripts/ep.txt"),
    ("/app/output/feeds/f/run_x/transcripts/ep.txt", "transcripts/ep.txt"),
    # The LAST occurrence: a parent directory that shares the run's name is not the run.
    ("/srv/run_x/feeds/f/run_x/transcripts/ep.txt", "transcripts/ep.txt"),
    ("/app/output/feeds/f/run_other/transcripts/ep.txt", ""),
    ("/app/output/feeds/f/run_x", ""),
    ("", ""),
    (None, ""),
]


def _script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "measure" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(("value", "want"), CASES)
def test_the_rule(value: Any, want: str) -> None:
    assert run_relative(value, RUN) == want


@pytest.mark.parametrize(("value", "want"), CASES)
def test_m0025_keeps_the_same_rule(value: Any, want: str) -> None:
    assert m25_rule(value, RUN) == want


def test_the_measure_scripts_use_the_shared_rule() -> None:
    assert _script("naming_gold_cases").run_relative is run_relative
    assert _script("naming_replay_compare")._run_relative is run_relative
