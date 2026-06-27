"""Smoke tests for the chunk-2/3/4 eval scoring scripts.

These run --help and the no-gold path so the scaffolding is exercised
in CI even before gold fixtures are populated.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO_ROOT / "scripts" / "eval" / "score"

_SCRIPTS_TO_CHECK = [
    "enrichment_deterministic.py",
    "enrichment_topic_similarity.py",
    "enrichment_nli_contradiction.py",
]


@pytest.mark.parametrize("script_name", _SCRIPTS_TO_CHECK)
def test_eval_script_help_exits_zero(script_name: str) -> None:
    script = _SCRIPTS / script_name
    assert script.is_file(), f"missing eval scaffolding at {script}"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_enrichment_deterministic_no_gold_exits_zero(tmp_path: Path) -> None:
    """Empty gold dir → status=no_gold + exit 0 (scaffolding mode)."""
    script = _SCRIPTS / "enrichment_deterministic.py"
    result = subprocess.run(
        [sys.executable, str(script), "--gold", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "no_gold" in result.stdout
