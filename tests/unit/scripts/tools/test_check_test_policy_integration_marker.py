"""Rule I2 of scripts/tools/check_test_policy.py: an integration test file is selected by
``-m integration``. CI runs only that selection, so an unmarked file runs locally and never in CI
(#2180: 239 tests in 14 files on 2026-11-04). The rule reads the marks pytest reads, not the text:
a docstring that mentions the marker selects nothing."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "tools" / "check_test_policy.py"
_spec = importlib.util.spec_from_file_location("check_test_policy_under_test", _SCRIPT)
assert _spec and _spec.loader
ctp: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ctp)


def _check(tmp_path: Path, body: str) -> list:
    f = tmp_path / "test_x.py"
    f.write_text("import pytest\n\n" + body, encoding="utf-8")
    found: list = ctp.check_integration_marker([f])
    return found


@pytest.mark.parametrize(
    "body",
    [
        "pytestmark = pytest.mark.integration\n\ndef test_a():\n    pass\n",
        "pytestmark = [pytest.mark.integration, pytest.mark.slow]\n\ndef test_a():\n    pass\n",
        "@pytest.mark.integration\ndef test_a():\n    pass\n",
        "@pytest.mark.integration\nclass TestA:\n    def test_a(self):\n        pass\n",
        "class TestA:\n    @pytest.mark.integration\n    def test_a(self):\n        pass\n",
    ],
)
def test_a_file_whose_tests_all_carry_the_marker_passes(tmp_path: Path, body: str) -> None:
    assert _check(tmp_path, body) == []


@pytest.mark.parametrize(
    "body",
    [
        "def test_a():\n    pass\n",
        '"""Uses pytest.mark.integration."""\n\ndef test_a():\n    pass\n',
        "# pytest.mark.integration\ndef test_a():\n    pass\n",
        "@pytest.mark.integration\ndef test_a():\n    pass\n\ndef test_b():\n    pass\n",
        "@pytest.mark.integration\nclass TestA:\n    def test_a(self):\n        pass\n\n"
        "class TestB:\n    def test_b(self):\n        pass\n",
    ],
)
def test_a_test_the_marker_does_not_select_fails(tmp_path: Path, body: str) -> None:
    [v] = _check(tmp_path, body)
    assert v.rule == "I2-integration-marker"


def test_a_file_with_no_tests_and_conftest_are_skipped(tmp_path: Path) -> None:
    assert _check(tmp_path, "HELPER = 1\n") == []
    conf = tmp_path / "conftest.py"
    conf.write_text("def test_like():\n    pass\n", encoding="utf-8")
    assert ctp.check_integration_marker([conf]) == []


def test_every_integration_file_in_the_repo_is_selected() -> None:
    root = Path(__file__).resolve().parents[4] / "tests" / "integration"
    files = sorted(p for p in root.rglob("test_*.py"))
    assert files and ctp.check_integration_marker(files) == []


@pytest.mark.parametrize(
    "body",
    [
        "class TestA:\n    pytestmark = pytest.mark.integration\n\n"
        "    def test_a(self):\n        pass\n",
        "from pytest import mark\n\n@mark.integration\ndef test_a():\n    pass\n",
        "from pytest import mark\n\npytestmark = mark.integration\n\ndef test_a():\n    pass\n",
        "if True:\n    pytestmark = pytest.mark.integration\n\ndef test_a():\n    pass\n",
    ],
)
def test_other_ways_pytest_applies_the_marker_pass(tmp_path: Path, body: str) -> None:
    """Review 2, 2026-11-04: valid pytest the rule did not read."""
    assert _check(tmp_path, body) == []


def test_a_test_the_walk_cannot_see_is_not_a_pass(tmp_path: Path) -> None:
    """A ``def test_`` only nested somewhere the AST walk does not visit must not pass unmarked."""
    body = "class Helper:\n    def test_a(self):\n        pass\n"
    [v] = _check(tmp_path, body)
    assert v.rule == "I2-integration-marker"
