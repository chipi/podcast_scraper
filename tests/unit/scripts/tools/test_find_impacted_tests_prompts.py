"""``find_impacted_tests`` maps prompt template paths to ``module_prompts``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _load_find_impacted_tests_module():
    repo_root = Path(__file__).resolve().parents[4]
    path = repo_root / "scripts" / "tools" / "find_impacted_tests.py"
    spec = importlib.util.spec_from_file_location("_fit", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prompts_src_path_maps_to_module_prompts() -> None:
    mod = _load_find_impacted_tests_module()
    marker = mod.extract_module_marker(
        "src/podcast_scraper/prompts/gemini/cleaning/v1.j2",
    )
    assert marker == "module_prompts"


def test_prompts_unit_test_path_maps_to_module_prompts() -> None:
    mod = _load_find_impacted_tests_module()
    marker = mod.extract_module_marker(
        "tests/unit/podcast_scraper/prompts/test_cleaning_prompt_templates.py",
    )
    assert marker == "module_prompts"


def test_build_marker_expression_includes_module_prompts_for_unit() -> None:
    mod = _load_find_impacted_tests_module()
    expr = mod.build_marker_expression({"module_prompts"}, "unit", False)
    assert "module_prompts" in expr
    assert "unit" in expr
