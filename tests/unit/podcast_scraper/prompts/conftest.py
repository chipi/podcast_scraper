"""Offline prompt tests: markers for ``validate-files`` + prompt template cache reset."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from podcast_scraper.prompts.store import clear_cache

_PROMPTS_UNIT_DIR = Path(__file__).resolve().parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Register ``unit`` + ``module_prompts`` so ``-m`` discovery (validate-files) works.

    Module-level ``pytestmark`` in ``conftest.py`` does not apply these marks to
    collected tests for marker expressions in all pytest versions.
    """
    for item in items:
        raw_path = getattr(item, "path", None)
        if raw_path is not None:
            fspath = Path(raw_path)
        else:
            try:
                fspath = Path(item.fspath)  # type: ignore[attr-defined]
            except Exception:
                continue
        if _PROMPTS_UNIT_DIR in fspath.parents or fspath.parent == _PROMPTS_UNIT_DIR:
            item.add_marker(pytest.mark.unit)
            item.add_marker(pytest.mark.module_prompts)


@pytest.fixture(autouse=True)
def _clear_prompt_template_cache() -> Iterator[None]:
    clear_cache()
    yield
    clear_cache()
