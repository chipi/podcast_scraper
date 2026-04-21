"""Insight extraction prompts must embed the episode transcript (offline)."""

from __future__ import annotations

from podcast_scraper.prompts.store import get_prompt_dir
from tests.unit.podcast_scraper.prompts.prompt_contract_registry import (
    INSIGHT_EXTRACTION_V1_LOGICAL_NAMES,
)

_TRANSCRIPT = "{{ transcript }}"


def test_insight_extraction_v1_logical_names_match_disk() -> None:
    root = get_prompt_dir()
    discovered = sorted(
        p.relative_to(root).with_suffix("").as_posix()
        for p in root.glob("**/insight_extraction/*.j2")
    )
    expected = sorted(INSIGHT_EXTRACTION_V1_LOGICAL_NAMES)
    assert discovered == expected, (
        "Update INSIGHT_EXTRACTION_V1_LOGICAL_NAMES when adding/removing "
        f"insight_extraction templates (disk={discovered!r} expected={expected!r})"
    )


def test_insight_extraction_v1_templates_include_transcript_placeholder() -> None:
    root = get_prompt_dir()
    for path in sorted(root.glob("**/insight_extraction/*.j2")):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(root).as_posix()
        assert (
            _TRANSCRIPT in text
        ), f"{rel} must contain {_TRANSCRIPT} so the model receives episode text"
