"""Offline contracts for shared prompt templates (summarization + KG)."""

from __future__ import annotations

from typing import cast

import pytest

from podcast_scraper.prompts.store import get_prompt_source, render_prompt
from tests.unit.podcast_scraper.prompts.prompt_contract_registry import (
    assert_transcript_absent_from_source,
    assert_transcript_placeholder_in_source,
    SHARED_JSON_USER_TEMPLATE_SUBSTRINGS,
    shared_render_kwargs,
    SHARED_TEMPLATES_FORBID_TRANSCRIPT,
    SHARED_TEMPLATES_REQUIRE_TRANSCRIPT,
)


@pytest.mark.parametrize("logical_name", SHARED_TEMPLATES_REQUIRE_TRANSCRIPT)
def test_shared_templates_require_transcript_in_source(logical_name: str) -> None:
    assert_transcript_placeholder_in_source(logical_name)


@pytest.mark.parametrize("logical_name", SHARED_TEMPLATES_REQUIRE_TRANSCRIPT)
def test_shared_templates_render_includes_transcript_marker(logical_name: str) -> None:
    kwargs = shared_render_kwargs(logical_name)
    marker = cast(str, kwargs["transcript"])
    out = render_prompt(logical_name, **kwargs)
    assert marker in out, f"{logical_name} rendered output must include transcript body"


@pytest.mark.parametrize("logical_name", SHARED_TEMPLATES_FORBID_TRANSCRIPT)
def test_shared_system_and_bullets_templates_forbid_transcript_snippet(
    logical_name: str,
) -> None:
    assert_transcript_absent_from_source(logical_name)


@pytest.mark.parametrize(
    "logical_name",
    tuple(SHARED_JSON_USER_TEMPLATE_SUBSTRINGS.keys()),
)
def test_shared_json_user_templates_preserve_strict_json_instruction_copy(
    logical_name: str,
) -> None:
    src = get_prompt_source(logical_name)
    for fragment in SHARED_JSON_USER_TEMPLATE_SUBSTRINGS[logical_name]:
        assert fragment in src, (
            f"{logical_name} must still contain JSON instruction fragment "
            f"{fragment!r} (downstream parsers rely on stable wording)"
        )
