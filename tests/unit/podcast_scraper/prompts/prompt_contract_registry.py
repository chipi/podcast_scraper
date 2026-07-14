"""Central rules for offline Jinja prompt contracts (unit tests only).

Prompt templates: keep required substrings and path exclusions in one place
so shared vs per-provider tests stay aligned.
"""

from __future__ import annotations

from typing import Final

# Logical names passed to ``get_prompt_source`` / ``render_prompt`` (no ``.j2``).
SHARED_TEMPLATES_REQUIRE_TRANSCRIPT: Final[tuple[str, ...]] = (
    "shared/summarization/bullets_json_v1",
    "shared/summarization/bundled_clean_summary_user_v1",
    "shared/kg_graph_extraction/v1",
    "shared/kg_graph_extraction/v2",
    "shared/kg_graph_extraction/v5",
)

# Instruction-only or bullet-input prompts: must not silently gain ``transcript``.
SHARED_TEMPLATES_FORBID_TRANSCRIPT: Final[tuple[str, ...]] = (
    "shared/summarization/bundled_clean_summary_system_v1",
    "shared/summarization/system_bullets_v1",
)

_TRANSCRIPT_SNIP: Final[str] = "{{ transcript }}"

# Under ``prompts/**/summarization/*.j2`` but not ``shared/`` (covered above).
SUMMARIZATION_TRANSCRIPT_EXCLUDE_FILENAMES: Final[frozenset[str]] = frozenset(
    {
        "system_v1.j2",
        "bundled_clean_summary_system_v1.j2",
        "system_bullets_v1.j2",
        # R1-Distill anti-reasoning system prompt — system messages set the
        # role + output contract; the matching long_no_thinking_v1.j2 carries
        # the transcript. See #961.
        "system_no_thinking_v1.j2",
    }
)

# Every insight-extraction template on disk (no ``shared/`` insight template).
#
# ``v2`` is the shipped prompt, and every profile now PINS it by name rather than inheriting a
# default. ``v3`` is the speech-act variant that LOST its A/B (route kappa 0.57 against v2's 0.67);
# it stays on disk as the record of a measured experiment and is selected by nothing.
#
# v3 and ``system_v1`` exist for EVERY provider now, not just ollama. Prompt asymmetry was the
# quietest bias in the codebase: a model can only be compared against another if the harness around
# it is the same, and v3 was reachable by ollama alone while the others were pinned to v2 in code.
# Whichever prompt wins the next A/B, every provider must be able to run it.
#
# ``system_v1`` is the one-line output-shape instruction ("one per line, no numbering") that used to
# be a string literal inside all seven providers. A prompt in a file is versioned, diffable and
# reviewable; a prompt in code is a bug with nowhere to be seen (#1179).
INSIGHT_EXTRACTION_V1_LOGICAL_NAMES: Final[tuple[str, ...]] = (
    "anthropic/insight_extraction/system_v1",
    "anthropic/insight_extraction/v1",
    "anthropic/insight_extraction/v2",
    "anthropic/insight_extraction/v3",
    "deepseek/insight_extraction/system_v1",
    "deepseek/insight_extraction/v1",
    "deepseek/insight_extraction/v2",
    "deepseek/insight_extraction/v3",
    "gemini/insight_extraction/system_v1",
    "gemini/insight_extraction/v1",
    "gemini/insight_extraction/v2",
    "gemini/insight_extraction/v3",
    "grok/insight_extraction/system_v1",
    "grok/insight_extraction/v1",
    "grok/insight_extraction/v2",
    "grok/insight_extraction/v3",
    "mistral/insight_extraction/system_v1",
    "mistral/insight_extraction/v1",
    "mistral/insight_extraction/v2",
    "mistral/insight_extraction/v3",
    "ollama/insight_extraction/system_v1",
    "ollama/insight_extraction/v1",
    "ollama/insight_extraction/v2",
    "ollama/insight_extraction/v3",
    "openai/insight_extraction/system_v1",
    "openai/insight_extraction/v1",
    "openai/insight_extraction/v2",
    "openai/insight_extraction/v3",
)

# Parser-facing JSON instructions must not drift silently.
SHARED_JSON_USER_TEMPLATE_SUBSTRINGS: Final[dict[str, tuple[str, ...]]] = {
    "shared/summarization/bullets_json_v1": (
        "Return ONLY valid JSON",
        "Use this shape:",
        '{"title": null or a short episode headline, "bullets":',
    ),
    "shared/summarization/bundled_clean_summary_user_v1": (
        "Return ONLY valid JSON",
        "Required shape:",
        '"summary"',
    ),
}


def shared_render_kwargs(logical_name: str) -> dict[str, object]:
    """Minimal kwargs so ``render_prompt`` succeeds for contract checks."""
    marker = f"__contract_marker_{logical_name.replace('/', '_')}__"
    if logical_name == "shared/summarization/bullets_json_v1":
        return {"transcript": marker, "title": "Contract Title"}
    if logical_name == "shared/summarization/bundled_clean_summary_user_v1":
        return {"transcript": marker, "title": "Contract Title"}
    if logical_name == "shared/kg_graph_extraction/v1":
        return {
            "title": "Contract Title",
            "max_topics": 3,
            "max_entities": 5,
            "transcript": marker,
        }
    if logical_name == "shared/kg_graph_extraction/v2":
        return {
            "title": "Contract Title",
            "max_topics": 3,
            "max_entities": 5,
            "transcript": marker,
        }
    if logical_name == "shared/kg_graph_extraction/v5":
        return {
            "title": "Contract Title",
            "max_topics": 3,
            "max_entities": 5,
            "transcript": marker,
            "ner_entity_hints": [],
        }
    raise ValueError(f"No render kwargs registered for {logical_name!r}")


def assert_transcript_placeholder_in_source(logical_name: str) -> None:
    from podcast_scraper.prompts.store import get_prompt_source

    src = get_prompt_source(logical_name)
    assert (
        _TRANSCRIPT_SNIP in src
    ), f"{logical_name} must contain {_TRANSCRIPT_SNIP} so episode text reaches the model"


def assert_transcript_absent_from_source(logical_name: str) -> None:
    from podcast_scraper.prompts.store import get_prompt_source

    src = get_prompt_source(logical_name)
    assert _TRANSCRIPT_SNIP not in src, (
        f"{logical_name} must not embed {_TRANSCRIPT_SNIP} " "(system / bullets-only prompt)"
    )
