"""Regression: cleaning prompts must embed the transcript.

No LLM calls — checks Jinja source and local ``render_prompt`` output only.
"""

from __future__ import annotations

from podcast_scraper.prompts.store import (
    get_prompt_dir,
    get_prompt_source,
    render_prompt,
)

# Every ``<provider>/cleaning/v1.j2`` on disk must be listed here (see sync test below).
_CONTRACT_CLEANING_V1_PROVIDERS: tuple[str, ...] = (
    "anthropic",
    "deepseek",
    "gemini",
    "grok",
    "mistral",
    "ollama",
    "openai",
)


def _assert_cleaning_v1_contract(provider: str) -> None:
    logical_name = f"{provider}/cleaning/v1"
    raw = get_prompt_source(logical_name)
    assert "{{ transcript }}" in raw, (
        f"{logical_name} must contain Jinja reference {{ transcript }} "
        "(otherwise cleaning runs without episode text)"
    )
    marker = f"__test_transcript_marker_{provider}__"
    rendered = render_prompt(logical_name, transcript=marker)
    assert marker in rendered, f"{logical_name} rendered prompt must include transcript body"


def test_cleaning_v1_on_disk_matches_contract_provider_list() -> None:
    root = get_prompt_dir()
    discovered = sorted(p.parent.parent.name for p in root.glob("*/cleaning/v1.j2"))
    expected = sorted(_CONTRACT_CLEANING_V1_PROVIDERS)
    assert discovered == expected, (
        "Update _CONTRACT_CLEANING_V1_PROVIDERS when adding/removing "
        f"**/cleaning/v1.j2 (disk={discovered!r} contract={expected!r})"
    )


def test_anthropic_cleaning_v1_embeds_transcript() -> None:
    _assert_cleaning_v1_contract("anthropic")


def test_deepseek_cleaning_v1_embeds_transcript() -> None:
    _assert_cleaning_v1_contract("deepseek")


def test_gemini_cleaning_v1_embeds_transcript() -> None:
    _assert_cleaning_v1_contract("gemini")


def test_grok_cleaning_v1_embeds_transcript() -> None:
    _assert_cleaning_v1_contract("grok")


def test_mistral_cleaning_v1_embeds_transcript() -> None:
    _assert_cleaning_v1_contract("mistral")


def test_ollama_cleaning_v1_embeds_transcript() -> None:
    _assert_cleaning_v1_contract("ollama")


def test_openai_cleaning_v1_embeds_transcript() -> None:
    _assert_cleaning_v1_contract("openai")
