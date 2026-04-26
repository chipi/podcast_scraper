"""Regression guard: Jinja prompt templates must ship with the wheel.

setuptools' default behaviour is to strip non-Python files out of the
distribution unless ``[tool.setuptools.package-data]`` lists them.
Without that config, an editable install (``pip install -e .``) reads
``.j2`` templates from the source tree fine — but a wheel install
(Docker images, ``pip install podcast-scraper``) loses them and
Gemini / Groq / OpenAI provider paths fail at runtime with
``PromptNotFoundError`` (``stack-test cloud-thin`` was the canonical
trip wire — #666 follow-up).

This test fails if a future ``pyproject.toml`` change drops the
``package-data`` entry, or if a template is moved out of the
package tree without the manifest noticing.
"""

from __future__ import annotations

import importlib.resources as resources
from pathlib import Path

import pytest


def _prompts_root() -> Path:
    """Resolve ``podcast_scraper.prompts`` package root via importlib.

    Works for both editable installs (returns the source path) and
    wheel installs (returns the unpacked site-packages path).
    """
    files = resources.files("podcast_scraper.prompts")
    return Path(str(files))


@pytest.mark.parametrize(
    "rel_path",
    [
        # Cloud providers — exercised by the stack-test cloud-thin run.
        "gemini/ner/guest_host_v1.j2",
        "gemini/summarization/long_v1.j2",
        "gemini/summarization/system_v1.j2",
        "openai/ner/guest_host_v1.j2",
        "anthropic/ner/guest_host_v1.j2",
        # Shared (KG extraction). Used by all providers.
        "shared/kg_graph_extraction/v1.j2",
        # Local providers (airgapped path) — also bundled.
        "groq/ner/guest_host_v1.j2",
        "deepseek/cleaning/v1.j2",
    ],
)
def test_prompt_template_present_under_package(rel_path: str) -> None:
    """Each canonical prompt must be discoverable under the installed
    ``podcast_scraper.prompts`` package, not just the source tree.
    """
    full = _prompts_root() / rel_path
    assert full.exists(), (
        f"Missing prompt template after install: {rel_path}\n"
        f"Expected at: {full}\n"
        f"Likely cause: ``pyproject.toml`` lost its "
        f"``[tool.setuptools.package-data]`` entry for ``prompts/**/*.j2``."
    )


def test_render_prompt_can_load_a_real_template() -> None:
    """End-to-end: ``render_prompt`` loads + parses a known template
    against the installed package.
    """
    from podcast_scraper.prompts.store import render_prompt

    # ``shared/kg_graph_extraction/v1.j2`` takes a few params; pass a
    # minimal subset so the Jinja render runs successfully. We only
    # care that the template is found + parses, not the output content.
    rendered = render_prompt(
        "shared/kg_graph_extraction/v1",
        transcript="(stub transcript for the test)",
        max_topics=10,
        max_entities=15,
    )
    assert isinstance(rendered, str)
    assert len(rendered) > 0
