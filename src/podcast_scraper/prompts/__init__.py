"""Prompt template management for LLM providers.

This package contains:
- Prompt store (store.py): Loading and caching of Jinja2 prompt templates
- Provider-specific prompts: Subdirectories for each LLM provider (openai/, anthropic/, etc.)
"""

from .store import (
    clear_cache,
    get_prompt_dir,
    get_prompt_metadata,
    get_prompt_source,
    hash_text,
    PromptNotFoundError,
    render_prompt,
    set_prompt_dir,
)

__all__ = [
    "PromptNotFoundError",
    "clear_cache",
    "get_prompt_dir",
    "get_prompt_metadata",
    "get_prompt_source",
    "hash_text",
    "render_prompt",
    "set_prompt_dir",
]
