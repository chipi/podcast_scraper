"""Lightweight prompt management for LLM experiments and production use.

Features:
- File-based prompts (Jinja2 templates)
- Loading by logical name (e.g. "summarization/long_v1")
- In-memory caching to avoid repeated disk I/O
- Optional templating parameters via Jinja2
- SHA256 hashes for reproducible experiment metadata

This module implements RFC-017: Prompt Management and Loading.
"""

from __future__ import annotations

import os
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict

from jinja2 import Template

# Root directory where all your prompt templates live.
# Default: project_root/prompts/
# Can be overridden via environment variable PROMPT_DIR
_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


class PromptNotFoundError(FileNotFoundError):
    """Raised when a requested prompt template is not found on disk."""


def set_prompt_dir(path: str | Path) -> None:
    """Set the root directory for prompt templates.

    Useful for testing or custom prompt locations.

    Args:
        path: Path to prompt directory
    """
    global _PROMPT_DIR
    _PROMPT_DIR = Path(path).resolve()
    # Clear cache when directory changes
    _load_template.cache_clear()


def get_prompt_dir() -> Path:
    """Get the current prompt directory.

    Returns:
        Path to prompt directory
    """
    return _PROMPT_DIR


@lru_cache(maxsize=None)
def _load_template(name: str) -> Template:
    """
    Load and cache a Jinja2 template by logical name.

    Example:
        name="summarization/long_v1" -> prompts/summarization/long_v1.j2

    Args:
        name: Logical name without .j2 extension

    Returns:
        Jinja2 Template object

    Raises:
        PromptNotFoundError: If template file doesn't exist
    """
    # Check environment variable for custom prompt directory
    env_prompt_dir = os.getenv("PROMPT_DIR")
    if env_prompt_dir:
        prompt_dir = Path(env_prompt_dir).resolve()
    else:
        prompt_dir = _PROMPT_DIR

    # Normalize: allow both "summarization/long_v1" and "summarization/long_v1.j2"
    if name.endswith(".j2"):
        rel_path = Path(name)
    else:
        rel_path = Path(name + ".j2")

    path = prompt_dir / rel_path

    if not path.exists():
        raise PromptNotFoundError(
            f"Prompt template not found: {path}\n"
            f"  Searched in: {prompt_dir}\n"
            f"  Requested name: {name}"
        )

    text = path.read_text(encoding="utf-8")
    return Template(text)


def render_prompt(name: str, **params: Any) -> str:
    """
    Render a prompt template with optional parameters.

    Args:
        name: Logical name, e.g. "summarization/long_v1"
        **params: Template parameters passed to Jinja2 .render()

    Returns:
        Rendered prompt string (stripped of leading/trailing whitespace).

    Example:
        >>> render_prompt("summarization/long_v1", paragraphs_min=3, paragraphs_max=6)
        "You are summarizing a podcast episode.\\n\\nWrite a detailed..."

    Raises:
        PromptNotFoundError: If template file doesn't exist
    """
    tmpl = _load_template(name)
    return tmpl.render(**params).strip()


def get_prompt_source(name: str) -> str:
    """
    Return the raw template source text (without rendering).
    Useful for hashing / metadata.

    Args:
        name: Logical name, e.g. "summarization/long_v1"

    Returns:
        Raw template source as string

    Raises:
        PromptNotFoundError: If template file doesn't exist
    """
    # Check environment variable for custom prompt directory
    env_prompt_dir = os.getenv("PROMPT_DIR")
    if env_prompt_dir:
        prompt_dir = Path(env_prompt_dir).resolve()
    else:
        prompt_dir = _PROMPT_DIR

    if name.endswith(".j2"):
        rel_path = Path(name)
    else:
        rel_path = Path(name + ".j2")
    path = prompt_dir / rel_path
    return path.read_text(encoding="utf-8")


def hash_text(text: str) -> str:
    """
    Return a SHA256 hex digest for arbitrary text.

    Args:
        text: Text to hash

    Returns:
        SHA256 hash as hex string
    """
    return sha256(text.encode("utf-8")).hexdigest()


def get_prompt_metadata(
    name: str,
    params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Return metadata describing a prompt configuration.

    Includes:
        - logical name ("summarization/long_v1")
        - filename (relative path)
        - sha256 hash of template source
        - params used for rendering (if any)

    Args:
        name: Logical name, e.g. "summarization/long_v1"
        params: Optional template parameters

    Returns:
        Dictionary with prompt metadata

    Raises:
        PromptNotFoundError: If template file doesn't exist
    """
    # Check environment variable for custom prompt directory
    env_prompt_dir = os.getenv("PROMPT_DIR")
    if env_prompt_dir:
        prompt_dir = Path(env_prompt_dir).resolve()
    else:
        prompt_dir = _PROMPT_DIR

    if name.endswith(".j2"):
        rel_path = Path(name)
    else:
        rel_path = Path(name + ".j2")

    path = prompt_dir / rel_path
    source = get_prompt_source(name)

    metadata: Dict[str, Any] = {
        "name": name,
        "file": str(path.relative_to(prompt_dir)),
        "sha256": hash_text(source),
    }

    if params:
        metadata["params"] = params

    return metadata


def clear_cache() -> None:
    """Clear the prompt template cache.

    Useful for testing or when prompts are updated during development.
    """
    _load_template.cache_clear()
