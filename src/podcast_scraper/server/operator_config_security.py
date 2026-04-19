"""Reject operator YAML that embeds secrets (viewer PUT / operator file checks).

Only **top-level** mapping keys are checked. Nested mappings (e.g. ``providers:``) are
not scanned — secrets there would still load in the real pipeline if present on disk;
operators should treat this API as a convenience editor with a shallow denylist, not
a full secret scanner.
"""

from __future__ import annotations

from typing import Any

import yaml
from fastapi import HTTPException, status

# Known Config credential fields (snake_case). Extend when new provider keys ship.
_FORBIDDEN_NORMALIZED: frozenset[str] = frozenset(
    {
        "openai_api_key",
        "gemini_api_key",
        "anthropic_api_key",
        "mistral_api_key",
        "deepseek_api_key",
        "grok_api_key",
        "huggingface_token",
        "hf_token",
        "api_key",
    }
)


def _norm_key(key: str) -> str:
    return key.strip().lower().replace("-", "_")


def forbidden_operator_top_level_keys(data: dict[str, Any]) -> list[str]:
    """Return original key names that must not appear at the YAML root."""
    bad: list[str] = []
    for raw_key in data:
        nk = _norm_key(str(raw_key))
        if nk in _FORBIDDEN_NORMALIZED:
            bad.append(str(raw_key))
            continue
        if nk.endswith("_api_key"):
            bad.append(str(raw_key))
            continue
        if "password" in nk:
            bad.append(str(raw_key))
            continue
        if "secret" in nk:
            bad.append(str(raw_key))
    return bad


def assert_operator_yaml_safe_for_persist(text: str) -> None:
    """Raise HTTPException when YAML is invalid or contains forbidden top-level keys."""
    if not str(text).strip():
        return
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid YAML: {exc}",
        ) from exc
    if parsed is None:
        return
    if not isinstance(parsed, dict):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Operator YAML must parse to a mapping (object) at the root.",
        )
    bad = forbidden_operator_top_level_keys(parsed)
    if bad:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail={"error": "forbidden_operator_keys", "keys": bad},
        )
