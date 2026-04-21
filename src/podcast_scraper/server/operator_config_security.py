"""Reject operator YAML that embeds secrets or feed-list keys (viewer PUT / file checks).

**Top-level** mapping keys only: forbidden credential names, ``*_api_key`` / password /
secret heuristics, and feed-source keys (``rss``, ``rss_url``, ``rss_urls``, ``feeds``)
that belong in corpus ``feeds.spec.yaml`` (Feeds API) for the viewer workflow.

Nested mappings (e.g. ``providers:``) are **not** scanned — secrets there would still
load in the real pipeline if present on disk. Treat this API as a **shallow** gate, not
a full secret or feed scanner.

When both feed keys and other forbidden keys appear, ``detail.error`` is
``forbidden_operator_keys`` (not ``forbidden_operator_feed_keys``); ``detail.keys`` lists
all offending top-level keys.

Raises :class:`OperatorYamlUnsafeError` (no FastAPI dependency) so unit tests can import
this module under ``.[dev]`` only; HTTP routes translate to ``HTTPException``.
"""

from __future__ import annotations

from typing import Any

import yaml

# Known Config credential fields (snake_case). Extend when new provider keys ship.
# Top-level keys that duplicate the canonical feeds list / RSS sources (Feeds API +
# feeds.spec.yaml).
_FORBIDDEN_FEED_KEYS_NORMALIZED: frozenset[str] = frozenset(
    {
        "rss",
        "rss_url",
        "rss_urls",
        "feeds",
    }
)

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


class OperatorYamlUnsafeError(Exception):
    """Invalid or forbidden operator YAML (routes map to ``HTTPException``)."""

    def __init__(self, status_code: int, detail: str | dict[str, Any]) -> None:
        super().__init__(str(detail))
        self.status_code = status_code
        self.detail = detail


def _norm_key(key: str) -> str:
    return key.strip().lower().replace("-", "_")


def forbidden_operator_top_level_keys(data: dict[str, Any]) -> list[str]:
    """Return original key names that must not appear at the YAML root."""
    bad: list[str] = []
    for raw_key in data:
        nk = _norm_key(str(raw_key))
        if nk in _FORBIDDEN_FEED_KEYS_NORMALIZED:
            bad.append(str(raw_key))
            continue
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
    """Raise :class:`OperatorYamlUnsafeError` when YAML is invalid or has forbidden keys."""
    if not str(text).strip():
        return
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise OperatorYamlUnsafeError(
            422,
            f"Invalid YAML: {exc}",
        ) from exc
    if parsed is None:
        return
    if not isinstance(parsed, dict):
        raise OperatorYamlUnsafeError(
            422,
            "Operator YAML must parse to a mapping (object) at the root.",
        )
    bad = forbidden_operator_top_level_keys(parsed)
    if bad:
        feed_hits = [k for k in bad if _norm_key(k) in _FORBIDDEN_FEED_KEYS_NORMALIZED]
        err = (
            "forbidden_operator_feed_keys"
            if feed_hits and len(feed_hits) == len(bad)
            else "forbidden_operator_keys"
        )
        raise OperatorYamlUnsafeError(
            400,
            {"error": err, "keys": bad},
        )
