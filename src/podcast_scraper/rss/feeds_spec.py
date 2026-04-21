"""Structured corpus feed list: ``{ feeds: [...] }`` in YAML or JSON (RFC-077 / #626).

Each entry is either a URL string or an object with required ``url`` plus optional
per-feed overrides (download resilience, timeouts, episode window). Validated with
Pydantic (same style as main :class:`~podcast_scraper.config.Config`).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypeVar
from urllib.parse import urlparse

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

_CfgT = TypeVar("_CfgT", bound=BaseModel)

logger = logging.getLogger(__name__)

# Canonical basename when the server resolves a path under the corpus root.
FEEDS_SPEC_DEFAULT_BASENAME = "feeds.spec.yaml"

# Keys allowed on a feed object beyond ``url`` (must exist on Config for model_copy).
RSS_FEED_ENTRY_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {
        "user_agent",
        "timeout",
        "http_retry_total",
        "http_backoff_factor",
        "rss_retry_total",
        "rss_backoff_factor",
        "episode_retry_max",
        "episode_retry_delay_sec",
        "delay_ms",
        "host_request_interval_ms",
        "host_max_concurrent",
        "circuit_breaker_enabled",
        "circuit_breaker_failure_threshold",
        "circuit_breaker_window_seconds",
        "circuit_breaker_cooldown_seconds",
        "circuit_breaker_scope",
        "rss_conditional_get",
        "rss_cache_dir",
        "max_episodes",
        "episode_order",
        "episode_offset",
        "episode_since",
        "episode_until",
    }
)


def _validate_http_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"RSS URL must use http or https (got {parsed.scheme!r}): {url}")
    if not parsed.netloc:
        raise ValueError(f"RSS URL must include a valid hostname: {url}")
    return url


class RssFeedEntry(BaseModel):
    """One feed: ``url`` plus optional per-inner-run overrides (allowlist only)."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    url: str = Field(validation_alias=AliasChoices("url", "rss", "rss_url"))
    user_agent: Optional[str] = None
    timeout: Optional[int] = Field(default=None, ge=1)
    http_retry_total: Optional[int] = Field(default=None, ge=0, le=20)
    http_backoff_factor: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    rss_retry_total: Optional[int] = Field(default=None, ge=0, le=20)
    rss_backoff_factor: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    episode_retry_max: Optional[int] = Field(default=None, ge=0, le=10)
    episode_retry_delay_sec: Optional[float] = Field(default=None, ge=0.0, le=120.0)
    delay_ms: Optional[int] = Field(default=None, ge=0)
    host_request_interval_ms: Optional[int] = Field(default=None, ge=0, le=600_000)
    host_max_concurrent: Optional[int] = Field(default=None, ge=0, le=64)
    circuit_breaker_enabled: Optional[bool] = None
    circuit_breaker_failure_threshold: Optional[int] = Field(default=None, ge=1, le=100)
    circuit_breaker_window_seconds: Optional[int] = Field(default=None, ge=1, le=86400)
    circuit_breaker_cooldown_seconds: Optional[int] = Field(default=None, ge=1, le=86400)
    circuit_breaker_scope: Optional[Literal["feed", "host"]] = None
    rss_conditional_get: Optional[bool] = None
    rss_cache_dir: Optional[str] = None
    max_episodes: Optional[int] = None
    episode_order: Optional[Literal["newest", "oldest"]] = None
    episode_offset: Optional[int] = Field(default=None, ge=0)
    episode_since: Optional[str] = None
    episode_until: Optional[str] = None

    @field_validator("url", mode="after")
    @classmethod
    def _check_url(cls, v: str) -> str:
        return _validate_http_url(v.strip())

    def override_update_dict(self) -> Dict[str, Any]:
        """Flat dict for ``Config.model_copy(update=...)`` — excludes ``url``."""
        data = self.model_dump(exclude_none=True)
        data.pop("url", None)
        return data


class FeedsSpecDocument(BaseModel):
    """Root document: ``{ feeds: [...] }`` plus optional ``_comment*`` keys (ignored)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    feeds: List[RssFeedEntry] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_feed_items(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        raw = out.get("feeds")
        if raw is None:
            out["feeds"] = []
            return out
        if not isinstance(raw, list):
            raise TypeError("feeds must be a list")
        normalized: List[Any] = []
        for item in raw:
            if item is None:
                continue
            if isinstance(item, str):
                s = item.strip()
                if s:
                    normalized.append({"url": s})
            elif isinstance(item, dict):
                normalized.append(dict(item))
            else:
                raise TypeError("each feeds[] entry must be a string or object")
        out["feeds"] = normalized
        return out


def load_feeds_spec_file(path: str | Path) -> FeedsSpecDocument:
    """Load and validate ``feeds.spec.{yaml,yml,json}`` (root object with ``feeds`` array only).

    Callers must pass a path already bound to a trusted corpus root (HTTP routes use
    ``normpath_if_under_root`` before calling); the CLI may pass explicit local paths.
    """
    p = Path(path).expanduser()
    # codeql[py/path-injection] -- caller supplies trusted path (Type 1; CODEQL_DISMISSALS.md).
    if not p.is_file():
        raise ValueError(f"Feeds spec file not found: {p}")
    suffix = p.suffix.lower()
    # codeql[py/path-injection] -- same as is_file guard above.
    text = p.read_text(encoding="utf-8")
    if suffix == ".json":
        raw = json.loads(text)
    elif suffix in (".yaml", ".yml"):
        raw = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported feeds spec extension (use .json, .yaml, .yml): {p}")
    if not isinstance(raw, dict):
        raise ValueError("Feeds spec must be a JSON/YAML object at the top level")
    unknown = [k for k in raw if k != "feeds" and not str(k).startswith("_")]
    if unknown:
        raise ValueError(
            "Unknown top-level keys in feeds spec (only `feeds` and `_comment*` allowed): "
            + ", ".join(sorted(unknown))
        )
    return FeedsSpecDocument.model_validate(raw)


def dump_feeds_spec_yaml(doc: FeedsSpecDocument) -> str:
    """Serialize document to YAML (default for viewer/API writes)."""
    items: List[Any] = []
    for e in doc.feeds:
        d = e.model_dump(mode="json", exclude_none=True)
        if set(d.keys()) == {"url"}:
            items.append(d["url"])
        else:
            items.append(d)
    payload: Dict[str, Any] = {"feeds": items}
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, default_flow_style=False)


def append_normalized_feed_items(bucket: List[dict], items: Optional[List[Any]]) -> None:
    """Append coerced feed dicts to *bucket*, deduping by ``url`` (first wins)."""
    if not items:
        return
    seen = {str(b.get("url", "")).strip() for b in bucket if b.get("url")}
    for u in items:
        if u is None:
            continue
        if isinstance(u, str):
            t = u.strip()
            if not t or t in seen:
                continue
            seen.add(t)
            bucket.append({"url": t})
        elif isinstance(u, dict):
            url = str(u.get("url") or u.get("rss") or u.get("rss_url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            d: Dict[str, Any] = {"url": url}
            for k, v in u.items():
                if k in ("url", "rss", "rss_url"):
                    continue
                if k in RSS_FEED_ENTRY_OVERRIDE_KEYS:
                    d[k] = v
            bucket.append(d)
        else:
            raise TypeError("feeds/rss_urls entries must be a string or a mapping with url")


def merge_feed_entry_into_config(cfg: _CfgT, entry: RssFeedEntry) -> _CfgT:
    """Return a new Config with ``rss_url`` set and per-feed overrides applied."""
    updates: Dict[str, Any] = {"rss_url": entry.url, "rss_urls": None}
    updates.update(entry.override_update_dict())
    return cfg.model_copy(update=updates)
