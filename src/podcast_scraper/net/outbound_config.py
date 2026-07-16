"""Outbound-network config models + parse/validate/redact.

Read from the operator YAML's ``outbound:`` block. All fields are optional; a
missing block or missing keys defaults to "proxy off, verify=on, system trust".

Validation runs at parse time so a bad PUT is rejected before the registry
swap. Redaction runs on every echo (GET response, log line, warning) so the
proxy password never leaks. Redaction is intentionally paranoid: masks anything
that looks like userinfo even if the field label suggests it is safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

_ALLOWED_PROXY_SCHEMES = frozenset({"http", "https"})


class OutboundConfigError(ValueError):
    """Raised when an ``outbound:`` block is invalid (routes map to HTTP 422)."""


@dataclass(frozen=True)
class ProxyConfig:
    """Global outbound HTTP proxy config."""

    enabled: bool = False
    url: str | None = None
    no_proxy: tuple[str, ...] = ()

    def validate(self) -> None:
        """Raise :class:`OutboundConfigError` if the block is inconsistent."""
        if not self.enabled:
            return
        if not self.url:
            raise OutboundConfigError("outbound.proxy.enabled=true requires outbound.proxy.url")
        parsed = urlparse(self.url)
        if parsed.scheme not in _ALLOWED_PROXY_SCHEMES:
            raise OutboundConfigError(
                f"outbound.proxy.url scheme must be http or https (got {parsed.scheme!r})"
            )
        if not parsed.hostname:
            raise OutboundConfigError("outbound.proxy.url missing host")


@dataclass(frozen=True)
class TlsConfig:
    """Outbound TLS trust config (custom CA bundle + verify toggle + mTLS)."""

    verify: bool = True
    ca_bundle: str | None = None
    client_cert: str | None = None
    client_key: str | None = None

    def validate(self) -> None:
        """Raise :class:`OutboundConfigError` on missing/unpaired file paths."""
        for label, raw in (
            ("ca_bundle", self.ca_bundle),
            ("client_cert", self.client_cert),
            ("client_key", self.client_key),
        ):
            if raw is None:
                continue
            if not Path(raw).is_file():
                raise OutboundConfigError(
                    f"outbound.tls.{label} does not exist or is not a file: {raw}"
                )
        if bool(self.client_cert) ^ bool(self.client_key):
            raise OutboundConfigError(
                "outbound.tls.client_cert and outbound.tls.client_key must be set together (mTLS)"
            )


@dataclass(frozen=True)
class OutboundConfig:
    """Top-level ``outbound:`` block combining proxy + TLS trust."""

    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    tls: TlsConfig = field(default_factory=TlsConfig)

    def validate(self) -> None:
        """Validate both blocks; raises on the first offender."""
        self.proxy.validate()
        self.tls.validate()

    @classmethod
    def defaults(cls) -> OutboundConfig:
        """Return the safe default (proxy off, verify=on, system trust)."""
        return cls()


def _coerce_str_list(raw: Any, label: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise OutboundConfigError(f"{label} must be a list of strings")
    items: list[str] = []
    for i, v in enumerate(raw):
        if not isinstance(v, str):
            raise OutboundConfigError(f"{label}[{i}] must be a string")
        items.append(v)
    return tuple(items)


def _coerce_bool(raw: Any, label: str, *, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    raise OutboundConfigError(f"{label} must be true/false")


def _coerce_optional_str(raw: Any, label: str) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise OutboundConfigError(f"{label} must be a string")
    s = raw.strip()
    return s or None


def load_from_operator_yaml(parsed: dict[str, Any] | None) -> OutboundConfig:
    """Extract + validate an ``outbound:`` block from parsed operator YAML.

    ``parsed`` is the already-``yaml.safe_load``-ed operator YAML root (dict or None).
    Missing block returns defaults. Any nested type error raises :class:`OutboundConfigError`.
    """
    if not parsed:
        return OutboundConfig.defaults()
    block = parsed.get("outbound")
    if block is None:
        return OutboundConfig.defaults()
    if not isinstance(block, dict):
        raise OutboundConfigError("outbound: must be a mapping")

    proxy_raw = block.get("proxy") or {}
    if not isinstance(proxy_raw, dict):
        raise OutboundConfigError("outbound.proxy must be a mapping")
    proxy = ProxyConfig(
        enabled=_coerce_bool(proxy_raw.get("enabled"), "outbound.proxy.enabled", default=False),
        url=_coerce_optional_str(proxy_raw.get("url"), "outbound.proxy.url"),
        no_proxy=_coerce_str_list(proxy_raw.get("no_proxy"), "outbound.proxy.no_proxy"),
    )

    tls_raw = block.get("tls") or {}
    if not isinstance(tls_raw, dict):
        raise OutboundConfigError("outbound.tls must be a mapping")
    tls = TlsConfig(
        verify=_coerce_bool(tls_raw.get("verify"), "outbound.tls.verify", default=True),
        ca_bundle=_coerce_optional_str(tls_raw.get("ca_bundle"), "outbound.tls.ca_bundle"),
        client_cert=_coerce_optional_str(tls_raw.get("client_cert"), "outbound.tls.client_cert"),
        client_key=_coerce_optional_str(tls_raw.get("client_key"), "outbound.tls.client_key"),
    )

    cfg = OutboundConfig(proxy=proxy, tls=tls)
    cfg.validate()
    return cfg


def _redact_proxy_url(url: str | None) -> str | None:
    """Mask the ``user:pass@`` segment of a proxy URL, keeping scheme/host/port/path."""
    if not url:
        return url
    try:
        parsed = urlparse(url)
    except ValueError:
        return "[REDACTED]"
    if not parsed.hostname:
        return url
    host = parsed.hostname
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    userinfo_present = bool(parsed.username or parsed.password)
    netloc = f"[REDACTED]@{host}" if userinfo_present else host
    return urlunparse(parsed._replace(netloc=netloc))


def redact_for_echo(cfg: OutboundConfig) -> dict[str, Any]:
    """Return a JSON-safe, redacted view of ``cfg`` for GET responses / logs.

    Also carries ``insecure_mode_flags`` so operators see (and admins accept)
    that verify=false is active — never silent.
    """
    return {
        "proxy": {
            "enabled": cfg.proxy.enabled,
            "url": _redact_proxy_url(cfg.proxy.url),
            "no_proxy": list(cfg.proxy.no_proxy),
        },
        "tls": {
            "verify": cfg.tls.verify,
            "ca_bundle": cfg.tls.ca_bundle,
            "client_cert": cfg.tls.client_cert,
            "client_key": cfg.tls.client_key,
        },
        "insecure_mode_flags": {
            "tls_verify_off": not cfg.tls.verify,
        },
    }


def with_overrides(
    cfg: OutboundConfig,
    *,
    proxy: ProxyConfig | None = None,
    tls: TlsConfig | None = None,
) -> OutboundConfig:
    """Return a new :class:`OutboundConfig` with the given blocks swapped in."""
    return replace(
        cfg,
        proxy=proxy if proxy is not None else cfg.proxy,
        tls=tls if tls is not None else cfg.tls,
    )
