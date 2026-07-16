"""Central outbound-network plumbing (#1129 proxy + #1130 TLS trust).

See ``docs/wip/1129-1130-OUTBOUND-HTTP-FACTORY.md`` for the design and
per-subsystem migration matrix.
"""

from __future__ import annotations

from podcast_scraper.net.outbound_config import (
    load_from_operator_yaml,
    OutboundConfig,
    ProxyConfig,
    redact_for_echo,
    TlsConfig,
)
from podcast_scraper.net.outbound_http import (
    create_async_client,
    create_client,
    sdk_http_client,
)
from podcast_scraper.net.outbound_registry import (
    get_registry,
    OutboundConfigRegistry,
)

__all__ = [
    "OutboundConfig",
    "ProxyConfig",
    "TlsConfig",
    "OutboundConfigRegistry",
    "create_async_client",
    "create_client",
    "get_registry",
    "load_from_operator_yaml",
    "redact_for_echo",
    "sdk_http_client",
]
