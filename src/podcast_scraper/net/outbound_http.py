"""httpx client factory reading proxy + TLS trust from the registry.

Call sites use ``create_client(...)``/``create_async_client(...)`` in place of
``httpx.Client(...)`` / ``httpx.AsyncClient(...)``. Long-lived clients that
need to survive a config swap should subscribe via
``get_registry().add_listener(...)`` and rebuild themselves.

Vendor SDK clients that accept a custom ``http_client`` (OpenAI, Anthropic,
etc.) pass a client built here; SDKs that own their transport internally
(``huggingface_hub``, pyannote) rely on the env-mirror in
:mod:`podcast_scraper.net.outbound_registry`.

Proxy + no_proxy routing uses httpx ``mounts=`` (per-URL transport table)
rather than the top-level ``proxy=`` kwarg — ``proxy=`` alone routes *every*
request through the proxy and silently ignores ``no_proxy``. ``mounts=`` gives
us a bypass entry per pattern plus a default catch-all with the proxy set.
``trust_env=False`` keeps factory-built clients deterministic: the registry
mirrors env vars for third-party libs, not for our own clients.

TLS trust (verify, ca_bundle, client_cert/key) is baked into an
``ssl.SSLContext`` and passed via ``verify=`` — httpx 0.28+ deprecates the
top-level ``cert=`` kwarg and the string form of ``verify=``, so the context
carries everything (CA trust roots + mTLS chain) in one object.
"""

from __future__ import annotations

import logging
import ssl
from typing import Any, Callable, List, Optional, Tuple

import httpx

from podcast_scraper.net.outbound_config import OutboundConfig
from podcast_scraper.net.outbound_registry import get_registry

logger = logging.getLogger(__name__)

SyncTransportWrapper = Callable[[httpx.HTTPTransport], httpx.BaseTransport]
AsyncTransportWrapper = Callable[[httpx.AsyncHTTPTransport], httpx.AsyncBaseTransport]
# TCP setsockopt tuples: `(level, optname, value)` per Python `socket` docs.
SocketOptions = Optional[List[Tuple[int, int, int]]]


def _build_verify_arg(cfg: OutboundConfig) -> bool | ssl.SSLContext:
    """Return the ``verify=`` value for ``httpx.Client`` / ``HTTPTransport``.

    - ``verify=False``          → registry has ``tls.verify=false`` (foot-gun)
    - ``ssl.SSLContext``        → custom CA and/or mTLS client cert configured
    - ``True``                  → default: verify with the system trust store
    """
    if not cfg.tls.verify:
        return False
    if cfg.tls.ca_bundle or cfg.tls.client_cert:
        ctx = ssl.create_default_context(cafile=cfg.tls.ca_bundle)
        if cfg.tls.client_cert and cfg.tls.client_key:
            ctx.load_cert_chain(certfile=cfg.tls.client_cert, keyfile=cfg.tls.client_key)
        return ctx
    return True


def _no_proxy_mount_key(pattern: str) -> str:
    """Normalize a no_proxy entry to an httpx mount URL pattern.

    Examples:
      ``"localhost"``    → ``"all://localhost"``
      ``"*.internal"``   → ``"all://*.internal"``
      ``"10.0.0.0/8"``   → ``"all://10.0.0.0/8"`` (best-effort; httpx pattern-matches host)
    """
    return f"all://{pattern.strip()}"


def _sync_mounts(
    cfg: OutboundConfig,
    verify_arg: bool | ssl.SSLContext,
    wrapper: SyncTransportWrapper | None = None,
    socket_options: SocketOptions = None,
) -> dict[str, httpx.BaseTransport] | None:
    if not cfg.proxy.enabled or not cfg.proxy.url:
        return None
    transport_kwargs: dict[str, Any] = {"verify": verify_arg}
    if socket_options is not None:
        transport_kwargs["socket_options"] = socket_options
    proxied_base = httpx.HTTPTransport(proxy=cfg.proxy.url, **transport_kwargs)
    proxied: httpx.BaseTransport = wrapper(proxied_base) if wrapper else proxied_base
    if not cfg.proxy.no_proxy:
        return {"all://": proxied}
    direct_base = httpx.HTTPTransport(**transport_kwargs)
    direct: httpx.BaseTransport = wrapper(direct_base) if wrapper else direct_base
    mounts: dict[str, httpx.BaseTransport] = {}
    for pat in cfg.proxy.no_proxy:
        mounts[_no_proxy_mount_key(pat)] = direct
    mounts["all://"] = proxied
    return mounts


def _async_mounts(
    cfg: OutboundConfig,
    verify_arg: bool | ssl.SSLContext,
    wrapper: AsyncTransportWrapper | None = None,
    socket_options: SocketOptions = None,
) -> dict[str, httpx.AsyncBaseTransport] | None:
    if not cfg.proxy.enabled or not cfg.proxy.url:
        return None
    transport_kwargs: dict[str, Any] = {"verify": verify_arg}
    if socket_options is not None:
        transport_kwargs["socket_options"] = socket_options
    proxied_base = httpx.AsyncHTTPTransport(proxy=cfg.proxy.url, **transport_kwargs)
    proxied: httpx.AsyncBaseTransport = wrapper(proxied_base) if wrapper else proxied_base
    if not cfg.proxy.no_proxy:
        return {"all://": proxied}
    direct_base = httpx.AsyncHTTPTransport(**transport_kwargs)
    direct: httpx.AsyncBaseTransport = wrapper(direct_base) if wrapper else direct_base
    mounts: dict[str, httpx.AsyncBaseTransport] = {}
    for pat in cfg.proxy.no_proxy:
        mounts[_no_proxy_mount_key(pat)] = direct
    mounts["all://"] = proxied
    return mounts


def _resolve_common_kwargs(cfg: OutboundConfig, kwargs: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(kwargs)
    # trust_env=False so factory-built clients don't accidentally pick up
    # env vars that our own registry set for third-party libs — the factory
    # owns proxy/TLS routing explicitly here.
    resolved.setdefault("trust_env", False)
    resolved.setdefault("verify", _build_verify_arg(cfg))
    resolved.setdefault("headers", {})
    resolved["headers"].setdefault("X-Outbound-Subsystem", "?")
    return resolved


def create_client(
    *,
    subsystem: str,
    cfg: OutboundConfig | None = None,
    transport_wrapper: SyncTransportWrapper | None = None,
    socket_options: SocketOptions = None,
    **kwargs: Any,
) -> httpx.Client:
    """Build a sync ``httpx.Client`` honoring the current outbound config.

    Args:
        subsystem: Short label for logs / metrics ("rss", "llm_openai", …).
        cfg: Override the registry snapshot; testing hook only.
        transport_wrapper: Optional callable that wraps each underlying
            ``httpx.HTTPTransport`` — the RSS downloader uses this to compose a
            retry transport around whatever proxy/TLS transport the factory
            builds. Applied to every mount in the proxy case, and to the
            default transport in the no-proxy case.
        socket_options: Optional TCP-level ``setsockopt`` tuples applied at
            transport construction. Used by :func:`hardened_http_client` to
            inject TCP keepalive for long-blocking inference POSTs so the
            factory's proxy/TLS routing still governs those requests.
        **kwargs: Forwarded to ``httpx.Client``. Explicit values win over the
            factory defaults (e.g. a call site pre-baking its own ``verify``).
    """
    resolved_cfg = cfg or get_registry().current()
    resolved = _resolve_common_kwargs(resolved_cfg, kwargs)
    resolved["headers"]["X-Outbound-Subsystem"] = subsystem
    mounts = _sync_mounts(resolved_cfg, resolved["verify"], transport_wrapper, socket_options)
    if mounts is not None:
        resolved.setdefault("mounts", mounts)
    elif transport_wrapper is not None or socket_options is not None:
        transport_kwargs: dict[str, Any] = {"verify": resolved["verify"]}
        if socket_options is not None:
            transport_kwargs["socket_options"] = socket_options
        base = httpx.HTTPTransport(**transport_kwargs)
        wrapped = transport_wrapper(base) if transport_wrapper is not None else base
        resolved.setdefault("transport", wrapped)
    return httpx.Client(**resolved)


def create_async_client(
    *,
    subsystem: str,
    cfg: OutboundConfig | None = None,
    transport_wrapper: AsyncTransportWrapper | None = None,
    socket_options: SocketOptions = None,
    **kwargs: Any,
) -> httpx.AsyncClient:
    """Async equivalent of :func:`create_client`."""
    resolved_cfg = cfg or get_registry().current()
    resolved = _resolve_common_kwargs(resolved_cfg, kwargs)
    resolved["headers"]["X-Outbound-Subsystem"] = subsystem
    mounts = _async_mounts(resolved_cfg, resolved["verify"], transport_wrapper, socket_options)
    if mounts is not None:
        resolved.setdefault("mounts", mounts)
    elif transport_wrapper is not None or socket_options is not None:
        transport_kwargs: dict[str, Any] = {"verify": resolved["verify"]}
        if socket_options is not None:
            transport_kwargs["socket_options"] = socket_options
        base = httpx.AsyncHTTPTransport(**transport_kwargs)
        wrapped = transport_wrapper(base) if transport_wrapper is not None else base
        resolved.setdefault("transport", wrapped)
    return httpx.AsyncClient(**resolved)


def sdk_http_client(*, subsystem: str, **kwargs: Any) -> httpx.Client | None:
    """Build an ``httpx.Client`` suitable for passing as SDK ``http_client=``.

    Vendor SDKs (openai, anthropic, mistral, etc.) accept an ``http_client``
    kwarg — passing a factory-built client routes their traffic through the
    admin-configured proxy (#1129) / TLS trust (#1130).

    On construction error, logs at ``ERROR`` and returns ``None`` so SDK init
    can proceed with the SDK's own default transport. The SDK's default
    transport reads ``HTTPS_PROXY`` and ``SSL_CERT_FILE`` from the process env
    (which the registry mirrors, see
    :meth:`OutboundConfigRegistry._apply_env`) — so a factory-build failure
    still keeps proxy + CA-bundle routing intact.

    Env-mirror does NOT cover ``tls.verify=False`` or mTLS
    (``client_cert``/``client_key``). If the registry sets those, the
    factory-build error means the SDK will silently diverge from operator
    intent — verify=False intent → SDK still verifies (conn may fail);
    mTLS intent → SDK never presents the client cert (server rejects).
    That divergence is what the ERROR log + returned ``None`` are meant to
    surface: operators are expected to page on the log and either fix the
    misconfig or accept the fallback.
    """
    try:
        return create_client(subsystem=subsystem, **kwargs)
    except Exception as exc:  # pragma: no cover - exercised via unit test
        logger.error(
            "sdk_http_client(subsystem=%s) build failed: %s. SDK will fall back "
            "to its default transport (proxy + CA bundle covered by env-mirror; "
            "verify=False + mTLS silently ignored).",
            subsystem,
            exc,
        )
        return None
