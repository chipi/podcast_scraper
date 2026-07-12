"""Process-wide singleton holding the current :class:`OutboundConfig`.

Every ``create_client(...)`` reads from ``get_registry().current()`` at build
time. Config PUTs call ``get_registry().swap(new)`` under a lock; a listener
list lets long-lived callers (RSS session, provider SDK clients) rebuild.

Env-mirror — scope and limits
-----------------------------

Also mirrors the config into standard env vars (``HTTPS_PROXY`` /
``SSL_CERT_FILE`` / etc.) so third-party libraries whose transport we cannot
inject (huggingface_hub, pyannote, some SDK internals) inherit the settings
transparently.

The mirror is NOT a general fallback. It covers exactly two knobs:

- **``proxy.url`` + ``proxy.no_proxy``** → ``HTTP_PROXY``, ``HTTPS_PROXY``,
  ``NO_PROXY``. httpx and requests both read these by default.
- **``tls.ca_bundle``** → ``SSL_CERT_FILE`` + ``REQUESTS_CA_BUNDLE``. Python's
  stdlib ``ssl`` module and the requests library both read these.

It does NOT cover:

- **``tls.verify=False``**. There is no widely-honored env var to disable
  cert verification; SDK defaults are ``verify=True``. A ``verify=False``
  intent that falls back to env-only routing will silently keep verifying
  and reject self-signed certs.
- **``tls.client_cert`` / ``tls.client_key`` (mTLS)**. Each SDK has its own
  API for client-cert plumbing; there is no cross-vendor env convention.
  A mTLS intent that falls back to env-only routing will silently omit the
  client cert.

Consequences of that scope:

- ``huggingface_hub`` and ``pyannote`` (transport-internal SDKs) work only
  for proxy + CA bundle. If operators need ``verify=False`` or mTLS toward
  those services, that's a design change, not a config change.
- ``sdk_http_client(...)`` returning ``None`` on error is safe for proxy +
  CA bundle, silently wrong for ``verify=False`` and mTLS — the helper logs
  ``ERROR`` when this happens so operators can page on the divergence.
- The DGX Whisper + diarize multipart POSTs go through
  :func:`hardened_http_client`, which delegates to ``create_client(...)`` —
  so those requests get the full registry treatment (proxy + verify + mTLS)
  and never depend on env-mirror.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Callable

from podcast_scraper.net.outbound_config import OutboundConfig, redact_for_echo

_logger = logging.getLogger(__name__)

# The env vars we mirror on every swap. Kept as a tuple so tests can iterate.
_PROXY_ENV_VARS: tuple[str, ...] = ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY")
_TLS_ENV_VARS: tuple[str, ...] = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")

Listener = Callable[[OutboundConfig], None]


class OutboundConfigRegistry:
    """Thread-safe holder for the current outbound config."""

    def __init__(self, initial: OutboundConfig | None = None) -> None:
        self._lock = threading.RLock()
        self._current: OutboundConfig = initial or OutboundConfig.defaults()
        self._listeners: list[Listener] = []
        self._env_snapshot: dict[str, str | None] = {}
        self._apply_env(self._current)

    def current(self) -> OutboundConfig:
        """Return the current registry snapshot under the lock."""
        with self._lock:
            return self._current

    def swap(self, new: OutboundConfig) -> OutboundConfig:
        """Replace the current config atomically; emit env mirror + notify listeners."""
        new.validate()
        with self._lock:
            old = self._current
            self._current = new
            self._apply_env(new)
            listeners = list(self._listeners)
        _log_swap(old, new)
        for cb in listeners:
            try:
                cb(new)
            except Exception:  # pragma: no cover - defensive
                _logger.exception("OutboundConfigRegistry listener raised")
        return old

    def add_listener(self, cb: Listener) -> Callable[[], None]:
        """Register ``cb`` to fire on every swap. Returns an unsubscribe fn."""
        with self._lock:
            self._listeners.append(cb)

        def _unsub() -> None:
            with self._lock:
                try:
                    self._listeners.remove(cb)
                except ValueError:
                    pass

        return _unsub

    # --- env-mirror ---

    def _apply_env(self, cfg: OutboundConfig) -> None:
        """Sync ``os.environ`` with the current config.

        Restores the pre-first-swap values on empty fields so tests + repeated
        swaps don't accumulate stale env state.
        """
        if not self._env_snapshot:
            for name in _PROXY_ENV_VARS + _TLS_ENV_VARS:
                self._env_snapshot[name] = os.environ.get(name)

        proxy_url = cfg.proxy.url if cfg.proxy.enabled else None
        no_proxy = ",".join(cfg.proxy.no_proxy) if cfg.proxy.no_proxy else None
        _set_or_restore(os.environ, "HTTP_PROXY", proxy_url, self._env_snapshot["HTTP_PROXY"])
        _set_or_restore(os.environ, "HTTPS_PROXY", proxy_url, self._env_snapshot["HTTPS_PROXY"])
        _set_or_restore(os.environ, "NO_PROXY", no_proxy, self._env_snapshot["NO_PROXY"])

        ca = cfg.tls.ca_bundle
        _set_or_restore(os.environ, "SSL_CERT_FILE", ca, self._env_snapshot["SSL_CERT_FILE"])
        _set_or_restore(
            os.environ,
            "REQUESTS_CA_BUNDLE",
            ca,
            self._env_snapshot["REQUESTS_CA_BUNDLE"],
        )


def _set_or_restore(
    env: os._Environ[str],
    name: str,
    value: str | None,
    original: str | None,
) -> None:
    if value is None:
        if original is None:
            env.pop(name, None)
        else:
            env[name] = original
        return
    env[name] = value


def _log_swap(old: OutboundConfig, new: OutboundConfig) -> None:
    if not new.tls.verify:
        _logger.warning(
            "Outbound TLS verification is DISABLED (outbound.tls.verify=false). "
            "All external HTTPS traffic will accept any certificate. This is a foot-gun; "
            "use only in trusted / dev environments."
        )
    if old != new:
        _logger.info("outbound config swapped: %s", redact_for_echo(new))


_registry_lock = threading.Lock()
_registry: OutboundConfigRegistry | None = None


def get_registry() -> OutboundConfigRegistry:
    """Return the process-wide registry, creating it lazily on first call."""
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = OutboundConfigRegistry()
        return _registry


def _reset_registry_for_tests() -> None:
    """Reset the singleton + scrub env-mirror vars — internal test helper only.

    The env-mirror on ``OutboundConfigRegistry.__init__`` snapshots
    ``os.environ`` before the first ``swap`` so it can restore later. If a
    previous test left ``HTTPS_PROXY`` / ``SSL_CERT_FILE`` set from its own
    swap, the next test's snapshot would think those were "pre-existing"
    values and never clear them — leaking config between tests. Explicit
    scrub here matches the isolation contract callers expect.
    """
    global _registry
    with _registry_lock:
        _registry = None
    for name in _PROXY_ENV_VARS + _TLS_ENV_VARS:
        os.environ.pop(name, None)
