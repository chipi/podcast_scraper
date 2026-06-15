"""Connection-level resilience primitives for any inference-service HTTP client.

Used to be ``providers/tailnet_dgx/resilience.py`` — extracted here in
2026-06-15 so the primitives stop carrying deployment-specific words in their
import path. Nothing about ``TimeoutLike``, ``CircuitBreaker``, the watchdog,
or the hardened HTTP client is DGX-specific or Tailnet-specific; they apply
to any single-GPU inference contention scenario or any long-blocking HTTP
client.

Sub-modules:

- :mod:`.exceptions` — :data:`TimeoutLike`
- :mod:`.breakers` — :class:`CircuitBreaker`
- :mod:`.deadlines` — :func:`run_with_watchdog`, :data:`WATCHDOG_GRACE_SEC`
- :mod:`.sockets` — :func:`keepalive_socket_options`, :func:`hardened_http_client`,
  :func:`probe_audio_duration_sec`, :func:`effective_timeout_sec`
"""

from __future__ import annotations

from .breakers import CircuitBreaker
from .deadlines import run_with_watchdog, WATCHDOG_GRACE_SEC
from .exceptions import TimeoutLike
from .sockets import (
    effective_timeout_sec,
    hardened_http_client,
    keepalive_socket_options,
    probe_audio_duration_sec,
)

__all__ = [
    "CircuitBreaker",
    "TimeoutLike",
    "WATCHDOG_GRACE_SEC",
    "effective_timeout_sec",
    "hardened_http_client",
    "keepalive_socket_options",
    "probe_audio_duration_sec",
    "run_with_watchdog",
]
