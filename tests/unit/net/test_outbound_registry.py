"""Unit tests for the OutboundConfigRegistry env-mirror + listener behavior.

Findings from the pre-push harden audit (2026-07-16):

- ``_apply_env`` / ``_set_or_restore`` — mirrors config to ``HTTP_PROXY`` /
  ``HTTPS_PROXY`` / ``NO_PROXY`` / ``SSL_CERT_FILE`` / ``REQUESTS_CA_BUNDLE``
  on swap; restores pre-first-swap originals when a subsequent swap has empty
  fields. Regression in the restore logic would silently corrupt the env for
  ``huggingface_hub`` / ``pyannote`` which read those vars directly.
- ``add_listener`` round-trip (subscribe → swap → observe → unsubscribe →
  swap → NOT observe).
- ``_log_swap`` WARNING branch when ``tls.verify=False``.

None of these were covered before this file — a bug would have surfaced only
in integration tests, if at all.
"""

from __future__ import annotations

import logging
import os
from typing import Iterator

import pytest

from podcast_scraper.net.outbound_config import OutboundConfig, ProxyConfig, TlsConfig
from podcast_scraper.net.outbound_registry import (
    _reset_registry_for_tests,
    get_registry,
    OutboundConfigRegistry,
)

pytestmark = pytest.mark.unit


_MIRRORED_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Ensure every mirrored env var starts unset for the test."""
    for name in _MIRRORED_VARS:
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture
def reset_registry() -> Iterator[None]:
    _reset_registry_for_tests()
    yield
    _reset_registry_for_tests()


# ---- env-mirror on swap ----------------------------------------------------


def test_swap_sets_proxy_env_vars(clean_env, reset_registry) -> None:
    """Swapping to a config with a proxy sets both HTTP_PROXY + HTTPS_PROXY."""
    reg = OutboundConfigRegistry()
    reg.swap(OutboundConfig(proxy=ProxyConfig(enabled=True, url="http://p:3128")))
    assert os.environ.get("HTTP_PROXY") == "http://p:3128"
    assert os.environ.get("HTTPS_PROXY") == "http://p:3128"


def test_swap_sets_no_proxy_from_list(clean_env, reset_registry) -> None:
    """no_proxy list is joined with commas into the NO_PROXY env var."""
    reg = OutboundConfigRegistry()
    reg.swap(
        OutboundConfig(
            proxy=ProxyConfig(
                enabled=True, url="http://p:3128", no_proxy=("localhost", "*.internal")
            )
        )
    )
    assert os.environ.get("NO_PROXY") == "localhost,*.internal"


def test_swap_sets_ca_bundle_env_vars(clean_env, reset_registry, tmp_path) -> None:
    """SSL_CERT_FILE and REQUESTS_CA_BUNDLE both receive the ca_bundle path."""
    ca = tmp_path / "ca.pem"
    ca.write_text("-----BEGIN CERTIFICATE-----\n")
    reg = OutboundConfigRegistry()
    reg.swap(OutboundConfig(tls=TlsConfig(verify=True, ca_bundle=str(ca))))
    assert os.environ.get("SSL_CERT_FILE") == str(ca)
    assert os.environ.get("REQUESTS_CA_BUNDLE") == str(ca)


def test_swap_to_proxy_disabled_removes_env_vars(clean_env, reset_registry) -> None:
    """A swap to `proxy.enabled=False` must clear HTTP_PROXY / HTTPS_PROXY."""
    reg = OutboundConfigRegistry()
    reg.swap(OutboundConfig(proxy=ProxyConfig(enabled=True, url="http://p:3128")))
    assert "HTTP_PROXY" in os.environ

    reg.swap(OutboundConfig())  # proxy disabled
    assert "HTTP_PROXY" not in os.environ
    assert "HTTPS_PROXY" not in os.environ
    assert "NO_PROXY" not in os.environ


def test_apply_env_restores_originals_after_swap_to_empty(monkeypatch, reset_registry) -> None:
    """If HTTPS_PROXY was set BEFORE registry init, a later swap to an empty
    config must restore the pre-first-swap value, not clear it. Confirms the
    ``_env_snapshot`` capture + restore branch of ``_set_or_restore``.
    """
    # Pre-existing operator/system env value.
    monkeypatch.setenv("HTTPS_PROXY", "http://original.example:3128")

    reg = OutboundConfigRegistry()
    # First swap sets our proxy.
    reg.swap(OutboundConfig(proxy=ProxyConfig(enabled=True, url="http://ours:3128")))
    assert os.environ.get("HTTPS_PROXY") == "http://ours:3128"

    # Second swap disables → must restore the original, NOT delete.
    reg.swap(OutboundConfig())
    assert os.environ.get("HTTPS_PROXY") == "http://original.example:3128"


# ---- add_listener / unsubscribe --------------------------------------------


def test_add_listener_fires_on_swap(clean_env, reset_registry) -> None:
    """A subscribed listener is called with the new config on every swap."""
    reg = OutboundConfigRegistry()
    calls: list[OutboundConfig] = []

    reg.add_listener(calls.append)
    new_cfg = OutboundConfig(proxy=ProxyConfig(enabled=True, url="http://p:3128"))
    reg.swap(new_cfg)

    assert len(calls) == 1
    assert calls[0] is new_cfg


def test_unsubscribe_stops_listener(clean_env, reset_registry) -> None:
    """The returned unsubscribe callable removes the listener; subsequent swaps
    don't fire it.
    """
    reg = OutboundConfigRegistry()
    calls: list[OutboundConfig] = []

    unsub = reg.add_listener(calls.append)
    reg.swap(OutboundConfig(proxy=ProxyConfig(enabled=True, url="http://a:1")))
    assert len(calls) == 1

    unsub()
    reg.swap(OutboundConfig(proxy=ProxyConfig(enabled=True, url="http://b:2")))
    assert len(calls) == 1, "listener should not fire after unsubscribe"


def test_unsubscribe_is_idempotent(clean_env, reset_registry) -> None:
    """Calling unsubscribe twice is a no-op (guards against operator scripts
    that unsub in a finally block after already unsub'd in the happy path).
    """
    reg = OutboundConfigRegistry()
    unsub = reg.add_listener(lambda _: None)
    unsub()
    # Must not raise.
    unsub()


# ---- _log_swap verify=False WARNING ---------------------------------------


def test_swap_logs_warning_when_verify_false(clean_env, reset_registry, caplog) -> None:
    """A swap to ``tls.verify=False`` emits a WARNING that says "DISABLED".
    This is the foot-gun log line an operator sees when they mis-configure.
    """
    reg = OutboundConfigRegistry()
    caplog.set_level(logging.WARNING, logger="podcast_scraper.net.outbound_registry")
    reg.swap(OutboundConfig(tls=TlsConfig(verify=False)))

    assert any(
        "DISABLED" in rec.message and rec.levelno == logging.WARNING for rec in caplog.records
    ), f"expected DISABLED WARNING; got {[r.message for r in caplog.records]}"


def test_swap_does_not_log_when_verify_true(clean_env, reset_registry, caplog) -> None:
    """Verify=True (the safe default) must NOT emit the DISABLED WARNING —
    otherwise the log would false-positive on every deploy.
    """
    reg = OutboundConfigRegistry()
    caplog.set_level(logging.WARNING, logger="podcast_scraper.net.outbound_registry")
    reg.swap(OutboundConfig())

    assert not any(
        "DISABLED" in rec.message for rec in caplog.records
    ), "verify=True should not emit the disabled-TLS warning"


# ---- get_registry() singleton -------------------------------------------


def test_get_registry_returns_singleton(clean_env, reset_registry) -> None:
    """The module-level ``get_registry()`` returns the same instance across
    calls — factory clients + admin API surface both operate on one config.
    """
    assert get_registry() is get_registry()
