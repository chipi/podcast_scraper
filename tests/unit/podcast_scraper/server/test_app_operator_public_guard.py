"""Unit coverage for the operator-public open-signup boot guard (app.py, RFC-108).

``create_app`` refuses to boot the public operator surface under ``APP_SIGNUP_MODE=open``:
the viewer self-grants ``creator`` via its ``?grant=creator`` login hint, so the email
allowlist is the only authZ boundary on the operator-read corpus — one env flip must not
open the whole surface. The integration serve-mode tests cover this end-to-end through
``create_app``; this exercises the guard directly at unit tier (codecov's patch coverage
only sees the unit upload).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.server.app import _guard_operator_public_open_signup, create_app


def test_guard_raises_under_operator_public_open_signup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_SIGNUP_MODE", "open")
    with pytest.raises(RuntimeError, match="APP_SIGNUP_MODE=open"):
        _guard_operator_public_open_signup(True)


def test_guard_allows_operator_public_with_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_SIGNUP_MODE", "allowlist")
    _guard_operator_public_open_signup(True)  # must not raise


def test_guard_ignores_open_when_not_operator_public(monkeypatch: pytest.MonkeyPatch) -> None:
    # ``open`` signup is legitimate for the non-operator surfaces — the guard is
    # operator-public-only, so it must NOT raise when operator_public is False.
    monkeypatch.setenv("APP_SIGNUP_MODE", "open")
    _guard_operator_public_open_signup(False)  # must not raise


def test_create_app_wires_the_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard is actually called from create_app (covers the wiring at the call site):
    open signup under operator-public refuses to boot; allowlist boots fine."""
    monkeypatch.delenv("PODCAST_SERVE_APP_ONLY", raising=False)
    monkeypatch.setenv("PODCAST_SERVE_OPERATOR_PUBLIC", "1")

    monkeypatch.setenv("APP_SIGNUP_MODE", "open")
    with pytest.raises(RuntimeError, match="APP_SIGNUP_MODE=open"):
        create_app(output_dir=tmp_path)

    monkeypatch.setenv("APP_SIGNUP_MODE", "allowlist")
    create_app(output_dir=tmp_path)  # must not raise
