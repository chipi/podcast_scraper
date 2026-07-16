"""``podcast serve`` wires proxy/forwarded-header handling (#1163).

Behind the Caddy edge → nginx → uvicorn chain, ``request.url_for`` (the OAuth callback)
must resolve to the public https origin. That needs uvicorn's ``proxy_headers`` +
``forwarded_allow_ips`` — this locks that the serve command passes them, and honors the
``FORWARDED_ALLOW_IPS`` env (default loopback; ``*`` on a proxied deployment).
"""

from __future__ import annotations

from argparse import Namespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

import logging

from podcast_scraper.server import cli_handlers

pytestmark = [pytest.mark.integration]


def _run(monkeypatch, tmp_path, forwarded_env: str | None, *, reload: bool = False) -> dict:
    captured: dict = {}

    def _fake_run(app, **kwargs):  # noqa: ANN001
        captured.update(kwargs)
        return None

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", _fake_run)
    if forwarded_env is None:
        monkeypatch.delenv("FORWARDED_ALLOW_IPS", raising=False)
    else:
        monkeypatch.setenv("FORWARDED_ALLOW_IPS", forwarded_env)

    args = Namespace(
        output_dir=str(tmp_path), host="0.0.0.0", port=8000, reload=reload, no_static=True
    )
    rc = cli_handlers.run_serve(args, logging.getLogger("test"))
    assert rc == 0
    return captured


def test_serve_passes_proxy_headers_default_loopback(monkeypatch, tmp_path) -> None:
    kw = _run(monkeypatch, tmp_path, forwarded_env=None)
    assert kw.get("proxy_headers") is True
    assert kw.get("forwarded_allow_ips") == "127.0.0.1"


def test_serve_honors_forwarded_allow_ips_env(monkeypatch, tmp_path) -> None:
    kw = _run(monkeypatch, tmp_path, forwarded_env="*")
    assert kw.get("proxy_headers") is True
    assert kw.get("forwarded_allow_ips") == "*"


def test_serve_reload_mode_also_passes_proxy_headers(monkeypatch, tmp_path) -> None:
    # The --reload branch (factory mode) must carry the same proxy/forwarded wiring —
    # a refactor that drops it from the reload path would otherwise pass silently.
    kw = _run(monkeypatch, tmp_path, forwarded_env="*", reload=True)
    assert kw.get("reload") is True
    assert kw.get("factory") is True
    assert kw.get("proxy_headers") is True
    assert kw.get("forwarded_allow_ips") == "*"
