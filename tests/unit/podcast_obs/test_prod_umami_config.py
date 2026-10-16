"""A prod-shaped obs config reads Umami per frontend, with a deploy-rendered URL."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from podcast_obs.config import ObservabilityConfig

pytestmark = pytest.mark.unit

_PROD_SHAPED = textwrap.dedent("""\
    default_target: prod
    targets:
      prod:
        umami:
          username: podcast-obs
          password_env: PODCAST_OBS_UMAMI_PASSWORD
          website_ids:
            player: 11111111-1111-1111-1111-111111111111
            operator: 22222222-2222-2222-2222-222222222222
    """)


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "observability.prod.yaml"
    path.write_text(_PROD_SHAPED, encoding="utf-8")
    return path


def test_prod_config_has_per_frontend_sites_and_rendered_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PODCAST_OBS_UMAMI_URL", "https://umami.example.ts.net")
    monkeypatch.setenv("PODCAST_OBS_UMAMI_PASSWORD", "pw")
    t = ObservabilityConfig.from_yaml(_config(tmp_path)).target("prod")
    assert t.umami_url == "https://umami.example.ts.net"
    assert t.umami_username == "podcast-obs"
    assert t.umami_password == "pw"
    assert set(t.umami_website_ids) == {"player", "operator"}
    assert t.umami_website_ids["player"] != t.umami_website_ids["operator"]


def test_without_the_rendered_url_umami_stays_unconfigured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PODCAST_OBS_UMAMI_URL", raising=False)
    t = ObservabilityConfig.from_yaml(_config(tmp_path)).target("prod")
    assert t.umami_url is None
