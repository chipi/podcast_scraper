"""The prod obs config reads Umami per frontend, with a deploy-rendered URL (never committed)."""

from __future__ import annotations

import pytest

from podcast_obs.config import ObservabilityConfig

pytestmark = pytest.mark.unit


def test_prod_config_has_per_frontend_sites_and_rendered_url(monkeypatch) -> None:
    monkeypatch.setenv("PODCAST_OBS_UMAMI_URL", "https://umami.example.ts.net")
    monkeypatch.setenv("PODCAST_OBS_UMAMI_PASSWORD", "pw")
    t = ObservabilityConfig.from_yaml("config/observability.prod.yaml").target("prod")
    assert t.umami_url == "https://umami.example.ts.net"
    assert t.umami_username == "podcast-obs"
    assert t.umami_password == "pw"
    assert set(t.umami_website_ids) == {"player", "operator"}
    assert t.umami_website_ids["player"] != t.umami_website_ids["operator"]


def test_without_the_rendered_url_umami_stays_unconfigured(monkeypatch) -> None:
    monkeypatch.delenv("PODCAST_OBS_UMAMI_URL", raising=False)
    t = ObservabilityConfig.from_yaml("config/observability.prod.yaml").target("prod")
    assert t.umami_url is None
