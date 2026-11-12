"""An installed extension declares its own settings (ADR-182): flat keys route to its model."""

from __future__ import annotations

import logging

import pytest
from pydantic import BaseModel, ConfigDict

from podcast_scraper import config as config_mod
from podcast_scraper.config import Config, config_fields, config_fields_set, config_value
from podcast_scraper.extensions import Extension, use_extensions


class _ProbeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    probe_threshold: float = 0.5
    probe_label: str = "off"


_PROBE = Extension(name="probe", config_model=_ProbeSettings)
_RSS = "https://example.com/feed.xml"


def _cfg(**settings: object) -> Config:
    return Config.model_validate({"rss_url": _RSS, **settings})


@pytest.fixture
def probe_installed():
    with use_extensions([_PROBE]):
        yield


@pytest.fixture
def nothing_installed():
    with use_extensions([]):
        yield


def test_a_flat_key_routes_to_the_extension_model(probe_installed) -> None:
    cfg = _cfg(probe_threshold=0.8)
    assert cfg.extension("probe").probe_threshold == 0.8
    assert cfg.extension("probe").probe_label == "off"


def test_defaults_when_the_extension_key_is_not_set(probe_installed) -> None:
    cfg = _cfg()
    assert cfg.extension("probe") == _ProbeSettings()


def test_a_profile_layer_key_routes_too(probe_installed, monkeypatch) -> None:
    real = config_mod.resolve_profile_layers

    def layers(name, **kwargs):
        if name == "probe_profile":
            return {"probe_threshold": 0.9}, True
        return real(name, **kwargs)

    monkeypatch.setattr(config_mod, "resolve_profile_layers", layers)
    cfg = _cfg(profile="probe_profile")
    assert cfg.extension("probe").probe_threshold == 0.9
    explicit = _cfg(profile="probe_profile", probe_threshold=0.1)
    assert explicit.extension("probe").probe_threshold == 0.1


def _registry_preset(monkeypatch) -> None:
    from podcast_scraper.providers.ml import model_registry

    def preset(name, **kwargs):
        return {"probe_threshold": 0.4, "_meta": 1, "no_such_field": 2}

    monkeypatch.setattr(model_registry, "resolve_profile_to_settings", preset)


def test_a_registry_preset_keeps_an_installed_extension_key(probe_installed, monkeypatch) -> None:
    _registry_preset(monkeypatch)
    assert config_mod.resolve_profile_layers("no_yaml_preset") == ({"probe_threshold": 0.4}, True)


def test_a_registry_preset_drops_the_key_without_the_extension(
    nothing_installed, monkeypatch
) -> None:
    _registry_preset(monkeypatch)
    assert config_mod.resolve_profile_layers("no_yaml_preset") == ({}, False)


def test_an_unknown_extension_is_a_lookup_error(probe_installed) -> None:
    with pytest.raises(LookupError, match="probe"):
        _cfg().extension("absent")


def test_the_old_attribute_read_still_answers_and_warns_once(probe_installed, caplog) -> None:
    config_mod._warned_moved_reads.discard("probe_label")
    cfg = _cfg(probe_label="on")
    with caplog.at_level(logging.WARNING, logger="podcast_scraper.config"):
        assert cfg.probe_label == "on"
        assert cfg.probe_label == "on"
    warned = [r for r in caplog.records if "cfg.probe_label" in r.getMessage()]
    assert len(warned) == 1
    assert "cfg.extension('probe').probe_label" in warned[0].getMessage()


def test_a_missing_attribute_is_still_an_attribute_error(probe_installed) -> None:
    with pytest.raises(AttributeError):
        _cfg().no_such_setting  # noqa: B018


def test_model_dump_is_flat_and_round_trips(probe_installed) -> None:
    cfg = _cfg(probe_threshold=0.7)
    dumped = cfg.model_dump()
    assert dumped["probe_threshold"] == 0.7
    assert "extension_settings" not in dumped
    again = Config.model_validate(dumped)
    assert again.extension("probe").probe_threshold == 0.7
    assert "probe_threshold" not in cfg.model_dump(exclude={"probe_threshold"})
    assert cfg.model_dump(include={"probe_threshold"}) == {"probe_threshold": 0.7}


def test_model_copy_updates_extension_and_public_keys(probe_installed) -> None:
    cfg = _cfg()
    new = cfg.model_copy(update={"probe_threshold": 0.3, "max_episodes": 2})
    assert new.extension("probe").probe_threshold == 0.3
    assert new.max_episodes == 2
    assert cfg.extension("probe").probe_threshold == 0.5
    assert Config.model_validate(new.model_dump()).extension("probe").probe_threshold == 0.3


def test_an_unknown_key_is_refused_naming_the_installed_extensions(probe_installed) -> None:
    with pytest.raises(ValueError, match=r"no_such_key.*probe"):
        _cfg(no_such_key=1)


def test_an_extension_key_is_refused_without_the_extension(nothing_installed) -> None:
    with pytest.raises(ValueError, match="probe_threshold"):
        _cfg(probe_threshold=0.8)


def test_config_fields_and_values_cover_the_extension(probe_installed) -> None:
    assert "probe_threshold" in config_fields()
    assert "extension_settings" not in config_fields()
    cfg = _cfg(probe_threshold=0.2)
    assert config_value(cfg, "probe_threshold") == 0.2
    assert config_value(cfg, "max_episodes") == cfg.max_episodes
    assert config_value(cfg, "absent", "dflt") == "dflt"
    assert "probe_threshold" in config_fields_set(cfg)
    assert "probe_label" not in config_fields_set(cfg)


def test_nothing_changes_without_an_extension_model(nothing_installed) -> None:
    cfg = _cfg()
    with pytest.raises(LookupError):
        cfg.extension("probe")
    assert "extension_settings" not in cfg.model_dump()
    assert set(cfg.model_dump()) == set(Config.model_fields) - {"extension_settings"}
    assert set(config_fields()) == set(Config.model_fields) - {"extension_settings"}
