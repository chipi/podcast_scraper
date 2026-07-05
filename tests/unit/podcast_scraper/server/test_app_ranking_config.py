"""Unit tests for the ranking-signal registry (``app_ranking_config``).

The registry is the single source of truth for discovery/digest ranking: every signal is
independently on/off + weight-tunable, parsed total-ly from a config dict so a bad override can
never empty ranking or silently drop a signal the composition relies on.
"""

from __future__ import annotations

from podcast_scraper.server.app_ranking_config import (
    DEFAULT_RANKING_CONFIG,
    ranking_config_from_dict,
    ranking_config_to_dict,
    SIGNAL_INTEREST_AFFINITY,
    SIGNAL_SIGNIFICANCE,
    SIGNAL_TREND_VELOCITY,
)


def test_default_has_the_four_known_signals() -> None:
    names = [s.name for s in DEFAULT_RANKING_CONFIG.signals]
    assert names == ["significance", "interest_affinity", "trend_velocity", "recency"]


def test_trend_velocity_defaults_off() -> None:
    assert DEFAULT_RANKING_CONFIG.is_enabled(SIGNAL_TREND_VELOCITY) is False
    # Disabled → effective weight 0 regardless of the configured weight.
    assert DEFAULT_RANKING_CONFIG.weight_of(SIGNAL_TREND_VELOCITY) == 0.0


def test_weight_of_enabled_returns_configured_weight() -> None:
    assert DEFAULT_RANKING_CONFIG.weight_of(SIGNAL_INTEREST_AFFINITY) == 2.0


def test_weight_of_unknown_signal_is_default() -> None:
    assert DEFAULT_RANKING_CONFIG.weight_of("nope", default=1.5) == 1.5


def test_params_of_significance() -> None:
    p = DEFAULT_RANKING_CONFIG.params_of(SIGNAL_SIGNIFICANCE)
    assert p["gi_bonus"] == 2.0 and p["bullet_cap"] == 5


def test_from_dict_merges_partial_override_onto_defaults() -> None:
    # Only trend_velocity supplied → it flips on, the other three signals keep their defaults.
    cfg = ranking_config_from_dict(
        {
            "signals": [
                {"name": "trend_velocity", "enabled": True, "weight": 0.9, "params": {"cap": 2.0}}
            ]
        }
    )
    assert cfg.is_enabled("trend_velocity") is True
    assert cfg.weight_of("trend_velocity") == 0.9
    assert cfg.params_of("trend_velocity")["cap"] == 2.0
    # Untouched signals survive the merge.
    assert cfg.weight_of(SIGNAL_INTEREST_AFFINITY) == 2.0
    assert [s.name for s in cfg.signals] == [s.name for s in DEFAULT_RANKING_CONFIG.signals]


def test_from_dict_falls_back_on_garbage() -> None:
    assert ranking_config_from_dict(None) is DEFAULT_RANKING_CONFIG
    assert ranking_config_from_dict({"signals": "nope"}) is DEFAULT_RANKING_CONFIG
    assert ranking_config_from_dict({"signals": []}) is DEFAULT_RANKING_CONFIG
    # A wholly invalid list yields no overrides → identity fallback.
    assert ranking_config_from_dict({"signals": [42, {"noname": 1}]}) is DEFAULT_RANKING_CONFIG


def test_from_dict_defaults_missing_fields_from_base() -> None:
    # Only 'enabled' given → weight + params inherited from the default signal.
    cfg = ranking_config_from_dict({"signals": [{"name": "interest_affinity", "enabled": False}]})
    sig = cfg.get("interest_affinity")
    assert sig is not None and sig.enabled is False and sig.weight == 2.0


def test_from_dict_bad_weight_falls_back_to_base() -> None:
    cfg = ranking_config_from_dict({"signals": [{"name": "interest_affinity", "weight": "high"}]})
    assert cfg.weight_of("interest_affinity") == 2.0


def test_from_dict_appends_unknown_signal() -> None:
    cfg = ranking_config_from_dict(
        {"signals": [{"name": "novelty", "enabled": True, "weight": 1.0}]}
    )
    assert [s.name for s in cfg.signals][-1] == "novelty"
    assert cfg.weight_of("novelty") == 1.0


def test_to_dict_roundtrips() -> None:
    d = ranking_config_to_dict(DEFAULT_RANKING_CONFIG)
    cfg = ranking_config_from_dict(d)
    assert ranking_config_to_dict(cfg) == d
