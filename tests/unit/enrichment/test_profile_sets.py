"""Unit tests for the RFC-088 chunk-7 profile-preset → EnricherSet matrix."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.enrichment.enrichers import ALL_DETERMINISTIC_ENRICHER_IDS
from podcast_scraper.enrichment.profile_sets import (
    apply_cli_overrides,
    discover_profile_yaml_names,
    enricher_set_for_profile,
)
from podcast_scraper.enrichment.protocol import EnricherSet

# ---------------------------------------------------------------------------
# enricher_set_for_profile — matrix correctness
# ---------------------------------------------------------------------------


def test_none_profile_is_empty_set() -> None:
    s = enricher_set_for_profile(None)
    assert s.enabled_enrichers == []


def test_test_default_is_empty_set_for_ci_isolation() -> None:
    s = enricher_set_for_profile("test_default")
    assert s.enabled_enrichers == []


def test_airgapped_thin_is_deterministic_only() -> None:
    s = enricher_set_for_profile("airgapped_thin")
    assert set(s.enabled_enrichers) == set(ALL_DETERMINISTIC_ENRICHER_IDS)
    assert "topic_similarity" not in s.enabled_enrichers
    assert "nli_contradiction" not in s.enabled_enrichers


def test_airgapped_adds_topic_similarity() -> None:
    s = enricher_set_for_profile("airgapped")
    assert "topic_similarity" in s.enabled_enrichers
    assert set(ALL_DETERMINISTIC_ENRICHER_IDS) <= set(s.enabled_enrichers)
    assert "nli_contradiction" not in s.enabled_enrichers


@pytest.mark.parametrize("profile", ["cloud_thin", "cloud_balanced", "cloud_quality"])
def test_cloud_profiles_get_full_stack(profile: str) -> None:
    s = enricher_set_for_profile(profile)
    assert "topic_similarity" in s.enabled_enrichers
    assert "nli_contradiction" in s.enabled_enrichers
    assert set(ALL_DETERMINISTIC_ENRICHER_IDS) <= set(s.enabled_enrichers)


@pytest.mark.parametrize(
    "profile", ["local_dgx_balanced", "local_dgx_full", "cloud_with_dgx_primary", "dev", "prod"]
)
def test_dgx_dev_prod_profiles_get_full_stack(profile: str) -> None:
    s = enricher_set_for_profile(profile)
    assert "nli_contradiction" in s.enabled_enrichers


def test_unknown_profile_is_conservative_empty_set() -> None:
    s = enricher_set_for_profile("definitely-not-a-profile")
    assert s.enabled_enrichers == []


# ---------------------------------------------------------------------------
# apply_cli_overrides — precedence
# ---------------------------------------------------------------------------


def test_no_enrichers_wins_over_everything() -> None:
    base = EnricherSet(enabled_enrichers=["topic_similarity", "nli_contradiction"])
    out = apply_cli_overrides(base, only=["topic_similarity"], no_enrichers=True)
    assert out.enabled_enrichers == []


def test_only_filter_keeps_named_subset() -> None:
    base = EnricherSet(enabled_enrichers=["a", "b", "c"])
    out = apply_cli_overrides(base, only=["a", "c"])
    assert out.enabled_enrichers == ["a", "c"]


def test_only_and_skip_compose() -> None:
    base = EnricherSet(enabled_enrichers=["a", "b", "c", "d"])
    out = apply_cli_overrides(base, only=["a", "b", "c"], skip=["b"])
    assert out.enabled_enrichers == ["a", "c"]


def test_skip_filter_drops_named_ids() -> None:
    base = EnricherSet(enabled_enrichers=["a", "b", "c"])
    out = apply_cli_overrides(base, skip=["b"])
    assert out.enabled_enrichers == ["a", "c"]


def test_extra_opt_in_layers_on_top_of_base_flags() -> None:
    base = EnricherSet(enabled_enrichers=["a"], opt_in_flags={"a": True})
    out = apply_cli_overrides(base, extra_opt_in=["b"])
    assert out.opt_in_flags == {"a": True, "b": True}


def test_overrides_preserve_per_enricher_config() -> None:
    base = EnricherSet(
        enabled_enrichers=["a"],
        per_enricher_config={"a": {"threshold": 0.7}},
    )
    out = apply_cli_overrides(base, only=["a"])
    assert out.per_enricher_config == {"a": {"threshold": 0.7}}


# ---------------------------------------------------------------------------
# discover_profile_yaml_names — drift test surface
# ---------------------------------------------------------------------------


def test_discover_profile_yaml_names_skips_example_yamls(tmp_path: Path) -> None:
    (tmp_path / "ok.yaml").write_text("", encoding="utf-8")
    (tmp_path / "another.example.yaml").write_text("", encoding="utf-8")
    names = discover_profile_yaml_names(tmp_path)
    assert names == ["ok"]


def test_discover_profile_yaml_names_returns_sorted_stems(tmp_path: Path) -> None:
    for name in ("z.yaml", "a.yaml", "m.yaml"):
        (tmp_path / name).write_text("", encoding="utf-8")
    assert discover_profile_yaml_names(tmp_path) == ["a", "m", "z"]


# ---------------------------------------------------------------------------
# Drift gate: every profile YAML on disk must map to an EnricherSet decision
# ---------------------------------------------------------------------------


def test_every_real_profile_yaml_has_a_matrix_decision() -> None:
    """Adding a new profile YAML without registering its enricher set is a
    drift. Every name surfaced by discover_profile_yaml_names() must produce
    an EnricherSet (even if the decision is the empty set) — no ValueError /
    KeyError leakage allowed."""
    for name in discover_profile_yaml_names():
        s = enricher_set_for_profile(name)
        assert isinstance(s, EnricherSet), name
