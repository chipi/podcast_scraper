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
    "profile",
    [
        "local_dgx_balanced",
        "local_dgx_full",
        "prod_dgx_balanced",
        "prod_dgx_full_with_fallback",
        "cloud_with_dgx_primary",
        "dev",
        "local",
    ],
)
def test_dgx_dev_local_profiles_get_full_stack(profile: str) -> None:
    """Every profile with a real LLM somewhere gets the full enricher set.
    'prod' is intentionally NOT here — there is no config/profiles/prod.yaml;
    the production profiles are prod_dgx_*."""
    s = enricher_set_for_profile(profile)
    assert "nli_contradiction" in s.enabled_enrichers


def test_phantom_prod_profile_is_unknown() -> None:
    """A 'prod' literal in the matrix without a corresponding YAML used to
    be dead code (chunk-8 sweep finding #3). The matrix now treats
    unknown 'prod' as the conservative empty set."""
    s = enricher_set_for_profile("prod")
    assert s.enabled_enrichers == []


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
    """extra_opt_in MUST name an enricher that's in the active set —
    opting in to an enricher that won't run is almost always a typo
    (A3 fix)."""
    base = EnricherSet(enabled_enrichers=["a", "b"], opt_in_flags={"a": True})
    out = apply_cli_overrides(base, extra_opt_in=["b"])
    assert out.opt_in_flags == {"a": True, "b": True}


def test_extra_opt_in_for_unregistered_enricher_raises() -> None:
    """A3 follow-up: --opt-in nly_contradiction (typo) should fail loudly
    instead of silently setting a flag for an enricher that won't run."""
    from podcast_scraper.enrichment.profile_sets import UnknownOptInError

    base = EnricherSet(enabled_enrichers=["a"])
    with pytest.raises(UnknownOptInError):
        apply_cli_overrides(base, extra_opt_in=["nly_contradiction"])


def test_overrides_preserve_per_enricher_config() -> None:
    base = EnricherSet(
        enabled_enrichers=["a"],
        per_enricher_config={"a": {"threshold": 0.7}},
    )
    out = apply_cli_overrides(base, only=["a"])
    assert out.per_enricher_config == {"a": {"threshold": 0.7}}


def test_empty_only_list_is_a_noop() -> None:
    """only=[] (empty list, not None) must not filter — empty restriction
    would clear the active set silently."""
    base = EnricherSet(enabled_enrichers=["a", "b", "c"])
    out = apply_cli_overrides(base, only=[])
    assert out.enabled_enrichers == ["a", "b", "c"]


def test_empty_skip_list_is_a_noop() -> None:
    """skip=[] is a no-op too — the active set is preserved verbatim."""
    base = EnricherSet(enabled_enrichers=["a", "b"])
    out = apply_cli_overrides(base, skip=[])
    assert out.enabled_enrichers == ["a", "b"]


def test_skip_with_id_not_in_base_is_a_noop() -> None:
    """Operator can skip an enricher that wasn't active — should not raise."""
    base = EnricherSet(enabled_enrichers=["a"])
    out = apply_cli_overrides(base, skip=["zzz"])
    assert out.enabled_enrichers == ["a"]


def test_no_enrichers_beats_only_and_extra_opt_in() -> None:
    """no_enrichers wins over everything — should not raise on extra_opt_in
    either, since the active set is empty by construction."""
    base = EnricherSet(enabled_enrichers=["a", "b"])
    out = apply_cli_overrides(base, only=["a"], extra_opt_in=["zzz"], no_enrichers=True)
    assert out.enabled_enrichers == []
    assert out.opt_in_flags == {}


def test_only_with_empty_base_force_includes() -> None:
    """No profile + ``--only a,b,c`` runs a,b,c rather than silently no-op'ing.

    The pre-fix CLI failure mode was: with no ``--profile`` and no YAML, the
    base set is empty; ``--only`` would *filter* an empty set, leaving zero
    enrichers active, and the CLI would print ``status=ok duration_ms=0``
    without running anything. The natural read of ``--only a,b,c`` is "run
    a,b,c", so when there is no base to filter we treat it as the set.
    """
    base = EnricherSet()  # no YAML, no profile — what the bare CLI gives
    out = apply_cli_overrides(base, only=["topic_cooccurrence", "grounding_rate"])
    assert out.enabled_enrichers == ["topic_cooccurrence", "grounding_rate"]


def test_only_filters_when_base_is_non_empty() -> None:
    """The force-include path does NOT change profile + --only filtering."""
    base = EnricherSet(enabled_enrichers=["a", "b", "c"])
    out = apply_cli_overrides(base, only=["a", "c", "missing"])
    # ``missing`` not in base → dropped (this is filter semantics).
    assert out.enabled_enrichers == ["a", "c"]


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


def test_every_real_profile_yaml_lands_in_an_explicit_branch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Stronger drift gate: every profile YAML must hit a known branch in
    ``enricher_set_for_profile()``. If a profile YAML falls through to the
    unknown-profile WARNING, this test fails — that's exactly the drift the
    A9 follow-up is supposed to catch.
    """
    caplog.clear()
    with caplog.at_level("WARNING", logger="podcast_scraper.enrichment.profile_sets"):
        for name in discover_profile_yaml_names():
            enricher_set_for_profile(name)
    unknown_warnings = [rec for rec in caplog.records if "unknown profile" in rec.getMessage()]
    assert not unknown_warnings, (
        f"profile YAML(s) without an explicit enricher_set_for_profile() branch: "
        f"{[w.getMessage() for w in unknown_warnings]}"
    )


def test_yaml_enrichment_block_matches_python_matrix() -> None:
    """The advisory ``enrichment.enrichers`` dict in each
    ``config/profiles/*.yaml`` MUST declare the same enrichers as the
    Python matrix returns. The Python matrix is the source of truth;
    the YAML block is the operator-facing mirror. Drift between them
    is a bug.

    Shape B (RFC-088 v2) accepts either:
      - the legacy list-of-ids form (``enrichers: [id1, id2]``)
      - the dict-of-blocks form (``enrichers: {id1: {...}, id2: {}}``)
    The drift check normalises both to a set of ids before comparing.
    """
    import yaml

    profiles_dir = Path(__file__).resolve().parents[3] / "config" / "profiles"
    for name in discover_profile_yaml_names(profiles_dir):
        yaml_path = profiles_dir / f"{name}.yaml"
        body = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        yaml_block = body.get("enrichment")
        if yaml_block is None:
            continue
        raw = yaml_block.get("enrichers") or {}
        if isinstance(raw, dict):
            yaml_set = set(raw.keys())
        elif isinstance(raw, list):
            yaml_set = set(raw)
        else:
            yaml_set = set()
        py_set = set(enricher_set_for_profile(name).enabled_enrichers)
        assert yaml_set == py_set, (
            f"profile {name!r}: YAML enrichment.enrichers ({sorted(yaml_set)}) "
            f"differs from Python matrix ({sorted(py_set)})"
        )


def test_top_level_profiles_have_an_enrichment_block() -> None:
    """Every top-level profile YAML (under config/profiles/, not the
    freeze/ subdirectory or *.example.yaml) must declare an
    `enrichment:` block — operators read these YAMLs to discover what's
    active, so the block must be present even when the matrix says
    `enabled: false`.
    """
    import yaml

    profiles_dir = Path(__file__).resolve().parents[3] / "config" / "profiles"
    missing: list[str] = []
    for name in discover_profile_yaml_names(profiles_dir):
        yaml_path = profiles_dir / f"{name}.yaml"
        body = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        if body.get("enrichment") is None:
            missing.append(name)
    assert not missing, f"profiles missing `enrichment:` block: {missing}"
