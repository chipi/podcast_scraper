"""Unit tests for packaged profile name listing (operator-config API)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_list_packaged_profile_names_includes_repo_profiles() -> None:
    from podcast_scraper.server import profile_presets

    names = profile_presets.list_packaged_profile_names()
    assert len(names) >= 3
    assert names == sorted(names)
    assert "cloud_balanced" in names
    assert "profile_freeze" not in names  # profile_freeze.example stem ends with .example


def test_profile_directories_prefers_cwd_then_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from podcast_scraper.server import profile_presets

    cwd_prof = tmp_path / "config" / "profiles"
    cwd_prof.mkdir(parents=True)
    (cwd_prof / "from_cwd_only.yaml").write_text("x: 1\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    names = profile_presets.list_packaged_profile_names()
    assert "from_cwd_only" in names


# --- PODCAST_AVAILABLE_PROFILES env allowlist (#692, RFC-081) -------------------


def test_env_allowlist_unset_returns_full_disk_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No env var set: behaves identically to before (every on-disk profile)."""
    from podcast_scraper.server import profile_presets

    monkeypatch.delenv("PODCAST_AVAILABLE_PROFILES", raising=False)
    names = profile_presets.list_packaged_profile_names()
    assert "cloud_thin" in names
    assert "cloud_balanced" in names
    # Sanity: at least 3 packaged profiles ship in the repo.
    assert len(names) >= 3


def test_env_allowlist_blank_treated_as_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whitespace-only env value = no filtering (forgiving operator UX)."""
    from podcast_scraper.server import profile_presets

    monkeypatch.setenv("PODCAST_AVAILABLE_PROFILES", "   ")
    names = profile_presets.list_packaged_profile_names()
    assert "cloud_thin" in names
    assert "cloud_balanced" in names


def test_env_allowlist_filters_to_intersection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env allowlist intersects with on-disk profile names — preprod default case."""
    from podcast_scraper.server import profile_presets

    monkeypatch.setenv("PODCAST_AVAILABLE_PROFILES", "cloud_thin")
    names = profile_presets.list_packaged_profile_names()
    assert names == ["cloud_thin"]


def test_env_allowlist_handles_commas_and_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trailing commas / spacing don't change the resolved allowlist."""
    from podcast_scraper.server import profile_presets

    monkeypatch.setenv(
        "PODCAST_AVAILABLE_PROFILES",
        " cloud_thin , cloud_balanced , ",
    )
    names = profile_presets.list_packaged_profile_names()
    assert names == ["cloud_balanced", "cloud_thin"]


def test_env_allowlist_typo_does_not_advertise_nonexistent_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the operator typos a profile name, it stays out of the dropdown.

    Defense in depth: the intersection is taken AFTER on-disk discovery, so
    ``cloud_quaility`` (typo) doesn't appear even though the env says so —
    keeps the API contract honest.
    """
    from podcast_scraper.server import profile_presets

    monkeypatch.setenv("PODCAST_AVAILABLE_PROFILES", "cloud_thin,cloud_quaility")
    names = profile_presets.list_packaged_profile_names()
    assert names == ["cloud_thin"]


def test_env_allowlist_all_typos_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allowlist of only-typos returns empty list (operator UI shows no options)."""
    from podcast_scraper.server import profile_presets

    monkeypatch.setenv("PODCAST_AVAILABLE_PROFILES", "ttt,uuu")
    names = profile_presets.list_packaged_profile_names()
    assert names == []


# --- validate_operator_profile_allowed defense-in-depth (#692) -----------------


def test_validate_profile_allowed_no_file_returns_none(tmp_path: Path) -> None:
    """Missing operator file → ``None`` (lets ``Config._resolve_profile`` fall back)."""
    from podcast_scraper.server import profile_presets

    op = tmp_path / "viewer_operator.yaml"
    assert profile_presets.validate_operator_profile_allowed(op) is None


def test_validate_profile_allowed_no_profile_line_returns_none(tmp_path: Path) -> None:
    """Operator file present but no ``profile:`` line → ``None`` (no-op default)."""
    from podcast_scraper.server import profile_presets

    op = tmp_path / "viewer_operator.yaml"
    op.write_text("max_episodes: 1\n", encoding="utf-8")
    assert profile_presets.validate_operator_profile_allowed(op) is None


def test_validate_profile_allowed_blank_profile_line_returns_none(tmp_path: Path) -> None:
    """``profile:`` line with empty value treated as no-op default."""
    from podcast_scraper.server import profile_presets

    op = tmp_path / "viewer_operator.yaml"
    op.write_text('profile: ""\nmax_episodes: 1\n', encoding="utf-8")
    assert profile_presets.validate_operator_profile_allowed(op) is None


def test_validate_profile_allowed_in_allowlist_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``cloud_thin`` allowed in preprod → returns the validated name."""
    from podcast_scraper.server import profile_presets

    monkeypatch.setenv("PODCAST_AVAILABLE_PROFILES", "cloud_thin")
    op = tmp_path / "viewer_operator.yaml"
    op.write_text("profile: cloud_thin\nmax_episodes: 1\n", encoding="utf-8")
    assert profile_presets.validate_operator_profile_allowed(op) == "cloud_thin"


def test_validate_profile_allowed_outside_allowlist_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``cloud_quality`` outside the preprod allowlist → ValueError + helpful detail."""
    from podcast_scraper.server import profile_presets

    monkeypatch.setenv("PODCAST_AVAILABLE_PROFILES", "cloud_thin")
    op = tmp_path / "viewer_operator.yaml"
    op.write_text("profile: cloud_quality\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        profile_presets.validate_operator_profile_allowed(op)
    detail = str(exc_info.value)
    assert "cloud_quality" in detail
    assert "cloud_thin" in detail  # allowed list shown so operator can re-pick


def test_validate_profile_allowed_no_env_no_filtering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No env allowlist (dev / CI default) → any on-disk profile passes."""
    from podcast_scraper.server import profile_presets

    monkeypatch.delenv("PODCAST_AVAILABLE_PROFILES", raising=False)
    op = tmp_path / "viewer_operator.yaml"
    op.write_text("profile: cloud_quality\n", encoding="utf-8")
    assert profile_presets.validate_operator_profile_allowed(op) == "cloud_quality"


def test_validate_profile_allowed_typo_profile_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Operator typo'd profile name (not on disk) → ValueError even when no env set."""
    from podcast_scraper.server import profile_presets

    monkeypatch.delenv("PODCAST_AVAILABLE_PROFILES", raising=False)
    op = tmp_path / "viewer_operator.yaml"
    op.write_text("profile: clud_thin\n", encoding="utf-8")  # typo
    with pytest.raises(ValueError) as exc_info:
        profile_presets.validate_operator_profile_allowed(op)
    assert "clud_thin" in str(exc_info.value)


# --- env_default_profile (#692 default-profile concept) ----------------------


def test_default_profile_unset_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """No env var set: default-profile is None (dev / CI default)."""
    from podcast_scraper.server import profile_presets

    monkeypatch.delenv("PODCAST_DEFAULT_PROFILE", raising=False)
    monkeypatch.delenv("PODCAST_AVAILABLE_PROFILES", raising=False)
    assert profile_presets.env_default_profile() is None


def test_default_profile_blank_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whitespace-only env value treated as unset."""
    from podcast_scraper.server import profile_presets

    monkeypatch.setenv("PODCAST_DEFAULT_PROFILE", "   ")
    assert profile_presets.env_default_profile() is None


def test_default_profile_resolves_when_in_packaged_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``cloud_thin`` is on disk → returns ``cloud_thin``."""
    from podcast_scraper.server import profile_presets

    monkeypatch.delenv("PODCAST_AVAILABLE_PROFILES", raising=False)
    monkeypatch.setenv("PODCAST_DEFAULT_PROFILE", "cloud_thin")
    assert profile_presets.env_default_profile() == "cloud_thin"


def test_default_profile_filtered_by_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default must also pass the allowlist filter — no fallback to a hidden profile.

    If allowlist excludes the requested default (operator misconfiguration),
    return ``None`` rather than serving a profile the operator UI deliberately
    hides. Better than half-broken UX where dropdown filters but server still
    falls back to the disallowed profile.
    """
    from podcast_scraper.server import profile_presets

    monkeypatch.setenv("PODCAST_AVAILABLE_PROFILES", "cloud_thin")
    monkeypatch.setenv("PODCAST_DEFAULT_PROFILE", "cloud_balanced")
    assert profile_presets.env_default_profile() is None


def test_default_profile_typo_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Typo'd default name (not on disk) → None (no surprise fallback)."""
    from podcast_scraper.server import profile_presets

    monkeypatch.delenv("PODCAST_AVAILABLE_PROFILES", raising=False)
    monkeypatch.setenv("PODCAST_DEFAULT_PROFILE", "cloud_thinn")  # typo
    assert profile_presets.env_default_profile() is None
