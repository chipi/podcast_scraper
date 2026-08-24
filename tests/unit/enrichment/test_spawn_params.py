"""``derive_enrichment_job_params`` — profile-driven --profile/--with-ml (RFC-118).

The operator-triggered surfaces (HTTP route, MCP reenrich) must derive the child's
profile and ML wiring from the operator YAML exactly like the pipeline auto-chain,
or a force re-derive silently warn-skips the ML pair.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.enrichment.spawn_params import derive_enrichment_job_params

pytestmark = pytest.mark.unit


def _yaml(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "viewer_operator.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_none_yaml_fails_open():
    assert derive_enrichment_job_params(None) == (None, False)


def test_missing_file_fails_open(tmp_path):
    assert derive_enrichment_job_params(tmp_path / "absent.yaml") == (None, False)


def test_ml_profile_drives_with_ml(tmp_path):
    # airgapped enables topic_similarity (provider_requirement) → --with-ml.
    profile, with_ml = derive_enrichment_job_params(
        _yaml(tmp_path, "profile: airgapped\nenrichment:\n  enabled: true\n")
    )
    assert profile == "airgapped"
    assert with_ml is True


def test_deterministic_only_profile_needs_no_ml(tmp_path):
    # airgapped_thin = the deterministic six only → no provider_requirement.
    profile, with_ml = derive_enrichment_job_params(
        _yaml(tmp_path, "profile: airgapped_thin\nenrichment:\n  enabled: true\n")
    )
    assert profile == "airgapped_thin"
    assert with_ml is False


def test_explicit_provider_block_forces_with_ml(tmp_path):
    profile, with_ml = derive_enrichment_job_params(
        _yaml(
            tmp_path,
            "enrichment:\n"
            "  enabled: true\n"
            "  enrichers:\n"
            "    topic_similarity:\n"
            "      provider: {type: fake_for_test}\n",
        )
    )
    assert profile is None
    assert with_ml is True


def test_unknown_profile_fails_open(tmp_path):
    profile, with_ml = derive_enrichment_job_params(_yaml(tmp_path, "profile: no_such_profile\n"))
    assert profile == "no_such_profile"
    assert with_ml is False


def test_corrupt_yaml_fails_open(tmp_path):
    assert derive_enrichment_job_params(_yaml(tmp_path, "{{not yaml")) == (None, False)
