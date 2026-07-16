"""Every model a shipped profile pins must be on the known-model allowlist.

The operator's ask: "when somebody enters something in a profile we can validate that." This is that
gate at CI time — a bad or fictional cloud model committed to ``config/profiles/*.yaml`` fails here
before it can reach a run. Local providers (ollama, transformers, summllama) are ungoverned by
design and pass through untouched.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from podcast_scraper.providers.known_models import (
    iter_profile_model_refs,
    UnknownModelError,
    validate_profile_or_raise,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PROFILE_DIR = _REPO_ROOT / "config" / "profiles"


def _profile_files():
    return sorted(_PROFILE_DIR.glob("*.yaml"))


def test_profile_dir_is_present() -> None:
    assert _PROFILE_DIR.is_dir(), f"expected profiles at {_PROFILE_DIR}"
    assert _profile_files(), "no profile YAMLs found — the sweep would be a no-op"


@pytest.mark.parametrize("profile_path", _profile_files(), ids=lambda p: p.name)
def test_every_profile_pins_only_known_models(profile_path: Path) -> None:
    """THE PROFILE GATE. Load each real profile and validate every model it names."""
    data = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    # A profile YAML may carry an operator sub-body; validate the flat top-level model fields, which
    # is where summary/value-gate models live.
    validate_profile_or_raise(data, context=profile_path.name)


def test_the_extractor_maps_provider_prefixes_and_the_bare_pair() -> None:
    profile = {
        "summary_provider": "anthropic",
        "summary_model": "claude-haiku-4-5",
        "gemini_summary_model": "gemini-2.5-flash-lite",
        "openai_summary_model": "autoresearch",
        "ollama_summary_model": "qwen3.5:35b",
        "gi_value_gate_provider": "grok",
        "gi_value_gate_model": "grok-4.3",
    }
    refs = {(p, m) for p, m, _ in iter_profile_model_refs(profile)}
    assert ("gemini", "gemini-2.5-flash-lite") in refs
    assert ("openai", "autoresearch") in refs  # local vLLM served-name, allowlisted
    assert ("ollama", "qwen3.5:35b") in refs  # extracted, but ungoverned at validation
    assert ("anthropic", "claude-haiku-4-5") in refs
    assert ("grok", "grok-4.3") in refs
    validate_profile_or_raise(profile, context="synthetic")  # all known → no raise


def test_a_profile_with_a_bad_cloud_model_is_rejected() -> None:
    bad = {"summary_provider": "anthropic", "summary_model": "claude-sonnet-4-6"}
    with pytest.raises(UnknownModelError, match="claude-sonnet-4-6"):
        validate_profile_or_raise(bad, context="bad_profile.yaml")
