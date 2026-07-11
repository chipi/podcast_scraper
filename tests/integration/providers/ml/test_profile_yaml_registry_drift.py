"""Drift detection: profile YAML routing must match the declared registry preset.

When a YAML under ``config/profiles/`` declares a top-level ``profile:`` field
that names a known ``_PROFILE_PRESETS`` entry, the YAML's routing fields must
agree with what ``resolve_profile_to_settings(name, ...)`` returns. Otherwise
the eval reports document a decision the runtime never adopts.

Today most profile YAMLs do NOT declare ``profile:`` (they predate #907's
runtime wiring). Those YAMLs are skipped — adding ``profile:`` to them is the
opt-in for drift-checking.

Adding a new profile preset to the registry without a matching YAML
declaration is fine and won't trip this test. The test only fires once a YAML
has explicitly opted in by declaring ``profile:``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import yaml

from podcast_scraper.providers.ml.model_registry import (
    _PROFILE_PRESETS,
    resolve_profile_to_settings,
)

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]

_PROFILE_DIR = Path(__file__).resolve().parents[4] / "config" / "profiles"

_ROUTING_FIELDS = (
    "transcription_provider",
    "summary_provider",
    "summary_model",
    "kg_extraction_source",
    "kg_max_topics",
    "kg_max_entities",
    "speaker_detector_provider",
    "ner_model",
    "gi_insight_source",
    "gi_max_insights",
    "gi_require_grounding",
    "gil_evidence_quote_mode",
    "gil_evidence_nli_mode",
    "topic_cluster_threshold",
    "insight_cluster_threshold",
    "diarization_model",
    "dgx_diarize_model",
)


def _opted_in_yamls() -> List[Tuple[Path, Dict[str, Any]]]:
    """Yield (path, parsed_dict) for every YAML that declares a ``profile:`` field."""
    opted_in: List[Tuple[Path, Dict[str, Any]]] = []
    if not _PROFILE_DIR.is_dir():
        return opted_in
    for path in sorted(_PROFILE_DIR.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        if isinstance(data, dict) and "profile" in data:
            opted_in.append((path, data))
    return opted_in


_OPTED_IN = _opted_in_yamls()


@pytest.mark.parametrize(
    "path,data",
    _OPTED_IN,
    ids=[p.name for p, _ in _OPTED_IN] or ["no-opted-in-yamls"],
)
def test_profile_yaml_matches_registry_preset(path: Path, data: Dict[str, Any]) -> None:
    """Every routing field present in the YAML must equal the registry's value."""
    profile_name = data["profile"]
    assert profile_name in _PROFILE_PRESETS, (
        f"{path.name} declares profile={profile_name!r}, which is not in the "
        f"registry. Known presets: {sorted(_PROFILE_PRESETS)}"
    )
    expected = resolve_profile_to_settings(profile_name, dgx_tailnet_host="drift-check.example")
    for field in _ROUTING_FIELDS:
        if field not in data:
            continue  # YAML omits this field — registry default applies, no drift.
        # nosec B608 — not SQL; just a multi-line f-string assert message.
        msg = (
            f"{path.name}: field {field!r} = {data[field]!r} disagrees with "  # nosec B608
            f"registry preset {profile_name!r} "
            f"(expected {expected.get(field)!r}). "
            f"Update either the YAML or the registry preset "
            f"(with new research_ref) to bring them back in sync."
        )
        assert data[field] == expected.get(field), msg


# #1076 chunk 4-A — overlay-only fields that ProfilePreset does NOT carry.
# A YAML drop of a `True` here silently regresses the airgapped profiles
# back to the regex-only MENTIONS_PERSON path. Pin the expected value
# per profile so the drift test catches it.
_OVERLAY_EXPECTED: Dict[str, Dict[str, Any]] = {
    "airgapped": {
        "gi_typed_mentions_use_ner": True,
        "kg_organizations_use_ner": True,
        "kg_topic_corpus_clustering": True,
    },
    "airgapped_thin": {
        "gi_typed_mentions_use_ner": True,
        "kg_organizations_use_ner": True,
        "kg_topic_corpus_clustering": True,
    },
}


@pytest.mark.parametrize(
    "path,data",
    _OPTED_IN,
    ids=[p.name for p, _ in _OPTED_IN] or ["no-opted-in-yamls"],
)
def test_profile_yaml_overlay_fields_match_expected(path: Path, data: Dict[str, Any]) -> None:
    """YAML-overlay-only fields (not in ProfilePreset) MUST match the
    pinned expected value per profile. Without this, a silent YAML
    drop wouldn't be caught by the routing-field drift test above —
    the registry has no slot to compare against.

    Add new overlay fields to ``_OVERLAY_EXPECTED`` as they're
    introduced; profiles that don't appear in the map are skipped.
    """
    profile_name = data["profile"]
    if profile_name not in _OVERLAY_EXPECTED:
        pytest.skip(f"no overlay expectations for profile {profile_name!r}")
    for field, expected_value in _OVERLAY_EXPECTED[profile_name].items():
        assert field in data, (
            f"{path.name}: overlay field {field!r} is MISSING. "
            f"_OVERLAY_EXPECTED[{profile_name!r}] expects {expected_value!r}. "
            f"Either restore the field or remove the pin in the test."
        )
        assert data[field] == expected_value, (
            f"{path.name}: overlay field {field!r} = {data[field]!r} drifted "
            f"from expected {expected_value!r}. Update either the YAML or "
            f"the _OVERLAY_EXPECTED pin to bring them back in sync."
        )


def test_no_opted_in_yamls_is_self_documenting() -> None:
    """Smoke test that surfaces an informative message if no YAML has opted in yet.

    Without this, the parametrized test above generates a single ``no-opted-in-yamls``
    case that silently passes — making the drift check look broken at a glance.
    """
    if not _OPTED_IN:
        pytest.skip(
            "No profile YAML declares `profile: <name>` yet — drift check is a no-op. "
            "Opt in by adding `profile: <preset_name>` to a YAML under config/profiles/."
        )
    # When at least one YAML opts in, this test passes trivially.
    assert _OPTED_IN


def test_every_registry_preset_has_a_matching_yaml() -> None:
    """Orphan-preset detector: every `_PROFILE_PRESETS` entry must have a
    corresponding YAML under `config/profiles/` declaring `profile: <name>`.

    Adding a new preset to the registry without shipping a YAML for it is a
    smell — the preset is then load-bearing only via
    ``resolve_profile_to_settings`` from code, with no operator-facing
    profile YAML to look at. This test catches that asymmetry as a
    regression.

    To exempt a preset (e.g. for transient experimentation), add its name
    to ``_PRESET_WITHOUT_YAML_ALLOWLIST`` below WITH a rationale.
    """
    _PRESET_WITHOUT_YAML_ALLOWLIST: set[str] = set()  # empty today; documented exemptions go here

    opted_in_names = {data["profile"] for _, data in _OPTED_IN}
    presets = set(_PROFILE_PRESETS)
    orphans = presets - opted_in_names - _PRESET_WITHOUT_YAML_ALLOWLIST
    assert not orphans, (
        f"Registry presets without a matching YAML opt-in: {sorted(orphans)}. "
        "Either ship a YAML at config/profiles/<name>.yaml declaring "
        "`profile: <name>` at the top, OR add the preset to "
        "_PRESET_WITHOUT_YAML_ALLOWLIST in this test with a rationale comment."
    )


_PROVIDER_TO_MODEL_KEY = {
    "openai": "openai_summary_model",
    "anthropic": "anthropic_summary_model",
    "gemini": "gemini_summary_model",
    "mistral": "mistral_summary_model",
    "deepseek": "deepseek_summary_model",
    "grok": "grok_summary_model",
    "ollama": "ollama_summary_model",
}


def test_every_profile_pins_model_for_its_summary_provider() -> None:
    """Profile YAMLs that pin a non-default summary provider should ALSO pin
    that provider's specific summary model — or accept the Field default
    explicitly via documentation.

    Catches the class of "profile half-set" mistake: e.g. setting
    ``summary_provider: anthropic`` without pinning
    ``anthropic_summary_model`` means the run silently uses the Field
    default (PROD = claude-3-5-sonnet-20241022), which might or might not
    be what the profile author intended. The drift bug we just fixed
    (2026-06-23) was a structural example of this class — defaults
    silently varying.

    Enforces operator's "profiles are source of truth" framing.

    Test/legacy profiles are exempted (test_default, profile_freeze.example)
    — their model pins live in non-profile-name keys.
    """
    _EXEMPT_FROM_CHECK: set[str] = {
        # test_default pins MODELS across vendors; provider is chosen by Field
        # default for the path that runs. Hence summary_provider unset.
        "test_default",
        # profile_freeze.example is a legacy stub; skip.
        "profile_freeze.example",
        # transformers + summllama profiles: model controlled by summary_model
        # (alias-based), not provider-specific *_summary_model knob.
        "airgapped",
        "airgapped_thin",
        "dev",
    }

    if not _PROFILE_DIR.is_dir():
        pytest.skip("profile dir absent")

    violations: List[str] = []
    for path in sorted(_PROFILE_DIR.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        if not isinstance(data, dict):
            continue
        stem = path.stem
        if stem in _EXEMPT_FROM_CHECK:
            continue
        provider = data.get("summary_provider")
        if not provider or provider == "transformers":
            continue  # transformers path uses summary_model alias, not vendor-specific knob
        expected_key = _PROVIDER_TO_MODEL_KEY.get(provider)
        if expected_key and expected_key not in data:
            violations.append(
                f"{path.name}: pins `summary_provider: {provider}` but not "
                f"`{expected_key}`. Pin it explicitly (or add stem to the "
                f"exempt list with a rationale)."
            )

    assert not violations, "\n".join(violations)


def test_every_yaml_only_profile_documents_its_status() -> None:
    """YAML-only profiles (no `profile:` top-level field) MUST carry a
    comment explaining why they don't have a `ProfilePreset` — otherwise
    they look like accidental orphans.

    Catches "added a YAML, forgot to opt into the registry" mistakes by
    requiring the operator to document the design choice on every
    YAML-only profile.
    """
    _YAML_ONLY_DOC_MARKERS = (
        "yaml-only",
        "no entry in `_profile_presets`",
        "registry status: yaml-only",
    )
    if not _PROFILE_DIR.is_dir():
        pytest.skip("profile dir absent")

    yaml_only: List[Path] = []
    for path in sorted(_PROFILE_DIR.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        if not (isinstance(data, dict) and "profile" in data):
            yaml_only.append(path)

    undocumented: List[Path] = []
    for path in yaml_only:
        raw = path.read_text(encoding="utf-8").lower()
        if not any(m in raw for m in _YAML_ONLY_DOC_MARKERS):
            undocumented.append(path)

    assert not undocumented, (
        "YAML-only profiles (no `profile:` opt-in) must carry a comment explaining "
        "why they aren't in `_PROFILE_PRESETS`. Add a comment containing one of "
        f"{list(_YAML_ONLY_DOC_MARKERS)} to: {[p.name for p in undocumented]}"
    )


# Vendor-specific summary-model keys that the test_default profile pins. The
# "pins MODEL across vendors without pinning PROVIDER" property — which makes
# the 6-tuple ProfilePreset shape structurally inapplicable — is asserted by
# requiring the YAML to set at least three of these so a future edit can't
# accidentally collapse the profile into the regular preset shape and
# then silently look like an orphan candidate.
_TEST_DEFAULT_VENDOR_SUMMARY_MODEL_KEYS = (
    "openai_summary_model",
    "anthropic_summary_model",
    "mistral_summary_model",
    "gemini_summary_model",
    "deepseek_summary_model",
    "grok_summary_model",
    "ollama_summary_model",
)


def test_test_default_is_intentionally_yaml_only_not_a_drift_bug() -> None:
    """`test_default` is YAML-only by design (#1060 D1 = path B): it pins
    MODEL across vendors without pinning PROVIDER, which is structurally
    incompatible with the 6-tuple `ProfilePreset` shape (every Preset entry
    names exactly one StageOption per stage = one PROVIDER per stage).

    This test makes that intent structural rather than implicit:

    1. `test_default` MUST NOT appear in `_PROFILE_PRESETS`. Promoting it
       would force test-tier StageOption coverage for every provider, with
       no product driver to justify the registry surface expansion.
    2. `test_default.yaml` MUST NOT carry a `profile:` opt-in. The drift
       test must continue to skip it.
    3. The YAML MUST carry one of the "YAML-only" documentation markers
       (the documented-status check already enforces this; we re-assert
       here so a future change to that test surfaces in one place).
    4. The YAML MUST pin at least three vendor-specific
       `<vendor>_summary_model` keys. That's the structural fingerprint
       of "pin MODEL across vendors". If a future edit collapses
       test_default down to a single-vendor preset shape, this assertion
       fires and the operator either promotes it to `_PROFILE_PRESETS` or
       restores the multi-vendor pins.

    Removing or relaxing this test is allowed ONLY if Decision 1 (B) is
    revisited. See `config/profiles/README.md` § "Registry status — two
    first-class shapes" and the 2026-06-23 amendment in
    `src/podcast_scraper/providers/ml/model_registry.py`.
    """
    yaml_path = _PROFILE_DIR / "test_default.yaml"
    assert yaml_path.is_file(), "test_default.yaml is missing — promotion or rename?"

    # (1) Not in _PROFILE_PRESETS.
    assert "test_default" not in _PROFILE_PRESETS, (
        "`test_default` was added to `_PROFILE_PRESETS`. If that promotion is "
        "intentional (Decision 1 = A path), delete this test with a written "
        "rationale; otherwise revert the registry addition."
    )

    # (2) No `profile:` opt-in.
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    assert isinstance(data, dict), "test_default.yaml did not parse as a dict"
    assert "profile" not in data, (
        "test_default.yaml gained a `profile:` field; the drift check would "
        "now compare it against `_PROFILE_PRESETS['test_default']`. Either "
        "delete this test (path A) or remove the `profile:` field (path B)."
    )

    # (3) Documentation marker present (re-asserted, kept narrow).
    _YAML_ONLY_DOC_MARKERS = (
        "yaml-only",
        "no entry in `_profile_presets`",
        "registry status: yaml-only",
    )
    raw_lower = yaml_path.read_text(encoding="utf-8").lower()
    assert any(m in raw_lower for m in _YAML_ONLY_DOC_MARKERS), (
        "test_default.yaml is missing the YAML-only documentation marker. "
        f"Add one of {list(_YAML_ONLY_DOC_MARKERS)}."
    )

    # (4) Structural fingerprint — pins MODEL across multiple vendors.
    pinned_vendor_models = [k for k in _TEST_DEFAULT_VENDOR_SUMMARY_MODEL_KEYS if k in data]
    assert len(pinned_vendor_models) >= 3, (
        f"test_default.yaml pins only {pinned_vendor_models!r} vendor summary "
        "models. The 'pin MODEL across vendors' property requires at least 3 "
        f"of {list(_TEST_DEFAULT_VENDOR_SUMMARY_MODEL_KEYS)}. If the test_default "
        "profile collapsed to a single vendor, promote it to `_PROFILE_PRESETS` "
        "and delete this test."
    )
