"""Being on the governed list is not the same as being governed.

`REGISTRY_GOVERNED_FIELDS` is the list of settings a profile YAML may not diverge from. The drift
check builds its expectation as `{k: resolved[k] for k in REGISTRY_GOVERNED_FIELDS if k in
resolved}` — so a field the resolver never emits is filtered out, compared against nothing, and
passes on whatever value the YAML happens to carry, including a hand-edit.

Measured 2026-09-30: `translate_api_base` and `translate_model` were declared on `ProfilePreset`
AND listed as governed, and `resolve_profile_to_settings` emitted neither. Setting
`translate_model: totally/unsanctioned-model` in `prod_dgx_full.yaml` produced *"All 18
registry-governed profiles match the registry."* ADR-157 asserted the opposite in as many words,
on the strength of the declaration.

The check's own success message is part of why it stayed hidden: it counts PROFILES, not fields,
so a shrinking field set never shows up in the output.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "config" / "materialize_profiles.py"


@pytest.fixture(scope="module")
def mp() -> Any:
    spec = importlib.util.spec_from_file_location("materialize_profiles_under_test", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestEveryGovernedFieldIsActuallyEmitted:
    def test_the_audit_is_clean(self, mp: Any) -> None:
        assert mp.check_every_governed_field_is_emitted() == []

    def test_the_translate_fields_are_emitted_now(self, mp: Any) -> None:
        """Pinned by name: these are the three that were governed in name only, and the DGX
        presets are the ones that carry them."""
        settings = mp.resolve_profile_to_settings("prod_dgx_full")
        assert settings.get("translate_model") == "google/translategemma-12b-it"
        assert "8005" in str(settings.get("translate_api_base"))

    def test_an_unemitted_governed_field_FAILS_the_audit(self, mp: Any, monkeypatch) -> None:
        """The audit's whole job. Without it, adding a field to the governed list and forgetting
        the resolver is a silent no-op that reads as a new protection."""
        monkeypatch.setattr(
            mp,
            "REGISTRY_GOVERNED_FIELDS",
            tuple(mp.REGISTRY_GOVERNED_FIELDS) + ("a_field_nobody_emits",),
        )
        problems = mp.check_every_governed_field_is_emitted()
        assert len(problems) == 1
        assert "a_field_nobody_emits" in problems[0]
        assert "governed in name only" in problems[0]

    def test_the_partial_set_is_empty_and_documented(self, mp: Any) -> None:
        """`PARTIALLY_EMITTED_FIELDS` is for a field only SOME presets have. A field emitted by
        no preset can never be legitimate, so nothing belongs there for that case — and an entry
        added without a reason is an escape hatch."""
        for field, reason in mp.PARTIALLY_EMITTED_FIELDS.items():
            assert reason.strip(), f"{field} has no stated reason"
            assert len(reason.split()) >= 6, f"{field}: {reason!r} is not a reason"


class TestTheDriftCheckSeesTheTranslationRouting:
    def test_governed_settings_includes_the_translate_fields(self, mp: Any) -> None:
        """Directly on `governed_settings`, because that is the function whose `if k in resolved`
        filter made the omission silent."""
        got = mp.governed_settings("prod_dgx_full")
        assert "translate_api_base" in got
        assert "translate_model" in got

    def test_a_profile_with_no_translator_does_not_grow_empty_keys(self, mp: Any) -> None:
        """Emitted only when set, like every other optional field in the resolver. A cloud
        profile gaining two empty translate keys would be noise in 16 YAMLs."""
        got = mp.governed_settings("cloud_balanced")
        assert "translate_api_base" not in got
        assert "translate_model" not in got
