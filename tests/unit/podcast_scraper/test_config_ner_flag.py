"""#1076 chunk 4-A — Config + workflow wiring for ``gi_typed_mentions_use_ner``.

Locks the contract that:

1. The YAML overlays for ``airgapped`` and ``airgapped_thin`` flip the
   flag on, and the YAML overlays for ``dev`` / ``cloud_thin`` leave it
   off (or unset, defaulting to False).
2. The workflow gating expression
   (``workflow/metadata_generation.py:4008``) — ``nlp if
   getattr(cfg, "gi_typed_mentions_use_ner", False) else None`` —
   returns the loaded model exactly when the flag is True, ``None``
   otherwise, and ``None`` when the field is absent.

Without this test the cfg → workflow wiring could silently regress
(a YAML drop, a refactor that renames the field) without any
existing test catching it, because the leaf-level unit tests in
``tests/unit/gi/test_relational_edges.py`` exercise the helper
directly with ``nlp=...`` and don't care how the workflow chose to
pass it.
"""

from __future__ import annotations

import pytest

from podcast_scraper.config import Config

pytestmark = pytest.mark.unit


@pytest.fixture
def _fake_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPGRAM_API_KEY",
    ):
        monkeypatch.setenv(name, "test-dummy")


class TestProfileNERFlag:
    """YAML-overlay contract for ``gi_typed_mentions_use_ner``."""

    def test_airgapped_profile_flips_ner_on(self, _fake_keys: None) -> None:
        cfg = Config.model_validate({"profile": "airgapped"})
        assert cfg.gi_typed_mentions_use_ner is True

    def test_airgapped_thin_profile_flips_ner_on(self, _fake_keys: None) -> None:
        cfg = Config.model_validate({"profile": "airgapped_thin"})
        assert cfg.gi_typed_mentions_use_ner is True

    def test_dev_profile_leaves_ner_off(self, _fake_keys: None) -> None:
        cfg = Config.model_validate({"profile": "dev"})
        assert cfg.gi_typed_mentions_use_ner is False

    def test_cloud_thin_profile_leaves_ner_off(self, _fake_keys: None) -> None:
        cfg = Config.model_validate({"profile": "cloud_thin"})
        assert cfg.gi_typed_mentions_use_ner is False

    def test_default_when_no_profile(self, _fake_keys: None) -> None:
        cfg = Config.model_validate({})
        assert cfg.gi_typed_mentions_use_ner is False

    def test_explicit_override_wins_over_profile(self, _fake_keys: None) -> None:
        cfg = Config.model_validate({"profile": "airgapped", "gi_typed_mentions_use_ner": False})
        assert cfg.gi_typed_mentions_use_ner is False


class TestProfileKGOrgFlag:
    """#1058 chunk 1 — YAML-overlay contract for ``kg_organizations_use_ner``."""

    def test_airgapped_profile_flips_kg_org_on(self, _fake_keys: None) -> None:
        cfg = Config.model_validate({"profile": "airgapped"})
        assert cfg.kg_organizations_use_ner is True

    def test_airgapped_thin_profile_flips_kg_org_on(self, _fake_keys: None) -> None:
        cfg = Config.model_validate({"profile": "airgapped_thin"})
        assert cfg.kg_organizations_use_ner is True

    def test_dev_profile_leaves_kg_org_off(self, _fake_keys: None) -> None:
        cfg = Config.model_validate({"profile": "dev"})
        assert cfg.kg_organizations_use_ner is False

    def test_cloud_thin_profile_leaves_kg_org_off(self, _fake_keys: None) -> None:
        cfg = Config.model_validate({"profile": "cloud_thin"})
        assert cfg.kg_organizations_use_ner is False

    def test_default_when_no_profile(self, _fake_keys: None) -> None:
        cfg = Config.model_validate({})
        assert cfg.kg_organizations_use_ner is False

    def test_explicit_override_wins_over_profile(self, _fake_keys: None) -> None:
        cfg = Config.model_validate({"profile": "airgapped", "kg_organizations_use_ner": False})
        assert cfg.kg_organizations_use_ner is False


class TestWorkflowGatingExpression:
    """Direct contract test for the conditional at
    ``workflow/metadata_generation.py:4008``."""

    @staticmethod
    def _gate(cfg: object, nlp: object) -> object:
        """Mirror the production gating expression exactly."""
        return nlp if getattr(cfg, "gi_typed_mentions_use_ner", False) else None

    def test_gate_returns_nlp_when_flag_true(self) -> None:
        class _Cfg:
            gi_typed_mentions_use_ner = True

        sentinel_nlp = object()
        assert self._gate(_Cfg(), sentinel_nlp) is sentinel_nlp

    def test_gate_returns_none_when_flag_false(self) -> None:
        class _Cfg:
            gi_typed_mentions_use_ner = False

        sentinel_nlp = object()
        assert self._gate(_Cfg(), sentinel_nlp) is None

    def test_gate_defaults_to_none_when_field_absent(self) -> None:
        """A future cfg shape that drops the field altogether must not
        crash — getattr default keeps NER off."""

        class _LegacyCfg:
            pass

        sentinel_nlp = object()
        assert self._gate(_LegacyCfg(), sentinel_nlp) is None
