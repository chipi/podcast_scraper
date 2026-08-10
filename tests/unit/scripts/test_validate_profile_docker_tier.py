"""#1545: a profile that declares ``ner_model`` (spaCy NER pre-pass) must derive the ``ml`` tier.

spaCy ships only in ``[ml]`` — NOT in ``[search]`` (torch is not spaCy) — so a profile that sets
``ner_model`` with the pre-pass on genuinely needs the ml image. Before this guard the tier
derivation keyed ml only off ``speaker_detector_provider=spacy`` / ``vector_search`` / local
``whisper``, so a profile whose ONLY ml requirement was NER would derive ``llm`` and the mismatch
went unflagged (the KG entity-recall silent-degrade class).

NB: this is the STATIC profile check. It recognises NER as an ml-requiring signal and guards future
profiles; it does not by itself resolve cloud_balanced being DEPLOYED on the ``[llm,search]`` image
despite declaring ml (the [search]/[ml] tier-model split — tracked in #1545).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "validate_profile_docker_tier",
    _REPO_ROOT / "scripts" / "tools" / "validate_profile_docker_tier.py",
)
assert _SPEC and _SPEC.loader
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)

pytestmark = [pytest.mark.unit]

derive = _mod._derive_tier_from_yaml


def test_ner_model_requires_ml_tier() -> None:
    # No spaCy speaker / no vector_search / cloud transcription — the ONLY ml requirement is NER.
    data = {
        "transcription_provider": "deepgram",
        "summary_provider": "litellm",
        "ner_model": "en_core_web_trf",
    }
    assert derive(data) == "ml"


def test_ner_model_with_prepass_disabled_does_not_force_ml() -> None:
    data = {
        "transcription_provider": "deepgram",
        "summary_provider": "litellm",
        "ner_model": "en_core_web_trf",
        "kg_extraction_use_ner_prepass": False,
    }
    assert derive(data) == "llm"


def test_plain_cloud_profile_stays_llm() -> None:
    assert derive({"transcription_provider": "deepgram", "summary_provider": "litellm"}) == "llm"


def test_existing_ml_signals_still_derive_ml() -> None:
    assert derive({"speaker_detector_provider": "spacy"}) == "ml"
    assert derive({"vector_search": True}) == "ml"
    assert derive({"transcription_provider": "whisper"}) == "ml"


def test_packaged_profiles_pass_the_validator(monkeypatch) -> None:
    """The real config/profiles/*.yaml stay consistent after the NER guard (no regression)."""
    # main() parses sys.argv; give it a clean argv so pytest's flags don't reach argparse.
    monkeypatch.setattr("sys.argv", ["validate_profile_docker_tier"])
    monkeypatch.delenv("EXPECTED_TIER", raising=False)
    assert _mod.main() == 0
