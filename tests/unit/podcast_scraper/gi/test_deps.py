"""Tests for GIL optional dependency validation."""

import builtins
from unittest.mock import MagicMock

import pytest

from podcast_scraper.exceptions import ProviderDependencyError
from podcast_scraper.gi.deps import (
    create_gil_evidence_providers,
    validate_gil_grounding_dependencies,
)
from tests.conftest import create_test_config

pytestmark = [pytest.mark.unit]

_real_import = builtins.__import__


def test_validate_skips_when_gi_disabled():
    """No check when generate_gi is false."""
    cfg = MagicMock()
    cfg.generate_gi = False
    cfg.gi_require_grounding = True
    cfg.entailment_provider = "transformers"
    validate_gil_grounding_dependencies(cfg)


def test_validate_skips_when_grounding_disabled():
    """No check when gi_require_grounding is false."""
    cfg = MagicMock()
    cfg.generate_gi = True
    cfg.gi_require_grounding = False
    cfg.entailment_provider = "transformers"
    validate_gil_grounding_dependencies(cfg)


def test_validate_skips_for_api_entailment():
    """Local sentence-transformers not required for API entailment."""
    cfg = MagicMock()
    cfg.generate_gi = True
    cfg.gi_require_grounding = True
    cfg.entailment_provider = "openai"
    validate_gil_grounding_dependencies(cfg)


def test_validate_raises_when_local_entailment_and_st_missing(monkeypatch):
    """ProviderDependencyError when transformers entailment and import fails."""

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sentence_transformers":
            raise ImportError("No module named 'sentence_transformers'")
        return _real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    cfg = MagicMock()
    cfg.generate_gi = True
    cfg.gi_require_grounding = True
    cfg.entailment_provider = "transformers"

    with pytest.raises(ProviderDependencyError) as ei:
        validate_gil_grounding_dependencies(cfg)
    err = str(ei.value).lower()
    assert "sentence" in err and "transform" in err.replace("_", "-")


def test_validate_ok_when_sentence_transformers_present():
    """No error when package is importable."""
    cfg = MagicMock()
    cfg.generate_gi = True
    cfg.gi_require_grounding = True
    cfg.entailment_provider = "transformers"
    pytest.importorskip("sentence_transformers")
    validate_gil_grounding_dependencies(cfg)


def test_create_gil_evidence_providers_one_factory_call_when_quote_entail_match_summary(
    monkeypatch,
):
    """When quote, entail, and summary are the same backend, reuse one provider instance."""
    instances = []

    def fake_create(cfg, provider_type_override=None):
        m = MagicMock()
        instances.append(provider_type_override)
        return m

    monkeypatch.setattr(
        "podcast_scraper.summarization.factory.create_summarization_provider",
        fake_create,
    )
    cfg = create_test_config(
        summary_provider="transformers",
        quote_extraction_provider="transformers",
        entailment_provider="transformers",
    )
    q, e = create_gil_evidence_providers(cfg, summary_provider=None)
    assert q is e
    assert len(instances) == 1
    assert instances[0] is None


def test_create_gil_evidence_providers_reuses_passed_summary_provider(monkeypatch):
    """When cfg points quote/entail at summary_provider, use the passed instance."""
    calls = []

    def fake_create(*_a, **_k):
        calls.append(1)
        return MagicMock()

    monkeypatch.setattr(
        "podcast_scraper.summarization.factory.create_summarization_provider",
        fake_create,
    )
    sp = MagicMock()
    cfg = create_test_config(
        summary_provider="transformers",
        quote_extraction_provider="transformers",
        entailment_provider="transformers",
    )
    q, e = create_gil_evidence_providers(cfg, summary_provider=sp)
    assert q is sp and e is sp
    assert calls == []
