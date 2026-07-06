"""Unit tests for :func:`_download_hf_evidence_model` — the Phase G
cache-warm helper introduced by #382.

Covers all three ``kind`` branches (``qa``, ``nli``, ``embedding``) plus
the unknown-kind error path. Imports are mocked at ``sys.modules`` level
so no real HF Hub download happens.
"""

from __future__ import annotations

import sys
from unittest import mock

import pytest

pytest.importorskip("transformers")

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


def _install_transformers_mock(monkeypatch):
    """Replace `transformers` module with a MagicMock — captures the
    AutoTokenizer / AutoModelForQuestionAnswering call sites."""
    m = mock.MagicMock()
    monkeypatch.setitem(sys.modules, "transformers", m)
    return m


class _FakeSTModule:
    """Stand-in for the sentence_transformers module — a real object with
    real classes so inspect.signature works."""

    def __init__(self):
        self.CrossEncoder = None  # overwritten per-test
        self.SentenceTransformer = mock.MagicMock()


def _install_sentence_transformers_mock(monkeypatch):
    m = _FakeSTModule()
    monkeypatch.setitem(sys.modules, "sentence_transformers", m)
    return m


class TestDownloadHFEvidenceModelQA:
    def test_qa_kind_calls_autotokenizer_and_qa_model_from_pretrained(self, monkeypatch, tmp_path):
        """kind='qa' → AutoTokenizer + AutoModelForQuestionAnswering warmed."""
        tr = _install_transformers_mock(monkeypatch)
        monkeypatch.setattr(
            "podcast_scraper.providers.ml.model_loader.get_transformers_cache_dir",
            lambda: tmp_path,
        )
        from podcast_scraper.providers.ml.model_loader import _download_hf_evidence_model

        _download_hf_evidence_model("qa", "deepset/roberta-base-squad2")

        # Both must have been called with the same download-permitted kw set.
        assert tr.AutoTokenizer.from_pretrained.called
        assert tr.AutoModelForQuestionAnswering.from_pretrained.called
        _, kw = tr.AutoTokenizer.from_pretrained.call_args
        # Preload path — local_files_only=False (downloads allowed)
        assert kw["local_files_only"] is False
        assert kw["trust_remote_code"] is False
        assert kw["low_cpu_mem_usage"] is False
        assert str(tmp_path) in kw["cache_dir"]


class TestDownloadHFEvidenceModelNLI:
    def test_nli_kind_supports_both_ce_kwargs(self, monkeypatch, tmp_path):
        """kind='nli' with a CrossEncoder that accepts local_files_only + cache_folder."""
        st = _install_sentence_transformers_mock(monkeypatch)
        # Replace the mocked class with a MagicMock again so we can inspect calls,
        # but keep the fake __init__ signature we just installed.
        recorded = {}

        class FakeCE:
            def __init__(self, model_id, local_files_only=None, cache_folder=None):
                recorded["model_id"] = model_id
                recorded["local_files_only"] = local_files_only
                recorded["cache_folder"] = cache_folder

        st.CrossEncoder = FakeCE
        monkeypatch.setattr(
            "podcast_scraper.providers.ml.model_loader.get_transformers_cache_dir",
            lambda: tmp_path,
        )
        from podcast_scraper.providers.ml.model_loader import _download_hf_evidence_model

        _download_hf_evidence_model("nli", "cross-encoder/nli-deberta-v3-base")

        assert recorded["model_id"] == "cross-encoder/nli-deberta-v3-base"
        assert recorded["local_files_only"] is False
        assert str(tmp_path) in recorded["cache_folder"]

    def test_nli_kind_with_ce_supporting_no_extra_kwargs(self, monkeypatch, tmp_path):
        """CrossEncoder without local_files_only / cache_folder — both branches skip."""
        st = _install_sentence_transformers_mock(monkeypatch)
        recorded = {"count": 0}

        class FakeCE:
            def __init__(self, model_id):
                recorded["count"] += 1
                recorded["model_id"] = model_id

        st.CrossEncoder = FakeCE
        monkeypatch.setattr(
            "podcast_scraper.providers.ml.model_loader.get_transformers_cache_dir",
            lambda: tmp_path,
        )
        from podcast_scraper.providers.ml.model_loader import _download_hf_evidence_model

        _download_hf_evidence_model("nli", "some/nli-model")
        assert recorded["count"] == 1
        assert recorded["model_id"] == "some/nli-model"


class TestDownloadHFEvidenceModelEmbedding:
    def test_embedding_kind_calls_sentence_transformer(self, monkeypatch, tmp_path):
        """kind='embedding' → SentenceTransformer(model_id, cache_folder=…)."""
        st = _install_sentence_transformers_mock(monkeypatch)
        monkeypatch.setattr(
            "podcast_scraper.providers.ml.model_loader.get_transformers_cache_dir",
            lambda: tmp_path,
        )
        from podcast_scraper.providers.ml.model_loader import _download_hf_evidence_model

        _download_hf_evidence_model("embedding", "sentence-transformers/all-MiniLM-L6-v2")

        st.SentenceTransformer.assert_called_once()
        args, kw = st.SentenceTransformer.call_args
        assert args[0] == "sentence-transformers/all-MiniLM-L6-v2"
        assert str(tmp_path) in kw["cache_folder"]


class TestDownloadHFEvidenceModelUnknownKind:
    def test_unknown_kind_raises_value_error(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "podcast_scraper.providers.ml.model_loader.get_transformers_cache_dir",
            lambda: tmp_path,
        )
        from podcast_scraper.providers.ml.model_loader import _download_hf_evidence_model

        with pytest.raises(ValueError, match="Unknown evidence kind"):
            _download_hf_evidence_model("bogus", "some/model")  # type: ignore[arg-type]
