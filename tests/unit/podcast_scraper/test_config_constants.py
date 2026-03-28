"""Tests for config_constants (revision helpers, evidence defaults, etc.)."""

import pytest

from podcast_scraper import config_constants

pytestmark = [pytest.mark.unit]


class TestEvidenceStackDefaults:
    """GIL evidence stack default model IDs (Issue #435)."""

    def test_embedding_default_is_full_hf_id(self):
        """Default embedding model is a full HuggingFace ID."""
        assert "/" in config_constants.DEFAULT_EMBEDDING_MODEL
        assert "MiniLM" in config_constants.DEFAULT_EMBEDDING_MODEL

    def test_extractive_qa_default_is_full_hf_id(self):
        """Default extractive QA model is a full HuggingFace ID."""
        assert "/" in config_constants.DEFAULT_EXTRACTIVE_QA_MODEL
        assert "roberta" in config_constants.DEFAULT_EXTRACTIVE_QA_MODEL.lower()

    def test_nli_default_is_full_hf_id(self):
        """Default NLI model is a full HuggingFace ID."""
        assert "/" in config_constants.DEFAULT_NLI_MODEL
        assert "nli" in config_constants.DEFAULT_NLI_MODEL.lower()


class TestIsShaRevision:
    """Tests for is_sha_revision (Issue #428)."""

    def test_accepts_40_lowercase_hex(self):
        """40 lowercase hex chars are accepted as SHA."""
        sha = "38335783885b338d93791936c54bb4be46bebed9"
        assert config_constants.is_sha_revision(sha) is True

    def test_accepts_40_uppercase_hex(self):
        """40 uppercase hex chars are accepted (normalized to lower)."""
        sha = "38335783885B338D93791936C54BB4BE46BEBED9"
        assert config_constants.is_sha_revision(sha) is True

    def test_rejects_main(self):
        """Branch name 'main' is not a SHA."""
        assert config_constants.is_sha_revision("main") is False

    def test_rejects_short_string(self):
        """Short string is not a SHA."""
        assert config_constants.is_sha_revision("abc") is False

    def test_rejects_long_string(self):
        """41+ chars are not a SHA."""
        assert config_constants.is_sha_revision("0" * 41) is False

    def test_rejects_empty(self):
        """Empty string is not a SHA."""
        assert config_constants.is_sha_revision("") is False

    def test_rejects_non_hex(self):
        """Non-hex character (e.g. 'g') is rejected."""
        assert config_constants.is_sha_revision("g" + "0" * 39) is False


class TestGetPinnedRevisionForModel:
    """Tests for get_pinned_revision_for_model (FLAN-T5 and LongT5 pinning)."""

    def test_flan_t5_base_returns_pinned(self):
        """google/flan-t5-base returns FLAN_T5_BASE_REVISION."""
        rev = config_constants.get_pinned_revision_for_model("google/flan-t5-base")
        assert rev == config_constants.FLAN_T5_BASE_REVISION

    def test_flan_t5_large_returns_pinned(self):
        """google/flan-t5-large returns FLAN_T5_LARGE_REVISION."""
        rev = config_constants.get_pinned_revision_for_model("google/flan-t5-large")
        assert rev == config_constants.FLAN_T5_LARGE_REVISION

    def test_long_t5_tglobal_base_returns_pinned(self):
        """long-t5-tglobal-base returns LONG_T5_TGLOBAL_BASE_REVISION."""
        rev = config_constants.get_pinned_revision_for_model("google/long-t5-tglobal-base")
        assert rev == config_constants.LONG_T5_TGLOBAL_BASE_REVISION

    def test_long_t5_tglobal_large_returns_pinned(self):
        """long-t5-tglobal-large returns LONG_T5_TGLOBAL_LARGE_REVISION."""
        rev = config_constants.get_pinned_revision_for_model("google/long-t5-tglobal-large")
        assert rev == config_constants.LONG_T5_TGLOBAL_LARGE_REVISION

    def test_unknown_model_returns_none(self):
        """Unknown model returns None."""
        assert config_constants.get_pinned_revision_for_model("allenai/led-base-16384") is None
        assert config_constants.get_pinned_revision_for_model("other/model") is None


class TestRevisionConstantsAreShas:
    """All ML revision constants must be 40-char SHAs (no 'main')."""

    def test_all_revision_constants_are_40_char_sha(self):
        """Every *_REVISION constant is a valid SHA for reproducibility."""
        rev_attrs = [
            "PEGASUS_CNN_DAILYMAIL_REVISION",
            "LED_BASE_16384_REVISION",
            "LED_LARGE_16384_REVISION",
            "FLAN_T5_BASE_REVISION",
            "FLAN_T5_LARGE_REVISION",
            "LONG_T5_TGLOBAL_BASE_REVISION",
            "LONG_T5_TGLOBAL_LARGE_REVISION",
        ]
        for attr in rev_attrs:
            rev = getattr(config_constants, attr, None)
            assert rev is not None, f"Missing {attr}"
            assert config_constants.is_sha_revision(rev), (
                f"{attr}={rev!r} is not a 40-char SHA; "
                "pin with HfApi().model_info(..., revision='main').sha"
            )
