"""Unit tests for podcast_scraper.workflow.run_manifest module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.workflow.run_manifest import (
    _get_config_hash,
    _revision_for_summary_model,
    create_run_manifest,
    RunManifest,
)


@pytest.mark.unit
class TestGetConfigHash:
    """Tests for _get_config_hash."""

    def test_config_with_model_dump_returns_sha_and_path(self):
        """Config with model_dump returns (sha256, path, json_string)."""
        cfg = MagicMock()
        cfg.model_dump.return_value = {"a": 1, "b": 2}
        cfg.config_path = "/path/to/config.yaml"
        sha, path, full = _get_config_hash(cfg)
        assert sha is not None
        assert len(sha) == 64
        assert path == "/path/to/config.yaml"
        assert "a" in full or "b" in full

    def test_config_secrets_redacted_in_hash(self):
        """Secrets in config are redacted before hashing."""
        cfg = MagicMock()
        cfg.model_dump.return_value = {"api_key": "sk-secret", "title": "Podcast"}
        cfg.config_path = None
        sha1, _, full1 = _get_config_hash(cfg)
        cfg.model_dump.return_value = {"api_key": "sk-other", "title": "Podcast"}
        sha2, _, full2 = _get_config_hash(cfg)
        assert sha1 == sha2
        assert "__redacted__" in full1

    def test_config_dict_fallback(self):
        """Config with .dict() but no model_dump is supported."""
        cfg = MagicMock(spec=["dict", "config_path"])
        cfg.dict.return_value = {"x": 1}
        cfg.config_path = None
        sha, path, _ = _get_config_hash(cfg)
        assert sha is not None


@pytest.mark.unit
class TestRevisionForSummaryModel:
    """Tests for _revision_for_summary_model."""

    def test_none_returns_none(self):
        """None model name returns None."""
        assert _revision_for_summary_model(None) is None

    def test_led_base_returns_revision_when_constant_defined(self):
        """LED base model returns revision if config_constants has it."""
        with patch(
            "podcast_scraper.config_constants.LED_BASE_16384_REVISION",
            "pinned-rev",
            create=True,
        ):
            assert _revision_for_summary_model("allenai/led-base-16384") == "pinned-rev"

    def test_unknown_model_returns_none(self):
        """Unknown model returns None."""
        assert _revision_for_summary_model("unknown/model") is None


@pytest.mark.unit
class TestCreateRunManifest:
    """Tests for create_run_manifest."""

    @patch("podcast_scraper.workflow.run_manifest._get_gpu_info", return_value=None)
    @patch("podcast_scraper.workflow.run_manifest._get_config_hash")
    @patch("podcast_scraper.workflow.run_manifest._get_git_info")
    def test_returns_manifest(
        self, mock_git: MagicMock, mock_config_hash: MagicMock, mock_gpu: MagicMock
    ):
        """create_run_manifest returns RunManifest with expected fields."""
        mock_git.return_value = ("abc123", "main", False)
        mock_config_hash.return_value = ("sha256hex", "/config.yaml", "{}")
        cfg = MagicMock()
        cfg.whisper_model = None
        cfg.summary_model = None
        cfg.summary_reduce_model = None
        cfg.whisper_device = None
        cfg.summary_device = None
        cfg.temperature = None
        cfg.seed = None

        manifest = create_run_manifest(cfg, "/out", run_id="test-run")

        assert isinstance(manifest, RunManifest)
        assert manifest.run_id == "test-run"
        assert manifest.git_commit_sha == "abc123"
        assert manifest.config_sha256 == "sha256hex"
        mock_git.assert_called_once()
        mock_config_hash.assert_called_once_with(cfg)

    @patch("podcast_scraper.workflow.run_manifest._get_gpu_info", return_value=None)
    @patch("podcast_scraper.workflow.run_manifest._get_config_hash")
    @patch("podcast_scraper.workflow.run_manifest._get_git_info")
    def test_create_run_manifest_tolerates_mocked_modules_without_version(
        self, mock_git: MagicMock, mock_config_hash: MagicMock, mock_gpu: MagicMock
    ):
        """create_run_manifest does not raise when torch/transformers/whisper lack __version__.

        Other tests may mock sys.modules; those mocks often have no __version__.
        The manifest uses getattr(..., '__version__', None) so creation still succeeds.
        """
        import sys

        mock_git.return_value = ("abc123", "main", False)
        mock_config_hash.return_value = ("sha256hex", "/config.yaml", "{}")
        cfg = MagicMock()
        cfg.whisper_model = None
        cfg.summary_model = None
        cfg.summary_reduce_model = None
        cfg.whisper_device = None
        cfg.summary_device = None
        cfg.temperature = None
        cfg.seed = None

        # Mocks with spec=[] have no __version__; getattr(..., "__version__", None) returns None
        mock_torch = MagicMock(spec=[])
        mock_transformers = MagicMock(spec=[])
        mock_whisper = MagicMock(spec=[])

        with patch.dict(
            sys.modules,
            {"torch": mock_torch, "transformers": mock_transformers, "whisper": mock_whisper},
        ):
            manifest = create_run_manifest(cfg, "/out", run_id="test-run")

        assert isinstance(manifest, RunManifest)
        assert manifest.torch_version is None
        assert manifest.transformers_version is None
        assert manifest.whisper_version is None


@pytest.mark.unit
class TestRunManifestDataclass:
    """Tests for RunManifest dataclass."""

    def test_to_dict_returns_dict(self):
        """to_dict returns dictionary."""
        m = RunManifest(
            run_id="r1",
            created_at="2026-01-01T00:00:00Z",
            created_by="test",
        )
        d = m.to_dict()
        assert d["run_id"] == "r1"
        assert "created_at" in d

    def test_to_json_returns_json_string(self):
        """to_json returns JSON string."""
        m = RunManifest(
            run_id="r1",
            created_at="2026-01-01T00:00:00Z",
            created_by="test",
        )
        s = m.to_json()
        assert "r1" in s
        assert "created_at" in s
