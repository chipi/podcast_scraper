#!/usr/bin/env python3
"""Tests for experiment configuration module.

This module tests Pydantic models and utility functions for experiment configuration.
"""

import os
import sys
import unittest
from pathlib import Path

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pydantic import ValidationError

from podcast_scraper import config, experiment_config


class TestPromptConfig(unittest.TestCase):
    """Test PromptConfig Pydantic model."""

    def test_prompt_config_minimal(self):
        """Test PromptConfig with minimal required fields."""
        cfg = experiment_config.PromptConfig(user="summarization/system_v1")
        self.assertEqual(cfg.user, "summarization/system_v1")
        self.assertIsNone(cfg.system)
        self.assertEqual(cfg.params, {})

    def test_prompt_config_with_system(self):
        """Test PromptConfig with system prompt."""
        cfg = experiment_config.PromptConfig(
            user="summarization/long_v1", system="summarization/system_v1"
        )
        self.assertEqual(cfg.user, "summarization/long_v1")
        self.assertEqual(cfg.system, "summarization/system_v1")
        self.assertEqual(cfg.params, {})

    def test_prompt_config_with_params(self):
        """Test PromptConfig with template parameters."""
        cfg = experiment_config.PromptConfig(
            user="summarization/long_v1", params={"paragraphs_min": 3, "paragraphs_max": 6}
        )
        self.assertEqual(cfg.user, "summarization/long_v1")
        self.assertEqual(cfg.params["paragraphs_min"], 3)
        self.assertEqual(cfg.params["paragraphs_max"], 6)

    def test_prompt_config_missing_user(self):
        """Test PromptConfig validation fails without required user field."""
        with self.assertRaises(ValidationError):
            experiment_config.PromptConfig()


class TestHFBackendConfig(unittest.TestCase):
    """Test HFBackendConfig Pydantic model."""

    def test_hf_backend_config_minimal(self):
        """Test HFBackendConfig with minimal fields."""
        cfg = experiment_config.HFBackendConfig()
        self.assertEqual(cfg.type, "hf_local")
        self.assertIsNone(cfg.map_model)
        self.assertIsNone(cfg.reduce_model)
        self.assertIsNone(cfg.model)

    def test_hf_backend_config_with_models(self):
        """Test HFBackendConfig with model names."""
        cfg = experiment_config.HFBackendConfig(
            map_model="facebook/bart-base", reduce_model="allenai/led-base-16384"
        )
        self.assertEqual(cfg.type, "hf_local")
        self.assertEqual(cfg.map_model, "facebook/bart-base")
        self.assertEqual(cfg.reduce_model, "allenai/led-base-16384")

    def test_hf_backend_config_single_model(self):
        """Test HFBackendConfig with single model."""
        cfg = experiment_config.HFBackendConfig(model="facebook/bart-base")
        self.assertEqual(cfg.type, "hf_local")
        self.assertEqual(cfg.model, "facebook/bart-base")


class TestOpenAIBackendConfig(unittest.TestCase):
    """Test OpenAIBackendConfig Pydantic model."""

    def test_openai_backend_config(self):
        """Test OpenAIBackendConfig with model name."""
        cfg = experiment_config.OpenAIBackendConfig(model=config.TEST_DEFAULT_OPENAI_SUMMARY_MODEL)
        self.assertEqual(cfg.type, "openai")
        self.assertEqual(cfg.model, config.TEST_DEFAULT_OPENAI_SUMMARY_MODEL)

    def test_openai_backend_config_missing_model(self):
        """Test OpenAIBackendConfig validation fails without required model field."""
        with self.assertRaises(ValidationError):
            experiment_config.OpenAIBackendConfig()


class TestDataConfig(unittest.TestCase):
    """Test DataConfig Pydantic model."""

    def test_data_config_minimal(self):
        """Test DataConfig with minimal required fields."""
        cfg = experiment_config.DataConfig(episodes_glob="*.txt")
        self.assertEqual(cfg.episodes_glob, "*.txt")
        self.assertEqual(cfg.id_from, "parent_dir")  # default

    def test_data_config_with_id_from_stem(self):
        """Test DataConfig with id_from='stem'."""
        cfg = experiment_config.DataConfig(episodes_glob="*.txt", id_from="stem")
        self.assertEqual(cfg.episodes_glob, "*.txt")
        self.assertEqual(cfg.id_from, "stem")

    def test_data_config_with_id_from_parent_dir(self):
        """Test DataConfig with id_from='parent_dir'."""
        cfg = experiment_config.DataConfig(episodes_glob="*.txt", id_from="parent_dir")
        self.assertEqual(cfg.episodes_glob, "*.txt")
        self.assertEqual(cfg.id_from, "parent_dir")

    def test_data_config_missing_episodes_glob(self):
        """Test DataConfig validation fails without required episodes_glob field."""
        with self.assertRaises(ValidationError):
            experiment_config.DataConfig()


class TestExperimentParams(unittest.TestCase):
    """Test ExperimentParams Pydantic model."""

    def test_experiment_params_minimal(self):
        """Test ExperimentParams with minimal fields."""
        params = experiment_config.ExperimentParams()
        self.assertIsNone(params.max_length)
        self.assertIsNone(params.min_length)
        self.assertIsNone(params.chunk_size)
        self.assertEqual(params.extra, {})

    def test_experiment_params_with_values(self):
        """Test ExperimentParams with explicit values."""
        params = experiment_config.ExperimentParams(max_length=512, min_length=128, temperature=0.7)
        self.assertEqual(params.max_length, 512)
        self.assertEqual(params.min_length, 128)
        self.assertEqual(params.temperature, 0.7)

    def test_experiment_params_collect_extra(self):
        """Test ExperimentParams collects extra fields."""
        # Extra fields should be passed via extra dict
        params = experiment_config.ExperimentParams(
            max_length=512, extra={"custom_param": "value", "another_param": 123}
        )
        self.assertEqual(params.max_length, 512)
        self.assertEqual(params.extra["custom_param"], "value")
        self.assertEqual(params.extra["another_param"], 123)

    def test_experiment_params_collect_extra_none(self):
        """Test ExperimentParams handles None extra fields."""
        params = experiment_config.ExperimentParams(max_length=512, extra=None)
        self.assertEqual(params.max_length, 512)
        self.assertEqual(params.extra, {})

    def test_experiment_params_collect_extra_dict(self):
        """Test ExperimentParams handles dict extra fields."""
        params = experiment_config.ExperimentParams(max_length=512, extra={"custom": "value"})
        self.assertEqual(params.max_length, 512)
        self.assertEqual(params.extra["custom"], "value")


class TestExperimentConfig(unittest.TestCase):
    """Test ExperimentConfig Pydantic model."""

    def test_experiment_config_minimal(self):
        """Test ExperimentConfig with minimal required fields."""
        cfg = experiment_config.ExperimentConfig(
            id="test_experiment",
            backend=experiment_config.HFBackendConfig(),
            prompts=experiment_config.PromptConfig(user="summarization/system_v1"),
            data=experiment_config.DataConfig(episodes_glob="*.txt"),
        )
        self.assertEqual(cfg.id, "test_experiment")
        self.assertEqual(cfg.task, "summarization")  # default
        self.assertEqual(cfg.backend.type, "hf_local")
        self.assertIsInstance(cfg.params, experiment_config.ExperimentParams)

    def test_experiment_config_with_openai(self):
        """Test ExperimentConfig with OpenAI backend."""
        cfg = experiment_config.ExperimentConfig(
            id="test_openai",
            backend=experiment_config.OpenAIBackendConfig(
                model=config.TEST_DEFAULT_OPENAI_SUMMARY_MODEL
            ),
            prompts=experiment_config.PromptConfig(user="summarization/system_v1"),
            data=experiment_config.DataConfig(episodes_glob="*.txt"),
        )
        self.assertEqual(cfg.id, "test_openai")
        self.assertEqual(cfg.backend.type, "openai")
        self.assertEqual(cfg.backend.model, config.TEST_DEFAULT_OPENAI_SUMMARY_MODEL)

    def test_experiment_config_ensure_non_empty_id(self):
        """Test ExperimentConfig validation ensures non-empty ID."""
        with self.assertRaises(ValidationError) as cm:
            experiment_config.ExperimentConfig(
                id="",
                backend=experiment_config.HFBackendConfig(),
                prompts=experiment_config.PromptConfig(user="summarization/system_v1"),
                data=experiment_config.DataConfig(episodes_glob="*.txt"),
            )
        self.assertIn("non-empty", str(cm.exception))

    def test_experiment_config_ensure_non_empty_id_whitespace(self):
        """Test ExperimentConfig validation ensures ID is not just whitespace."""
        with self.assertRaises(ValidationError) as cm:
            experiment_config.ExperimentConfig(
                id="   ",
                backend=experiment_config.HFBackendConfig(),
                prompts=experiment_config.PromptConfig(user="summarization/system_v1"),
                data=experiment_config.DataConfig(episodes_glob="*.txt"),
            )
        self.assertIn("non-empty", str(cm.exception))

    def test_experiment_config_with_task(self):
        """Test ExperimentConfig with explicit task."""
        cfg = experiment_config.ExperimentConfig(
            id="test_ner",
            task="ner_guest_host",
            backend=experiment_config.OpenAIBackendConfig(
                model=config.TEST_DEFAULT_OPENAI_SPEAKER_MODEL
            ),
            prompts=experiment_config.PromptConfig(user="ner/guest_host_v1"),
            data=experiment_config.DataConfig(episodes_glob="*.txt"),
        )
        self.assertEqual(cfg.task, "ner_guest_host")


class TestLoadExperimentConfig(unittest.TestCase):
    """Test load_experiment_config function."""

    def test_load_experiment_config_success(self):
        """Test loading valid experiment config from YAML."""
        import tempfile

        config_yaml = """
id: test_experiment
task: summarization
backend:
  type: hf_local
  map_model: facebook/bart-base
prompts:
  user: summarization/system_v1
data:
  episodes_glob: "*.txt"
  id_from: parent_dir
params:
  max_length: 512
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            config_path = Path(f.name)

        try:

            cfg = experiment_config.load_experiment_config(config_path)
            self.assertEqual(cfg.id, "test_experiment")
            self.assertEqual(cfg.task, "summarization")
            self.assertEqual(cfg.backend.type, "hf_local")
            self.assertEqual(cfg.backend.map_model, "facebook/bart-base")
            self.assertEqual(cfg.prompts.user, "summarization/system_v1")
            self.assertEqual(cfg.data.episodes_glob, "*.txt")
            self.assertEqual(cfg.params.max_length, 512)
        finally:
            config_path.unlink(missing_ok=True)

    def test_load_experiment_config_file_not_found(self):
        """Test load_experiment_config raises FileNotFoundError for missing file."""
        config_path = Path("/nonexistent/path/config.yaml")
        with self.assertRaises(FileNotFoundError):
            experiment_config.load_experiment_config(config_path)

    def test_load_experiment_config_empty_file(self):
        """Test load_experiment_config raises ValueError for empty file."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_path = Path(f.name)
            # File is empty (just created)

        try:
            with self.assertRaises(ValueError) as cm:
                experiment_config.load_experiment_config(config_path)
            self.assertIn("empty", str(cm.exception).lower())
        finally:
            config_path.unlink(missing_ok=True)

    def test_load_experiment_config_invalid_yaml(self):
        """Test load_experiment_config raises YAMLError for invalid YAML."""
        import tempfile

        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = Path(f.name)

        try:
            with self.assertRaises(yaml.YAMLError):
                experiment_config.load_experiment_config(config_path)
        finally:
            config_path.unlink(missing_ok=True)

    def test_load_experiment_config_invalid_schema(self):
        """Test load_experiment_config raises ValidationError for invalid schema."""
        import tempfile

        config_yaml = """
id: test_experiment
# Missing required fields: backend, prompts, data
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            config_path = Path(f.name)

        try:
            with self.assertRaises(ValidationError):
                experiment_config.load_experiment_config(config_path)
        finally:
            config_path.unlink(missing_ok=True)

    def test_load_experiment_config_with_path_string(self):
        """Test load_experiment_config accepts string path."""
        import tempfile

        config_yaml = """
id: test_experiment
backend:
  type: hf_local
prompts:
  user: summarization/system_v1
data:
  episodes_glob: "*.txt"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            config_path = Path(f.name)

        try:
            cfg = experiment_config.load_experiment_config(str(config_path))
            self.assertEqual(cfg.id, "test_experiment")
        finally:
            config_path.unlink(missing_ok=True)


class TestDiscoverInputFiles(unittest.TestCase):
    """Test discover_input_files function."""

    def test_discover_input_files_simple_glob(self):
        """Test discovering files with simple glob pattern."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            (Path(temp_dir) / "file1.txt").write_text("content1")
            (Path(temp_dir) / "file2.txt").write_text("content2")
            (Path(temp_dir) / "file3.md").write_text("content3")

            data_cfg = experiment_config.DataConfig(episodes_glob="*.txt")
            files = experiment_config.discover_input_files(data_cfg, base_dir=Path(temp_dir))

            self.assertEqual(len(files), 2)
            file_names = {f.name for f in files}
            self.assertEqual(file_names, {"file1.txt", "file2.txt"})

    def test_discover_input_files_nested_glob(self):
        """Test discovering files with nested glob pattern."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested structure
            (Path(temp_dir) / "ep01").mkdir(parents=True, exist_ok=True)
            (Path(temp_dir) / "ep01" / "transcript.txt").write_text("content1")
            (Path(temp_dir) / "ep02").mkdir(parents=True, exist_ok=True)
            (Path(temp_dir) / "ep02" / "transcript.txt").write_text("content2")

            data_cfg = experiment_config.DataConfig(episodes_glob="ep*/transcript.txt")
            files = experiment_config.discover_input_files(data_cfg, base_dir=Path(temp_dir))

            self.assertEqual(len(files), 2)
            file_names = {f.name for f in files}
            self.assertEqual(file_names, {"transcript.txt"})  # Both have same name

    def test_discover_input_files_no_matches(self):
        """Test discovering files when no files match glob."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            data_cfg = experiment_config.DataConfig(episodes_glob="*.nonexistent")
            files = experiment_config.discover_input_files(data_cfg, base_dir=Path(temp_dir))

            self.assertEqual(len(files), 0)

    def test_discover_input_files_filters_directories(self):
        """Test that discover_input_files filters out directories."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a directory that matches the glob
            (Path(temp_dir) / "file.txt").mkdir(exist_ok=True)
            (Path(temp_dir) / "actual_file.txt").write_text("content")

            data_cfg = experiment_config.DataConfig(episodes_glob="*.txt")
            files = experiment_config.discover_input_files(data_cfg, base_dir=Path(temp_dir))

            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].name, "actual_file.txt")

    # Note: test_discover_input_files_default_base_dir removed
    # Testing base_dir=None would require filesystem access to current directory,
    # which is blocked in unit tests. The function's behavior with explicit base_dir
    # is already tested in other test methods.

    def test_discover_input_files_sorted(self):
        """Test that discover_input_files returns sorted files."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files in non-alphabetical order
            (Path(temp_dir) / "z_file.txt").write_text("content")
            (Path(temp_dir) / "a_file.txt").write_text("content")
            (Path(temp_dir) / "m_file.txt").write_text("content")

            data_cfg = experiment_config.DataConfig(episodes_glob="*.txt")
            files = experiment_config.discover_input_files(data_cfg, base_dir=Path(temp_dir))

            self.assertEqual(len(files), 3)
            self.assertEqual(files[0].name, "a_file.txt")
            self.assertEqual(files[1].name, "m_file.txt")
            self.assertEqual(files[2].name, "z_file.txt")


class TestEpisodeIdFromPath(unittest.TestCase):
    """Test episode_id_from_path function."""

    def test_episode_id_from_path_stem(self):
        """Test episode ID extraction from file stem."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ep01" / "transcript.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("content")

            data_cfg = experiment_config.DataConfig(episodes_glob="*.txt", id_from="stem")
            episode_id = experiment_config.episode_id_from_path(path, data_cfg)

            self.assertEqual(episode_id, "transcript")

    def test_episode_id_from_path_parent_dir(self):
        """Test episode ID extraction from parent directory."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ep01" / "transcript.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("content")

            data_cfg = experiment_config.DataConfig(episodes_glob="*.txt", id_from="parent_dir")
            episode_id = experiment_config.episode_id_from_path(path, data_cfg)

            self.assertEqual(episode_id, "ep01")

    def test_episode_id_from_path_parent_dir_default(self):
        """Test episode ID extraction defaults to parent_dir."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ep01" / "transcript.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("content")

            data_cfg = experiment_config.DataConfig(episodes_glob="*.txt")  # default id_from
            episode_id = experiment_config.episode_id_from_path(path, data_cfg)

            self.assertEqual(episode_id, "ep01")

    def test_episode_id_from_path_stem_with_dots(self):
        """Test episode ID extraction from stem with dots in filename."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ep01" / "transcript.v2.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("content")

            data_cfg = experiment_config.DataConfig(episodes_glob="*.txt", id_from="stem")
            episode_id = experiment_config.episode_id_from_path(path, data_cfg)

            self.assertEqual(episode_id, "transcript.v2")

    def test_episode_id_from_path_root_parent(self):
        """Test episode ID extraction when file is in root directory."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "transcript.txt"
            path.write_text("content")

            data_cfg = experiment_config.DataConfig(episodes_glob="*.txt", id_from="parent_dir")
            episode_id = experiment_config.episode_id_from_path(path, data_cfg)

            # Parent of temp_dir would be system temp, so we check it's not empty
            self.assertIsInstance(episode_id, str)
            self.assertGreater(len(episode_id), 0)
