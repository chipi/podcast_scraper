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

from podcast_scraper import config
from podcast_scraper.evaluation import experiment_config


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


class TestHybridMLBackendConfig(unittest.TestCase):
    """Test HybridMLBackendConfig Pydantic model."""

    def test_hybrid_ml_backend_config_defaults(self):
        """Test HybridMLBackendConfig with defaults."""
        cfg = experiment_config.HybridMLBackendConfig()
        self.assertEqual(cfg.type, "hybrid_ml")
        self.assertEqual(cfg.map_model, "longt5-base")
        self.assertEqual(cfg.reduce_model, "google/flan-t5-base")
        self.assertEqual(cfg.reduce_backend, "transformers")

    def test_hybrid_ml_backend_config_explicit(self):
        """Test HybridMLBackendConfig with explicit map/reduce/backend."""
        cfg = experiment_config.HybridMLBackendConfig(
            map_model="longt5-base",
            reduce_model="qwen2.5:7b",
            reduce_backend="ollama",
        )
        self.assertEqual(cfg.type, "hybrid_ml")
        self.assertEqual(cfg.map_model, "longt5-base")
        self.assertEqual(cfg.reduce_model, "qwen2.5:7b")
        self.assertEqual(cfg.reduce_backend, "ollama")


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
            experiment_config.OpenAIBackendConfig()  # type: ignore[call-arg]


class TestGeminiBackendConfig(unittest.TestCase):
    """Test GeminiBackendConfig Pydantic model."""

    def test_gemini_backend_config(self):
        """Test GeminiBackendConfig with model name."""
        cfg = experiment_config.GeminiBackendConfig(model="gemini-2.0-flash")
        self.assertEqual(cfg.type, "gemini")
        self.assertEqual(cfg.model, "gemini-2.0-flash")

    def test_gemini_backend_config_missing_model(self):
        """Test GeminiBackendConfig validation fails without required model field."""
        with self.assertRaises(ValidationError):
            experiment_config.GeminiBackendConfig()  # type: ignore[call-arg]


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


class TestExperimentConfig(unittest.TestCase):
    """Test ExperimentConfig Pydantic model."""

    def test_experiment_config_minimal(self):
        """Test ExperimentConfig with minimal required fields."""
        cfg = experiment_config.ExperimentConfig(
            id="test_experiment",
            backend=experiment_config.HFBackendConfig(),
            prompts=experiment_config.PromptConfig(user="summarization/system_v1"),
            data=experiment_config.DataConfig(episodes_glob="*.txt"),
            map_params=experiment_config.GenerationParams(max_new_tokens=200, min_new_tokens=80),
            reduce_params=experiment_config.GenerationParams(
                max_new_tokens=650, min_new_tokens=220
            ),
            tokenize=experiment_config.TokenizeConfig(
                map_max_input_tokens=1024, reduce_max_input_tokens=4096
            ),
        )
        self.assertEqual(cfg.id, "test_experiment")
        self.assertEqual(cfg.task, "summarization")  # default
        self.assertEqual(cfg.backend.type, "hf_local")

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

    def test_experiment_config_transcript_cleaning_strategy_optional(self):
        """transcript_cleaning_strategy is optional; when set it validates."""
        minimal = experiment_config.ExperimentConfig(
            id="test_openai_clean",
            backend=experiment_config.OpenAIBackendConfig(
                model=config.TEST_DEFAULT_OPENAI_SUMMARY_MODEL
            ),
            prompts=experiment_config.PromptConfig(user="openai/summarization/long_v1"),
            data=experiment_config.DataConfig(dataset_id="curated_5feeds_smoke_v1"),
            transcript_cleaning_strategy="pattern",
        )
        self.assertEqual(minimal.transcript_cleaning_strategy, "pattern")

    def test_load_transcript_cleaning_experiment_yaml_arm_a(self):
        """Load Arm A transcript-cleaning experiment config from disk."""
        repo_root = Path(__file__).resolve().parents[4]
        path = (
            repo_root
            / "data"
            / "eval"
            / "configs"
            / "experiment_openai_gpt4o_smoke_cleaning_arm_a_pattern_v1.yaml"
        )
        self.assertTrue(path.is_file(), f"missing fixture {path}")
        loaded = experiment_config.load_experiment_config(path)
        self.assertEqual(loaded.id, "experiment_openai_gpt4o_smoke_cleaning_arm_a_pattern_v1")
        self.assertEqual(loaded.transcript_cleaning_strategy, "pattern")

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

    def test_experiment_config_ollama_backend_requires_prompts(self):
        """Ollama backend requires prompts.user."""
        with self.assertRaises(ValueError):
            experiment_config.ExperimentConfig(
                id="test_ollama",
                backend=experiment_config.OllamaBackendConfig(model="qwen2.5:7b"),
                data=experiment_config.DataConfig(dataset_id="curated_5feeds_smoke_v1"),
                params={"max_length": 800, "min_length": 200, "temperature": 0.0},
            )

    def test_experiment_config_gemini_backend_requires_prompts(self):
        """Gemini backend requires prompts (validated like OpenAI)."""
        with self.assertRaises(ValueError):
            experiment_config.ExperimentConfig(
                id="test_gemini",
                backend=experiment_config.GeminiBackendConfig(model="gemini-2.0-flash"),
                data=experiment_config.DataConfig(dataset_id="curated_5feeds_smoke_v1"),
                params={"max_length": 800, "temperature": 0.0},
            )

    def test_experiment_config_with_hybrid_ml_backend(self):
        """Test ExperimentConfig with hybrid_ml backend."""
        cfg = experiment_config.ExperimentConfig(
            id="test_hybrid",
            backend=experiment_config.HybridMLBackendConfig(
                map_model="longt5-base",
                reduce_model="google/flan-t5-base",
                reduce_backend="transformers",
            ),
            data=experiment_config.DataConfig(dataset_id="curated_5feeds_smoke_v1"),
            map_params=experiment_config.GenerationParams(max_new_tokens=200, min_new_tokens=80),
            reduce_params=experiment_config.GenerationParams(
                max_new_tokens=650, min_new_tokens=220
            ),
            tokenize=experiment_config.TokenizeConfig(
                map_max_input_tokens=1024, reduce_max_input_tokens=4096
            ),
        )
        self.assertEqual(cfg.id, "test_hybrid")
        self.assertEqual(cfg.backend.type, "hybrid_ml")
        self.assertEqual(cfg.backend.map_model, "longt5-base")
        self.assertEqual(cfg.backend.reduce_model, "google/flan-t5-base")
        self.assertEqual(cfg.backend.reduce_backend, "transformers")


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
map_params:
  max_new_tokens: 200
  min_new_tokens: 80
reduce_params:
  max_new_tokens: 650
  min_new_tokens: 220
tokenize:
  map_max_input_tokens: 1024
  reduce_max_input_tokens: 4096
  truncation: true
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
            self.assertEqual(cfg.map_params.max_new_tokens, 200)
            self.assertEqual(cfg.reduce_params.max_new_tokens, 650)
            self.assertEqual(cfg.tokenize.map_max_input_tokens, 1024)
        finally:
            config_path.unlink(missing_ok=True)

    def test_load_experiment_config_ollama_backend_pure_llm(self):
        """Test loading autoresearch Ollama paragraph smoke config (pure LLM, not hybrid_ml)."""
        repo_root = Path(__file__).resolve().parents[4]
        cfg_name = "autoresearch_prompt_ollama_qwen25_7b_smoke_paragraph_v1.yaml"
        path = repo_root / "data/eval/configs/summarization" / cfg_name
        if not path.exists():
            self.skipTest(f"Config missing: {path}")
        cfg = experiment_config.load_experiment_config(path)
        self.assertEqual(cfg.backend.type, "ollama")
        self.assertEqual(cfg.backend.model, "qwen2.5:7b")
        self.assertIn("long_v1", cfg.prompts.user)
        self.assertIsNone(cfg.map_params)
        self.assertIsNone(cfg.reduce_params)

    def test_load_experiment_config_ollama_mistral_nemo_and_small32_smoke(self):
        """Test loading Mistral Nemo 12B and Mistral Small 3.2 Ollama smoke configs."""
        repo_root = Path(__file__).resolve().parents[4]
        cases = (
            (
                "summarization/autoresearch_prompt_ollama_mistral_nemo_12b_smoke_paragraph_v1.yaml",
                "autoresearch_prompt_ollama_mistral_nemo_12b_smoke_paragraph_v1",
                "mistral-nemo:12b",
                "ollama/mistral-nemo_12b/summarization/long_v1",
                "ollama/mistral-nemo_12b/summarization/system_v1",
            ),
            (
                "summarization/autoresearch_prompt_ollama_mistral_small3_2_smoke_paragraph_v1.yaml",
                "autoresearch_prompt_ollama_mistral_small3_2_smoke_paragraph_v1",
                "mistral-small3.2:latest",
                "ollama/mistral-small3.2/summarization/long_v1",
                "ollama/mistral-small3.2/summarization/system_v1",
            ),
        )
        for filename, exp_id, exp_model, exp_user, exp_system in cases:
            path = repo_root / "data/eval/configs" / filename
            if not path.exists():
                self.skipTest(f"Config missing: {path}")
            cfg = experiment_config.load_experiment_config(path)
            self.assertEqual(cfg.id, exp_id)
            self.assertEqual(cfg.backend.type, "ollama")
            self.assertEqual(cfg.backend.model, exp_model)
            self.assertEqual(cfg.prompts.user, exp_user)
            self.assertEqual(cfg.prompts.system, exp_system)
            self.assertIsNone(cfg.map_params)
            self.assertIsNone(cfg.reduce_params)

    def test_load_experiment_config_gemini_smoke(self):
        """Test loading autoresearch Gemini paragraph smoke config."""
        repo_root = Path(__file__).resolve().parents[4]
        path = (
            repo_root
            / "data/eval/configs/summarization/autoresearch_prompt_gemini_smoke_paragraph_v1.yaml"
        )
        if not path.exists():
            self.skipTest(f"Config missing: {path}")
        cfg = experiment_config.load_experiment_config(path)
        self.assertEqual(cfg.backend.type, "gemini")
        self.assertEqual(cfg.backend.model, "gemini-2.0-flash")
        self.assertEqual(cfg.prompts.user, "gemini/summarization/long_v1")
        self.assertEqual(cfg.prompts.system, "gemini/summarization/system_v1")
        self.assertIsNone(cfg.map_params)
        self.assertIsNone(cfg.reduce_params)

    def test_load_experiment_config_hybrid_ml_backend(self):
        """Test loading experiment config with hybrid_ml backend from YAML."""
        import tempfile

        config_yaml = """
id: test_hybrid_ml
task: summarization
backend:
  type: hybrid_ml
  map_model: longt5-base
  reduce_model: google/flan-t5-base
  reduce_backend: transformers
data:
  dataset_id: curated_5feeds_smoke_v1
map_params:
  max_new_tokens: 200
  min_new_tokens: 80
reduce_params:
  max_new_tokens: 650
  min_new_tokens: 220
tokenize:
  map_max_input_tokens: 1024
  reduce_max_input_tokens: 4096
  truncation: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            config_path = Path(f.name)

        try:
            cfg = experiment_config.load_experiment_config(config_path)
            self.assertEqual(cfg.id, "test_hybrid_ml")
            self.assertEqual(cfg.backend.type, "hybrid_ml")
            self.assertEqual(cfg.backend.map_model, "longt5-base")
            self.assertEqual(cfg.backend.reduce_model, "google/flan-t5-base")
            self.assertEqual(cfg.backend.reduce_backend, "transformers")
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
map_params:
  max_new_tokens: 200
  min_new_tokens: 80
reduce_params:
  max_new_tokens: 650
  min_new_tokens: 220
tokenize:
  map_max_input_tokens: 1024
  reduce_max_input_tokens: 4096
  truncation: true
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
