"""Experiment configuration models for LLM evaluation.

- Keeps prompts, models, params, and data paths in a single typed config.
- Designed to work with both local HF models and OpenAI backends.
- Uses Pydantic for validation and type safety.

This module implements RFC-017: Prompt Management and Loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

# ----- Prompt configuration -----


class PromptConfig(BaseModel):
    """Configuration for prompts used in an experiment.

    Example YAML:
      prompts:
        system: "summarization/system_v1"
        user:   "summarization/long_v2_more_narrative"
        params:
          paragraphs_min: 3
          paragraphs_max: 6
    """

    system: Optional[str] = Field(
        default=None,
        description="Logical name for system prompt template (or None).",
    )
    user: str = Field(
        description="Logical name for user prompt template.",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Template parameters to render into prompts.",
    )


# ----- Backend / model configuration -----


class HFBackendConfig(BaseModel):
    """Config for local Hugging Face models (your existing BART/LED setup)."""

    type: Literal["hf_local"] = "hf_local"
    map_model: Optional[str] = Field(
        default=None,
        description="Model name for map stage (optional, summarization only).",
    )
    reduce_model: Optional[str] = Field(
        default=None,
        description="Model name for reduce stage (optional, summarization only).",
    )
    model: Optional[str] = Field(
        default=None,
        description="Single HF model (for e.g. single-pass tasks).",
    )


class OpenAIBackendConfig(BaseModel):
    """Config for OpenAI models (summarization, NER, etc.)."""

    type: Literal["openai"] = "openai"
    model: str = Field(
        description="OpenAI model name, e.g. 'gpt-4o-mini'.",
    )


BackendConfig = HFBackendConfig | OpenAIBackendConfig


# ----- Data configuration -----


class DataConfig(BaseModel):
    """Where to find input data for this experiment.

    Example YAML:
      data:
        episodes_glob: "data/episodes/ep*/transcript.txt"
        id_from: "parent_dir"   # or "stem"
    """

    episodes_glob: str = Field(
        description="Glob pattern to discover episode input files.",
    )
    id_from: Literal["stem", "parent_dir"] = Field(
        default="parent_dir",
        description=(
            "How to derive episode_id from path. "
            "'stem' -> filename without extension; "
            "'parent_dir' -> parent folder name."
        ),
    )


# ----- Top-level experiment config -----


class ExperimentParams(BaseModel):
    """Task-specific parameters.

    For summarization, you might use:
      max_length, min_length, chunk_size, etc.

    For NER, maybe:
      max_output_tokens, schema variant, etc.
    """

    # Common parameters
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    chunk_size: Optional[int] = None
    word_chunk_size: Optional[int] = None
    word_overlap: Optional[int] = None
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None

    # Allow arbitrary extra keys for specific experiments
    extra: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("extra", mode="before")
    @classmethod
    def collect_extra(cls, v: Any) -> Dict[str, Any]:
        """Collect any extra fields not explicitly defined."""
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        return {}


class ExperimentConfig(BaseModel):
    """Full configuration for a single experiment run.

    Example YAML:

      id: "summarization_openai_long_v2"
      task: "summarization"

      backend:
        type: "openai"
        model: "gpt-4o-mini"

      prompts:
        system: "summarization/system_v1"
        user:   "summarization/long_v2_more_narrative"
        params:
          paragraphs_min: 3
          paragraphs_max: 6

      data:
        episodes_glob: "data/episodes/ep*/transcript.txt"
        id_from: "parent_dir"

      params:
        max_output_tokens: 900
    """

    id: str
    task: Literal["summarization", "ner_guest_host", "ner_generic", "transcription"] = Field(
        default="summarization"
    )
    backend: BackendConfig
    prompts: PromptConfig
    data: DataConfig
    params: ExperimentParams = Field(default_factory=ExperimentParams)

    @field_validator("id")
    @classmethod
    def ensure_non_empty_id(cls, v: str) -> str:
        """Ensure experiment ID is non-empty."""
        if not v.strip():
            raise ValueError("Experiment id must be non-empty")
        return v


# ----- Loader helpers -----


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML experiment config into a typed ExperimentConfig.

    Args:
        path: Path to YAML config.

    Returns:
        ExperimentConfig instance.

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        ValidationError: If config doesn't match ExperimentConfig schema
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment config file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"Experiment config file is empty: {path}")

    return ExperimentConfig(**raw)


def discover_input_files(data_cfg: DataConfig, base_dir: Path | None = None) -> List[Path]:
    """Discover input files according to the experiment's data config.

    Args:
        data_cfg: Data configuration
        base_dir: Base directory for glob (default: current directory)

    Returns:
        List of discovered file paths, sorted
    """
    if base_dir is None:
        base_dir = Path(".")
    paths = sorted(base_dir.glob(data_cfg.episodes_glob))
    return [p for p in paths if p.is_file()]


def episode_id_from_path(path: Path, data_cfg: DataConfig) -> str:
    """Convert a file path to an episode_id using the data config's id_from rule.

    Args:
        path: File path
        data_cfg: Data configuration

    Returns:
        Episode ID string
    """
    if data_cfg.id_from == "stem":
        return path.stem
    # default: parent_dir
    return path.parent.name
