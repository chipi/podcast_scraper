"""Experiment configuration models for LLM evaluation.

- Keeps prompts, models, params, and data paths in a single typed config.
- Designed to work with both local HF models and OpenAI backends.
- Uses Pydantic for validation and type safety.

This module implements RFC-017: Prompt Management and Loading.

Note: This module was moved from root-level experiment_config.py to evaluation/experiment_config.py
for better organization. The evaluation system uses this configuration exclusively.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# ----- Prompt configuration -----


class PromptConfig(BaseModel):
    """Configuration for prompts used in an experiment.

    Example YAML:
      prompts:
        system: "openai/summarization/system_v1"
        user:   "openai/summarization/long_v2_more_narrative"
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


class SpacyBackendConfig(BaseModel):
    """Config for local spaCy models (NER tasks)."""

    type: Literal["spacy_local"] = "spacy_local"
    model: str = Field(
        description="spaCy model name, e.g. 'en_core_web_sm' or 'en_core_web_trf'.",
    )


class OpenAIBackendConfig(BaseModel):
    """Config for OpenAI models (summarization, NER, etc.)."""

    type: Literal["openai"] = "openai"
    model: str = Field(
        description="OpenAI model name, e.g. 'gpt-4o-mini'.",
    )


class GeminiBackendConfig(BaseModel):
    """Config for Google Gemini models (summarization)."""

    type: Literal["gemini"] = "gemini"
    model: str = Field(
        description="Gemini model name, e.g. 'gemini-2.0-flash'.",
    )


class HybridMLBackendConfig(BaseModel):
    """Config for hybrid MAP-REDUCE (RFC-042): classic MAP + instruction-tuned REDUCE."""

    type: Literal["hybrid_ml"] = "hybrid_ml"
    map_model: str = Field(
        default="longt5-base",
        description="MAP model alias or ID (e.g. longt5-base, google/long-t5-tglobal-base).",
    )
    reduce_model: str = Field(
        default="google/flan-t5-base",
        description=(
            "REDUCE model: HuggingFace ID for transformers, "
            "Ollama tag for ollama, or path to GGUF for llama_cpp."
        ),
    )
    reduce_backend: Literal["transformers", "ollama", "llama_cpp"] = Field(
        default="transformers",
        description="REDUCE backend: transformers (FLAN-T5), ollama, or llama_cpp.",
    )
    reduce_instruction_style: Optional[Literal["structured", "paragraph"]] = Field(
        default=None,
        description=(
            "REDUCE instruction: 'structured' = Takeaways/Outline/Actions; "
            "'paragraph' = silver-style paragraphs. Default: structured."
        ),
    )


class AnthropicBackendConfig(BaseModel):
    """Config for Anthropic models (summarization)."""

    type: Literal["anthropic"] = "anthropic"
    model: str = Field(
        description="Anthropic model name, e.g. 'claude-3-5-sonnet-20241022'.",
    )


class MistralBackendConfig(BaseModel):
    """Config for Mistral models (summarization)."""

    type: Literal["mistral"] = "mistral"
    model: str = Field(
        description="Mistral model name, e.g. 'mistral-small-latest'.",
    )


class OllamaBackendConfig(BaseModel):
    """Config for local Ollama models (summarization only, single-pass LLM)."""

    type: Literal["ollama"] = "ollama"
    model: str = Field(
        description="Ollama model tag, e.g. 'qwen2.5:7b', 'llama3.1:8b'.",
    )


class GrokBackendConfig(BaseModel):
    """Config for Grok models (summarization)."""

    type: Literal["grok"] = "grok"
    model: str = Field(
        description="Grok model name, e.g. 'grok-2'.",
    )


class DeepSeekBackendConfig(BaseModel):
    """Config for DeepSeek models (summarization)."""

    type: Literal["deepseek"] = "deepseek"
    model: str = Field(
        description="DeepSeek model name, e.g. 'deepseek-chat'.",
    )


class EvalStubBackendConfig(BaseModel):
    """Eval-only backend: no external model (GIL/KG stub pipeline in ``run_experiment``)."""

    type: Literal["eval_stub"] = "eval_stub"


BackendConfig = (
    HFBackendConfig
    | OpenAIBackendConfig
    | GeminiBackendConfig
    | SpacyBackendConfig
    | HybridMLBackendConfig
    | AnthropicBackendConfig
    | MistralBackendConfig
    | OllamaBackendConfig
    | GrokBackendConfig
    | DeepSeekBackendConfig
    | EvalStubBackendConfig
)


# ----- Data configuration -----


class DataConfig(BaseModel):
    """Where to find input data for this experiment.

    Preferred: dataset-based mode with dataset_id and files under data/eval/datasets/.
    Glob-based mode (episodes_glob) and benchmarks/datasets/ are supported for
    backward compatibility; prefer data/eval/datasets/ for new experiments.

    Supports two modes:
    1. Dataset-based (recommended): Use dataset_id to load from data/eval/datasets/*.json
    2. Glob-based (legacy): Use episodes_glob pattern for backward compatibility

    Example YAML (dataset-based):
      data:
        dataset_id: "indicator_v1"

    Example YAML (glob-based):
      data:
        episodes_glob: "data/episodes/ep*/transcript.txt"
        id_from: "parent_dir"   # or "stem"
    """

    dataset_id: Optional[str] = Field(
        default=None,
        description=(
            "Dataset identifier (e.g., 'indicator_v1'). "
            "Loads from data/eval/datasets/{dataset_id}.json "
            "(or benchmarks/datasets/ as fallback)."
        ),
    )
    episodes_glob: Optional[str] = Field(
        default=None,
        description="Glob pattern to discover episode input files (legacy mode).",
    )
    id_from: Literal["stem", "parent_dir"] = Field(
        default="parent_dir",
        description=(
            "How to derive episode_id from path (legacy mode only). "
            "'stem' -> filename without extension; "
            "'parent_dir' -> parent folder name."
        ),
    )
    max_episodes: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "If set, only the first N episodes (after stable sort) are processed or scored. "
            "Used for smoke runs and autoresearch cost control (RFC-057)."
        ),
    )

    @model_validator(mode="after")
    def validate_data_source(self) -> "DataConfig":
        """Ensure exactly one data source is provided."""
        if not self.dataset_id and not self.episodes_glob:
            raise ValueError(
                "Either 'dataset_id' or 'episodes_glob' must be provided in data config"
            )
        if self.dataset_id and self.episodes_glob:
            raise ValueError(
                "Cannot specify both 'dataset_id' and 'episodes_glob'. "
                "Use 'dataset_id' for dataset-based mode or 'episodes_glob' for legacy mode."
            )
        return self


# ----- Generation parameters -----


class GenerationParams(BaseModel):
    """Generation parameters for transformer models.

    These parameters are passed directly to model.generate() calls.
    Uses max_new_tokens/min_new_tokens for clarity (output-only tokens).

    Example YAML:
      max_new_tokens: 200
      min_new_tokens: 80
      num_beams: 4
      no_repeat_ngram_size: 3
      length_penalty: 1.0
      early_stopping: true
    """

    max_new_tokens: int = Field(
        ge=1,
        description="Maximum number of new tokens to generate (output-only, not including input)",
    )
    min_new_tokens: int = Field(
        ge=1,
        description="Minimum number of new tokens to generate (output-only, not including input)",
    )
    num_beams: int = Field(
        default=4,
        ge=1,
        description="Number of beams for beam search (higher = better quality, slower)",
    )
    no_repeat_ngram_size: int = Field(
        default=3,
        ge=1,
        description="Size of n-grams to prevent repeating (prevents repetition/hallucination)",
    )
    length_penalty: float = Field(
        default=1.0,
        ge=0.0,
        description="Length penalty for beam search (1.0 = no penalty, >1.0 = prefer longer)",
    )
    early_stopping: bool = Field(
        default=True,
        description="Stop generation when all beams agree (beam search only)",
    )
    repetition_penalty: float = Field(
        default=1.3,
        ge=1.0,
        le=2.0,
        description=(
            "Repetition penalty (1.0 = no penalty, >1.0 = penalize repetition, "
            "prevents hallucinations)"
        ),
    )
    encoder_no_repeat_ngram_size: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Prevents copying n-grams directly from input "
            "(None = disabled, 3 = prevent 3-grams from input)"
        ),
    )


class TokenizeConfig(BaseModel):
    """Tokenization configuration for input text.

    Controls how input text is tokenized before being passed to the model.
    Separate from generation params to avoid confusion.

    Example YAML:
      map_max_input_tokens: 1024
      reduce_max_input_tokens: 4096
      truncation: true
    """

    map_max_input_tokens: int = Field(
        ge=1,
        description="Maximum input tokens for map stage (truncation limit)",
    )
    reduce_max_input_tokens: int = Field(
        ge=1,
        description="Maximum input tokens for reduce stage (truncation limit)",
    )
    truncation: bool = Field(
        default=True,
        description="Whether to truncate input text that exceeds max_input_tokens",
    )


class ChunkingConfig(BaseModel):
    """Chunking configuration for long text processing.

    Controls how long transcripts are split into chunks for processing.
    Chunking is critical for quality, cost, latency, and reproducibility.

    Example YAML:
      strategy: "word_chunking"
      word_chunk_size: 900
      word_overlap: 150
    """

    strategy: Literal["word_chunking", "token_chunking", "none"] = Field(
        default="word_chunking",
        description="Chunking strategy: 'word_chunking' (recommended for BART/PEGASUS), "
        "'token_chunking' (for models with large context windows), or 'none' (no chunking)",
    )
    word_chunk_size: int = Field(
        default=900,
        ge=1,
        description="Chunk size in words (for word_chunking strategy, 800-1200 recommended)",
    )
    word_overlap: int = Field(
        default=150,
        ge=0,
        description=(
            "Overlap in words between chunks " "(for word_chunking strategy, 100-200 recommended)"
        ),
    )


# ----- Top-level experiment config -----


class ExperimentConfig(BaseModel):
    """Full configuration for a single experiment run.

    Example YAML (new structure with explicit ML params):

      id: "baseline_bart_small_led_long_fast"
      task: "summarization"

      backend:
        type: "hf_local"
        map_model: "bart-small"
        reduce_model: "long-fast"

      data:
        dataset_id: "curated_5feeds_smoke_v1"

      map_params:
        max_new_tokens: 200
        min_new_tokens: 80
        num_beams: 4
        no_repeat_ngram_size: 3
        length_penalty: 1.0
        early_stopping: true

      reduce_params:
        max_new_tokens: 650
        min_new_tokens: 220
        num_beams: 4
        no_repeat_ngram_size: 3
        length_penalty: 1.0
        early_stopping: true

      tokenize:
        map_max_input_tokens: 1024
        reduce_max_input_tokens: 4096
        truncation: true

      chunking:
        strategy: "word_chunking"
        word_chunk_size: 900
        word_overlap: 150

    """

    id: str
    task: Literal[
        "summarization",
        "ner_entities",
        "ner_guest_host",
        "ner_generic",
        "transcription",
        "grounded_insights",
        "knowledge_graph",
    ] = Field(default="summarization")
    backend: BackendConfig
    prompts: Optional[PromptConfig] = Field(
        default=None,
        description=(
            "Prompt configuration (required for openai, gemini, and ollama backends; "
            "optional for hf_local / hybrid_ml)."
        ),
    )
    data: DataConfig
    # ML params (required for hf_local backend)
    map_params: Optional[GenerationParams] = Field(
        default=None,
        description="Generation parameters for map stage (required for hf_local backend)",
    )
    reduce_params: Optional[GenerationParams] = Field(
        default=None,
        description="Generation parameters for reduce stage (required for hf_local backend)",
    )
    tokenize: Optional[TokenizeConfig] = Field(
        default=None,
        description="Tokenization configuration for input text (required for hf_local backend)",
    )
    chunking: Optional[ChunkingConfig] = Field(
        default=None,
        description=(
            "Chunking configuration for long text processing "
            "(optional, uses defaults if not specified)"
        ),
    )
    preprocessing_profile: str = Field(
        default="cleaning_v3",
        description=(
            "Preprocessing profile ID to use for text cleaning "
            "(e.g., 'cleaning_v3', 'cleaning_v4')"
        ),
    )
    transcript_cleaning_strategy: Optional[Literal["pattern", "llm", "hybrid"]] = Field(
        default=None,
        description=(
            "When set, passed to podcast_scraper.config.Config.transcript_cleaning_strategy "
            "for summarization experiments (pattern / llm / hybrid LLM cleaning). "
            "When omitted, product default applies (typically hybrid for API LLM providers)."
        ),
    )
    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Backend-specific parameters " "(e.g., max_length, min_length, temperature for OpenAI)"
        ),
    )

    @field_validator("id")
    @classmethod
    def ensure_non_empty_id(cls, v: str) -> str:
        """Ensure experiment ID is non-empty."""
        if not v.strip():
            raise ValueError("Experiment id must be non-empty")
        return v

    @model_validator(mode="after")
    def validate_prompts_for_backend(self) -> "ExperimentConfig":
        """Validate that prompts are provided for OpenAI backend."""
        if self.backend.type == "eval_stub":
            return self
        if self.backend.type in ("openai", "gemini") and not self.prompts:
            raise ValueError(
                "prompts are required for openai and gemini backends. "
                "Provide a prompts section with at least a user prompt."
            )
        if self.backend.type == "ollama" and not self.prompts:
            raise ValueError(
                "prompts are required for Ollama backend. "
                "Provide prompts.user (and optionally prompts.system), "
                "e.g. ollama/summarization/long_v1."
            )
        return self

    @model_validator(mode="after")
    def validate_eval_stub_task_pairing(self) -> "ExperimentConfig":
        """eval_stub is only valid for GIL/KG eval tasks."""
        if self.backend.type == "eval_stub" and self.task not in (
            "grounded_insights",
            "knowledge_graph",
        ):
            raise ValueError(
                "eval_stub backend only supports tasks 'grounded_insights' or "
                f"'knowledge_graph' (got task={self.task!r})"
            )
        # grounded_insights / knowledge_graph may use eval_stub (cheap CI) or any
        # summarization-capable backend (regenerate summary, then GI/KG pipelines).
        return self

    @model_validator(mode="after")
    def validate_ml_params_for_hf_local(self) -> "ExperimentConfig":
        """Validate that ML params are required for hf_local and hybrid_ml backends."""
        if self.backend.type in ("hf_local", "hybrid_ml"):
            if not self.map_params:
                raise ValueError(
                    f"map_params is required for {self.backend.type} backend. "
                    "Provide map_params with generation parameters for map stage."
                )
            if not self.reduce_params:
                raise ValueError(
                    f"reduce_params is required for {self.backend.type} backend. "
                    "Provide reduce_params with generation parameters for reduce stage."
                )
            if not self.tokenize:
                raise ValueError(
                    f"tokenize is required for {self.backend.type} backend. "
                    "Provide tokenize with tokenization limits."
                )
        return self


# ----- Loader helpers -----


def load_dataset_json(dataset_id: str) -> Dict[str, Any]:
    """Load dataset JSON definition.

    Preferred location: data/eval/datasets/{dataset_id}.json.
    Checks multiple locations in order:
    1. data/eval/datasets/ (preferred)
    2. benchmarks/datasets/ (legacy, for backward compatibility)

    Args:
        dataset_id: Dataset identifier (e.g., "indicator_v1")

    Returns:
        Dataset dictionary with episodes array

    Raises:
        FileNotFoundError: If dataset JSON doesn't exist in any location
    """
    # Try current location first
    dataset_path = Path("data/eval/datasets") / f"{dataset_id}.json"
    if not dataset_path.exists():
        # Fallback to legacy location
        dataset_path = Path("benchmarks/datasets") / f"{dataset_id}.json"
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset definition not found: {dataset_id}\n"
                f"  Searched in: data/eval/datasets/{dataset_id}.json\n"
                f"  Searched in: benchmarks/datasets/{dataset_id}.json"
            )
    return dict(json.loads(dataset_path.read_text(encoding="utf-8")))  # type: ignore[no-any-return]


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

    Supports two modes:
    1. Dataset-based: Loads transcript paths from dataset JSON
    2. Glob-based: Uses glob pattern (legacy mode)

    Args:
        data_cfg: Data configuration
        base_dir: Base directory for glob (default: current directory, legacy mode only)

    Returns:
        List of discovered file paths, sorted

    Raises:
        ValueError: If neither dataset_id nor episodes_glob is provided
        FileNotFoundError: If dataset JSON doesn't exist or transcript files are missing
    """
    if data_cfg.dataset_id:
        # Dataset-based mode: load from materialized data ONLY
        dataset = load_dataset_json(data_cfg.dataset_id)

        # Materialized data is required - no fallback to sources
        materialized_dir = Path("data/eval/materialized") / data_cfg.dataset_id
        if not materialized_dir.exists():
            raise FileNotFoundError(
                f"Materialized dataset not found: {materialized_dir}\n"
                f"  Materialize the dataset first with: "
                f"make dataset-materialize DATASET_ID={data_cfg.dataset_id}"
            )

        meta_json = materialized_dir / "meta.json"
        if not meta_json.exists():
            raise FileNotFoundError(
                f"Materialized dataset metadata not found: {meta_json}\n"
                f"  Materialize the dataset first with: "
                f"make dataset-materialize DATASET_ID={data_cfg.dataset_id}"
            )

        paths = []
        for episode in dataset.get("episodes", []):
            episode_id = episode["episode_id"]
            materialized_path = materialized_dir / f"{episode_id}.txt"

            if not materialized_path.exists():
                raise FileNotFoundError(
                    f"Materialized transcript not found for episode "
                    f"{episode_id}: {materialized_path}\n"
                    f"  Materialize the dataset first with: "
                    f"make dataset-materialize DATASET_ID={data_cfg.dataset_id}"
                )

            paths.append(materialized_path)

        return sorted(paths)
    elif data_cfg.episodes_glob:
        # Legacy glob-based mode
        if base_dir is None:
            base_dir = Path(".")
        paths = sorted(base_dir.glob(data_cfg.episodes_glob))
        return [p for p in paths if p.is_file()]
    else:
        raise ValueError("Either dataset_id or episodes_glob must be provided in data config")


def discover_input_files_limited(data_cfg: DataConfig, base_dir: Path | None = None) -> List[Path]:
    """Like ``discover_input_files`` but applies ``data_cfg.max_episodes`` when set."""
    paths = discover_input_files(data_cfg, base_dir=base_dir)
    if data_cfg.max_episodes is not None:
        return paths[: data_cfg.max_episodes]
    return paths


def episode_id_from_path(path: Path, data_cfg: DataConfig) -> str:
    """Convert a file path to an episode_id.

    Supports two modes:
    1. Dataset-based: Looks up episode_id from dataset JSON by matching transcript_path
       or extracts from materialized path (e.g., p01_e01.txt -> p01_e01)
    2. Glob-based: Uses id_from rule (legacy mode)

    Args:
        path: File path
        data_cfg: Data configuration

    Returns:
        Episode ID string

    Raises:
        ValueError: If episode not found in dataset (dataset mode only)
    """
    if data_cfg.dataset_id:
        # Dataset-based mode: try to extract episode_id from materialized path first
        # Materialized paths are: data/eval/materialized/{dataset_id}/{episode_id}.txt
        materialized_dir = Path("data/eval/materialized") / data_cfg.dataset_id
        try:
            # Check if path is within materialized directory
            path_resolved = path.resolve()
            materialized_resolved = materialized_dir.resolve()
            if path_resolved.is_relative_to(materialized_resolved):
                # Extract episode_id from filename (e.g., p01_e01.txt -> p01_e01)
                episode_id = path.stem
                # Validate it exists in dataset
                dataset = load_dataset_json(data_cfg.dataset_id)
                episode_ids = {ep.get("episode_id") for ep in dataset.get("episodes", [])}
                if episode_id in episode_ids:
                    return episode_id
        except (ValueError, AttributeError):
            # Path is not relative to materialized dir, continue to lookup
            pass

        # Fallback: look up episode_id from dataset JSON by matching transcript_path
        dataset = load_dataset_json(data_cfg.dataset_id)
        for episode in dataset.get("episodes", []):
            episode_path = Path(episode["transcript_path"]).resolve()
            if episode_path == path.resolve():
                return str(episode["episode_id"])
        raise ValueError(f"Episode not found in dataset '{data_cfg.dataset_id}' for path: {path}")
    else:
        # Legacy glob-based mode: use id_from rule
        if data_cfg.id_from == "stem":
            return path.stem
        # default: parent_dir
        return path.parent.name
