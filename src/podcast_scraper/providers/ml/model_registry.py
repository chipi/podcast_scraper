"""Model Registry: single source of truth for model capabilities AND pipeline-stage defaults.

**Original scope (ADR-048 / RFC-044)**: ML model architecture limits + mode
configurations. `ModelCapabilities` carries max input tokens, chunk defaults,
model family; `ModeConfiguration` carries promoted baseline configurations.
``get_capabilities()`` uses fallback order: registry → dynamic detection →
pattern-based guess → safe default.

**2026-06-12 amendment**: expanded to also hold the canonical defaults per
pipeline stage with research provenance. ``StageOption`` is one
provider/model choice for a stage (transcription, summary, etc.) with
``research_ref`` pointing back at the eval report that justified it.
``ProfilePreset`` is a named composition of StageOptions per stage. The
profile YAMLs in ``config/profiles/`` are downstream views — they should
match the registry preset they claim to derive from. Drift = bug.

The flow for any autoresearch finding that changes a default:

    run experiment → score → write eval report → MATERIALIZE here →
    REGENERATE profile YAML → behavior test

See ``docs/wip/RESEARCH_POWERED_REGISTRY_PLAN.md`` for the vision /
migration path, ``docs/adr/ADR-048-centralized-model-registry.md`` for the
amendment, and ``docs/guides/EXPERIMENT_GUIDE.md`` § Step 6 for the flow.
"""

from dataclasses import dataclass
from typing import Any, Dict, Final, Optional


@dataclass(frozen=True)
class ModelCapabilities:
    """Model architecture capabilities and limits.

    Generalized to support all model families: summarizers, instruction-tuned
    models, embedding models, extractive QA, and NLI cross-encoders.
    """

    # ── Core (all models) ──────────────────────
    max_input_tokens: int  # Maximum input tokens
    model_type: str  # "bart", "led", "flan-t5", etc.
    model_family: str  # "map", "reduce", "embedding", "extractive_qa", "nli"
    supports_long_context: bool  # >=4096 tokens

    # ── Summarizer-specific ────────────────────
    default_chunk_size: Optional[int] = None
    default_overlap: Optional[int] = None

    # ── Instruction-tuned model fields ─────────
    supports_json_output: bool = False
    supports_extraction: bool = False

    # ── Embedding model fields ─────────────────
    embedding_dim: Optional[int] = None

    # ── Resource estimates ─────────────────────
    memory_mb: Optional[int] = None
    default_device: str = "cpu"


@dataclass(frozen=True)
class ModeConfiguration:
    """Complete runtime configuration for a summarization mode.

    Promoted from proven baseline configurations. These modes can
    become app defaults while keeping runtime code decoupled from `data/eval/`.
    """

    mode_id: str
    map_model: str
    reduce_model: str
    preprocessing_profile: str
    map_params: Dict[str, Any]
    reduce_params: Optional[Dict[str, Any]]
    tokenize: Dict[str, Any]
    chunking: Optional[Dict[str, Any]]
    promoted_from: str
    promoted_at: str
    # Hybrid ollama-reduce fields (set when reduce_backend == "ollama")
    reduce_backend: Optional[str] = None
    reduce_instruction_style: Optional[str] = None
    ollama_reduce_params: Optional[Dict[str, Any]] = None
    metrics_summary: Optional[Dict[str, Any]] = None
    deprecated_at: Optional[str] = None
    deprecation_reason: Optional[str] = None


def _bart_caps(
    memory_mb: int = 500,
    default_chunk_size: int = 600,
    default_overlap: int = 60,
) -> ModelCapabilities:
    """BART-style capabilities (1024 tokens)."""
    return ModelCapabilities(
        max_input_tokens=1024,
        model_type="bart",
        model_family="map",
        supports_long_context=False,
        default_chunk_size=default_chunk_size,
        default_overlap=default_overlap,
        memory_mb=memory_mb,
    )


def _pegasus_caps(memory_mb: int = 2000) -> ModelCapabilities:
    """PEGASUS-style capabilities (1024 tokens)."""
    return ModelCapabilities(
        max_input_tokens=1024,
        model_type="pegasus",
        model_family="map",
        supports_long_context=False,
        default_chunk_size=600,
        default_overlap=60,
        memory_mb=memory_mb,
    )


def _led_caps(
    memory_mb: int = 1000,
    default_chunk_size: int = 16384,
    default_overlap: int = 1638,
) -> ModelCapabilities:
    """LED-style capabilities (16384 tokens)."""
    return ModelCapabilities(
        max_input_tokens=16384,
        model_type="led",
        model_family="map",
        supports_long_context=True,
        default_chunk_size=default_chunk_size,
        default_overlap=default_overlap,
        memory_mb=memory_mb,
    )


def _longt5_caps(
    memory_mb: int = 1000,
    default_chunk_size: int = 8192,
    default_overlap: int = 819,
) -> ModelCapabilities:
    """LongT5-style capabilities (8192 tokens)."""
    return ModelCapabilities(
        max_input_tokens=8192,
        model_type="longt5",
        model_family="map",
        supports_long_context=True,
        default_chunk_size=default_chunk_size,
        default_overlap=default_overlap,
        memory_mb=memory_mb,
    )


class ModelRegistry:
    """Centralized registry of model capabilities and metadata."""

    _registry: Dict[str, ModelCapabilities] = {
        # ── BART models (1024 token limit) ─────
        "bart-large": _bart_caps(memory_mb=1600),
        "bart-small": _bart_caps(memory_mb=500),
        "facebook/bart-large-cnn": _bart_caps(memory_mb=1600),
        "facebook/bart-base": _bart_caps(memory_mb=500),
        "fast": _bart_caps(memory_mb=1200),  # DistilBART alias
        "sshleifer/distilbart-cnn-12-6": _bart_caps(memory_mb=1200),
        # ── PEGASUS models (1024 token limit) ──
        "pegasus": _pegasus_caps(),
        "pegasus-cnn": _pegasus_caps(),
        "pegasus-xsum": _pegasus_caps(),
        "google/pegasus-large": _pegasus_caps(),
        "google/pegasus-cnn_dailymail": _pegasus_caps(),
        "google/pegasus-xsum": _pegasus_caps(),
        # ── LED models (16384 token limit) ─────
        "long": _led_caps(memory_mb=2000),
        "long-large": _led_caps(memory_mb=2000),
        "long-fast": _led_caps(memory_mb=1000),
        "allenai/led-large-16384": _led_caps(memory_mb=2000),
        "allenai/led-base-16384": _led_caps(memory_mb=1000),
        # ── LongT5 models (8192 token limit) ────
        "longt5-base": _longt5_caps(memory_mb=1000),
        "longt5-large": _longt5_caps(memory_mb=2500),
        "google/long-t5-tglobal-base": _longt5_caps(memory_mb=1000),
        "google/long-t5-tglobal-large": _longt5_caps(memory_mb=2500),
        # ── FLAN-T5 (Tier 1 REDUCE — PyTorch) ──
        "google/flan-t5-base": ModelCapabilities(
            max_input_tokens=512,
            model_type="flan-t5",
            model_family="reduce",
            supports_long_context=False,
            supports_json_output=True,
            supports_extraction=True,
            memory_mb=1000,
        ),
        "google/flan-t5-large": ModelCapabilities(
            max_input_tokens=512,
            model_type="flan-t5",
            model_family="reduce",
            supports_long_context=False,
            supports_json_output=True,
            supports_extraction=True,
            memory_mb=3000,
            default_device="mps",
        ),
        "google/flan-t5-xl": ModelCapabilities(
            max_input_tokens=512,
            model_type="flan-t5",
            model_family="reduce",
            supports_long_context=False,
            supports_json_output=True,
            supports_extraction=True,
            memory_mb=12000,
            default_device="cuda",
        ),
        # ── Embedding Models ────────────────────
        "sentence-transformers/all-MiniLM-L6-v2": ModelCapabilities(
            max_input_tokens=256,
            model_type="sentence-transformer",
            model_family="embedding",
            supports_long_context=False,
            embedding_dim=384,
            memory_mb=90,
        ),
        "sentence-transformers/all-MiniLM-L12-v2": ModelCapabilities(
            max_input_tokens=256,
            model_type="sentence-transformer",
            model_family="embedding",
            supports_long_context=False,
            embedding_dim=384,
            memory_mb=120,
        ),
        "sentence-transformers/all-mpnet-base-v2": ModelCapabilities(
            max_input_tokens=384,
            model_type="sentence-transformer",
            model_family="embedding",
            supports_long_context=False,
            embedding_dim=768,
            memory_mb=420,
        ),
        # ── Extractive QA Models ────────────────
        "deepset/roberta-base-squad2": ModelCapabilities(
            max_input_tokens=512,
            model_type="roberta",
            model_family="extractive_qa",
            supports_long_context=False,
            memory_mb=500,
        ),
        "deepset/deberta-v3-base-squad2": ModelCapabilities(
            max_input_tokens=512,
            model_type="deberta",
            model_family="extractive_qa",
            supports_long_context=False,
            memory_mb=700,
        ),
        # ── NLI Cross-Encoder Models ────────────
        "cross-encoder/nli-deberta-v3-base": ModelCapabilities(
            max_input_tokens=512,
            model_type="deberta-nli",
            model_family="nli",
            supports_long_context=False,
            memory_mb=400,
        ),
        "cross-encoder/nli-deberta-v3-small": ModelCapabilities(
            max_input_tokens=512,
            model_type="deberta-nli",
            model_family="nli",
            supports_long_context=False,
            memory_mb=200,
        ),
    }

    # Aliases for evidence stack (embedding, extractive QA, NLI) for hybrid MAP-REDUCE
    _evidence_aliases: Dict[str, str] = {
        "minilm-l6": "sentence-transformers/all-MiniLM-L6-v2",
        "minilm-l12": "sentence-transformers/all-MiniLM-L12-v2",
        "mpnet-base": "sentence-transformers/all-mpnet-base-v2",
        "roberta-squad2": "deepset/roberta-base-squad2",
        "deberta-squad2": "deepset/deberta-v3-base-squad2",
        "nli-deberta-base": "cross-encoder/nli-deberta-v3-base",
        "nli-deberta-small": "cross-encoder/nli-deberta-v3-small",
    }

    @classmethod
    def resolve_evidence_model_id(cls, model_id: str) -> str:
        """Resolve evidence-stack model alias to full HuggingFace ID.

        Args:
            model_id: Alias (e.g. 'minilm-l6') or full HF ID.

        Returns:
            Full HuggingFace model ID for loading.
        """
        if not isinstance(model_id, str):
            model_id = str(model_id)
        if "/" in model_id:
            return model_id
        if model_id in cls._evidence_aliases:
            return cls._evidence_aliases[model_id]
        raise ValueError(
            f"Unknown evidence model id: {model_id}. "
            f"Available aliases: {list(cls._evidence_aliases.keys())}. "
            "Or use a full HuggingFace model ID."
        )

    _mode_registry: Dict[str, ModeConfiguration] = {
        # BEGIN MODE REGISTRY (append-only)
        "ml_small_authority": ModeConfiguration(
            mode_id="ml_small_authority",
            map_model="bart-small",
            reduce_model="long-fast",
            preprocessing_profile="cleaning_v4",
            map_params={
                "max_new_tokens": 200,
                "min_new_tokens": 80,
                "num_beams": 4,
                "no_repeat_ngram_size": 3,
                "length_penalty": 1.0,
                "early_stopping": True,
                "repetition_penalty": 1.3,
            },
            reduce_params={
                "max_new_tokens": 650,
                "min_new_tokens": 220,
                "num_beams": 4,
                "no_repeat_ngram_size": 3,
                "length_penalty": 1.0,
                "early_stopping": True,
                "repetition_penalty": 1.3,
            },
            tokenize={
                "map_max_input_tokens": 1024,
                "reduce_max_input_tokens": 4096,
                "truncation": True,
            },
            chunking={"strategy": "word_chunking", "word_chunk_size": 900, "word_overlap": 150},
            promoted_from="baseline_ml_dev_authority_smoke_v1",
            promoted_at="2026-02-12T07:23:49Z",
            metrics_summary={
                "dataset_id": "curated_5feeds_smoke_v1",
                "run_id": "baseline_bart_v7_cleaning_v4",
                "episode_count": 5,
                "intrinsic": {
                    "gates": {
                        "boilerplate_leak_rate": 0.0,
                        "speaker_label_leak_rate": 0.0,
                        "truncation_rate": 0.0,
                    },
                    "length": {"avg_tokens": 470.4, "min_tokens": 183, "max_tokens": 666},
                    "performance": {"avg_latency_ms": 33627.17385292053},
                },
            },
        ),
        "ml_prod_authority_v1": ModeConfiguration(
            mode_id="ml_prod_authority_v1",
            map_model="google/pegasus-cnn_dailymail",
            reduce_model="allenai/led-base-16384",
            preprocessing_profile="cleaning_v4",
            map_params={
                "do_sample": False,
                "num_beams": 6,
                "max_new_tokens": 200,
                "min_new_tokens": 80,
                "no_repeat_ngram_size": 3,
                "repetition_penalty": 1.1,
                "length_penalty": 1.0,
                "early_stopping": True,
            },
            reduce_params={
                "do_sample": False,
                "num_beams": 4,
                "max_new_tokens": 650,
                "min_new_tokens": 220,
                "no_repeat_ngram_size": 3,
                "repetition_penalty": 1.12,
                "length_penalty": 1.0,
                "early_stopping": False,
            },
            tokenize={
                "map_max_input_tokens": 1024,
                "reduce_max_input_tokens": 4096,
                "truncation": True,
            },
            chunking={"strategy": "word_chunking", "word_chunk_size": 900, "word_overlap": 150},
            promoted_from="baseline_ml_prod_authority_v1",
            promoted_at="2026-02-12T07:23:49Z",
            metrics_summary={
                "dataset_id": "curated_5feeds_benchmark_v1",
                "run_id": "baseline_ml_prod_candidate_v1_benchmark",
                "episode_count": 10,
                "intrinsic": {
                    "gates": {
                        "boilerplate_leak_rate": 0.0,
                        "speaker_label_leak_rate": 0.0,
                        "truncation_rate": 0.0,
                    },
                    "length": {"avg_tokens": 228.8, "min_tokens": 207, "max_tokens": 246},
                    "performance": {"avg_latency_ms": 22964.020609855652},
                },
            },
            deprecated_at="2026-04-03T00:00:00Z",
            deprecation_reason=(
                "Pegasus (CNN/DailyMail) produces near-duplicate chunk summaries on podcast "
                "transcripts due to its news GSG pretraining objective. LED reduce then exhausts "
                "its no_repeat_ngram budget on redundant input and stops at ~55-70 tokens "
                "regardless of param tuning. Architectural mismatch for podcast content — not "
                "fixable without retraining. Superseded by ml_small_authority (BART+LED) as prod "
                "default and ml_longt5_led_v1 (LongT5+LED) as the next candidate. "
                "NOTE: Pegasus is well-suited for news content; re-evaluate when news is added "
                "as a content type. "
                "See baseline_ml_pegasus_retirement_smoke_v1 for the tombstone experiment."
            ),
        ),
        "ml_bart_led_autoresearch_v1": ModeConfiguration(
            mode_id="ml_bart_led_autoresearch_v1",
            map_model="bart-small",
            reduce_model="long-fast",
            preprocessing_profile="cleaning_v4",
            map_params={
                "do_sample": False,
                "num_beams": 4,
                "max_new_tokens": 200,
                "min_new_tokens": 80,
                "no_repeat_ngram_size": 3,
                "repetition_penalty": 1.3,
                "length_penalty": 1.0,
                "early_stopping": True,
            },
            reduce_params={
                "do_sample": False,
                "num_beams": 6,
                "max_new_tokens": 550,
                "min_new_tokens": 220,
                "no_repeat_ngram_size": 3,
                "repetition_penalty": 1.3,
                "length_penalty": 1.0,
                "early_stopping": True,
            },
            tokenize={
                "map_max_input_tokens": 1024,
                "reduce_max_input_tokens": 4096,
                "truncation": True,
            },
            chunking={"strategy": "word_chunking", "word_chunk_size": 900, "word_overlap": 150},
            promoted_from="baseline_ml_dev_authority",
            promoted_at="2026-04-03T00:00:00Z",
            metrics_summary={
                "dataset_id": "curated_5feeds_smoke_v1",
                "reference_id": "silver_sonnet46_smoke_v1",
                "rouge_l_f1": 0.1882,
                "embedding_cosine": 0.7259,
                "sweep_rounds": 2,
                "experiments_run": 11,
                "gains": {
                    "reduce_max_new_tokens": "+2.89% (550 vs 650)",
                    "reduce_num_beams": "+1.15% (6 vs 4)",
                },
            },
        ),
        "ml_hybrid_bart_llama32_3b_autoresearch_v1": ModeConfiguration(
            mode_id="ml_hybrid_bart_llama32_3b_autoresearch_v1",
            map_model="bart-small",
            reduce_model="llama3.2:3b",
            reduce_backend="ollama",
            reduce_instruction_style="paragraph",
            preprocessing_profile="cleaning_v4",
            map_params={
                "do_sample": False,
                "num_beams": 6,
                "max_new_tokens": 200,
                "min_new_tokens": 80,
                "no_repeat_ngram_size": 3,
                "repetition_penalty": 1.1,
                "length_penalty": 1.0,
                "early_stopping": True,
            },
            reduce_params=None,
            ollama_reduce_params={
                "max_tokens": 1000,
                "temperature": 0.5,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
            },
            tokenize={
                "map_max_input_tokens": 1024,
                "reduce_max_input_tokens": 4096,
                "truncation": True,
            },
            chunking={"strategy": "word_chunking", "word_chunk_size": 900, "word_overlap": 150},
            promoted_from="hybrid_ml_bart_llama32_3b_smoke_paragraph_v1",
            promoted_at="2026-04-04T00:00:00Z",
            metrics_summary={
                "dataset_id": "curated_5feeds_smoke_v1",
                "reference_id": "silver_sonnet46_smoke_v1",
                "rouge_l_f1_approx": 0.223,
                "embedding_cosine_approx": 0.764,
                "note": "2-3pp sampling variance at temp=0.5; use 3+ runs for stable estimate",
                "sweep_rounds": 2,
                "experiments_run": 9,
                "gains": {
                    "ollama_temperature": "+10.0% carry-over from longt5 sweep (0.5 vs 0.3)",
                    "ollama_top_p": "+11.93% total across two steps (0.9 → 0.95 → 1.0)",
                    "ollama_max_tokens": "+6.96% (1000 vs 800)",
                },
                "vs_bart_led": "beats ml_bart_led_autoresearch_v1 (0.188) by ~+22%",
                "vs_cloud_gap": "closes ~70% of gap to Anthropic Claude 32.6%",
                "latency_s": 15.5,
            },
        ),
        "ml_hybrid_llama32_3b_autoresearch_v1": ModeConfiguration(
            mode_id="ml_hybrid_llama32_3b_autoresearch_v1",
            map_model="longt5-base",
            reduce_model="llama3.2:3b",
            reduce_backend="ollama",
            reduce_instruction_style="paragraph",
            preprocessing_profile="cleaning_v4",
            map_params={
                "do_sample": False,
                "num_beams": 6,
                "max_new_tokens": 200,
                "min_new_tokens": 80,
                "no_repeat_ngram_size": 3,
                "repetition_penalty": 1.1,
                "length_penalty": 1.0,
                "early_stopping": True,
            },
            reduce_params=None,
            ollama_reduce_params={
                "max_tokens": 800,
                "temperature": 0.5,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
            },
            tokenize={
                "map_max_input_tokens": 1024,
                "reduce_max_input_tokens": 4096,
                "truncation": True,
            },
            chunking={"strategy": "word_chunking", "word_chunk_size": 900, "word_overlap": 150},
            promoted_from="hybrid_ml_tier2_llama32_3b_smoke_paragraph_v1",
            promoted_at="2026-04-03T00:00:00Z",
            metrics_summary={
                "dataset_id": "curated_5feeds_smoke_v1",
                "reference_id": "silver_sonnet46_smoke_v1",
                "rouge_l_f1": 0.208,
                "embedding_cosine": 0.763,
                "sweep_rounds": 1,
                "experiments_run": 4,
                "gains": {
                    "ollama_temperature": "+10.0% (0.5 vs 0.3)",
                },
                "vs_baseline": "beats BART+LED autoresearch_v1 (0.188 ROUGE-L) by +10.6%",
                "latency_s": 15,
            },
        ),
        # END MODE REGISTRY (append-only)
    }

    @classmethod
    def get_capabilities(
        cls,
        model_id: str,
        model_instance: Optional[Any] = None,
    ) -> ModelCapabilities:
        """Get model capabilities with dynamic fallback.

        Priority:
        1. Registry lookup (by alias or full ID)
        2. Dynamic detection from model_instance.config
        3. Intelligent fallback based on model name patterns
        4. Safe default (1024 for unknown models)

        Args:
            model_id: Model identifier (alias or full HuggingFace ID)
            model_instance: Optional loaded model instance for dynamic detection

        Returns:
            ModelCapabilities with architecture limits
        """
        # Normalize model_id for lookup (e.g. mocks in tests may be non-string)
        if not isinstance(model_id, str):
            model_id = str(model_id)

        # 1. Check registry first
        if model_id in cls._registry:
            return cls._registry[model_id]

        # 2. Try dynamic detection from model instance
        if model_instance is not None:
            try:
                config = (
                    model_instance.model.config
                    if hasattr(model_instance, "model")
                    else model_instance.config
                )
                max_pos = getattr(config, "max_position_embeddings", None)
                if max_pos is not None:
                    model_type = cls._infer_model_type(config)
                    family = cls._infer_model_family(model_id, model_type)
                    supports_long = max_pos >= 4096
                    return ModelCapabilities(
                        max_input_tokens=max_pos,
                        model_type=model_type,
                        model_family=family,
                        supports_long_context=supports_long,
                    )
            except (AttributeError, TypeError):
                pass

        # 3. Pattern-based fallback
        lower_id = model_id.lower()
        if "led" in lower_id or "longformer" in lower_id:
            return _led_caps()
        if "longt5" in lower_id or "long-t5" in lower_id or "long_t5" in lower_id:
            return _longt5_caps()
        if "bart" in lower_id:
            return _bart_caps()
        if "pegasus" in lower_id:
            return _pegasus_caps()
        if "flan-t5" in lower_id or "flan_t5" in lower_id:
            return ModelCapabilities(
                max_input_tokens=512,
                model_type="flan-t5",
                model_family="reduce",
                supports_long_context=False,
                supports_json_output=True,
                supports_extraction=True,
            )
        if "sentence-transformer" in lower_id:
            return ModelCapabilities(
                max_input_tokens=256,
                model_type="sentence-transformer",
                model_family="embedding",
                supports_long_context=False,
                embedding_dim=384,
            )
        if "squad" in lower_id:
            return ModelCapabilities(
                max_input_tokens=512,
                model_type="qa",
                model_family="extractive_qa",
                supports_long_context=False,
            )
        if "nli" in lower_id:
            return ModelCapabilities(
                max_input_tokens=512,
                model_type="nli",
                model_family="nli",
                supports_long_context=False,
            )

        # 4. Safe default (conservative)
        return ModelCapabilities(
            max_input_tokens=1024,
            model_type="unknown",
            model_family="unknown",
            supports_long_context=False,
        )

    @classmethod
    def get_mode_configuration(cls, mode_id: str) -> ModeConfiguration:
        """Get a promoted mode configuration by ID.

        Args:
            mode_id: Mode identifier (e.g. "ml_prod_authority_v1")

        Returns:
            ModeConfiguration for the requested mode.

        Raises:
            ValueError: If the mode is not registered.
        """
        if mode_id not in cls._mode_registry:
            raise ValueError(f"Mode '{mode_id}' not found in registry")
        return cls._mode_registry[mode_id]

    @classmethod
    def _infer_model_type(cls, config: Any) -> str:
        """Infer model type from config."""
        model_type = getattr(config, "model_type", "").lower()
        if "bart" in model_type:
            return "bart"
        if "led" in model_type:
            return "led"
        if "longformer" in model_type:
            return "led"
        if "longt5" in model_type:
            return "longt5"
        if "pegasus" in model_type:
            return "pegasus"
        if "t5" in model_type:
            return "flan-t5"
        return "unknown"

    @classmethod
    def _infer_model_family(cls, model_id: str, model_type: str) -> str:
        """Infer model family from ID and type."""
        lower_id = model_id.lower()
        if "nli" in lower_id:
            return "nli"
        if "squad" in lower_id:
            return "extractive_qa"
        if "sentence-transformer" in lower_id:
            return "embedding"
        if model_type in ("bart", "led", "pegasus", "longt5"):
            return "map"
        if model_type == "flan-t5":
            return "reduce"
        return "unknown"

    @classmethod
    def register_model(cls, model_id: str, capabilities: ModelCapabilities) -> None:
        """Register a new model (for extensibility).

        Args:
            model_id: Model identifier
            capabilities: Model capabilities
        """
        cls._registry[model_id] = capabilities


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline-stage registry — RFC-044 expansion beyond ML-only models.
#
# RFC-044 (and the WIP plan at `docs/wip/RESEARCH_POWERED_REGISTRY_PLAN.md`)
# extends the model_registry concept from "just ML/HF models" to "canonical
# defaults for every pipeline stage with research provenance."
#
# Each StageOption below carries:
#   - the runtime knobs (provider, model, endpoint/port if local)
#   - the research evidence that justified it (eval report path + headline metric)
#   - a recommendation tier (primary / fallback / experimental)
#
# Profile YAMLs in `config/profiles/` derive their stage choices from these
# entries — a profile is a composition of StageOption references, not a
# free-form bag of settings. The eventual goal (per RFC-044) is that a
# profile YAML becomes a near-empty pointer:
#
#     profile: cloud_with_dgx_primary
#
# and every field is resolved from this registry. Until that resolver lands,
# the entries below are the **canonical source the profile YAMLs should match**;
# any drift between them is a bug to file.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StageOption:
    """A single provider/model option for a pipeline stage.

    Used for transcription, summary, GI, KG, NER, clustering. Holds the
    runtime knobs plus the research_ref that justified the choice.
    """

    # Identity
    stage: str  # "transcription" | "summary" | "gi" | "kg" | "ner" | "clustering"
    option_id: str  # stable key, e.g. "openai_whisper_1", "ollama_qwen35_35b"

    # Runtime knobs
    provider: str  # config.transcription_provider / summary_provider value
    model: Optional[str] = None  # provider-specific model name
    endpoint: Optional[str] = None  # for DGX/Tailscale-routed options
    extra_settings: Optional[Dict[str, Any]] = None  # e.g. {"think": False}

    # Research provenance
    research_ref: Optional[str] = None  # eval report path or issue ref
    headline_metric: Optional[str] = None  # one-line summary of why this won
    measured_at: Optional[str] = None  # YYYY-MM-DD of the latest measurement

    # Recommendation tier
    tier: str = "primary"  # "primary" | "fallback" | "experimental" | "deprecated"

    # Capacity hints (for GB10 unified-memory sizing per DGX_RUNBOOK)
    resident_memory_gb: Optional[float] = None  # GPU + CPU memory if loaded
    realtime_multiple: Optional[float] = None  # for transcription stages


# Transcription stage — answers RFC-044's #592 placeholder.
_TRANSCRIPTION_OPTIONS: Dict[str, StageOption] = {
    # Cloud Whisper API — reliable, cheap, 25 MB file cap unless chunked.
    "openai_whisper_1": StageOption(
        stage="transcription",
        option_id="openai_whisper_1",
        provider="openai",
        model="whisper-1",
        research_ref="docs/guides/eval-reports/EVAL_TRANSCRIPTION_3WAY_2026_06.md",
        headline_metric="quality ceiling for v2 fixtures; ~$0.006/min",
        measured_at="2026-06-11",
        tier="primary",
    ),
    # DGX whisper-openai container — production winner per #929 + #963.
    # Fast (4.6× realtime), clean output, code we own.
    "tailnet_dgx_whisper_openai": StageOption(
        stage="transcription",
        option_id="tailnet_dgx_whisper_openai",
        provider="tailnet_dgx_whisper",
        model="large-v3",
        endpoint="http://{dgx_tailnet_host}:8002/v1/audio/transcriptions",
        research_ref="docs/guides/eval-reports/EVAL_TRANSCRIPTION_3WAY_2026_06.md",
        headline_metric=(
            "mean WER 0.102 / 4.56× realtime on v2; " "verified stable under vLLM contention (#963)"
        ),
        measured_at="2026-06-11",
        tier="primary",
        resident_memory_gb=3.0,
        realtime_multiple=4.56,
    ),
    # DGX speaches/faster-whisper container — usable secondary post #968 Thread B.
    # Quality competitive (0.066 WER) but slow (~1× realtime); int8-only on GB10.
    "tailnet_dgx_speaches_thread_b": StageOption(
        stage="transcription",
        option_id="tailnet_dgx_speaches_thread_b",
        provider="tailnet_dgx_whisper",
        model="Systran/faster-whisper-large-v3",
        endpoint="http://{dgx_tailnet_host}:8000/v1/audio/transcriptions",
        extra_settings={"WHISPER__COMPUTE_TYPE": "default"},
        research_ref="docs/guides/eval-reports/EVAL_SPEACHES_COMPUTE_TYPE_2026_06.md",
        headline_metric=(
            "mean WER 0.083 / 2.38× realtime on v2 — "
            "speaches:latest-cuda-gb10 (#948 source-built ctranslate2) "
            "+ #968 Thread B temperature-fallback patch"
        ),
        measured_at="2026-06-12",
        tier="fallback",
        resident_memory_gb=3.0,
        realtime_multiple=2.38,
    ),
    # Laptop MPS — local production default for cloud_with_dgx fallback or no-DGX.
    "local_mps_large_v3": StageOption(
        stage="transcription",
        option_id="local_mps_large_v3",
        provider="whisper",
        model="large-v3",
        extra_settings={"LOCAL_WHISPER_DEVICE": "mps"},
        research_ref="docs/guides/eval-reports/EVAL_TRANSCRIPTION_3WAY_2026_06.md",
        headline_metric="mean WER 0.096 / 1.6× realtime on v2 (laptop default)",
        measured_at="2026-06-09",
        tier="primary",
        resident_memory_gb=3.0,
        realtime_multiple=1.6,
    ),
}


# Summary stage — local-DGX default holds per #928 + #958 Cell D.
_SUMMARY_OPTIONS: Dict[str, StageOption] = {
    # Cloud cheap — current production default for cloud_balanced / cloud_thin /
    # cloud_with_dgx_primary's summary path.
    "gemini_flash_lite": StageOption(
        stage="summary",
        option_id="gemini_flash_lite",
        provider="gemini",
        model="gemini-2.5-flash-lite",
        research_ref="docs/guides/eval-reports/EVAL_HELDOUT_V2_2026_04.md",
        headline_metric=(
            "0.564 bullets / 1.5s / $0.00047/ep — " "best compound (cost × latency × quality) score"
        ),
        measured_at="2026-04-16",
        tier="primary",
    ),
    # Local DGX winner per #928 — unchanged after #958 Cell D's same-model quant
    # isolation check.
    "ollama_qwen35_35b": StageOption(
        stage="summary",
        option_id="ollama_qwen35_35b",
        provider="ollama",
        model="qwen3.5:35b",
        endpoint="http://{dgx_tailnet_host}:11434/v1",
        extra_settings={"think": False},  # required for direct /api/chat (#959)
        research_ref="docs/guides/eval-reports/EVAL_SUMMARY_DGX_LOCAL_2026_06.md",
        headline_metric="G-Eval 5.00 mean on #928 finale; Q4_K_M robust per #958 Cell D",
        measured_at="2026-06-11",
        tier="primary",
        resident_memory_gb=23.0,
    ),
    # Legitimate panelist after #961 prompt fix; still slightly under qwen3.5:35b.
    "vllm_r1_distill_32b_with_prompt_fix": StageOption(
        stage="summary",
        option_id="vllm_r1_distill_32b_with_prompt_fix",
        provider="openai",  # via the #960 first-class vLLM path
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        endpoint="http://{dgx_tailnet_host}:8003/v1",
        extra_settings={
            "api_key_env": "VLLM_NO_AUTH_NEEDED",
            "anti_reasoning_prompt": (
                "vllm/r1_distill_32b/summarization/{system,long}_no_thinking_v1"
            ),
            "postprocess": "strip_r1_reasoning",
        },
        research_ref="docs/guides/eval-reports/EVAL_SUMMARY_DGX_LOCAL_2026_06.md",
        headline_metric="G-Eval 4.05 mean (post-#961 prompt fix; pre-fix was 3.25)",
        measured_at="2026-06-11",
        tier="experimental",
        resident_memory_gb=64.0,
    ),
    # Ollama R1-Distill Q4 — surprising same-model winner over vLLM bf16.
    # Useful fallback when vLLM is busy.
    "ollama_deepseek_r1_32b_q4": StageOption(
        stage="summary",
        option_id="ollama_deepseek_r1_32b_q4",
        provider="ollama",
        model="deepseek-r1:32b",
        endpoint="http://{dgx_tailnet_host}:11434/v1",
        research_ref="docs/guides/eval-reports/EVAL_SUMMARY_DGX_LOCAL_2026_06.md",
        headline_metric=(
            "G-Eval 4.15 mean (#958 Cell D) — " "beats vLLM-R1-bf16 by +0.90 on same model"
        ),
        measured_at="2026-06-11",
        tier="fallback",
        resident_memory_gb=19.0,
    ),
    # Smaller local for laptop / dev profiles.
    "ollama_qwen35_9b": StageOption(
        stage="summary",
        option_id="ollama_qwen35_9b",
        provider="ollama",
        model="qwen3.5:9b",
        endpoint="http://{dgx_tailnet_host}:11434/v1",
        extra_settings={"think": False, "bundled_mode_caveat": "fragile in bundled mode"},
        research_ref="docs/guides/eval-reports/EVAL_HELDOUT_V2_2026_04.md",
        headline_metric="0.529 bullets / 0.509 para — local DGX entry-level",
        measured_at="2026-04-16",
        tier="fallback",
        resident_memory_gb=6.6,
    ),
}


@dataclass(frozen=True)
class ProfilePreset:
    """A named composition of StageOptions — the registry-driven view of a profile.

    Each preset names one StageOption per stage as the canonical default. The
    profile YAMLs in `config/profiles/` are downstream consumers — they should
    encode either (a) the exact StageOption keys named here, or (b) explicit
    overrides for niche use cases. Drift between a profile YAML and its
    declared preset is a bug.
    """

    name: str
    transcription: str  # StageOption.option_id key into _TRANSCRIPTION_OPTIONS
    summary: str  # StageOption.option_id key into _SUMMARY_OPTIONS
    notes: Optional[str] = None


# Profile presets — canonical compositions. Profile YAMLs are downstream views.
_PROFILE_PRESETS: Dict[str, ProfilePreset] = {
    "cloud_balanced": ProfilePreset(
        name="cloud_balanced",
        transcription="openai_whisper_1",
        summary="gemini_flash_lite",
        notes="Production cloud default. Best compound (quality × cost × latency).",
    ),
    "cloud_thin": ProfilePreset(
        name="cloud_thin",
        transcription="openai_whisper_1",
        summary="gemini_flash_lite",
        notes="Minimal cloud feature set. Same providers as cloud_balanced.",
    ),
    "cloud_with_dgx_primary": ProfilePreset(
        name="cloud_with_dgx_primary",
        transcription="tailnet_dgx_whisper_openai",  # post-#929/#966 winner
        summary="gemini_flash_lite",
        notes=(
            "Production hybrid: DGX whisper-openai for transcription, cloud Gemini for summary. "
            "Was previously tailnet_dgx_speaches; flipped after #968 Thread B confirmed "
            "speaches viability but with 5× speed cost vs whisper-openai."
        ),
    ),
    "local_dgx_balanced": ProfilePreset(
        name="local_dgx_balanced",
        transcription="local_mps_large_v3",
        summary="ollama_qwen35_35b",  # #928 winner; #958 Cell D confirms Q4 robustness
        notes="Local pipeline with DGX summary; laptop runs MPS transcription.",
    ),
    "local_dgx_full": ProfilePreset(
        name="local_dgx_full",
        transcription="local_mps_large_v3",
        summary="ollama_qwen35_35b",
        notes="Higher resident memory budget; same registry choices as balanced today.",
    ),
}


def get_transcription_options() -> Dict[str, StageOption]:
    """All registered transcription options (id → StageOption)."""
    return dict(_TRANSCRIPTION_OPTIONS)


def get_summary_options() -> Dict[str, StageOption]:
    """All registered summary options (id → StageOption)."""
    return dict(_SUMMARY_OPTIONS)


def get_transcription_option(option_id: str) -> StageOption:
    """Look up a single transcription option by id."""
    if option_id not in _TRANSCRIPTION_OPTIONS:
        raise ValueError(f"Unknown transcription option '{option_id}'")
    return _TRANSCRIPTION_OPTIONS[option_id]


def get_summary_option(option_id: str) -> StageOption:
    """Look up a single summary option by id."""
    if option_id not in _SUMMARY_OPTIONS:
        raise ValueError(f"Unknown summary option '{option_id}'")
    return _SUMMARY_OPTIONS[option_id]


def get_profile_preset(name: str) -> ProfilePreset:
    """Look up a profile preset by name. The canonical source for profile YAMLs."""
    if name not in _PROFILE_PRESETS:
        raise ValueError(f"Unknown profile preset '{name}'. Known: {sorted(_PROFILE_PRESETS)}")
    return _PROFILE_PRESETS[name]


# Hostname placeholder used in StageOption.endpoint templates. Operators
# supply the real value via the DGX_TAILNET_HOST env var (or directly via
# the dgx_tailnet_host arg to resolve_endpoint) so the operator's specific
# Tailscale MagicDNS name doesn't get checked into the repo. See
# CONTRIBUTING.md "Hostnames in the registry" section.
_DGX_TAILNET_HOST_PLACEHOLDER: Final[str] = "REPLACE_ME_DGX_TAILNET_HOST"


def resolve_endpoint(
    template: Optional[str], dgx_tailnet_host: Optional[str] = None
) -> Optional[str]:
    """Substitute the `{dgx_tailnet_host}` placeholder in a StageOption endpoint.

    Resolution order: explicit arg → DGX_TAILNET_HOST env var →
    DGX_TAILNET_HOST_PLACEHOLDER (which makes downstream HTTP calls fail
    obviously rather than silently routing to a placeholder hostname). Pass
    None to get None back (no endpoint configured for this StageOption).
    """
    import os

    if template is None:
        return None
    if "{dgx_tailnet_host}" not in template:
        return template
    host = dgx_tailnet_host or os.getenv("DGX_TAILNET_HOST") or _DGX_TAILNET_HOST_PLACEHOLDER
    return template.format(dgx_tailnet_host=host)


def resolve_profile_to_settings(name: str) -> Dict[str, Any]:
    """Resolve a profile preset to the runtime settings the pipeline expects.

    Returns a flat dict the Config can ingest. Profile YAMLs become thin
    viewers — eventually a YAML reads `profile: cloud_with_dgx_primary` and
    this resolver fills in the rest.
    """
    preset = get_profile_preset(name)
    tx = get_transcription_option(preset.transcription)
    sm = get_summary_option(preset.summary)

    settings: Dict[str, Any] = {
        "transcription_provider": tx.provider,
    }
    if tx.model is not None:
        # Field name depends on provider — keeping the canonical mapping minimal here.
        # Consumers may need additional translation; this returns the registry view.
        settings.setdefault("transcription_model", tx.model)
    if tx.endpoint is not None:
        settings["transcription_endpoint"] = tx.endpoint
    if tx.extra_settings:
        settings["transcription_extra"] = dict(tx.extra_settings)

    settings["summary_provider"] = sm.provider
    if sm.model is not None:
        settings["summary_model"] = sm.model
    if sm.endpoint is not None:
        settings["summary_endpoint"] = sm.endpoint
    if sm.extra_settings:
        settings["summary_extra"] = dict(sm.extra_settings)

    settings["_profile_preset"] = preset.name
    settings["_transcription_research_ref"] = tx.research_ref
    settings["_summary_research_ref"] = sm.research_ref

    return settings
