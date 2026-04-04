"""Model Registry: single source of truth for model capabilities and architecture limits.

Implements RFC-044: centralizes model metadata (max input tokens, chunk defaults,
model family) and provides get_capabilities() with fallback order: registry →
dynamic detection → pattern-based guess → safe default. Keeps the codebase
decoupled from hardcoded limits and supports future mode configurations
(promoted baselines).

See docs/rfc/RFC-044-model-registry.md for design and migration strategy.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


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

    Promoted from proven baseline configurations (see RFC-044). These modes can
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

    # Aliases for evidence stack (embedding, extractive QA, NLI) per RFC-042 §12.1
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
