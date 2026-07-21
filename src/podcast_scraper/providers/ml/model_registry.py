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

**2026-07-05 note (#382, transformers v5)**: consumers of this registry no
longer instantiate transformers.pipeline(...) (removed in v5) — they
use AutoModel*.from_pretrained + generate() / forward pass via
:class:`HFEvidenceBackend` (evidence stack) and :class:`HFSeq2SeqBackend`
(summarizer + hybrid reduce). Capability fields still describe the
underlying checkpoint, not the pipeline wrapper. Model IDs and pinned
revisions unchanged.

**2026-06-23 amendment (#1060)**: YAML-only profiles are first-class. A
profile YAML may legitimately have no ``ProfilePreset`` when it pins only
MODEL across vendors (not PROVIDER, e.g. ``test_default``), or when its
shape is otherwise incompatible with the rigid 6-tuple preset model. Such
profiles MUST carry a "Registry status: YAML-only" comment with the
gating reason; the drift test
(``tests/integration/providers/ml/test_profile_yaml_registry_drift.py``)
enforces this. See ``config/profiles/README.md`` § "Registry status — two
first-class shapes" for the operator-facing rule.

The flow for any autoresearch finding that changes a default:

    run experiment → score → write eval report → MATERIALIZE here →
    REGENERATE profile YAML → behavior test

See ``docs/wip/RESEARCH_POWERED_REGISTRY_PLAN.md`` for the vision /
migration path, ``docs/adr/ADR-048-centralized-model-registry.md`` for the
amendment, and ``docs/guides/EXPERIMENT_GUIDE.md`` § Step 6 for the flow.
"""

from dataclasses import dataclass
from typing import Any, Dict, Final, Optional, Tuple


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

    Used for transcription, summary, GI, KG, NER, clustering, diarization.
    Holds the runtime knobs plus the research_ref that justified the choice.
    """

    # Identity
    stage: str  # "transcription"|"summary"|"gi"|"kg"|"ner"|"clustering"|"diarization"
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

    # Why a superseded option was abandoned. Superseded research is kept as HISTORY rather than
    # deleted — knowing that n=12 was never derived from anything (and that the providers clamped to
    # 10 regardless, so no run could ever honour it) is worth more than the 12 ever was.
    deprecated_at: Optional[str] = None  # YYYY-MM-DD
    deprecation_reason: Optional[str] = None

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
        research_ref="docs/guides/eval-reports/EVAL_WHISPER_ENGINE_DRIFT_2026_06_16.md",
        headline_metric=(
            "DGX transcription winner (#952): 10.23% WER vs Deepgram silver on real "
            "podcasts — beats whisper-openai (:8002) by 2.74pp AND ~20% faster. "
            "speaches:latest-cuda-gb10 int8 (#948) + #968 Thread B temperature fallback"
        ),
        measured_at="2026-06-16",
        tier="primary",
        resident_memory_gb=3.0,
        realtime_multiple=2.38,
    ),
    # DGX MOSS-Transcribe-Diarize (:8004) — the #1174 bake-off transcription winner.
    # Single 0.9B model does transcribe+diarize in one pass; we use it for TRANSCRIPTION
    # and keep pyannote community-1 for diarization (MOSS lost diarization on real audio).
    # Long audio is chunked upstream (AudioChunker, #1174) so a request never exceeds the
    # model's ~30 min single-pass ceiling. The moss provider defaults moss_port=8004 +
    # moss_model, so the profile needs only transcription_provider=moss.
    "moss_transcribe_diarize": StageOption(
        stage="transcription",
        option_id="moss_transcribe_diarize",
        provider="moss",
        model="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        endpoint="http://{dgx_tailnet_host}:8004/v1/transcribe",
        research_ref="docs/guides/eval-reports/EVAL_MOSS_BAKEOFF_2026_07.md",
        headline_metric=(
            "DGX transcription winner (#1174): 5.2% WER vs Deepgram silver on 10 real "
            "prod episodes — beats faster-whisper (8.5%) on all 10. Diarization loses to "
            "pyannote on real audio, so pair with pyannote. 2.1-3.8x realtime (bare "
            "transformers, no speed win vs large-v3). English quality resolved: excellent."
        ),
        measured_at="2026-07-16",
        tier="primary",
        resident_memory_gb=16.0,
        realtime_multiple=2.7,
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
    # Laptop CPU/MPS — quality upgrade over base.en at higher latency.
    # Picked by local.yaml (operator's laptop default profile).
    "local_whisper_small_en": StageOption(
        stage="transcription",
        option_id="local_whisper_small_en",
        provider="whisper",
        model="small.en",
        research_ref="docs/guides/eval-reports/EVAL_WHISPER_SMALL_EN_2026_06_13.md",
        headline_metric=(
            "mean WER 0.029 on v2 (-25% vs base.en); 30.6s/ep on M4 Pro CPU "
            "(2.6× base.en latency — laptop trade)"
        ),
        measured_at="2026-06-13",
        tier="primary",
        resident_memory_gb=0.5,
    ),
    # Cloud quality — Deepgram nova-3 picked by cloud_quality.yaml.
    "deepgram_nova_3": StageOption(
        stage="transcription",
        option_id="deepgram_nova_3",
        provider="deepgram",
        model="nova-3",
        research_ref="docs/guides/eval-reports/EVAL_DEEPGRAM_TRANSCRIPTION_2026_06_13.md",
        headline_metric=(
            "mean WER 0.0248 on v2 — best accuracy AND best latency (1.2s/ep) "
            "across all measured models. ≈$0.0043/min."
        ),
        measured_at="2026-06-13",
        tier="primary",
    ),
    # Dev / airgapped_thin floor — fastest local Whisper. Smoke_v2 numbers
    # use the FU4 clean-reference preprocessing (markdown headers + speaker
    # labels + timestamps stripped) so WER is comparable to the v2 fixture
    # baseline in EVAL_WHISPER_SMALL_EN_2026_06_13.md. FU5 DGX figures pinned
    # alongside for cross-device portability.
    "local_whisper_tiny_en": StageOption(
        stage="transcription",
        option_id="local_whisper_tiny_en",
        provider="whisper",
        model="tiny.en",
        research_ref="docs/guides/eval-reports/EVAL_DEV_TIER_REGISTRY_2026_06_23.md",
        headline_metric=(
            "mean WER 17.2% (M4 Pro CPU) / 16.0% (DGX GB10 CUDA) on smoke_v2 "
            "with FU4 clean reference; 9.8 s/ep CPU vs 5.4 s/ep CUDA "
            "(1.8× speedup). Dev/CI floor — quality intentionally traded for "
            "speed; not for production transcription."
        ),
        measured_at="2026-06-23",
        tier="primary",
        resident_memory_gb=0.1,
    ),
    # Airgapped quality default — Whisper medium.en. Quality upgrade over
    # tiny.en; runs comfortably on a server (DGX GB10) and on a beefy laptop
    # CPU but not on a Docker stack-test box (~3 GB working set).
    "local_whisper_medium_en": StageOption(
        stage="transcription",
        option_id="local_whisper_medium_en",
        provider="whisper",
        model="medium.en",
        research_ref="docs/guides/eval-reports/EVAL_DEV_TIER_REGISTRY_2026_06_23.md",
        headline_metric=(
            "mean WER 8.1% on smoke_v2 with FU4 clean reference (M4 Pro CPU "
            "and DGX GB10 CUDA agree to 1pp); 82.7 s/ep CPU vs 34.3 s/ep CUDA "
            "(2.4× speedup). Airgapped quality default; -53% relative WER vs "
            "tiny.en at ~6× CPU wallclock."
        ),
        measured_at="2026-06-23",
        tier="primary",
        resident_memory_gb=1.5,
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
    # #113 small-model standoff (2026-06-20): under #1035 NER pre-pass, this
    # 9B Q4 model rides spaCy PERSON+ORG hints to 97% entity coverage (29/30
    # on silver_opus47_kg_dev_v1) — within 3pp of Cell F NVFP4's 100%, at
    # 73% smaller weight footprint. Topic coverage 66% essentially ties Cell
    # F (65%). NER pre-pass effectively closes the model-size gap for KG /
    # entity extraction. Candidate for `airgapped_thin` / `local_dgx_balanced`
    # default after benchmark_v2 held-out validation lands.
    "ollama_qwen35_9b": StageOption(
        stage="summary",
        option_id="ollama_qwen35_9b",
        provider="ollama",
        model="qwen3.5:9b",
        endpoint="http://{dgx_tailnet_host}:11434/v1",
        extra_settings={"think": False, "bundled_mode_caveat": "fragile in bundled mode"},
        research_ref="docs/wip/EVAL_113_SMALL_MODEL_STANDOFF.md",
        headline_metric=(
            "Summary 0.529 bullets / 0.509 para (#928). #113 small-model "
            "standoff under #1035 NER pre-pass: KG topic 66% / entity 97% — "
            "within 3pp of Cell F NVFP4 at 6.6 GB footprint."
        ),
        measured_at="2026-06-20",
        tier="fallback",
        resident_memory_gb=6.6,
    ),
    # Cloud-quality cheap-and-fast pick — bullets-bundled compound winner.
    # Picked by cloud_quality.yaml.
    "anthropic_haiku_4_5": StageOption(
        stage="summary",
        option_id="anthropic_haiku_4_5",
        provider="anthropic",
        model="claude-haiku-4-5",
        research_ref="docs/guides/eval-reports/EVAL_HELDOUT_V2_2026_04.md",
        headline_metric=(
            "bullets-bundled compound winner (0.552); 4th quality / 2nd fastest "
            "on the cost-latency-quality frontier; 4.8s / $0.00416/ep"
        ),
        measured_at="2026-04-16",
        tier="primary",
    ),
    # Laptop summary default (no DGX, no cloud). Picked by local.yaml.
    "ollama_hermes3_8b_laptop": StageOption(
        stage="summary",
        option_id="ollama_hermes3_8b_laptop",
        provider="ollama",
        model="hermes3:8b",
        research_ref="docs/guides/eval-reports/EVAL_HYBRID_ROUTING_2026_06.md",
        headline_metric=(
            "laptop-default summary per #949 finale; ~50× realtime on Ollama CPU. "
            "Trade vs base llama3.1:8b documented in EVAL_SMOKE_V2_DGX_REFRESH_2026_06."
        ),
        measured_at="2026-06-10",
        tier="primary",
        resident_memory_gb=6.0,
    ),
    # vLLM-served Qwen3.5-35B-A3B — #1016 Round 3 top-dog on summary + KG.
    # As of #1022 (2026-06-19), supplanted by Qwen3-30B-A3B-NVFP4 for
    # **daily-driver use** (Cell F: 1.7× faster end-to-end at -4.7% quality
    # sum). Retain this option for **highest-stakes one-shot evals** where
    # summary or KG quality matters more than wall-clock — operator manually
    # swaps the homelab compose for those runs. Requires
    # `--reasoning-parser=qwen3` server flag. +3.6 Sonnet-mimicry on summary —
    # cross-vendor judge panel required to validate top-dog claim
    # ([[silver_judge_vendor_bias]]).
    "vllm_qwen3_5_35b_a3b": StageOption(
        stage="summary",
        option_id="vllm_qwen3_5_35b_a3b",
        provider="openai",
        model="autoresearch",  # served-model-name on the autoresearch slot
        endpoint="http://{dgx_tailnet_host}:8003/v1",
        extra_settings={
            "api_key_env": "VLLM_API_KEY",
            "compose_flag_required": "--reasoning-parser=qwen3",
            "chat_template_kwargs": {"enable_thinking": False},
            "vendor_sampling": {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "presence_penalty": 1.5,
            },
        },
        research_ref="docs/wip/EVAL_1016_FINAL_REPORT_2026_06_17.md",
        headline_metric=(
            "ROUGE-1 vs Opus 59.4%, cosine 83.7% (cohort leader); GI 36%, KG 38% "
            "(KG cohort leader). 13.6s/ep. Δ S−O = +3.6 (Sonnet-mimicry — needs "
            "cross-vendor judge panel)."
        ),
        measured_at="2026-06-17",
        tier="primary",
        resident_memory_gb=67.0,
    ),
    # vLLM-served Moonlight-16B-A3B-Instruct — #1016 Round 3 included it as the
    # style-neutral "safe pick" (Δ S−O = 0.0 across all 3 stages). That role
    # has since compressed to "speed-only choice when entity quality is not a
    # gate":
    #   - #1022 (2026-06-19): supplanted on autoresearch tier by Cell F NVFP4
    #     (Cell F wins GI +161% / KG +45% / speed -5% / footprint -44%)
    #   - #113 (2026-06-20) under #1035 NER pre-pass: lowest topic coverage
    #     in the cohort (54%, vs Cell F 65% / Qwen3.5:9b 66%) AND the only
    #     candidate emitting NER false-positive entities (+4 extras). MoE
    #     small active-params (~3B/token) trade discrimination for speed —
    #     under NER hints, denser models filter the candidate list better.
    # Retained here only as a speed-priority option when KG entity quality
    # isn't gating.
    # Requires `--max-model-len=8192` (model's max_position_embeddings).
    "vllm_moonlight_16b_a3b": StageOption(
        stage="summary",
        option_id="vllm_moonlight_16b_a3b",
        provider="openai",
        model="autoresearch",  # served-model-name on the autoresearch slot
        endpoint="http://{dgx_tailnet_host}:8003/v1",
        extra_settings={
            "api_key_env": "VLLM_API_KEY",
            "compose_flag_required": "--trust-remote-code --max-model-len=8192",
            "vendor_sampling": "greedy (no vendor recommendation)",
        },
        research_ref="docs/wip/EVAL_113_SMALL_MODEL_STANDOFF.md",
        headline_metric=(
            "Summary ROUGE-1 vs Opus 57.5%, cosine 78.6% (#1016 R3). "
            "#113 under #1035 NER pre-pass: KG topic 54% (cohort floor), "
            "entity 93% (4 false positives — only candidate with extras). "
            "Speed-priority choice only; not for entity-quality-gated work."
        ),
        measured_at="2026-06-20",
        tier="primary",
        resident_memory_gb=32.0,
    ),
    # vLLM-served Qwen3-30B-A3B-NVFP4 — #1022 Cell F daily-driver champion
    # (corrected-pipeline rankings per #1033, 2026-06-19).
    # NVFP4 quantization of Qwen3-30B-A3B-Instruct-2507 (NVIDIA Model
    # Optimizer team's official quant — `NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4`).
    # On the corrected `provider`-source pipeline: GI avg_sim 0.595 (cohort
    # #2 behind Qwen3.5-35B-A3B 0.618), KG topic cov 45% (cohort #2 behind
    # Qwen3.5-35B-A3B 50%). 16% faster end-to-end on the GI+KG path (440s
    # vs Qwen3.5-35B-A3B 526s) at a bounded -4% GI / -5pp KG quality gap.
    # Same architecture + sampling as Qwen3-30B-A3B-Instruct-2507 baseline —
    # drop-in replacement at the homelab compose level. Held-out validated
    # on benchmark_v2 (Sonnet 4.6 silver) — quality holds cross-dataset +
    # cross-vendor. For one-shot highest-stakes evals, operator can
    # manually swap the compose to Qwen3.5-35B-A3B for the cohort top-quality
    # candidate.
    "vllm_qwen3_30b_a3b_nvfp4": StageOption(
        stage="summary",
        option_id="vllm_qwen3_30b_a3b_nvfp4",
        provider="openai",
        model="autoresearch",  # served-model-name on the autoresearch slot
        endpoint="http://{dgx_tailnet_host}:8003/v1",
        extra_settings={
            "api_key_env": "VLLM_API_KEY",
            "underlying_hf_model": "NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4",
            "chat_template_kwargs": {"enable_thinking": False},
            "vendor_sampling": {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "presence_penalty": 1.5,
            },
        },
        research_ref="docs/wip/EVAL_1033_COHORT_RERUN_2026-06-19.md",
        headline_metric=(
            "Corrected-pipeline (#1033) cohort: GI avg_sim 0.595 (rank 2), "
            "KG topic cov 45% (rank 2), summary rouge1_f1 vs Opus 0.5407 "
            "(rank 5). End-to-end GI+KG 440s — 16% faster than #1 "
            "Qwen3.5-35B-A3B at -4% GI / -5pp KG quality gap. 18 GB weight "
            "footprint (3.7× smaller). Best speed-quality trade-off; "
            "Qwen3.5-35B-A3B is the top-quality reserve for one-shot evals."
        ),
        measured_at="2026-06-19",
        tier="primary",
        resident_memory_gb=18.0,
    ),
    # Airgapped summary default — SummLlama 3.2-3B paragraph. Local MPS,
    # no cloud, no Ollama. #571 / #652 / #653 history continues in the
    # 2026-06-23 smoke_v2 measurement. Laptop-class scope only by design —
    # operators with a GPU pick a larger ollama/vLLM model instead.
    "summllama_3_2_3b_paragraph": StageOption(
        stage="summary",
        option_id="summllama_3_2_3b_paragraph",
        provider="summllama",
        model="DISLab/SummLlama3.2-3B",
        extra_settings={
            "summllama_summary_style": "paragraph",
            "summllama_device": "mps",
            "summllama_max_tokens": 600,
        },
        research_ref="docs/guides/eval-reports/EVAL_DEV_TIER_REGISTRY_2026_06_23.md",
        headline_metric=(
            "ROUGE-L 0.251 / ROUGE-1 0.499 / cosine 0.823 on smoke_v2 "
            "paragraph vs silver_sonnet46_smoke_v2; cross-vendor judge mean "
            "0.735 (gpt-4o-mini + claude-haiku-4-5; contested on individual "
            "episodes so Track-A scalar = ROUGE-L). 53.3 s/ep on M4 Pro MPS "
            "(laptop-class scope — not DGX-benched). Airgapped paragraph "
            "quality default — beats bart-small+long-fast by +67% ROUGE-L, "
            "+25% cosine, and +0.36 judge at 2.9× the latency."
        ),
        measured_at="2026-06-23",
        tier="primary",
        resident_memory_gb=7.0,
    ),
    # Dev / airgapped_thin summary default — pure-transformers map-reduce,
    # no GPU required, no external model server. Backs the
    # `ml_small_authority` ModeConfiguration above. Laptop-class scope only.
    "transformers_bart_small_long_fast_authority": StageOption(
        stage="summary",
        option_id="transformers_bart_small_long_fast_authority",
        provider="transformers",
        model="bart-small",
        extra_settings={
            "summary_reduce_model": "long-fast",
            "summary_mode_id": "ml_small_authority",
        },
        research_ref="docs/guides/eval-reports/EVAL_DEV_TIER_REGISTRY_2026_06_23.md",
        headline_metric=(
            "ROUGE-L 0.150 / ROUGE-1 0.311 / cosine 0.655 on smoke_v2 "
            "paragraph vs silver_sonnet46_smoke_v2; cross-vendor judge mean "
            "0.370 (gpt-4o-mini + claude-haiku-4-5; contested so Track-A "
            "scalar = ROUGE-L). 18.3 s/ep on M4 Pro MPS (laptop / stack-test "
            "scope — not DGX-benched). Dev/airgapped-thin floor — 8.1× "
            "compression, no GPU or external model server required."
        ),
        measured_at="2026-06-23",
        tier="primary",
        resident_memory_gb=1.5,
    ),
}


# --- GI / KG / NER / clustering stages -------------------------------------
#
# Materialized 2026-06-13 from #853 + #904 + #906 eval reports per the
# materialize-decisions discipline. GI sweep (#978) still needs a v2-fixture
# run before its options can land — that registry slot stays empty.
#
# NEVER add entries here without a ``research_ref`` pointing at an eval
# report that justifies the choice — the whole point of this registry is
# that every default is backed by measured evidence, not opinion.

# GI — summary-derived provider mode is the v2 winner (#978).
#
# `EVAL_GI_AUTORESEARCH_V2_2026_06_13.md` measured direct-from-transcript
# extraction across n ∈ {6, 8, 10, 12, 16} against the v2 silver. Coverage
# capped at 10% regardless of n; the summary-derived pipeline hits 72% on
# the same provider in the same eval window. The "bypass summary" historic
# claim is reversed on v2.
#
# Bundling (`bundled_ab`) is the cross-provider champion per #921
# (EVAL_GIL_BUNDLING_2026_05). Grounding is treated as universal-on.
_GI_OPTIONS: Dict[str, StageOption] = {
    # SUPERSEDED by provider_chunked_gated_v3. Kept as history, not deleted: n=12 was never derived
    # from anything (the providers clamped to 10 anyway, so no run could honour it), and recording
    # WHY a value was abandoned is worth more than the value was.
    "provider_n12_grounded_bundled": StageOption(
        stage="gi",
        option_id="provider_n12_grounded_bundled",
        provider="provider",
        tier="deprecated",
        deprecated_at="2026-07-14",
        deprecation_reason=(
            "n=12 was a historic default, never measured; providers clamped to 10 regardless. "
            "Superseded by provider_chunked_gated_v3, whose every value cites an eval."
        ),
        extra_settings={
            "max_insights": 12,
            "require_grounding": True,
            "evidence_quote_mode": "bundled",
            "evidence_nli_mode": "bundled",
        },
        research_ref="docs/guides/eval-reports/EVAL_GI_AUTORESEARCH_V2_2026_06_13.md",
        headline_metric=(
            "summary-derived provider mode beats direct-from-transcript by "
            "~60pp on v2 silver (72% vs 10%); n=12 historic default holds; "
            "bundled per EVAL_GIL_BUNDLING_2026_05"
        ),
        measured_at="2026-06-13",
    ),
    # The v3 tuning. Every value below was measured, not chosen — the registry now carries the
    # PARAMS an eval established, not just the model, so a profile cannot quietly run a different
    # configuration than the one we validated.
    #
    # n=12 was never derived from anything; it was a historic default that the providers clamped to
    # 10 anyway, so no run could ever honour it. Lifting the ceiling and letting the gates trim is
    # what the evals actually support: the value gate drops filler, and grounding drops anything
    # unsupported, so the ceiling is not what protects quality.
    "provider_chunked_gated_v3": StageOption(
        stage="gi",
        option_id="provider_chunked_gated_v3",
        provider="provider",
        extra_settings={
            "max_insights": 50,
            "require_grounding": True,
            "evidence_quote_mode": "bundled",
            "evidence_nli_mode": "bundled",
            # Local models saturate per CALL, not per episode: qwen emits ~18 insights however long
            # the episode is, while gemini scales with the material. Context was never the limit —
            # so give it more calls, not a bigger window.
            "insight_chunk_chars": 30000,
            "insight_dedupe_threshold": 0.75,
            # Filler rises with chunking, which is what the gate is for. Trimming filler is cheap;
            # nothing recovers knowledge that was never extracted.
            "value_gate_enabled": True,
            "value_gate_min_tier": 2,
            # Reproducibility. The providers hardcoded 0.3 and ignored config: the SAME config run
            # twice gave 28.0 vs 18.3 insights/episode and grounding on either side of the 80%
            # floor. A corpus that cannot be reproduced cannot be debugged. Worse, at 0.3 a model
            # disagrees with ITSELF between runs, and a head-to-head reports that disagreement as
            # "the other model found knowledge this one missed".
            "insight_temperature": 0.0,
            # The prompt decides what an insight IS. v3 rewrote extraction as speech acts and LOST
            # its A/B (route kappa 0.57 vs v2's 0.67), so v2 is the MEASURED choice — recorded here
            # rather than left to a code default, which is how a regression nearly shipped as an
            # upgrade.
            "insight_prompt_version": "v2",
            # The evidence floors: what counts as SUPPORT. Asking for strict textual entailment is
            # not the question this pipeline means, and asking it strictly cost 60% of the evidence
            # a trusted annotator accepted (#1179).
            "qa_score_min": 0.3,
            "nli_entailment_min": 0.5,
            # An LLM grounds with an LLM. This switch is what makes the evidence providers follow
            # the summariser; without it they fall back to the local transformers stack, so a
            # profile that asks for LLM grounding quietly grounds with DeBERTa instead and the
            # model gets the blame. It is a researched guarantee, not an implementation detail.
            "evidence_match_summary_provider": True,
        },
        research_ref="docs/guides/eval-reports/EVAL_GEMINI_VS_QWEN_10EP_2026_07.md",
        headline_metric=(
            "chunked extraction lifts grounded insights/episode 17.1 -> 43.2 (gemini) and "
            "16.1 -> 27.7 (qwen) on 10 pinned episodes; temperature 0 cuts re-run drift from "
            "9.7 insights / 14.7pp grounding to 0.7 / 0.9pp. OPEN: chunked qwen grounds at 75.1%, "
            "below the 80% floor — see the report before shipping this on a local-only profile."
        ),
        measured_at="2026-07-13",
        tier="experimental",
    ),
}


# KG — max_topics / max_entities defaults.
#
# Per-provider extraction reads the cleaned transcript directly via the
# configured summary provider's KG-extraction endpoint. The legacy
# `summary_bullets` source was removed in #1034 (per the #1033 audit) —
# its option entry here is gone. The (max_topics, max_entities) = (10, 15)
# tuple is the universal default across every YAML — these are caps, not
# sweeps, so a "winner" doesn't meaningfully apply. The canonicalization
# thresholds 0.65 / 0.70 baked into ``entity_clusters.py`` ARE measured
# (see #853 report).
_GROUNDING_RESEARCH_REF = "docs/guides/eval-reports/EVAL_GROUNDING_WHO_FINDS_THE_QUOTE_2026_07.md"

# GROUNDING — who finds the quote that backs an insight.
#
# This stage had no registry entry at all until now: the QA and NLI models were loose config fields,
# so a profile could not pin its grounder the way it pins every other stage. That is precisely how
# the choice drifted invisibly — every LLM profile silently fell back to the ML stack and grounded
# 0% of its insights, and nothing in the registry could contradict it.
#
# Writing the insight and finding its evidence are different jobs. The measurement holds the insight
# set FIXED and varies only the grounder, so a model that writes fewer insights cannot look like a
# better grounder.
_GROUNDING_OPTIONS: Dict[str, StageOption] = {
    "llm_matched_to_summary": StageOption(
        stage="grounding",
        option_id="llm_matched_to_summary",
        # Not a model id: the grounder IS the summarising LLM. Config resolves it in
        # _auto_promote_evidence_providers, so an LLM grounds the insights it wrote.
        provider="match_summary",
        extra_settings={"nli_entailment_min": 0.75},
        research_ref=_GROUNDING_RESEARCH_REF,
        headline_metric=(
            "82% of insights grounded (vs 8% for the ML QA+NLI stack) on a frozen 100-insight set; "
            "100% of returned quotes verbatim, 0% drift, 0% fabricated — an LLM asked for a quote "
            "copies it exactly, so the silent-drop risk is absent in practice"
        ),
        measured_at="2026-07-13",
        tier="primary",
    ),
    "ml_qa_nli": StageOption(
        stage="grounding",
        option_id="ml_qa_nli",
        provider="transformers",
        extra_settings={
            "qa_model": "deepset/roberta-base-squad2",
            "nli_model": "cross-encoder/nli-deberta-v3-base",
            "qa_window_chars": 1800,
        },
        research_ref=_GROUNDING_RESEARCH_REF,
        headline_metric=(
            "8% of insights grounded on the same frozen set. Two structural faults, neither "
            "fixable by a threshold: QA answers WITHIN a window and nothing asks which of ~40 "
            "windows is about the claim (no retrieval step); and the NLI head demands strict "
            "entailment while insights are abstractive (transcript hedges, insight asserts -> "
            "0.007). Needs embedding retrieval + a graded verifier. For the local/offline "
            "profiles, which have no LLM."
        ),
        measured_at="2026-07-13",
        tier="fallback",
    ),
}


_KG_OPTIONS: Dict[str, StageOption] = {
    "provider_n10_15": StageOption(
        stage="kg",
        option_id="provider_n10_15",
        provider="provider",
        extra_settings={"max_topics": 10, "max_entities": 15},
        research_ref="docs/guides/eval-reports/EVAL_ENTITY_CANON_2026_06_08.md",
        headline_metric=(
            "canonicalization thresholds 0.65/0.70 (+18pp recall vs pre-#853 baseline at "
            "100% precision); per-provider KG-extraction default for cloud + DGX profiles"
        ),
        measured_at="2026-06-08",
        tier="primary",
    ),
}


# NER / speaker detection — provider choice + spaCy model variant.
#
# Tier 3 report (#906) compared `en_core_web_sm` vs `en_core_web_trf`:
# `_trf` is +13pp v2-spec recall (96.7% vs 83.3%) at 2× latency, but 600 MB
# and not present in every deploy path. The Tier 3 finding: keep `_sm` as
# the explicit "lighter" default and prefer `_trf` where it's installed.
# Cloud profiles route NER through Gemini instead of spaCy (the pipeline-llm
# image doesn't ship spaCy models at all).
_NER_OPTIONS: Dict[str, StageOption] = {
    "gemini_speaker_detector": StageOption(
        stage="ner",
        option_id="gemini_speaker_detector",
        provider="gemini",
        research_ref="docs/guides/eval-reports/EVAL_FIXTURES_V2_TIER3_TUNING_2026_06_08.md",
        headline_metric=(
            "cloud-profile default — pipeline-llm image lacks spaCy models, "
            "Gemini handles NER + speaker detection inline with summary calls"
        ),
        measured_at="2026-06-08",
        tier="primary",
    ),
    "ollama_speaker_detector": StageOption(
        stage="ner",
        option_id="ollama_speaker_detector",
        provider="ollama",
        model="en_core_web_trf",  # the local entity stage still runs spaCy alongside
        research_ref="docs/guides/eval-reports/EVAL_SPEAKER_DETECTION_NAMING_2026_06_15.md",
        headline_metric=(
            "speaker/host detection served by the same DGX-resident LLM, so an all-DGX profile "
            "needs no cloud call for it (#1169)"
        ),
        measured_at="2026-06-15",
        tier="primary",
    ),
    "spacy_trf": StageOption(
        stage="ner",
        option_id="spacy_trf",
        provider="spacy",
        model="en_core_web_trf",
        research_ref="docs/guides/eval-reports/EVAL_FIXTURES_V2_TIER3_TUNING_2026_06_08.md",
        headline_metric=(
            "v2 spec recall 96.7% (+13pp vs en_core_web_sm); 2× more PERSON "
            "mentions/ep; ~1s latency. Preferred where the 600MB model is installed."
        ),
        measured_at="2026-06-08",
        tier="primary",
    ),
    "spacy_sm": StageOption(
        stage="ner",
        option_id="spacy_sm",
        provider="spacy",
        model="en_core_web_sm",
        research_ref="docs/guides/eval-reports/EVAL_FIXTURES_V2_TIER3_TUNING_2026_06_08.md",
        headline_metric=(
            "lightweight fallback — 83.3% v2 spec recall, 0.55s/ep; the "
            "thin-deploy default when `_trf` isn't available"
        ),
        measured_at="2026-06-08",
        tier="fallback",
    ),
}


# Topic / insight clusters — similarity threshold.
#
# Tier 1 report (#904) swept clustering threshold ∈ {0.65, 0.70, 0.75, 0.80,
# 0.85}. The current default of 0.75 is Pareto-optimal: max cross-feed
# parent clusters (4) at the lowest tc:* total (6). No Config field exposes
# this knob today — the threshold is a function-default in
# ``search/topic_clusters.py`` + ``search/insight_clusters.py``. The
# registry entry is documentation/provenance until a config surface lands;
# ``resolve_profile_to_settings`` does NOT yet plumb a clustering field
# into runtime settings.
_CLUSTERING_OPTIONS: Dict[str, StageOption] = {
    "topic_clusters_default_0_75": StageOption(
        stage="clustering",
        option_id="topic_clusters_default_0_75",
        provider="default",
        extra_settings={"threshold": 0.75},
        research_ref="docs/guides/eval-reports/EVAL_FIXTURES_V2_TIER1_TUNING_2026_06_08.md",
        headline_metric=(
            "0.75 Pareto-optimal on v2 fixtures (6 tc:* parents / 4 cross-feed); "
            "lower thresholds add near-singletons without cross-feed lift, "
            "higher collapse cross-feed clusters"
        ),
        measured_at="2026-06-08",
        tier="primary",
    ),
}


# Diarization — pyannote pipeline choice (in-process ML provider vs the DGX
# diarize service). community-1 (v4) beats 3.1 on the full v3 fixture set
# (count 40/45 vs 32/45, DER 7.1% vs 10.8%) by fixing 3.1's multi-speaker
# panel merges; 3.1 kept as fallback (better on brief-cameo count). community-1
# is non-gated (no HF token, unlike 3.1). See the eval report.
_DIARIZATION_RESEARCH_REF = (
    "docs/guides/eval-reports/EVAL_DIARIZATION_31_VS_COMMUNITY1_RTTM_2026_07.md"
)
_DIARIZATION_OPTIONS: Dict[str, StageOption] = {
    "pyannote_diarization_community1": StageOption(
        stage="diarization",
        option_id="pyannote_diarization_community1",
        # 'local' matches the diarization_provider config Literal (in-process pyannote); the DGX
        # sibling uses 'tailnet_dgx'. The registry provider IS the dispatchable backend so the
        # RFC-106 fallback chain can construct this tier directly.
        provider="local",
        model="pyannote/speaker-diarization-community-1",
        research_ref=_DIARIZATION_RESEARCH_REF,
        headline_metric=(
            "count 40/45 + DER 7.1% on v3 fixtures (vs 3.1's 32/45 / 10.8%); "
            "fixes 3.1's multi-speaker panel merges; non-gated"
        ),
        measured_at="2026-07-11",
        tier="primary",
        resident_memory_gb=2.0,
    ),
    "tailnet_dgx_diarization_community1": StageOption(
        stage="diarization",
        option_id="tailnet_dgx_diarization_community1",
        provider="tailnet_dgx",
        model="pyannote/speaker-diarization-community-1",
        endpoint="http://{dgx_tailnet_host}:8001",
        research_ref=_DIARIZATION_RESEARCH_REF,
        headline_metric=(
            "same community-1 win, served by the DGX diarize service on a "
            "parallel pyannote-4 container (3.1 kept ready for rollback)"
        ),
        measured_at="2026-07-11",
        tier="primary",
        resident_memory_gb=2.0,
    ),
    "pyannote_diarization_31": StageOption(
        stage="diarization",
        option_id="pyannote_diarization_31",
        provider="local",  # in-process pyannote; matches the diarization_provider Literal
        model="pyannote/speaker-diarization-3.1",
        research_ref=_DIARIZATION_RESEARCH_REF,
        headline_metric="prior default; wins brief-cameo count, merges panels. Gated (HF token).",
        measured_at="2026-07-11",
        tier="fallback",
        resident_memory_gb=2.0,
    ),
    "tailnet_dgx_diarization_31": StageOption(
        stage="diarization",
        option_id="tailnet_dgx_diarization_31",
        provider="tailnet_dgx",
        model="pyannote/speaker-diarization-3.1",
        endpoint="http://{dgx_tailnet_host}:8001",
        research_ref=_DIARIZATION_RESEARCH_REF,
        headline_metric="prior DGX default; kept deployable for instant rollback.",
        measured_at="2026-07-11",
        tier="fallback",
        resident_memory_gb=2.0,
    ),
    # Deepgram standalone diarization pass — the cloud profiles' backend
    # (``diarization_provider: deepgram``). Not pyannote: Deepgram diarizes as a
    # separate cloud call, so these profiles never touch the pyannote pipeline.
    "deepgram_diarization_nova3": StageOption(
        stage="diarization",
        option_id="deepgram_diarization_nova3",
        provider="deepgram",
        model="nova-3-general",
        research_ref="docs/guides/eval-reports/EVAL_DEEPGRAM_TRANSCRIPTION_2026_06_13.md",
        headline_metric="cloud-profile diarization via Deepgram nova-3-general (standalone pass)",
        measured_at="2026-06-13",
        tier="primary",
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
    kg: str  # StageOption.option_id key into _KG_OPTIONS
    ner: str  # StageOption.option_id key into _NER_OPTIONS
    clustering: str  # StageOption.option_id key into _CLUSTERING_OPTIONS
    gi: str  # StageOption.option_id key into _GI_OPTIONS
    diarization: str  # StageOption.option_id key into _DIARIZATION_OPTIONS
    # Who finds the quote that backs an insight. Writing the insight and grounding it are different
    # jobs, and until this stage existed the choice was made by a config fallback nobody could see.
    grounding: str = "llm_matched_to_summary"  # key into _GROUNDING_OPTIONS
    # Ordered failover ladders per stage (StageOption ids, tried after the primary on an infra
    # failure). Registry-governed: the resolver emits <stage>_fallback_providers into the
    # materialized profile so the chain is in the YAML, not inferred at runtime (RFC-106 / #1198).
    # Convention (DGX profiles): remaining on-prem tiers first, then the cloud_balanced option for
    # that stage. Empty = no fallback. Tuples because ProfilePreset is frozen.
    transcription_fallback: Tuple[str, ...] = ()
    diarization_fallback: Tuple[str, ...] = ()
    summary_fallback: Tuple[str, ...] = ()
    # RFC-106 (#1198) fail-closed switch. When False, the resolver refuses to emit any cloud
    # StageOption into a fallback chain — the ladder ends at the last on-prem tier. This is what
    # makes a no-cloud / airgapped profile safe to give a chain at all: it can degrade across its
    # DGX/local tiers but never phones out.
    allow_cloud_fallback: bool = True
    # ADR-119 (#1253) resilience posture, canonicalized so every profile self-declares it rather
    # than relying on the Config default (the gap that let a reprocess run drop to serve/failover).
    # Registry-backed presets are all serving pipelines -> serve/failover; the yaml-only
    # reprocess_dgx_* profiles declare reprocess/hold directly (they have no preset here).
    resilience_run_context: str = "serve"
    resilience_failure_strategy: str = "failover"
    notes: Optional[str] = None


# The value gate's judge, in preference order. NOT a fixed choice — a POLICY, because the right
# judge depends on who is being judged.
#
# #939: a model asked to grade its own output is lenient. Measured across 7 providers, self-grading
# drops ~10% of insights where an independent judge drops ~25% of the SAME output — half as strict.
# So a pinned literal judge is a trap: pin `anthropic` and the Anthropic candidate grades itself and
# collects a free pass, while every other arm is held to the stricter bar. The bake-off would report
# our judge assignment as model quality.
#
# The resolver therefore DERIVES the judge: first entry whose vendor differs from the summariser.
_VALUE_GATE_JUDGES: Tuple[Tuple[str, str], ...] = (
    ("anthropic", "claude-haiku-4-5-20251001"),
    ("gemini", "gemini-2.5-flash-lite"),
    ("openai", "gpt-5.4-mini"),
)

# THE GATE IS AN LLM ASKING A QUESTION. That single fact decides all of this.
#
# On the pure-ML path — sentence-transformers, summllama, the local extractive stack — there is no
# LLM to ask. The gate is not "disabled by preference" there; it is INAPPLICABLE, and there is no
# such thing as a judge on that path. Enabling it would either crash or, far worse, reach for a
# hosted judge: an `airgapped` profile making a network call is the one thing airgapped means it
# cannot do, and a `dev` profile doing it would put real paid LLM calls into CI.
_LLM_PROVIDERS: frozenset = frozenset(
    {"anthropic", "deepseek", "gemini", "grok", "mistral", "ollama", "openai"}
)

# LLMs that run locally. They CAN judge, but no INDEPENDENT judge is reachable from them — the only
# model on the box is the one being graded. So the gate self-grades, which is lenient (#939) and
# recorded here explicitly rather than being quietly true: an offline run still trims filler, but
# its gate counts are not comparable with a cloud arm's.
_LOCAL_ONLY_LLM: frozenset = frozenset({"ollama"})


def resolve_value_gate(summary_provider: str) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """``(gate_enabled, judge)`` for a summariser. The registry's voice on who grades what.

    Three tiers, because there are three genuinely different situations:

    * **not an LLM** (transformers / summllama) -> NO GATE. Nothing to judge with.
    * **local LLM** (ollama) -> gate on, no independent judge; it self-grades.
    * **hosted LLM** -> gate on, judged by the first policy vendor that is NOT the defendant.
    """
    if summary_provider not in _LLM_PROVIDERS:
        return False, None
    if summary_provider in _LOCAL_ONLY_LLM:
        return True, None
    for provider, model in _VALUE_GATE_JUDGES:
        if provider != summary_provider:
            return True, (provider, model)
    raise RuntimeError(
        f"No vendor-disjoint judge available for summariser '{summary_provider}'. Every judge in "
        f"_VALUE_GATE_JUDGES shares its vendor, so the gate would self-grade (#939)."
    )


# The fields the REGISTRY owns. A profile YAML does not get a vote on these — it is a downstream
# VIEW, regenerated from here by `make profiles-materialize`, and the drift test fails if it says
# anything different.
#
# Declared ONCE, and read by both the materialiser and the drift test, because the alternative is a
# fourth hand-maintained allowlist and the first three are what put us here: a key nobody copies
# does not error, it silently takes a default, and the default is usually "off". Concretely, this
# repo shipped `gi_max_insights` = 12 in the registry, 50 in the profile and 20 in the Config
# default — three doors, three answers — while the researched configuration
# (`provider_chunked_gated_v3`, with the temperature pin an eval had already established) sat in the
# registry as an ORPHAN that no preset pointed at.
#
# To promote a newly-measured value: change it HERE (in the StageOption, with a research_ref), run
# `make profiles-materialize`, and every profile inherits it. That is the whole loop.
REGISTRY_GOVERNED_FIELDS: Tuple[str, ...] = (
    # Routing — which model runs each stage.
    "transcription_provider",
    "summary_provider",
    "summary_model",
    "kg_extraction_source",
    "kg_max_topics",
    "kg_max_entities",
    "speaker_detector_provider",
    "ner_model",
    "topic_cluster_threshold",
    "insight_cluster_threshold",
    "diarization_model",
    "dgx_diarize_model",
    "deepgram_diarization_model",
    # RFC-106 (#1198): the ordered per-stage failover ladders are registry-governed too, so a stale
    # chain in a profile is caught by profiles-check like any other drift.
    "transcription_fallback_providers",
    "diarization_fallback_providers",
    "summary_fallback_providers",
    # ADR-119 (#1253): resilience posture — governed so a profile can't silently diverge from its
    # preset's serve/failover (or, in future, a hold) declaration.
    "resilience_run_context",
    "resilience_failure_strategy",
    # GI tuning — what an insight IS, and what evidence it must carry. Every one of these was
    # measured; leaving any of them to a code default is how the eval and the pipeline came to run
    # two different configurations.
    "gi_insight_source",
    "gi_max_insights",
    "gi_require_grounding",
    "gil_evidence_quote_mode",
    "gil_evidence_nli_mode",
    "gi_insight_chunk_chars",
    "gi_insight_dedupe_threshold",
    "gi_insight_temperature",
    "gi_insight_prompt_version",
    "gi_value_gate_enabled",
    "gi_value_gate_min_tier",
    "gi_qa_score_min",
    "gi_nli_entailment_min",
    "gil_evidence_match_summary_provider",
)


# Profile presets — canonical compositions. Profile YAMLs are downstream views.
_PROFILE_PRESETS: Dict[str, ProfilePreset] = {
    "cloud_balanced": ProfilePreset(
        name="cloud_balanced",
        transcription="openai_whisper_1",
        summary="gemini_flash_lite",
        kg="provider_n10_15",
        ner="gemini_speaker_detector",
        clustering="topic_clusters_default_0_75",
        gi="provider_chunked_gated_v3",
        diarization="deepgram_diarization_nova3",
        notes="Production cloud default. Best compound (quality × cost × latency).",
    ),
    "cloud_thin": ProfilePreset(
        name="cloud_thin",
        transcription="openai_whisper_1",
        summary="gemini_flash_lite",
        kg="provider_n10_15",
        ner="gemini_speaker_detector",
        clustering="topic_clusters_default_0_75",
        gi="provider_chunked_gated_v3",
        diarization="deepgram_diarization_nova3",
        notes="Minimal cloud feature set. Same providers as cloud_balanced.",
    ),
    "cloud_with_dgx_primary": ProfilePreset(
        name="cloud_with_dgx_primary",
        transcription="moss_transcribe_diarize",  # #1174 winner (MOSS :8004; whisper fallback)
        summary="gemini_flash_lite",
        kg="provider_n10_15",
        ner="gemini_speaker_detector",
        clustering="topic_clusters_default_0_75",
        gi="provider_chunked_gated_v3",
        diarization="tailnet_dgx_diarization_community1",
        # RFC-106 (#1198): free tiers first, paid cloud last. Transcription: MOSS -> DGX
        # faster-whisper -> local in-process whisper -> cloud whisper. Diarization: DGX pyannote ->
        # local in-process pyannote -> deepgram. summary is already the cloud tier (gemini) so it
        # needs no fallback. The local-whisper id is referenced for its provider ('whisper'); the
        # actual device is resolved from the profile, not the StageOption's mps measurement hint.
        transcription_fallback=(
            "tailnet_dgx_speaches_thread_b",
            "local_mps_large_v3",
            "openai_whisper_1",
        ),
        diarization_fallback=("pyannote_diarization_community1", "deepgram_diarization_nova3"),
        notes=(
            "Production hybrid: DGX faster-whisper (:8000 Speaches) for transcription, cloud "
            "Gemini for summary. Routing flipped :8002 whisper-openai -> :8000 Speaches on "
            "2026-06-16 (#952): int8 Speaches wins WER (10.23% vs 12.97% on Deepgram silver) "
            "and speed. :8002 stays a comparison sibling only, not the prod path."
        ),
    ),
    "local_dgx_balanced": ProfilePreset(
        name="local_dgx_balanced",
        transcription="local_mps_large_v3",
        summary="ollama_qwen35_35b",  # #928 winner; #958 Cell D confirms Q4 robustness
        kg="provider_n10_15",
        ner="spacy_trf",  # +13pp v2 recall vs sm per #906 Tier 3
        clustering="topic_clusters_default_0_75",
        gi="provider_chunked_gated_v3",
        diarization="tailnet_dgx_diarization_community1",
        notes="Local pipeline with DGX summary; laptop runs MPS transcription.",
    ),
    "local_dgx_full": ProfilePreset(
        name="local_dgx_full",
        transcription="local_mps_large_v3",
        summary="ollama_qwen35_35b",
        kg="provider_n10_15",
        ner="spacy_trf",
        clustering="topic_clusters_default_0_75",
        gi="provider_chunked_gated_v3",
        diarization="tailnet_dgx_diarization_community1",
        notes="Higher resident memory budget; same registry choices as balanced today.",
    ),
    # The profile the v3 work is actually measured on. Everything the DGX evals ran — the grounding
    # bake-off, the gemini-vs-qwen comparison, the chunking and value-gate tuning — ran THIS shape,
    # and until now it had no registry entry at all: it was a YAML-only profile, so the registry
    # could not govern the one profile that mattered. Meanwhile the registered prod_dgx_* presets
    # point at the vLLM `autoresearch` slot, a stack GI has never been evaluated on.
    #
    # Named `experiment_` deliberately: the DGX profile set is being consolidated once the qwen work
    # lands, and calling this `prod` before that decision is made would be claiming a conclusion we
    # have not reached.
    "experiment_dgx_only": ProfilePreset(
        name="experiment_dgx_only",
        transcription="tailnet_dgx_speaches_thread_b",
        summary="ollama_qwen35_35b",
        kg="provider_n10_15",
        ner="ollama_speaker_detector",
        clustering="topic_clusters_default_0_75",
        gi="provider_chunked_gated_v3",  # the tuned params, carried BY the registry
        diarization="tailnet_dgx_diarization_community1",
        grounding="llm_matched_to_summary",  # qwen grounds its own insights: 82% cov, 0% fabricated
        notes=(
            "All-DGX, no cloud call at any stage: faster-whisper (:8000), pyannote community-1 "
            "(:8001), qwen3.5:35b on Ollama (:11434). The GI stage carries the v3 tuning "
            "(uncapped ceiling + chunking + value gate + temperature 0)."
        ),
    ),
    "prod_dgx_full_with_fallback": ProfilePreset(
        name="prod_dgx_full_with_fallback",
        transcription="moss_transcribe_diarize",  # #1174 winner (MOSS :8004) + cloud fallback
        # #1022 Cell F daily-driver champion (supersedes Qwen3.5-35B-A3B top dog for routine prod)
        summary="vllm_qwen3_30b_a3b_nvfp4",
        kg="provider_n10_15",
        ner="gemini_speaker_detector",  # sub-cent, better than spacy on names
        clustering="topic_clusters_default_0_75",
        gi="provider_chunked_gated_v3",
        diarization="tailnet_dgx_diarization_community1",
        # RFC-106 (#1198): free tiers first, paid cloud last. MOSS -> DGX faster-whisper -> local
        # in-process whisper -> cloud whisper; DGX pyannote -> local pyannote -> deepgram; DGX vLLM
        # summary -> cloud gemini (the cloud_balanced summary tier). All-DGX intent = exhaust the
        # free/on-prem ladder before paying cloud.
        transcription_fallback=(
            "tailnet_dgx_speaches_thread_b",
            "local_mps_large_v3",
            "openai_whisper_1",
        ),
        diarization_fallback=("pyannote_diarization_community1", "deepgram_diarization_nova3"),
        summary_fallback=("gemini_flash_lite",),
        notes=(
            "Prod-ready all-DGX (#923): whisper + summary + GI + KG on the GB10, "
            "Gemini for the cheap speaker-detect, cloud Gemini as the summary "
            "degradation_policy.fallback_provider_on_failure. Marginal cost ≈ $0 "
            "vs cloud_with_dgx_primary's ~$0.85/mo at ~100ep/mo. "
            "2026-06-17 UPDATE: summary stage flipped from ollama qwen3.5:35b "
            "(#928 Cell C winner) to vLLM Qwen3.5-35B-A3B (#1016 R3 top dog). "
            "Wins 2 of 3 stages decisively (summary 59.4% R-1 vs Opus cohort "
            "leader, KG 38% cohort leader, GI 36% second). Cost: needs "
            "--reasoning-parser=qwen3 server flag; carries +3.6 Sonnet-mimicry "
            "on summary so any prod judge panel must be cross-vendor. If "
            "judge-bias concerns dominate (third-party leaderboards, customer-"
            "facing audited rankings), use prod_dgx_balanced (Moonlight safe "
            "pick) instead. "
            "OPERATIONAL GATE: the #963 / #996 catastrophic-tail risk applies — "
            "do not run autoresearch sweeps on coder-next vLLM concurrent with a "
            "pipeline run on this profile. The transcription and summary stages "
            "are sequential per-episode so on-profile they don't overlap, but "
            "external vLLM sweeps will. "
            "2026-06-19 UPDATE (#1022 Cell F): the autoresearch-slot model has "
            "been swapped from Qwen3.5-35B-A3B (bf16) to "
            "NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4 — 1.7× faster end-to-end at "
            "-4.7% quality sum, wins GI stage outright. The "
            "vllm_qwen3_5_35b_a3b StageOption is retained for highest-stakes "
            "one-shot evals (manual compose swap)."
        ),
    ),
    "prod_dgx_balanced": ProfilePreset(
        name="prod_dgx_balanced",
        transcription="moss_transcribe_diarize",  # #1174 winner (MOSS :8004)
        # #1022 Cell F (supersedes Moonlight safe pick: same speed, +161% GI, +45% KG, -44% mem)
        summary="vllm_qwen3_30b_a3b_nvfp4",
        kg="provider_n10_15",
        ner="gemini_speaker_detector",
        clustering="topic_clusters_default_0_75",
        gi="provider_chunked_gated_v3",
        diarization="tailnet_dgx_diarization_community1",
        # RFC-106 (#1198): free tiers first, paid cloud last. MOSS -> DGX faster-whisper -> local
        # in-process whisper -> cloud whisper; DGX pyannote -> local pyannote -> deepgram; DGX vLLM
        # summary -> cloud gemini (the cloud_balanced summary tier).
        transcription_fallback=(
            "tailnet_dgx_speaches_thread_b",
            "local_mps_large_v3",
            "openai_whisper_1",
        ),
        diarization_fallback=("pyannote_diarization_community1", "deepgram_diarization_nova3"),
        summary_fallback=("gemini_flash_lite",),
        notes=(
            "Prod variant of prod_dgx_full_with_fallback that swaps the summary "
            "stage to vLLM-served Moonlight-16B-A3B (#1016 Round 3 safe pick). "
            "Trade-off vs prod_dgx_full_with_fallback (Qwen3.5-35B-A3B top dog): "
            "- 33% faster (9s/ep vs 13.6s/ep) "
            "- half the resident memory (32 GB vs 67 GB) — DGX-contention friendly "
            "- style-neutral (Δ S−O = 0.0) — no judge-vendor caveat "
            "- ROUGE-1 -1.9pt vs the top-dog Qwen3.5 (57.5% vs 59.4%) "
            "Pick this profile when (a) DGX is shared with other workloads, "
            "(b) downstream judge vendor is unknown/Sonnet-heavy, or (c) "
            "audit-clean ranking matters more than peak quality. Otherwise "
            "use prod_dgx_full_with_fallback. "
            "2026-06-19 UPDATE (#1022 Cell F): the autoresearch-slot model "
            "behind this profile's summary stage has been swapped from "
            "Moonlight-16B-A3B (bf16, 32 GB) to "
            "NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4 (18 GB). Cell F dominates "
            "Moonlight in 4 of 5 dimensions: loses summary rouge1 by 6% but "
            "wins GI by 161%, KG by 45%, end-to-end speed by 5%, weight "
            "footprint by 44%. Style-neutrality note: Cell F has not been "
            "Δ S−O measured; if strict bias-clean ranking is essential, "
            "manually swap to Moonlight via homelab compose for that run."
        ),
    ),
    "eval_default": ProfilePreset(
        name="eval_default",
        transcription="tailnet_dgx_speaches_thread_b",
        # #1022 Cell F (supersedes Moonlight; same architecture + faster + GI winner)
        summary="vllm_qwen3_30b_a3b_nvfp4",
        kg="provider_n10_15",
        ner="gemini_speaker_detector",
        clustering="topic_clusters_default_0_75",
        gi="provider_chunked_gated_v3",
        diarization="tailnet_dgx_diarization_community1",
        notes=(
            "Internal autoresearch eval-loop default. Uses Moonlight-16B-A3B "
            "(#1016 Round 3 safe pick) as the summary stage because we're our "
            "own downstream consumer — bias-clean baseline gives more defensible "
            "self-comparison when running future autoresearch cohorts. "
            "Style-neutral (Δ S−O = 0.0) means new candidates' rankings don't "
            "get distorted by judge-vendor bias. The autoresearch compose at "
            "~/agentic-ai-homelab/infra/vllm/autoresearch/docker-compose.yml has "
            "had --max-num-seqs=4 + --enforce-eager defaults applied since "
            "2026-06-17 (lever A + C from #1016 §5) — these are right for the "
            "eval-loop iteration cadence. "
            "2026-06-19 UPDATE (#1022 Cell F): summary stage swapped from "
            "Moonlight-16B-A3B to NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4 "
            "after Cell F entered the cohort. Same speed as Moonlight, "
            "significantly better GI/KG, half the memory. Style-neutrality "
            "caveat (above) applies."
        ),
    ),
    "cloud_quality": ProfilePreset(
        name="cloud_quality",
        transcription="deepgram_nova_3",
        summary="anthropic_haiku_4_5",
        kg="provider_n10_15",
        ner="gemini_speaker_detector",  # 2026-06-17 drift fix: YAML chose Gemini per v3 research
        clustering="topic_clusters_default_0_75",
        gi="provider_chunked_gated_v3",
        diarization="deepgram_diarization_nova3",
        notes=(
            "Cloud quality-first profile: Deepgram for transcription (best WER + best "
            "latency on v2 fixtures), Anthropic Haiku 4.5 for summary "
            "(compound-winner per EVAL_HELDOUT_V2). NER flipped from spacy_trf to "
            "gemini_speaker_detector per the YAML's v3 (22ep) research note."
        ),
    ),
    "local": ProfilePreset(
        name="local",
        transcription="local_whisper_small_en",
        summary="ollama_hermes3_8b_laptop",
        kg="provider_n10_15",
        ner="spacy_trf",
        clustering="topic_clusters_default_0_75",
        gi="provider_chunked_gated_v3",
        diarization="pyannote_diarization_community1",
        notes=(
            "Laptop-only profile: small.en whisper (quality upgrade over base.en) + "
            "Ollama hermes3:8b summary. No DGX, no cloud."
        ),
    ),
    # ── Dev / airgapped tier (#1060 — promoted 2026-06-23) ─────────────
    # All three profiles below ship the same NER + clustering + GI/KG
    # caveat: provider-source GI/KG is a no-op when summary_provider is
    # not an LLM (summllama / transformers are summary-only). Same shape
    # as the cloud/DGX presets so the drift test treats them uniformly.
    "airgapped": ProfilePreset(
        name="airgapped",
        transcription="local_whisper_medium_en",
        summary="summllama_3_2_3b_paragraph",
        kg="provider_n10_15",
        ner="spacy_trf",
        clustering="topic_clusters_default_0_75",
        gi="provider_chunked_gated_v3",
        diarization="pyannote_diarization_community1",
        grounding="ml_qa_nli",  # no LLM in this profile
        # Airgapped: never phone out. Fail-closed so any fallback ladder ends on-prem (RFC-106).
        allow_cloud_fallback=False,
        notes=(
            "Airgapped quality default: medium.en Whisper + SummLlama 3.2-3B "
            "paragraph + spaCy trf. No network, no Ollama. SummLlama is "
            "summary-only, so the provider-side GI/KG paths are no-ops on "
            "this profile (KG/GI are effectively disabled until an LLM "
            "summary_provider replaces summllama). #571 / #652 / #653 — "
            "EVAL_DEV_TIER_REGISTRY_2026_06_23.md re-measured on smoke_v2."
        ),
    ),
    "airgapped_thin": ProfilePreset(
        name="airgapped_thin",
        transcription="local_whisper_tiny_en",
        summary="transformers_bart_small_long_fast_authority",
        kg="provider_n10_15",
        ner="spacy_sm",
        clustering="topic_clusters_default_0_75",
        gi="provider_chunked_gated_v3",
        diarization="pyannote_diarization_community1",
        grounding="ml_qa_nli",  # no LLM in this profile
        # Airgapped: never phone out. Fail-closed so any fallback ladder ends on-prem (RFC-106).
        allow_cloud_fallback=False,
        notes=(
            "Airgapped thin: tiny.en Whisper + bart-small+long-fast "
            "(ml_small_authority mode) + spaCy sm. Lowest RAM / startup "
            "cost in the registry — matches Docker stack-test constraints. "
            "transformers is summary-only so provider-side GI/KG paths "
            "are no-ops here too. Tier-3 stack-test asserts the strict "
            "no-LLM v3.0 capability subset (Insight + Person + Topic + "
            "ABOUT/SPOKEN_BY/SUPPORTED_BY/HAS_INSIGHT edges) per the YAML's "
            "capability-set documentation."
        ),
    ),
    "dev": ProfilePreset(
        name="dev",
        transcription="local_whisper_tiny_en",
        summary="transformers_bart_small_long_fast_authority",
        kg="provider_n10_15",
        ner="spacy_sm",
        clustering="topic_clusters_default_0_75",
        gi="provider_chunked_gated_v3",
        diarization="pyannote_diarization_community1",
        grounding="ml_qa_nli",  # no LLM in this profile
        notes=(
            "Dev / CI: same shape as airgapped_thin (tiny.en + "
            "bart-small+long-fast + spaCy sm) but the YAML disables GI / "
            "KG / vector_search via generate_gi=false etc. — the registry "
            "preset captures the COMPOSITIONAL choice; the YAML's "
            "boolean toggles still apply on top via Field defaults / "
            "explicit overrides. The drift test only checks routing "
            "fields, not the on/off toggles."
        ),
    ),
    # ── Pre-prod / dress-rehearsal tier (#1060 follow-up FU1, 2026-06-23) ──
    "preprod_local_whisper": ProfilePreset(
        name="preprod_local_whisper",
        transcription="local_whisper_small_en",
        summary="gemini_flash_lite",
        kg="provider_n10_15",
        ner="gemini_speaker_detector",
        clustering="topic_clusters_default_0_75",
        gi="provider_chunked_gated_v3",
        diarization="pyannote_diarization_community1",
        notes=(
            "Stage-A dress rehearsal for cloud_with_dgx_primary: mirrors "
            "every stage choice EXCEPT transcription, which runs on the "
            "laptop's small.en Whisper (~10× realtime on MPS) instead of "
            "the DGX whisper-openai container. Validates the full prod "
            "pipeline shape (cloud Gemini LLM + screenplay + diarize + GI "
            "evidence + KG + vectors) from a laptop before the DGX Whisper "
            "service exists. For the FINAL dress rehearsal the operator "
            "overrides whisper_model to large-v3 to match the DGX service "
            "(~0.6× realtime on M-series) — that override is a YAML edit "
            "or CLI flag, not a preset change. See DGX_RUNBOOK § Pre-prod "
            "laptop validation."
        ),
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


def get_gi_options() -> Dict[str, StageOption]:
    """All registered GI options (id → StageOption). Empty pending #978."""
    return dict(_GI_OPTIONS)


def get_kg_options() -> Dict[str, StageOption]:
    """All registered KG options (id → StageOption)."""
    return dict(_KG_OPTIONS)


def get_ner_options() -> Dict[str, StageOption]:
    """All registered NER options (id → StageOption)."""
    return dict(_NER_OPTIONS)


def get_clustering_options() -> Dict[str, StageOption]:
    """All registered clustering options (id → StageOption)."""
    return dict(_CLUSTERING_OPTIONS)


def get_kg_option(option_id: str) -> StageOption:
    """Look up a single KG option by id."""
    if option_id not in _KG_OPTIONS:
        raise ValueError(f"Unknown KG option '{option_id}'")
    return _KG_OPTIONS[option_id]


def get_ner_option(option_id: str) -> StageOption:
    """Look up a single NER option by id."""
    if option_id not in _NER_OPTIONS:
        raise ValueError(f"Unknown NER option '{option_id}'")
    return _NER_OPTIONS[option_id]


def get_clustering_option(option_id: str) -> StageOption:
    """Look up a single clustering option by id."""
    if option_id not in _CLUSTERING_OPTIONS:
        raise ValueError(f"Unknown clustering option '{option_id}'")
    return _CLUSTERING_OPTIONS[option_id]


def get_grounding_options() -> Dict[str, StageOption]:
    """All registered grounding options (id → StageOption)."""
    return dict(_GROUNDING_OPTIONS)


def get_grounding_option(option_id: str) -> StageOption:
    """Look up a single grounding option by id."""
    if option_id not in _GROUNDING_OPTIONS:
        raise ValueError(f"Unknown grounding option '{option_id}'")
    return _GROUNDING_OPTIONS[option_id]


def get_diarization_options() -> Dict[str, StageOption]:
    """All registered diarization options (id → StageOption)."""
    return dict(_DIARIZATION_OPTIONS)


def get_diarization_option(option_id: str) -> StageOption:
    """Look up a single diarization option by id."""
    if option_id not in _DIARIZATION_OPTIONS:
        raise ValueError(f"Unknown diarization option '{option_id}'")
    return _DIARIZATION_OPTIONS[option_id]


def get_gi_option(option_id: str) -> StageOption:
    """Look up a single GI option by id."""
    if option_id not in _GI_OPTIONS:
        raise ValueError(f"Unknown GI option '{option_id}'")
    return _GI_OPTIONS[option_id]


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


# Cloud vendors a fallback tier can reach. A cloud vendor pointed at a DGX/local endpoint (e.g.
# vLLM served over the openai protocol) is on-prem, not cloud — see _is_cloud_option.
_CLOUD_FALLBACK_VENDORS: Final = frozenset(
    {"openai", "gemini", "anthropic", "deepgram", "mistral", "deepseek", "grok", "cohere"}
)


def _is_cloud_option(opt: StageOption) -> bool:
    """Whether ``opt`` is a hosted-cloud tier (RFC-106 fail-closed classification).

    A cloud vendor served from a DGX/local endpoint (vLLM over the openai protocol, a localhost
    Ollama, etc.) is on-prem despite the vendor name, so it is NOT treated as cloud.
    """
    if opt.provider not in _CLOUD_FALLBACK_VENDORS:
        return False
    endpoint = opt.endpoint or ""
    if "{dgx_tailnet_host}" in endpoint or "localhost" in endpoint or "127.0.0.1" in endpoint:
        return False
    return True


def _emit_fallback_chains(preset: ProfilePreset, settings: Dict[str, Any]) -> None:
    """RFC-106 (#1198): map each stage's ordered fallback StageOption ids to their provider values
    and emit ``<stage>_fallback_providers`` into ``settings``.

    Keeps the failover ladder in the materialized profile (the runtime FallbackChain reads it) and
    lets ``profiles-check`` guard it like every other registry-governed field. Empty ladder -> no
    key emitted (a stage with no fallback stays absent rather than emitting ``[]``).

    Fail-closed: when ``preset.allow_cloud_fallback`` is False, cloud tiers are dropped so the
    ladder ends at the last on-prem tier and a no-cloud profile never phones out.
    """
    chains = (
        (
            preset.transcription_fallback,
            get_transcription_option,
            "transcription_fallback_providers",
        ),
        (preset.diarization_fallback, get_diarization_option, "diarization_fallback_providers"),
        (preset.summary_fallback, get_summary_option, "summary_fallback_providers"),
    )
    for ids, get_option, key in chains:
        options = [get_option(oid) for oid in ids]
        if not preset.allow_cloud_fallback:
            options = [opt for opt in options if not _is_cloud_option(opt)]
        if options:
            settings[key] = [opt.provider for opt in options]


def resolve_profile_to_settings(
    name: str,
    dgx_tailnet_host: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve a profile preset to the runtime settings the pipeline expects.

    Returns a flat dict the Config can ingest. Profile YAMLs become thin
    viewers — a YAML reads `profile: cloud_with_dgx_primary` and this
    resolver fills in the rest at validation time.

    Args:
        name: ProfilePreset name (e.g. "cloud_with_dgx_primary").
        dgx_tailnet_host: optional explicit Tailscale host; threaded into
            resolve_endpoint() for each stage option's endpoint template.
            When None, the resolver's per-endpoint call falls back to the
            DGX_TAILNET_HOST env var, then to the fail-fast sentinel.
    """
    preset = get_profile_preset(name)
    tx = get_transcription_option(preset.transcription)
    sm = get_summary_option(preset.summary)
    kg = get_kg_option(preset.kg)
    ner = get_ner_option(preset.ner)
    clustering = get_clustering_option(preset.clustering)
    gi = get_gi_option(preset.gi)
    dia = get_diarization_option(preset.diarization)

    settings: Dict[str, Any] = {
        "transcription_provider": tx.provider,
    }
    if tx.model is not None:
        # Field name depends on provider — keeping the canonical mapping minimal here.
        # Consumers may need additional translation; this returns the registry view.
        settings.setdefault("transcription_model", tx.model)
    resolved_tx_endpoint = resolve_endpoint(tx.endpoint, dgx_tailnet_host)
    if resolved_tx_endpoint is not None:
        settings["transcription_endpoint"] = resolved_tx_endpoint
    if tx.extra_settings:
        settings["transcription_extra"] = dict(tx.extra_settings)

    settings["summary_provider"] = sm.provider
    if sm.model is not None:
        settings["summary_model"] = sm.model
    resolved_sm_endpoint = resolve_endpoint(sm.endpoint, dgx_tailnet_host)
    if resolved_sm_endpoint is not None:
        settings["summary_endpoint"] = resolved_sm_endpoint
    if sm.extra_settings:
        settings["summary_extra"] = dict(sm.extra_settings)

    _emit_fallback_chains(preset, settings)

    # ADR-119 (#1253): resilience posture is registry-governed, so every materialized profile
    # self-declares it (invocation-agnostic — the reprocess make targets load via --config, which
    # bypasses name-derivation) and profiles-check catches drift.
    settings["resilience_run_context"] = preset.resilience_run_context
    settings["resilience_failure_strategy"] = preset.resilience_failure_strategy

    # GI: insight source + caps + grounding + evidence-stack bundling + the tuned params.
    #
    # The registry carries the PARAMS an eval established, not just the model. Anything measured in
    # data/eval and then left out of this mapping is a setting production silently does not run —
    # which is exactly how the pipeline came to be evaluated in one configuration and shipped in
    # another. Add the key here when you add it to a StageOption.
    settings["gi_insight_source"] = gi.provider

    # The judge is DERIVED, never copied: it must not share a vendor with the model it grades
    # (#939). A literal in the YAML cannot satisfy that, because the correct judge changes with the
    # summariser — which is exactly the bug waiting in the bake-off, where the Anthropic arm would
    # have been graded by the pinned Anthropic judge and scored against rivals held to a stricter
    # bar.
    _gate_on, _judge = resolve_value_gate(sm.provider)
    if _judge is not None:
        settings["gi_value_gate_provider"], settings["gi_value_gate_model"] = _judge

    _GI_SETTING_TO_CONFIG_KEY = {
        "max_insights": "gi_max_insights",
        "require_grounding": "gi_require_grounding",
        "evidence_quote_mode": "gil_evidence_quote_mode",
        "evidence_nli_mode": "gil_evidence_nli_mode",
        "insight_chunk_chars": "gi_insight_chunk_chars",
        "insight_dedupe_threshold": "gi_insight_dedupe_threshold",
        "value_gate_enabled": "gi_value_gate_enabled",
        "value_gate_min_tier": "gi_value_gate_min_tier",
        "value_gate_provider": "gi_value_gate_provider",
        "value_gate_model": "gi_value_gate_model",
        # A DEDICATED field, not `{provider}_temperature`. That field also drives summarisation and
        # speaker detection, so routing the insight pin through it silently re-tuned two unrelated
        # stages — and a "model difference" in a bake-off would have included a summariser we
        # quietly changed underneath it.
        "insight_temperature": "gi_insight_temperature",
        # The prompt decides what an insight IS, so it is a tuned parameter like any other and
        # belongs to the researched configuration, not to a code default. (v3 LOST its A/B — route
        # kappa 0.57 vs v2's 0.67 — so v2 is the measured choice, and the registry must say so
        # rather than leaving it to whoever last edited a YAML.)
        "insight_prompt_version": "gi_insight_prompt_version",
        # The evidence floors. These decide what counts as SUPPORT, which is the difference between
        # a grounded corpus and a plausible one.
        "qa_score_min": "gi_qa_score_min",
        "nli_entailment_min": "gi_nli_entailment_min",
        "evidence_match_summary_provider": "gil_evidence_match_summary_provider",
    }
    for key, value in (gi.extra_settings or {}).items():
        config_key = _GI_SETTING_TO_CONFIG_KEY.get(key)
        if config_key is None:
            # NOT ValueError: Config._resolve_profile catches ValueError to mean "not a registry
            # preset" and silently drops to YAML-only. A typo here would therefore disable the
            # whole registry for this profile without a word — the exact silent-fallback class of
            # bug this stage exists to prevent.
            raise RuntimeError(
                f"GI option '{gi.option_id}' sets '{key}', which this resolver does not map to a "
                "Config field. A param the registry records but never plumbs is a setting "
                "production silently does not run — add it to _GI_SETTING_TO_CONFIG_KEY."
            )
        settings[config_key] = value

    # The GI option enables the gate for the researched (LLM) configuration. It CANNOT apply where
    # there is no LLM to run it: on the pure-ML path the "judge" would have to be a hosted model,
    # and handing one to `airgapped` means a network call — the single thing airgapped exists to
    # forbid — while handing one to `dev` puts paid LLM calls into CI.
    #
    # So capability overrides the blanket default. This is not the registry being ignored; it IS the
    # registry, speaking through the resolver, about a profile that cannot honour the setting.
    if not _gate_on:
        settings["gi_value_gate_enabled"] = False

    # Grounding: who finds the quote. `match_summary` means the summarising LLM grounds its own
    # insights (Config._auto_promote_evidence_providers resolves it); anything else is explicit.
    grounding = get_grounding_option(preset.grounding)
    if grounding.provider != "match_summary":
        settings["quote_extraction_provider"] = grounding.provider
        settings["entailment_provider"] = grounding.provider
    for key, value in (grounding.extra_settings or {}).items():
        if key == "nli_entailment_min":
            settings["gi_nli_entailment_min"] = value
        elif key == "qa_model":
            settings["gi_qa_model"] = value
        elif key == "nli_model":
            settings["gi_nli_model"] = value
        elif key == "qa_window_chars":
            settings["gi_qa_window_chars"] = value

    # KG: extraction source + caps.
    settings["kg_extraction_source"] = kg.provider
    if kg.extra_settings:
        if "max_topics" in kg.extra_settings:
            settings["kg_max_topics"] = kg.extra_settings["max_topics"]
        if "max_entities" in kg.extra_settings:
            settings["kg_max_entities"] = kg.extra_settings["max_entities"]

    # NER / speaker detection: provider + (when spaCy) the model name.
    settings["speaker_detector_provider"] = ner.provider
    if ner.model is not None:
        settings["ner_model"] = ner.model

    # Clustering: threshold flows through to runtime Config (#991). The same
    # value drives both ``topic_cluster_threshold`` and ``insight_cluster_threshold``
    # today — they share the registry's clustering option because the v2-fixture
    # eval (#904 Tier 1) treated them as a coupled knob. Future autoresearch
    # can split them by introducing separate StageOptions.
    if clustering.extra_settings and "threshold" in clustering.extra_settings:
        threshold = clustering.extra_settings["threshold"]
        settings["topic_cluster_threshold"] = threshold
        settings["insight_cluster_threshold"] = threshold
    settings["_clustering_research_ref"] = clustering.research_ref

    # Diarization: route the model id to the backend's config field by the option's
    # provider — pyannote in-process (``diarization_model``), the DGX diarize service
    # (``dgx_diarize_model``), or the Deepgram standalone pass
    # (``deepgram_diarization_model``). The stage on/off (``diarize``) is a YAML toggle.
    if dia.model is not None:
        if dia.provider == "tailnet_dgx":
            settings["dgx_diarize_model"] = dia.model
        elif dia.provider == "deepgram":
            settings["deepgram_diarization_model"] = dia.model
        else:  # pyannote / local
            settings["diarization_model"] = dia.model

    settings["_profile_preset"] = preset.name
    settings["_transcription_research_ref"] = tx.research_ref
    settings["_summary_research_ref"] = sm.research_ref
    settings["_kg_research_ref"] = kg.research_ref
    settings["_ner_research_ref"] = ner.research_ref
    settings["_gi_research_ref"] = gi.research_ref
    settings["_diarization_research_ref"] = dia.research_ref

    return settings
