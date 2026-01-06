# RFC-016: Modularization for AI Experiment Pipeline

- **Status**: 🟡 **80% Complete** - Core provider pattern implemented, cleanup and experiment enhancements needed
- **Authors**:
- **Stakeholders**: Maintainers, developers implementing AI experiment pipeline, developers maintaining core workflow
- **Related RFCs**: `docs/rfc/RFC-015-ai-experiment-pipeline.md`, `docs/rfc/RFC-017-prompt-management.md`, `docs/rfc/RFC-021-modularization-refactoring-plan.md` (historical reference), `docs/rfc/RFC-029-provider-refactoring-consolidation.md` (completed)
- **Related Issues**: [#303](https://github.com/chipi/podcast_scraper/issues/303) (RFC-016 Implementation)
- **Updated**: 2026-01-08

---

## 📊 Implementation Status (80% Complete)

### ✅ Completed (Core Foundation)

1. **Provider Protocols** ✅ (100% Complete)
   - `src/podcast_scraper/summarization/base.py` → `SummarizationProvider` protocol
   - `src/podcast_scraper/transcription/base.py` → `TranscriptionProvider` protocol
   - `src/podcast_scraper/speaker_detectors/base.py` → `SpeakerDetector` protocol
   - All protocols have consistent interfaces and comprehensive docstrings

2. **Unified Provider Implementations** ✅ (100% Complete)
   - `src/podcast_scraper/ml/ml_provider.py` → `MLProvider` (implements all 3 protocols)
     - Whisper (transcription)
     - spaCy NER (speaker detection)
     - Transformers BART/LED (summarization)
   - `src/podcast_scraper/openai/openai_provider.py` → `OpenAIProvider` (implements all 3 protocols)
     - Whisper API (transcription)
     - GPT API (speaker detection)
     - GPT API (summarization)

3. **Factory Functions** ✅ (100% Complete)
   - `src/podcast_scraper/summarization/factory.py` → `create_summarization_provider()`
   - `src/podcast_scraper/transcription/factory.py` → `create_transcription_provider()`
   - `src/podcast_scraper/speaker_detectors/factory.py` → `create_speaker_detector()`

4. **Production Integration** ✅ (100% Complete)
   - `src/podcast_scraper/workflow.py` uses new provider factories
   - Clean separation between orchestration and implementation
   - Proper lifecycle management

### 🟡 Remaining Work (20%)

**Phase 1: Legacy Module Cleanup** (Optional - Low Priority)
- Deprecate `src/podcast_scraper/summarizer.py`
- Deprecate `src/podcast_scraper/speaker_detection.py`
- Deprecate `src/podcast_scraper/whisper_integration.py`

**Phase 2: Experiment-Specific Factory Enhancements** (Critical - 3-5 days)
- Enable factories to accept experiment-style params dict (not just `Config`)
- Required for RFC-015 Phase 1

**Phase 3: Extract Evaluation Infrastructure** (Important - 1 week)
- Create `src/podcast_scraper/experiments/evaluation.py`
- Extract ROUGE, BLEU from `scripts/eval_summaries.py`
- Add WER calculation (jiwer)
- Add semantic similarity (sentence-transformers)
- Required for RFC-015 Phase 2

**See [GitHub Issue #303](https://github.com/chipi/podcast_scraper/issues/303) for detailed implementation plan.**

---

---

## Abstract

**🎯 Quick Summary:** This RFC is **80% complete**. The core provider pattern (protocols, implementations, factories) is production-ready. Remaining work (20%) focuses on legacy cleanup, experiment-specific factory enhancements, and evaluation infrastructure extraction. **You can start RFC-015 as soon as Phase 2 (factory enhancements) is complete (~3-5 days).**

---

This RFC identifies the refactoring and modularization needed to support the AI experiment pipeline (RFC-015) while maintaining the existing production workflow. **This refactoring aligns with and builds upon the modularization already completed in RFC-029 (Provider Refactoring Consolidation)**. The goal is to create clean abstractions that allow the experiment pipeline to run independently without duplicating code or interfering with the main pipeline.

**Key Concept**: The experiment pipeline wraps existing pieces (gold data, HF baseline, eval scripts) in a repeatable pipeline. The provider system enables this by allowing the experiment runner to reuse production providers without code duplication. Once the structure is in place, adding new providers is just "add config + small backend class".

**Note**: The provider pattern refactoring described here is the same refactoring planned in RFC-013. This RFC focuses specifically on how that refactoring enables the AI experiment pipeline, ensuring we don't duplicate work.

## Problem Statement

The current codebase has tight coupling between:

1. **Workflow orchestration** (`workflow.py`) and **model implementations** (`summarizer.py`, `metadata.py`)
2. **Evaluation scripts** (`eval_summaries.py`) and **internal data structures**
3. **Model loading** and **pipeline execution**
4. **Configuration** and **implementation details**

To support the AI experiment pipeline, we need:

- **Independent backends** that can be instantiated outside the main workflow
- **Standardized interfaces** for different provider types (local HF, OpenAI, etc.)
- **Reusable evaluation logic** that works with both production and experiment outputs
- **Separation of concerns** between experiment pipeline and production pipeline

## Goals

1. **No Duplication**: Experiment pipeline reuses production code, not duplicates it
2. **Independent Execution**: Experiments can run without running the full production pipeline
3. **Clean Interfaces**: Clear protocols/interfaces for backends (summarization, NER, transcription)
4. **Backward Compatibility**: Existing production workflow continues to work unchanged
5. **Testability**: Each component can be tested independently
6. **Extensibility**: Easy to add new backends or experiment types

## Current Architecture Analysis

### Current Structure

````text
podcast_scraper/
├── workflow.py              # Main pipeline orchestration
├── metadata.py              # Metadata generation (calls summarizer)
├── summarizer.py            # Local transformer models
├── speaker_detection.py     # NER-based speaker detection
├── config.py                # Configuration model
├── prompt_store.py          # Prompt management (RFC-017)
├── prompts/                 # Versioned prompt templates
│   ├── summarization/
│   └── ner/
└── scripts/
    ├── eval_summaries.py    # Evaluation script
    └── eval_cleaning.py     # Evaluation script
```text

- Directly imports and instantiates `summarizer.SummaryModel`
- Calls `metadata.generate_episode_metadata()` which internally calls summarizer
- Tightly coupled to specific model implementations

**`metadata.py`**:

- Directly imports `summarizer` module
- Calls `summarizer.clean_transcript()`, `summarizer.SummaryModel()`
- Hard-coded to use local transformer models

**`eval_summaries.py`**:

- Reads predictions from specific file structure
- Expects specific data formats
- Not easily reusable for experiment outputs

## Required Refactoring

### 1. Extract Provider Interfaces (Protocols)

**Create `podcast_scraper/providers/` package:**

```text
podcast_scraper/providers/
├── __init__.py
├── base.py                  # Protocol definitions
├── summarization/
│   ├── __init__.py
│   ├── base.py              # SummarizationProvider protocol
│   ├── local.py              # LocalSummarizationProvider (current summarizer.py logic)
│   ├── openai.py             # OpenAISummarizationProvider
│   └── factory.py            # SummarizationProviderFactory
├── speaker_detection/
│   ├── __init__.py
│   ├── base.py               # SpeakerDetector protocol
│   ├── ner.py                 # NERSpeakerDetector (current speaker_detection.py logic)
│   ├── openai.py              # OpenAISpeakerDetector
│   └── factory.py             # SpeakerDetectorFactory
└── transcription/
    ├── __init__.py
    ├── base.py                # TranscriptionProvider protocol
    ├── whisper.py              # WhisperTranscriptionProvider (current whisper integration)
    ├── openai.py               # OpenAITranscriptionProvider
    └── factory.py              # TranscriptionProviderFactory
```python

# podcast_scraper/providers/summarization/base.py

from typing import Protocol, Dict, Any, Optional
from .. import config

class SummarizationProvider(Protocol):
    """Protocol for summarization providers."""

    def summarize(
        self,
        text: str,
        episode_title: Optional[str] = None,
        episode_description: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Summarize text.

        Args:
            text: Transcript text to summarize
            episode_title: Optional episode title
            episode_description: Optional episode description
            params: Optional parameters (max_length, min_length, etc.)

```text
        Returns:
            Dictionary with summary results:
            {
                "summary": str,
                "summary_short": Optional[str],
                "metadata": {...}
            }
        """
        ...
```python

# podcast_scraper/providers/summarization/local.py

from typing import Dict, Any, Optional
from .base import SummarizationProvider
from .. import config

class LocalSummarizationProvider:
    """Local transformer-based summarization provider."""

    def __init__(self, cfg: config.Config):
        """Initialize local summarization provider."""
        self.cfg = cfg
        self.model = self._load_model()

    def summarize(
        self,
        text: str,
        episode_title: Optional[str] = None,
        episode_description: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

```text

        """Summarize using local transformer models."""

```

        # Current summarizer.py logic, but as a provider

```python

# podcast_scraper/metadata.py

from typing import Optional
from .providers.summarization.base import SummarizationProvider

def generate_episode_metadata(
    ...,
    summarization_provider: Optional[SummarizationProvider] = None,
    ...
) -> Optional[str]:
    """Generate episode metadata.

    Args:
        summarization_provider: Optional summarization provider.
                               If None, creates default provider from config.
    """
    if summarization_provider is None:

```python

        # Create default provider from config (backward compatibility)

```python

        from .providers.summarization.factory import create_summarization_provider
        summarization_provider = create_summarization_provider(cfg)

```

    summary_result = summarization_provider.summarize(
        text=transcript_text,
        episode_title=episode.title,
        episode_description=episode.description,
        params={"max_length": cfg.summary_max_length}
    )
    ...

```text

# podcast_scraper/preprocessing.py

"""Provider-agnostic preprocessing functions."""

def clean_transcript(
    text: str,
    remove_timestamps: bool = True,
    normalize_speakers: bool = True,
    collapse_blank_lines: bool = True,
    remove_fillers: bool = False,
) -> str:
    """Clean transcript text (moved from summarizer.py)."""

    # Current clean_transcript() implementation

    ...

def remove_sponsor_blocks(text: str) -> str:

```python

    """Remove sponsor blocks (moved from summarizer.py)."""

```python

- `metadata.py` imports from `preprocessing` instead of `summarizer`
- `providers/summarization/local.py` imports from `preprocessing` if needed
- All providers receive preprocessed text

## 3.1 Prompt Management Integration

**Prompt management is a provider-specific concern** (see RFC-017):

- **Provider Autonomy**: Each provider that needs prompts (OpenAI) uses `prompt_store` internally
- **Protocol-Agnostic**: Prompts don't appear in protocol definitions
- **Versioned Assets**: Prompts stored as `.j2` files in `prompts/` directory
- **Config-Driven**: Prompt selection via config fields

**Example Provider Implementation:**

```python

# podcast_scraper/summarization/openai_provider.py

from ..prompt_store import render_prompt

class OpenAISummarizationProvider:
    def summarize(self, text: str, cfg: config.Config, ...) -> Dict[str, Any]:

        # Load prompts internally (provider-specific)

        system_prompt = render_prompt(
            cfg.summary_system_prompt or "summarization/system_v1",
            **cfg.summary_prompt_params,
        ) if cfg.summary_system_prompt else None

        user_prompt = render_prompt(
            cfg.summary_user_prompt or "summarization/long_v1",
            transcript=text,
            **cfg.summary_prompt_params,
        )

```text

        # Use prompts in API call...

```

- ✅ Same `prompt_store` used in both application and experiments

See RFC-017 for complete prompt management design.

## 4. Create Experiment Backend Adapters

**Create `scripts/experiments/backends/`:**

```text

scripts/experiments/
├── __init__.py
├── backends/
│   ├── __init__.py
│   ├── summarization_backend.py    # Adapter for experiment pipeline
│   ├── ner_backend.py               # Adapter for experiment pipeline
│   └── transcription_backend.py    # Adapter for experiment pipeline
└── ...

```python

# scripts/experiments/backends/summarization_backend.py

from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Add parent directory to path to import podcast_scraper

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from podcast_scraper.providers.summarization.factory import create_summarization_provider
from podcast_scraper import config

class SummarizationBackend:
    """Backend adapter for experiment pipeline."""

    def __init__(self, experiment_config: Dict[str, Any]):
        """Initialize backend from experiment config."""
        self.config = experiment_config

```text

        # Convert experiment config to Config object

```
```
        self.provider = create_summarization_provider(cfg)

```python

    def summarize(
        self,
        transcript: str,
        episode_title: Optional[str] = None,
        episode_description: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Summarize transcript using configured provider."""

```
        # Merge experiment params with method params

```

            text=transcript,
            episode_title=episode_title,
            episode_description=episode_description,
            params=merged_params,
        )

```python

    def _create_config_from_experiment(self, exp_config: Dict[str, Any]) -> config.Config:
        """Convert experiment config to Config object."""

```
```python

# scripts/eval_summaries.py (refactored)

from typing import List, Dict, Any
from pathlib import Path

def evaluate_summaries(
    predictions: List[Dict[str, Any]],
    gold_data: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    """Evaluate summaries against golden dataset.

    Args:
        predictions: List of prediction dicts with:
            {
                "episode_id": str,
                "prediction": {
                    "summary": str,
                    "summary_short": Optional[str]
                }
            }
        gold_data: Dict mapping episode_id to gold summary:

```text

            {
                "ep01": {
                    "summary": "...",
                    "summary_short": "..."
                }
            }

```json

    Returns:
        Metrics dictionary:
        {
            "global": {
                "rouge1_f": float,
                "rougeL_f": float,
                "avg_compression": float
            },
            "episodes": {
                "ep01": {...},
                "ep02": {...}
            }
        }
    """

```python

    ...

# CLI entry point (preserved for backward compatibility)

def main():

```text

    # Parse CLI args

```
```

```python

# scripts/experiments/utils.py

from typing import List, Dict, Any
from pathlib import Path
import json

def load_predictions_from_jsonl(predictions_file: Path) -> List[Dict[str, Any]]:
    """Load predictions from JSONL file."""
    predictions = []
    with open(predictions_file, "r") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    return predictions

def load_predictions_from_metadata(metadata_dir: Path) -> List[Dict[str, Any]]:
    """Load predictions from production metadata files (for comparison)."""

```python

    # Load from existing metadata.json files

```python

# podcast_scraper/workflow.py

from .providers.summarization.factory import create_summarization_provider
from .providers.speaker_detection.factory import create_speaker_detector
from .providers.transcription.factory import create_transcription_provider

def run_pipeline(cfg: config.Config) -> Tuple[int, str]:
    """Run production pipeline."""

    # Create providers using factory

    summarization_provider = None
    if cfg.generate_summaries and not cfg.dry_run:
        summarization_provider = create_summarization_provider(cfg)

    speaker_detector = None
    if cfg.auto_speakers:
        speaker_detector = create_speaker_detector(cfg)

    transcription_provider = None
    if cfg.transcribe_missing:

```text

        transcription_provider = create_transcription_provider(cfg)

```

    # ...

```python

# scripts/experiments/config_mapper.py

from typing import Dict, Any
from podcast_scraper import config

def map_experiment_to_config(experiment_config: Dict[str, Any]) -> config.Config:
    """Map experiment config to Config object.

    Example:
        experiment_config = {
            "models": {
                "summarizer": {"type": "openai", "name": "gpt-4o-mini"}
            },
            "params": {"max_length": 500, "temperature": 0.7}
        }

        Returns Config object with:
        - summary_provider = "openai"
        - openai_summarization_model = "gpt-4o-mini"
        - summary_max_length = 500
        - (temperature mapped to OpenAI-specific config if supported)
    """

    config_dict = {}

```text

    # Map models

```
            config_dict["summary_provider"] = "openai"
            config_dict["openai_summarization_model"] = model_config["name"]
        elif model_config["type"] == "hf_local":
            config_dict["summary_provider"] = "transformers"
            config_dict["summary_model"] = model_config["name"]

```

            if "reduce" in experiment_config.get("models", {}):
                reduce_config = experiment_config["models"]["reduce"]
                if reduce_config["type"] == "hf_local":
                    config_dict["summary_reduce_model"] = reduce_config["name"]

```
        if "max_length" in params:
            config_dict["summary_max_length"] = params["max_length"]
        if "min_length" in params:
            config_dict["summary_min_length"] = params["min_length"]
        if "chunk_size" in params:
            config_dict["summary_chunk_size"] = params["chunk_size"]
        if "word_chunk_size" in params:
            config_dict["summary_word_chunk_size"] = params["word_chunk_size"]
        if "word_overlap" in params:
            config_dict["summary_word_overlap"] = params["word_overlap"]
        if "device" in params:
            config_dict["summary_device"] = params["device"]

```

        # Note: Some OpenAI params (temperature, etc.) may need to be passed

```python

    # Map prompts (using prompt_store from RFC-017)

```

        prompts = experiment_config["prompts"]

```

            config_dict["summary_system_prompt"] = prompts["system"]
        if "user" in prompts:
            config_dict["summary_user_prompt"] = prompts["user"]
        if "params" in prompts:
            config_dict["summary_prompt_params"] = prompts["params"]

```

- Versioned prompt files (`.j2` templates)
- Jinja2 parameterization
- Caching and tracking
- Provider-specific usage (prompts loaded internally by providers)

## Migration Strategy

### Phase 1: Extract Protocols (Non-Breaking)

1. Create `providers/` package structure
2. Define protocols/interfaces
3. Create factory functions
4. **No changes to existing code yet**

### Phase 2: Refactor Current Implementations (Backward Compatible)

1. Move `summarizer.py` → `providers/summarization/local.py`
2. Create `LocalSummarizationProvider` class
3. Update `summarizer.py` to import and re-export (backward compatibility)
4. Update `metadata.py` to optionally accept provider
5. Update `workflow.py` to use factory, but fall back to old code

### Phase 3: Extract Preprocessing (Non-Breaking)

1. Create `preprocessing.py`
2. Move cleaning functions from `summarizer.py`
3. Update imports gradually
4. Keep old functions in `summarizer.py` as wrappers initially

### Phase 4: Standardize Evaluation (Non-Breaking)

1. Refactor `eval_summaries.py` to have core function
2. Keep CLI entry point unchanged
3. Create helper functions for loading predictions

### Phase 5: Create Experiment Backends

1. Create `scripts/experiments/backends/`
2. Implement backend adapters
3. Test with experiment pipeline

### Phase 6: Clean Up (Breaking Changes - Major Version)

1. Remove deprecated wrappers
2. Update all imports
3. Remove old code paths

## Benefits

**Note**: For product benefits, see `docs/prd/PRD-007-ai-experiment-pipeline.md`. This section focuses on technical benefits.

1. **No Code Duplication**: Experiment pipeline reuses production providers
2. **Independent Execution**: Experiments don't require full pipeline
3. **Testability**: Each provider can be tested independently
4. **Extensibility**: Easy to add new providers
5. **Backward Compatibility**: Existing code continues to work during migration
6. **Clear Separation**: Experiment pipeline is separate from production pipeline

## Risks and Mitigation

**Risk**: Breaking existing production workflow

**Mitigation**:

- Gradual migration with backward compatibility wrappers
- Extensive testing at each phase
- Feature flags if needed

**Risk**: Increased complexity

**Mitigation**:

- Clear documentation
- Well-defined interfaces
- Examples and tests

## Success Criteria

**Note**: For product success criteria, see `docs/prd/PRD-007-ai-experiment-pipeline.md`. This section focuses on technical success criteria.

- ✅ Experiment pipeline can run independently
- ✅ Production workflow continues to work unchanged
- ✅ No code duplication between experiment and production
- ✅ Easy to add new providers
- ✅ Evaluation scripts work with both experiment and production outputs
- ✅ Clear separation of concerns
- ✅ Provider interfaces are well-defined and testable

---

## 🚀 Evolution & Improvements (2026-01-10 Update)

### Critical Enhancements for Phase 2 Implementation

Based on lessons learned from manual experiments and comparison issues, the following improvements are **critical** for Phase 2 (factory enhancements).

---

### 1. Strengthen Phase 2: Stable Typed Contract for Experiment Params

**Problem:** Current plan says "accept experiment-style params dict" but doesn't specify the contract. This leads to fuzzy parameter resolution and incomparable experiments.

**Solution:** Define typed `ProviderParams` models per task, ensuring consistent parameter handling.

#### Typed Parameter Models

```python
# src/podcast_scraper/providers/params.py

from pydantic import BaseModel, Field
from typing import Optional, Literal

class SummarizationParams(BaseModel):
    """Typed params for summarization providers."""

    # Chunking
    chunk_size: int = Field(2048, description="Max tokens per chunk")
    word_chunk_size: int = Field(900, description="Max words per chunk")
    word_overlap: int = Field(150, description="Word overlap between chunks")

    # Generation
    max_length: int = Field(160, description="Max summary length")
    min_length: int = Field(60, description="Min summary length")
    temperature: Optional[float] = Field(None, description="Sampling temperature")
    top_p: Optional[float] = Field(None, description="Nucleus sampling")

    # Hardware
    device: Literal["cpu", "cuda", "mps", "auto"] = Field("auto", description="Device for inference")

    # Model-specific
    use_16bit: bool = Field(False, description="Use 16-bit precision")
    batch_size: int = Field(1, description="Batch size for MAP phase")

class TranscriptionParams(BaseModel):
    """Typed params for transcription providers."""

    model_size: str = Field("large-v3", description="Whisper model size")
    device: Literal["cpu", "cuda", "mps", "auto"] = Field("auto")
    language: Optional[str] = Field(None, description="Force language (None = auto-detect)")
    beam_size: int = Field(5, description="Beam search width")
    temperature: float = Field(0.0, description="Sampling temperature")

class SpeakerDetectionParams(BaseModel):
    """Typed params for speaker detection providers."""

    spacy_model: str = Field("en_core_web_lg", description="spaCy NER model")
    min_confidence: float = Field(0.7, description="Min confidence for NER")
    context_window: int = Field(100, description="Context chars around mention")
```

#### Factory Integration

```python
# src/podcast_scraper/providers/summarization/factory.py

def create_summarization_provider(
    config: Config,
    params: Optional[SummarizationParams] = None,
) -> SummarizationProvider:
    """Create summarization provider with typed params."""

    # Use params if provided, otherwise create from config
    if params is None:
        params = SummarizationParams(
            chunk_size=config.chunk_size,
            max_length=config.max_length,
            device=config.device,
            # ... map all config fields to params
        )

    # Log resolved params
    logger.info(f"Summarization params: {params.dict()}")

    # Create provider with typed params
    if config.summarization_provider == "transformers":
        return LocalSummarizationProvider(config, params)
    elif config.summarization_provider == "openai":
        return OpenAISummarizationProvider(config, params)
    else:
        raise ValueError(f"Unknown provider: {config.summarization_provider}")
```

#### Experiment Config Usage

```yaml
# experiments/summarization_experiment_v1.yaml

models:
  summarizer:
    type: "transformers"
    name: "facebook/bart-large-cnn"

# Typed params (validated against SummarizationParams)
params:
  chunk_size: 2048
  word_chunk_size: 900
  word_overlap: 150
  max_length: 160
  min_length: 60
  device: "mps"
  use_16bit: false
  batch_size: 1
```

**Why:** Explicit types prevent parameter confusion. Pydantic validation catches errors early. Logged params enable exact reproduction.

---

### 2. Add Provider Fingerprinting to Outputs

**Problem:** "Why did results change?" is impossible to answer without knowing exact provider state (model version, device, precision, dependencies).

**Solution:** Record comprehensive provider fingerprint in every output.

#### Provider Fingerprint

```python
# src/podcast_scraper/providers/base.py

from dataclasses import dataclass
import torch
import transformers
from typing import Dict, Any

@dataclass
class ProviderFingerprint:
    """Complete provider state for reproducibility."""

    # Provider info
    provider_type: str  # "transformers", "openai", "hybrid_ml"
    provider_version: str  # Package version

    # Model info
    model_names: Dict[str, str]  # {"map": "facebook/bart-large-cnn", "reduce": "allenai/led-base-16384"}
    model_versions: Dict[str, str]  # Model file hashes or API versions

    # Hardware
    device: str  # "cuda:0", "mps", "cpu"
    device_name: Optional[str]  # GPU name if available
    precision: str  # "fp32", "fp16", "int8"

    # Environment
    git_commit: str  # Current git commit hash
    git_branch: str  # Current git branch
    git_dirty: bool  # Uncommitted changes

    # Dependencies
    python_version: str
    torch_version: Optional[str]
    transformers_version: Optional[str]
    cuda_version: Optional[str]

    # Params (resolved final values)
    params: Dict[str, Any]

    # Preprocessing
    preprocessing_profile: str  # "cleaning_v3"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON storage."""
        return asdict(self)

def capture_provider_fingerprint(
    provider_type: str,
    model_names: Dict[str, str],
    params: BaseModel,
    preprocessing_profile: str,
) -> ProviderFingerprint:
    """Capture complete provider state."""

    return ProviderFingerprint(
        provider_type=provider_type,
        provider_version=get_package_version("podcast_scraper"),
        model_names=model_names,
        model_versions=get_model_versions(model_names),
        device=str(params.device),
        device_name=get_device_name(params.device),
        precision=get_precision(params),
        git_commit=get_git_commit(),
        git_branch=get_git_branch(),
        git_dirty=has_uncommitted_changes(),
        python_version=sys.version,
        torch_version=torch.__version__ if torch else None,
        transformers_version=transformers.__version__ if transformers else None,
        cuda_version=torch.version.cuda if torch and torch.cuda.is_available() else None,
        params=params.dict(),
        preprocessing_profile=preprocessing_profile,
    )
```

#### Output Metadata

```json
{
  "run_id": "summarization_bart_led_v1_20260110_143022",
  "fingerprint": {
    "provider_type": "transformers",
    "provider_version": "2.4.0",
    "model_names": {
      "map": "facebook/bart-large-cnn",
      "reduce": "allenai/led-base-16384"
    },
    "model_versions": {
      "map": "abc123...",
      "reduce": "def456..."
    },
    "device": "mps",
    "device_name": "Apple M1 Max",
    "precision": "fp32",
    "git_commit": "abc123def456",
    "git_branch": "feat/experiment-pipeline",
    "git_dirty": false,
    "python_version": "3.10.12",
    "torch_version": "2.1.0",
    "transformers_version": "4.35.2",
    "cuda_version": null,
    "params": {
      "chunk_size": 2048,
      "max_length": 160,
      "device": "mps"
    },
    "preprocessing_profile": "cleaning_v3"
  }
}
```

**Why:** Complete reproducibility. When results change, you know exactly what changed (model version? CUDA? preprocessing?).

---

### 3. Preprocessing Should Be Explicitly Provider-Agnostic and Versioned

**Problem:** Preprocessing changes are as impactful as model changes, but aren't tracked or versioned.

**Solution:** Make preprocessing profiles first-class, versioned, and tracked in metadata.

#### Preprocessing Profiles

```python
# src/podcast_scraper/preprocessing/profiles.py

from dataclasses import dataclass
from typing import Callable

@dataclass
class PreprocessingProfile:
    """Versioned preprocessing configuration."""

    profile_id: str  # "cleaning_v3"
    version: str  # "3.0"
    description: str

    # Cleaning steps
    remove_timestamps: bool = True
    normalize_speakers: bool = True
    collapse_blank_lines: bool = True
    remove_fillers: bool = False
    remove_sponsors: bool = True
    remove_outros: bool = True

    # Custom transformations
    custom_transforms: list[Callable[[str], str]] = None

    def apply(self, text: str) -> str:
        """Apply preprocessing pipeline."""
        # Apply transformations in order
        ...

# Registry of profiles
PREPROCESSING_PROFILES = {
    "cleaning_v1": PreprocessingProfile(
        profile_id="cleaning_v1",
        version="1.0",
        description="Basic cleaning (timestamps, speakers, whitespace)",
        remove_timestamps=True,
        normalize_speakers=True,
        collapse_blank_lines=True,
        remove_fillers=False,
        remove_sponsors=False,
        remove_outros=False,
    ),
    "cleaning_v2": PreprocessingProfile(
        profile_id="cleaning_v2",
        version="2.0",
        description="Basic + sponsor removal",
        remove_timestamps=True,
        normalize_speakers=True,
        collapse_blank_lines=True,
        remove_fillers=False,
        remove_sponsors=True,
        remove_outros=False,
    ),
    "cleaning_v3": PreprocessingProfile(
        profile_id="cleaning_v3",
        version="3.0",
        description="Full cleaning (default)",
        remove_timestamps=True,
        normalize_speakers=True,
        collapse_blank_lines=True,
        remove_fillers=False,
        remove_sponsors=True,
        remove_outros=True,
    ),
}

def get_preprocessing_profile(profile_id: str) -> PreprocessingProfile:
    """Get preprocessing profile by ID."""
    if profile_id not in PREPROCESSING_PROFILES:
        raise ValueError(f"Unknown preprocessing profile: {profile_id}")
    return PREPROCESSING_PROFILES[profile_id]
```

#### Integration with Experiments

```yaml
# experiments/summarization_experiment_v1.yaml

preprocessing:
  profile_id: "cleaning_v3"  # Explicit, versioned

models:
  summarizer:
    type: "transformers"
```

#### Metadata Recording

```json
{
  "preprocessing": {
    "profile_id": "cleaning_v3",
    "version": "3.0",
    "steps": {
      "remove_timestamps": true,
      "remove_sponsors": true,
      "remove_outros": true
    }
  }
}
```

**Why:** Preprocessing changes can silently improve ROUGE while deleting signal. Versioned profiles make changes explicit and traceable.

---

### 4. Baseline Reference in Provider Metadata

**Problem:** Providers don't know which baseline they're being compared against, making "delta from baseline" reporting impossible.

**Solution:** Pass `baseline_id` to providers and include in fingerprint.

```python
def create_summarization_provider(
    config: Config,
    params: Optional[SummarizationParams] = None,
    baseline_id: Optional[str] = None,  # NEW
) -> SummarizationProvider:
    """Create provider with baseline reference."""

    provider = LocalSummarizationProvider(config, params)
    provider.baseline_id = baseline_id
    return provider
```

**Why:** Enables automatic "vs baseline" reporting without external coordination.

---

## Updated Phase 2 Deliverables

Based on these improvements, Phase 2 should deliver:

### Core Deliverables

1. ✅ Typed `ProviderParams` models (Summarization, Transcription, SpeakerDetection)
2. ✅ Factory functions accept typed params
3. ✅ Provider fingerprinting (model versions, device, git commit, dependencies)
4. ✅ Preprocessing profiles (versioned, registered, tracked)
5. ✅ Baseline ID integration

### Code Changes

- `src/podcast_scraper/providers/params.py` - Typed parameter models
- `src/podcast_scraper/providers/base.py` - ProviderFingerprint
- `src/podcast_scraper/preprocessing/profiles.py` - Preprocessing profiles
- `src/podcast_scraper/providers/*/factory.py` - Updated factory signatures

### Metadata Enhancements

Every provider output includes:
- ✅ Complete fingerprint (models, device, precision, git state)
- ✅ Resolved params (actual values used)
- ✅ Preprocessing profile used
- ✅ Baseline ID reference (if applicable)

**Timeline:** 1 week (instead of 3-5 days) due to added rigor.

---

## Related Documents

- `docs/prd/PRD-007-ai-experiment-pipeline.md`: Product requirements, use cases, and functional specifications
- `docs/rfc/RFC-015-ai-experiment-pipeline.md`: Technical design and implementation details (updated with baseline-first concepts)
- `docs/rfc/RFC-041-podcast-ml-benchmarking-framework.md`: Benchmarking framework (complementary, shares baseline concept)
- `docs/rfc/RFC-013-openai-provider-implementation.md`: OpenAI provider design (shared refactoring plan)
- `docs/rfc/RFC-017-prompt-management.md`: Prompt management and loading implementation
- `docs/prd/PRD-006-openai-provider-integration.md`: Product requirements for OpenAI integration

## Alignment with RFC-013

**Key Point**: The provider pattern refactoring described in this RFC is **the same refactoring planned in RFC-013** for OpenAI provider integration. This ensures:

1. **No Duplication**: We don't refactor twice - the provider pattern is implemented once
2. **Shared Benefits**: Both OpenAI integration and AI experiments benefit from the same modularization
3. **Unified Timeline**: Implementation phases align, reducing overall work
4. **Consistent Architecture**: Same patterns and interfaces for both use cases

**Differences**: This RFC adds:

- Experiment backend adapters (Phase 4-5)
- Standardized evaluation interfaces (Phase 4)
- Focus on experiment pipeline independence

These additions build on top of the planned provider pattern refactoring, not replacing it.

````
