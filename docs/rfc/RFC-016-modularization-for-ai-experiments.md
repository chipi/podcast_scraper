# RFC-016: Modularization for AI Experiment Pipeline

- **Status**: Draft
- **Authors**:
- **Stakeholders**: Maintainers, developers implementing AI experiment pipeline, developers maintaining core workflow
- **Related RFCs**: `docs/rfc/RFC-015-ai-experiment-pipeline.md`, `docs/rfc/RFC-017-prompt-management.md`, `docs/rfc/RFC-021-modularization-refactoring-plan.md` (historical reference)
- **Related Issues**: (to be created)

## Abstract

This RFC identifies the refactoring and modularization needed to support the AI experiment pipeline (RFC-015) while maintaining the existing production workflow. **This refactoring aligns with and builds upon the modularization already planned in RFC-013 (OpenAI Provider Implementation)**. The goal is to create clean abstractions that allow the experiment pipeline to run independently without duplicating code or interfering with the main pipeline.

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

```
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

```
    # Current remove_sponsor_blocks() implementation

```
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

- ✅ Local providers (transformers, Whisper) don't use prompts
- ✅ Prompt management doesn't affect protocol compliance
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

        cfg = self._create_config_from_experiment(experiment_config)

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
```

        return self.provider.summarize(
            text=transcript,
            episode_title=episode_title,
            episode_description=episode_description,
            params=merged_params,
        )

```python
    def _create_config_from_experiment(self, exp_config: Dict[str, Any]) -> config.Config:
        """Convert experiment config to Config object."""
```

        # Map experiment config to Config fields

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

```
    # Current evaluation logic, but with standardized input/output

```python

    ...

# CLI entry point (preserved for backward compatibility)

def main():

```text

    # Parse CLI args

```

    # Load predictions and gold data

```
    # Call evaluate_summaries()

```

    # Print results

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

```

```text
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
    if "summarizer" in experiment_config.get("models", {}):
        model_config = experiment_config["models"]["summarizer"]
        if model_config["type"] == "openai":
            config_dict["summary_provider"] = "openai"
            config_dict["openai_summarization_model"] = model_config["name"]
        elif model_config["type"] == "hf_local":
            config_dict["summary_provider"] = "transformers"
            config_dict["summary_model"] = model_config["name"]

```
```
            if "reduce" in experiment_config.get("models", {}):
                reduce_config = experiment_config["models"]["reduce"]
                if reduce_config["type"] == "hf_local":
                    config_dict["summary_reduce_model"] = reduce_config["name"]

```
```
    if "params" in experiment_config:
        params = experiment_config["params"]
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
```
        # Note: Some OpenAI params (temperature, etc.) may need to be passed

```
```python

    # Map prompts (using prompt_store from RFC-017)

```

        prompts = experiment_config["prompts"]

```
        # Map prompt names to config fields

```

            config_dict["summary_system_prompt"] = prompts["system"]
        if "user" in prompts:
            config_dict["summary_user_prompt"] = prompts["user"]
        if "params" in prompts:
            config_dict["summary_prompt_params"] = prompts["params"]

```
    return config.Config(**config_dict)

```go

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

## Related Documents

- `docs/prd/PRD-007-ai-experiment-pipeline.md`: Product requirements, use cases, and functional specifications
- `docs/rfc/RFC-015-ai-experiment-pipeline.md`: Technical design and implementation details
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
