# RFC-017: Prompt Management and Loading

- **Status**: Completed
- **Authors**:
- **Stakeholders**: Maintainers, developers implementing AI features, researchers running experiments
- **Related PRDs**: `docs/prd/PRD-006-openai-provider-integration.md`
- **Related RFCs**: `docs/rfc/RFC-012-episode-summarization.md`, `docs/rfc/RFC-013-openai-provider-implementation.md`, `docs/rfc/RFC-015-ai-experiment-pipeline.md`
- **Related Issues**: (to be created)

## Abstract

Design and implement a lightweight, unified prompt management system that enables versioned, parameterized prompts for both production application use and AI experiment pipelines. This system treats prompts as first-class, versioned assets with optional templating, avoiding heavy frameworks while providing essential functionality: file-based organization, Jinja2 templating, caching, and tracking.

**Architecture Alignment:** This RFC aligns with RFC-016 (Modularization for AI Experiments) by treating prompts as a **provider-specific concern**. Prompts are used internally by providers (e.g., OpenAI providers) but are not part of the core protocol interface, maintaining provider autonomy and backward compatibility.

## Problem Statement

Currently, prompts are:

- **Embedded in code**: Hard-coded strings scattered throughout the codebase
- **Not versioned**: Changes to prompts require code changes, making it hard to track what prompt produced which results
- **Not reusable**: Same prompts duplicated across application code and experiment scripts
- **Not parameterized**: Manual string formatting for dynamic content
- **Hard to compare**: No easy way to compare prompt variants or track which prompt was used in production

**Use Cases:**

1. **Production Application**: Use prompts for OpenAI summarization, NER, and other AI features
2. **Experiment Pipeline**: Test different prompt variants without code changes
3. **Prompt Engineering**: Iterate on prompts by editing files, not code
4. **Reproducibility**: Always know which exact prompt (including version) produced results
5. **A/B Testing**: Compare prompt variants side-by-side

## Goals

1. **Unified Interface**: Same prompt loading mechanism for application and experiments
2. **File-Based**: Prompts stored as versioned files, not embedded in code
3. **Templating**: Optional parameterization via Jinja2 templates
4. **Lightweight**: Minimal dependencies (Jinja2, Pydantic) - no heavy frameworks
5. **Caching**: Efficient in-memory caching to avoid repeated disk I/O
6. **Tracking**: SHA256 hashes and metadata for reproducibility
7. **Versioning**: Explicit versioning via filenames (v1, v2, etc.)
8. **Type Safety**: Typed configs using Pydantic

## Design & Implementation

### 1. Prompt Directory Structure

Prompts are organized in a `prompts/` directory with task-specific subdirectories:

````text
prompts/
  summarization/
    system_v1.j2
    long_v1.j2
    long_v2_more_narrative.j2
    long_v2_focus_on_frameworks.j2
    short_v1.j2
  ner/
    system_ner_v1.j2
    guest_host_v1.j2
    guest_host_v2_strict_roles.j2
    entities_generic_v1.j2
```text
```jinja2
You are summarizing a podcast episode.

Write a detailed, narrative summary with a clear story arc.
Guidelines:

- Aim for {{ paragraphs_min }}–{{ paragraphs_max }} paragraphs.
- Focus on key decisions, arguments, and lessons.
- Ignore sponsorships, ads, and housekeeping.
- Do not use quotes or speaker names.
- Do not invent information not implied by the transcript.
```text

**Example: `prompts/summarization/system_v1.j2`**

```jinja2
You are an expert at creating concise, informative summaries of podcast episodes.
```text
```python
"""
Lightweight prompt management for LLM experiments and production use.

Features:
- File-based prompts (Jinja2 templates)
- Loading by logical name (e.g. "summarization/long_v1")
- In-memory caching to avoid repeated disk I/O
- Optional templating parameters via Jinja2
- SHA256 hashes for reproducible experiment metadata
"""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict

from jinja2 import Template

# Root directory where all your prompt templates live.

# Default: project_root/prompts/

# Can be overridden via environment variable PROMPT_DIR

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

class PromptNotFoundError(FileNotFoundError):

```text

    """Raised when a requested prompt template is not found on disk."""

```python
def set_prompt_dir(path: str | Path) -> None:

```text

    """Set the root directory for prompt templates.

```python
    Useful for testing or custom prompt locations.
    """
    global _PROMPT_DIR
    _PROMPT_DIR = Path(path).resolve()

def get_prompt_dir() -> Path:

```text

    """Get the current prompt directory."""
    return _PROMPT_DIR

```python
@lru_cache(maxsize=None)
def _load_template(name: str) -> Template:

```text

    """
    Load and cache a Jinja2 template by logical name.

```python

    Example:
        name="summarization/long_v1" -> prompts/summarization/long_v1.j2

```

    Args:
        name: Logical name without .j2 extension

```
    Raises:
        PromptNotFoundError: If template file doesn't exist
    """

```

    if name.endswith(".j2"):
        rel_path = Path(name)
    else:
        rel_path = Path(name + ".j2")

```
            f"Prompt template not found: {path}\n"
            f"  Searched in: {_PROMPT_DIR}\n"
            f"  Requested name: {name}"
        )

```
```python

def render_prompt(name: str, **params: Any) -> str:

```text

    """
    Render a prompt template with optional parameters.

```

```
```
    """
    tmpl = _load_template(name)
    return tmpl.render(**params).strip()

```python

def get_prompt_source(name: str) -> str:

```text

    """
    Return the raw template source text (without rendering).
    Useful for hashing / metadata.

```
```
    tmpl = _load_template(name)

```
```python

    # Fallback: reload from disk

```

        rel_path = Path(name + ".j2")
    path = _PROMPT_DIR / rel_path
    return path.read_text(encoding="utf-8")

```python

def hash_text(text: str) -> str:

```text

    """
    Return a SHA256 hex digest for arbitrary text.

```

```

    return sha256(text.encode("utf-8")).hexdigest()

```python

def get_prompt_metadata(
    name: str,
    params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:

```text

    """
    Return metadata describing a prompt configuration.

```
        - filename (relative path)
        - sha256 hash of template source
        - params used for rendering (if any)

```
```
    """
    if name.endswith(".j2"):
        rel_path = Path(name)
    else:
        rel_path = Path(name + ".j2")

```

        "name": name,
        "file": str(path.relative_to(_PROMPT_DIR)),
        "sha256": hash_text(source),
    }

```text

    if params:
        metadata["params"] = params

```python

def clear_cache() -> None:

```text

    """Clear the prompt template cache.

```python

"""
Experiment configuration models for LLM evaluation.

- Keeps prompts, models, params, and data paths in a single typed config.
- Designed to work with both local HF models and OpenAI backends.
- Uses Pydantic for validation and type safety.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, validator

# ----- Prompt configuration -----

class PromptConfig(BaseModel):

```text

    """
    Configuration for prompts used in an experiment.

```

        system: "summarization/system_v1"
        user:   "summarization/long_v2_more_narrative"
        params:
          paragraphs_min: 3
          paragraphs_max: 6
    """

```yaml
    )
    user: str = Field(
        description="Logical name for user prompt template.",
    )
    params: Dict[str, object] = Field(
        default_factory=dict,
        description="Template parameters to render into prompts.",
    )

# ----- Backend / model configuration -----

class HFBackendConfig(BaseModel):

```text

    """Config for local Hugging Face models (your existing BART/LED setup)."""

```
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

```text

    """Config for OpenAI models (summarization, NER, etc.)."""

```yaml

    type: Literal["openai"] = "openai"
    model: str = Field(
        description="OpenAI model name, e.g. 'gpt-4o-mini'.",
    )

BackendConfig = HFBackendConfig | OpenAIBackendConfig

# ----- Data configuration -----

class DataConfig(BaseModel):

```text

    """
    Where to find input data for this experiment.

```
        id_from: "parent_dir"   # or "stem"
    """

```python

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

```text

    """
    Task-specific parameters.

```

    """

```
    chunk_size: Optional[int] = None
    word_chunk_size: Optional[int] = None
    word_overlap: Optional[int] = None
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None

```text

    # Allow arbitrary extra keys for specific experiments

```python

    @validator("extra", pre=True, always=True)
    def collect_extra(cls, v, values):  # type: ignore[override]

```python

class ExperimentConfig(BaseModel):

```text

    """
    Full configuration for a single experiment run.

```

      backend:
        type: "openai"
        model: "gpt-4o-mini"

```
        user:   "summarization/long_v2_more_narrative"
        params:
          paragraphs_min: 3
          paragraphs_max: 6

```

```
    """

```

    data: DataConfig
    params: ExperimentParams = Field(default_factory=ExperimentParams)

```python

    @validator("id")
    def ensure_non_empty_id(cls, v):  # type: ignore[override]
        if not v.strip():
            raise ValueError("Experiment id must be non-empty")
        return v

```yaml

# ----- Loader helpers -----

def load_experiment_config(path: str | Path) -> ExperimentConfig:

```text

    """
    Load a YAML experiment config into a typed ExperimentConfig.

```

    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ExperimentConfig(**raw)

```python

def discover_input_files(data_cfg: DataConfig, base_dir: Path | None = None) -> List[Path]:

```text

    """
    Discover input files according to the experiment's data config.

```

```

    """
    if base_dir is None:
        base_dir = Path(".")
    paths = sorted(base_dir.glob(data_cfg.episodes_glob))
    return [p for p in paths if p.is_file()]

```python

def episode_id_from_path(path: Path, data_cfg: DataConfig) -> str:

```python

    """
    Convert a file path to an episode_id using the data config's id_from rule.

```

```python

    Returns:
        Episode ID string
    """
    if data_cfg.id_from == "stem":
        return path.stem

```
1. **Provider-Agnostic Core**: The core system (workflow, factories) doesn't know about prompts
2. **Provider-Specific Implementation**: Each provider that needs prompts handles them internally
3. **Optional Usage**: Providers that don't need prompts (e.g., local transformers) aren't forced to use them
4. **Protocol Compliance**: Prompt usage doesn't affect protocol compliance

**Example: Using prompts in OpenAI summarization provider**

```python

# podcast_scraper/summarization/openai_provider.py

from typing import Protocol, Optional, Dict, Any
from .. import config
from ..prompt_store import render_prompt, get_prompt_metadata
from .base import SummarizationProvider

class OpenAISummarizationProvider:
    """OpenAI provider for summarization (implements SummarizationProvider protocol)."""

    def __init__(self, cfg: config.Config):
        self.cfg = cfg
        self.client = self._setup_openai_client()

        # Prompts are loaded on-demand, cached automatically via prompt_store

    def initialize(self, cfg: config.Config) -> Optional[Any]:
        """Initialize OpenAI client (provider-specific resource)."""
        return self.client

```python

    def summarize(
        self,
        text: str,
        cfg: config.Config,
        resource: Any,  # OpenAI client
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Summarize text using OpenAI API with prompts from prompt_store.

```

        """

```python

        # Load prompts from prompt_store (provider-specific)

```

            system_prompt = render_prompt(
                cfg.summary_system_prompt,
                **cfg.summary_prompt_params,
            )

        user_prompt = render_prompt(
            cfg.summary_user_prompt or "summarization/long_v1",
            transcript=text,
            title=cfg.episode_title or "",
            paragraphs_min=(min_length or cfg.summary_min_length) // 100,
            paragraphs_max=(max_length or cfg.summary_max_length) // 100,
            **cfg.summary_prompt_params,
        )

```text

        # Call OpenAI API (provider-specific implementation)

```json

        response = resource.chat.completions.create(
            model=cfg.summary_model,
            messages=[
                {"role": "system", "content": system_prompt} if system_prompt else None,
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_length or cfg.summary_max_length,
            temperature=cfg.summary_temperature,
        )

```text

        return {
            "summary": response.choices[0].message.content,
            "metadata": {
                "model": cfg.summary_model,
                "prompts": {
                    "system": cfg.summary_system_prompt,
                    "user": cfg.summary_user_prompt,
                }
            }
        }

```python

# podcast_scraper/speaker_detectors/openai_detector.py

from typing import Protocol, Set, List, Tuple, Optional, Dict, Any
from .. import config
from ..prompt_store import render_prompt
from .base import SpeakerDetector

class OpenAISpeakerDetector:
    """OpenAI provider for speaker detection (implements SpeakerDetector protocol)."""

    def __init__(self, cfg: config.Config):
        self.cfg = cfg
        self.client = self._setup_openai_client()

    def detect_speakers(
        self,
        episode_title: str,
        episode_description: Optional[str],
        known_hosts: Set[str],
    ) -> Tuple[List[str], Set[str], bool]:

```python

        """Detect speakers using OpenAI API with prompts from prompt_store.

```

        """
        user_prompt = render_prompt(
            self.cfg.ner_user_prompt or "ner/guest_host_v1",
            episode_title=episode_title,
            episode_description=episode_description or "",
            known_hosts=", ".join(known_hosts) if known_hosts else "",
            **self.cfg.ner_prompt_params,
        )

```text

        # Call OpenAI API (provider-specific implementation)

```

        return (detected_speakers, detected_hosts, success)

```python

# podcast_scraper/summarization/transformers_provider.py

from typing import Protocol, Optional, Dict, Any
from .. import config
from .base import SummarizationProvider

class TransformersSummarizationProvider:
    """Local HuggingFace transformers provider (implements SummarizationProvider protocol)."""

    def summarize(
        self,
        text: str,
        cfg: config.Config,
        resource: Any,  # Local model
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Summarize text using local transformers model.

        This method implements the SummarizationProvider protocol.
        Local models don't use prompts - they use model-specific tokenization.
        """

```text

        # No prompts needed - local models work differently

```

```

                "model": cfg.summary_model,
            }
        }

```python

- ✅ **Protocol Compliance**: All providers implement the same protocol, regardless of prompt usage
- ✅ **Provider Autonomy**: Each provider decides how to use (or not use) prompts
- ✅ **No Core Dependencies**: The workflow/factory code doesn't import `prompt_store`
- ✅ **Backward Compatible**: Existing providers (transformers, Whisper) continue working without prompts

## 5. Integration with Experiment Pipeline

**File: `scripts/run_experiment.py`**

```python

"""
Run a single AI experiment from a YAML config file.
"""

from pathlib import Path
from typing import Any, Dict

from podcast_scraper.experiment_config import (
    ExperimentConfig,
    load_experiment_config,
    discover_input_files,
    episode_id_from_path,
)
from podcast_scraper.prompt_store import (
    render_prompt,
    get_prompt_metadata,
)

def run_experiment(cfg_path: str | Path) -> Dict[str, Any]:

```python

    """
    Run an experiment from a config file.

```

    Returns:
        Dictionary with experiment results and metadata
    """
    cfg = load_experiment_config(cfg_path)

```
        system_prompt = render_prompt(
            cfg.prompts.system,
            **cfg.prompts.params,
        )

    user_prompt = render_prompt(
        cfg.prompts.user,
        **cfg.prompts.params,
    )

```text

    # Get prompt metadata for tracking

```
            if cfg.prompts.system
            else None
        ),
        "user": get_prompt_metadata(cfg.prompts.user, cfg.prompts.params),
    }

```text

    # Discover data files

```
```python

    if cfg.task == "summarization":
        from podcast_scraper.summarization import SummarizationProviderFactory
        provider = SummarizationProviderFactory.create(cfg)
        resource = provider.initialize(cfg) if provider else None
    elif cfg.task.startswith("ner_"):
        from podcast_scraper.speaker_detectors import SpeakerDetectorFactory
        provider = SpeakerDetectorFactory.create(cfg)
        resource = provider.initialize(cfg) if provider else None
    else:
        raise ValueError(f"Unknown task: {cfg.task}")

```

```text

        episode_id = episode_id_from_path(file_path, cfg.data)

```

```

                text=episode_data["transcript"],
                cfg=cfg,
                resource=resource,
                max_length=cfg.params.max_length,
                min_length=cfg.params.min_length,
            )
            prediction = prediction_dict["summary"]
        elif cfg.task.startswith("ner_") and provider:
            speakers, hosts, success = provider.detect_speakers(
                episode_title=episode_data.get("title", ""),
                episode_description=episode_data.get("description"),
                known_hosts=set(),
            )
            prediction = {"speakers": speakers, "hosts": hosts, "success": success}
        else:
            raise ValueError(f"Provider not available for task: {cfg.task}")

```
```text

    # Cleanup provider resources

```
```

        "backend": cfg.backend.dict(),
        "prompts": prompt_meta,
        "metrics": metrics,
        "params": cfg.params.dict(),
    }

```text

    # Save results

```
```python

class Config(BaseModel):

    # ... existing fields ...

    # Prompt configuration

    summary_system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt name for summarization (e.g. 'summarization/system_v1')",
    )
    summary_user_prompt: str = Field(
        default="summarization/long_v1",
        description="User prompt name for summarization",
    )
    summary_prompt_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Template parameters for summary prompts",
    )

    ner_system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt name for NER",
    )
    ner_user_prompt: str = Field(
        default="ner/guest_host_v1",
        description="User prompt name for NER",
    )
    ner_prompt_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Template parameters for NER prompts",
    )

```text

**Example: `results/summarization_openai_long_v2/metrics.json`**

```json

{
  "experiment_id": "summarization_openai_long_v2",
  "task": "summarization",
  "backend": {
    "type": "openai",
    "model": "gpt-4o-mini"
  },
  "prompts": {
    "system": {
      "name": "summarization/system_v1",
      "file": "summarization/system_v1.j2",
      "sha256": "abc123def456...",
      "params": {}
    },
    "user": {
      "name": "summarization/long_v2_more_narrative",
      "file": "summarization/long_v2_more_narrative.j2",
      "sha256": "789ghi012jkl...",
      "params": {
        "paragraphs_min": 3,
        "paragraphs_max": 6
      }
    }
  },
  "metrics": {
    "global": {
      "rouge1_f": 0.322,
      "rougeL_f": 0.142,
      "avg_compression": 39.7
    },
    "episodes": {
      "ep01": {...}
    }
  },
  "params": {
    "max_output_tokens": 900,
    "temperature": 0.7
  }
}

```yaml

- **Jinja2**: Templating engine (already used in many Python projects)
- **Pydantic**: Type validation and config parsing (already a dependency)
- **PyYAML**: YAML parsing (already a dependency)

No new heavy dependencies required.

## File Structure

```text

podcast_scraper/
  prompt_store.py          # Core prompt loading/rendering
  experiment_config.py     # Experiment config models
  prompts/                 # Prompt templates directory
    summarization/
      system_v1.j2
      long_v1.j2
      long_v2_more_narrative.j2
    ner/
      system_ner_v1.j2
      guest_host_v1.j2
scripts/
  run_experiment.py        # Experiment runner (uses prompt_store)
experiments/               # Experiment configs
  summarization_openai_long_v1.yaml
  summarization_openai_long_v2.yaml

```text

### 1. Provider-Specific Concern

- **Prompts are internal to providers**: Providers that need prompts (OpenAI) use `prompt_store` internally
- **Protocol-agnostic**: The core protocol interfaces (`SummarizationProvider`, `SpeakerDetector`) don't mention prompts
- **Optional usage**: Providers that don't need prompts (local transformers, Whisper) aren't forced to use them

### 2. Protocol-Based Design

- **Protocol compliance**: All providers implement the same protocol, regardless of prompt usage
- **Factory pattern**: Factories create providers based on config; prompts are passed via config
- **Provider autonomy**: Each provider decides how to use (or not use) prompts

### 3. Backward Compatibility

- **No breaking changes**: Existing providers continue working without prompts
- **Gradual adoption**: Prompts can be added incrementally to providers that need them
- **Config-driven**: Prompt selection via config, not code changes

### 4. Separation of Concerns

- **Preprocessing**: Provider-agnostic preprocessing happens before provider selection (see RFC-016)
- **Prompt management**: Provider-specific prompt loading happens inside providers
- **Core workflow**: Workflow code doesn't import `prompt_store` directly

## Benefits

1. **Unified Interface**: Same `prompt_store` module used everywhere
2. **No Code Changes**: Edit prompt files, not code, to change prompts
3. **Versioning**: Explicit versioning via filenames
4. **Reproducibility**: SHA256 hashes track exact prompt versions
5. **Parameterization**: Jinja2 templates enable dynamic content
6. **Performance**: LRU cache avoids repeated disk I/O
7. **Type Safety**: Pydantic ensures configs are valid
8. **Lightweight**: Minimal dependencies, no heavy frameworks
9. **Provider-Agnostic Core**: Core system doesn't depend on prompts
10. **Protocol Compliance**: Prompts don't affect protocol interfaces

## Migration Path

1. **Phase 1**: Create `prompt_store.py` and initial prompt files
2. **Phase 2**: Update OpenAI providers to use `prompt_store`
3. **Phase 3**: Create `experiment_config.py` for experiment pipeline
4. **Phase 4**: Migrate existing prompts from code to files
5. **Phase 5**: Update config model to support prompt names

## Testing Strategy

- **Unit Tests**: Test prompt loading, rendering, caching, hashing
- **Integration Tests**: Test with real prompt files and configs
- **Experiment Tests**: Test experiment pipeline with prompt_store
- **Application Tests**: Test providers using prompt_store

## Open Questions

1. Should prompts support includes/extends for shared components?
2. Should we support prompt validation (e.g., required parameters)?
3. Should we add a CLI tool for prompt management?

## Relationship to Modularization Plan

This RFC implements prompt management as part of the broader modularization effort described in `docs/rfc/RFC-021-modularization-refactoring-plan.md`. Key alignment points:

### Provider Pattern Integration

- **Protocol-Based**: Prompts don't appear in protocol definitions (`SummarizationProvider`, `SpeakerDetector`)
- **Factory Pattern**: Factories create providers; prompts are configured via `config.Config`
- **Provider Autonomy**: Each provider decides internally whether and how to use prompts

### Implementation Phases

Following the modularization plan's incremental approach:

1. **Phase 1**: Create `prompt_store.py` and initial prompt files (no breaking changes)
2. **Phase 2**: Update OpenAI providers to use `prompt_store` internally
3. **Phase 3**: Add prompt config fields to `config.py` (backward compatible defaults)
4. **Phase 4**: Update experiment pipeline to use prompt_store via providers

### Backward Compatibility

- Existing providers (transformers, Whisper) continue working without prompts
- New prompt fields in config have defaults matching current behavior
- No changes required to protocol interfaces or factory patterns

## References

- RFC-015: AI Experiment Pipeline
- RFC-013: OpenAI Provider Implementation
- RFC-016: Modularization for AI Experiments
- RFC-021: Modularization Refactoring Plan - Overall modularization strategy (historical reference)
- [Jinja2 Documentation](https://jinja.palletsprojects.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

````
