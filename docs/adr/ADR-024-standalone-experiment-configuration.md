# ADR-024: Standalone Experiment Configuration

- **Status**: Accepted
- **Date**: 2026-01-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-015](../rfc/RFC-015-ai-experiment-pipeline.md)
- **Related PRDs**: [PRD-007](../prd/PRD-007-ai-quality-experiment-platform.md)

## Context & Problem Statement

Iterating on AI models, prompts, and parameters (like chunk size or temperature) often required code changes in the core workflow or manual script overrides. This made it difficult to compare different approaches systematically or to track what specific settings produced a particular set of results.

## Decision

We define **Standalone Experiment Configurations** as YAML files in `data/eval/configs/` directory.

- Each YAML file defines a specific research goal (e.g., `summarization_gpt4_v2.yaml`).
- The config includes model identifiers, typed parameters, prompt template references, and dataset links.
- The `run_experiment.py` script consumes these files to generate predictions without modifying production code.

## Rationale

- **Decoupling**: Separates research and "tuning" parameters from the production application logic.
- **Reproducibility**: The configuration file serves as a complete record of the experiment's inputs.
- **Velocity**: Prompts and parameters can be swapped and tested in seconds without git commits to the core library.

## Alternatives Considered

1. **Code-Based Experiments**: Rejected as it clutters the library with research-only branches and logic.
2. **CLI Flag Overrides**: Rejected as it makes it impossible to track complex experiments with 10+ varying parameters.

## Consequences

- **Positive**: Clean separation of concerns; high-velocity research; immutable experiment records.
- **Negative**: Requires maintaining a separate directory of configuration assets.

## References

- [RFC-015: AI Experiment Pipeline](../rfc/RFC-015-ai-experiment-pipeline.md)
