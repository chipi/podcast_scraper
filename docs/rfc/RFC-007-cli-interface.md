# RFC-007: CLI Interface & Validation

- **Status**: Completed
- **Authors**: GPT-5 Codex (initial documentation)
- **Stakeholders**: Maintainers, operators, documentation writers
- **Related PRD**: `docs/prd/PRD-003-user-interface-config.md`

## Abstract

Specify the structure and behavior of the command-line interface, including argument parsing, validation, configuration merging, and integration with the pipeline.

## Problem Statement

The CLI is the primary user entry point. It must expose all critical functionality while preventing invalid runs through proactive validation. Additionally, it needs to support configuration files without surprising precedence rules.

## Constraints & Assumptions

- CLI entry is `python -m podcast_scraper.cli` or `podcast_scraper.cli.main()` from Python.
- Arguments are parsed using `argparse`; we must remain compatible with Python 3.10+ standard library.
- Validation should surface actionable errors without stack traces (exit code 1).

## Design & Implementation

1. **Argument parsing**
   - `parse_args` defines all flags documented in README (RSS URL, output, max episodes, Whisper flags, etc.).
   - Supports `--config` for JSON/YAML files; merges validated values into parser defaults.
   - `--version` prints version string and exits.
2. **Validation**
   - `validate_args` enforces URL schemes, numeric ranges, Whisper model choices, speaker name counts, and output directory validity.
   - Raises `ValueError` with aggregated error messages for user-friendly output.
3. **Config merging**
   - Config files loaded via `config.load_config_file` then parsed through `config.Config` for schema enforcement.
   - CLI arguments override config defaults; unspecified CLI flags inherit config values.
4. **Config construction**
   - `_build_config` transforms CLI namespace into `config.Config` (populating derived fields like output dir and speaker list).
5. **Integration hooks**
   - `main` accepts injectable `apply_log_level_fn`, `run_pipeline_fn`, and `logger` for testing.
   - Registers toolkit progress factory (`progress.set_progress_factory`) with CLI-specific `tqdm` wrapper.
6. **Exit semantics**
   - Validation or configuration errors return exit code 1 without stack traces.
   - Pipeline exceptions are caught, logged, and return exit code 1.

## Key Decisions

- **Two-phase validation** (argparse + Pydantic) catches both syntactic and semantic errors before running pipeline.
- **Config precedence** ensures reproducible defaults while allowing on-the-fly overrides.
- **Injectable dependencies** improve testability (e.g., verifying CLI surfaces errors correctly).

## Alternatives Considered

- **Click/Typer frameworks**: Rejected to minimize dependencies and maintain explicit control over parsing/validation flow.
- **Silent failure on validation errors**: Rejected; clear logging and exit status are vital for automation.

## Testing Strategy

- CLI tests in `tests/test_podcast_scraper.py` cover success cases, invalid arguments, config loading precedence, and version flag behavior.
- Unit tests simulate argument lists to hit edge cases (e.g., invalid speaker counts, unknown config keys).

## Rollout & Monitoring

- Help text (`--help`) reviewed for accuracy each release.
- Version string maintained in `cli.__version__` and exported via `__init__` for tooling.

## References

- Source: `podcast_scraper/cli.py`
- Config schema: `docs/rfc/RFC-008-config-model.md`
- Progress integration: `docs/rfc/RFC-009-progress-integration.md`
