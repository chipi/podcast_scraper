# RFC-008: Configuration Model & Validation

- **Status**: Accepted
- **Authors**: GPT-5 Codex (initial documentation)
- **Stakeholders**: Maintainers, API consumers, automation engineers
- **Related PRDs**: `docs/prd/PRD-001-transcript-pipeline.md`, `docs/prd/PRD-002-whisper-fallback.md`, `docs/prd/PRD-003-user-interface-config.md`

## Abstract
Outline the design of the immutable Pydantic `Config` model, including field normalization, validation rules, and serialization behavior that underpins both CLI and Python API usage.

## Problem Statement
Multiple modules need consistent configuration data with guarantees around types, ranges, and normalization (e.g., trimmed strings, positive integers). A central schema ensures downstream logic can assume invariants without duplicating validation.

## Constraints & Assumptions
- Configuration should be immutable post-construction to prevent accidental runtime mutation.
- Validation must handle inputs from CLI strings, config files (JSON/YAML), and direct Python instantiation.
- The model should forbid unknown fields to catch typos early.

## Design & Implementation
1. **Model definition**
   - `Config` inherits from `pydantic.BaseModel` with `frozen=True`, `populate_by_name=True`, and `extra="forbid"`.
   - Field aliases align with CLI flags (e.g., `rss` -> `rss_url`).
2. **Default values**
   - Defaults centralized in `config.py` (timeout, worker count, log level, etc.).
   - `DEFAULT_WORKERS` derived from CPU count bounded between 1 and 8.
3. **Field validators**
   - Strip whitespace from string fields, enforce positive numbers, ensure valid Whisper models, etc.
   - Normalize speaker names into lists, convert `prefer_type` to list of strings, and coerce integers from strings when needed.
4. **Output directory derivation**
   - Stored `output_dir` is already normalized; CLI calculates it via `filesystem.derive_output_dir` before model instantiation.
5. **Serialization**
   - `Config.model_dump` with `exclude_none=True` and `by_alias=True` used when exporting config data back to CLI defaults.
6. **Integration**
   - CLI constructs `Config` for pipeline; Python API consumers can instantiate directly with keyword arguments.
   - `load_config_file` (JSON/YAML) returns a dict ready for `Config.model_validate`.

## Key Decisions
- **Frozen model** ensures modules treat configuration as read-only, promoting functional-style architecture.
- **Validators** centralize edge-case handling (e.g., negative delays) instead of scattering checks.
- **Alias usage** keeps CLI and internal naming aligned while allowing Pythonic field names in code.

## Alternatives Considered
- **Custom dataclasses**: Rejected; Pydantic provides superior validation and parsing out of the box.
- **Mutable configs**: Rejected to avoid accidental mutation and thread-safety issues.

## Testing Strategy
- Unit tests validate coercion logic, error messages, and alias handling.
- Integration tests confirm that CLI + config files properly instantiate `Config`.

## Rollout & Monitoring
- New configuration options require updates to `Config`, CLI parser, PRDs/RFCs, and README.
- Breaking changes (field renames) should bump minor version and update `__version__` references.

## References
- Source: `podcast_scraper/config.py`
- CLI usage: `docs/rfc/RFC-007-cli-interface.md`
- Filesystem validation: `docs/rfc/RFC-004-filesystem-layout.md`
