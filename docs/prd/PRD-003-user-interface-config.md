# PRD-003: User Interfaces & Configuration

- **Status**: ✅ Implemented (v2.0.0)
- **Related RFCs**: RFC-007, RFC-008, RFC-009

## Summary

Define how operators interact with the podcast scraper via CLI flags and configuration files. Ensure a consistent experience across modes while exposing progress feedback and logging controls.

## Background & Context

- Users frequently run the tool from terminals or automation scripts, requiring a predictable CLI surface.
- Many production runs rely on reusable configuration files for reproducibility.
- The CLI is also the public API showcase; Python consumers call into the same `Config` and `run_pipeline` primitives.

## Goals

- Provide a single configuration model (`Config`) that powers both CLI and Python integration.
- Ensure CLI validation guards against common mistakes before work begins.
- Allow reusable JSON/YAML config files that slot into automation pipelines.
- Offer progress and logging visibility tuned for terminal usage while remaining embeddable.

## Non-Goals

- Building a GUI or web interface.
- Secret management or credential storage (handled externally if needed).
- Rich analytics dashboards (out of scope).

## Personas

- **Operator Owen**: Runs the CLI manually, tweaking flags to experiment with output runs.
- **Automation Alex**: Integrates the scraper into nightly jobs using configuration files.
- **Integrator Iris**: Imports the Python API into another application and needs a stable programmatic interface.

## User Stories

- _As Operator Owen, I can run `python -m podcast_scraper.cli <rss_url>` with sensible defaults and see progress bars and status logs._
- _As Automation Alex, I can maintain a JSON/YAML config file checked into version control and use `--config` to load it._
- _As a user, I can request version info (`--version`) and set log level verbosity per run._
- _As Integrator Iris, I can call `podcast_scraper.Config` + `podcast_scraper.run_pipeline` directly in Python with the same semantics._
- _As any user, I can enable automatic speaker name detection (`--auto-speakers`) without manually specifying names for each episode (RFC-010)._
- _As any user, I can configure the podcast language (`--language`) to optimize both Whisper transcription and speaker name detection._
- _As any user, I can provide manual speaker names (`--speaker-names`) as fallback when automatic detection fails._

## Functional Requirements

- **FR1**: CLI must validate inputs (RSS URL, numeric ranges, Whisper model choices) and surface actionable error messages.
- **FR2**: CLI flags map to `Config` fields; precedence is CLI > config file defaults (with validation).
- **FR3**: Support both JSON and YAML configuration files loaded via `--config` with schema validation.
- **FR4**: Expose logging controls (`--log-level`) and default to INFO.
- **FR5**: Default progress reporter uses `tqdm`; expose abstraction (`progress.set_progress_factory`) to override in embedded contexts.
- **FR6**: Provide `--dry-run`, `--skip-existing`, `--clean-output`, `--workers`, and other operational flags documented in README.
- **FR7**: Ensure exit codes communicate success (0) vs. validation or runtime failures (1).
- **FR8**: Export Python API surface (`Config`, `load_config_file`, `run_pipeline`, `cli.main`) from `podcast_scraper.__init__`.
- **FR9**: Support `--language` flag (default `"en"`) that configures both Whisper transcription language and NER model selection (RFC-010).
- **FR10**: Support `--auto-speakers` flag (default `true`) to enable/disable automatic speaker name detection via NER (RFC-010).
- **FR11**: Support `--ner-model` flag for advanced users to override default spaCy model selection (RFC-010).
- **FR12**: Support `--cache-detected-hosts` flag (default `true`) to control host detection memoization (RFC-010).
- **FR13**: Maintain fallback chain: automatic detection > manual `--speaker-names` fallback (when detection fails) > default `["Host", "Guest"]`.

## Success Metrics

- CLI onboarding: a new user can run the default command with only an RSS URL and receive useful output/logging.
- Config file onboarding: loading an invalid config produces a clear validation error (no partial runs).
- Python API: integration tests confirm parity with CLI semantics.

## Dependencies

- Validation and configuration logic described in `docs/rfc/RFC-007-cli-interface.md` and `docs/rfc/RFC-008-config-model.md`.
- Progress abstraction detailed in `docs/rfc/RFC-009-progress-integration.md`.
- Automatic speaker name detection and language configuration in `docs/rfc/RFC-010-speaker-name-detection.md`.

## Release Checklist

- [ ] CLI help text audited and examples verified in README.
- [ ] Integration tests cover CLI happy path, invalid args, config file precedence, programmatic usage.
- [ ] Version string maintained in sync (`__version__`).

## Open Questions

- Should we support environment variable substitution in config files? Not currently planned.
- Do we need subcommands for future expansion (e.g., `inspect`, `clean`)? Monitor user feedback.

## RFC-010 Integration

This PRD integrates with RFC-010 (Automatic Speaker Name Detection) to provide new configuration options:

- **Language Configuration**: The `--language` flag (default `"en"`) controls both Whisper model selection and NER model selection. Config file supports `language` field.
- **Automatic Speaker Detection**: The `--auto-speakers` flag (default `true`) enables/disables automatic extraction of speaker names from episode metadata. Config file supports `auto_speakers` boolean field.
- **NER Model Override**: Advanced users can specify `--ner-model` to override default spaCy model selection (e.g., `en_core_web_sm`). Config file supports `ner_model` field.
- **Caching Control**: The `--cache-detected-hosts` flag (default `true`) controls whether host detection is memoized across episodes. Config file supports `cache_detected_hosts` boolean field.
- **Precedence Rules**:
  - Automatic detection runs first when `--auto-speakers` is enabled.
  - Manual `--speaker-names` are ONLY used as fallback when automatic detection fails (not as override).
  - Manual names format: first item = host, second item = guest (e.g., `["Lenny", "Guest"]`).
  - When guest detection fails: keep detected hosts (if any) + use manual guest name as fallback.
  - If detection succeeds, manual names are ignored; if detection fails, manual names are used as fallback.
- **Validation**: CLI validates language codes, NER model names, and ensures speaker name lists meet minimum requirements.
