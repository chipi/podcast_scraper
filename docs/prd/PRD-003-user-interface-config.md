# PRD-003: User Interfaces & Configuration

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
- *As Operator Owen, I can run `python -m podcast_scraper.cli <rss_url>` with sensible defaults and see progress bars and status logs.*
- *As Automation Alex, I can maintain a JSON/YAML config file checked into version control and use `--config` to load it.*
- *As a user, I can request version info (`--version`) and set log level verbosity per run.*
- *As Integrator Iris, I can call `podcast_scraper.Config` + `podcast_scraper.run_pipeline` directly in Python with the same semantics.*

## Functional Requirements
- **FR1**: CLI must validate inputs (RSS URL, numeric ranges, Whisper model choices) and surface actionable error messages.
- **FR2**: CLI flags map to `Config` fields; precedence is CLI > config file defaults (with validation).
- **FR3**: Support both JSON and YAML configuration files loaded via `--config` with schema validation.
- **FR4**: Expose logging controls (`--log-level`) and default to INFO.
- **FR5**: Default progress reporter uses `tqdm`; expose abstraction (`progress.set_progress_factory`) to override in embedded contexts.
- **FR6**: Provide `--dry-run`, `--skip-existing`, `--clean-output`, `--workers`, and other operational flags documented in README.
- **FR7**: Ensure exit codes communicate success (0) vs. validation or runtime failures (1).
- **FR8**: Export Python API surface (`Config`, `load_config_file`, `run_pipeline`, `cli.main`) from `podcast_scraper.__init__`.

## Success Metrics
- CLI onboarding: a new user can run the default command with only an RSS URL and receive useful output/logging.
- Config file onboarding: loading an invalid config produces a clear validation error (no partial runs).
- Python API: integration tests confirm parity with CLI semantics.

## Dependencies
- Validation and configuration logic described in `docs/rfc/RFC-007-cli-interface.md` and `docs/rfc/RFC-008-config-model.md`.
- Progress abstraction detailed in `docs/rfc/RFC-009-progress-integration.md`.

## Release Checklist
- [ ] CLI help text audited and examples verified in README.
- [ ] Integration tests cover CLI happy path, invalid args, config file precedence, programmatic usage.
- [ ] Version string maintained in sync (`__version__`).

## Open Questions
- Should we support environment variable substitution in config files? Not currently planned.
- Do we need subcommands for future expansion (e.g., `inspect`, `clean`)? Monitor user feedback.
