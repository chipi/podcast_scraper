# ADR-182: An installed extension declares its own configuration

- **Status**: Proposed
- **Date**: 2026-11-12
- **Authors**: Marko Dragoljevic
- **Related**: [ADR-168](ADR-168-installed-extensions.md) (installed extensions),
  [ADR-170](ADR-170-private-repo-split-platform-common-apps.md) (the split),
  [ADR-176](ADR-176-extension-seams-for-the-split.md) (the seams; this amends its decision 7)

## Context & Problem Statement

ADR-176 decision 7 kept a moved stage's switches as ordinary `Config` fields, so operator YAML and
profiles kept working through the move. The move is done, and that choice now has a cost:

- **Measured 2026-11-12** (every `Config` field, read as an attribute or a `getattr` name outside
  `config.py`): 47 fields are read only by the extension's code, 58 by both, 289 by public only,
  96 by nothing outside `config.py`. The 47 include the diarization tuning (`diarization_*`,
  `dgx_diarize_*`), the GI value gate and evidence modes (`gi_value_gate_*`, `gil_evidence_*`),
  speaker naming and labelling (`transcript_naming`, `labeling_profile`, `speaker_naming_passes`,
  `voice_identity_*`) and the per-provider speaker prompts.
- **On a public-only install those keys validate and do nothing.** A profile can set
  `voice_identity_rung` and the run silently ignores it; nothing says the key needs an extension.
- **Public documents and validates the extension's settings.** Their validators, defaults and
  descriptions are the extension's design, kept in public's `config.py`.

## Decision

1. **An extension declares its settings as a model.** `Extension.config_model` is a pydantic
   model of the fields the extension alone reads, with today's names, types, defaults and
   validators.
2. **One flat configuration, as today.** Operator YAML, profiles, environment variables and
   `Config(...)` keep their flat keys. Loading splits them: public's fields go to `Config`, each
   installed extension's fields to its model, kept on the config (`cfg.extension("stages")`). A
   read of a moved field through `cfg.<name>` keeps working during the change and warns once.
3. **A key nobody declares is refused.** `Config` forbids unknown keys today; a key that neither
   public nor an installed extension declares is still an error, now naming the extensions that
   are installed. A profile written for an extension fails loudly on an install without it.
4. **What moves is measured, then read.** The 47 fields read only by the extension are the
   candidates. Each is read before it moves (a field read by a public default or validator stays);
   fields read by both repos stay public.
5. **Flags follow their fields.** A moved field's CLI flag is the extension's, added through
   `cli_hooks` like its subcommands.
6. **Documentation follows too.** `docs/api/CONFIGURATION.md` documents public's fields; an
   extension documents its own.

## Consequences

- **Positive**: public's configuration is the platform's; an unsupported key fails on a
  public-only install; the extension owns its settings' design, validation and documentation.
- **Negative**: a configuration now depends on what is installed, so the same YAML can load on
  one install and fail on another; two places define fields; every moved field's readers change
  from `cfg.x` to the extension's model (the warning bridges the change, not forever).
- **Neutral**: operator YAML, profiles and prod's environment do not change; no corpus change.

## Alternatives considered

1. **Keep ADR-176 decision 7 (all switches in `Config`).** No migration, but public keeps 47
   settings it never reads, and a public-only install accepts them silently.
2. **A namespaced section (`extensions: {stages: {...}}`).** Clear ownership, but every operator
   YAML, profile and environment variable changes, and prod's with them.
3. **Merge extension fields into `Config` at import (`create_model`).** One model and flat keys,
   but `Config`'s type then depends on import order, static typing sees none of the moved fields,
   and the fields are still documented as public's.
4. **Accept unknown keys with a warning.** A public-only install could load prod's profiles, but a
   typo or a missing extension would pass silently, which `extra=forbid` exists to prevent.

## Not covered

- The order of the move (which of the 47 first) and its tests.
- Settings that both repos read: whether any should become the extension's with a public default.
- Profiles and operator YAML that set moved keys (ADR-170 item 8 moves the tuned profiles).
