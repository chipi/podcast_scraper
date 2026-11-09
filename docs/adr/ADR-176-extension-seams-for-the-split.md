# ADR-176: The stages that leave public are reached through seams, with today's code built in

- **Status**: Accepted (2026-11-09); implemented in `podcast_scraper/stage_hooks.py`,
  `podcast_scraper/read_hooks.py`, `podcast_scraper/cli_hooks.py` and their `builtin_*` modules,
  `podcast_scraper/extensions.py` (`Extension.pipeline_hooks`, `read_hooks`, `routers`,
  `cli_commands`), `podcast_scraper/identity/display.py`, and the `insight_stage` and
  `speaker_stage` packages (public #2446)
- **Date**: 2026-11-09
- **Authors**: Marko Dragoljevic
- **Related**: [ADR-168](ADR-168-installed-extensions.md) (installed extensions),
  [ADR-170](ADR-170-private-repo-split-platform-common-apps.md) (the private repo split),
  [ADR-174](ADR-174-extension-corpus-migrations.md) (extension corpus migrations)

## Context & Problem Statement

The split (ADR-170, the E04 split line) moves grounded insights (GI), the GI–KG bridge, speaker
detection and naming, diarization, ad cleaning and the identity ledger out of this package into a
private extension. Public keeps a plain pipeline that runs end to end without them.

Before the split, public code called that code directly: the pipeline imported GI and naming, the
server's read routes imported GI views, the CLI imported the GI verbs, and every surface that shows
a person's name imported the ledger. Moving the code in one change would also change every caller,
with no way to prove each caller on prod before the move.

## Decision

1. **Every caller reaches a moving stage through a named hook, never by import.** There are three
   registries, split by who calls them:
   - `stage_hooks`: the pipeline, the multi-feed batch and the corpus migrations.
   - `read_hooks`: the server's read side and search. It is kept apart so the pipeline's registry
     does not import the server's read models.
   - `cli_hooks`: CLI verbs and the batch-end ledger refresh. It is kept apart because the outcome
     hook measures a corpus migration by everything it can import (ADR-174), and a migration must
     not reach CLI-only code. Reaching it through `stage_hooks` added `refresh_guard` to migration
     `0039`'s measured closure.
2. **Today's code is registered as the built-in owner.** `builtin_stage_hooks`,
   `builtin_read_hooks` and `builtin_cli_hooks` hold it. An installed extension adds its own on
   `Extension.pipeline_hooks`, `Extension.read_hooks` and `Extension.cli_commands`.
   - A **list hook** (`run_started`, `index_rows`, `artifact_kinds`, …) runs the built-in entries,
     then each extension's, in load order.
   - A **single-owner hook** (`insight_grounding`, `entity_id_map`, …) is answered by the first
     owner that sets it, the built-in one first. Without an owner it returns a plain default: no
     speakers, no insights, no joins.
   - A hook keeps its own failure behaviour: one that must never fail a run catches its own errors.
     The registry adds no guard of its own.
3. **The moving code lives in its own packages.** `insight_stage` holds GI, the bridge, the context
   digest, the GI views, `cil_queries`, the insight clusters and the GI routes. `speaker_stage`
   holds hosts, guests, diarization and naming. They are reached only through the hooks. Each
   provider's GI and naming methods sit in a `*_provider_gi_naming` module beside the provider.
4. **Read routes that serve GI are extension routers.** The explore, CIL and corpus-persons routes
   are `RouterMount`s from `read_hooks.builtin_routers()`. Each declares the
   `/api/health` flags it serves (`advertises`). The server sets `explore_api`, `cil_queries_api`
   and `corpus_persons_api` from what it actually mounted in the running serve posture, and the
   viewer shows GI panels only when they are set.
5. **Artifact kinds are registered.** The kinds an episode can carry are
   `search.corpus_scope.PUBLIC_ARTIFACT_KINDS` plus each owner's `artifact_kinds` (built in: `gi`
   and `bridge`). `/api/artifacts` serves those kinds, and `/api/health` reports them as
   `artifact_kinds`.
6. **A person's shown name goes through the display seam.** Every surface reads
   `identity.display.person_display(root)`. The built-in builder is the ledger's
   (`identity.ledger_display`). `set_display_builder` replaces it, and `plain_display` (no joins)
   is what public alone shows.
7. **Configuration and flags stay plain.** A stage's switches stay ordinary `Config` fields, so
   operator YAML and profiles keep working. A stage-owned value is validated through its hook (the
   labeling profile through `stage_hooks.validate_labeling_profile`).

## Consequences

- **No behaviour change while the code is built in.** The proof for release 1:
  - every suite passed;
  - the mounted route set equals main's in each serve posture;
  - the fixture pipeline outputs are identical except run manifests;
  - a before/after comparison of every GET route on a copy of prod's corpus (2026-11-09) found no
    difference from the seams beyond the new `/api/health` fields.
- **The move becomes deletion plus registration.** In the core release a built-in entry is deleted
  and the extension registers the same function, so no caller changes. Player and Common import
  the moved modules at their new paths (`insight_stage.*`, `speaker_stage.*`).
- **A public-only install runs on the plain defaults.** That path is not proven yet: it is the core
  release's step 8, with its own fixture corpus and stack run.
- **A new caller has to use a hook.** Importing `insight_stage` or `speaker_stage` from platform
  code brings back the coupling the seams removed. `test_public_imports_load_no_speaker_stage.py`
  checks that the public entry points load without the speaker stage.

## Alternatives considered

- **Move the code in one change, without seams.** Rejected. Every caller would change in the same
  change as the code, with no step that could be proven on prod by itself.
- **One registry for every caller.** Rejected. A migration's measured closure would reach the
  server's read models and the CLI. The `0039` closure change showed the cost.
- **Load the stages by module string (`import_module`) instead of hooks.** Rejected. A string
  import is invisible to type checking and to the outcome hook's closure, and it would hide which
  stage a caller depends on.
