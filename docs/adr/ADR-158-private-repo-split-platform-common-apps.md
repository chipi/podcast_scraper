# ADR-158: Split the apps, MCP, enrichers and sign-in providers into private repos

- **Status**: Proposed
- **Date**: 2026-10-07 (revised the same day with the probe's measurements)
- **Authors**: Marko Dragoljevic
- **See Also**: [Splitting a surface into a private repo](../guides/SPLITTING_A_SURFACE_INTO_A_PRIVATE_REPO.md),
  [Mounting the private eval research repo](../guides/EVAL_RESEARCH_MOUNT.md),
  [ADR-153](ADR-153-mcp-stays-read-only-over-http.md)

## Context & Problem Statement

Everything in this repo is public: the platform (pipeline, corpus, search, operator viewer) and
the consumer product built on it (the player app, its backend, its mobile shells, the MCP servers,
the enrichers). The product logic and the higher-value algorithms should not be public. A second
consumer app (news) is planned; it will add its own content types and endpoints but should
reuse sign-in, users and sessions rather than copy them.

The eval split (PR #2134) moved research out under the rule *tradecraft moves, runtime stays*.
That rule does not decide this split: the player is runtime.

The code does not mark the boundary. Measured on `main` at `96963f1cc`:

- 91 `server/app_*` and `server/routes/app_*` modules, ~21.7k lines; 42 of them are reached from
  platform code. The `app_` prefix covers four kinds of code: corpus read-models the platform
  uses, the identity kernel, endpoints the **operator viewer** calls, and player product logic.
- `enrichment/` is 67 files, ~14.6k lines; 18 modules outside it import from it.
- `mcp/` is 20 files (~3.0k lines); `podcast_obs/` is 18 files (~2.8k lines).
- `web/learning-player/` is 784 files (~134.6k lines) including `android/` and `ios/`.

### What the probe measured

`scripts/tools/split_copy.py` copies what `scripts/tools/split_manifest.yaml` lists into local,
unpublished repos under `apps/` and rewrites every import of a moved module. A probe then deletes
those files from a throwaway worktree and imports every module in both directions:

- **Public without private:** 35 public modules fail to import. Every failure traces to one of
  the seams in Decision 4: a public file reaching into moved code.
- **Private on top of public:** 82 of 117 private modules import; every one of the other 35 fails
  on a public module broken by those seams. The private code has no problem of its own.
- **The observability split (Decision 3) was applied and re-probed:** the Ops view's two routes no
  longer reach moved code.
- **Two misclassifications corrected:** `app_comms_store` and `app_release_store` are kernel (the
  outbox, account deletion, health and admin import them).
- **One leftover from the eval split** (`search/llm_judge.py`, unimportable on `main`) was deleted.

## Decision

### 1. Four layers, one repo each

| Layer | Visibility | Holds |
| --- | --- | --- |
| **Platform** (this repo) | public | pipeline, corpus, search, operator viewer, corpus read-models, the public kernel, the enrichment framework and two example enrichers, the observability data layer, the fixture corpora |
| **Common** | private | Google and Apple sign-in providers; the corpus MCP server (`mcp/`), its OAuth authorization server, MCP tokens and rate limiter; the observability MCP server; all other enrichers and their eval scorers; further IP chosen later |
| **Player** | private | player backend (product logic), `web/learning-player`, `android/`, `ios/`, its tests, make targets and CI jobs, player ranking-eval scripts |
| **News** | private, future | same shape as Player |

Dependencies point one way: Player and News depend on Common and Platform; Common depends on
Platform; Platform depends on nothing private and must run, test and deploy without it. Working
package names: `closelistening_common`, `closelistening_player`.

### 2. What stays public, by name

- **Kernel:** sessions, users, roles, CSRF, access and access store, magic link, outbox, push,
  comms consent (`app_comms_store`), released-app version (`app_release_store`), audit, operator
  guard, account deletion, user seed and CLI, the `OAuthProvider` protocol, the provider registry
  and `MockOAuthProvider`. `GoogleProvider` and `AppleProvider` move to Common.
- **Read-models:** `app_kg_view`, `app_gi_view`, `app_relational_view`, `app_momentum`,
  `app_catalog_cache`, `app_corpus_access`, `app_content_source`, `app_slugs`, `app_artwork`,
  `app_kg_index`. They leave the `app_` namespace (rename only); the private helpers
  `feed_signals.py` imports (`_role_of`, `_aggregate_role`, `_ROLE_RANK`) become public names.
- **What the operator viewer calls under `/api/app`:** auth, preferences, admin, graph events,
  profile. These were inside the `app_` prefix and stay with the viewer.

### 3. Observability splits along its own seam

`podcast_obs` has a clean internal boundary: its data layer (`config`, `result`, `_http`,
`sources/*`, `aggregate`) imports nothing from `mcp_server`, `cli` or `auth`. The data layer stays
public, because the operator Ops view uses it (`routes/ops.py`, `routes/llm_gateway.py`). The MCP
server, its CLI and its auth move to Common.

### 4. Extension interface, discovered by entry points

The platform loads installed private packages through Python entry points. A package registers
routers, scheduled jobs, startup hooks, CLI subcommands, capability-audit checks, user-lifecycle
hooks (account created, account deleted), sign-in providers, enrichers, data sources for public
read-models, and its data folder. Absent package, absent feature.

The probe's seams, and what each becomes (file:line in the plan note):

| Seam | Becomes |
| --- | --- |
| `server/app.py` mounts 17 player routers and 3 Common routers | registered routers |
| `server/app.py` starts the player cache warmer and digest-health metrics | startup hooks |
| `server/scheduler.py` imports the digest dispatcher | registered scheduled jobs |
| `capability_audit.py` imports discovery ranking | registered audit checks |
| `routes/corpus_enrichments.py` imports `filtered_entity_signals` from a player route | the function moves into the platform |
| `app_account_deletion.py` deletes player state and MCP tokens and grants | account-deleted hook, one per package |
| `routes/app_auth.py` writes `account_created` into player state on sign-up | account-created hook |
| `app_momentum.py` (public read-model) reads the player's engagement series | engagement is a registered data source; absent, momentum uses content signals only |
| `cli.py` imports the MCP CLI handlers | registered CLI subcommands |
| `enrichers/__init__.py` and `query_enrichers/__init__.py` import every enricher | registry fed by entry points |
| `ml_wiring.py`, `web_wiring.py`, `routes/enrichment_config.py`, `enrichment/eval/admission.py` name enricher classes | wiring looks enrichers up by id |
| `og/build.py` calls image-path helpers that live inside `person_web` and `org_web` | the helpers move into the platform |
| `tests/conftest.py` imports `person_web` | the fixture moves to Common's tests |
| `scripts/eval/score/rank_*.py` import player ranking | move to Player |
| `scripts/mcp_e2e_pivot_chain.py` imports MCP tools | moves to Common |
| `enrichment/eval/scorers/__init__.py` imports the moved scorers | registry fed by entry points, like the enrichers |
| `routes/ops.py`, `routes/llm_gateway.py` import `podcast_obs` | resolved by Decision 3 (re-probed: no edge left) |

Two consequences the import graph cannot show:

- Public code that reads a private enricher's output by name (`og/build.py`, `feed_signals.py`,
  `cil_queries.py`, the public search route's query enrichment, the operator viewer's enrichment
  panels) treats that output as optional.
- The operator viewer calls `/api/app/mcp` (MCP token management). Without Common that route is
  not mounted, and the viewer hides the UI.

### 5. Enrichers: two public examples, the rest private

`insight_density` (episode scope) and `guest_coappearance` (corpus scope) stay public. Both are
deterministic, read only GI and metadata, need no model, and the operator viewer already reads
them. Every other enricher moves to Common, with the eval scorers and gate metrics that grade it
(`enrichment/eval/scorers/grounding_rate.py`, `topic_similarity.py`,
`gate_metrics/enrichment/topic_consensus/`). The enrichment framework, including the eval runner,
stays public.

### 6. Fixture corpora keep private enrichers' outputs as frozen data

The public fixture corpora hold outputs of enrichers that move (95 `insight_sentiment.json` files,
`person_web` images, and others). They stay: they are derived data, not code, and public tests
and the operator viewer's e2e read them. Regenerating them (`make enrich-viewer-fixture`, which
runs every enricher) needs Common mounted. Public tests must never require regeneration.

### 7. One data folder per app

Kernel data stays at `data/users/<id>/profile.json`. App data moves to
`data/apps/<app>/users/<id>/…` (today the player writes `data/users/<id>/<name>.json` beside the
profile). The same split applies everywhere data is named: backups, compose volumes, and env
prefixes (shared sign-in settings keep `APP_*`; app-only settings take an app prefix). Prod moves
with a migration that has a dry run, verification and undo.

### 8. The mount, the copy and the probe

Private repos live in a gitignored `apps/` folder in this checkout, as `eval-data/` does. The
`.gitignore` entries (both `/apps/` and `/apps`) and every tool exclusion (flake8, black, isort,
markdownlint, bandit) were added and tested with planted files before the first `git init`.

The copy is a script, not an event. `split_copy.py` wipes `apps/<repo>/` (keeping `.git`) and
copies from the current tracked tree on every run, so the private side never drifts from `main`
while the seams are being fixed. The manifest is the single list of what moves. The repos stay
local until the probe is clean.

After the cutover, a change that needs both sides lands as two PRs: public first, with a contract
test against a fake app in this repo's tests; then the private one, which moves its pin. Private
CI runs against its pin and nightly against public `main`.

### 9. Images and deploy

The Player repo builds the production API image as the public API image plus Common and Player,
and the player web image. The pipeline image gets Common, because enrichment runs inside the
pipeline (`workflow/orchestration.py`). Deploying belongs to the infrastructure repo (PR #2138
moves the deploy, smoke and backup workflows there). iOS and Android build on a developer
machine. Until #2138 merges, this work stays out of `.github/workflows/`, `infra/`, `Makefile` and
`mkdocs.yml` beyond what it strictly needs.

### 10. Docs move with their surface

ADRs, RFCs, PRDs, UXS documents and WIP notes whose subject moves go with it, chosen by the
playbook's path sweep and then by reading each candidate. Public citations of moved docs use the
`repo-name:path` form. Round 1 moves whole documents (including
`docs/wip/PUBLIC-EXPOSURE-AND-PRIVATE-SPLIT.md`); round 2 later repairs mixed documents, listed
but not edited in round 1.

## Sequence

1. This ADR, the mount, the copy script and the probe. **Done.**
2. Fix the seams in public, one slice at a time, re-running the probe after each. Order: the
   extension interface with a fake app in tests; the enricher registry and wiring; the
   user-lifecycle hooks; router, startup, job, CLI and audit registration; the `podcast_obs` and
   OAuth-provider splits; the read-model renames. Done when both probe directions import cleanly.
3. Run the test suites in both directions and fix what fails.
4. Per-app data folders and the prod migration.
5. Read the round-1 documents; adjust the manifest.
6. Publish the private repos, set up their CI and image builds.
7. Cutover (playbook arc 2): delete from public, after PR #2138 has merged.
8. Later: further seams, then pipeline logic with IP value moves to Common.

## Consequences

- **Positive**: product logic and IP leave the public tree; a second app reuses sign-in, users
  and sessions without copying them; the platform gains an explicit extension boundary instead of
  an import web; progress is measured by the probe, not estimated.
- **Negative**: a cross-cutting change becomes two PRs in two repos; public CI cannot see the
  apps, so a public change can break them until the nightly run catches it; the production API
  and pipeline images are built in a private repo; public fixtures carry outputs the public repo
  cannot regenerate on its own.
- **Neutral**: everything already pushed stays in public history; this protects future work,
  not past commits.

## Alternatives Considered

1. **Make the whole repo private.** Simplest, but the platform stops being public at all.
2. **Two layers (platform and player).** A news app would have to depend on the player or copy
   its sign-in code.
3. **Keep the kernel private as well.** The operator viewer's login and the public platform
   would then depend on private code.
4. **Use `app_` as the boundary.** It covers read-models, kernel and viewer endpoints too; moving
   them breaks MCP tools, OG cards, corpus routes and the operator viewer.
5. **A one-off copy into the private repos, then refactor there.** The copy would drift from
   `main` from the first day, and the seams would be fixed in two places.
6. **Move the whole `podcast_obs` package to Common.** The public Ops view would lose its data.
7. **Git submodules instead of the mount.** Records the private repos' names and SHAs in public
   history, which the eval split avoided on purpose.

## Open questions

- Which further modules count as IP for Common (deferred until the refactor is seen).
- Repo names for Common and Player.
- Whether the 128 copied test files pass on the private side (not run yet: imports must be clean
  first).
- Whether the prod host can pull private registry images, and Actions-minute cost for private CI.
