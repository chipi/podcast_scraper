# ADR-158: Split the apps, MCP, enrichers and sign-in providers into private repos

- **Status**: Proposed
- **Date**: 2026-10-07
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
That rule does not decide this split: the player is runtime. It is built, published, deployed
and smoke-tested from this repo.

The code does not mark the boundary today. Measured on `main` at `96963f1cc` with an AST import
graph over `src/`:

- 91 `server/app_*` and `server/routes/app_*` modules, ~21.7k lines.
- **42** of them (~12.0k lines) are reached from platform code other than the router mounts in
  `server/app.py`, so they cannot simply move. Reading them, they are three different kinds of
  code that share one prefix:
  - **Corpus read-models** used by the platform itself: `app_kg_view`, `app_gi_view`,
    `app_relational_view`, `app_momentum`, `app_catalog_cache`, `app_corpus_access`,
    `app_content_source`, `app_slugs`, `app_artwork`, `app_kg_index`. MCP tools, OG cards,
    `feed_signals.py` and the `corpus_*` routes import them.
  - **Identity and comms**: `routes/app_auth`, `app_oauth`, `app_oauth_server`, `app_sessions`,
    `app_csrf`, `app_roles`, `app_user_store`, `app_rate_limit`, `app_audit`, `app_mcp_tokens`,
    `app_magic_link`, `app_access`, `app_access_store`, `app_operator_guard`, `app_user_seed`,
    `app_users_cli`, `app_account_deletion`, `app_outbox_store`, `app_push_store`,
    `app_release_store`.
  - **Player product logic** pulled in transitively (`app_user_state`, `app_recap`, `app_stats`,
    `app_discover_view`, `app_ranking_config`, `app_digest_*`).
- **50** modules (~9.7k lines) are reached only through `app.py` router mounts or other `app_*`
  modules.
- `enrichment/` is 67 files, ~14.6k lines; 18 modules outside it import from it, including
  `workflow/orchestration.py`.
- `mcp/` is 20 files (~3.0k lines); `podcast_obs/` is 18 files (~2.8k lines).
- `web/learning-player/` is 784 files (~134.6k lines) including `android/` and `ios/`.

## Decision

### 1. Four layers, one repo each

| Layer | Visibility | Holds |
| --- | --- | --- |
| **Platform** (this repo) | public | pipeline, corpus, search, operator viewer, corpus read-models, the public kernel, the enrichment framework, the fixture corpus |
| **Common** | private | Google and Apple sign-in providers, both MCP servers (`mcp/`, `podcast_obs/`) and the MCP OAuth server, all enrichers except two, further IP chosen later |
| **Player** | private | player backend (product logic), `web/learning-player`, `android/`, `ios/`, its tests, make targets, workflows, compose and deploy |
| **News** | private, future | same shape as Player |

Dependencies point one way: Player and News depend on Common and Platform; Common depends on
Platform; Platform depends on nothing private and must run, test and deploy without it.

### 2. The public kernel

Sessions, users, roles, CSRF, rate limiting, audit, magic link, outbox and push stay public, as
does the `OAuthProvider` protocol (`app_oauth.py`), the provider registry and
`MockOAuthProvider`. The operator viewer's login needs them, and a second app reuses them.
`GoogleProvider` and `AppleProvider` move to Common.

### 3. Corpus read-models are platform code

The ten read-models listed above leave the `app_` namespace (rename only, no behaviour change).
Private helpers that platform code imports across modules (`feed_signals.py` imports `_role_of`,
`_aggregate_role`, `_ROLE_RANK`) become public names.

### 4. Extension interface, discovered by entry points

The platform loads installed private packages through Python entry points. An app registers:
routers, scheduled jobs, startup hooks, capability-audit checks, an account-deletion hook and its
data folder. Common registers sign-in providers, enrichers and the MCP servers. Absent package,
absent feature. Each place where public code reaches into private code today becomes a
registration:

| Where | Today | Becomes |
| --- | --- | --- |
| `server/app.py` | mounts every `app_*` router | mounts registered routers |
| `server/app.py` (cache warmer, digest-health metrics) | starts player machinery | registered startup hooks |
| `server/scheduler.py` | imports `app_digest_dispatch` | registered scheduled jobs |
| `capability_audit.py` | imports `app_discover_view`, `app_ranking_config` | registered audit checks |
| `routes/corpus_enrichments.py` | imports `filtered_entity_signals` from `routes.app_enrichment` | function moves into the platform |
| `routes/app_auth.py` | imports `player_client_health` | client health supplied per app |
| `app_account_deletion.py` | deletes player state | calls each app's deletion hook |
| `enrichers/__init__.py` | registers all enrichers | registers the two public ones; Common registers the rest |

Public code that reads one enricher's output by name (`og/build.py`, `feed_signals.py`,
`cil_queries.py` and others) must treat that output as optional.

### 5. Two public example enrichers

`insight_density` (episode scope) and `guest_coappearance` (corpus scope) stay public. Both are
deterministic, read only GI and metadata, need no model, and the operator viewer already reads
them. They are the working reference for writing an enricher. Every other enricher, and the
query enricher, moves to Common.

### 6. One data folder per app

Kernel data stays at `data/users/<id>/profile.json`. App data moves to
`data/apps/<app>/users/<id>/…` (today the player writes `data/users/<id>/<name>.json` beside the
profile, `app_user_state.py`). The same split applies everywhere data is named: backups, compose
volumes, and env prefixes (shared sign-in settings keep `APP_*`; app-only settings take an app
prefix). Prod moves with a migration that has a dry run, verification and undo.

### 7. Working on all repos at once: the mount pattern

Private repos clone into a gitignored `apps/` folder in this checkout, as `eval-data/` does
today. One venv installs all of them editable (`pip install -e . -e apps/common -e apps/player`).
`.gitignore` carries both `/apps/` and `/apps`, and the linter and formatter exclusions are added,
before the first clone.

A change that needs both sides lands as two PRs: the public one first, with a contract test
against a small fake app in this repo's tests; then the private one, which moves its pin to that
public SHA. Private CI runs against its pin and nightly against public `main`, because public CI
cannot see private code.

### 8. Images and deploy

The Player repo builds the production API image as the public API image plus the Common and
Player packages, and builds the player web image. Both publish to private registry packages.
The pipeline image gets the same treatment with Common, because enrichment runs inside the
pipeline (`workflow/orchestration.py`).

Deploying is not the Player repo's job. The infrastructure split (PR #2138) already removes the
deploy, smoke and backup workflows (`deploy-player.yml`, `smoke-player.yml`,
`backup-player-appdata-prod.yml` and the rest) from this repo; the deploy side consumes the
images the Player repo publishes. iOS and Android builds run on a developer machine, not in CI.

To avoid conflicts with #2138, the in-repo refactor (step 2 of the sequence) stays out of
`.github/workflows/`, `infra/`, `Makefile` and `mkdocs.yml` beyond what it strictly needs,
until PR #2138 has merged.

### 9. Docs move with their surface

ADRs, RFCs, PRDs, UXS documents and WIP notes whose subject moves go with it, chosen by the
playbook's path sweep and then by reading each candidate, never by keyword. Public citations of
moved docs use the `repo-name:path` form. Two rounds:

1. **Whole documents** whose subject is entirely private move as they are. This includes the
   earlier analysis of what the public repo exposes (`docs/wip/PUBLIC-EXPOSURE-AND-PRIVATE-SPLIT.md`).
2. **Mixed documents** that cover both sides get a later repair: the private part moves, the
   public part stays. Round 1 produces the list of these; it does not edit them.

## Sequence

1. This ADR.
2. Refactor inside this repo, still public: renames, separate top-level packages for kernel,
   Common and Player, the extension interface, the data-folder migration. Every test still runs
   in one repo.
3. Copy into the private repos (playbook arc 1).
4. Delete here and fix every caller (playbook arc 2).
5. Later: more seams, then the pipeline logic with IP value moves to Common.

## Consequences

- **Positive**: product logic and IP leave the public tree; a second app reuses sign-in, users
  and sessions without copying them; the platform gains an explicit extension boundary instead of
  an import web.
- **Negative**: a cross-cutting change is two PRs in two repos; public CI cannot see the apps,
  so a public change can break them until the nightly run catches it; the production API image is
  built in a private repo and the single image tag splits.
- **Neutral**: everything already pushed stays in public history; this protects future work,
  not past commits.

## Alternatives Considered

1. **Make the whole repo private.** Simplest, but the platform stops being public at all.
2. **Two layers (platform and player).** A news app would have to depend on the player or copy
   its sign-in code.
3. **Keep the kernel private as well.** The operator viewer's login and the public platform
   would then depend on private code.
4. **Use `app_` as the boundary.** About ten of those modules are platform read-models; moving
   them breaks MCP tools, OG cards and corpus routes.
5. **Git submodules instead of the mount.** Records the private repos' names and SHAs in public
   history, which the eval split avoided on purpose.

## Open questions

- Which further modules count as IP for Common (deliberately deferred until the refactor is seen).
- Repo names for Common and Player.
- Who outside `app_*` pulls `app_recap` and `app_stats` into the platform-needed set: not yet
  traced.
- How the 105 Python test files that touch `app_*` split.
- Whether the prod host can pull private registry images, and Actions-minute cost for private CI.
