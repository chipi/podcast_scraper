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

`scripts/tools/split_copy.py` copies what `scripts/tools/split_manifest.yaml` lists into the
private repos under `apps/` and rewrites every import of a moved module.
`scripts/tools/split_probe.py` then deletes those files from a throwaway worktree and imports every
module in both directions. With the manifest as of decision 5 below:

- **Public without private:** 27 of 575 public modules fail to import, and 65 imports in 20 files
  still name moved code. Every failure traces to one of the seams in Decision 4.
- **Private on top of public:** 20 of 100 private modules fail, every one on a public module broken
  by those seams. The private code has no problem of its own.
- **The observability split (Decision 3) was applied and re-probed:** the Ops view's two routes no
  longer reach moved code.
- **Tier A (Decision 5) is not in the manifest yet:** its scope is measured below, but the modules
  that move with it are still to be read and listed.
- **Two misclassifications corrected:** `app_comms_store` and `app_release_store` are kernel (the
  outbox, account deletion, health and admin import them).
- **One leftover from the eval split** (`search/llm_judge.py`, unimportable on `main`) was deleted.

## Decision

### 1. Four layers, one repo each

| Layer | Visibility | Holds |
| --- | --- | --- |
| **Platform** (this repo) | public | pipeline, corpus, plain search, operator viewer, corpus read-models, the public kernel, the enrichment framework and the five enrichers without IP, the observability data layer, the fixture corpora (public outputs only) |
| **Common** | private | Google and Apple sign-in providers; the corpus MCP server (`mcp/`), its OAuth authorization server, MCP tokens and rate limiter; the observability MCP server; the six enrichers with real logic, their eval scorers, and the platform features built on their outputs (Decision 5); further IP chosen later |
| **Player** | private | player backend (product logic), `web/learning-player`, `android/`, `ios/`, its tests, make targets and CI jobs, player ranking-eval scripts |
| **News** | private, future | same shape as Player |

Dependencies point one way: Player and News depend on Common and Platform; Common depends on
Platform; Platform depends on nothing private and must run, test and deploy without it.
Repositories: `chipi/closelistening-common`, `chipi/closelistening-player` (private, created
2026-10-07). Package names: `closelistening_common`, `closelistening_player`.

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

### 5. Enrichers: the line is IP, and the features built on private outputs go with them

**Public:** the five enrichers with nothing to protect: `insight_density`, `guest_coappearance`,
`insight_sentiment` (a wrapper around the VADER lexicon), `grounding_rate`,
`topic_cooccurrence_corpus`, with the `grounding_rate` scorer. They are the working reference for
writing an enricher.

**Private (Common):** the six with real logic: `person_web`, `org_web`, `temporal_velocity`,
`topic_consensus`, `topic_theme_clusters`, `topic_similarity`, the query enricher, the
`topic_similarity` scorer and the `topic_consensus` gate metrics.

Hiding the code is not enough: public features that consume an output publish what it contains.
So outputs are tiered by who consumes them, and the consumer moves with the producer:

- **Tier A, outputs public features are built on:** `topic_consensus`, `topic_theme_clusters`,
  `topic_similarity`, `temporal_velocity`. Consensus search, storylines, topic clusters, graph
  lenses, trending and OG cards read them. **These features move private too** (operator decision
  2026-10-07); the public platform keeps ingest, corpus and plain search. Measured scope: the four
  ids appear in 46 public `src/` files (search operators, storylines, topic clusters,
  `routes/corpus_storylines.py`, `routes/search.py`, `og/build.py`, schemas, enrichment wiring)
  and 25 operator-viewer files (graph lenses, the search operator bar, dashboard trending, the
  theme legend, enrichment panels). Which of those move and which only stop reading the output is
  the next manifest step; how the operator viewer loses these features is an open question.
- **Tier B, outputs only private code consumes:** `person_web`, `org_web`. Their schemas
  (`AppPersonWeb`, `AppOrgWeb`), fixtures and the two image-path helpers move to Common.
- **Tier C, pipeline IP:** prompts, GI and KG extraction. Untouched by this split; it is the later
  phase (sequence step 8).

### 6. Fixture corpora carry public outputs only

The public fixture corpora keep what the public pipeline and the five public enrichers produce.
Outputs of private enrichers (tier A and B) move to a fixture overlay in Common, which private
tests apply on top of the public corpus; public tests that assert on those outputs move with them.
Regenerating fixtures (`make enrich-viewer-fixture`) becomes public-only; Common gets its own
target for the overlay.

### 7. One data folder per app

Kernel data stays at `data/users/<id>/profile.json`. App data moves to
`data/apps/<app>/users/<id>/…` (today the player writes `data/users/<id>/<name>.json` beside the
profile). The same split applies everywhere data is named: backups, compose volumes, and env
prefixes (shared sign-in settings keep `APP_*`; app-only settings take an app prefix). Prod moves
with a migration that has a dry run, verification and undo.

### 8. The mount, the copy and the probe

Private repos live in a gitignored `apps/` folder in this checkout, as `eval-data/` does. The
`.gitignore` entries (both `/apps/` and `/apps`) and every tool exclusion (flake8, black, isort,
markdownlint, bandit, the doc-structure check) were added and tested with planted files before
the first `git init`.

The copy is a script, not an event. `split_copy.py` wipes `apps/<repo>/` (keeping `.git`) and
copies from the current tracked tree on every run, so the private side never drifts from `main`
while the seams are being fixed. The manifest is the single list of what moves, and
`split_probe.py` is the measure. The GitHub repos exist early (created 2026-10-07) so private CI,
image builds and pinning can be set up while the seams are fixed; what is pushed to them is
regenerated by the copy until the cutover.

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
playbook's path sweep and then by reading each candidate. Round 1 moves whole documents
(including `docs/wip/PUBLIC-EXPOSURE-AND-PRIVATE-SPLIT.md`); round 2 later repairs mixed
documents, listed but not edited in round 1.

References across the boundary run one way only:

- **Private → public: allowed, by ID, never as a link.** A private document names a public one by
  its identifier ("ADR-158", "PRD-039", "RFC-088"), not by a relative path or URL. A path breaks
  as soon as either repo's layout changes, which is how the copied player docs failed the doc
  check.
- **Public → private: not allowed.** No public document, comment or code names a private
  document, by link, path or ID. When a document moves, public references to it are removed, not
  converted. Measured cost, accepted by the operator: 825 references to the 33 moving document IDs
  in 237 public files (230 in `src/` docstrings, 62 in `mkdocs.yml`), and 288 lines in 99 files
  that name the earlier eval split's private repo (index stubs, registry evidence citations and
  their tests, onboarding docs). The mount tooling may name the `eval-data/` and `apps/`
  directories, because `.gitignore` has to.

## Sequence

1. This ADR, the mount, the copy script, the probe, and the two empty private repos. **Done.**
2. Fix the seams in public, one slice at a time, re-running the probe after each. Order: the
   extension interface with a fake app in tests; the enricher registry and wiring; the
   user-lifecycle hooks; router, startup, job, CLI and audit registration; the `podcast_obs` and
   OAuth-provider splits; the read-model renames. Done when both probe directions import cleanly.
3. Run the test suites in both directions and fix what fails.
4. Per-app data folders and the prod migration.
5. Read the round-1 documents; adjust the manifest.
6. Private CI, image builds and pinning (can start in parallel with steps 2–5).
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
- **The limit**: this hides future work, not past work, and not what clients see. The repo is
  public with 3 forks and 8 stars; every enricher, the player and the MCP servers as they stand
  today are out permanently. Anything the phone receives from `/api/app`, and the MCP tool
  descriptions any token holder can list, stay observable whatever the code's visibility.
- **Also**: the public pipeline image no longer runs the private enrichers; enrichment with them
  needs the image the private repo builds.

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
- How the operator viewer loses the tier-A features: strip them from the public viewer, make the
  viewer load private panels, or move the viewer private.
- Whether Common is two packages (identity: sign-in, MCP auth; intelligence: enrichers, MCP
  tools), so the news app can depend on identity alone.
- Versioning between repos: how a private repo pins the public one (git SHA, a deploy key, the
  public image tag), and how tags are cut.
- MCP tokens live under `data/users/<id>/` today: the per-app data migration and the
  account-deleted hook have three owners (kernel, Common, Player), not two.
- Whether the 128 copied test files pass on the private side (not run yet: imports must be clean
  first).
- Whether the prod host can pull private registry images, and Actions-minute cost for private CI.
