# Private split — initial plan (2026-10-07)

**Decision record:** [ADR-158](../adr/ADR-158-private-repo-split-platform-common-apps.md).
This note is the working list behind it: the order of work, the round-1 document list, and the
state of the infrastructure PR that has to land first.

**Status:** plan only. Nothing has moved. All counts are from `main` at `96963f1cc`.

---

## 0. First: land the infrastructure split (PR #2138)

Operator decision 2026-10-07: merge #2138 first and get rid of infra before this split starts,
because it removes the deploy, smoke and backup workflows this split would otherwise have to
carve up. Reviewed 2026-10-07; it cannot merge as it stands.

| Finding | Detail |
| --- | --- |
| Branch is stale | merge base `45c68dbe1` (2026-09-23); 5 commits ahead, 222 behind `main` |
| `main` kept changing infra after the split | 29 files; the infra repo (`chipi/podcast_scraper-infra`, last push 2026-09-29) has none of those changes |
| — new on `main`, absent in infra | 5: `scripts/ops/prod_health_lib.sh`, `scripts/ops/prod_recovery_check.sh`, `scripts/obs/umami_views.py`, `scripts/obs/verify_dashboard.py`, `config/grafana/dashboards/podcast-player/beta-usage.json` |
| — infra still holds the 09-23 version | 15: `deploy-all-prod`, `deploy-prod`, `drill-deploy`, `gi-repair-prod`, `reprocess-prod`, `restage-prod-secrets` workflows; `config/obs/tenants.yaml`; `config/observability.prod.yaml`; `infra/caddy/player.caddy`; `infra/cloud-init/prod.user-data`; `infra/deploy/deploy-{operator,player}.sh`; `infra/observability/base.alloy`; `scripts/ops/restage_prod_recreate.sh` |
| — changed on both sides (3-way merge from `45c68dbe1`) | 9: `deploy-player`, `deploy-operator`, `backup-corpus-prod`, `prod-restore-corpus` workflows; `tailscale/policy.hujson`; `PLAYER_PUBLIC_LAUNCH.md`; `PROD_RUNBOOK.md`; `DEPLOY_GOTCHAS.md`; `OBSERVABILITY_ARCHITECTURE.md`; `test_stack_contract_restore_scripts.py` |
| Trial rebase onto `main` | stops on the first commit with 26 conflicts: 24 modify/delete (the files above) plus `Makefile` and `mkdocs.yml` |
| New references on `main` to deleted paths | tests: `test_public_media_routes_are_exempt_at_the_edge.py` (reads `infra/caddy/player.caddy`, `infra/caddy/validate.sh`), `test_dev_obs_env_switch_obs.py` (reads `config/observability.homelab.yaml`); docs: `ADR-157`, `OPERATOR_SMOKE_TEST.md`, `UXS-018` |
| Replacement deploy path | not active: the infra repo keeps all 46 workflows in `.github/workflows-staged/` (its README: "None of them can run"); `gh run list` shows no runs; secrets and the Actions setting are not readable with the session token (HTTP 403) |

Order to land it: port the 29 files into infra → activate infra and prove one deploy → rebase
PR #2138 (the 24 modify/delete conflicts then resolve to delete; `Makefile` and `mkdocs.yml` by
hand) → move or invert the two tests and delink the three docs → rerun the gates the PR did not
run locally (`test-ui`, `build-viewer`, `test-app`, `build-app`, `stack-test`) → no infra edits
on `main` between the port and the merge.

Not covered by that review: a file-by-file correctness read of the PR's deletions; its other four
commits beyond their titles; indirect references (variables, globs).

---

## 1. Seams, measured (ADR-158 sequence step 2)

Everything stays in this repo until the probe is clean. Each slice is its own PR, and the probe
is re-run after each one; the failure count is the progress measure. Work that touches
`Makefile`, `.github/workflows/`, `compose/` or `mkdocs.yml` waits for #2138.

### How the probe works

`scripts/tools/split_copy.py` copies what `scripts/tools/split_manifest.yaml` lists into the
local, unpublished repos `apps/common` and `apps/player`, rewriting every import of a moved
module. The probe deletes those files from a throwaway worktree and checks both directions:
public without private (import every public module; list every import edge into moved code), and
private on top of public (import every private module, with `PYTHONPATH`, nothing installed).

### Current numbers (`scripts/tools/split_probe.py`, after ADR-158 decisions 3 and 5, 2026-10-07)

| | Result |
| --- | --- |
| Baseline, unpruned tree | 0 import failures (after deleting the eval split's leftover `search/llm_judge.py`) |
| Copied | Player: 53 modules, 68 tests, 786 web files, 67 docs. Common: 38 modules, 57 tests, 16 docs |
| Copy check | 0 stale references (AST verifier; it catches a planted stale import and `mock.patch` string) |
| Public without private | 27 of 575 modules fail; roots: MCP tokens via account deletion (13), enricher registry (6), query-enricher registry (3), engagement series via momentum (2), scorer registry, discovery ranking, enrichment route function (1 each). 65 imports in 20 files still name moved code |
| Private on top of public | 20 of 100 modules fail, all on a public module broken by a seam |
| Not yet in the manifest | tier A features (ADR-158 decision 5): 46 public `src/` files and 25 operator-viewer files read the four tier-A outputs; to be read and listed |

### Seams

| # | Seam (public file:line → moved code) | Fix | Order |
| --- | --- | --- | --- |
| 1 | `enrichment/enrichers/__init__.py:17-27`, `query_enrichers/__init__.py:5`, `enrichment/eval/scorers/__init__.py:19,21` import every enricher or scorer | registries fed by entry points; public registers its two examples | 2 |
| 2 | `enrichment/ml_wiring.py:27,30`, `web_wiring.py:13-14`, `routes/enrichment_config.py:230-234`, `enrichment/eval/admission.py:101` name enricher classes | wiring looks enrichers up by id | 2 |
| 3 | `server/og/build.py:366,400` call `person_web.person_image_path`, `org_web.org_logo_path` | the two file-path helpers move into the platform | 2 |
| 4 | `tests/conftest.py:1070` imports `person_web` | the fixture moves to Common's tests | 2 |
| 5 | `server/app_account_deletion.py:32` deletes MCP tokens and OAuth grants | account-deleted hook per package | 3 |
| 6 | `server/routes/app_auth.py:392,871` writes `append_account_created` into player state | account-created hook | 3 |
| 7 | `server/app_momentum.py:25` reads the player's engagement series | registered data source; absent, content signals only | 3 |
| 8 | `server/app.py:30` mounts 17 player and 3 Common routers | registered routers | 4 |
| 9 | `server/app.py:497` cache warmer, `:663` digest-health metrics | startup hooks | 4 |
| 10 | `server/scheduler.py:424` digest dispatch | registered scheduled jobs | 4 |
| 11 | `capability_audit.py:44,1178,1184` → discovery ranking | registered audit checks | 4 |
| 12 | `routes/corpus_enrichments.py:36` → `filtered_entity_signals` | the function moves into the platform | 4 |
| 13 | `cli.py:3970,5282` → MCP CLI handlers | registered CLI subcommands | 4 |
| 14 | `routes/ops.py`, `routes/llm_gateway.py` → `podcast_obs` | **resolved** by decision 3 (obs data layer public, MCP server private); re-probe shows no edge | done |
| 15 | `scripts/eval/score/rank_discover_v1.py`, `rank_scenarios_v1.py` → player ranking | move to Player | manifest |
| 16 | `scripts/mcp_e2e_pivot_chain.py` → MCP tools | move to Common | manifest |

Order 1 is the extension interface itself (protocol, entry-point loader, a fake app in tests
exercising every hook). Then 2–4 as numbered, then the OAuth-provider split and the read-model
renames, which no import depends on.

### Plan review — risks the probe surfaced (2026-10-07)

1. **Entry points need an install.** The probe runs on `PYTHONPATH`, which cannot see entry
   points. Testing real registration needs `pip install -e apps/common -e apps/player` into some
   venv. Installing into the shared `.venv` needs the operator's approval first; the alternative
   is a separate venv for the private side. Decide before order 1 lands.
2. **Player e2e assumes the repo root two levels up.** Four files under `web/learning-player`
   (Playwright configs and e2e helpers) reference `../../src`, `../../.venv/bin/python` and
   `../../scripts/tools/run_e2e_mock_server.py` 18 times. From `apps/player/web` that resolves to
   `apps/`. Those paths need to come from one setting (a platform-root variable) before the
   private e2e can run.
3. **The "what" stays visible; only the "how" goes private.** The operator viewer is public and
   renders the private enrichers' outputs; the public fixture corpora contain those outputs; the
   player's `/api/app` surface is visible to anyone who runs the app. Moving the code hides how
   results are computed, not what is computed. If an output schema itself is IP, that is a
   separate decision.
4. **Public fixtures cannot be regenerated without Common.** `make enrich-viewer-fixture`
   (`Makefile:1794`) runs every enricher over the public fixture corpus. ADR-158 decision 6 keeps
   the outputs as frozen data; regeneration then requires the mount.
5. **The released-version store names one app.** `app_release_store` holds `player_version`. It
   stays in the kernel because health and admin read it, but a second app needs it per app.
6. **Private tests may depend on public test helpers.** The copy brings the `conftest.py` chain
   but not helper modules under `tests/`. Unknown until the private tests run (sequence step 3).
7. **Copied docs link into public docs by relative path.** With the mount populated,
   `make check-doc-structure` went red on links such as `../../../docs/guides/E2E_TESTING_GUIDE.md`
   in the copied player docs: same cause as risk 2. The gate itself also walked `apps/` (and
   would have walked `eval-data/`); fixed with a top-level-only skip and a test. Round 1 of the
   doc move has to rewrite these links as plain IDs ("ADR-158", "PRD-039"): private docs may
   name public docs by ID, never by path or link; public docs name no private doc at all
   (operator rule, 2026-10-07; ADR-158 decision 10).
8. **The probe checks imports, not behaviour.** A clean probe means everything loads. Runtime
   paths (a router missing at request time, a hook never called) show up only in the test runs.

## 2. Documents — round 1 (whole documents that move)

Found by the path sweep (references to moving code) plus a title pass over every ADR, RFC, PRD
and UXS. **Each one still has to be read before it moves** — the playbook's four false positives
in the eval split were found only by opening the file. One false positive is already known:
`AGENT_BROWSER_LOOP_GUIDE.md` matched "MCP" 80 times but is about the DevTools MCP; it stays.

### To Player

- **PRD**: 035 Learning Platform, 036 Foundation / Identity, 037 Discovery, 038 Catalog,
  039 Player, 040 Capture, 041 Consolidation, 042 Home, 043 Knowledge Layer, 046 Delivery &
  Curation
- **RFC**: 098 Learning Platform Foundation, 099 Consumer Client, 100 Audio Bridge, 101 Personal
  Knowledge Corpus, 102 Knowledge Clusters, 111 Curation Surfaces, 113 PKM Export, 114 Personal
  Corpus, 119 Holistic Collections, 120 Login-first, 121 Unified Saved, 122 Post-episode recap
- **ADR**: 146 Your Week
- **UXS**: 011 Consumer Learning App, 012 Consumer Home, 013 Knowledge clusters (consumer),
  014 Interaction patterns (consumer)
- **API**: `api/PLATFORM_API.md` (`/api/app`)
- **Guides**: `CONSUMER_LEARNING_PLAYER_GUIDE`, `MOBILE_E2E_TESTING`,
  `MOBILE_STORE_RELEASE_RUNBOOK`, `E2E_ON_INTEL_MAC`, `RECOMMENDATION_GUIDE`,
  `NOTIFICATIONS_GUIDE`
- **WIP**: `PLAYER-BACKEND-WAVES-2026-09-10`, `PLAYER-GOLDEN-WALKTHROUGH-v3`,
  `PLAYER-NEXT-WAVE-G-J-2026-09-10`, `DAILY-RECAP-EMAIL-HANDOFF`, `DEVICE-FEEDBACK-2026-09-19`,
  `DEVICE-ONLY-DEFECTS-2026-09-27`, `DEVICE-TIERS-HANDOVER-2026-09-27`,
  `APP-LINKS-PRE-BUILD-TODO-2026-10-05`, `HANDOFF-BETA-USER-SHOWS-SMOKE-2026-10-01`,
  `REDESIGN-CRITIC-FEEDBACK`, `wip/player/`, `wip/knowledge-retention/`, the three `recap-*.png`

### To Common

- **MCP**: PRD-034, RFC-093, RFC-095, RFC-112, ADR-153; guides `EXPLORATION_MCP`,
  `LOCAL_MCP_SERVERS`, `MCP_SERVER_GUIDE`, `OBS_MCP_HOMELAB_DEPLOY`,
  `OBSERVABILITY_CONTROL_PLANE`; WIP `OBS-MCP-ON-VPS-PLAN`
- **Enrichers**: ADR-108 (NLI enrichers); WIP `ENRICHER-HARDENING-ROADMAP`,
  `ONBOARDING-SHOWS-FOR-ENRICHER-VALUE`, `2026-09-02-enricher-reenable-prep`
- **The split itself**: `PUBLIC-EXPOSURE-AND-PRIVATE-SPLIT` (operator, 2026-10-07)

### Owned by #2138, not this split

`PLAYER_PUBLIC_LAUNCH.md` and the other deploy runbooks are deleted or moved by the infra PR.

---

## 3. Documents — round 2 (mixed; repaired later, listed only)

These cover a surface that stays public and one that moves. Round 2 moves the private part and
leaves the public part.

| Document | Why mixed |
| --- | --- |
| RFC-088, `ENRICHMENT_LAYER_GUIDE`, `api/ENRICHMENT_LAYER_API`, ADR-104 | enrichment framework stays public, enrichers move |
| RFC-103 Momentum | `app_momentum` is a public read-model (OG cards); trending surfaces are player and MCP |
| RFC-110, ADR-144, ADR-145 (outbox and delivery seam) | outbox is public kernel; digests and recaps are player |
| ADR-125 (user-scoped UI state) | applies to the operator viewer and the player |
| UXS-017 Shareable cards, `TOKEN-VOCABULARY-CROSSMAP` | operator viewer and consumer app |
| `api/HTTP_API.md`, `architecture/ARCHITECTURE.md`, `POLYGLOT_REPO_GUIDE`, `E2E_TESTING_GUIDE`, `TESTING_GUIDE`, `SERVER_GUIDE`, `DEVELOPMENT_GUIDE` | general docs with player, MCP or enricher sections |
| `GRAPH_VISUALIZATION_GUIDE`, PRD-026, PRD-045, RFC-107, UXS-007, UXS-016 | operator surfaces that describe enricher output |
| RFC-097, RFC-118, `ontology-v3-forward-look` | platform design with MCP and enricher sections |
| PRD-031/032/033, RFC-090, RFC-094 | search docs that describe MCP tools |
| `AI_DESIGN_RUNBOOK`, `OBSERVABILITY_RUNBOOK`, `PILOT_RUN_AND_OBSERVE`, `CORPUS_REPROCESSING` | mention moving surfaces; not yet read |
| ADR/RFC/PRD/UXS indexes, `guides/index.md`, `mkdocs.yml` | entries for every moved document |

---

## Not covered by this plan

- No document above has been read in full; round 1 is a candidate list, not a verdict.
- The sweep regexes are narrow (named paths, "MCP", enricher ids, "learning player",
  "Apple/Google sign-in"); a document that discusses a moving surface without those words is
  missed. The title pass catches some of them, not all.
- WIP notes were only checked by filename for round 1; the 36 the sweep found are not all
  classified.
- No effort estimate in days.
- The list of further IP for Common is deliberately deferred by the operator.
- Repo names for Common and Player are not chosen.
