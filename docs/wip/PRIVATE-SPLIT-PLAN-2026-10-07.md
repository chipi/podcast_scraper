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

## 1. In-repo refactor (ADR-158 step 2)

Everything stays in this repo and every test keeps running. Each slice is its own PR. Slices that
touch `Makefile`, `.github/workflows/`, `compose/` or `mkdocs.yml` wait for #2138.

| Slice | What | Size signal | Waits for #2138 |
| --- | --- | --- | --- |
| A | Rename the ten corpus read-models out of `app_` (`app_kg_view`, `app_gi_view`, `app_relational_view`, `app_momentum`, `app_catalog_cache`, `app_corpus_access`, `app_content_source`, `app_slugs`, `app_artwork`, `app_kg_index`); make `_role_of`, `_aggregate_role`, `_ROLE_RANK` public | rename + import edits | no |
| B | Extension interface: protocol, entry-point loader, a fake app in tests that exercises every hook | new module + contract tests | no |
| C | Convert the seams one at a time: `app.py` router mounts, cache warmer, digest-health metrics, `scheduler.py` digest dispatch, `capability_audit.py` checks, account deletion, `player_client_health`, move `filtered_entity_signals` into the platform | 8 seams | no |
| D | Sign-in providers: `GoogleProvider` and `AppleProvider` behind the registry; mock stays | small | no |
| E | Enricher registry: register `insight_density` and `guest_coappearance` publicly, the rest from the Common package; public readers of named enricher output (`og/build.py`, `feed_signals.py`, `cil_queries.py`, others) treat it as optional | 18 importing modules to check | no |
| F | Separate top-level packages in `src/`: public kernel, `common` (providers, `mcp/`, `podcast_obs/`, enrichers), `player` (the 50 player-only modules plus the player logic currently pulled in transitively) | directory moves | no |
| G | Per-app data folders: `data/apps/<app>/users/<id>/…`, prod migration with dry run, verify, undo | migration | partly (backups, compose volumes) |
| H | Split the tests: 105 test files touching `app_*`, plus enrichment and MCP tests, assigned to the package they test | classification | no |
| I | Player make targets (42 by name), player CI jobs in `python-app.yml`, `stack-test.yml` image entries | config | **yes** |

Still to trace before slice F: which platform entry point pulls `app_recap` and `app_stats` into
the platform-needed set.

---

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
