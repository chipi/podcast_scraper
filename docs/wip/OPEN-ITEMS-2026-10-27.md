# Open items, all current activities (started 2026-10-27)

One list of what is open across the naming release, prod, the tenants and the epics, kept current
as items close. Update the status column and the "Closed" table in place; add new items at the end
of their section with the date. Evidence goes next to every "done".

Owner: *prod* = the prod session; *ingestion* = the naming / ingest session; *eval* = the
eval-harness session; *operator* = a decision only the operator makes.

## Naming release: what is left (prod)

| # | Item | Status |
|---|---|---|
| 6 | Update `NAMING-RELEASE-POST-DEPLOY-PLAN-2026-10-25.md` status with today's results; commit it with this file and `EPIC-2097-CLOSEOUT-PLAN-2026-10-27.md` (piled for the final push) | open |

## Confirm tomorrow (2026-10-28)

| # | Item | Check |
|---|---|---|
| 7 | Corpus-scope enrichers stale after the migrations (grounding_rate, guest_coappearance, temporal_velocity, topic_cooccurrence_corpus, topic_theme_clusters, person_web) | after the 10-28 nightly: `corpus_status` (MCP) shows none stale for `corpus_artifacts_newer` |
| 9 | PLAYER-1L fix (ca4ce2843, 9ee56fcd3) | after the next player deploy: its live smoke green, no new PLAYER-* audio issue in GlitchTip, player events carry release sha-<7> |
| 9a | Alloy `app` labels (6370dde6b) | after `deploy-config.yml`: `app` on the five podcast file sources and the three api scrape targets |
| 9b | GlitchTip DSNs in the app containers | checked 2026-10-27 (docker inspect, values not printed): compose-api, player-api, operator-api via secret files (`/run/secrets/podcast_sentry_dsn_*`, exported by the shim); player-mcp, player-obs via env, set; board-summarizer via secret. **player-digest-scheduler has no DSN at all** (compose declares none): its errors never reach GlitchTip. *operator*: add one or accept |

## Ingestion, in progress

| # | Item | Status |
|---|---|---|
| 8 | Plan item 13: blind read of 21 sampled episodes (files on the mini, `~/prod-naming-20261027/item13/`) | 2026-10-27 read by two blind readers: 4 voices in 3 episodes carry another person's name (`substack:post:216929530` swapped pair, 89 insights; `d99b4720-…` SPEAKER_04, 14; `https://iono.fm/e/1732482` SPEAKER_07, 3). Evidence in the private eval repo, `data/eval/naming-night1-20261027/`. Served fix = item 22 (C) |

## Eval-harness

| # | Item | Status |
|---|---|---|

## Epic #2097 close-out (plan: `EPIC-2097-CLOSEOUT-PLAN-2026-10-27.md`)

| # | Item | Status |
|---|---|---|
| 14 | #2094 D4 / D5 / D6: measure on today's corpus; fix what remains (pipeline first, then text backfill) | 2026-10-27 measured on 3006 served: D6 = 0 (fixed). D5 STILL SWAPPED (Ground Truths Couric/Topol: SPEAKER_01 says "I'm thrilled to be talking to Dr. Eric Topol" yet named Eric Topol). D4: 43 candidate episodes (named voice <5% talk + unnamed voice >=40%), list `~/prod-naming-20261027/d4_candidates.json` on the mini. *ingestion* owns D4 + D5 (took them 2026-10-27): pipeline first, then a text backfill. Plan and status: items 22 (D5 = A) and 23 (D4 = B) |
| 15 | Epic activities 6-7: re-derive the #2075 measurements (#2093, #2095, the #2075 PR) on the repaired corpus; evidence on the issues; *operator* decides closures | moved to *ingestion* 2026-10-27 (operator): run from snapshot-prod-20261027 (11:04Z, after 0034-0036 + the 0030 re-run), no prod access; waits for 14 (D4/D5 change the numbers) |

## Main: fixes landing as hypotheses, one validation pass at the end (*ingestion*)

Operator, 2026-10-27: keep landing the fixes; park every re-measure / gate / review until the scope
for main is settled, then validate all at once. Nothing below is validated yet.

| # | Item | Status |
|---|---|---|
| 22 | A. D5 (Ground Truths `substack:post:165726942`, names swapped). Cause from the diagnostics: `known_hosts` seated the feed host on the voice acting as host; no third-person guard covers `known_hosts` seating (guards cover LLM names, roster guests, the 5% floor). Plan: (1) code: apply the third-person check to `known_hosts` seating, change record steps 1-5; (2) served: a small text-only migration (~0037) swapping the pair back, plus item 8's 4 voices (C). Done when: a replay shows no host name on a voice that speaks of the host; on prod both voices and C's 4 carry their own names | in progress |
| 23 | B. D4: classify the 43 candidates read-only (floor covers it / genuine / wrong name / undecidable); real wrong names join 22's migration | after 22's code fix |
| 24 | Local commits on main, not pushed: 2b035c180 (gate runs full `upgrade verify`, verdicts completeness test, method rules, ad-cut resync step), 9153ca153 (ledger + outcome files chowned to the corpus owner; undo rewinds `version`). They move the 0034-0036 code digest, so the landing test stays red until the final pass rebuilds the outcomes | push waits for the operator |
| 25 | Final validation pass: gates on the snapshot clone (0034 -> 0035 -> 0036 -> new), rebuild outcomes, ci-fast + test-integration, push, then delete the clone (`/Users/claude/projects/remeasure-20261027`, 9 G) | after 22-23 |
| 26 | Known and accepted: `upgrade verify` flags 0025 on the 5 known episodes, spelling only; the shared spelling-precedence rule is debt B10 | accepted |

## Naming plan, later

| # | Item | Status |
|---|---|---|
| 17 | 0032 (`naming_voices_rederived`) held on its own gate (#2380), outside this plan | moved to *ingestion* 2026-10-27: assess still-needed vs retire after 0031/0033-0036; *operator* decides on the assessment |

## Found along the way, not yet followed up

| # | Item | Status |
|---|---|---|
| 19 | player-api OOM-killed at the 6 GB cap 3 times since the 10-25 deploy (kernel: 10-26 13:46:07 and 13:46:31, 10-27 07:13:37, anon-rss ~6.27 GB), during real app use (beta traffic; an app-launch burst); the 1 s watchdog logged nothing before any kill, so the spike is sub-second or the watchdog is starved; the killing request never completes, so the access log cannot name it. Ruled out: the index pool warm-up, `PUT /interests` itself. Next: reproduce off prod (dev engine, snapshot-prod-20261027, same image + 6 GB cap, replay the launch burst endpoint by endpoint) — route awaiting the *operator* | *prod* |
| 21 | Upgrade snapshots live in compose-api-1's writable layer: a deploy silently deletes them, and "rollback = this dir" only holds until then; note it in the corpus-migration skill, plus ~corpus-size free disk per snapshot (30 GB now) | moved to *ingestion* 2026-10-27 |
| 27 | deploy-verify's Umami check reported 0 events after the deploys: homelab found traffic sparse (first Player event 15 min / 2 h after each deploy) and a real defect (a failed psql query read as 0 events); 0 now INCONCLUSIVE until the pre-deploy quiet gap has passed | *homelab*: fixed locally 2e41270 (agentic-ai-homelab), push waits for the *operator* |

## Closed (with evidence)

| Date | Item | Evidence |
|---|---|---|
| 2026-10-27 | Migration leftovers removed (operator: whole dir; naming session agreed) | `/app/output/.podcast_scraper/upgrade-backups/` (1.7 GB) deleted 14:2xZ after confirming `snapshot-prod-20261027` (11:04Z, run 40491913697) holds it off-box; kept ledger, receipts, upgrade-outcomes; `upgrade status` 2.7.30 up to date |
| 2026-10-27 | Housekeeping: 5 DGX / step-5 worktrees on the mini | `git worktree remove --force` as claude for podcast_scraper-0029-dgx, -dgx-new, -s5-branch, -s5-deployed, -s5-v54 (0 open files first); scratch corpora already deleted by the naming session; close-arc is the naming session's |
| 2026-10-27 | Umami on the player (operator: beta users are all on Android) | Homelab: Player site events from `android-app`, e.g. 10-26 89 page views / 193 custom events |
| 2026-10-27 | 0036's snapshot deleted (operator go) | `/app/.corpus-upgrade-backup-2.7.29-20261027T101817Z` removed; no snapshot left; 75 GB free |
| 2026-10-27 | #2082 on served data (operator: no action needed) | all 23 audit hits superseded runs; 0 served |
| 2026-10-27 | #2100 reconcile (operator: no action needed) | 102 served episodes in the 3 feeds all with correct speakers; DGX-DEFERRED row removed |
| 2026-10-27 | evals-inspector OOM (eval fix c91d48b0, run 40492927985) | 12:55Z: restarts 0, OOMKilled false since 11:53Z start, 136 MiB / 256 MiB, 0 kernel OOM kills since 11:53 |
| 2026-10-27 | Final `upgrade verify` (after 0036 + 0030 re-run) | 34 OK incl. 0030/0035/0036; only FAIL = 0025 on the 5 known files |
| 2026-10-27 | Ownership after the migrations | 2 + 5 paths chowned to uid 1000; OWNERSHIP PASS |
| 2026-10-27 | Derived surfaces | enrich insight_density/sentiment 248 ok, 0 failed; index scanned=3004 skipped=2756 reindexed=248, EXIT=0 |
| 2026-10-27 | Plan item 18.7 report to ingestion | sent with receipts, verify, plan diffs, derived-surface runs |
| 2026-10-27 | #2097: the 51 text-prefixed publisher transcripts | 0 across 3006 served (via each transcript_file_path) |
| 2026-10-27 | #2094 D6 (placeholder published as a speaker) | 0 in 3006 served rosters |
| 2026-10-27 | sha-2914e56 deployed | run 40473459447 success; `prod-deploy verify` VERIFY PASS 10/10; post_deploy_smoke green |
| 2026-10-27 | 0034 applied (152 episodes, 175 names) | receipt = dry run = clone (21 counters); verify OK |
| 2026-10-27 | 0035 applied (99 episodes, 113 voices) | receipt = dry run = clone (27 counters); verify OK; 0025 = the 5 known |
| 2026-10-27 | 0036 applied (101 identities, 34 episodes, 16 bridges) | receipt = dry run = gate line; 2.7.30, 0 pending |
| 2026-10-27 | 0030 re-run on 2 episodes after 0035's renames exposed two host-read sponsor blocks | guarded dry run = expected 2 episodes + 8 counters; 14 files written |
| 2026-10-26 | Night 1 on the new code | 14 episodes, 0 errors, 0 naming exceptions; per-episode timing inside baseline |
| 2026-10-27 | Plan item 12 (new rungs ran) | naming_shadow_report JSON on the mini (sha256 07b33b17…) |
| 2026-10-27 | Plan item 14 (night timing) | traces: metadata p50 305/460/324 s, transcribe 321/404/331 s |
| 2026-10-27 | Plan item 15 (read-only #2097 pass) | findings → items 11-15 |
| 2026-10-26 | Plan item 22 (archived audio) | 2,950 of 2,980 episodes; 18 size mismatches, 30 none |
| 2026-10-27 | player-mcp traces on the new release | `mcp.tool.corpus_status` trace at 10:42:45Z, sha-2914e56 |
| 2026-10-27 | Homelab: corpus-migration `to_version`; advisor on fable + docs realigned | agentic-ai-homelab `3290b0b`, `9b4d806`; CI skills-index + docs success |
| 2026-10-26 | Beta testers added to the player allowlist | `/app/appdata/access_policy.json` 12 addresses; audit record |
| 2026-10-27 | Prod SSH access | key re-added to the agent; deploy@ and root@ ok |
| 2026-10-26 | eval-harness live on evals.humancopilot.eu | deploy 40286075873; edge 18 site files; probes 200 |
| 2026-10-27 | Nightly CI | `gh run list --workflow nightly.yml`: success 10-26 and 10-27 |
