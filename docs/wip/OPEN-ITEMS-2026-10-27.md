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

## Ingestion, in progress

| # | Item | Status |
|---|---|---|
| 8 | Plan item 13: blind read of 21 sampled episodes (files on the mini, `~/prod-naming-20261027/item13/`) | waiting for its table |

## Eval-harness

| # | Item | Status |
|---|---|---|

## Epic #2097 close-out (plan: `EPIC-2097-CLOSEOUT-PLAN-2026-10-27.md`)

| # | Item | Status |
|---|---|---|
| 14 | #2094 D4 / D5 / D6: measure on today's corpus; fix what remains (pipeline first, then text backfill) | 2026-10-27 measured on 3006 served: D6 = 0 (fixed). D5 STILL SWAPPED (Ground Truths Couric/Topol: SPEAKER_01 says "I'm thrilled to be talking to Dr. Eric Topol" yet named Eric Topol). D4: 43 candidate episodes (named voice <5% talk + unnamed voice >=40%), list `~/prod-naming-20261027/d4_candidates.json` on the mini. *ingestion* owns D4 + D5 (took them 2026-10-27): pipeline first, then a text backfill |
| 15 | Epic activities 6-7: re-derive the #2075 measurements on the repaired corpus; evidence on the issues; *operator* decides closures | *prod* runs the measurements, *ingestion* reads them; waits for 14 (D4/D5 change the numbers) |

## Naming plan, later

| # | Item | Status |
|---|---|---|
| 16a | Plan item 19: split-voice merge audit at 0.8 on new episodes | weeks 1-2 |
| 16b | Plan item 20: voice identity rung promotion — *operator* decides from its table | weeks 1-2 |
| 16c | Plan item 21: transcript naming stays in shadow; data accumulates | ongoing |
| 17 | 0032 (`naming_voices_rederived`) held on its own gate (#2380), outside this plan | held |
| 18 | Housekeeping (plan items 26-28): DGX / step-5 worktrees (path by path, *operator*), scratch corpora, close the arc (`close-arc`) | after the migrations |

## Found along the way, not yet followed up

| # | Item | Status |
|---|---|---|
| 19 | player-api memory: watchdog fired once (rss 3125 MB, 2026-10-25 19:43Z, cache warmer + 6 cold-start requests); under the 6 GB cap; the earlier OOM's cause not established | open |
| 20 | Umami: 0 Player events in both post-deploy checks (10-25, 10-27); whether anyone used the player then is unchecked | open |
| 21 | Upgrade snapshots live in compose-api-1's writable layer: a deploy silently deletes them, and "rollback = this dir" only holds until then; note it in the corpus-migration skill | open |

## Closed (with evidence)

| Date | Item | Evidence |
|---|---|---|
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
