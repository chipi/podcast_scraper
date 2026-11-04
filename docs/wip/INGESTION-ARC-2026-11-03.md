# Ingestion arc, branch `feat/ingestion-arc` (draft, 2026-11-03)

One long branch from main `e1206af1`. It deploys only after the infra and project-structure work
lands, so it takes **only work that changes nothing about deployment**: no compose, Docker,
workflow, deploy-contract or package-layout change, and nothing that counts as finished only once
it runs on prod or the DGX. Code here ships with a later deploy; it adds no step to that deploy's
runbook unless an item says so.

Sources: `HANDOVER-NAMING-2026-11-03.md` (§2, §4), the deleted `INGESTION-ARC-2026-10-19.md`
(`git show 775b09c8^:docs/wip/INGESTION-ARC-2026-10-19.md`), eval-data
`docs/wip/NAME-CONSOLIDATION-ARC-V3-2026-10-30.md`, open GH issues, and the ingestion session's
pointers (2026-11-03). Every issue state below is from `gh issue view` on 2026-11-03.

**Status: draft.** The order and the cut are proposals. The operator has not ranked them.

## How the method applies, per tier

[CORPUS_CHANGE_METHOD.md](../guides/CORPUS_CHANGE_METHOD.md) v0.9 governs every item that changes
what the corpus serves. Per tier (from the ingestion session's D1/D2 practice, 2026-11-03):

| Tier | Record | Measured by | Outcome file / prod hook |
| --- | --- | --- | --- |
| A, read-time fix (A2, A6) | none | repro test first; before/after of the served output on a snapshot | no |
| A, migration code (A3, A4) | yes: `upgrade/` is `content_code`; the gate decides full or light | tests, plus the gate on a clone | only if a migration is built |
| B, forward rule | yes, steps 1–5 before code (hook) | gold S7/S8 (design on dev_v1, val_v2 once) and an old-vs-new replay over a snapshot's inputs | no |
| L, ledger rule | yes, as a full change, though the hook does not force it (`identity/` is not `content_code`) | prediction from the planned code (record s6h), ledger rebuilt on a clone, diffed per group, respell writes counted | no; the respell runs at the next refresh |
| M, instruments | the guide's step 0 ("if a gate is missing, building it is the change") | tests | no |
| C, tooling | per item | tests | no |

Measuring needs a prod snapshot clone, which is not on this machine (see below). The gold cases and
labels are in eval-data (`data/eval/references/silver/naming_speakers_v1/<set>/cases/` and
`labels_v1/`); use those labels, not an older local copy. D1/D2's absolute S7 counts were scored on
labels older than eval-data `1019b1c` (2026-10-24) and were not re-measured; any baseline here is
taken fresh on the current labels.

## Tier A: code, tests and docs; no corpus change, no deploy step

| # | Item | Source | Notes |
| --- | --- | --- | --- |
| A0 | The operator's working rules for a multi-step arc, written into `AGENTS.md` § "Running a multi-step arc" | operator, 2026-11-03 | **done** `249ada02` |
| A1 | Set the 14 rule records to `landed` (d1, s5a–s5d, s5o, s6a–s6h; d2 already was) | ingestion session | **done** `2ee536af`: 6 had steps only D2 covered, now `MISSED:` with the operator's acceptance (2026-11-03). The "#2405 waits" line was not in the handover |
| A2 | `person_profile` empty for `person:robert-caro`, `person:anita-anand` | handover §2.1 | **done** `218e2858`: every CIL reader kept one run per feed (289 of 3,668 bridges on snapshot-prod-20261103); now the central membership rule (3,130). Profiles: robert-caro 0 → 75 quotes, anita-anand 0 → 1,052 |
| A3 | m0023 `_write_text_with_backup` creates backup dirs without `match_corpus_owner` | handover §2.2 | **done** `afcb74d9` (record `a3_text_backup_dir_owner`, light, open until a deploy carries it). Also covers `identity/respell.py`, which uses the same writer on every prod refresh |
| A4 | `write_with_backup` keeps the first backup, so a twice-written file undoes to its pre-first-run state | handover §4 | **done** `36ef4b53` (record `a4_backup_per_before_state`, light): a second backup per replaced state; undo restores only the receipt's before-state |
| A5 | A test that checks each verdict row's `line_sha256` against the served segment | handover §4 | **done** `4439151e`: `scripts/measure/verdict_lines_check.py`; snapshot-prod-20261103: 652 of 652 hashed rows match |
| A6 | CIL readers read each feed's latest run only, so a fix in an older run's bridge is invisible | handover §4 | **done** with A2 (`218e2858`): same cause |
| A7 | #2253 `gi inspect` slices `.adfree.txt` spans from `.txt` | GH, open | **done** `5d9d011b`: the residual case, an absolute `transcript_ref` (7,481 quotes in 52 episodes on the snapshot). 8 quotes remain off their text in the right body: offset drift, not #2253 |
| A8 | #2131 `parse_webvtt` / `parse_srt` never unescape entities | GH, open | **already fixed** `caa9987e` (#2131 still open on GitHub); 0 of 4,950 served raw transcripts carry an entity |
| A9 | Mixed-voice majority is measured in seconds: into `CORPUS_CHANGE_METHOD.md` | handover §4 | **already done**: method guide, operator decisions 2026-10-26/27, item 1 |
| A10 | Review budgets (method v0.9 §4a) are not enforced | handover §4 | **done** `da2ae4be` (method v1.0): the change gate refuses a review from 2026-11-04 with no minutes, past 10 minutes without INCOMPLETE, or with no stated budget; numbers are self-reported |
| A11 | #2180 203 integration tests carry no `integration` marker | GH, open | **done** `84ace6f6`: 239 tests in 14 files marked; `check_test_policy` rule I2 guards it |
| A12 | eval-data: measuring scripts skip episodes with an absolute transcript path; recount D1's gold moves with the corrected script | ingestion session | **done** `bb819aca`: naming_gold_cases and naming_replay_compare read absolute transcript paths (pool 3,009 to 3,130 on the snapshot). The archived D1/D2 scripts keep their code; the D1 recount cannot run (its clones are gone) |
| A13 | `server/routes/artifacts.py:85` lists artifacts with the same one-run-per-feed name rule A2 removed from the CIL readers | found in A2 | **done** `d73856d4`: per-episode membership shared with the CIL readers, cached per corpus change (snapshot: 3,130 per kind, 1.45 s cold, 0.02 s warm). In the viewer, episodes eligible for the 7-day graph: 8 before, 104 after (22 shown, the cap) |

## Tier B: naming rules for new episodes

Each one changes what the corpus serves, so each starts with a change record
(`docs/wip/changes/`, steps 1–5 before code) and is measured on a snapshot clone. Stored episodes
stay as they are unless a migration is built; **a migration adds an apply step to the deploy**, so
any migration here is built and gated on the branch, and its apply waits for the operator.

| # | Item | Source |
| --- | --- | --- |
| B5 | A name in the middle of an episode title is a stated spelling | #2363, open on GitHub. **Shipped** in `fe231664` (2026-10-15), on prod; new episodes get it. The stored episodes from before it are L6 |
| B6 | Patreon supporter shout-outs published as guests | #2364, open on GitHub. **Shipped** in `fc4e0dbd`, on prod for new episodes; the stored ones are in 0043 (class B) |
| B7 | An eponymous host treated as the show | #2136, open on GitHub. **Shipped** for new episodes (self-intro exemption) and m0014 for stored ones; residual: a host who never self-introduces, 0 measured losses |
| B8 | A house-ad read published as a guest | #2123, open on GitHub. **Shipped** for new episodes (ad signatures) and m0016 for stored ones; m0016's gaps are in 0043 (classes A, C) |
| B9 | Rosters wrong beyond a missing name | #2367, open. **Tried and rejected** on replay (`70a041d7`: 42 lost, 2 gained) |
| B10 | Recurrence-scan blind spots | handover §4 (record 0034, round 6). **Closed, no change** (below) |
| B11 | The hardfork-guest-in-record golden failure: cause not established | handover §4. **Closed: cause established** (below) |
| B12 | Stored names on voices today's roster types commercial (ad, promo, shout-out readers) | found in B6 / B8 triage. **Built**: migration 0043 (record `0043_non_participant_names_removed`), 111 voices in 101 episodes, step 8 GO WITH CONDITIONS, ci-fast green on `0758c88cf`. **Apply held** for the deploy (below) |

## Tier L: ledger rules, which rewrite stored files on prod after a deploy, without a migration

0041 is applied, so on prod the ledger refresh after every `enrich` and `upgrade run` respells the
episode files to the current group spellings (`identity/refresh_guard.py`). A deployed change to the
ledger's join or display rules, or to a reviewed list, therefore rewrites stored files at the next
refresh. Each is a corpus change: change record, prediction, and a clone gate counting what the
respell writes (the near-spelling joins report, plus the ledger change-log's respell counts).
**The deploy that carries any of these changes stored data.**

**Decided (operator, 2026-11-04): in this arc, shipped switched off.** The rules are built as a new
named rule mode beside today's (`identity/ledger.py` selects one through `PERSON_MODE =
"cluster_person"`), through the full method, and measured on a snapshot clone: the ledger rebuilt
in the new mode, diffed per group, the respell's writes counted, and the rules scored once on a
fresh unseen split. `PERSON_MODE` stays on today's rules, so the deploy after the infra work
changes no stored file. Switching on is its own one-line change and its own deploy, on the
operator's go after reading the measured diff.

**Scoring set (operator, 2026-11-04):** tiers B and L are scored on a fresh unseen split (seeded,
excluding every existing cases folder, labelled blind by two agents, audited as val_v6 was), built
before the first rule.

| # | Item | Source |
| --- | --- | --- |
| L1 | Roster same-person rules unified or justified (RFC-131 Phase 5), if they change what the ledger reads | #2403, open. **Out of the ledger mode** (record `l_publisher_verified_mode` §1): a roster rule at naming time; its prerequisites (the shadow namers' decision, a voice-level truth set) are unmet |
| L2 | Feed-host spelling rule: a feed's stated host matched by first name and host role (`dara-obrey`, `luisa-lemi`, `john-glenn-hill`) | handover §2.3. **Built, switched off**: record `l_publisher_verified_mode` (V2, V3), step 8 GO WITH CONDITIONS |
| L3 | `known_hosts` as mixed evidence, fixed at the pipeline (`_recurrent_hosts_from_disk`); a ledger change if it feeds the ledger | handover §2.4. **Ledger half built, switched off** (V1: 1,063 of 4,307 name-episodes not in the publisher's text); the pipeline half is its own forward change |
| L4 | Strict-tier losses D1 → D2: `sarah-gonzalez`, `erika-barris`, `david-sanger` | handover §2.5. **Two built, switched off** (`sarah-gonzales`, `erica-barris` take the publisher's spelling); `david-sanger` is a middle-initial scoring artifact, not a ledger error |
| L5 | Check #2056 (duplicate people from spelling variants) against D2; close it or name what is left | GH, open. **Checked**: neither reported duplicate remains on snapshot-prod-20261103 (one pair is one group, the other has one spelling left); closing is the operator's call |
| L6 | A voice's heard spelling joins the person its own episode title names when the roster's stated-name match pairs them (nickname pair, surname within one edit); the title's spelling is shown. On snapshot-prod-20261103, 20 served episodes still carry the heard spelling (13 only it, 7 beside the title's); D2's s5d keeps the two one-episode ids apart. Replay: 27 voices of 3,130 episodes, every one the same person | B5 (#2363). **Built as migration 0044** (record `0044_title_spelling_voices_renamed`), not a ledger rule: 22 of the 27 ids are in no ledger group, so a join cannot rename them. 18 voices in 18 episodes after a blind review (2 refused). Its apply is a deploy step after 0043's |

## Tier M: measuring instruments (from the 2026-10-19 plan's Stage 2; D1/D2 did not do them)

| # | Item | Notes |
| --- | --- | --- |
| M1 | Gold refresh: re-cut the stale gold cases on a new baseline, and draw a new unseen validation set | D1/D2 scored on `labels_v1` |
| M2 | Split S7: same person misspelled vs another person (`gold_insights` counts an ASR spelling of the right person as wrong) | eval-data. **Tried, not kept** (2026-11-04): split by the roster's ASR rule (`_loosely_same_person`), 0 of 392 (dev_v1) and 0 of 299 (val_v7) S7_wrong come out misspelled: the gate's fuzzy tier already counts those as correct, and what is left (`dara-obrey`'s 12) is ASR no rule calls the same person. A split needs a human-read variant list |
| M3 | The loss-ranking script: `origin/wip/loss-ranking`, 85 commits ahead of main, never merged, not read | **Checked, recommend drop** (2026-11-04): `git diff --quiet origin/wip/loss-ranking bc8ab25b` is empty and `bc8ab25b` is on main, so the branch's tree is already in main. Deleting the remote branch is the operator's action (below) |
| M4 | V3 S7, roster vs voices: `quality/name_report.py` measures R5 (`listed_not_heard_entries`); the operator's decision on how a listed-but-unheard name shows was never asked, so no read-time fix exists | **Measured, closed with no read-time change** (below): on snapshot-prod-20261103, 3,889 listed-but-unheard names in 1,983 of 3,130 episodes |

Measuring tiers L, B and M locally needs a snapshot copy; none is on this machine. Source: the
backup repo, `scripts/ops/corpus_snapshot/download_and_verify_snapshot.sh --tag
snapshot-prod-20261030` (about 10 GB; the one D1/D2 were measured on), then `upgrade run --to
2.7.36 --no-snapshot` to bring it to D2's state, or a newer daily snapshot. A working copy is about
8.3 GB, one base plus a gate copy per run. D2's gate chain: eval-data
`data/eval/name-consolidation/d2out/scripts/chain5.sh`.

## Tier C: tooling for later migrations (built here, used on prod later)

| # | Item | Source |
| --- | --- | --- |
| C1 | The code digest covers a migration and most of the package; narrow it | handover §4. **Measured, a decision for later** ([C1-CODE-DIGEST-2026-11-04.md](C1-CODE-DIGEST-2026-11-04.md)): 0043's closure is 610 of 778 modules; following no parent `__init__` imports gives 603, so the breadth is the migrations' own imports. Recommendation: keep it, and re-gate open migrations after a branch's last package change (deploy inventory step 2) |
| C2 | Re-run an applied migration for named episodes (today: calling the class in the api container) | handover §4. **Built**: `upgrade rerun` (record `c2_episode_rerun`) |
| C3 | A naming migration that re-renders text keeps the stored ad cut; recompute it or end with 0030's resync | handover §4. **Closed: measured for 0043, and already guarded.** 0030's read-only plan recomputes today's cut and compares it with the stored one. It reads `ad-free cut plan: 0 episode(s)` on snapshot-prod-20261103 and again on the gate copy after 0043. The cut depends on the transcript text (`workflow/adfree_transcript.build_adfree_artifacts` re-renders the `Name:` markers), so a later naming migration can leave it stale. The guard exists: the clone gate's `verify_all` runs 0030's check after every apply, and on 0043 it is ok with 0 regressions |
| C4 | A gate row "wrong-person voices before / after" | handover §4. **Built**: the gate's `targets` row (`69802a6d`); 0043: 111 before, 0 after |

## The deploy and what follows it (inventory, 2026-11-04)

**Order** (operator, 2026-11-04; in the database session's plan, Common `SEQUENCE.md`, "Then: the
ingestion deploy"): infra's PR B (#2420) merges and deploys first; then this branch, rebased onto
main after it, with the 0043 apply and the targeted reindex; then the first database moment
(#2417 and companions). PR B merges only on the operator's go; the session that owns it messages
the merge sha.

**Before the branch's PR** (local, nothing on origin yet):

1. Wait for PR B's merge sha; rebase onto main. PR B deletes the 4 `tests/integration/server/`
   files this branch marked (`test_app_audio_drift`, `test_app_mcp_routes`,
   `test_app_moments_no_ads`, `test_mcp_oauth_routes`): drop those edits. It also modifies
   `README.md`, `docs/api/CONFIGURATION.md` and `docs/wip/WIP_README.md`: resolve by hand.
2. Re-gate 0043 on the clean tree and rebuild its outcome **after the last package change**: its
   code digest covers 610 modules, so any later edit makes the outcome stale (C1).
3. After the rebase: `make ci-fast` again, the docs strict build, the secrets scan, and the 0043
   landing check; then the operator's go to push and open the PR (ship skill).
4. Tell the database session when the PR merges (it rebases #2417 onto it).

**The deploy** (prod session, through the database session until the database moment):

1. A new private-image release built from the merged public main (prod runs release `64d910c`,
   public `ba5e9a7`); the api, pipeline and player-api images from it.
2. Deploy verify: every container on the new sha (prod-deploy skill), telemetry clean
   (deploy-verify skill).

**After the deploy, in this order** (each on the operator's go):

1. 0043 apply (`upgrade run --to 2.7.37`; rehearsed on a snapshot copy 2026-11-04, hook receipt
   PASS) with the corpus-migration skill: preflight (not within 90 min before the 00:00Z
   nightly, not during it; no backup, reprocess or sweep running; snapshot to
   `/srv/corpus-upgrade-snapshots`, 77 GB free on 2026-11-04 for a 30 GB corpus); the dry run and
   the outcome hook's plan diff against the clone's (sha256 `d167d230…`, 101 episodes); an accept
   file for the `prod_only` rows of episodes newer than snapshot-prod-20261103; the apply; verify;
   the ownership check.
2. The ledger refresh that `upgrade run` ends with respells the episode files (since 0041):
   measured on a clone (record 0043, section 7, "Ledger after 0043"): 0 files written, 3 variant
   joins dropped (652 -> 649), 0 spellings changed. Check prod's change log shows the same.
3. Search index: the incremental `python -m podcast_scraper.cli index --output-dir /app/output` in
   a pipeline-llm run (6g/8g; never `index-two-tier`). It re-embeds an episode only when its
   embeddable text changed; measured on a clone (record 0043, section 7, "Re-embed check"): 100
   of the 101 written episodes re-embed, 0 rows stay stale. Expect `reindexed` 100.
4. Enrichment of the written episodes: the nightly `corpus_enrichment` picks them up, or run
   `enrich` at once in pipeline-llm.
4b. 0044 apply, on its own go, in its own run after 0043's (`upgrade run --to 2.7.38`; a run
   without `--to` with both pending is refused): 18 voices in 18 episodes (record
   `0044_title_spelling_voices_renamed`), its ledger refresh (3 episodes respelled on the clone),
   then the same incremental index for its 18 episodes.
4c. 0045 apply, on its own go, in its own run after 0044's (`upgrade run --to 2.7.39`): 1 voice in
   1 episode (record `0045_episode_voices_corrected`), its ledger refresh (0 files on the clone),
   then the incremental index for that episode.
5. Record 0043 steps 10 and 11 (prod's plan diff, the apply counts, the after-apply scoreboard),
    status `landed`; `config/last_deployed_prod_version.json` brought up to the applied list (it
    lists up to 0037; prod has through 0042).
6. Tell the database session when the reindex has finished and the next nightly ran clean: the
    database day must not overlap the apply, the reindex or the nightly.

**What the deploy changes without a migration** (check on prod after step 5):

1. A2 / A6: the CIL readers use the corpus membership rule; person profiles grow (snapshot:
    `person:robert-caro` 0 -> 75 quotes, `person:anita-anand` 0 -> 1,052). Spot-read both.
2. A13: `/api/artifacts` lists every member episode (viewer's 7-day graph: 8 -> 104 eligible).
3. A7: `gi inspect` on an episode whose GI stores the absolute `transcript_ref`.
4. A3 / A4: the next migration's backups are owned by the corpus owner and kept per before-state;
    nothing to run, read the next apply's backup dir.
5. Tier L's code ships in this deploy switched off (`EVIDENCE_MODE = "extracted"`, record
    `l_publisher_verified_mode`): the ledger builds byte-identical on the snapshot, so no stored
    file changes until its own switch-on deploy. The one default-path change is the diagnostics
    path through `utils.run_relative`. Switching on: set the mode and `RULE_VERSION`, deploy, and
    measure the switch-on again on a fresh snapshot (its step 8, condition 2; the trace by id is
    done, 8b).
6. C2's `upgrade rerun` and C4's gate row ship as tooling: nothing runs them, no stored file
    changes. Each prod rerun is its own corpus change with its own record and go (record
    `c2_episode_rerun`).

**GitHub issues fixed here but still open** (closing is the operator's call; after the deploy
proves each): #2131 (A8), #2180 (A11), #2253 (A7), #2363 (B5), #2364 (B6), #2123 (B8), #2136
(B7, with its residual).

**eval-data** (private repo, this checkout's own clone): branches `naming-val-v7` (the val_v7
split) and `ingestion-arc-l6` (L6's replay) are local; pushing them is the operator's go.

## The arc's open items, closed (2026-11-04)

Operator, 2026-11-04: close them rather than ask. What is left for the operator is three actions
only the operator takes; they go with the push go for the branch's PR.

**For the operator (with the PR's push go):**

1. Tier L switch-on: its own one-line change and deploy (operator's rule of 2026-11-04). Ready:
   the 8 episodes between prototype and build are traced (record `l_publisher_verified_mode` 8b);
   the switch-on is measured again on a fresh snapshot first.
2. Delete `origin/wip/loss-ranking`: its tree equals `bc8ab25b` on main (M3).
3. Push the eval-data branches `naming-val-v7` and `ingestion-arc-l6` to the private remote.

**Closed:**

| Item | How |
| --- | --- |
| C1 | Kept as it is ([C1-CODE-DIGEST-2026-11-04.md](C1-CODE-DIGEST-2026-11-04.md)); re-gating open migrations after the last package change is deploy inventory step 2 |
| C2 | Built: `upgrade rerun` (record `c2_episode_rerun`); measurement on the snapshot in that record |
| C3 | Already guarded: the clone gate's `verify_all` runs 0030's check after every apply (added after 0035, 2026-10-27); on 0043: 0030 ok, 0 regressions |
| C4 | Built: the gate's `targets` row (`69802a6d`); 0043: 111 wrong-person voices in 101 episodes before, 0 after |
| L6 | Built: migration 0044 (record `0044_title_spelling_voices_renamed`); 18 voices renamed to their title's spelling after a blind review; apply after 0043's, on its own go |
| Wrong-person voice found by 0044's review | Built: migration 0045 (record `0045_episode_voices_corrected`); the host's voice takes the host's name and role in episode `a02c151a…`; apply after 0044's, on its own go |
| M3 | Drop the branch (action 2) |
| M4 | Measured, no read-time change. The player's episode page lists KG persons with their roster role. Of the 3,889 listed-but-unheard names: 1,731 shown as host or guest, 429 as mentioned, 1,729 not shown (snapshot-prod-20261103, scratchpad `m4_player_role.py`). A host absent from this episode and a host who spoke but is not placed look the same in the text on disk, so no text-only rule separates them: changing the role would trade one error for the other. Junk names ("CEO Sam Rodriguez") are the descriptor rule's shape (0019) |
| B10 | No change: the recurrence scan is a reviewer's aid for 0034's lists, its misses are listed in record 0034 (round 6), and every listed voice was read by a person; nothing consumes its completeness |
| B11 | Cause established: the snapshot serves the episode's 2026-08-13 run, made before the fix (`metadata_named: []`). The 2026-08-19 run carries `metadata_named: ['MacKenzie Price', …]` but wrote no metadata file for this episode, so it is not served. The input, not the code |
| Scope split | L1 and L3's pipeline half are their own forward changes (operator, record `l_publisher_verified_mode` §9) |
| Deploy windows | The deploy inventory's last step: message the database session at the PR merge, after the reindex and after a clean nightly |

**Mine, scheduled:**

| Item | When |
| --- | --- |
| The episode's other guest-named voice | Episode `a02c151a-dde6-4804-9827-a3df80b8352c` SPEAKER_01: refused by 0045's reader on its first 20 of 109 segments (record 0045, Not covered); needs a reading of the whole voice |
| M1 | After the deploy, on the first snapshot with 0043 applied: re-cut the stale gold cases and draw a new unseen split |
| GitHub issues #2131, #2180, #2253, #2363, #2364, #2123, #2136, #2056 | Closed after the deploy proves each |

## Not in this arc, and why

| Item | Why not |
| --- | --- |
| Diarizer merges promo/ad/clip into a host voice | upstream diarization, DGX |
| #2187, #2251, #2353, #2370, #2284, #2292 | ASR and translation runs on the DGX |
| 0035's reviewer points, 0032's 91 voices, the near-spelling 0-pairs check, `robert-carrow` trace | need prod's served data or a deploy |
| Bare names pooled again (#1801), cross-episode voice identity (#2333), shadow-eval round 2 | wait on the voice-fingerprint shadow validation |
| 12 KG-only persons left out of the bridge | needs the operator's approval first |
| CI interpreter pin (ubuntu-24.04 / Python 3.11.8) | workflow change |
| V3 S9's remaining parts (a daily name report, the R7 alert, promoting a shadow namer) | scheduling and alerting on prod; the ledger-after-ingest step, the ledger state gauge and the regression tests are done |
| #2407, #2408 (index and MCP memory) | no owner recorded; the operator decides whether they belong here |
| Triage-fleet escalations, mobile, player, translation epic issues | not ingestion |

## Not covered by this draft

- About 200 open issues were read by title only. Only the ones above were checked one by one.
- `origin/wip/loss-ranking`'s content was not read; its tree equals `bc8ab25b` on main (M3).
- Eval-data's 2026-10-22 naming notes and `MIGRATION_ARC_PLAN` / `MIGRATION_INVENTORY` were not
  re-read; parts may still be open.
- No item has a size estimate.
- The snapshot download (about 10 GB, plus working copies) was not tried. Free disk: 1.2 TB
  (`df -h /work`, 2026-11-03).
- No naming split is fully unread (eval-data `naming_speakers_v1/README.md`): val_v1, val_v2 and
  val_v3 were read; val_v6 was scored at the 2026-10-25 release gate and 7 of its voices were
  panel-read; val_v4 and val_v5's read state is not checked. A tier-B or tier-L rule needs a fresh
  split (seeded, excluding every existing cases folder, labelled blind and audited), or val_v6
  once, totals only, with that caveat in the record. No unseen feed exists.
