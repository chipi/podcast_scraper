# Naming release: deploy and after, in time order (2026-10-25)

What happens from the merge of PR #2393 (merge plan step 7) to the end of the naming arc, in the
order it has to happen. The merge plan is `NAMING-BRANCH-MERGE-PLAN-2026-10-24.md`; its step 5
result is the evidence this deploy rests on.

## Status, 2026-10-27 13:00Z

**Deployed: `sha-2914e56` (main `2914e562e`) at ~08:30Z** (deploy-all-prod run 40473459447;
`prod-deploy verify` VERIFY PASS, 10 / 10). **Item 18 done:** 0034 (152 episodes), 0035 (99) and
0036 (34) applied one at a time with `--to`, each receipt equal to its dry run and the measured
clone, plan diff 0 each; corpus at 2.7.30, 0 pending. 0035's renames exposed two host-read sponsor
blocks to today's ad cut (0030's verify failed on 2 episodes); fixed forward by re-running 0030's
own apply on those 2 (operator: no restore, no undo). Final `upgrade verify`: every migration OK
except 0025 on exactly the 5 known episodes. Ownership PASS; `enrich --only
insight_density,insight_sentiment` 248 episodes ok; incremental `index` reindexed 248. Upgrade
snapshots deleted (operator). Items 12 and 14 done (night 1 clean, timing inside the baseline);
item 13 with the naming session; item 15's findings and close-out plan:
`EPIC-2097-CLOSEOUT-PLAN-2026-10-27.md`. Item 22 done (2,950 of 2,980 episodes have archived
audio). Open items across everything: `OPEN-ITEMS-2026-10-27.md`.

## Status, 2026-10-25 20:00Z

**Deployed: `sha-89590c6` (main `89590c65a`) at 19:43:59Z** (deploy-all-prod run 40226957536;
`prod-deploy verify` VERIFY PASS, 10 / 10 app containers). Items 1-10 are done (marked below);
night 1 (item 11) runs at 00:00Z on the new code. The candidate moved once: the merge commit
`bc8ab25b9` failed main's full `test-integration` (3 declarations the release owed), so it got
no images; the hotfix `89590c65a` (pushed to main directly) fixed it and is what is deployed.

**Owners:** *me* = the naming session; *prod* = the prod session; *operator* = a decision or a go
only the operator gives. Every corpus change goes through `docs/guides/CORPUS_CHANGE_METHOD.md`
(outcome table, fresh reviewer, the operator's go).

**What the deploy changes in prod** (`config/profiles/prod_dgx_full.yaml`, deployed `b60a89e` ->
the merge): the talk floor at 5% (`speaker_name_min_talk_share`), the E1d / S30s rescues
(`ladder_rescue_rules`), transcript naming and the voice rung in shadow, speaker detection at
temperature 0, the split-voice merge at 0.8; and in code the ladder changes v39-v55. Also in the
image, already on main before the merge: pillow 13 (`faeff5aa3`, `2eb7dd9ca`) and the memory
watchdog; and from the branch, preshed < 3.0.14 and filelock 4.2.1.

## Does anything have to run between the deploy and the first nightly?

**No corpus change has to.** New code reads an older corpus with a warning, not a refusal
(`corpus_version.assess_corpus_version_compat` returns a warning; CI's "N-1 corpus compat" check,
current code against the previous fixture, passed on `8e3c3af2e`). 0034 and 0035 act on episodes
already served; the nightly writes new episodes with the new code. They do not depend on each
other, and neither has to precede the nightly.

## Before the deploy (today, after the merge)

1. **Done.** CI green on main at `89590c65a` (every workflow success), Stack test published
   `sha-89590c6`. *me*
2. **The "before" numbers for the nightly** (prod, 2026-10-25, from the job registry,
   VictoriaLogs and VictoriaTraces). Last 7 nightlies that ran (10-24 held; 10-16 / 10-17 failed
   in ~4 min, not counted):

   | night | start -> end (UTC) | new episodes | image |
   | --- | --- | --- | --- |
   | 10-18 | 00:00 -> 23:53 | 145 | ded80e0 (the 13-new-feeds night) |
   | 10-19 | 00:05 -> 02:43 | 13 | ded80e0 |
   | 10-20 | 00:00 -> 04:05 | 25 | e151a5c |
   | 10-21 | 00:00 -> 03:20 | 21 | b60a89e |
   | 10-22 | 00:00 -> 06:44 | 27 | b60a89e |
   | 10-23 | 00:00 -> 03:45 | 23 | b60a89e |
   | 10-25 | 00:00 -> 04:15 | 27 | b60a89e |

   - Per episode (median / p90 of the traces): `episode.transcribe` 248-458 / 459-799 s,
     `episode.metadata` 283-428 / 411-722 s, `episode.process` 10-12 / 11-29 s.
   - **Naming has no timing of its own** (its `pipeline_stage` events carry no `duration_s`, and no
     span covers it).
   - **Estimate, not verified:** the step-5 relabel took 71 min for the branch against 19 min for
     deployed code on the same 212 episodes (+52 min, about 15 s per episode; cause not
     confirmed, the shadow naming passes presumably). If that carries over to the full pipeline, a
     normal nightly (20-27 new episodes) gets about 6-7 min longer.
3. **Done** (prod, at preflight). **Telemetry baseline:** `deploy-verify snapshot`. *prod*
4. **Done:** `prod-deploy preflight 89590c65a` PASS (4 CI PASS, 5 images `sha-89590c6` HTTP 200).
   *prod*

## The deploy (T0)

5. **Done,** 19:43:59Z, on the operator's go to prod. **Dispatch and approve** (`prod-deploy
   dispatch` / `approve`). *operator, prod*
6. **Done:** VERIFY PASS at 19:44Z, 10 / 10 app containers on `sha-89590c6`. *prod*
7. **Done:** the deploy run's operator and player live smokes, and
   `scripts/ops/post_deploy_smoke.sh` (6 / 6 probes). **Not** `scripts/smoke/post_deploy.py`: it
   POSTs an ingest job into the corpus (the codespace smoke), so it is not a prod smoke. *prod*
8. **Ran at 19:54:30Z** (prod reports only a failure; none reported). **Telemetry gate:**
   `deploy-verify compare --release sha-89590c6`, then `o11y-review` over the window; every new
   GlitchTip issue looked at. *prod / me*

## Between the deploy and the nightly (same evening)

9. **Confirmed:** nothing corpus-side. Prod's `upgrade status` on `sha-89590c6`: 2.7.27 current =
   target; the registry ends 0030, 0031, 0033 (0032 is held unregistered, #2380; 0034 / 0035 are
   not registered until their lists and verdicts exist).
10. **Done:** a one-episode canary on `sha-89590c6` before the deploy (19:24-19:36Z, EXIT 0, 0
    ERROR / Traceback): a throwaway container writing to a temp dir, corpus mounted read-only,
    `prod_dgx_full`. Its sidecar carries the `transcript_naming` / `voice_identity` shadow keys;
    2 / 2 voices named. The talk floor, v54, the rescues and the split merge had nothing to act
    on there, so night 1 is their first prod exercise (item 12). *prod, me*

## Night 1: the first nightly on the new code (00:00Z)

11. **Watched live** by prod: duration against item 2's baseline, failures, naming errors. If it
    runs far past its window, the operator decides (stop it, or let it finish). *prod, operator*

## Day 1 (the morning after)

12. **The new rungs ran:** the night's speaker-diagnostics sidecars carry `talk_floor`,
    `llm_third_person`, the rescue and split-merge steps where they fire, and the shadow keys
    `transcript_naming` / `voice_identity`; no naming exceptions. *me (read-only)*
13. **Served names on the night's episodes, wrong-person names first:** a sample read against the
    transcripts. *me*
14. **Nightly time against item 2's baseline** (night 1's start -> end and per-episode
    `episode.metadata` / `episode.transcribe` medians). Far over the 6-7 min estimate: the operator decides whether a shadow pass
    comes off (a profile change, its own small deploy). *me, operator*
15. **Epic #2097 carry-over, a read-only pass:** the #2082 transcript-pairing audit and repairs,
    the #2100 rediarize of 51 episodes, the 51 text-prefixed publisher transcripts. The epic records
    no "done" for these; establish what is still open and report it, nothing more. *me*

## Days 1-3: the two corpus changes, one at a time, never during a nightly

**Schedule:** 0034 on day 1 (2026-10-26), in the day after items 12-14 show night 1 clean;
0035 on day 2, after 0034 is applied and verified. Each: about half a day to the operator's go;
applied in daytime, never inside a nightly (00:00 to about 04:30Z). Each is registered in
`upgrade/registry.py` only once its frozen list and verdicts exist (whether prod's mounted-source
migration tooling needs a new image for that is checked on day 1). 0032 stays held under its own
gate (#2380), outside this plan.

16. **Status 2026-10-27: 0034, 0035 and 0036 are built, measured, reviewed and on main**
    (`2914e562e`; records `docs/wip/changes/0034_talk_floor_names_removed.md`,
    `0035_voice_identity_names.md`, `0036_bridge_identities_restored.md`). Each has its frozen list
    and verdicts, a step-8 GO WITH CONDITIONS with the conditions closed, an outcome bound by sha256
    in step 8, and the operator's go on the 12 step-9 points (2026-10-26, recorded in the records).
    Measured on snapshot-prod-20261025: 0034 removes 175 names in 152 episodes; 0035 renames or
    unnames 113 voices in 99 episodes; 0036 restores 101 bridge identities in 34 episodes and
    rebuilds 16 damaged bridges. Every undo cycle on the clones PASS.
17. **Deploy the image that carries them** (`sha-2914e56`): `prod-deploy` preflight, dispatch,
    verify (as items 5-9). The migrations run from mounted source (prod tooling), but the ledger
    target versions 2.7.28 / 2.7.29 / 2.7.30 are this code's. *prod*
18. **Apply, after the deploy, in daytime, never inside a nightly (00:00 to about 04:30Z), with the
    operator's go given to the prod session directly**, using the `corpus-migration` skill for
    each (preflight, the CLI's dry run, gated apply, verify, ownership check, undo recorded). *prod*
    1. **0034 alone** (`--to 2.7.28`): dry run first; the plan must equal the outcome's (0
       differences; prod has episodes the snapshot lacks, each difference accepted with a cause by
       the operator). After the apply: verify OK, and `https://iono.fm/e/1728203`'s files equal the
       clone's (0035 also writes that episode).
    2. **0035 alone** (`--to 2.7.29`): dry run with 0 plan differences against its outcome; apply;
       verify OK.
    3. **0036** (`--to 2.7.30`): dry run, apply, verify OK.
    4. **`upgrade verify` (the 2.7 runbook's step 5): 0025 is expected to flag exactly 5 episodes,
       accepted as known by the operator (spelling only, one person id each, no duplicate):**
       `998e8fc4-c6fc-11f0-b555-4bf8880724ca`, `https://iono.fm/e/1729872`,
       `https://iono.fm/e/1728106`, `Buzzsprout-19147714`, `Buzzsprout-19066916`. Any other 0025
       flag, or any other migration failing verify, is a stop.
    5. **Derived surfaces, for the episodes the three wrote** (each apply's `details.episodes`):
       the sidecar refresh `enrich --only insight_density,insight_sentiment`, then an incremental
       `index` for those episodes only. No full rebuild, no LLM, no DGX.
    6. **Undo, if needed, in reverse:** 0036, then 0035, then 0034. After an undo of 0035 reset the
       ledger `version` to 2.7.28 by hand (the runbook has no such step; the undo removes the
       ledger entry but leaves the version).
    7. **Report back:** each apply's receipt line and episode count, the verify output, the plan
       diffs, and the derived-surface runs. *prod*
**What the served corpus does not get:** names on episodes served before the deploy stay as
they are except where 0034 / 0035 / 0036 reach; v39-v55 act on new episodes.

## Week 1-2: measurements that need the new code's own output

19. **Split-voice merge audit at 0.8** on the new episodes' merges, before 0.7 is considered
    (ladder doc v50). *me*
20. **Voice rung promotion:** its shadow answers on new episodes against the ladder; on step 5,
    filling only unnamed voices: 33 right / 5 bad, and 5 wrong names repeating within a feed (the
    error-spreading check). The operator decides from that table. *me, operator*
21. **Transcript naming (LLM pass) stays in shadow** (operator); its shadow data accumulates
    (step 5: filling unnamed 103 right / 43 bad; publishing everywhere 437 / 75 against the
    ladder's 360 / 51). *-*
22. **Archived audio on prod:** how many episodes have it (the voice rung's reach on old
    episodes). *prod*

## Later naming rounds (not scheduled)

23. The "stated host answered as guest" refusal hides mostly right names in both code versions
    (step 5): a finer rule.
24. Two co-hosts typed as guests; c084 / v224; the surname-only refusal (y007). *(deferred by the
    operator)*
25. The resolver prompt's "says this ABOUT them" evidence line (an LLM-side change: a DGX
    measurement).

## When the migrations are done: housekeeping

26. The DGX and step-5 worktrees (`podcast_scraper-0029-dgx`, `-dgx-new`, `-s5-deployed`,
    `-s5-branch`, `-s5-v54`), each path approved. *operator, me*
27. Scratch corpora (`s5c_*`, `dgxcorpus`; `snap1016` once 0034 / 0035 no longer need it). *me*
28. ~~Close the arc (`close-arc`)~~: not applicable: this repo has no `docs/history`, decisions log or NEXT_STEPS (operator, 2026-10-27: drop close-arc here); the arc's record is the change records, the method doc and the naming handovers.

## Not covered by this plan

- Item 2's naming-time estimate assumes the relabel difference carries over to the full
  pipeline; item 10 is a proposal, not agreed.
- A correction: an earlier draft listed "missing `pipeline_stage` events" as a defect. They are
  shipped; since 2026-10-15 they are stored as fields with no `_msg`, so a text search finds
  nothing and `event_type:pipeline_stage` finds them (prod; health check fixed in `06e3d3ae5`).
- Epic #2097 was not read item by item (item 15 does that).
- No other post-deploy checklist for this arc was searched for beyond the merge plan and #2097.
