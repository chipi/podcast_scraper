# Naming release: deploy and after, in time order (2026-10-25)

What happens from the merge of PR #2393 (merge plan step 7) to the end of the naming arc, in the
order it has to happen. The merge plan is `NAMING-BRANCH-MERGE-PLAN-2026-10-24.md`; its step 5
result is the evidence this deploy rests on.

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

16. **0034, the talk floor on served episodes** (`m0034_talk_floor_names_removed`; measured on the
    10-17 snapshot: 84 bad names off, 7 right lost). Not started: no frozen list or verdicts exist.
    On a fresh prod snapshot (the day-1 backup): `find_changes`, freeze the list, a fresh
    reviewer's verdicts, the outcome table, the operator's go, apply with the `corpus-migration`
    skill, then verify every derived surface (GI, KG, bridge, search rows). *me, reviewer, operator*
17. **0035, voice-rung names on fingerprinted served episodes** (`m0035_voice_identity_names`; 413
    of 2,939 on 10-23, growing): the same, **after 0034** (0035 leaves voices under the floor to
    0034). *me, reviewer, operator*
18. **What the served corpus does not get:** names on episodes served before the deploy stay as
    they are except where 0034 / 0035 reach; v39-v55 act on new episodes.

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
28. Close the arc: history entry, NEXT_STEPS, decisions log (`close-arc`). *me*

## Not covered by this plan

- Item 2's naming-time estimate assumes the relabel difference carries over to the full
  pipeline; item 10 is a proposal, not agreed.
- A correction: an earlier draft listed "missing `pipeline_stage` events" as a defect. They are
  shipped; since 2026-10-15 they are stored as fields with no `_msg`, so a text search finds
  nothing and `event_type:pipeline_stage` finds them (prod; health check fixed in `06e3d3ae5`).
- Epic #2097 was not read item by item (item 15 does that).
- No other post-deploy checklist for this arc was searched for beyond the merge plan and #2097.
