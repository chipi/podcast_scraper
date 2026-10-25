# Naming branch: plan to merge (2026-10-24)

The operator approved this plan on 2026-10-24. Branch `wip/loss-ranking`, PR #2393. The eval repo
branch `feat/naming-voice-eval` merges to its master alongside it.

Merging to main is what the next deploy ships. Once merged, the prod profile's naming changes go
out with the next images:

- the 5% talk floor;
- the E1d / S30s rescues;
- transcript naming in shadow mode;
- the voice rung in shadow mode.

So "deploy only if better" is settled before the merge, not after it.

## Steps

1. **Done, 2026-10-24: y007 / y008 traced offline.**
   - **Cause:** main's rule, not this branch's. The third-person guard refused the host's name
     on the host's own voice ("On <platform>, I'm at <name>"), and the two-voice complement or the
     swap then moved it onto the guest's voice.
   - **Fix:** ladder doc v39, with a failing test first.
   - **Replay:** old code from git, over the 2026-10-16 prod snapshot and the R1-GI / R4 copies.
     - The old code agrees with every recorded verdict.
     - 2 pairs change on the snapshot (host voices; served names already right) and 2 on each
       run copy (the two broken episodes).
     - No new refusal; 3 mutants killed.
     - The gate script is `scratchpad/replay_tp/gate.sh`, not committed.
   - **What it fixes (the independent review, 2026-10-24):** y008 at the resolver
     step (the roster after it was not replayed). In y007 the host is now accepted, but the guest's own name is still refused: her
     surname alone occurs in her text inside another person's name. So she is left unnamed instead
     of carrying the host's name. The surname-only refusal is left unchanged: an independent
     advisor found the y007 shape in 0 of 124 snapshot refusals, a narrow fix would restore about
     3-5 names and needs guards against titles, middle names and nicknames, and today the guard
     only hides names, it never publishes a wrong one. The operator doubted its value; not decided.
   - **S7 corrected:** R4 / R5 ran with GI switched off, so the first R4 / R5 S7 readings
     measured R1-GI's names on stale offsets. S7 now places each quote by its text and scores
     the run's own roster (`--namer ladder`). See the S7 change doc.
2. **Ship the branch:**
   - rebase onto main;
   - run `make ci-fast` once;
   - push only with the operator's go;
   - watch CI to green.
3. **Ship the eval repo:** secrets scan; with the operator's go, push and merge to master (the
   scorer's voice map, the S7 readings, the DGX block plan).
4. **Done, 2026-10-25: a fresh held-out set (val_v6), added 2026-10-24 after run A:** val_v3 was
   read while building the episode host statements, and val_v5 was used to choose the split-voice
   threshold (run A's fingerprints). The release score needs labelled episodes nobody has looked
   at: picked, labelled blind, and not opened until step 5 scores them.
   - **val_v6** (eval PR #11): 80 blind cases from `snap1016/corpus` (seed 20261025), 27 feeds,
     labelled by two Sonnet agents, audited by two Opus agents (205 labels, 27 changes); 364
     voices, 205 named and scored.
   - **gate5_v1** (eval PR #12): 51 cases picked by condition: 30 where v51 can act, 1 where the
     E1d rescue can act, 20 random (nothing should move); same labelling and audit (147 labels,
     20 changes).
   - **Limit:** no unseen feed exists in prod (all 68 eligible feeds have earlier cases; the 10-24
     snapshot has the same 81 feeds), so the gate measures new episodes of known shows.
5. **Release gate on a prod-snapshot clone:** a before/after outcome table of what the deploy
   changes on served names, with S7 (insight-weighted) beside the per-voice score, on val_v6.
   - **Relabel, not rediarize:** run A rediarized from the publisher's current files (this corpus
     has no audio archive); on the episodes whose audio had changed since ingest, wrong or
     non-participant names went 22 -> 89 (unchanged audio: 17 -> 12). A rediarize in the gate
     uses archived audio only, or it measures the audio, not the code.
   - **The rediarize audio guard** (`73d93a40a`) is checked there: it refuses an episode whose
     downloaded audio is not the transcribed audio.
   - **The split-voice merge** (v49, on at 0.8 in the DGX profiles since v50) is part of what ships, so the
     table and the reviewer cover it.
   - **Design, folded in by the operator on 2026-10-25** (one DGX block, shipped CLI, prod hold):
     - **Three arms on the same episodes, in one block:** the deployed code, the deployed code
       again, and the branch. The two deployed-code arms measure the noise floor (step 3 showed
       the LLM's answers differ run to run on the same input); only branch-vs-deployed
       differences beyond that floor count, each attributed from the decision trace.
     - **Speaker detection pinned at temperature 0** in the profile copy (v47), so the candidate
       lists do not move between arms.
     - **Episodes:** val_v6, plus the episodes where v51-v53 can act (statement-only hosts:
       217 on the snapshot; third-person-refused names E1d could take; floor-exemption cases),
       plus a random sample where nothing should move.
     - **Scored wrong names first;** a voice with no label row that a change names is judged by its
       own words (the v417 shape) and reported.
     - **The voice rung, as a promotion candidate,** scored from the shadow sidecars against the
       ladder on the same arms: (a) publishing everywhere, (b) publishing only where the ladder
       left a voice unnamed or the LLM resolution refused. Plus the error-spreading check: on
       feeds with several episodes in the set, does a wrong history voice carry its name forward?
       The operator decides promotion from that table.
     - **Cost:** relabel ran at about 24 s per episode (step 3), so 3 arms x about 130 episodes is
       about 2.5-3 h of DGX time. Batch-invariant serving on the DGX (a DGX decision) would make
       the two deployed-code arms identical and the design cleaner.
   - **As run, 2026-10-25 (operator: scope everything, the panel rule, run at once):**
     - **Arms:** deployed code `b60a89e` (the 10-20 deploy-all image `sha-b60a89e`; no pipeline
       deploy since) twice, the branch `8e3c3af2e` once, in that order; each arm its own code's
       shipped `dev_dgx_full` with GI, KG, vector search and summaries off
       (`config/dgx_block/step5_*_nameonly.yaml` in the eval repo), and its own fresh copy of
       `snap1016/corpus` (48 whole feeds, hash-checked identical before the run).
     - **Detection temperature:** deployed code has no `speaker_detection_temperature` and its
       config forbids unknown keys, so its arms detect at the provider's 0.3, as prod does; only
       the branch arm is at 0. The two deployed arms measure exactly that run-to-run noise.
     - **Episodes:** 212 labelled episodes: val_v6 77, gate5_v1 51, v51 episodes labelled in
       earlier splits, the E1d candidates, the 3 v53 cases from the stored runs. 6 (3 val_v6, 3
       v51) are out: their feeds are not in the corpus's `feeds.spec.yaml`, so a relabel cannot
       reach them. The 103 v51 episodes with no labels are not run.
     - **Panel rule:** every voice the branch alone changes (dep1 = dep2 != branch) is judged by
       its label when it has a scored one; otherwise by a blind two-model panel (Opus + Fable,
       full transcript, no served names). Voices where dep1 != dep2 are reported as noise, not
       attributed. Tooling: eval `scripts/eval/naming/step5_table.py`, `step5_panel_prep.py`.
     - **Checks before the run:** a 1-episode test per code version; on an episode with two
       copies under one guid, the relabel rewrote the case's own copy; the scorer's copy choice
       is checked per case after each arm.
   - **Result, 2026-10-25** (10:34:58Z-12:23:37Z; 212 episodes per arm; scored 212 of 212 in
     every arm, the scorer reading the rewritten copy in all 212). Labelled voices (1,090):

     | | Deployed run 1 | Deployed run 2 | Branch |
     | --- | --- | --- | --- |
     | Bad names (wrong person, spurious, non-participant) | 87 | 78 | 51 |
     | Right (named right + rightly unnamed) | 842 | 852 | 891 |
     | Missing | 150 | 147 | 136 |
     | Role errors | 11 | 13 | 12 |
     | S7: insights on the wrong person (of 8,592) | 608 | 510 | 472 |
     | S7: insights on the right person | 5,307 | 5,517 | 5,939 |

     - **What clears the deployed noise** (the two deployed runs as its measure; step 6 review):
       labelled bad names 87 / 78 -> 51 (runs differ by 9, gain 27-36), right 842 / 852 -> 891, S7
       right-person insights 5,307 / 5,517 -> 5,939. **What does not:** S7 wrong-person insights
       608 / 510 -> 472 (runs differ by 98, the branch 38 below run 2): "not worse", not shown
       better.
     - **val_v6 (held-out), branch vs deployed run 1:** bad 33 -> 16, right 239 -> 259; S7 wrong
       287 (195 in run 2) -> 171, within the deployed noise. Role errors 2 -> 4: one deployed noise, two the same in every
       arm, one a wrong name made right with the wrong role.
     - **Per voice (1,302):** 1,152 stable, 56 noise (the deployed runs differ), 94 where both
       deployed runs agree and the branch differs (82 labelled: right +49 / -13, bad -35 / +9; 12
       unlabelled, to the panel). One deployed pair cannot tell a branch change from noise per
       voice: from 56 of 1,302 differing, about 29 of the 94 are expected by noise alone (step 6
       review; the branch arm's own noise, at detection temperature 0, is not measured). The
       direction holds (+49 / -13, -35 / +9 is far more one-sided than noise); the per-change
       attribution below does not.
     - **The 18 harmful labelled changes, read one by one:** the talk floor taking self-introduced
       guests and audience questioners off (5); the LLM resolution answering differently (6 bad:
       co-host picks the labels cannot decide, a swap, a credit name on interview tape); v45 (2
       right lost, 2 wrong removed, against deployed); a stated host forced onto a narrator; a
       split-voice rescue; a third-person mention the resolver's passage sample did not hold. That
       last one became **v54** (replay over seven stored runs and the gate: 4 bad off, 1-2 right
       lost per arm). Measured and not built: a floor exemption for voices the LLM calls guest
       (3 right, 6 wrong over the stored runs), "with me <name>" as a self-introduction (107
       resolver verdicts moved).
     - **Rediarize audio guard (`73d93a40a`), checked on the DGX:** on y029 (publisher file
       changed since ingest) a branch rediarize refused before diarizing ("downloaded audio is
       7660914 bytes; the transcribed audio was 6909843"), episode failed `AudioChanged`.
     - **Shadow namers (promotion is a later decision):** voice rung filling only unnamed voices
       33 right / 5 bad; LLM pass filling unnamed 103 right / 43 bad, publishing everywhere 437
       right / 75 bad (ladder 360 / 51); 5 wrong voice-rung names repeat within a feed.
     - **Panel:** Opus and Fable, blind (full transcript, no served names, no labels), agree on all 17 voices:
       the 12 unlabelled branch changes and the 5 LLM-resolution cases. Unlabelled: the branch removes
       8 bad names (3 on a promo, an announcer and a merged voice; 5 wrong people left unnamed, one of
       them an ASR spelling of the right person), loses 2 right ones, adds 1 bad (an announcer given
       the host's name: one of v54's refusals), corrects 1 spelling. The 5 LLM cases are all wrong per
       the panel, the two co-host picks included. On these 17: branch 4 right / 7 unnamed / 6 wrong,
       deployed 6 / 3 / 8 in both runs.
     - **Not covered:** no unseen feed (all prod feeds have earlier cases); 6 episodes unreachable
       (feeds not in `feeds.spec.yaml`); the 103 unlabelled v51 episodes; the voice rung and the
       split-voice merge act only where the snapshot holds fingerprints; v54 ran through the real
       pipeline on one episode where the LLM did not repeat the wrong name, so its in-pipeline
       proof is the ladder scenario; labels are silver.
     - **v54 is not held-out:** found on the gate's own episodes; its evidence is the replay over the
       stored runs and the gate's arms (the table's branch arm is `8e3c3af2e`, without v54). It only
       removes names. Accepting it on that evidence is the operator's call.
     - **Non-English episodes:** the snapshot holds none, so nothing here measured them. The step 6
       review found v53 / v54 reading every voice as English (fixed: the episode language is passed,
       as every resolver caller passes it).
     - **Relabel hosts:** a relabel reads the feed's frozen sibling hosts, so the episode-statement
       hosts (v43 / v44, and v51) reached the gate only where that list was empty (36 of 212 cases
       differ in known hosts); on ingest they always apply. Their served effect on feeds whose blurb
       already names a host is not measured.
6. **Fresh independent reviewer:** reads the diff and the outcome table, and gives a verdict.
7. **The operator's go, then merge #2393.**
8. **Deploy:** a separate go, then the post-deploy checks.
   - The 0034 / 0035 migrations are separate corpus changes under `CORPUS_CHANGE_METHOD.md`.

## Loop status after follow-ups, round 1 (2026-10-24)

Per work-list item (eval repo `NAMING-GATES-2026-10-22.md`), counted by the executor. Per-case
detail is in the eval repo's DGX block plan.

| # | Item | Status | Open |
| --- | --- | --- | --- |
| 1 | E1d + S30s rescues | 9 of 10 labelled fires right on held-out; the wrong one was S30s on a guest whose name is never stated | done (v40): S30s requires matching fingerprints when both exist; run B adds fingerprint pairs |
| 3 | Talk floor | Gate 1: 6 / 6 right. Gate 2: 15 / 20 right, 5 hosts' right names removed (co-hosts, moderators, 1.5-4.4% of talk) | done (v40): a seated host the LLM resolution also calls host keeps the name (+3 right, 1 unconfirmable); open: why the pipeline treats 2 of the 5 co-hosts as guests |
| 6 | LLM pass | better than the ladder on both gates; Gate 1 wrong 2.7% (budget 5%), unnamed 5.4% (criterion 2%); Gate 2 wrong 12.5% (criterion 10%) | decision: promotion scope |
| 7 | All-agree | no measurable gain, a real cost | closed: passes = 1 |
| 8 | Gate 2 | swap study done: the LLM fixes 58 of the ladder's swaps, adds 26, and names 21 voices the labels leave nameless | label check of the nameless hosts; voice rung needs a rediarize |
| 2 | Voice rung | shadow safe; publish not measured on held-out | decision: a held-out rediarize run (DGX) |
| 9 | Split-voice merge | 2 / 2 right; recall not measured | decision: a rediarize run on the split cases (DGX) |
| 10 | Widening | on its own target population (run D) it publishes job titles and phrases as speakers; net worse | closed: removed (v41) |
| 11 | C1 rediarize | keeps the transcript; misleading log fixed; y029's collapse is the publisher's file having changed (6.9 MB -> 9.2 MB) under an old transcript | open: decide a guard (refuse or flag when the downloaded audio differs from the transcribed one) |

## Loop status, round 3 (2026-10-24 evening)

Supersedes the table above where they differ. Per-case detail: the eval repo's DGX block plan,
"Follow-ups, round 3".

| # | Item | Status | Open |
| --- | --- | --- | --- |
| 1 | E1d + S30s rescues | as above (v40) | none |
| 3 | Talk floor | as above (v40) | 2 co-hosts treated as guests: deferred to a later round (operator) |
| 6 | LLM pass | Gate 2 wrong 14.4% on the corrected labels (was 12.5%) | stays in shadow (operator) |
| 7 | All-agree | closed | none |
| 8 | Gate 2 | the nameless-host label gaps are filled (eval `1019b1c`, 17 voices) | none |
| 2 | Voice rung | run A, val_v5 with unchanged audio (28 episodes, 74 voices): right 45, wrong 0 (ladder 41 / 2, LLM 44 / 2); on 60 named host voices it fixes 10 and breaks 0 | stays in shadow; promotion on more data. Its reach on old episodes is limited: fingerprints need a fresh diarization, safe only from archived audio, and how many prod episodes have archived audio is not known here (prod to report) |
| 9 | Split-voice merge | run B as above; held-out (run A): 181 different-person pairs, max 0.598 | done: on in the DGX profiles (v49) at 0.8 (v50; 0.7 waits for a post-deploy merge audit) |
| 10 | Widening | redone as typed steps (v42-v44). Step 3 (paired relabel, 28 episodes): the gains came through the pool (a host seated, spellings); the one effect through the LLM was a wrong name (x046, via an E1d defect) | done: v51 (stated hosts pooled, not LLM-bindable; replay: candidates equal the pre-v43 ones on 2,938 episodes), v52 (E1d fix) |
| 11 | C1 rediarize | guard built (refuses when the downloaded audio is not the transcribed audio; duration fallback when no size was recorded); run A, without it, shows the harm: changed audio 22 -> 89 bad names | must ship; checked in the release gate |
| new | One name on two voices | v45: neither voice keeps it; resolver replay 30 names removed, nothing else; served on 6 faithful episodes: 3 wrong gone, 1 right gained, 0 right lost; step 3: 2 wrong and 2 non-participant names removed, 0 right lost | none |
| new | Title words in every language | v46: fixed a v42 defect of mine (a test failing since v42) | none |
| new | Speaker detection temperature | v47: detection can be pinned to 0 alone; both DGX profiles do | not measured on the DGX; the serving's batch-invariant mode is off (DGX decision) |
| new | Host phrase about an event | v48: "a platform / initiative hosted by X" no longer adds a host; replay: exactly the 2 junk names removed | none |
| new | Talk-floor host exemption (v40) | step 3: kept the host's name on the show's announcer (v417, unscored) | done: v53 (no exemption for a voice that talks about the host in the third person) |
| new | E1d rescue | step 3: its self-introduction cue read "so I'm hoping that <guest>'s going to ..." | done: v52; replay over the stored runs: 3 firings stopped, all guest names on the host; 0 right lost |

## Not covered by step 1

- **Prompt evidence:** the resolver prompt's own evidence line ("says this ABOUT them") still
  labels "I'm at <name>" as a mention. The model answered right anyway in both episodes; changing
  the prompt is an LLM-side change and needs a DGX measurement.
- **The relabel-only shape:** why the snapshot's own runs of this show ended with the right served
  names while the relabel did not (a later rung put the host back there) is not traced.
- **Other handle forms:** searched 2026-10-24 (round 3): 51 handle phrases, 7 followed by a stated
  name, 1 wrongly refuted (a speech-to-text misspelling of the host's name); left, a hidden name.
- **The roster after the resolver:** replayed 2026-10-24 (round 3), the old side reproducing the
  recorded run exactly: with the branch, both episodes' served names are right except one guest
  left unnamed by the surname refusal.
- **A false third-person refusal of a host:** on a two-voice show it can still move the host's
  name onto the guest, overriding a correct known-hosts baseline. Not seen on the snapshot.
- **A rediarize with GI on:** that it rewrites the GI was read in the code, not run.
