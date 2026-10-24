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
4. **A fresh held-out set (val_v6), added 2026-10-24 after run A:** val_v3 was read while building
   the episode host statements, and val_v5 was used to choose the split-voice threshold (run A's
   fingerprints). The release score needs labelled episodes nobody has looked at: picked, labelled
   blind, and not opened until step 5 scores them.
5. **Release gate on a prod-snapshot clone:** a before/after outcome table of what the deploy
   changes on served names, with S7 (insight-weighted) beside the per-voice score, on val_v6.
   - **Relabel, not rediarize:** run A rediarized from the publisher's current files (this corpus
     has no audio archive); on the episodes whose audio had changed since ingest, wrong or
     non-participant names went 22 -> 89 (unchanged audio: 17 -> 12). A rediarize in the gate
     uses archived audio only, or it measures the audio, not the code.
   - **The rediarize audio guard** (`73d93a40a`) is checked there: it refuses an episode whose
     downloaded audio is not the transcribed audio.
   - **The split-voice merge** (v49, on at 0.7 in the DGX profiles) is part of what ships, so the
     table and the reviewer cover it.
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
| 9 | Split-voice merge | run B as above; held-out (run A): 181 different-person pairs, max 0.598 | done (v49): on at 0.7 in the DGX profiles |
| 10 | Widening | redone as typed steps (v42-v44): served effect not yet measured | step 3, a paired relabel after run A |
| 11 | C1 rediarize | guard built (refuses when the downloaded audio is not the transcribed audio; duration fallback when no size was recorded); run A, without it, shows the harm: changed audio 22 -> 89 bad names | must ship; checked in the release gate |
| new | One name on two voices | v45: neither voice keeps it; resolver replay 30 names removed, nothing else; served on 6 faithful episodes: 3 wrong gone, 1 right gained, 0 right lost | served effect on the rest: step 3 |
| new | Title words in every language | v46: fixed a v42 defect of mine (a test failing since v42) | none |
| new | Speaker detection temperature | v47: detection can be pinned to 0 alone; both DGX profiles do | not measured on the DGX; the serving's batch-invariant mode is off (DGX decision) |
| new | Host phrase about an event | v48: "a platform / initiative hosted by X" no longer adds a host; replay: exactly the 2 junk names removed | none |

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
