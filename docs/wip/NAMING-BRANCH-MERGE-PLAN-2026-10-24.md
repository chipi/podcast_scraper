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
4. **Release gate on a prod-snapshot clone:** a before/after outcome table of what the deploy
   changes on served names, with S7 (insight-weighted) beside the per-voice score.
5. **Fresh independent reviewer:** reads the diff and the outcome table, and gives a verdict.
6. **The operator's go, then merge #2393.**
7. **Deploy:** a separate go, then the post-deploy checks.
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

## Not covered by step 1

- **Prompt evidence:** the resolver prompt's own evidence line ("says this ABOUT them") still
  labels "I'm at <name>" as a mention. The model answered right anyway in both episodes; changing
  the prompt is an LLM-side change and needs a DGX measurement.
- **The relabel-only shape:** why the snapshot's own runs of this show ended with the right served
  names while the relabel did not (a later rung put the host back there) is not traced.
- **Other handle forms:** "find me at <name>", "follow me @<name>" were not searched for.
- **The roster after the resolver:** the y007 / y008 replay stops at the resolver; the later
  rungs were not replayed.
- **A false third-person refusal of a host:** on a two-voice show it can still move the host's
  name onto the guest, overriding a correct known-hosts baseline. Not seen on the snapshot.
- **A rediarize with GI on:** that it rewrites the GI was read in the code, not run.
