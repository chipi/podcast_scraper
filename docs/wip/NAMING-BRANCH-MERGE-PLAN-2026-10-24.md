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
     on the host's own voice ("On Twitter, I'm at <name>"), and the two-voice complement or the
     swap then moved it onto the guest's voice.
   - **Fix:** ladder doc v39, with a failing test first.
   - **Replay:** old code from git, over the 2026-10-16 prod snapshot and the R1-GI / R4 copies.
     - The old code agrees with every recorded verdict.
     - 2 pairs change on the snapshot (host voices; served names already right) and 2 on each
       run copy (the two broken episodes).
     - No new refusal; 3 mutants killed.
     - The gate script is `scratchpad/replay_tp/gate.sh`, not committed.
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

## Not covered by step 1

- **Prompt evidence:** the resolver prompt's own evidence line ("says this ABOUT them") still
  labels "I'm at <name>" as a mention. The model answered right anyway in both episodes; changing
  the prompt is an LLM-side change and needs a DGX measurement.
- **The relabel-only shape:** why the snapshot's own runs of this show ended with the right served
  names while the relabel did not (a later rung put the host back there) is not traced.
- **Other handle forms:** "find me at <name>", "follow me @<name>" were not searched for.
