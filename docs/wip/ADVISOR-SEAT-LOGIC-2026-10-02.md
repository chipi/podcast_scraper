# Advisor report — host seat logic (v4) and review of commits since the last pass

Fable, read-only on prod; replays one at a time; no LLM; nothing written under `/app/output`.
Variant code, diff and replay outputs are in the session scratchpad (`advisor/`): `roster_adv.diff`
(535 lines vs HEAD `roster.py`), `adv4.jsonl` / `adv4_view.txt`, `verdicts.py`, `notes_tests.md`.
They contain real episode text and are NOT to be committed.

Incident: an upload via `docker cp /dev/stdin` left a symlink to `/proc/self/fd/0`; a later
`cat >` into it spun at 100% CPU in compose-api-1 for ~15 min until killed. Load returned to normal
(verified 17:19 UTC: load 0.47). Five finished `docker exec` processes remain as zombies under the
api's `serve` process (no CPU/memory; cleared on the next container restart).

## Verdict
Gating step 4 on the LLM's host verdict (draft 1) is the wrong lever: the verdict is not stored on
disk (`llm_delta` holds final-vs-baseline role DIFFS only) and is itself often wrong. Replacement
v4 does not depend on LLM roles.

## v4 design
1. The feed's host count is a CAP. Steps 1-3 unchanged. Step 4 fills seats by arithmetic only when
   it is exact (candidates == empty seats); a spare voice → abstain.
2. Candidates: intro-window voices that are PRESENT (share ≥ 5% and first-to-last turn span ≥ 50%
   of the episode), not heard/judged/stated as someone else, not addressing the absent host by first
   name (vocative counts, 2:1 ratio guard against diarization bleed).
3. "Guest present" split: (a) a stated/introduced third party not bound to any voice, or an
   introduction cue in a seated host's opening → no fill; (b) the dominant (≥50%) voice is the
   guest and absorbs one unplaced person.
4. Feed memory: the k-th seat (k ≥ 2) needs the feed's sibling sidecars to show ≥ k EVIDENCE-named
   hosts (self_intro / publisher / llm_resolution — never forced) in ≥ 25% of episodes. Bimodal on
   prod: The Journal 2/94, The Daily 7/100 vs Unhedged 43/79, Odd Lots 41/50, Empire 20/48,
   Hard Fork 28/71, No Priors 21/70. Overridden by an in-room co-host cue ("I'm joined, as usual,
   by … mister Rob Armstrong"). Also gates the forced NAME for seats steps 2/3 filled.
5. Name-bearing cues after the LLM merge, unclaimed hosts only: "and me, <First>", "<other host>
   and I are here with" (opening 400 chars, two-host feeds). Absence cue ("Just me today … Anita
   has another project") removes that host's seat and forced name.
6. Unfilled seat → the stated host is published unplaced (`_unplaced_speakers`, verified).

## Replay (v4 vs HEAD, 2,102 episodes, every changed voice hand-read)
`{'episodes': 2102, 'role_changed': 69, 'renamed': 4, 'lost': 69, 'gained': 1}` →
74 name changes: **55 correct removals**, 3 correct renames, **5 wrong losses** (2 clearly real:
Ben Horowitz, Erik Torenberg on a16z's polluted 9-name pool; 2 where the feed lists the guest as a
host: Vladimir Keil, Aaron Levie; 1 ambiguous: Rob on "What's bothering bonds?"), 1 wrong gain
(Tracy on a trailer clip), 1 wrong rename ("Tracy" → "Tracey Alloway": a canonicalization gap),
9 unclear. Draft 1's ~20 real-host losses are all kept (Unhedged ×4, Hard Fork ×3, Dwarkesh,
Gergely, Runciman, Adam Reichardt, RJ Honicky, Dalrymple ×7). Iterations: v1 81/14/12, v2 76/13/12,
v3 90/34/13 (cues before the LLM merge — reverted), v4 69/4/1.

Residual wrong names v4 leaves as HEAD has them (not regressions): Unhedged "Triangle", Empire
369/372/375/360, No Priors ×3 (fixable by an opening-only formula override; needs its own replay).
Forgone correct gains: Rob on two Unhedged episodes first speaking at 100-153 s (step-4 window
~180 s is a candidate; needs a replay).

## Tests
One existing test changes (`test_two_hosts_and_two_voices_names_NEITHER_without_evidence`: give H2
a late turn). New tests, one behaviour each, listed in `advisor/notes_tests.md`. Integration left:
pipeline glue for `host_copresence` (sibling sidecars, like `_feed_recurring_text`), sibling index
in `roster_replay.py`, a provenance label for formula-named voices.

## Recommendations independent of v4
- Drop the `v in llm_host_voices` tie-break clause to keep the design replay-faithful.
- Persist the raw per-voice LLM verdicts in the sidecar (`resolution_attribution.llm_delta.verdicts`).
- `roster_replay.py` docstring overstates fidelity: the sidecar stores FINAL roles of
  `llm_resolution` voices, not the model's verdicts.

## Review of commits since the last pass
1. `e4156ded4` — corpus-neutral today (census of 3,738 names: 0 differ). New weakness: the gate now
   accepts "Host Mike", "Guest Mike", "Speaker Mike", "Guest Host Tim", "Mister Rob". Fix: refuse a
   two-token name whose first token is a role word; "Christopher Guest" stays publishable.
2. `roster_replay.py` — docstring fidelity claim (above); skipped episodes not counted.
3. Docs — no stale hashes.
4. Rebase onto `99d1e26f6` — zero file overlap; `5619 passed, 6 skipped` on the touched suites.
5. Gaps on the unplaced path: org-like stated hosts pass the gate ("Americas Online", "Trivium
   China", "Machine Learning Street", "Turkey Book", "Norman Conquest", "Carnegie India",
   "Africa Tech Summit"); company mononyms pass ("Apple", "Google", "Gemini", "Claude"); an LLM
   name was published on a cluster whose first words address that person ("Casey, where are you?"
   → Casey Newton) — not traced.
