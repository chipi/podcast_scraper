# Naming scoreboard — every slice attempt, and an honest conclusion at the end

Operator, 2026-10-02: finish the planned slices, keep track, and conclude honestly whether the
rule-by-rule approach is hitting a wall or how far it can be pushed. Method:
`docs/guides/NAMING_GATE_RUNBOOK.md` (gold sets, gate; loops per problem: 3 for seats, then 2 max
from 2026-10-02).

Numbers are voices scored by `scripts/measure/naming_gate.py`: development = 101 episodes (393
scored voices, labels_v1), validation = 500 episodes (2,010 scored voices, labels_v1).

## The ceiling (what a careful reader gets from the same text)

The labellers worked only from the case files (each voice's opening/closing and the first turns —
less than the pipeline has). On validation they put a supported name on **1,368 of 2,382 voices**;
an independent audit agreed with 89–99% of the labels it checked. The pipeline today publishes a
correct name on **602**. Most of the gap is recoverable from text the pipeline already holds; the
question this scoreboard answers is how much of it RULES can recover.

## Plan — one step at a time, each result recorded here (operator, 2026-10-03)

1. Measure the LLM step's marginal value on the gold sets (same code with vs without its answers).
2. Problem 2, last loop (host evidence outranks a lone guest phrase; self-introduced hosts).
3. Problem 3: host pools — give the right names to voices the person check emptied.
4. LLM guards (veto/check the LLM's answers; replayable).
5. Feed-level analysis with the Fable advisor: per-show patterns; local beats global.
6. Mini autoresearch loop on the gate: propose → score on dev → keep only if better with zero host
   regressions → validate once on the 500; rules and per-show profiles first, the LLM step after
   (DGX budget agreed first).
7. Honest conclusion: how far this can be pushed.

## Cumulative table (validation, 500)

| Step | correct name | wrong name | missing name | promo/ad/clip published | host/guest swapped | hosts correct |
| --- | --- | --- | --- | --- | --- | --- |
| today (baseline) | 602 | 155 | 384 | 49 | 31 | 342 |
| + seats (v4, accepted) | 604 | 138 | 399 | 47 | 31 | 342 |
| + person check (accepted) | 604 | 119 | 418 | 47 | 31 | 342 |
| + presenter evidence (accepted) | 615 | 118 | 417 | 47 | 22 | 352 |
| (pools rebuilt from metadata with the committed detector — stale stored pools, not a change) | 623 | 118 | 408 | 47 | 23 | 360 |
| + host pools (accepted) | 644 | 112 | 395 | 46 | 21 | 378 |

From the host-pool row on, every column is measured with `naming_gate.py --repool` (each variant
rebuilds the episode's pool from stored metadata with its own hosts module); the rows above it
replay the stored pools. The unlabelled row between them is the repool itself.

## Attempts

| # | Problem | Loop | Change | Dev better / worse | Val better / worse | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | host seats | 1 | seat logic v4 | 6 / 1 | 20 / 1 (host) | loop again |
| 1 | host seats | 2 | + late-guest absorption, dominant-voice guest name, Tracy/Tracey snap | 0 / 0 | 2 / 1 (host) | loop again |
| 1 | host seats | 3 | + snap guard (no claim on a host another voice already is) | 0 / 0 | 2 / 0 | accepted with one known host regression (operator) |
| 2 | host/guest judged across the whole voice | 1 | a guest phrase counts only in the first 60% of a voice | 3 / 0 | 0 / 3 | rejected (guests whose only "thanks for having me" is a farewell) |
| 2 | host/guest judged across the whole voice | 2 | presenter evidence (advisor p7): a voice naming the show in a presenting formula, introducing the stated guest, or one half of an "I'm A / And I'm B" pair is a host even outside the pool, and outranks a lone guest phrase; episode-stated guest host joins the pool; show-name pool entries never named | 22 / 0 | 13 / 2 | accepted (last loop; 2 worse: Hard Fork Kevin's voice named Casey, an unnamed Business of Africa host voice named) |
| 3 | junk names passing the person check | 1 | show/brand and region tails, count words, captured role words, job titles in long names, product mononyms, stray `?` | 4 / 0 | 20 / 0 | accepted (census: 15 distinct names newly refused, all junk) |
| 4 | host pools | 1 | statement ∪ author tags with a per-name person check, wider stated-name grammar, episode-described hosts (byline silenced when the description names one), respelt/merged host not a stranger, solo/ownership/show-mononym guards (measured with `--repool` on both sides; repooling alone with the committed detector = +8 correct on val, stale stored pools) | 11 / 1 | 28 / 6 (3 guests given the host's name or role, 2 co-hosts given the pooled host's name, 1 pooled host unnamed) | loop again |
| 4 | host pools | 2 | + unnamed introducer seat (1c), a forced pool name declines when another voice presents, a self-introduced stated guest who sounds like one is never a stand-in host, a respelt guest name is not spare, a guest the description names after the cue is not a described host, a carried pool name is never forced onto a second voice, a pool host's own self-introduced name stays on a non-host voice (production wiring: the roster pool also gets the episode's stated people) | 2 / 0 vs loop 1 (13 / 1 vs before) | 4 / 0 vs loop 1 (31 / 5 vs before) | accepted (last loop; 5 worse remain: an MLST guest named Tim Scarfe, a Flip guest named Justin Norman, two Latin America in Focus co-host voices named Carin Zissis, one Carin Zissis voice unnamed) |

## Step 1 result — the LLM step's marginal value (2026-10-03)

Same committed code (seats + person check), replayed without vs with the LLM's stored per-voice
answers (stored FINAL answers, not raw verdicts — an approximation of the live step):

| Set | correct name | hosts correct | missing | wrong | host/guest swapped | better / worse voices |
| --- | --- | --- | --- | --- | --- | --- |
| val 500, no LLM | 502 | 316 | 572 | 78 | 20 | — |
| val 500, with LLM | 604 (+102) | 342 (+26) | 418 (-154) | 119 (+41) | 31 (+11) | 138 / 87 |
| dev 101, no LLM | 81 | 49 | 114 | 26 | 13 | — |
| dev 101, with LLM | 104 (+23) | 55 (+6) | 76 (-38) | 36 (+10) | 18 (+5) | 32 / 25 |

Reading: the LLM is the only component adding correct names in volume (102 of 604), and it also
causes about a third of the wrong names (41 of 119) and of the swaps (11 of 31): roughly one damaged
voice for every two improved. Steps 4 (guards on its answers) and 6 (the LLM loop) target exactly
that.

## Running conclusion (updated after every problem)

After 4 accepted slices and 1 rejected loop: correct names 602 -> 644 on validation (+42, of
which +8 is only the stale stored pools being rebuilt), wrong names 155 -> 112, hosts correct
342 -> 378, swaps 31 -> 21. The first two slices turned WRONG names into NO name; presenter
evidence (+11) and host pools (+21) are the first to ADD correct names, and both did it by giving
the roster better METADATA (who presents, who the feed and the episode say host), not cleverer seat
arithmetic. Each accepted slice still leaves a few new errors (host pools: 5 on validation, three
of them a guest given a host's name), and the last loop of host pools moved validation by only 4
voices: the per-episode rule tail is getting thin. Against the labeller ceiling (1,368 named
voices), the gap was 766 voices; four slices closed 42 of them (5%). The rejected loop is the
held-out set working as intended (the development set alone would have accepted it). Not yet
enough slices to call a wall; whether per-show profiles (step 5) and the LLM step (steps 4 and 6)
move more than this is the open question those steps measure.

Next after the planned slices (operator, 2026-10-03): a FEED-LEVEL analysis with the Fable advisor —
patterns per show (who hosts it, who recurs across its episodes, how its transcripts/voices look),
using the show sidecar (`feeds/<feed>/show.json`) and the gold sets, to find rules that work at the
show level rather than per episode. Framing (operator): at some point LOCAL (per-show) optimisation beats
GLOBAL rules — a learned per-show profile, kept honest by the same gate (validation spans 53 shows).

Then (operator, 2026-10-03): LLM-step optimisation as its own loop — prompt, candidate list and the
per-voice case view, with real DGX calls on dev, validated once on the 500, 2 loops max. The
"LLM reads like a labeller" experiment below belongs to this loop.

Candidate beyond rules (not yet tried): make the production LLM resolver read each voice the way
the labellers did (the labelling guide as the prompt, the case view as input, the rules as guards),
measured on the same gate with real DGX calls on the development set first.
