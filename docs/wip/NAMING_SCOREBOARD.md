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

## Cumulative table (validation, 500)

| Step | correct name | wrong name | missing name | promo/ad/clip published | host/guest swapped | hosts correct |
| --- | --- | --- | --- | --- | --- | --- |
| today (baseline) | 602 | 155 | 384 | 49 | 31 | 342 |
| + seats (v4, accepted) | 604 | 138 | 399 | 47 | 31 | 342 |
| + person check (accepted) | 604 | 119 | 418 | 47 | 31 | 342 |

## Attempts

| # | Problem | Loop | Change | Dev better / worse | Val better / worse | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | host seats | 1 | seat logic v4 | 6 / 1 | 20 / 1 (host) | loop again |
| 1 | host seats | 2 | + late-guest absorption, dominant-voice guest name, Tracy/Tracey snap | 0 / 0 | 2 / 1 (host) | loop again |
| 1 | host seats | 3 | + snap guard (no claim on a host another voice already is) | 0 / 0 | 2 / 0 | accepted with one known host regression (operator) |
| 2 | host/guest judged across the whole voice | 1 | a guest phrase counts only in the first 60% of a voice | 3 / 0 | 0 / 3 | rejected (guests whose only "thanks for having me" is a farewell) |
| 3 | junk names passing the person check | 1 | show/brand and region tails, count words, captured role words, job titles in long names, product mononyms, stray `?` | 4 / 0 | 20 / 0 | accepted (census: 15 distinct names newly refused, all junk) |

## Running conclusion (updated after every problem)

After 2 accepted slices and 1 rejected loop: rules move single- to low-double-digit voices per
slice on 2,010 scored validation voices. Both accepted slices mostly turn WRONG names into NO name
(wrong 155 -> 119, missing 384 -> 418); correct names barely move (602 -> 604). Removing bad
answers is what rules do well here; producing right answers is where they have not moved yet. The rejected loop is the held-out set working as intended
(the development set alone would have accepted it). Not yet enough slices to call a wall.

Next after the planned slices (operator, 2026-10-03): a FEED-LEVEL analysis with the Fable advisor —
patterns per show (who hosts it, who recurs across its episodes, how its transcripts/voices look),
using the show sidecar (`feeds/<feed>/show.json`) and the gold sets, to find rules that work at the
show level rather than per episode.

Candidate beyond rules (not yet tried): make the production LLM resolver read each voice the way
the labellers did (the labelling guide as the prompt, the case view as input, the rules as guards),
measured on the same gate with real DGX calls on the development set first.
