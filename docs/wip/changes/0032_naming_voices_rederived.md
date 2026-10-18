---
id: 0032_naming_voices_rederived
class: full           # full | light (see README.md); the gate measurement can overrule light
status: open          # open | landed
issue: 2380
touches:              # every content-code path this change writes (globs allowed)
  - src/podcast_scraper/upgrade/migrations/m0032_naming_voices_rederived.py
  - src/podcast_scraper/upgrade/migrations/m0025_one_person_one_entry.py
---

# 0032 — the names the deployed naming code gives 126 served episodes (held, #2380)

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Every section is filled
before the step after it starts. Was 0031 until 2026-10-18; never ran on
prod; unregistered until it passes. As first built it named voices and left their insights hidden
(S1n +1,085 on the 2026-10-16 clone); its rewrite is fixed in 1bc9d18d0, unmeasured since. A skipped step is written `MISSED: <what and why>`, and it counts
only once this record has a line `Operator accepted: YYYY-MM-DD <what>`.

## 1. Intake

Reading list, one line each: every open and recent issue, doc, scoreboard and metric on the topic,
and what is already known and measured.

_TODO_

## 2. Outcome and baseline

The scoreboard numbers the change targets, the baseline with its command, output and snapshot id.

_TODO_

## 3. Diagnosis by loss

Causes ranked by outcome lost, the committed script that ranks them, and the samples read.

_TODO_

## 4. Prediction

Dated: YYYY-MM-DD

Integers on the named snapshot for each gating number and the change's own numbers; population
reached and not reached; trade-rule moves by identity; what must not move.

_TODO_

## 5. Acceptance

Written by a fresh reviewer from the goal and the prediction, without seeing code.

_TODO_

## 6. Build

The scenario tests (written first) and the code, by commit.

_TODO_

## 7. Measure

Scoreboard before/after on a snapshot clone, counts per path, the full-clone undo, the per-episode
plan, the derived-surface census, and the gate report (`0032_naming_voices_rederived.gate.json`, from
`scripts/measure/corpus_change_gate.py`).

_TODO_

## 8. Verdict

The fresh reviewer's verdict against each acceptance criterion, with its own numbers and its
verdicts file.

_TODO_

## 9. Operator review

_TODO_

## 10. Before apply / deploy

Prod's own per-episode plan diffed against the clone's; the run steps, with every derived surface.

_TODO_

## 11. After apply / deploy

_TODO_

## 12. Retrospective

_TODO_

## Not covered

_TODO_
