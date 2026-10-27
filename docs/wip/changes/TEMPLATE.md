---
id: NNNN_short_name   # the migration id, or a name for a code change
class: full           # full | light (see README.md); the gate measurement can overrule light
status: open          # open | landed
issue: NNNN
touches:              # every content-code path this change writes (globs allowed)
  - src/podcast_scraper/upgrade/migrations/mNNNN_short_name.py
---

# NNNN — one line on what the change is for

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Every section is filled
before the step after it starts. A skipped step is written `MISSED: <what and why>`, and it counts
only once this record has a line `Operator accepted: YYYY-MM-DD <what>`.

## 0. Why light

Only for `class: light`: which files it writes and why none of them changes what a listener or
reader is shown. Delete for `full`.

_TODO_

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

Prediction read by a fresh agent (method §4a), 5 minutes, before any code: rows added to section 4 and the step-8 spot checks, or "complete". No acceptance list.

_TODO_

## 6. Build

The scenario tests (written first) and the code, by commit.

_TODO_

## 7. Measure

Scoreboard before/after on a snapshot clone, counts per path, the full-clone undo, the per-episode
plan, the derived-surface census, and the gate report (`NNNN_short_name.gate.json`, from
`scripts/measure/corpus_change_gate.py`), the per-voice before/after excerpts (up to 10), and the
builder's own GO or NO-GO with its deciding number.

_TODO_

## 8. Verdict

The §4a review block from a fresh agent: prediction vs gate, required items, the after-state read,
open calls, VERDICT. No numbers of its own.

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
