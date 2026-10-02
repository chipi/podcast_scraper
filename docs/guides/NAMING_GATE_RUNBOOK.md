# Naming gate runbook — labelled speaker sets and how every naming change is judged

Speaker naming (who is host, who is guest, which voice is which person) is changed by many small
rules, and each change is a trade: it fixes some voices and can break others. Reading a replay diff
by hand cannot tell whether a trade is good. This runbook replaces that judgement with a test:
real episodes labelled with the truth, and a gate that scores old code against new code on them.

Introduced 2026-10-02 (operator: "input, output, we compare").

## The two labelled sets

| Set | Size | Role |
| --- | --- | --- |
| **Development** (`dev_v1`) | 101 episodes, 449 voices | Iterate against it. Read every regression. Tune rules here. |
| **Validation** (`val_v1`) | 500 episodes, 2,382 voices, disjoint from dev | Score each candidate ONCE before accepting it. Never read these cases while tuning. |

The split is the same as training versus held-out data: a rule that improves development but not
validation was fitted to the 101 development episodes, not to podcasts.

**Where they live.** `.test_outputs/naming_gold/` in the main checkout — git-ignored. They are
derived from real episodes (voice texts, titles, names) and are NEVER committed. How to share or
back them up is an open operator decision; until then they are local only.

```text
.test_outputs/naming_gold/
  README.md               what is where
  LABELING_GUIDE.md       the labelling standard (current = v2)
  LABELING_GUIDE_v1.md    the first version (dev labels_v0 and val wave 1 were made under it)
  dev_v1/cases/           cNNN.json — blind case files (no published names)
  dev_v1/labels_v0/       first labels
  dev_v1/labels_v1/       after the audit — SCORE AGAINST THIS
  val_v1/cases/           vNNN.json
  val_v1/labels_v*/       validation labels, versioned the same way
```

Never edit a labels version in place: copy to the next version and change that. The gate report
says which version it scored.

## A case file

One episode, built read-only from the corpus (`meta_relpath` points at its metadata):
feed title/description/authors, episode title/description, and per diarized voice its talk share,
first/last second, opening ~1,800 and closing 400 characters, plus the first ~45 turns in order.

It deliberately does NOT contain the names the pipeline published, so a label cannot copy our own
answer.

## A label

Per voice: `role` (host / guest / ad / promo / clip / unknown), `name` (only if the episode's own
text or metadata supports it, else null), `name_evidence` (the quote), `confidence`
(high / medium / low), `note`. The full standard is `LABELING_GUIDE.md`; its v2 rules came from the
first audit (announcers are `unknown`, merged two-person voices are `unknown`, promos for other
shows are `promo`, a name the text contradicts is null, no names from ASR artifacts).

`unknown` and `low` labels are not scored: they are the honest "the text cannot tell".

## Making or extending a set

1. **Select** episodes read-only on the box (newest run per episode, diarized, at least two
   voices), seeded so the selection is reproducible. Validation must exclude every development
   episode. Write blind case files.
2. **Label** with agents, about 25 episodes each, every agent with its own helper folder (agents
   sharing one folder overwrote each other's scripts in the first round). Each agent self-checks:
   every voice covered, valid values, evidence whenever a name is set.
3. **Validate** all files mechanically (voice sets match the cases, no stray files).
4. **Audit** with a stronger model: every low/medium label plus a random 20% of high ones. The
   high-confidence sample estimates the error rate of the unaudited labels (first audit: 45/46
   role, 46/46 name). Apply the audit's corrections as the next labels version; fold its rule
   proposals into the guide.
5. **Save** cases and labels under `.test_outputs/naming_gold/`.

Cost on 2026-10-02: 101 episodes ≈ 4 Sonnet labelling agents + 1 Fable audit; 500 episodes ≈ 20
labelling agents plus an audit.

## Running the gate

`scripts/measure/naming_gate.py` replays the speaker roster over each labelled episode twice — OLD
code and NEW code — from the stored artifacts (voice texts, stored LLM answers; no LLM call, nothing
written) and scores every voice:

| Outcome | Meaning |
| --- | --- |
| `correct_name` / `correct_unnamed` | right |
| `missing_name` | labelled name, published nobody |
| `role_error` | right person, host/guest swapped |
| `wrong_name` | a different person |
| `spurious_name` | a name the text does not support |
| `non_participant` | an ad / promo / clip voice published as a speaker |

It prints the table for both sides (and a hosts-only column), every REGRESSION (old right, new
not), every FIX, and every voice that got BETTER or WORSE by severity
(wrong > role swap > missing > correct) — so "a wrong name became no name" is visible.

```bash
# on the box (the corpus is there); variant files are module copies, e.g. from git show
python scripts/measure/naming_gate.py --corpus /app/output \
    --cases <naming_gold>/dev_v1/cases --labels <naming_gold>/dev_v1/labels_v1 \
    --old roster=/tmp/rp/roster_head.py --new roster=/tmp/rp/roster_candidate.py \
    --signatures --json /tmp/rp/gate_report.json
```

`--old`/`--new` take `MODULE=PATH` for `roster`, `hosts`, `resolution`, `ad_signatures` — exactly
as `scripts/measure/roster_replay.py`, which the gate uses. A variant that reads the feed's history
(the seat-logic `host_copresence` prior) gets the other episodes of the same feed automatically.

**Replay fidelity.** Checked on 2026-10-02: the committed replay reproduced the advisor's seat-v4
numbers exactly (`lost 69, renamed 4, gained 1`). Limits: stored LLM answers are FINAL roles, not the
model's raw verdicts, so a rule keyed on LLM verdicts is only partly measurable; a change to what
the LLM is ASKED cannot be measured offline at all.

**Where to run.** Today the replay runs inside the api container on the box. A read-only local
snapshot of the needed artifacts (metadata, segments, speaker diagnostics — no audio) into
`.test_outputs/` would let it run on the workstation without touching prod; planned.

## Accepting a change

1. Development gate: read every regression and every "worse" row. A regression on a labelled
   host blocks the change unless the label is shown wrong (then fix the label in a new version).
2. Validation gate, once, at the end: no worse than development in direction. If development
   improves and validation does not, the change was fitted to the development set.
3. Unit tests for the rule itself, one behaviour per test, with synthetic fixtures shaped like the
   real cases (never real episode text).

## Loop discipline (operator, 2026-10-02)

One problem at a time, no rush. For each problem (slice):

1. **Goal**, set before starting, on the development set. Default: fix at least half of the
   slice's failing voices, with zero regressions on labelled hosts.
2. **Loop** on development: design, replay, read every change, adjust.
3. **Validate** once on the 500. Same direction as development, or it is not accepted.
4. If validation disagrees, loop again. **At most 3 loops per problem.** After 3, stop: record
   where it landed and why in the table, and move to the next problem.
5. After all problems, review the cumulative table and decide what comes next.

The cumulative table (`naming_gate.py --step ...`) is the scoreboard: one column per accepted slice,
each on top of all previous ones.

| Problem | Method fit |
| --- | --- |
| Host seats (v4) | first slice, measured |
| Host/guest judged across the whole voice (bled lines) | fully replayable |
| Polluted host pools | fully replayable (stored pools) |
| Junk names passing the name check | fully replayable + name census |
| The LLM's own naming | guards replayable; prompt changes need DGX test calls on dev |
| Diarizer splits | mostly upstream (audio); text-level merges replayable |

## Today's baseline (dev_v1, labels_v0, 2026-10-02)

Of 393 scored voices with the current code: 102 correct names, 148 correct unnamed, 41 wrong names,
68 missing names, 1 spurious, 15 non-participants published, 18 host/guest swaps. Seat v4 alone:
wrong 41 → 36, missing 68 → 72, one regression (a trailer clip named as a host).

On the audited labels (`labels_v1`): today 103 correct / 43 wrong / 70 missing; +seat_v4 104 / 38 /
74; better 6, worse 1.
