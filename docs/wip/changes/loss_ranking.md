---
id: loss_ranking
class: full           # an instrument, not a corpus change; run through every step anyway
status: open
issue: none (the operator opens issues; INGESTION-ARC Stage 2, items 2.2 and 2.3)
touches:
  - scripts/measure/loss_ranking.py
  - src/podcast_scraper/quality/loss_ranking.py
---

# Loss ranking: why names are wrong (gold) and why insights are hidden (corpus)

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Plan:
[INGESTION-ARC-2026-10-19.md](../INGESTION-ARC-2026-10-19.md), Stage 2, items 2.2 (honest S7) and
2.3 (loss ranking). Operator, 2026-10-20: measure first and get to a loss ranking, then plan from it;
the ranking before the gold refresh (2.1), which checks it afterwards on unseen episodes.

## 1. Intake

- **S7** (`quality/gold_insights.py`, record `s7_s8_gold_insight_weighted`): per insight on the gold
  sets, correct / wrong / unnamed by the stored name against the label. Its `S7_wrong` counts every
  stored name that is not the labelled person: a misspelling, a fragment, junk and another person
  alike. The plan's 2.2 assumed misspellings inflate it.
- **S1** (`quality/corpus_scoreboard.py`): insights hidden because their voice is unnamed, over the
  served corpus; it does not say why.
- **The stored naming decision** (`speakers.diagnostics.json` `decision_trace.inputs.llm_resolution`)
  exists only for runs since #2276; the record's roster (`content.speakers`: name, role, placed,
  voices, source) exists for every run.
- **The gold** (`.test_outputs/naming_gold`, git-ignored, never committed): dev_v1 101, val_v1 500,
  val_v2 150 cases, `labels_v1`; val_v1 is development-tier (read while tuning).
- **Known pieces of the same classes:** #2333 (K1 swaps), #2352 (swapped first-name hosts), F4/F4a
  (a guest seated as host, the guest's name on the host's voice), 0033 (a stated guest unplaced).
- **Exploration, 2026-10-20** (scratch scripts, not this instrument): the numbers in step 4.

## 2. Outcome and baseline

The outcome is a worklist, not a corpus change: every wrong-name insight on the gold and every S1
insight in the corpus in exactly one class, with counts, examples and the per-insight rows, so the
next changes are chosen by measured loss. Baseline: S7 by bucket only, S1 by count only.

## 3. Diagnosis by loss

- **Wrong names are mostly the episode's own people on the wrong voice.** Of the 2,947 `S7_wrong`
  insights (dev 380, val_v1 2,128, val_v2 439, all shown), the stored name is another labelled
  voice's in the same episode for about 1,897: names rotated across voices, two voices exchanged,
  or one name on two voices. Misspellings and fragments are about 102.
- **Hidden insights cluster where the episode states a person no voice holds.** Of S1's 14,841,
  about 5,407 sit on a principal voice (at least 30% of the talk) in an episode whose record names
  a person `placed: false`; 3,247 of them where it is one stated person and one anonymous principal
  voice. On the gold, seating that person by elimination is right in 16 of 27 cases: elimination
  alone mints wrong names, so the class is reported with that precision, never as a fix.
- **Two text signals tried and recorded as negative** (not part of this instrument): a voice
  addressing its own published first name flags wrong names at 45-46% precision, 5% recall
  (quoted speech; diarization merges two speakers into one segment); the published host asking far
  fewer questions than another voice, 62% precision, 11% recall (co-hosts and panels). No single
  text cue sizes the binding class on the corpus.

## 4. Prediction

Dated: 2026-10-20T08:13

On `lost/gateworkB6c` (snapshot-prod-20261016 + 0030 + 0031) and the gold's `labels_v1`, the
committed instrument reports:

- **Gold, wrong names** (`S7_wrong` insights, by the stored name's relation to the labels): total
  2,947 (380 / 2,128 / 439), every one in exactly one class:
  - `binding` (the name is another labelled voice's in the same episode): 1,897, split
    `rotated` 1,152 (the owner voice carries a third name or none), `swapped` 544 (the two
    voices exchanged), `duplicated` 201 (the owner keeps its own name too);
  - `spelling` (the same person spelled differently, or a part of the name): 102 plus the
    first-name forms the exploration put under other person (51): 153, within ±25 (the rule is
    defined before the build and the difference listed);
  - `other_person`: the rest, 897 ± 25.
- **Corpus, hidden** (S1 insights, by the episode's record): total 14,841, split by whether a
  stated person is unplaced and whether the insight's voice is principal: unplaced + principal
  5,407 (one-to-one 3,247 in 85 episodes); unplaced + minor 1,275; none unplaced + principal 3,834;
  none unplaced + minor 768; nobody named on the episode 3,557. Each within ±0 (the definitions are
  the exploration's).
- **Gold, the one-to-one seat checked:** 27 episodes scored, 16 right, 11 wrong.
- Read-only: no corpus or gold file written; conservation asserted (classes sum to the bucket).

### Restated after the step-5 reviewer, before any code

Dated: 2026-10-20T08:22

The reviewer's definitions (step 5, A3–A8) replace the exploration's thresholds; its own scratch
code measured them on the same clone and gold. The restated integers, each ±0:

- **Gold, `S7_wrong`** 2,947 (380 / 2,128 / 439, all shown): `binding` 1,897 (`swapped` 544,
  `duplicated` 201, `rotated` 1,152: `third_name` 346, `owner_unnamed` 652, `owner_no_insight`
  154); `spelling` 52 (val_v1 only); `other_person` 998 (`shares_label_token` 101). By set: dev
  binding 200, other 180; val_v1 binding 1,397, spelling 52, other 679; val_v2 binding 300, other
  139.
- **Gold, wrong names outside `S7_wrong`:** `name_without_support` 81, `undecidable_named` 887,
  `unplaceable_named` 31, all shown.
- **Corpus, S1** 14,841: gaps 6 (`voice_not_found`, `no_diagnostics`); `nobody_named` 3,557;
  `unplaced_principal` 5,407 (`one_to_one` 3,247 in 85 episodes); `unplaced_minor` 1,269;
  `placed_not_recorded_principal` 3,293, `placed_not_recorded_minor` 638; `none_unplaced_principal`
  541, `none_unplaced_minor` 130.
- **Seat check (served gold cases):** 26 scored: 16 right, 9 another person, 1 no supported name.

What the exploration had wrong, corrected here: "misspellings ~102" (65 of them were a greeting
fused to a first name; spelling is 52); the seat check counted a case not served on the clone (16
of 26, not 27); "none unplaced" 3,834 / 768 was mostly records that do not state `placed` (only 671
truly have nobody unplaced); 6 insights with no voice had been called minor; the rotated split
counted raw `SPEAKER_NN` labels as names (346 / 806, not 932 / 220).

## 5. Acceptance

Written by a fresh step-5 reviewer (Opus advisor), dated 2026-10-20T08:13 UTC, before any code:
[loss_ranking.acceptance.md](loss_ranking.acceptance.md), criteria A0–A15.

## 6. Build

- `src/podcast_scraper/quality/loss_ranking.py`: `gold_wrong_names` (A1–A5, on the committed S7
  rows, `gold_insights` unchanged), `corpus_hidden` (A6, A7), `seat_check` (A8), `is_spelling`
  (A3's spelling rule on `same_person`'s own threshold), conservation asserted in each (A9).
- `scripts/measure/loss_ranking.py`: the CLI; `--out` must lie outside the git checkout (A11); rows
  as JSONL and `summary.json` there; the record carries counts and case ids only.
- `tests/unit/podcast_scraper/quality/test_loss_ranking.py`: 31 scenario tests on real-shaped temp
  corpora with invented names (A14).
- **One definition line dropped as implied:** A3's "no more tokens than the label" needs no check of
  its own, because each stored token takes a different label token; its mutant is equivalent (no
  test can kill it), so the line went and a comment says why.

**Read while defining the rules (A12), by case id:** the exploration printed 41 (label, stored)
pairs (the top 8 per class and up to 4 per sub-class), in 49 gold cases: dev_v1 c005, c008, c014,
c019, c022, c040, c060, c085, c087, c091, c094; val_v1 v017, v039, v071, v073, v075, v102, v131,
v132, v133, v134, v148, v182, v192, v199, v222, v226, v239, v259, v268, v270, v277, v279, v296,
v313, v316, v343, v374, v392, v393, v440, v450, v485, v493; val_v2 w038, w045, w087, w106, w112.
The seat-check and text-signal probes printed these too: dev_v1 c006, c013, c018, c027, c042,
c043, c044, c065, c066, c070, c084, c089; val_v1 v064, v070, v094, v095, v100, v137, v149, v189,
v194, v207, v243, v262, v474. The step-8 reviewer's held-out sample is drawn from the rest.

## 7. Measure

`scripts/measure/loss_ranking.py --corpus lost/gateworkB6c --gold .test_outputs/naming_gold
--snapshot "snapshot-prod-20261016+0030+0031" --out <scratch>` on **6d3018123** (clean tree),
identical summaries on e9cc19c46 and 6d3018123:

- **Gold, `S7_wrong`** 2,947 (380 / 2,128 / 439): binding 1,897 (swapped 544, duplicated 201,
  rotated third_name 346, owner_unnamed 652, owner_no_insight 154); spelling 52 (val_v1);
  other_person 998. By set: dev 200 / 0 / 180, val_v1 1,397 / 52 / 679, val_v2 300 / 0 / 139. All
  as A5, ±0.
- **Flags:** two or more owners 117, `same_person` owner sensitivity 83, shares a label token 101,
  stored on another voice with an unnamed label 22, as accepted. **`owner_label_weak` 56, not
  A1's 32:** the reviewer's probe checked only the first owner (`owners[:1]`); A3 says "an owner",
  order-independent, which the code applies to every owner. The difference is the 24 insights whose
  weak-labelled owner is not the first in label order.
- **Gold, outside `S7_wrong`:** name_without_support 81, undecidable_named 887, unplaceable_named
  31, as A2.
- **Corpus, S1** 14,841: gaps 6 (no_diagnostics 5, voice_not_found 1), nobody_named 3,557,
  unplaced_principal 5,407 (one_to_one 3,247 in 85 episodes), unplaced_minor 1,269,
  placed_not_recorded 3,293 / 638, none_unplaced 541 / 130. All as A7, ±0; the scoreboard's S1
  equals the classed total (asserted).
- **Seat check:** 26 served cases: 16 right, 9 another person, 1 the label supports no name, as A8.
  Hidden insights on the seated voice: right 457, another person 171, no name 4. So elimination
  alone would put 171 insights under the wrong person to show 457.
- **Mutations (builder's own, before step 8):** 18 mutants over every definition, baseline 69
  passed and 0 failed on the export; 18 killed, 0 survivors (`scratchpad/mutlr/mutate.py`). A first
  run had a failing baseline (the export had no `.git`, so the CLI test could not find the repo):
  every "kill" in it was void and is not counted; a second showed 7 survivors, each now killed by a
  new test (one, the token count, is equivalent and was removed instead).

**Not covered:** the gold is development-tier in part (val_v1 read while tuning) and not a random
sample (val_v2 half weighted to feeds with hidden insights), so the gold classes are rates, not
corpus counts; 2.1's unseen set checks them. The corpus classes describe the record, not the cause
(a stated person "unplaced" may never speak). 85% of S1 lacks a stored LLM decision, so "would the
LLM have solved it" is not measured. The clone is the 10-16 snapshot; prod has moved.
