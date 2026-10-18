---
id: s7_s8_gold_insight_weighted
class: full           # an instrument, not a corpus change; run through every step anyway
status: open
issue: 2380
touches:
  - scripts/measure/naming_gate.py
  - src/podcast_scraper/quality/gold_insights.py
---

# S7 / S8 — insight-weighted gold numbers (method step 0, before 0031's step 10)

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §3 (S7, S8) and §4.
Operator 2026-10-18: 0031 waits for S7/S8 and the migration outcome hook (step 0, option B).

## 1. Intake

- Method §3 S7: insight → its first grounded quote → `char_start` → the segment holding it → that
  segment's RAW `speaker` (the pipeline's `_segment_char_spans`); among insights on scored, nameable
  labelled voices: correct / wrong person / unnamed; insights with no grounded quote counted apart.
  S8: insights on voices labelled `confidence: low` or `role: unknown`.
- The v0.1 prototype (`scratchpad/lost/goldgate.py`, 2026-10-17) joined by stored label, which
  double-counts when one name sits on two voices (795 insights on val_v1): NOT VALID.
- Gold sets (`.test_outputs/naming_gold/`, copied to `scratchpad/gold/naming_gold`): dev_v1 101
  cases, val_v1 500, val_v2 150 (751 labelled cases); 51 no longer the served run (dev 9, val_v1 39,
  val_v2 3); val_v1 is a development-tier set now (read while tuning).
- Matcher: `kg.speaker_coherence.same_person` accepts token subsets ("Josh" = "Josh Elman").
- Speakerless insights (no `speaker`): 348 / 1,159 / 160 (dev / val_v1 / val_v2) fell out of v0.1.
- Reuse: `quality/corpus_scoreboard.py` already has the raw-voice quote join (`_Texts`,
  `_voice_of_quote`); `naming_gate.py` has `is_scored` / `outcome` and the label loader.
- 0031's overlap, measured 2026-10-18 (`scratchpad/lost/gold0031.py`): 19 of its 54 episodes are
  gold cases on the served run; gold agrees with the review on 19 of 19.

## 2. Outcome and baseline

The instrument's own numbers, per gold set and per language, on a snapshot clone: S7 correct /
wrong person / unnamed (each split shown / hidden), S8 undecidable, insights with no grounded quote,
insights on unscored voices, cases skipped as no longer served. Baseline: none valid (v0.1 is not).

## 3. Diagnosis by loss

Not a loss diagnosis: the defect to remove is the v0.1 join (by stored label, so a name on two
voices counts their insights twice) and the speakerless gap (an insight with no `speaker` dropped
from the denominator although its quote places it on a voice).

## 4. Prediction

Dated: 2026-10-18T09:50

Known answers, stated before the instrument exists (on `lost/clone30`, gold copy
`scratchpad/gold/naming_gold`, labels = each set's newest `labels_v*`):

- val_v1 v447 (The Flip, Jessica Espinoza): before 0031, SPEAKER_00 (labelled "Jessica Espinoza",
  high) carries 29 insights stored as "Justin Norman": wrong person 29, all hidden; SPEAKER_01
  (labelled "Justin Norman", medium) carries 3, stored unnamed: unnamed 3. After 0031 revision B.3
  (applied on a fresh clone30 copy): v447 has wrong person 0 and unnamed 32.
- Cases skipped as no longer the served run: 51 (dev 9, val_v1 39, val_v2 3), as in the method.
- Conservation, exact per set: correct + wrong + unnamed + S8 undecidable + unscored-role +
  no-grounded-quote = every Insight node of the included cases; no insight counted twice (the v0.1
  defect), checked by insight id.
- The 19 cases 0031 overlaps: their entry voices hold no wrong-person insight after 0031.
- What it must not do: write anything; read a label set it was not given; join by stored label.

## 5. Acceptance

Dated: 2026-10-18T10:05

The fresh reviewer's criteria, written before code after its own reference measurement: [s7_s8_gold_insight_weighted.acceptance.md](s7_s8_gold_insight_weighted.acceptance.md).

## 6. Build

_TODO_

## 7. Measure

_TODO_

## 8. Verdict

_TODO_

## 9. Operator review

_TODO_

## 10. Before apply / deploy

Not applicable (no corpus write); S7/S8 then run on 0031 (before and after) for its step 10.

## 11. After apply / deploy

_TODO_

## 12. Retrospective

_TODO_

## Not covered

_TODO_
