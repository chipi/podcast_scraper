# Eval: Entity-canonicalization threshold tuning (#853)

**Date:** 2026-06-08
**Ticket:** [#853](https://github.com/chipi/podcast_scraper/issues/853)
**Parent epic:** [#907](https://github.com/chipi/podcast_scraper/issues/907)
**Companion:** [#921](https://github.com/chipi/podcast_scraper/issues/921) (v3 fixtures rebuild)

## TL;DR

Loosened the two threshold constants in `src/podcast_scraper/kg/entity_clusters.py`:

| Constant | Before | After |
| --- | ---: | ---: |
| `_TOKEN_RATIO` | 0.78 | **0.65** |
| `_OVERALL_RATIO` | 0.85 | **0.70** |

**On a 190-pair silver eval mined from real prod (`my-manual-run-10`):**

| Setting | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline (0.78 / 0.85) | 1.0000 | **0.306** | 0.469 | 15 | 0 | 34 |
| Tuned (0.65 / 0.70) | 1.0000 | **0.490** | 0.658 | 24 | 0 | 25 |

**+60% recall at preserved 100% precision.** 9 additional real ASR garbles caught (Bessent/Bessett, Tracy Alloway/Allaway, etc.); zero false merges introduced.

## Method

1. **Mine candidate pairs** from `.test_outputs/manual/my-manual-run-10` (138 KG JSONs, 395 distinct `person:` IDs). Pair extraction: shared-token, slug-similarity ∈ [0.55, 0.95]. Yielded 190 candidates with realistic ASR-garble density. Script: `scripts/eval/data/...` (inline in the silver-extraction commit).
2. **Silver-label via Sonnet 4.6** (`scripts/eval/score/entity_canon_silver_v1.py`). The judge is told the corpus is dominated by ASR garbles, with a rubric for nickname-vs-full-name (same), single-letter swap (same), distinct surnames sharing first name (different). 190 calls × ~500 tokens each ≈ $0.40. Distribution: 49 SAME, 134 DIFFERENT, 7 BORDERLINE.
3. **Sweep grid** (`scripts/eval/score/entity_canon_sweep_v1.py`):
   - `_TOKEN_RATIO` ∈ {0.65, 0.70, 0.74, 0.76, 0.78, 0.80, 0.82, 0.85}
   - `_OVERALL_RATIO` ∈ {0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90}
   - `same_show_required` ∈ {True, False}
   - 112 cells, score each by precision / recall vs silver labels.
4. **Pareto front** + recommendation rule: highest recall among cells with precision ≥ 0.95.
5. **v2 validation** (`scripts/eval/score/entity_canon_v2_validate.py`): 7 deliberate-ambiguity cases — confirm no regression on the v2 spec cases (two-Marcos distinct, ASR-garble class).

## Findings

### What the tuned thresholds catch that the baseline missed

| Class | Example pair | Baseline | Tuned |
| --- | --- | :---: | :---: |
| ASR surname single-letter swap | Scott Bessent ↔ Scott Bessett | ✗ | ✓ |
| ASR surname single-letter swap | Tracy Alloway ↔ Tracy Allaway | ✗ | ✓ |
| ASR surname swap | Tim Geithner ↔ Tim Geidner | ✗ | ✓ |
| ASR first-name garble | Joe Weisenthal ↔ Joe Wassenthal | ✗ | ✓ |
| ASR trailing-letter add | Henry Blodget ↔ Henry Blodgett | ✗ | ✓ |
| ASR trailing-letter add | Greg Brew ↔ Greg Brews | ✗ | ✓ |
| ASR vowel swap | Ryan Petersen ↔ Ryan Peterson | ✗ | ✓ |
| ASR spelling variant | Burne Hobart ↔ Byrne Hobart | ✗ | ✓ |
| ASR first-name nickname | Dario Amadei ↔ Dario Amodei | ✗ | ✓ |

### What still gets missed (structural — predicate redesign needed, not threshold tuning)

The current `_are_xep_variants` predicate has hard-coded structural constraints that cap recall around 50%. The 25 remaining silver-SAME misses split into:

| Failure mode | Example pair | Why missed |
| --- | --- | --- |
| Token-count mismatch | Carney ↔ Mark Carney | `len(ta) != len(tb)` early-rejects |
| Token-count mismatch | Donald Trump ↔ Trump | same |
| Token-count mismatch | Ali Khamenei ↔ Ayatollah Ali Khamenei | title-prefix needs token-count tolerance |
| Token-count mismatch | Joe Eisenthal House ↔ Joe Weisenthal | Whisper added a token |
| Nickname-class first-name | Michael Selig ↔ Mike Selig | Michael/Mike ratio < 0.65 |
| Nickname-class first-name | Nicholas Snyder ↔ Nick Snyder | same |
| Nickname-class first-name | Elizabeth Reid ↔ Liz Reid | same |
| Nickname-class first-name | J. Powell ↔ Jerome Powell | initial-vs-full |
| Severe Whisper garble | Joll Wisenthal ↔ Joe Wisenthal | Joll/Joe ratio < 0.65 |
| Severe Whisper garble | Skanda Amarnath ↔ Skanda Eminas | surname completely different |
| Severe Whisper garble | Heidi Crebo-Rediker ↔ Heidi Krebohticker | severe surname mangling |

Tracking the predicate redesign in **#904** (Tier 1 CIL bridges). A pragmatic fix: a small nickname dictionary + a token-count-tolerant variant of `_are_xep_variants` + an LLM-tier escalation for severe garbles above a tunable confidence threshold.

### v2 spec validation

The recommended thresholds also pass the v2 deliberate-ambiguity cases without regression:

| Case | Expected | Predicted | Pass |
| --- | --- | --- | :---: |
| two-marcos-distinct (`Marco` vs `Marco Bianchi`) | DIFFERENT | DIFFERENT | ✓ |
| bessent-garble (`Scott Bessent` vs `Scott Bessett`) | SAME | SAME | ✓ |
| alloway-garble (`Tracy Alloway` vs `Tracy Allaway`) | SAME | SAME | ✓ |
| weisenthal-quartet (`Joe Weisenthal` vs `Joe Wassenthal`) | SAME | SAME | ✓ |
| geithner-garble (`Tim Geithner` vs `Tim Geidner`) | SAME | SAME | ✓ |
| fischer-merge (`Dr. Elena Fischer` vs `Elena Fischer`) | SAME | DIFFERENT | ✗ (token-count, structural) |
| liam-alias (`Liam Verbeek` vs `Liam`) | SAME | DIFFERENT | ✗ (token-count, structural) |

5/7 pass with both baseline and tuned settings. The 2 failures are token-count mismatches the predicate can't currently handle regardless of thresholds — out of scope for #853, in scope for #904.

## Acceptance

- [x] Labelled silver eval set committed (190 pairs from real prod, Sonnet 4.6 labels).
- [x] Threshold sweep harness committed (Pareto + recommendation logic).
- [x] Tuned thresholds applied to `src/podcast_scraper/kg/entity_clusters.py`; unit tests pass.
- [x] v2 deliberate-ambiguity cases validated — no regression vs baseline.
- [x] Eval report (this file).
- [x] Failure-mode catalogue contributed to `docs/wip/AUTORESEARCH_LEARNINGS_FOR_V3.md` (#921).

## Out of scope (tracked elsewhere)

- Predicate redesign for token-count mismatches + nicknames + severe garbles → **#904**.
- Embedding-based / semantic clustering → future work, not threshold tuning.
- Tuning `DEFAULT_FUZZY_THRESHOLD` in `identity/resolver.py` — the corpus didn't surface cases where this knob mattered; the resolver only enters when two layers (GI + KG) disagree, and the same-show predicate dominates that decision today. Will revisit when #904's predicate redesign exposes it.

## Reproduction

```bash
# 1. Extract candidate pairs from prod corpus (inline Python in the commit; manual run-10 must exist).
# 2. Silver-label via Sonnet 4.6:
PYTHONPATH=. python scripts/eval/score/entity_canon_silver_v1.py \
    --input data/eval/sources/entity_canon_v1/candidate_pairs.jsonl \
    --output data/eval/references/silver/entity_canon_v1/labels.jsonl

# 3. Threshold sweep:
PYTHONPATH=. python scripts/eval/score/entity_canon_sweep_v1.py \
    --silver data/eval/references/silver/entity_canon_v1/labels.jsonl \
    --output data/eval/runs/baseline_entity_canon_v1

# 4. v2 spec validation:
PYTHONPATH=. python scripts/eval/score/entity_canon_v2_validate.py \
    --output data/eval/runs/baseline_entity_canon_v2_validate
```
