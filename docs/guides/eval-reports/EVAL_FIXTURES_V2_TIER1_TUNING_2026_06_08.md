# Eval: Fixtures v2 Tier 1 tuning — sponsor cleaning + CIL bridges + topic clusters (#904)

**Date:** 2026-06-08
**Ticket:** [#904](https://github.com/chipi/podcast_scraper/issues/904)
**Parent epic:** [#907](https://github.com/chipi/podcast_scraper/issues/907)
**Companion:** [#921](https://github.com/chipi/podcast_scraper/issues/921) (v3 fixtures rebuild), inherits eval setup from [#853](https://github.com/chipi/podcast_scraper/issues/853) and [#594](https://github.com/chipi/podcast_scraper/issues/594)

## TL;DR

| Sub-task | Change shipped | Lift |
| --- | --- | --- |
| A. CIL bridges — predicate redesign | Nickname dictionary + title-prefix strip + family-only-reference matcher in `kg/entity_clusters.py` | Recall **49% → 67%** at preserved 97% precision (vs #853 baseline) on real-prod silver eval |
| B. Sponsor cleaning thresholds | None (defaults verified optimal on v2; real-prod has a coverage gap that's not threshold-tunable) | n/a |
| C. Topic clusters threshold | None (0.75 default is Pareto-optimal on v2) | n/a |

The bulk of #904's value landed in Sub-task A. B and C surfaced structural limitations that belong upstream (v3 fixtures #921 / pattern-set expansion).

---

## Sub-task A: CIL predicate redesign

### What changed

Three additions to `src/podcast_scraper/kg/entity_clusters.py`:

1. **`_NICKNAME_MAP`** — 30 common first-name nickname mappings (Michael/Mike, Elizabeth/Liz/Beth/Betsy, Jerome/Jerry/Jay/J., Robert/Rob/Bob/Bobby, …). Symmetric closure into `_NICKNAME_PAIRS` for either-direction lookup. Trailing-dot normalisation so `J.`, `j`, `j.` all canonicalise.
2. **`_TITLE_PREFIXES`** — ~30 title tokens (Dr / Mr / Mrs / Ms / Prof / Sir / Ayatollah / Rabbi / President / Senator / Judge / General / Ambassador …). Stripped from the start of either name in `_strip_title_prefix`.
3. **`_token_count_tolerant_match`** — handles three patterns the strict aligned predicate rejects:
   - Title-prefix strip on either side → retry strict predicate.
   - Family-only-reference: short side's single token equals long side's LAST token exactly (e.g. `Carney` ↔ `Mark Carney`).
   - First-name-only-alias intentionally NOT added — see "design decision" below.

Plus two surgical fixes to existing logic:

- **Reordered acronym guard** to run AFTER the token-count-tolerance check. Short proper nouns like `Trump` / `Liam` / `Brown` look acronym-ish to that guard but ARE valid family-only-reference candidates when they appear as a token in the other side's name. The tolerant matcher uses exact-token equality, not similarity, so the UPS-vs-USPS concern doesn't apply.
- **Up-front title-prefix normalisation** so `President Trump` vs `Donald Trump` (equal token count after stripping `President`) routes through the family-only-reference path instead of being rejected by the token-ratio test.

### Design decision: NO first-name-only-merge

Initially attempted a first-name-only-merge rule (short single token matches long's FIRST token), then removed. Same predicate shape produces opposite truth on two patterns:

- `Liam` ↔ `Liam Verbeek` — same person (ASR alias, Whisper invented surname)
- `Marco` ↔ `Marco Bianchi` — different people (v2 two-Marcos test from #903)

Without external signal (shared-episode evidence, same-show speaker role, or LLM escalation), the predicate can't distinguish them. Net-positive call: protect every real two-people-same-first-name case in real-prod corpora, lose the synthetic Liam-alias fixture case. The Liam-alias case is tracked for follow-up (#906 NER coalescing + #921 v3 fixtures may carry richer signal).

### Eval results

Silver eval set unchanged from #853: 190 candidate pairs from `.test_outputs/manual/my-manual-run-10`, Sonnet 4.6 silver labels (49 SAME / 134 DIFFERENT / 7 BORDERLINE). Sweep ran against the production predicate (no longer parametrised on thresholds since #853 froze them at 0.65/0.70).

| Predicate version | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pre-#853 (old 0.78/0.85) | 1.0000 | 0.306 | 0.469 | 15 | 0 | 34 |
| #853 ship (0.65/0.70) | 1.0000 | 0.490 | 0.658 | 24 | 0 | 25 |
| **#904 predicate redesign** | **0.971** | **0.673** | **0.795** | **33** | **1** | **16** |

**+18 pp recall** vs #853, **+37 pp recall** vs pre-#853, at 3 pp precision cost.

### Newly caught classes (the +9 TPs vs #853)

| Class | Sample pair | Notes |
| --- | --- | --- |
| Nickname | `Mike Selig` ↔ `Michael Selig`; `Nicholas Snyder` ↔ `Nick Snyder`; `Elizabeth Reid` ↔ `Liz Reid`; `Emmanuel Roman` ↔ `Manny Roman`; `Rich Clarida` ↔ `Richard Clarida` | Nickname dict |
| Initial-vs-full | `J. Powell` ↔ `Jerome Powell` | Dict + trailing-dot normalisation |
| Title prefix | `Dr. Elena Fischer` ↔ `Elena Fischer`; `Ayatollah Ali Khamenei` ↔ `Ali Khamenei`; `President Trump` ↔ `Donald Trump` | Title strip |
| Family-only | `Carney` ↔ `Mark Carney`; `Trump` ↔ `Donald Trump` | Token-count tolerance |

### The 1 FP

`Australia` vs `Western Australia`. Upstream KG misclassified both as `person:` entities. The token-count-tolerant family-only-reference correctly identifies that "Australia" matches the last token of "Western Australia" — but they're places. Can't fix at this layer without an additional signal (geographic stopword list, common-noun detection). Filed against the upstream KG extractor (#906 NER scope), not the predicate. Acceptable trade — 1 FP across 134 silver-DIFFERENT pairs in exchange for 9 new real-prod TPs.

### Remaining FNs (16, structurally out of scope)

Three classes still uncaught:

- **Severe surname garbles below 0.65 similarity** — `Joll Wisenthal` / `Joe Wisenthal`, `Skanda Amarnath` / `Skanda Eminas`, `Heidi Crebo-Rediker` / `Heidi Krebohticker`, `Dorothea Ioannou` / `Dorothea Yanu`. These need LLM-tier escalation OR a tighter ASR (per-tier confidence escalation tracked separately in the autoresearch programme).
- **Whisper-inserted token mid-name** — `Joe Eisenthal House` / `Joe Weisenthal` (Whisper added a spurious "House" token). Token-count mismatch but with internal-token-insertion, not title-prefix or family-only patterns. Would need a fuzzy-substring matcher.
- **First-name-only-alias** — `Liam Verbeek` ↔ `Liam` (intentionally not handled — see design decision above).

All three classes contributed to `docs/wip/AUTORESEARCH_LEARNINGS_FOR_V3.md` for #921.

### Unit-test coverage

14 new parametrised cases in `tests/unit/podcast_scraper/kg/test_entity_clusters.py`:

- 13 in `test_xep_variants_904_nickname_and_token_count_merge` covering nickname / title / family-only positive cases.
- 4 in `test_xep_variants_904_predicate_does_not_overmerge` covering the two-Marcos pattern and the org-side sub-product pattern (`Adobe` / `Adobe Creative Cloud`).
- 1 in `test_xep_variants_904_first_name_only_alias_deferred` pinning the deferred Liam-alias case so a future follow-up can flip it deliberately.

147 KG unit tests total, all pass.

---

## Sub-task B: Sponsor cleaning thresholds

### What we tested

Sweep over `_HIGH_CONFIDENCE_FOR_LARGE_BLOCK` ∈ {0.75, 0.80, 0.85, 0.90} × `_INLINE_STANDALONE_CONFIDENCE` ∈ {0.5, 0.6, 0.7, 0.8} = 16 cells against (a) v2 source corpus, (b) 54-episode `manual-run-10` real-prod corpus. Script: `scripts/eval/score/sponsor_threshold_sweep_v1.py`. Importantly the residual metric excludes `block_end` boundary patterns (transition markers like "welcome back to the show") which #594 incorrectly counted as residual content.

### Finding 1: v2 fixtures don't exercise these thresholds

Every cell on v2 produced identical results: 7.31% chars removed / 3 blocks detected / 100% content recall / 0 residual content. The sponsor blocks in v2 are scripted templates with confidence ≥ 0.85, which is already above the most permissive threshold tested. **No tuning change indicated on v2.**

### Finding 2: real-prod surfaces a coverage gap, not a tuning gap

| high_conf | inline_conf | chars% removed | blocks/ep | content recall | eps with residual |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.75 | 0.5 | 15.48% | 0.41 | 5.8% | 47/54 |
| 0.85 | 0.7 (**default**) | 13.76% | 0.37 | 1.9% | 47/54 |
| 0.90 | 0.7 | 3.93% | 0.22 | 1.9% | 47/54 |

87% of real-prod episodes leave residual sponsor content **regardless of threshold**. Content-pattern recall is 2–6% across the entire grid. The detector design (template-pattern matching) catches almost nothing in real podcasts where sponsor copy is host-read native ads with non-templated phrasing.

**Recommendation: keep defaults (0.85 / 0.7).** Lowering `_HIGH_CONFIDENCE_FOR_LARGE_BLOCK` to 0.75 picks up ~1.7% chars removed on prod but the gains are marginal on a corpus where the fundamental coverage is too thin. The real lever is `SPONSOR_PATTERNS` set expansion (host-read native-ad variants, native-CTA markers without "brought to you by" preamble) — significant scope, contributed to #921 as a v3 fixtures requirement and to the broader cleaning programme as a follow-up.

---

## Sub-task C: Topic clusters

### Threshold sweep

| Threshold | tc:* parents | Cross-feed parents | Notes |
| ---: | ---: | ---: | --- |
| 0.65 | 10 | 4 | Adds 4 near-singleton parents without adding cross-feed value |
| 0.70 | 6 | 4 | Same cross-feed as default |
| **0.75 (default)** | **6** | **4** | Pareto-optimal — max cross-feed, sensible total |
| 0.80 | 4 | 3 | Collapses one cross-feed cluster |
| 0.85 | 4 | 3 | Same as 0.80 |

**No change indicated.** The default 0.75 is Pareto-optimal: 6 tc:* parents (4 cross-feed) — matches the v2 spec targets (`risk-management`, `systems-thinking`, plus emergent `second-order-effects` and `downstream-costs`).

### Frame-negative test: infrastructure correct, live data can't exercise

`scripts/eval/score/topic_clusters_threshold_sweep_v1.py` injects a synthetic non-p04 frame-rooted topic to validate the `_frame_negative_test` assertion fires correctly on conflicting input. Result: even with injection, the test reports `exercised: false`. Reason: a single synthetic injection creates a SINGLETON topic that the output cluster list filters out (the cluster-output schema requires ≥2 unique topic IDs per cluster — same predicate as in the production code path).

**Infrastructure works** — unit tests in `tests/unit/scripts/eval/score/test_topic_clusters_baseline_v2.py` synthesize a violating cluster directly and confirm the detection logic fires.
**Live exercise** requires v3 fixtures (#921) to ship ≥2 non-p04 frame-rooted topics that genuinely cluster. Documented as a v3 requirement.

---

## v3 fixtures contribution (#921)

Two new entries appended to `docs/wip/AUTORESEARCH_LEARNINGS_FOR_V3.md`:

1. **Sponsor-coverage gap on real prod** — v3 should add fixtures with host-read native-ad copy that doesn't use "brought to you by"-style template markers, so detector tuning + pattern-set expansion can be evaluated against realistic content.
2. **Frame-negative live exercise** — v3 should add ≥2 non-p04 frame-rooted topics in genuinely different domains (financial reporting "frame", legal "frame for decision") so the `_frame_negative_test` becomes `exercised: true` on live data.

Plus the existing #853 contribution section gets reinforced — the first-name-only-alias case (`Liam` ↔ `Liam Verbeek`) remains uncaught at the predicate layer; v3 should encode an unambiguous shared-episode signal that future predicate redesigns can use to safely merge.

---

## Acceptance

- [x] CIL bridges: predicate redesign committed; recall lift measured on real-prod silver eval (49% → 67%, +18 pp).
- [x] Sponsor cleaning: threshold sweep on v2 + 54-ep real-prod sample; defaults preserved with explicit rationale.
- [x] Topic clusters: threshold sweep + frame-test exercise attempt; 0.75 default verified Pareto-optimal.
- [x] All v2 deliberate-ambiguity cases from #903 still pass.
- [x] 147 `kg` unit tests + 14 new #904 parametrised cases.
- [x] Eval report (this file).
- [x] v3 contributions logged in `docs/wip/AUTORESEARCH_LEARNINGS_FOR_V3.md`.

## Out of scope (tracked elsewhere)

- LLM-tier escalation for severe garbles — #906 NER tuning may carry richer signal that can be cascaded into the predicate.
- `SPONSOR_PATTERNS` set expansion for native-ad variants — significant scope, dedicated mini-ticket or rolled into #921.
- First-name-only-alias merge — needs same-episode signal or LLM tier; depends on #906 coalescing rules + #921 fixture redesign.
- `Australia` / `Western Australia` upstream KG misclassification — #906 NER scope.

## Reproduction

```bash
# Predicate sweep on the existing silver:
PYTHONPATH=. python scripts/eval/score/entity_canon_sweep_v1.py \
    --silver data/eval/references/silver/entity_canon_v1/labels.jsonl \
    --output data/eval/runs/baseline_entity_canon_v2

# Sponsor-detector threshold sweep on v2:
PYTHONPATH=. python scripts/eval/score/sponsor_threshold_sweep_v1.py \
    --sources data/eval/sources/curated_5feeds_raw_v2 \
    --output  data/eval/runs/baseline_sponsor_thresholds_v1

# Sponsor-detector threshold sweep on real prod:
PYTHONPATH=. python scripts/eval/score/sponsor_threshold_sweep_v1.py \
    --sources .test_outputs/manual/my-manual-run-10/run_20260421-190016_2606de6d/transcripts \
    --output  data/eval/runs/baseline_sponsor_thresholds_prod_v1

# Topic-clusters threshold sweep + frame-negative-test injection:
PYTHONPATH=. python scripts/eval/score/topic_clusters_threshold_sweep_v1.py \
    --kg-run kg_gemini_curated_5feeds_kg_v2_provider \
    --dataset curated_5feeds_kg_v2 \
    --output data/eval/runs/baseline_topic_clusters_threshold_sweep_v1
```
