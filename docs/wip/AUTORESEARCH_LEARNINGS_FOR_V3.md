# Autoresearch learnings → v3 fixtures (rolling notes)

**Tracking issue:** [#921](https://github.com/chipi/podcast_scraper/issues/921) (v3 fixtures rebuild — incorporate autoresearch learnings).
**Parent epic:** [#907](https://github.com/chipi/podcast_scraper/issues/907) (autoresearch programme).

Living document. Each #907 child contributes failure-mode patterns it
discovers, so v3 (when we build it) simulates real-prod defects directly in
the fixture corpus instead of relying on dev to encounter them in prod.

## How to use this doc

- When a #907 ticket closes, append a section under "Learnings collected" with:
  - The failure modes the ticket exposed (with concrete examples)
  - Pattern data that should land in v3 fixtures (with names, configs, payloads)
  - References to the eval report + tuned-thresholds commit
- Keep entries short — link to the eval report for detail.
- When all #907 children have contributed, the v3 generator pulls from here.

## Learnings collected

### #853 — Entity-canonicalization thresholds (in progress, 2026-06-08)

**Eval set:** 190 candidate near-duplicate person pairs mined from
`.test_outputs/manual/my-manual-run-10` KG outputs. Silver labels at
`data/eval/references/silver/entity_canon_v1/labels.jsonl` (Sonnet 4.6
auto-label, distribution 49 SAME / 7 BORDERLINE / 134 DIFFERENT).

**Real-prod failure-mode catalogue surfaced:**

ASR garble class (currently NOT representable in v2 because v2 doesn't
go through Whisper):

- **Surname single-letter swap** — `Scott Bessent` ↔ `Scott Bessett`,
  `Tracy Alloway` ↔ `Tracy Allaway`, `Ryan Petersen` ↔ `Ryan Peterson`,
  `Tim Geithner` ↔ `Tim Geidner`, `Henry Blodget` ↔ `Henry Blodgett`,
  `Greg Brew` ↔ `Greg Brews`.
- **First-name garble (same surname intact)** — `Joe Weisenthal` ↔
  `Joe Wassenthal`, ↔ `Joll Wisenthal`, ↔ `Jill Wisenthal`. The Odd Lots
  quartet is the canonical multi-garble test.
- **Trailing-letter drop / add** — `Zia Daoud` ↔ `Ziad Daoud`,
  `Dorothea Ioannou` ↔ `Dorothea Yanu`.
- **Vowel swap** — `Noah Brier` ↔ `Noah Bryer`, `Max Spero` ↔ `Max Spiro`,
  `Ray Wang` ↔ `Ray Wong`.
- **Spelling-variant first name** — `Burne Hobart` ↔ `Byrne Hobart`.

Same-first-name distinct-people class (representable in v2 today but
under-exercised):

- `Scott Baki` vs `Scott Bok` — distinct people, single-letter surname
  swap. Tests the SAME pattern as the garble class above; the model has to
  distinguish "garble of one person" from "two people who happen to have
  similar names".
- `Jacob Goldstein` vs `Rob Goldstein` — distinct people, shared surname,
  unrelated first names. v2 only has the two Marcos for this pattern.

Nickname / variant-formal class (v2 has none):

- `Rich Clarida` ↔ `Richard Clarida` — same person, nickname expansion.
- `Greg Brew` ↔ `Gregory Brew` — same person, nickname expansion.

**What v3 should add:**

- A `_garble_variants` field in the v2 spec's `Guest` dataclass. Generator
  emits each guest's name in N forms across episodes, one canonical + 1–2
  garbles drawn from a per-pattern template (single-letter swap, vowel
  swap, trailing-drop, etc.). KG ground truth records the canonical id so
  the bridge builder can be scored.
- A `_nickname_variants` field. Generator alternates between nickname and
  formal in different episodes.
- Two distinct guests with the same first name in different podcasts, each
  with their own garble cluster — combined version of two-Marcos + the
  Bessent garble pattern.
- An `_alias_inventions` pattern — generator picks a first-only guest, then
  in another episode a callback uses a fake invented surname (mimicking
  Whisper's "Liam Verbeek" pattern). KG ground truth records this as
  alias-to-same-person.

**Tuned thresholds (2026-06-08):**

- `_TOKEN_RATIO`: 0.78 → **0.65**
- `_OVERALL_RATIO`: 0.85 → **0.70**
- Recall 0.31 → 0.49 at preserved 1.00 precision.
- Eval report: `docs/guides/eval-reports/EVAL_ENTITY_CANON_2026_06_08.md`.

**Structural recall ceiling (predicate redesign → #904):**

Beyond ~50% recall, the predicate runs into structural constraints that no
threshold can fix:

- Token-count mismatches (`Carney` vs `Mark Carney`, `Donald Trump` vs `Trump`,
  `Ali Khamenei` vs `Ayatollah Ali Khamenei`, `Dr. Elena Fischer` vs
  `Elena Fischer`, `Liam Verbeek` vs `Liam`).
- Nickname-class first-name pairs (`Michael` vs `Mike`, `Nicholas` vs `Nick`,
  `Elizabeth` vs `Liz`, `J. Powell` vs `Jerome Powell`, `Manny` vs `Emmanuel`).
- Severe Whisper garbles where surname similarity drops below 0.65
  (`Joll Wisenthal` vs `Joe Wisenthal`, `Skanda Amarnath` vs `Skanda Eminas`,
  `Heidi Crebo-Rediker` vs `Heidi Krebohticker`).

These need a nickname dictionary + a token-count-tolerant predicate +
optionally an LLM-tier escalation for severe garbles. Tracked in #904.

### #594 — Cleaning autoresearch (2026-06-08)

**Sweep summary:** 5 v2 smoke episodes × 7 (provider, model) cells × 3 temperatures = 105 cleaning calls. Sonnet 4.6 silver baseline. Pairwise tournament + 3-episode prod validation. ~$3–4 total spend.

**Defaults bumped:**

- `anthropic_cleaning_temperature`: 0.2 → 0.4 (prod-validated 2W/0T/0L)
- `gemini_cleaning_temperature`: 0.2 → 0.4 (prod-validated 1W/2T/0L)

**Defaults NOT bumped (and why):**

- `openai_cleaning_temperature`: 0.2 kept. v2 sweep favored 0.4 by only +0.6%, prod validation showed over-cleaning risk (one episode dropped from 11962c to 4733c). Defer until #905's profile-selection harness gives stronger signal.
- `deepseek_cleaning_temperature`: 0.2 already optimal per the sweep; no change.
- Cleaning model choices for all 4 cloud providers: already correct pre-#594. The "16× cost reduction" case the ticket cited (replacing gpt-4o with gpt-4o-mini) had already shipped.

**Real cleaning failure modes for v3 to bake in:**

1. **Native-ad / non-templated sponsor blocks** — Gemini Flash Lite leaves ~0.8 pattern hits per episode on average. The misses are closing-CTA fragments without canonical "brought to you by" markers. v3 should add episodes where sponsor content uses non-template phrasing so cleaning eval doesn't just score template-pattern recall.
2. **Sponsor-shaped real content** — at higher temps OpenAI sometimes treats enthusiastic host recommendations as sponsor content and over-cleans them. v3 episodes that include "honest enthusiasm" host monologues let the cleaning-profile selection (#905) score "preserves real content" precisely.
3. **Larger-model regression** — `gpt-4o` cleaning was worse than `gpt-4o-mini` cleaning (similarity-to-silver 0.45 vs 0.62). Not a fixture artifact, but worth documenting as evidence that "bigger model = better cleaning" is an unsafe assumption — feeds into the #905 cleaning-profile sweep design.
4. **`gemini-2.5-flash` mid-tier instability** — sim drops 0.564 → 0.466 → 0.426 as temp climbs 0.0 → 0.2 → 0.4 on the same task. Worth documenting in the registry: this is a model that does NOT tolerate even modest temperature for structured-text tasks.

**Latency note (operational, not fixture):** Gemini Flash Lite is ~7× faster than gpt-4o-mini for the same prompt (4.5s vs 31s). On a corpus run of N episodes the cleaning stage wall-clock difference is meaningful — flag for #905 profile-selection scoring.

### #904 — Tier 1 sponsor cleaning + CIL bridges + topic clusters (queued)

### #905 — Tier 2 chunking + profile selection (queued)

### #906 — Tier 3 NER + Whisper + prompt tuning (queued)

### #816 — Reliability axis (queued)
