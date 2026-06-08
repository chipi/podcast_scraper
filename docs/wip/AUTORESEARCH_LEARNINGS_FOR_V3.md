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

### #904 — Tier 1 sponsor cleaning + CIL bridges + topic clusters (2026-06-08)

**CIL predicate-redesign findings:**

The first-name-only-alias case (`Liam` ↔ `Liam Verbeek`) was deliberately NOT
merged because it has identical predicate shape to the two-Marcos
distinct-people case. v3 should bake an unambiguous shared-episode or
shared-show signal so a future predicate redesign can safely first-name-merge
when external evidence supports it — e.g. both names appear in the same
episode's speaker list, OR both appear in the same show's recurring-guest
list under different surface forms.

Three FN classes still uncaught after #904's predicate redesign — v3 fixtures
should encode each so future predicate work has a test bed:

1. **Severe surname garbles (similarity < 0.65)** — `Joll Wisenthal`/`Joe
   Wisenthal`, `Skanda Amarnath`/`Skanda Eminas`, `Heidi Crebo-Rediker`/`Heidi
   Krebohticker`. Tests an LLM-tier escalation path.
2. **Whisper-inserted spurious mid-name token** — `Joe Eisenthal House`/`Joe
   Weisenthal`. Tests fuzzy-substring matching beyond title-prefix strip.
3. **First-name-only alias with shared-episode evidence** — same first name
   appears in same episode as the full-name version, e.g. `Liam` (in
   dialogue) + `Liam Verbeek` (in metadata header) of the same episode.

**Sponsor-coverage gap on real prod (Sub-task B finding):**

The detector's template-pattern approach catches ~2–6% of sponsor content on
real podcasts because most sponsor copy is host-read native ads with
non-templated phrasing. v3 should add:

1. Host-read native-ad blocks (no "brought to you by" / "today's episode is
   sponsored by" marker) where the host casually pivots into sponsor content
   mid-conversation.
2. Native-CTA markers without the canonical "visit X.com" preamble — phrases
   like "we have a special deal for our listeners", "head over to our
   sponsor's website", "use the link in show notes".

These let SPONSOR_PATTERNS-set expansion be evaluated meaningfully on
fixtures, not just on a sample of real prod where there's no ground truth.

**Frame-negative live exercise (Sub-task C finding):**

The `_frame_negative_test` infrastructure works (unit tests confirm it fires
on synthetic violating clusters), but the live v2 corpus can't exercise it
because `topic:frame` only exists in p04. v3 should add ≥2 non-p04
frame-rooted topics in genuinely different semantic domains (e.g. legal
"frame for decision-making", financial "frame the market reaction") so the
cluster predicate's discrimination can be tested live, not just by synthetic
injection. Per-domain "frame" labels also stress-test the embedding's
domain-disambiguation: do the photography and legal "frame" embeddings
diverge enough to stay in separate clusters?

### #905 — Tier 2 chunking + profile selection (2026-06-08)

**v4 over-cleaning evidence (Sub-task A finding):**

Pairwise judge on v2 smoke prefers `cleaning_v3` over `cleaning_v4` by 10W-0L-5T,
even though v4 has higher cheap-metric scores (most chars removed, highest
sim-to-silver). The judge consistently flags v4 as removing 3% of real content
along with sponsor blocks. v3 fixtures should add:

1. **"Sponsor-shaped real content"** — host enthusiastic recommendations, native
   product mentions in conversation, off-topic personal asides — content that
   *looks* like sponsor copy but isn't. Lets future profile sweeps measure
   "preserves real content" precisely instead of inferring it from char-removal
   bounds.
2. **Multiple silvers per episode** — `silver_aggressive` (Sonnet's most-pruning
   pass) and `silver_conservative` (Sonnet's preserve-everything pass) so cleaning
   profiles can be scored against an explicit point in the recall/precision
   space, not against a single silver whose bias dictates the answer.

**Chunking-quality test bed (Sub-task B finding):**

The v2 long-context fixtures (p07_e01 11.5k words, p08_e01 14.5k, p09_e0X
~5.7k) have synthetic text that produces sensible chunk distributions across
the recommended (chunk_words, overlap) range but doesn't stress the hard
cases. v3 long-context fixtures should add:

1. **Sentence-mid-chunk-boundary content** — a key claim split across two
   chunks at the default 900-word boundary; tests whether overlap captures it.
2. **Topic-shift inside the overlap zone** — speaker pivots mid-overlap from
   topic A to topic B; tests whether MAP-stage summary correctly attributes
   each side.
3. **Long episodes that span the hierarchical-vs-extractive switch point** —
   so future ticket can validate the strategy threshold lands in the
   intended regime per episode length, not just by accident.

**Production-default migration deferred:**

The judge result is strong but small-sample (30 verdicts on 5 episodes). Before
flipping `DEFAULT_PROFILE` from v4 to v3 we need either (a) broader judge sample
or (b) decision that the metric-vs-judge tradeoff is settled. Until then, the
4 hardcoded v4 fallbacks (`ml_provider.py`, `hybrid_ml_provider.py`,
`summarizer.py`, `model_registry.py`) stay on v4. v3 fixtures should make this
sample larger by including the "sponsor-shaped real content" + "multiple
silvers per episode" patterns above — they're what the broader judge pass
would need.

### #906 — Tier 3 NER + Whisper + prompt tuning (2026-06-08)

**NER coverage gap (Sub-task A finding):**

`en_core_web_sm` misses 17% of expected hosts/guests on v2 (96.7% recall with
`_trf`, 83.3% with `_sm`). The misses concentrate on (a) less-prominent
secondary persons and (b) callback references where the same person is named
via different surface forms across episodes. v3 fixtures should add:

1. Episodes with higher named-person density (host + 2 guests + 2 callback
   references), so NER recall can be measured at density that matches real
   podcast content. Current v2 max is ~4 named persons/ep which
   underrepresents the realistic case.
2. Explicit ground-truth "must-detect" list per episode (extending the
   `EPISODE_VOICES` style mapping in `scripts/eval/score/ner_model_sweep_v1.py`
   to include callback names too), so NER recall is testable without inferring
   from spec dataclass fields.

**Whisper accent stress (Sub-task B finding):**

`tiny.en` spikes to 23.14% WER on the p04_e01 episode (Daniel en-GB + Kathy
fr-CA voices) — other 4 v2 episodes stay at 4-6% WER. The accent combination
matters more than individual accents: one non-en-US voice is tolerable;
two co-existing non-en-US voices triggers degradation. v3 audio fixtures
should add:

1. **3-accent mix episodes** (e.g. host UK-en + guest 1 fr-CA + guest 2 es-MX)
   so future ASR-tier autoresearch tickets have explicit test beds for
   accent-stress evaluation.
2. **One synthetic Whisper-garble episode** — a transcript pre-mangled to
   resemble Whisper output on accented voices (Bessent → Bessett, etc.) —
   so entity-canon + CIL bridge work can be tested without needing actual
   ASR runs.

**Prompt v2-aware variant (Sub-task C finding):**

A hand-designed v2-aware paragraph prompt (adds explicit "position changes" +
"recurring guests" callouts) **sweeps 5-0** over the current production
`long_v1.j2` on v2 smoke. The win comes from explicitly surfacing structural
patterns (position arcs, callback recurrence) that v1 leaves implicit. The
v2-aware variant shipped at
`src/podcast_scraper/prompts/anthropic/summarization/long_v2.j2`. v3 fixtures
should:

1. Ensure each podcast has at least one episode with a `position_arc` (v2
   has 3 such episodes out of 15 — biased toward p01_e02, p02_e03, p05_e03).
2. Encode multi-episode recurring-guest patterns more aggressively (currently
   only the synthetic Marco-Bianchi callback in p05_e02 from the #904 work
   exercises this); v3 should have 2-3 cross-episode-recurring guests per
   podcast.

### #816 — Reliability axis (2026-06-08)

**Eval set:** single representative v2 transcript (`p01_e01`, ~6K chars), 4-candidate panel (`gemini-2.5-flash-lite`, `gemini-2.5-flash`, `gpt-4o-mini`, `claude-haiku-4-5`). Harness: `scripts/eval/score/summary_model_reliability_v1.py`.

**Real-prod failure mode (from 2026-05-24 manual run):** sustained ~15-20% Gemini 503 retry rate over ~3h of batched calls. Not visible at standard autoresearch eval scale (small batches, hand-paced).

**Methodology contributions:**

- Reliability is now a hard floor in summary-model ranking (`success_rate_pct >= 95` at the eval-scale operating point).
- "Effective $/successful-call" replaces nameplate $/call (same number when clean, meaningfully different when the model is taking 503s).
- p50 + p95 under sustained burst replace single-call latency.

**v3 fixtures contribution — what's missing today:**

- **Reliability-burst mode** for v3 fixtures. v2 has no sustained-load test bed. v3 should include a "burst" knob in the fixture-generation tooling so reliability evaluation can run against the same input shape every cycle without requiring a separate corpus.
- **Per-stage `ProviderCallMetrics` export wired into pipeline shutdown.** Today the retry counter is in-process only; future prod runs should auto-emit a `reliability_evidence_*.json` next to `metrics.json` so the autoresearch loop can ingest real-prod evidence without operator manual capture.
- **Time-of-day signal** — the 2026-05-24 prod observation is undifferentiated across the 3h window. v3 fixtures should drive a "stress at known load profile" pattern (ramp up over N minutes, hold, ramp down) so reliability measurement can show the 503-emergence curve, not just a steady-state number.

**Tuned thresholds shipped:** no — the methodology is shipped; the model selection is unchanged. Composite ranking with the reliability axis still favors `gemini-2.5-flash-lite` by a wide margin (3-10× cost dominance, 3× latency dominance).

**Cross-reference:** Reliability burst against the existing per-provider summarize endpoint is reusable for cross-stage evaluation when future tickets bring GI / KG / speaker models into autoresearch scope.
