# Autoresearch learnings → v3 fixtures (rolling notes)

**Tracking issue:** [#921](https://github.com/chipi/podcast_scraper/issues/921) (v3 fixtures rebuild — incorporate autoresearch learnings).
**Parent epic:** [#907](https://github.com/chipi/podcast_scraper/issues/907) (autoresearch programme).

Living document. Each #907 child contributes failure-mode patterns it
discovers, so v3 (when we build it) simulates real-prod defects directly in
the fixture corpus instead of relying on dev to encounter them in prod.

**Status (2026-06-09):** the v3 text/transcript/manifest side has landed
via `scripts/build_v3_fixtures.py`. See
`docs/guides/eval-reports/EVAL_FIXTURES_V3.md` for the v2→v3 delta report
and the failure-mode → episode coverage map. Per-failure-mode landing
status is tracked inline below (LANDED IN V3 / DEFERRED tags).

Audio (multi-voice TTS) is a separate operator PR. The v3 generator emits
per-episode voice/accent hints in transcript comments + manifest so the
audio PR can consume them without re-running the text generator.

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

**LANDED IN V3 (2026-06-09):** all four added. `GuestV3` now carries
`garble_variants`, `nickname_variants`, `severe_garble`, and
`alias_invention`. Episodes pick a surface form via
`guest_surface_overrides` (`canonical | garble:N | nickname:N | severe |
alias`). Ground truth records `{surface, canonical_id, kind}` per surface
form. Canonical garbles baked in: `Liam Verbeek` (p01) →
`Liam Vandermeer` alias, `Jonas Weisenthal` (p02) → full Odd Lots
quartet, `Scott Bessent` (p05) → `Bessett`/`Bessant`, `Rich`↔`Richard
Clarida` (p05), `Hanna Crebo-Rediker` (p03) → severe `Krebohticker`,
`Skanda Amarnath` (p07/p09) → severe `Skanda Eminas`, `Dr. Elena
Fischer` (p07/p09) → severe `Dr. Eliana Fishler`. Two-Marcos: p03 (Marco
Silva) vs p05 (Marco Bianchi). Two-Daniels: p04 (Daniel Olufemi) vs p05
(Daniel Cho).

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

   **LANDED IN V3 (2026-06-09):** `EpisodeV3.native_ad_block: str |
   None` + 4 non-templated host-read templates in
   `NATIVE_AD_TEMPLATES`. Exercised by p01_e02, p01_e03, p03_e01,
   p06_e01. Ground truth records `kind=native_ad` + brand + line index +
   note explaining no canonical marker.

2. **Sponsor-shaped real content** — at higher temps OpenAI sometimes treats enthusiastic host recommendations as sponsor content and over-cleans them. v3 episodes that include "honest enthusiasm" host monologues let the cleaning-profile selection (#905) score "preserves real content" precisely.

   **LANDED IN V3 (2026-06-09):** `EpisodeV3.genuine_recommendation: str
   | None` + 3 templates in `GENUINE_RECOMMENDATION_TEMPLATES`. Marked
   in ground truth with `kind=enthusiastic_recommendation` + explicit
   note `Real host recommendation — NOT a paid sponsor. Cleaner must
   preserve.`. Exercised by p02_e03, p04_e01, p04_e04, p09_e02.
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

**LANDED IN V3 (2026-06-09) for severe-garble + alias_invention:** the
three follow-on FN classes below are now exercised by v3 episodes. Per-class
status appears inline under each item.

Three FN classes still uncaught after #904's predicate redesign — v3 fixtures
should encode each so future predicate work has a test bed:

1. **Severe surname garbles (similarity < 0.65)** — `Joll Wisenthal`/`Joe
   Wisenthal`, `Skanda Amarnath`/`Skanda Eminas`, `Heidi Crebo-Rediker`/`Heidi
   Krebohticker`. Tests an LLM-tier escalation path.

   **LANDED IN V3:** `severe_garble` slot on `GuestV3`. Episodes:
   p02_e02 (Joll Wisenthal), p03_e03 (Hanna Krebohticker), p07_e02
   (Skanda Eminas), p09_e03 (Skanda Eminas in callback).

2. **Whisper-inserted spurious mid-name token** — `Joe Eisenthal House`/`Joe
   Weisenthal`. Tests fuzzy-substring matching beyond title-prefix strip.

   **DEFERRED — v3.1:** the `alias_invention` slot can carry this
   surface form but the canonical "spurious mid-name token" case isn't in
   v3 yet. Defer until #904's predicate redesign explicitly needs it.

3. **First-name-only alias with shared-episode evidence** — same first name
   appears in same episode as the full-name version, e.g. `Liam` (in
   dialogue) + `Liam Verbeek` (in metadata header) of the same episode.

   **LANDED IN V3:** `extra_alias_callbacks: list[tuple[str, str]]` on
   `EpisodeV3`. Exercised by p03_e03 (Marco first-name-only callback +
   Marco Silva canonical in earlier ep), p04_e04 (Ava first-name + Ava
   Lemoine canonical).

**Sponsor-coverage gap on real prod (Sub-task B finding):**

**LANDED IN V3 (2026-06-09):** both sub-items below land via the
`EpisodeV3.native_ad_block` slot + 4 host-read templates that omit the
canonical marker.

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

**LANDED IN V3 (2026-06-09):** `frame_topic_cross_domain` failure-mode tag
exercised by p02_e03 (legal: "frame the decision" / threat modeling), p04_e01
+ p04_e03 + p04_e04 (photography: "an underwater frame is built around
backscatter"), p05_e01 (financial: "frame the market reaction first"). Three
domains × the embedding-disambiguation question is now testable live.

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

   **LANDED IN V3:** see `genuine_recommendation` notes under #594 above.

2. **Multiple silvers per episode** — `silver_aggressive` (Sonnet's most-pruning
   pass) and `silver_conservative` (Sonnet's preserve-everything pass) so cleaning
   profiles can be scored against an explicit point in the recall/precision
   space, not against a single silver whose bias dictates the answer.

   **DEFERRED — silver generation is orthogonal to fixtures:** v3 ships
   transcripts + ground truth labels; the multi-silver pass is generated
   from those transcripts by a separate silver-gen run. Tracked
   alongside `#939` silver generation scripts; out of scope for #921.

**Chunking-quality test bed (Sub-task B finding):**

The v2 long-context fixtures (p07_e01 11.5k words, p08_e01 14.5k, p09_e0X
~5.7k) have synthetic text that produces sensible chunk distributions across
the recommended (chunk_words, overlap) range but doesn't stress the hard
cases. v3 long-context fixtures should add:

1. **Sentence-mid-chunk-boundary content** — a key claim split across two
   chunks at the default 900-word boundary; tests whether overlap captures it.

   **PARTIAL IN V3:** `long_context_chunk_boundary` failure-mode tag exists
   on p07_e01. Concrete sentence-mid-chunk content sketch deferred — the
   v3 long-context renderer currently produces medium-density episodes;
   v3.1 will port the v2 long-context renderer with knob support.

2. **Topic-shift inside the overlap zone** — speaker pivots mid-overlap from
   topic A to topic B; tests whether MAP-stage summary correctly attributes
   each side.

   **DEFERRED — v3.1:** same long-context-renderer port as item 1.

3. **Long episodes that span the hierarchical-vs-extractive switch point** —
   so future ticket can validate the strategy threshold lands in the
   intended regime per episode length, not just by accident.

   **DEFERRED — v3.1:** same long-context-renderer port as item 1.

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

   **LANDED IN V3:** `high_person_density` failure-mode tag. Exercised
   by p04_e02, p04_e04 (host + 3 guests + 2 callback references), and
   p07_e01 (host + 2 guests + 2 cross-episode references).

2. Explicit ground-truth "must-detect" list per episode (extending the
   `EPISODE_VOICES` style mapping in `scripts/eval/score/ner_model_sweep_v1.py`
   to include callback names too), so NER recall is testable without inferring
   from spec dataclass fields.

   **LANDED IN V3:** per-episode ground-truth JSON at
   `tests/fixtures/v3/ground_truth/<episode_id>.json` records every
   surface form (canonical + garbles + nicknames + aliases + callback
   surface forms) with `canonical_id` and `kind`. NER recall scoring
   can iterate over this without inferring from dataclass fields.

**Whisper accent stress (Sub-task B finding):**

`tiny.en` spikes to 23.14% WER on the p04_e01 episode (Daniel en-GB + Kathy
fr-CA voices) — other 4 v2 episodes stay at 4-6% WER. The accent combination
matters more than individual accents: one non-en-US voice is tolerable;
two co-existing non-en-US voices triggers degradation. v3 audio fixtures
should add:

1. **3-accent mix episodes** (e.g. host UK-en + guest 1 fr-CA + guest 2 es-MX)
   so future ASR-tier autoresearch tickets have explicit test beds for
   accent-stress evaluation.

   **LANDED IN V3 (text side):** `PodcastV3.host_accent` +
   `GuestV3.accent` per speaker; embedded in transcript comments and
   manifest `audio_voice_hints` for the multi-voice TTS PR. p04 carries
   host en-GB + Ava fr-CA + Tariq ar-EG + Daniel en-NG (4 accents);
   p07/p09 carry en-AU + de-DE + en-IN. Audio realization waits for the
   operator's separate TTS PR.

2. **One synthetic Whisper-garble episode** — a transcript pre-mangled to
   resemble Whisper output on accented voices (Bessent → Bessett, etc.) —
   so entity-canon + CIL bridge work can be tested without needing actual
   ASR runs.

   **LANDED IN V3:** every `asr_garble` and `asr_garble_severe`-tagged
   episode is the synthetic Whisper-garble episode for the guest in
   question. Entity-canon + CIL bridge work is testable on v3
   transcripts without an ASR pass.

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

   **LANDED IN V3:** `position_arc_multi` failure-mode tag (≥ 2 episodes
   carrying a position arc for the SAME canonical guest). Priya (p02
   e01 + e03 — centralized → embedded security ownership), Elena
   Fischer (p07 e01 → p09 e01 cross-podcast revisit), Skanda (p09 e02 +
   e03 — quarterly → annual labor-data trust).

2. Encode multi-episode recurring-guest patterns more aggressively (currently
   only the synthetic Marco-Bianchi callback in p05_e02 from the #904 work
   exercises this); v3 should have 2-3 cross-episode-recurring guests per
   podcast.

   **LANDED IN V3:** `recurring_guest` failure-mode tag on 11
   episodes. Priya (p02 e01 + e03), Marco Silva (p03 e01 callback in
   e02), Marco Bianchi (p05 cross-references), Ava + Tariq + Daniel
   Olufemi (p04 roundtable e04), Elena Fischer (p07 + p09 cross-pod
   recurrence), Skanda (p07 + p09 cross-pod recurrence).

### #816 — Reliability axis (2026-06-08)

**Eval set:** single representative v2 transcript (`p01_e01`, ~6K chars), 4-candidate panel (`gemini-2.5-flash-lite`, `gemini-2.5-flash`, `gpt-4o-mini`, `claude-haiku-4-5`). Harness: `scripts/eval/score/summary_model_reliability_v1.py`.

**Real-prod failure mode (from 2026-05-24 manual run):** sustained ~15-20% Gemini 503 retry rate over ~3h of batched calls. Not visible at standard autoresearch eval scale (small batches, hand-paced).

**Methodology contributions:**

- Reliability is now a hard floor in summary-model ranking (`success_rate_pct >= 95` at the eval-scale operating point).
- "Effective $/successful-call" replaces nameplate $/call (same number when clean, meaningfully different when the model is taking 503s).
- p50 + p95 under sustained burst replace single-call latency.

**v3 fixtures contribution — what's missing today:**

- **Reliability-burst mode** for v3 fixtures. v2 has no sustained-load test bed. v3 should include a "burst" knob in the fixture-generation tooling so reliability evaluation can run against the same input shape every cycle without requiring a separate corpus.

  **LANDED IN V3 (marker only):** `reliability_burst` failure-mode tag
  on p06_e01 — gives the reliability harness a stable input shape to
  target. The actual sustained-load 503 simulation lives in
  `scripts/eval/score/summary_model_reliability_v1.py` (harness side);
  the fixture is the input.

- **Per-stage `ProviderCallMetrics` export wired into pipeline shutdown.** Today the retry counter is in-process only; future prod runs should auto-emit a `reliability_evidence_*.json` next to `metrics.json` so the autoresearch loop can ingest real-prod evidence without operator manual capture.

  **OUT OF SCOPE for #921 (fixture-orthogonal):** this is a pipeline
  shutdown wiring change, not a fixture knob. Tracked separately.

- **Time-of-day signal** — the 2026-05-24 prod observation is undifferentiated across the 3h window. v3 fixtures should drive a "stress at known load profile" pattern (ramp up over N minutes, hold, ramp down) so reliability measurement can show the 503-emergence curve, not just a steady-state number.

  **DEFERRED — v3.1:** the time-of-day ramp is a harness-side
  concern; v3 ships the input shape. Defer until the reliability
  harness needs the ramp profile.

**Tuned thresholds shipped:** no — the methodology is shipped; the model selection is unchanged. Composite ranking with the reliability axis still favors `gemini-2.5-flash-lite` by a wide margin (3-10× cost dominance, 3× latency dominance).

**Cross-reference:** Reliability burst against the existing per-provider summarize endpoint is reusable for cross-stage evaluation when future tickets bring GI / KG / speaker models into autoresearch scope.

### #928 / #929 / #930 / #931 — DGX-vs-cloud autoresearch (batch 3, 2026-06-10/11)

**Eval set:** 5 v2 audio fixtures (`p0[1-5]_e01.mp3`, ~5 min each)
across four transcription backends (MPS / CPU / DGX `whisper-openai`
on `:8002` / DGX faster-whisper on `:8000`), three summary backends
(Ollama qwen3.5:35b at Q4 / vLLM R1-Distill-32B at bf16 / vLLM
Qwen3.6-35B-A3B at bf16 added via Cell C), and a 3-way pyannote
diarization run (MPS / CUDA / CPU). Reports under
`docs/guides/eval-reports/EVAL_{DIARIZATION,SUMMARY,TRANSCRIPTION,HYBRID_ROUTING}_*_2026_06.md`.

**Real-prod failure-mode catalogue surfaced (v3-relevant):**

1. **Single-voice TTS confirmed-blocks diarization at the fixture
   level.** The #930 3-way pyannote comparison was contaminated:
   all three platforms (MPS / CUDA / CPU) collapsed to a single
   detected speaker because every voice in v2 is `say --voice Alex`.
   This re-confirms the gap #934 already tracks — included here
   because the batch-3 numbers are the cleanest empirical evidence
   yet that the contamination is total, not partial.

2. **Long-form audio is the trigger condition for openai-whisper
   autoregressive runaway.** The DGX whisper service produced WER
   3.20 mean (5–9× extra hyp words) pre-fix because the API contract
   forced `temperature=0.0` scalar, disabling openai-whisper's
   built-in fallback schedule. Triggered reliably on the 5-min v2
   episodes; would trigger harder on the 30-90-min real podcasts
   the operator wants in production. Post-fix, the same episodes
   produced WER 0.102 / 4.56× realtime — within scoring noise of
   MPS. The bug fix lives in
   `infra/dgx/whisper-server/app.py` + the temperature-schedule
   nuance is documented in the consumer-contract section of the
   vllm-autoresearch README (moved to
   <https://github.com/chipi/agentic-ai-homelab/> on 2026-06-12).
   **Implication for v3
   audio**: episodes that span the autoregressive-runaway trigger
   window (multi-minute, conversational, technical-vocab) are
   exactly the v3 axis that will catch this class of bug before it
   reaches prod. The v3 text side already has them
   (`p07_e01` 11.5k words, `p08_e01` 14.5k); the v3 audio PR
   should render them as similarly-long voiced episodes, not just
   as 5-min clips.

3. **Per-episode acoustic difficulty asymmetry on temperature-
   fallback path.** Pure-CPU openai-whisper produced WER 0.281 on
   `p05_e01` while the other 4 episodes stayed at 0.07–0.12 mean
   (CPU 5-ep mean 0.137). Same voice, same fixture set —
   variance comes from openai-whisper's temperature schedule being
   non-deterministic once it fires (fallback retry at higher
   temperatures introduces stochasticity). The episode-level WER
   variance under fallback-mode decoding is ~3× the clean-decode
   variance. **Implication for v3 audio**: building a transcription-
   benchmark suite requires *deliberate* coverage of episodes that
   trigger the fallback path (faster-paced, denser vocab, longer
   uninterrupted spans). Clean-decode-only fixtures under-represent
   the failure regime.

4. **Same-model-different-server ties within scoring noise (#928
   Cell C).** vLLM-served `Qwen3.6-35B-A3B` at bf16 scored 4.90
   mean vs Ollama-served `qwen3.5:35b` at Q4_K_M's 5.00 mean
   (Sonnet 4.6 + GPT-5.4 cross-check, 100% agreement, no
   contested). The parent eval's 1.75-point gap (Ollama 5.00 vs
   vLLM R1-Distill 3.25) was the **model choice**, not the
   serving stack. **Implication for v3 / autoresearch
   methodology**, not v3 fixtures: future cross-stack model
   sweeps can hold one variable fixed without needing fixture
   changes. The fixture set produced well-separated scores
   across *different* models — which is the diversity v3 already
   delivers — so no new fixture coverage is needed for this
   class.

5. **R1-Distill reasoning prose leak as an output-side failure
   mode.** R1-Distill emits "thinking process" prose (`Okay, so
   I need to summarize this episode…`) before the actual summary
   when run without `chat_template_kwargs={enable_thinking:
   False}` (Qwen3.6 has a clean toggle; R1-Distill does not).
   This contributed to R1's low coherence (2.2) + fluency (2.4)
   in the parent #928. **Not a v3-fixture-side failure mode** —
   the leak is a property of the model's output distribution,
   not the input. Filed as **#961** (prompt-side fix). Mentioned
   here so future eval reports don't re-discover it from scratch.

**What v3 should add (audio side, consumed by the operator's TTS PR
when #934 lands):**

- **Multi-voice TTS realization of the v3 text fixtures.**
  Re-confirmed need; the text side already emits per-episode
  voice/accent hints via `PodcastV3.host_accent` /
  `GuestV3.accent` + manifest `audio_voice_hints` (per #906
  notes above). Batch-3 adds no new requirement here beyond
  "ship the audio side so #930 / #934 can verdict on diarization
  quality, not platform speed."
- **Long-form voiced episodes (5+ min, ideally 30+ min for
  prod-shape benchmarks).** The v3 text side already has
  `p07_e01` 11.5k words and `p08_e01` 14.5k words; the audio PR
  should render at least one of these as a multi-minute voiced
  episode. Real-podcast 90-min benchmarking is a separate axis
  tracked by **#959**.
- **Episodes that reliably trigger the temperature-fallback
  path** — see point 3 above. Fast-paced + technical-vocab
  episodes from the existing v3 text set already qualify; the
  audio renderer just needs to preserve the density when
  vocalizing.

**What v3 should NOT change (axes already adequately covered):**

- Diversity across summary candidates: v3 episodes produce well-
  separated scores across different summary models (Qwen3.6 vs
  R1-Distill = 1.75-point gap). Sufficient diversity for cross-
  model sweeps.
- Diversity across diarization candidates: blocked entirely by
  the single-voice TTS issue; once #934 lands multi-voice
  rendering, the existing v3 episode set has enough speaker
  variety (host + 2-4 guests per episode, per #906 notes) to
  produce a meaningful diarization eval. No new v3 fixture
  axis needed.

**Tuned thresholds shipped:** no — this batch shipped a
*configuration* fix (whisper-server `temperature` API contract
change) and *infrastructure* default flips (vLLM
`26.05-py3 + Qwen3.6-35B-A3B + --max-num-seqs 128` as the new
default in `deploy.py`; Ollama qwen3.5:35b stays the
`cloud_with_dgx_*` summary default with operational-simplicity
as the new justification, replacing quality as the prior
justification). Neither is a fixture-side threshold.

**Cross-ref:** Late-batch findings filed as follow-ups for next
sessions — **#956** (DGX-over-Tailscale client resilience),
**#957** (speaches/faster-whisper container root-cause; the
faster-whisper sweep had a separate bug from openai-whisper's;
empty output 4/5 episodes, hallucination 1/5),
**#958** (Cell D / Cell E quantization isolation for #928),
**#959** (real-podcast 90-min validation across the post-fix
4-way), **#960** (vLLM as a first-class backend in
`autoresearch_track_a.py`, replacing the standalone
`summary_vllm_predict_v1.py` workaround used this batch),
**#961** (R1-Distill reasoning-suppressed prompt — addresses
the failure mode in point 5), **#962** (Gemini speaker
detector provider deploy — unblocks the `cloud_*` slot of
#930's panel), **#963** (re-test DGX whisper under contention
now that the temperature bug is fixed), **#964** (Wave audio
hardening umbrella from the WIP audit).
