# Eval: Test Fixtures v2 — Baseline vs v1

**Date:** 2026-06-06
**Scope:** Text-derived metrics on the v1 and v2 fixture transcript sets.
**Branch:** `feat/fixtures-v2-rebuild` (issues [#109], [#111], [#900]).

[#109]: https://github.com/chipi/podcast_scraper/issues/109
[#111]: https://github.com/chipi/podcast_scraper/issues/111
[#900]: https://github.com/chipi/podcast_scraper/issues/900

## Why this report exists

The v2 fixtures rebuild is a three-issue change:

- **#111** — per-speaker TTS voice mapping (unblocks RFC-058 diarization testing).
- **#109** — commercial segments in mock transcripts (exercises
  `remove_sponsor_blocks` and the cleaning pipeline).
- **#900** — v2 transcript content rewrite for KG/GIL/CIL signal
  (recurring guests, cross-episode topics, position arcs, edge-case
  name ambiguity).

This report demonstrates — using cheap text-derived metrics — that v2
transcripts encode the patterns the three issues call for, while
remaining usable as drop-in replacements for v1 in the test suite.
Pipeline-derived KG/GIL/CIL metrics (entity counts, bridge counts,
topic-cluster spans) are deferred until the real ML pipeline runs
against the v2 corpus.

## Method

Metrics produced by `scripts/eval/score/fixtures_metrics.py`:

```bash
python scripts/eval/score/fixtures_metrics.py --version v1 \
    --output tests/fixtures/baselines/v1-metrics.json

python scripts/eval/score/fixtures_metrics.py --version v2 \
    --output tests/fixtures/baselines/v2-metrics.json
```

Each invocation reads `tests/fixtures/transcripts/<version>/*.txt`,
parses speaker/timestamp/body lines, and emits per-episode plus
aggregate metrics:

- `body_words` — non-header word count
- `unique_speakers` — count of distinct `Speaker:` labels (excluding
  metadata headers like `Host:`, `Guest:`)
- `proper_nouns` — capitalized words outside a common-titlecase
  allowlist (proxy for entity density)
- `type_token_ratio` — proxy for lexical diversity (higher = less
  repetitive)
- `sponsor_pattern_hits` — count of regex matches against
  `src/podcast_scraper/cleaning/commercial/patterns.py::SPONSOR_PATTERNS`

Aggregate adds cross-episode speaker recurrence per podcast and
cross-podcast proper-noun spans (a rough proxy for the cross-feed
topic spans #900 calls for).

## Headline numbers

| Metric | v1 | v2 | Delta |
| --- | ---: | ---: | ---: |
| Sponsor pattern hits (total) | 0 | 508 | **+508** |
| Sponsor hits per episode (mean) | 0.0 | 15.88 | **+15.88** |
| Type/token ratio (median) | 0.12 | 0.25 | **+0.13** (2.0×) |
| Type/token ratio (mean) | 0.21 | 0.41 | +0.20 |
| Cross-podcast proper nouns | 24 | 38 | +14 (+58%) |
| Words/episode (median) | 1754 | 1380 | -374 |
| Words/episode (mean) | 2857 | 2006 | -851 |
| Episode count | 32 | 32 | — |
| Podcast count | 9 | 9 | — |

### Reading the deltas

**Sponsor pattern hits 0 → 508.** v1 transcripts contained *no* content
matching any phrase in `SPONSOR_PATTERNS`, so the cleaning pipeline had
nothing to detect or remove against fixtures. v2 episodes carry 3
commercial blocks each (opening / mid-roll / closing) with phrases
deliberately drawn from `SPONSOR_PATTERNS`: "This episode is brought to
you by …", "Today's episode is sponsored by …", "Thanks again to our
friends at …". The mid-roll uses an `Ad:` speaker label, exercising the
multi-speaker parser. **Issue [#109] AC met for this metric.**

**Type/token ratio median 0.12 → 0.25.** v1 transcripts were
deliberately repetitive — the same fixed sentences ("On a berm, you
want your eyes through the exit …") repeated dozens of times per
episode. v2 doubles lexical diversity by sourcing claims, transitions,
and elaborations from the talking-points spec and template pools. Not
human-podcast quality, but a meaningful step away from platitude soup.

**Cross-podcast proper nouns 24 → 38.** v1's cross-podcast nouns are
mostly common given names that happen to appear in multiple feeds. v2
adds deliberate cross-podcast entity spans — see "Patterns landed"
below — that produce a real signal for [RFC-075] topic clustering.

[RFC-075]: ../../rfc/RFC-075-corpus-topic-clustering.md

**Words/episode median 1754 → 1380.** v2 medium episodes are ~20%
shorter on average. Acceptable — v1 episodes were padded with repeated
sentences; v2 reaches the same length with more diverse content but
less padding. Tests that depend on transcript length should still pass
(1380 words ≫ any chunking threshold).

## Patterns landed (issue #900 spec)

The v2 generator at `scripts/eval/data/generate_v2_transcripts.py`
encodes the #900 spec patterns as structured data:

### 1. Recurring guests within podcast

- **p02 Priya** — guest in `p02_e01` (on-call rotation) and `p02_e03`
  (security-as-design), referenced in `p02_e02` (engineering
  communication) via the callbacks field. CIL person-bridging
  acceptance target.
- **p01 Liam** — `p01_e01` (trail building) + `p01_e03` mention.
- **p01 Sophie** — `p01_e02` (enduro skills) + `p01_e03` callback
  (mechanic Noah pushing back on Sophie's pressure claim).

v2 baseline metrics show genuine cross-episode speaker recurrence for
the first time:

```text
p01: Liam → 4 eps (e01, e01_fast, multi_e01, multi_e04)
p01: Noah → 3 eps
p01: Sophie → 2 eps
p02: Priya → 2 eps   ← Priya recurrence pattern
p03: Marco → 1 ep    ← dive-Marco (paired with separate p05 Marco)
p06: Liam → 2 eps
```

### 2. Recurring orgs / products within podcast

Each podcast has a curated `recurring_orgs` list whose members appear
multiple times across the podcast's episodes:

| Podcast | Recurring orgs |
| --- | --- |
| p01 | Cascadia Alliance, Shimano, RockShox, Spoke & Wrench |
| p02 | Linear, Datadog, PagerDuty, Sentry, Notion |
| p03 | PADI, Suunto, GoPro, DAN |
| p04 | Adobe, Peak Design, Profoto, Capture One |
| p05 | Vanguard, Wealthfront, Morningstar, iShares |

Sponsor brands are partially drawn from the same pool so KG entity
counts will reflect both sponsor mentions and conversational mentions.

### 3. Cross-podcast topic spans (#900 pattern 4)

- `topic:reliability` — p02 (engineering) + p05 (financial planning)
- `topic:risk-management` — p02 + p03 + p05 + p07 + p09
- `topic:systems-thinking` — p02 + p05 + p07 + p09
- `topic:soil-erosion` — p01 (trail drainage) + p03 (reef
  sedimentation)
- `topic:frame` — p04 (photography framing) — deliberate
  common-word ambiguity for negative-test clustering

The v2 episodes name these `topic:*` strings in their `secondary_topics`
field; the generator surfaces the human form ("reliability",
"risk management") in dialogue prompts. Pipeline-derived
`search/topic_clusters.json` will show whether RFC-075 clustering
picks up these cross-feed spans — measured in a follow-up pipeline
rerun.

### 4. Position arcs

Three episodes encode a position-arc statement (one speaker's view
evolves over time):

- `p01_e02` Sophie — *"I used to think you could brake your way out of
  any line mistake. After 2024 nationals I revised that …"*
- `p02_e03` Priya — *"I used to argue for a single central security
  team. I changed my mind after the 2024 webhook incident — the
  embedded model gets faster fixes and better signal."*
- `p05_e03` Kasper — *"I used to write absolute rules — never whole
  life, always term. After enough edge cases I revised that — the
  right answer is 'it depends, here's how to think about it.'"*

CIL `/api/persons/{id}/positions` acceptance target.

### 5. Edge-case name mismatches

- **Two Marcos** — `p03_e01` "Marco" is a wreck diver with Italian-
  accented voice (mapped to `Luca`). `p05_e03` "Marco" is a tax-loss
  harvesting researcher — different domain, different episode, same
  first name. CIL bridge must keep these as distinct `person:` ids,
  not merge them into one.
- **Daniel disambiguation** — `p05_e01` "Daniel" is the index-investing
  guest; "Daniel" also appears as the v1 *host* voice for `Leo` (now
  superseded by the v2 voice map). The v2 SPEAKER_VOICE_MAP gives
  `p05` guest Daniel its own voice (`Oliver`), separating the surfaces.
- **Surface canonicalization** — `p07` introduces "Dr. Elena Fischer"
  in the welcome line and references "Elena Fischer" later; CIL
  should canonicalize these under one id.

### 6. Deliberate ambiguity for negative testing

`p04` uses `topic:frame` as a primary topic in `p04_e01`,
`p04_e02`, `p04_e03`. The word "frame" also appears in many other
contexts (KG/GIL noun extraction). The cluster pipeline should not
bundle unrelated uses of "frame" into a `tc:` parent — negative-test
target for RFC-075 clustering.

## Speaker voices (#111)

v2 audio uses the per-speaker voice map from RFC-059 §2, replacing the
v1 binary host/guest scheme. Each fixture speaker gets a distinct
macOS `say` voice with accent variety:

| Speaker | Voice | Why |
| --- | --- | --- |
| Maya, Ethan, Rina, Leo, Nora, Alex | Samantha, Alex, Karen, Daniel, Moira, Evan | Hosts — varied accents across podcasts |
| Liam, Sophie, Noah, Priya, Jonas | Fred, Flo, Tom, Isha, Eddy | Guests — distinct from hosts |
| Marco (p03 dive) | Luca | it_IT, separates from p05 Marco |
| Marco (p05 investing) | Hash fallback | en_US, separates from p03 Marco |
| Hanna, Camila, Owen, Ava, Tariq, Elise, Daniel (p05), Isabel, Kasper | Anna, Paulina, Reed, Kathy, Rishi, Amelie, Oliver, Monica, Ralph | Guests with regional accents matching the spec |
| Ad (synthetic mid-roll voice) | Zarvox | Deliberately robotic — distinct from all human voices |

Unmapped speakers fall back to a stable MD5-hash selection over
`FALLBACK_VOICES`. Python's built-in `hash()` is not used because it
varies across runs (`PYTHONHASHSEED`); MD5 is deterministic.

**Acceptance for diarization testing (RFC-058):** post-merge, run
pyannote against v2 fixture audio and verify it produces ≥ 2 speaker
clusters per episode where the spec has ≥ 2 speakers. (Out of scope
for this report — happens when RFC-058 is wired in.)

## What is NOT in this report (deferred)

| Capability | Why deferred | When |
| --- | --- | --- |
| Real KG entity counts per episode (KG extraction) | Requires running spaCy/NER models against v2 corpus | Phase 7b (separate compute-heavy step) |
| GIL Quote yield + offset-verify rate | Requires running summarizer + GIL extractor | Phase 7b |
| CIL bridge counts (`person:`/`topic:`/`org:` spanning episodes) | Requires bridge builder run against v2 KG + GIL | Phase 7b |
| `search/topic_clusters.json` `tc:` parent count | Requires embedding + clustering pipeline | Phase 7b |
| pyannote diarization speaker separation | Requires RFC-058 wired up | When RFC-058 lands |
| Summary content (no brand names in cleaned summaries) | Requires summarizer pipeline run | Phase 7b |

Phase 7b will produce a follow-up `EVAL_FIXTURES_V2_PIPELINE_<date>.md`
with the pipeline-derived metrics once the corpus rerun completes. The
text-derived deltas in this report are necessary but not sufficient
evidence that v2 is "better" — they establish that the spec patterns
are present in the input data; the pipeline rerun establishes that the
downstream layers actually pick them up.

## Risks worth flagging

- **v2 medium episodes shorter than v1.** Word count median dropped
  from 1754 to 1380. Tests that hardcode word counts or token counts
  will need updating in Phase 10. The chunking threshold tests (`p01_e03`
  ≈ 30k → 11.5k words) are the highest-risk consumers.
- **Long-form p08 shorter than v1.** v1 `p08_e01` was 30k+ words; v2 is
  ~14k. Long-context tests targeting the 6–10k token threshold may
  shift between strategies. To verify in Phase 10.
- **Generator-driven prose has visible templating.** Each talking point
  appears 3 times across the 3 passes ("structural" / "operational" /
  "contrarian"). KG/GIL benefit from the repetition; readability
  suffers. Future iteration could swap in LLM-generated prose if the
  pipeline rerun shows it changes nothing material downstream.
- **Same-first-name pattern needs CIL bridge verification.** The
  two-Marcos test is encoded in the spec but unverified — the bridge
  builder may currently merge them. Tracked for Phase 7b.

## Repro

```bash
# Baselines
python scripts/eval/score/fixtures_metrics.py --version v1 \
    --output tests/fixtures/baselines/v1-metrics.json
python scripts/eval/score/fixtures_metrics.py --version v2 \
    --output tests/fixtures/baselines/v2-metrics.json

# Regenerate v2 transcripts from spec
python scripts/eval/data/generate_v2_transcripts.py

# Regenerate v2 audio (macOS only)
python tests/fixtures/scripts/transcripts_to_mp3.py \
    tests/fixtures/transcripts/v2 --overwrite

# Regenerate v2 viewer-validation-corpus
python scripts/build_synthetic_validation_corpus.py \
    --rss-dir tests/fixtures/rss \
    --transcripts-dir tests/fixtures/transcripts/v2 \
    --output tests/fixtures/viewer-validation-corpus/v2
```

## Companion artifacts

- v1 baseline: [`tests/fixtures/baselines/v1-metrics.json`](../../../tests/fixtures/baselines/v1-metrics.json)
- v2 baseline: [`tests/fixtures/baselines/v2-metrics.json`](../../../tests/fixtures/baselines/v2-metrics.json)
- Generator: [`scripts/eval/data/generate_v2_transcripts.py`](../../../scripts/eval/data/generate_v2_transcripts.py)
- Voice map: [`tests/fixtures/scripts/transcripts_to_mp3.py`](../../../tests/fixtures/scripts/transcripts_to_mp3.py)
- Specs:
  [RFC-059](../../rfc/RFC-059-speaker-detection-refactor-test-audio.md),
  [RFC-072](../../rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md),
  [RFC-075](../../rfc/RFC-075-corpus-topic-clustering.md)
