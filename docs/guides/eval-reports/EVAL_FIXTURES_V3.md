# EVAL: Fixtures v3 — Autoresearch-Learnings-Driven Rebuild

**Status:** v3 generator landed (text/transcript/manifest side; audio is the
operator's separate multi-voice TTS PR).
**Issue:** [#921](https://github.com/chipi/podcast_scraper/issues/921)
**Parent epic:** [#907](https://github.com/chipi/podcast_scraper/issues/907)
**Generator:** `scripts/build_v3_fixtures.py`
**Output:** `tests/fixtures/transcripts/v3/`,
`tests/fixtures/v3/ground_truth/`, `tests/fixtures/v3/manifest.json`,
`data/eval/datasets/curated_5feeds_smoke_v3/manifest.{yaml,json}`,
`data/eval/datasets/curated_5feeds_smoke_v3.json` (flat-file mirror).
**Tests:** `tests/integration/eval/test_v3_fixtures.py` (22 assertions).

## Summary

v3 extends the v2 fixture model with explicit knobs for the failure-mode
catalogue harvested from the autoresearch programme
(`docs/wip/AUTORESEARCH_LEARNINGS_FOR_V3.md`,
`docs/wip/PROD_RUN_ANALYSIS_100EP.md`). Each prod-observed failure mode is
now a structured field on the `GuestV3` / `EpisodeV3` / `PodcastV3` data
classes, exercised by `≥ 1` episode, and recorded in the per-episode
ground-truth labels so eval scoring can score against it.

* **9 podcasts, 25 episodes** (vs v2's 9 podcasts, 32 episode files —
  several of v2's were edge-case singletons; v3 trades those for
  failure-mode-driven episodes).
* **16 failure-mode tags** in the vocabulary, every one exercised.
* **Deterministic**: re-running the generator produces bit-identical
  transcripts (MD5 of `pod_id:ep_id` seeds the per-episode RNG).
* **v2 untouched** — additive only. v2 transcripts, audio, and datasets
  remain on disk; `tests/fixtures/FIXTURES_VERSION` stays at `v2` until
  downstream tests are verified to pass on v3.

## Failure-mode coverage

| Tag                          | Episodes | Source learning                                     |
| ---------------------------- | -------: | --------------------------------------------------- |
| `asr_garble`                 |       12 | #853 — Whisper-style single-letter swap garbles     |
| `asr_garble_severe`          |        4 | #904 — severe surname garbles (similarity < 0.65)   |
| `nickname_variant`           |        2 | #853 — Rich ↔ Richard Clarida                       |
| `alias_invention`            |        2 | #853 / #904 — Liam → invented "Liam Verbeek" pattern|
| `same_first_distinct`        |        4 | #853 — two-Marcos test; Daniel p04 vs p05           |
| `position_arc_multi`         |        4 | #906 — position arcs across ≥ 2 episodes            |
| `recurring_guest`            |       11 | #906 — cross-episode callbacks per podcast          |
| `native_ad`                  |        2 | #594 — non-templated host-read sponsor copy         |
| `genuine_recommendation`     |        2 | #905 — sponsor-shaped real content                  |
| `low_grounding_dialogue`     |        2 | PROD_RUN — omnycontent-shape dialogue-heavy         |
| `zero_host_ner`              |        2 | PROD_RUN — NPR-shape host evading spaCy NER         |
| `multi_accent`               |        8 | #906 — ≥ 2 non-en-US voices per episode             |
| `frame_topic_cross_domain`   |        4 | #904 — frame: photography / legal / financial       |
| `high_person_density`        |        3 | #906 — host + 2 guests + ≥ 2 callbacks              |
| `long_context_chunk_boundary`|        1 | #905 — key claim across default 900-word boundary   |
| `reliability_burst`          |        1 | #816 — sustained-load 503 simulation hook           |

## Per-failure-mode design notes

### ASR garble (`asr_garble`, `asr_garble_severe`)

Each `GuestV3` carries a `garble_variants: list[str]` field. The generator
picks the surface form per episode via `guest_surface_overrides`
(`canonical` | `garble:0` | `garble:1` | `nickname:0` | `severe` | `alias`).
Ground truth records the chosen surface form and the canonical id so
entity-canon scoring can verify the bridge.

**Canonical garble examples landed in v3:**

* `Liam Verbeek` (p01) — alias_invention slot holds `Liam Vandermeer` (fake surname)
* `Jonas Weisenthal` (p02) — full Odd Lots quartet (`Jonas Wassenthal`, `Jonas Wisenthal`, severe `Joll Wisenthal`)
* `Scott Bessent` (p05) — `Scott Bessett`, `Scott Bessant`
* `Richard Clarida` (p05) — nickname `Rich Clarida`
* `Hanna Crebo-Rediker` (p03) — severe `Hanna Krebohticker`
* `Skanda Amarnath` (p07/p09) — severe `Skanda Eminas`
* `Dr. Elena Fischer` (p07/p09) — `Dr. Elena Fisher`, severe `Dr. Eliana Fishler`

### Native ad / non-templated sponsor blocks (`native_ad`)

`EpisodeV3.native_ad_block: str | None` — host-read pitch with no
`brought to you by` / `sponsored by` marker. Inserted between the
structural and operational passes so it's surrounded by real conversational
context. Ground truth records `kind="native_ad"` + brand + line index.

Drawn from #594's "non-templated phrasing" finding — Gemini Flash Lite
leaves ~0.8 native-ad pattern hits per real-prod episode unflagged because
they don't carry the template-pattern marker.

### Sponsor-shaped real content (`genuine_recommendation`)

`EpisodeV3.genuine_recommendation: str | None` — enthusiastic host
recommendation that LOOKS like sponsor copy ("real talk: Bloomberg is the
tool I'd pay double for") but isn't actually paid. Ground truth records
`kind="enthusiastic_recommendation"` with an explicit `note: "Real host
recommendation — NOT a paid sponsor. Cleaner must preserve."`.

Drawn from #905's `v4 over-cleaning` finding — the v4 cleaning profile was
over-pruning 3% real content along with sponsor blocks.

### Multi-accent stress (`multi_accent`)

`PodcastV3.host_accent` + `GuestV3.accent` per speaker. The generator
embeds the hints in a `#fixture-v3: voice=...` comment line that the
separate multi-voice TTS audio PR can consume.

p04 (Frame & Light) carries the host en-GB + Ava fr-CA + Tariq ar-EG +
Daniel en-NG quartet — exercises the accent-co-occurrence pattern from
the #906 `tiny.en spikes to 23.14% WER` finding.

### Position-arc episodes (`position_arc_multi`)

A guest's `position_arc` statement now appears on multiple episodes for
the same canonical id — exercised by:

* Priya (p02) e01 (pre-shift: centralized security) + e03 (post-shift: embedded)
* Elena Fischer (p09) e01 + cross-reference to p07 — two-years-later revision
* Skanda (p09) e02 + e03 — quarterly-to-annual labor-data trust revision

### Recurring-guest patterns (`recurring_guest`)

Same canonical guest appears in ≥ 2 episodes per podcast (and across
podcasts in p07 ↔ p09). Callbacks carry the guest's surface form which may
be a garble — exercises the CIL bridge under realistic ASR noise.

### Low-grounding feed patterns (`low_grounding_dialogue`)

`EpisodeV3.low_grounding_filler_turns: int` injects N host/guest
turn-pairs of meandering, non-attributable dialogue ("So, like, yeah —"
type filler) — modeled on the dialogue-heavy `omnycontent` pattern from
`PROD_RUN_ANALYSIS_100EP.md` Finding 1 (58% grounding rate on omnycontent
vs 91% corpus average).

p06 (The Drift) is the canonical low-grounding test podcast.

### NER zero-host patterns (`zero_host_ner`)

`PodcastV3.zero_host_ner: bool` + host name styled to evade spaCy
`en_core_web_sm` PERSON detection. p08 (`Public Hour` with host
`A. correspondent`) models the NPR-shape finding from
`PROD_RUN_ANALYSIS_100EP.md` Finding 5 (NPR shows 0 host detections / 147
mentioned).

### Long-context chunk-boundary stress (`long_context_chunk_boundary`)

p07_e01 (Elena Fischer sustainability) carries the long-context tag plus
a `high_person_density` tag — exercises both the chunk-boundary risk and
the dense-mention case in one episode. (Concrete sentence-mid-chunk
content is sketched as a v3.1 follow-up; the tag is exercised, the
content depth lands when long-context generation gets the same knob
treatment.)

### Reliability-burst hook (`reliability_burst`)

Marker tag on p06_e01. The actual sustained-load simulation lives in the
autoresearch reliability harness
(`scripts/eval/score/summary_model_reliability_v1.py`); the v3 fixture
just provides a stable input shape the harness can target every cycle.

## v2 → v3 deltas

### Schema additions

| Field                                | On                   | Purpose                                |
| ------------------------------------ | -------------------- | -------------------------------------- |
| `garble_variants: list[str]`         | `GuestV3`            | ASR-style surface forms                |
| `nickname_variants: list[str]`       | `GuestV3`            | Rich ↔ Richard                         |
| `severe_garble: str \| None`         | `GuestV3`            | similarity < 0.65 garble               |
| `alias_invention: str \| None`       | `GuestV3`            | Whisper-invented fake surname          |
| `accent: str`                        | `GuestV3`            | TTS hint for the audio PR              |
| `failure_modes: list[str]`           | `EpisodeV3`          | manifest coverage tag                  |
| `guest_surface_overrides: dict`      | `EpisodeV3`          | per-episode surface-form choice        |
| `native_ad_block: str \| None`       | `EpisodeV3`          | non-templated sponsor block            |
| `genuine_recommendation: str \| None`| `EpisodeV3`          | sponsor-shaped real content            |
| `low_grounding_filler_turns: int`    | `EpisodeV3`          | omnycontent-shape filler turn pairs    |
| `extra_alias_callbacks: list[tuple]` | `EpisodeV3`          | first-name / garble callbacks          |
| `host_accent: str`                   | `PodcastV3`          | TTS hint for the audio PR              |
| `zero_host_ner: bool`                | `PodcastV3`          | NPR-shape NER-evading host name        |

### Backwards compatibility

* `tests/fixtures/transcripts/v2/` — unchanged.
* `tests/fixtures/audio/v2/` — unchanged.
* `tests/fixtures/viewer-validation-corpus/v2/` — unchanged.
* `tests/fixtures/baselines/v2-metrics.json` — unchanged.
* `tests/fixtures/FIXTURES_VERSION` — still `v2`. Bump to `v3` only after
  every downstream test in `tests/integration/` is verified to pass on v3
  paths. That bump is **not** in this PR.
* `data/eval/datasets/curated_5feeds_smoke_v2.json` — unchanged. v3 ships
  alongside at `curated_5feeds_smoke_v3.json` with a `failure_modes` field
  per episode (v2 shape is a strict subset; the
  `test_v3_smoke_has_v2_shape_compatible_fields` integration test asserts
  this).

### Audio

This PR ships **no audio**. The operator has a separate PR coming with
multi-voice TTS for v3. The generator emits per-episode voice hints in:

* The transcript itself, as a comment line:
  `#fixture-v3: voice=fr-CA host_voice=en-GB`.
* The manifest:
  `"audio_voice_hints": {"host_voice_accent": "en-GB",
  "guest_voice_accent": "fr-CA", "guest_surface": "Ava Lemoine"}`.

The v1-pinned diarization e2e test from PR #941 stays pinned until that
audio PR + v3 meet downstream.

## How to use v3 in autoresearch

```bash
# Run a v3 smoke sweep through the existing autoresearch harness:
python scripts/eval/run_autoresearch.py \
    --dataset curated_5feeds_smoke_v3 \
    --... (rest of the usual flags)
```

The flat-file dataset is at the same path shape v2 uses; the autoresearch
loader picks it up by id with no other changes.

## Open follow-ups (intentionally out of scope)

* **Multi-voice TTS audio** — operator's separate PR.
* **Long-context fixture content** — v3 tags `long_context_chunk_boundary`
  but the generator currently renders long-context episodes at the same
  density as medium episodes. A v3.1 patch will repurpose the v2 long
  renderer with the same knob set.
* **Whisper-mid-name-token (`Joe Eisenthal House`)** — the
  alias_invention slot can hold this surface form but the canonical case
  isn't in v3 yet; defer until #904's predicate redesign explicitly
  needs it.
* **Per-stage `ProviderCallMetrics` export** — #816's "wire reliability
  evidence into pipeline shutdown" is fixture-orthogonal; tracked
  separately.

## Validation

```bash
# Generator self-check (deterministic, all modes covered):
python scripts/build_v3_fixtures.py --check

# Integration tests:
pytest tests/integration/eval/test_v3_fixtures.py -p no:randomly
# 22 passed in 0.17s
```

## Cross-references

* **Spec input:** `docs/wip/AUTORESEARCH_LEARNINGS_FOR_V3.md` (each section
  in this report points back to the source learning).
* **Prod baseline:** `docs/wip/PROD_RUN_ANALYSIS_100EP.md` (Finding 1, 5,
  12 — feed into the omnycontent / NPR / dialogue-insight patterns).
* **v2 generator:** `scripts/eval/data/generate_v2_transcripts.py` (the
  schema v3 extends).
* **v2 baseline report:**
  `docs/guides/eval-reports/EVAL_FIXTURES_V2_2026_06_06.md`.
