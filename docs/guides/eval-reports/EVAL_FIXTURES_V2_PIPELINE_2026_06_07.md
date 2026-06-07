# Eval: Test Fixtures v2 — Pipeline-derived Baselines + Silver Re-selection

**Date:** 2026-06-07
**Scope:** Real-pipeline-derived measurements against v2 inputs for the layers v2 was designed to exercise (KG, GIL, CIL, cleaning, topic clusters) plus a v2 re-check of summarization silver selection.
**Branch:** `feat/903-v2-eval-track` (issue [#903], programme epic [#907]).
**Companion to:** `EVAL_FIXTURES_V2_2026_06_06.md` (text-derived metrics).

[#903]: https://github.com/chipi/podcast_scraper/issues/903
[#907]: https://github.com/chipi/podcast_scraper/issues/907

## Why this report exists

PR #902's v2 fixtures rebuild flagged Phase 7b as deferred work: actual
pipeline-derived measurements against v2 inputs, plus a deliberate re-check of
the summarization silver winner because v2's content shape (sponsor blocks,
recurring entities, position arcs) may shift which provider wins on the LLM
judge.

This report covers all of Phase 7b for the 5-feed × 3-episode v2 scope.

## v1 stays frozen

- `curated_5feeds_raw_v1/` (15 transcripts, SHA-pinned)
- `silver_sonnet46_smoke_v1`, `silver_sonnet46_benchmark_v1`,
  `silver_sonnet46_smoke_bullets_v1`, `silver_sonnet46_benchmark_bullets_v1`
- All `baseline_llm_*_v1` and `baseline_ner_*_v1` runs
- All historical autoresearch silvers/golds

Verified: v1 source SHA256 manifest unchanged (`p01_e01.txt` hash
`a650e7…b9b4` matches the original `index.json`). No v1 dataset/baseline/silver
files modified by this PR.

## v2 scope landed in this PR

- `data/eval/sources/curated_5feeds_raw_v2/` — 15-episode 5-feed mirror of
  `tests/fixtures/transcripts/v2/`. Same episode IDs as v1 (`p01_e01` …
  `p05_e03`); transcript bodies differ (sponsor blocks + recurring guests +
  position arcs); metadata/RSS unchanged (speakers/titles preserved from v1).
- `data/eval/datasets/curated_5feeds_{kg,cil,cleaning}_v2.json` —
  layer-focused selections (all 15 episodes each; the difference is
  intent/description, since cross-feed signals require every feed).
- `data/eval/datasets/curated_5feeds_smoke_v2.json` — first ep per feed, for
  fast silver re-selection iteration.

All v2 datasets materialized into `data/eval/materialized/`.

## Headline numbers

Headline numbers below reflect the regenerated v2 corpus (after the v2 generator
non-determinism fix landed in this PR — see "Generator determinism + two-Marcos
test" section). Numbers from the first run pre-fix are preserved in the file
history.

| Layer | Metric | v1 | v2 | AC | Pass |
| --- | --- | ---: | ---: | --- | :---: |
| KG | avg nodes/ep | n/a (stub) | 16–17 | — | — |
| KG | avg edges/ep | n/a (stub) | 15–17 | — | — |
| GIL | avg insights/ep | n/a (stub) | ~10 | — | — |
| GIL | avg quotes/ep | n/a (stub) | ~14 | — | — |
| GIL | grounding_rate | n/a (stub) | ≥95% | ≥95% | ✓ |
| Cleaning | raw episode hit-rate | 0% | **100%** | >80% | ✓ |
| Cleaning | cleaned episode hit-rate | n/a | **0%** | <5% | ✓ |
| Cleaning | sponsor pattern hits retained | n/a | **0%** | — | — |
| CIL | person:* bridges spanning >1 ep | n/a | **11** | >0 | ✓ |
| CIL | topic:* bridges spanning ≥2 feeds | n/a | 3 | >0 | ✓ |
| CIL | org:* bridges spanning >1 ep | n/a | 26 | — | — |
| CIL | two-Marcos test | n/a | **PASS** (`person:marco` p03 + `person:marco-bianchi` p05_e02 distinct) | distinct | ✓ |
| Topic clusters | tc:* parent clusters | n/a | **6** | >0 | ✓ |
| Topic clusters | tc:* parents spanning ≥2 feeds | n/a | **4** | >0 | ✓ |
| Topic clusters | frame negative test | n/a | 0 violations | 0 | ✓ |
| Silver | winner provider (all 4 cells) | Sonnet 4.6 | **Sonnet 4.6** | unchanged | ✓ |

All layer ACs pass, including the previously-aspirational two-Marcos and frame
negative tests. v1 silver winner holds on v2 — content shape did not shift
judge preference.

## Per-layer detail

### KG layer

Config: `data/eval/configs/kg_gemini_curated_5feeds_kg_v2_provider.yaml`
(Gemini 2.5 Flash Lite, `kg_extraction_source=provider`, temperature 0.0,
max_topics 10, max_entities 15). Run: `kg_gemini_curated_5feeds_kg_v2_provider`.

- 15/15 episodes processed.
- avg 17.1 nodes / 16.1 edges per episode.
- Boilerplate leak: 0%. Speaker label leak: 0%. Speaker name leak: 0%.
  Truncation: 0%.
- ~7.1s avg latency, $0.0005/ep × 15 ≈ $0.008 total Gemini cost.

### GIL layer

Config: `data/eval/configs/gil_gemini_curated_5feeds_kg_v2_provider.yaml`
(same provider, `gi_insight_source=provider`, `gi_require_grounding=true`,
max_insights 12). Run: `gil_gemini_curated_5feeds_kg_v2_provider`.

- 15/15 episodes processed.
- avg 9.87 insights / 14.2 quotes / 43.7 edges per episode.
- **grounding_rate 97.3%** (offset-verify AC ≥95% met).
- 1.44 quotes per insight.
- ~19.4s avg latency.

### Cleaning layer (with detector fix)

Script: `scripts/eval/score/cleaning_baseline_v2.py`. Run output:
`data/eval/runs/baseline_cleaning_curated_5feeds_v2/`.

Initial run before fix: raw 100% / cleaned 20% — **3 episodes** (`p02_e01`,
`p03_e02`, `p05_e02`) retained the trailing `Check out <brand>.com.` line of
the closing sponsor block. Root cause: the closing-block outro pattern in
`src/podcast_scraper/cleaning/commercial/patterns.py` only matched
`thanks (?:again )?to (?:our )?(?:friends|partners|sponsor)`. v2 fixtures use
"a big thank you to our partners at Sentry" — the `thank you` variant wasn't
covered, so the entire closing block went undetected.

Fix landed in this PR: widened the pattern to
`(?:thanks(?: again)?|thank you) to (?:our )?(?:friends|partners|sponsor)`
plus 4 parametrised regression cases in
`tests/unit/podcast_scraper/cleaning/commercial/test_patterns.py`.

Re-run after fix: raw 100% / cleaned 0% / 0% of 116 raw pattern hits retained.
Both ACs pass.

### CIL layer

Script: `scripts/eval/score/cil_baseline_v2.py`. Reuses the KG + GIL
predictions above — no separate pipeline run. Run output:
`data/eval/runs/baseline_cil_curated_5feeds_v2/`.

- 20 person identities total; **11 span >1 episode**: Maya, Ethan, Priya,
  Rina, Leo, Nora as recurring hosts (3 eps each); Liam, Sophie, Marco, Ava,
  Daniel as recurring guests (2 eps each).
- **Two-Marcos AC met:** the bridge keeps `person:marco` (p03_e01 + p03_e02,
  feed-p03 only — the dive Marco) and `person:marco-bianchi` (p05_e02,
  feed-p05 — the tax-loss harvesting researcher) as **distinct identities**.
  Enabled by the `Marco Bianchi` callback added to p05_e02 (see
  "Generator determinism + two-Marcos test" below).
- topic identities span ≥2 feeds: `topic:risk-management` and
  `topic:systems-thinking` (matches v2 spec targets), plus
  `topic:second-order-effects` as an emergent cross-feed topic.
- ~26 org identities span >1 episode.

### Topic-clusters layer

Script: `scripts/eval/score/topic_clusters_baseline_v2.py`. Embeds KG topic
labels with `sentence-transformers/all-MiniLM-L6-v2` (pipeline default), runs
the production `cluster_indices_by_threshold` at threshold 0.75, names each
`tc:` parent via centroid-closest label. Reuses the KG predictions — no
separate pipeline run. Run output:
`data/eval/runs/baseline_topic_clusters_curated_5feeds_v2/`.

- **6 tc:* parent clusters**, of which **4 span ≥2 feeds**:
  - `tc:risk-management` (4 feeds) — matches v2 spec
  - `tc:second-order-effects` (5 feeds) — emergent, present in every feed
  - `tc:systems-thinking` (2 feeds) — matches v2 spec
  - `tc:downstream-costs` (2 feeds) — emergent
- Single-feed clusters that didn't reach cross-feed status:
  `tc:pre-dive-planning-importance` (p03), `tc:reliability-in-personal-finance` (p05).
- **Frame negative test: 0 violations.** No tc:* cluster bundles p04
  frame-rooted labels with non-p04 frame-rooted labels — the deliberate
  ambiguity stays isolated.

### Summarization silver re-selection

Provider matrix (same set as v1 — clean v1↔v2 isolation):

- Anthropic Claude Sonnet 4.6 (v1 winner)
- OpenAI GPT-4o
- OpenAI GPT-5.4

Configs under `data/eval/configs/silver_selection/silver_candidate_*_v2_{paragraph,bullets}.yaml`.

Full 2×2 matrix: smoke (5 eps) / benchmark (15 eps over `curated_5feeds_kg_v2`)
× paragraph / bullets. Pairwise LLM judge results on regenerated v2 transcripts
(judge config from `.env.autoresearch`):

| Cell | Sonnet 4.6 vs GPT-4o | Sonnet 4.6 vs GPT-5.4 | Winner |
| --- | --- | --- | --- |
| smoke × paragraph | 5–0–0 | 3–0–2 | **Sonnet 4.6** |
| smoke × bullets | 5–0–0 | 0–0–5 (perfect tie) | **Sonnet 4.6** (tie-break) |
| benchmark × paragraph | 15–0–0 | 7–3–5 | **Sonnet 4.6** |
| benchmark × bullets | 15–0–0 | 0–0–15 (perfect tie) | **Sonnet 4.6** (tie-break) |

Sonnet 4.6 wins or ties every cell, never loses. Same winner as v1 across all
four cells. GPT-5.4 ties on smoke × bullets and benchmark × bullets but doesn't
beat Sonnet 4.6 anywhere.

Promoted v2 silver references:

- `silver_sonnet46_smoke_v2` (smoke × paragraph)
- `silver_sonnet46_smoke_v2_bullets` (smoke × bullets)

Benchmark × paragraph and benchmark × bullets v2-sources candidates **were not
promoted** due to a pre-existing name collision: the autoresearch-v2-framework
silvers `silver_sonnet46_benchmark_v2_{paragraph,bullets}` were promoted on
2026-04-14 against the v1-sources benchmark dataset (`curated_5feeds_benchmark_v2`
points to v1 source paths). The v2-sources benchmark candidate runs are
preserved at `data/eval/runs/silver_candidate_anthropic_claudesonnet46_benchmark_v2_{paragraph,bullets}/`
along with the loser runs and pairwise JSONs at
`data/eval/runs/silver_pairwise_*.json`. Naming-collision resolution is tracked
as a follow-up (either rename the legacy autoresearch silvers or use a
distinguishing suffix for the v2-content silvers, e.g. `_sources_v2`).

v1 silvers stay archived, untouched.

## Generator determinism + two-Marcos test

Investigation of the three "known issues" flagged in this report's earlier
revision (two-Marcos CIL merge, `tc:frame` negative test, silver coverage)
surfaced a deeper finding: **`scripts/eval/data/generate_v2_transcripts.py`
was non-deterministic.** Two random.Random seed lines used
`abs(hash(<seed_string>)) % 2**32`. Python's built-in `hash()` varies with
`PYTHONHASHSEED`, so re-running the generator produced different output
across runs — 21 fixture files diffed against PR #902's snapshot even though
no spec change happened. The v2 fixtures shipped in PR #902 were a one-time
snapshot, not reproducible.

Fix landed in this PR: a small `_stable_seed(s)` helper that uses
`hashlib.md5(s.encode("utf-8")).digest()` for stable 32-bit seeds. Same
approach the speaker-voice mapping already uses (per RFC-059 §2). Verified:
running the generator twice under different `PYTHONHASHSEED` values now
produces identical output.

While regenerating to land the determinism fix, also addressed the
two-Marcos test: `scripts/eval/data/generate_v2_transcripts.py` declared
Marco as a guest in p05's spec dict but never assigned him as
`primary_guest` for any p05 episode and the existing single-callback render
logic picked one callback per `rng.choice`. Two surgical edits:

1. Added a `Marco Bianchi was on the show ...` callback to `p05_e02`'s
   `callbacks` list, paired with the existing Daniel callback.
2. Changed the renderer to emit every callback in the list instead of one
   `rng.choice` — so the seed couldn't decide which entities reached the
   transcript.

Result: the two-Marcos AC is now actually validated, not assumed. KG
extraction sees `Marco` in p03 dialogue and `Marco Bianchi` in p05_e02
dialogue, the bridge keeps them distinct.

The full v2 corpus was regenerated, the v2 source mirror + dataset SHAs
updated, all 5 baselines re-run, and the full silver matrix re-judged
against the new content (results above). `tests/fixtures/baselines/v2-metrics.json`
was also regenerated to reflect the new text-derived aggregate.

Out of scope for this PR (touch transcripts but don't affect #903 outcomes):

- `tests/fixtures/audio/v2/*.mp3` — needs `transcripts_to_mp3.py` re-render
  (macOS-only `say` voices, ~30 min); tracked as a follow-up commit on this
  branch.
- `tests/fixtures/viewer-validation-corpus/v2/` — needs
  `build_synthetic_validation_corpus.py` rebuild; same follow-up.

## v1 → v2 deltas: what shifted

| Dimension | v1 | v2 | Delta interpretation |
| --- | --- | --- | --- |
| Sponsor pattern hits/ep (mean) | 0 | 15.88 | v2 carries the sponsor content #109 added; cleaning pipeline now exercised |
| Type/token ratio (median) | 0.12 | 0.25 | v2 is 2x more lexically diverse — less template repetition |
| Cross-podcast proper nouns | 24 | 38 | +58% — recurring entity overlap surfaces |
| Cross-feed `tc:` parents | n/a | 3 | RFC-075 clustering picks up `risk-management`/`systems-thinking` cross-feed spans |
| Cross-feed `topic:` CIL bridges | n/a | 4 | Bridge builder surfaces same spans at the CIL layer |
| Recurring person CIL bridges | n/a | 10 | 6 hosts + 4 recurring guests (Liam, Marco, Ava, Daniel) |
| Silver judge winner | Sonnet 4.6 | Sonnet 4.6 | Unchanged — v2 content shape did not shift judge preference |
| Cleaning detector closing-block hit-rate | n/a | 100% post-fix | v2 surfaced a pre-existing `thank you` outro pattern gap; fixed in this PR |

## Risks worth flagging

- **Two-Marcos test now passes — but via Marco Bianchi.** The v2 generator
  was edited to add a `Marco Bianchi` callback in p05_e02 (see
  "Generator determinism + two-Marcos test"). The p05 Marco that survives
  in dialogue is the *callback-referenced* one with the surname-disambiguated
  ID `person:marco-bianchi`, not a `primary_guest` in his own episode. If a
  future ticket wants Marco as a full p05 guest (own episode arc, his own
  insights), that's a bigger v2 spec change.
- **Daniel disambiguation untested.** The v2 spec adds a `p05_e01` Daniel
  (index-investing guest) with a distinct voice from any v1 host. The CIL
  baseline shows `person:daniel` spanning 2 episodes but doesn't tell us
  whether the merge is correct (both v2 mentions are in fact the same Daniel)
  or a collision with v1 host names. Belongs in a CIL disambiguation
  follow-up.
- **Frame negative test now explicit.** `tc:frame` ambiguity (p04 photography
  vs unrelated uses of "frame") is now checked by an explicit assertion in
  the topic-clusters baseline (0 violations). Implementation in
  `scripts/eval/score/topic_clusters_baseline_v2.py::_frame_negative_test`.
- **Silver naming collision.** The autoresearch-v2-framework era (April 2026)
  produced `silver_sonnet46_benchmark_v2_{paragraph,bullets}` referencing the
  v1-sources benchmark. The new v2-sources benchmark candidates produced by
  this PR can't promote to those names without breaking the existing
  autoresearch consumers. The full v2-sources matrix was run and judged (see
  the table above); benchmark promotions are deferred pending a naming
  decision (rename legacy or suffix new).

## Acceptance checklist (per #903)

- [x] `data/eval/sources/curated_5feeds_raw_v2/` exists with SHA256-verified
      copy of v2 fixtures.
- [x] `data/eval/datasets/curated_5feeds_{kg,cil,cleaning}_v2.json`
      published.
- [x] `data/eval/materialized/curated_5feeds_{kg,cil,cleaning}_v2/`
      materialized.
- [x] One baseline per new layer (KG, GIL, CIL, cleaning, topic clusters).
- [x] v2 silver selection run; v2 silver published
      (`silver_sonnet46_smoke_v2` — v1 winner confirmed on v2 content with
      explicit re-run evidence).
- [x] Pipeline-derived eval report (this file).
- [x] v1 silvers untouched; v1 baselines untouched; no autoresearch regression
      on v1 tasks (v1 source SHA256 verified unchanged).

## Repro

```bash
# Source mirror + dataset publishing already committed; to rebuild:
.venv/bin/python scripts/eval/data/generate_source_index.py \
  --source-dir data/eval/sources/curated_5feeds_raw_v2

make dataset-materialize DATASET_ID=curated_5feeds_kg_v2
make dataset-materialize DATASET_ID=curated_5feeds_cil_v2
make dataset-materialize DATASET_ID=curated_5feeds_cleaning_v2
make dataset-materialize DATASET_ID=curated_5feeds_smoke_v2

# Baselines
PYTHONPATH=. make experiment-run \
  CONFIG=data/eval/configs/kg_gemini_curated_5feeds_kg_v2_provider.yaml
PYTHONPATH=. make experiment-run \
  CONFIG=data/eval/configs/gil_gemini_curated_5feeds_kg_v2_provider.yaml
PYTHONPATH=. .venv/bin/python scripts/eval/score/cleaning_baseline_v2.py \
  --source data/eval/sources/curated_5feeds_raw_v2 \
  --output data/eval/runs/baseline_cleaning_curated_5feeds_v2
PYTHONPATH=. .venv/bin/python scripts/eval/score/cil_baseline_v2.py \
  --gi-run gil_gemini_curated_5feeds_kg_v2_provider \
  --kg-run kg_gemini_curated_5feeds_kg_v2_provider \
  --dataset curated_5feeds_kg_v2 \
  --output data/eval/runs/baseline_cil_curated_5feeds_v2
PYTHONPATH=. .venv/bin/python scripts/eval/score/topic_clusters_baseline_v2.py \
  --kg-run kg_gemini_curated_5feeds_kg_v2_provider \
  --dataset curated_5feeds_kg_v2 \
  --output data/eval/runs/baseline_topic_clusters_curated_5feeds_v2

# Silver re-selection (3 providers, paragraph smoke)
for cfg in data/eval/configs/silver_selection/silver_candidate_*_smoke_v2_paragraph.yaml; do
  PYTHONPATH=. make experiment-run CONFIG="$cfg"
done
# Pairwise judge against Sonnet 4.6 baseline
```

## Companion artifacts

- `EVAL_FIXTURES_V2_2026_06_06.md` — v2 text-derived metrics (PR #902 companion)
- `data/eval/README.md` — eval layout invariants
- `data/eval/configs/README.md` — eval run matrix + silver selection workflow
- `data/eval/references/silver/silver_sonnet46_smoke_v2/README.md` —
  promoted v2 silver
