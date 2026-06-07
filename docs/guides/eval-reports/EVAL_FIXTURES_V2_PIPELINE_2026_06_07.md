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

| Layer | Metric | v1 | v2 | AC | Pass |
| --- | --- | ---: | ---: | --- | :---: |
| KG | avg nodes/ep | n/a (stub) | 17.1 | — | — |
| KG | avg edges/ep | n/a (stub) | 16.1 | — | — |
| GIL | avg insights/ep | n/a (stub) | 9.87 | — | — |
| GIL | avg quotes/ep | n/a (stub) | 14.2 | — | — |
| GIL | grounding_rate | n/a (stub) | **97.3%** | ≥95% | ✓ |
| Cleaning | raw episode hit-rate | 0% | **100%** | >80% | ✓ |
| Cleaning | cleaned episode hit-rate | n/a | **0%** | <5% | ✓ |
| Cleaning | sponsor pattern hits retained | n/a | **0%** | — | — |
| CIL | person:* bridges spanning >1 ep | n/a | 10 | >0 | ✓ |
| CIL | topic:* bridges spanning ≥2 feeds | n/a | 4 | >0 | ✓ |
| CIL | org:* bridges spanning >1 ep | n/a | 26 | — | — |
| Topic clusters | tc:* parent clusters | n/a | 4 | >0 | ✓ |
| Topic clusters | tc:* parents spanning ≥2 feeds | n/a | 3 | >0 | ✓ |
| Silver | winner provider | Sonnet 4.6 | **Sonnet 4.6** | unchanged | ✓ |

All five layer ACs pass. v1 silver winner holds on v2 — content shape did not
shift judge preference.

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

- 19 person identities total; **10 span >1 episode**: Maya, Ethan, Priya,
  Rina, Leo, Nora as recurring hosts (3 eps each); Liam, Marco, Ava, Daniel
  as recurring guests (2 eps each).
- 340 topic identities total; **4 span ≥2 feeds**:
  - `topic:second-order-effects` (4 feeds)
  - `topic:correlation-vs-causation` (3 feeds)
  - `topic:systems-thinking` (2 feeds) — matches v2 spec target
  - `topic:risk-management` (2 feeds) — matches v2 spec target
- 30 org identities total; 26 span >1 episode.

### Topic-clusters layer

Script: `scripts/eval/score/topic_clusters_baseline_v2.py`. Embeds KG topic
labels with `sentence-transformers/all-MiniLM-L6-v2` (pipeline default), runs
the production `cluster_indices_by_threshold` at threshold 0.75, names each
`tc:` parent via centroid-closest label. Reuses the KG predictions — no
separate pipeline run. Run output:
`data/eval/runs/baseline_topic_clusters_curated_5feeds_v2/`.

- 127 KG topic rows ingested.
- **4 tc:* parent clusters**, of which **3 span ≥2 feeds**:
  - `tc:risk-management` (4 feeds) — matches v2 spec
  - `tc:downstream-costs` (2 feeds) — emergent
  - `tc:systems-thinking` (2 feeds) — matches v2 spec
- Per-podcast: p01=2, p02=2, p03=2, p04=1, p05=2.

### Summarization silver re-selection

Provider matrix (same set as v1 — clean v1↔v2 isolation):

- Anthropic Claude Sonnet 4.6 (v1 winner)
- OpenAI GPT-4o
- OpenAI GPT-5.4

Configs under `data/eval/configs/silver_selection/silver_candidate_*_smoke_v2_paragraph.yaml`.

Scope of this re-check: v2 smoke (5 eps) × paragraph. Bullets and benchmark
cells deferred — the smoke × paragraph signal is unambiguous (see below) and
the v1 multi-cell matrix consistently agreed with the smoke-paragraph winner.

Pairwise LLM judge results (judge config from `.env.autoresearch`):

| Matchup | Wins A | Wins B | Ties |
| --- | ---: | ---: | ---: |
| Sonnet 4.6 vs GPT-4o | **5** | 0 | 0 |
| Sonnet 4.6 vs GPT-5.4 | **4** | 0 | 1 |

Sonnet 4.6 sweeps both matchups on v2. Same winner as v1.

Per AC: published `silver_sonnet46_smoke_v2` at
`data/eval/references/silver/silver_sonnet46_smoke_v2/`. v1 silvers stay
archived, untouched.

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

- **Two-Marcos test fails (CIL).** v2 spec encodes two distinct `person:marco`
  identities (`p03_e01` Marco the wreck diver vs `p05_e03` Marco the
  tax-loss-harvesting researcher). The CIL baseline shows them merged into a
  single `person:marco` identity (2 episodes, 2 feeds). The v1→v2 report
  already flagged this: `bridge_builder` needs a same-first-name disambiguation
  step. Out of scope for #903; tracked separately.
- **Daniel disambiguation untested.** The v2 spec adds a `p05_e01` Daniel
  (index-investing guest) with a distinct voice from any v1 host. The CIL
  baseline shows `person:daniel` spanning 2 episodes but doesn't tell us
  whether the merge is correct (both v2 mentions are in fact the same Daniel)
  or a collision with v1 host names. Belongs in the same CIL disambiguation
  follow-up.
- **`tc:frame` ambiguity negative test not yet checked.** The v2 spec
  deliberately introduces `topic:frame` (p04 photography) to verify that
  RFC-075 clustering doesn't bundle unrelated uses of "frame". The current
  baseline doesn't see a `tc:frame` cluster, but we haven't explicitly verified
  the negative — the absence may be because Gemini didn't extract `frame` as a
  topic label, not because clustering correctly rejected it. Worth a dedicated
  negative-test in a follow-up.
- **Silver re-selection coverage limited to smoke × paragraph.** Sonnet 4.6
  sweeping 5-0 / 4-0-1 is decisive enough that running bullets + benchmark
  cells is unlikely to change the conclusion, but it isn't impossible. The
  bullets-track in particular had a separate prompt-tuning ceiling in v1
  (silver_sonnet46_smoke_bullets_v1). If a future ticket needs strong evidence
  on bullets, the v2 bullets silver candidate runs can be added cheaply.

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
