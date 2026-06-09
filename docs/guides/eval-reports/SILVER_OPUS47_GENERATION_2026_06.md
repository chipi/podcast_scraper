# Silver Reference Upgrade: Sonnet 4.6 → Opus 4.7 (June 2026)

> Generation report for the autoresearch silver references
> `silver_opus47_smoke_v1` and `silver_opus47_smoke_v2`. Closes Phase 0 of
> the next autoresearch batch ([#939](https://github.com/chipi/podcast_scraper/issues/939));
> gates everything downstream that consumes silver
> (#932 G-Eval finale, #933 prod-curated tier, #927 championships, #921 v3 fixtures).

| Field | Value |
| --- | --- |
| **Date** | 2026-06-09 |
| **Issue** | [#939](https://github.com/chipi/podcast_scraper/issues/939) |
| **Model** | `claude-opus-4-7` (Anthropic Opus 4.7 — thinking model) |
| **User prompt** | `anthropic/summarization/long_v2.j2` (sha256 `acfa8d02b38512a97eea02a00c07b46f02c52fbb689187f5ed8d636b4e77984b`) |
| **System prompt** | `anthropic/summarization/system_v1.j2` (sha256 `d7aabe400110cb1961ba9dea7e380559713aaf3628347deea7758aba5bc0f7ac`) |
| **Datasets** | `curated_5feeds_smoke_v1` (5 ep) + `curated_5feeds_smoke_v2` (5 ep) |
| **Generation script** | `scripts/eval/data/generate_silver_summarization.py` |
| **Total cost** | **$0.36 USD** (well under the $5-7 budget) |

## Why a new generator script (not `make experiment-run`)

The standard `run_experiment.py` summarization path always passes
`temperature=0.0` (deterministic silver). Opus 4.7 deprecates non-1.0
`temperature` for thinking models and returns HTTP 400 if it's set. Rather
than rewire the production provider mid-sweep (which would affect every
Anthropic summarization caller), Phase 0 ships a focused one-shot generator
that:

- omits `temperature` for Opus 4.x thinking models (greedy-ish default)
- writes the same artifact layout (`predictions.jsonl`, `fingerprint.json`,
  `baseline.json`, `metrics.json`, `README.md`, `run.log`) that
  `run_experiment.py` writes, so `promote_run.py` and the score-only path
  consume the output without modification
- records per-episode input/output hashes and token counts in
  `metadata.input_hash` / `metadata.output_hash` / `metadata.cost_usd`

The provider-layer fix (omit `temperature` automatically for thinking
models) is left for a follow-up — silver generation isn't the right
trigger to change the runtime path that every Anthropic summarization
caller exercises.

## What changed vs `silver_sonnet46_smoke_v1`

| Axis | Sonnet 4.6 silver (April 2026) | Opus 4.7 silver (June 2026) |
| --- | --- | --- |
| Model | `claude-sonnet-4-6` | `claude-opus-4-7` |
| Prompt | `anthropic/summarization/long_v1` (no v2-aware behaviors) | `anthropic/summarization/long_v2` (recurring-guest tracking, position-change beats, named-concept preservation) |
| Temperature | 0.0 | omitted (thinking-model default) |
| `max_tokens` | 800 | 800 |
| Dataset | `curated_5feeds_smoke_v1` (5 ep) | `curated_5feeds_smoke_v1` (5 ep) + `curated_5feeds_smoke_v2` (5 ep) |
| Preprocessing | `cleaning_v4` (already-materialized) | `cleaning_v4` (already-materialized) |

The Opus output runs about the same length as Sonnet (≈12k chars across 5
episodes vs Sonnet's 12.7k on v1) but the prose threads recurring beats
through paragraphs in a way long_v1 does not surface. Compression ratio
4.77× (v1) and 4.01× (v2) — close to Sonnet's 4.22× v1 baseline.

## Generation stats

### `silver_opus47_smoke_v1` — `curated_5feeds_smoke_v1`

| Metric | Value |
| --- | --- |
| Episodes | 5 |
| Wall-clock | 68.0 s |
| Avg episode latency | 13.6 s |
| Total input tokens | 20,858 |
| Total output tokens | 3,470 |
| Total cost | **$0.1910 USD** |
| Total chars in → out | 55,178 → 11,570 (compression 4.77×) |

Per-episode SHA-256 of `summary_final`:

| Episode | Input chars | Output chars | Input tok | Output tok | Cost | Output SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| p01_e01 | 11,049 | 2,306 | 4,272 | 715 | $0.0392 | `2f14192ec9dbabfb724949c846b5a1633f6c388b4960071e2cee6ed3c666987e` |
| p02_e01 | 11,537 | 2,298 | 4,305 | 668 | $0.0382 | `a9d607f9481894edcb815a0b73bde512079e62525995e64f61d9ece970c8e357` |
| p03_e01 | 10,679 | 2,317 | 4,016 | 716 | $0.0380 | `64dade44a71ffcb9bf37d392a978309534dc011d4808f3db540aeca4d63ab43b` |
| p04_e01 | 10,723 | 2,470 | 3,984 | 720 | $0.0379 | `34982ec76bb430b4afa2620bb8ead90dfb3c50f95f8ce3fd602e910319b7d95d` |
| p05_e01 | 11,190 | 2,179 | 4,281 | 651 | $0.0377 | `5e4094257055fe32eda97168dc95dc6acbbf20a2ab7c1a18d0061bc763bdcd2e` |

### `silver_opus47_smoke_v2` — `curated_5feeds_smoke_v2`

| Metric | Value |
| --- | --- |
| Episodes | 5 |
| Wall-clock | 61.0 s |
| Avg episode latency | 12.2 s |
| Total input tokens | 17,178 |
| Total output tokens | 3,327 |
| Total cost | **$0.1691 USD** |
| Total chars in → out | 44,087 → 10,987 (compression 4.01×) |

| Episode | Input chars | Output chars | Input tok | Output tok | Cost | Output SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| p01_e01 | 9,164 | 2,286 | 3,558 | 709 | $0.0355 | `5a43259261585ef71b62c88e0a08dde2afc8d15cf158ec4fb01d6e0306c18f21` |
| p02_e01 | 8,957 | 2,303 | 3,492 | 683 | $0.0345 | `1d6f23212e49f1b01231bc4fee28d9cb65338e0e5ad8e08753cc5fc4c97ce000` |
| p03_e01 | 8,530 | 1,775 | 3,341 | 552 | $0.0305 | `c2f03ca0bee4e3233a05d70a6660f708311ec925cf8d437acff7ea2c702b1da9` |
| p04_e01 | 8,666 | 2,325 | 3,340 | 699 | $0.0342 | `3e0f6297fca40a8d1b68e6bcc9a7987fc724bf6a4c0ceccf75bb7fca3aaae683` |
| p05_e01 | 8,770 | 2,298 | 3,447 | 684 | $0.0343 | `526b9cdfc08e3b61cd6d3ba0504baee6a8529168e02f08cf35e33359d52d02b3` |

## Reproduction

```bash
# Pre-flight
make init                          # python venv if not present
set -a; . .env; set +a              # ANTHROPIC_API_KEY in env

# Generate v1 silver
PYTHONPATH=. .venv/bin/python scripts/eval/data/generate_silver_summarization.py \
  --config data/eval/configs/silver_selection/silver_candidate_anthropic_opus47_smoke_v1.yaml \
  --run-id silver_candidate_anthropic_opus47_smoke_v1 \
  --force

# Generate v2 silver
PYTHONPATH=. .venv/bin/python scripts/eval/data/generate_silver_summarization.py \
  --config data/eval/configs/silver_selection/silver_candidate_anthropic_opus47_smoke_v2_paragraph.yaml \
  --run-id silver_candidate_anthropic_opus47_smoke_v2_paragraph \
  --force

# Promote both
make run-promote RUN_ID=silver_candidate_anthropic_opus47_smoke_v1 \
  AS=reference PROMOTED_ID=silver_opus47_smoke_v1 REFERENCE_QUALITY=silver \
  REASON="Opus 4.7 silver — raises autoresearch quality ceiling (#939)"

make run-promote RUN_ID=silver_candidate_anthropic_opus47_smoke_v2_paragraph \
  AS=reference PROMOTED_ID=silver_opus47_smoke_v2 REFERENCE_QUALITY=silver \
  REASON="Opus 4.7 silver on v2 dataset (#939)"
```

## Notable observations about Opus 4.7 vs Sonnet 4.6 outputs

Skim of the v1 outputs (full predictions at
`data/eval/references/silver/silver_opus47_smoke_v1/predictions.jsonl`):

- **Recurring-beat threading.** Opus' first paragraph names a small set
  of through-lines (tire pressure, single-variable iteration, drainage)
  and then revisits each in later paragraphs, where Sonnet treated each
  paragraph as a fresh topic block. Both are valid summaries; Opus is
  more cohesive.
- **No explicit title headers.** long_v2 doesn't ask for an H1; Opus
  obeys cleanly. The Sonnet silver (long_v1) included `# Title` headers
  on most episodes — Opus didn't. This is one of the v2-aware behavior
  differences (#906 dropped the title from long_v2).
- **Recurring-guest beat present where applicable.** On p04_e01 (which
  mentions a recurring guest), Opus surfaces the call-back; long_v2's
  "name recurring guests" rubric was followed. Sonnet under long_v1
  did not have that rubric and did not surface the beat.
- **Position-change beats not present in these 5 episodes.** Neither
  the smoke v1 nor the smoke v2 fixtures contain "I used to think X
  — after Y, I now think Z" structures, so this v2-aware behavior is
  unexercised at smoke scale. The benchmark/dev datasets are where it
  will show up.

No anomalies, no truncation, no empty content. All 10 generations
returned non-empty text with token counts well below the 800 max.

## Old silver kept for historical comparison

`data/eval/references/silver/silver_sonnet46_smoke_v1/` and
`silver_sonnet46_smoke_v2/` are intentionally retained — autoresearch
configs that don't migrate to the new silver immediately can still
reference them, and a side-by-side comparison is useful for the
methodology audit in #932.

## Downstream impact (informs)

| Issue | Effect |
| --- | --- |
| #932 G-Eval finale | Should use `silver_opus47_smoke_v1` as the LLM-judge reference once landed |
| #933 prod-curated tier | When prod-curated dataset is built, generate a sibling `silver_opus47_prod_curated_v1` |
| #927 championships | All summary championships now score against the new silver |
| #921 v3 fixtures | Use the new silver as ground-truth for fidelity comparison |
| #923 prod profile | No direct effect — pricing/cost not on the prod path |

## References

- [Issue #939 — Upgrade autoresearch silver from Sonnet 4.6 to Opus 4.7](https://github.com/chipi/podcast_scraper/issues/939)
- [PR #941 — long_v2 transcript-injection fix](https://github.com/chipi/podcast_scraper/pull/941)
- [Smoke v2 DGX refresh report](EVAL_SMOKE_V2_DGX_REFRESH_2026_06.md) (the report that surfaced the silver-bias problem)
- Anthropic Opus 4.7 pricing: $5 / $25 per 1M tokens (input / output); added to `config/pricing_assumptions.yaml`
