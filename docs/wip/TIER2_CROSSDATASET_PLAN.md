# Tier-2 Cross-Dataset Validation Plan

Next phase after v2 closure. v2 gave us a trusted matrix on **one** corpus
(`curated_5feeds_benchmark_v2`, 5 podcast feeds). Tier-2 asks: **do these findings hold
on different corpora?**

Not started yet. This note exists so the scope is pinned down before the session begins.

**Predecessor:** [held-out v2 eval report](../guides/eval-reports/EVAL_HELDOUT_V2_2026_04.md)
+ [v2-closed header in AUTORESEARCH_V2_NEXT_STEPS.md](AUTORESEARCH_V2_NEXT_STEPS.md).

---

## Why a new phase

v2 gave us confident rankings on our podcast corpus. What it can't tell us:

- Would `qwen3.5:9b bundled` still beat `hybrid bart+qwen3.5:9b` on meeting transcripts
  (different structural shape — multi-speaker, less narrative)?
- Does SummLlama3.2-3B's DPO alignment generalise to domains it wasn't tuned on?
- Does the "bundled is fine for Anthropic, not Gemini" pattern replicate, or is it a
  podcast-specific artefact?
- At what cross-dataset score divergence does a provider "break"?

Any single-corpus result is a **measurement**, not a **pattern**. Tier-2 promotes the
strongest v2 findings to patterns (or knocks them down) with 2-3 additional corpora.

---

## Datasets to add (shortlist)

Pick 2-3 from this list. Each chosen for a distinct shape.

### Primary candidates

1. **QMSum** (meetings, 232 → 20-30 sample)
   - Academic / product meetings with multiple speakers
   - Human-annotated queries + reference summaries
   - Shape: structured dialog, 1k-10k words, different from podcast monologue
   - Access: HF `pszemraj/qmsum-cleaned` or the original `microsoft/QMSum`
   - **Why pick:** closest analogue to podcast (spoken, long-form) but different
     structure. If our numbers don't hold here, they likely don't generalise to
     meeting-style audio at all.

2. **DialogSum** (chitchat / task-oriented dialogs, 13k training → sample 20)
   - Short two-party dialogs with reference summaries
   - Shape: 300-1500 words — much shorter than our podcasts
   - Access: HF `knkarthick/dialogsum`
   - **Why pick:** stress-tests the opposite end of the length spectrum. Our current
     pipeline is tuned for 30-min transcripts. If SummLlama / qwen3.5:9b lose on short
     dialog, the tuning is length-coupled.

3. **One additional podcast corpus** (to de-risk podcast-specific overfit)
   - Options: Lex Fridman podcast scraped archive, Huberman Lab archive, or
     a curated academic podcast dataset (sample 10-20 episodes)
   - Shape: same as ours but different hosts / topics / pacing
   - **Why pick:** if our findings collapse on a different podcast source, the "podcast
     generalisation" claim was fragile.

### Deferred / skip for first pass

- **XSum / CNN-DM** — news articles, not spoken. Keep for validation of the ML track's
  BART-tuning only if relevant.
- **ArXiv-summarization** — academic papers. Wrong modality.
- **SummScreen / ScreenSum** — TV/movie transcripts. Interesting but narrower value
  than QMSum.

---

## Shortlist of champions to port

We have **a lot** of v2 champions. Cross-dataset measurement doesn't need to repeat all
100+ cells. Pick the **load-bearing ones** — if they replicate, the whole matrix probably
replicates. If they break, we know what to look at next.

Priority order:

1. **Balanced default** — `gemini-2.5-flash-lite` non-bundled bullets + paragraph
2. **Quality first** — `deepseek-chat` non-bundled bullets + paragraph
3. **Bundled alternative** — `claude-haiku-4-5` bundled (single-call, bullets + paragraph)
4. **Local top** — `qwen3.5:9b` bundled (bullets + paragraph)
5. **Local no-daemon** — `DISLab/SummLlama3.2-3B` standalone (paragraph; bullets if A.1 adds it)
6. **Pure-ML floor** — `ml_bart_led_autoresearch_v1` paragraph (sanity reference)

That's 6 champions × 2 tracks × 2-3 datasets = **24-36 cells per champion round**. Manageable.

Explicitly **not** in tier-2 first pass:

- All non-winning Ollama models (v1 left them as reference; v2 is the authoritative test)
- All the cheaper-tier variants that didn't win their cell (gpt-4o-mini, mistral-medium
  non-bundled etc.)
- Q4_K_M variants (null result on v2; no new information to gain)

---

## Methodology (reuse v2 framework where possible)

- **Framework**: same v2 harness (dev/held-out split, dual-judge, fraction-based contestation,
  Efficiency rubric, blended `0.70 * ROUGE-L + 0.30 * judge_mean`). No changes.
- **Silvers**: each new dataset needs a Sonnet-4.6 silver to score against. Generate per-dataset:
  - `silver_sonnet46_qmsum_paragraph`, `silver_sonnet46_qmsum_bullets`
  - ditto DialogSum, additional podcast
  - Estimated cost: ~$1-3/dataset in Sonnet-4.6 credits
- **Dev/held-out split**: same pattern. 60-70% for dev (only if iteration happens); 30-40%
  held-out (scored once). Smaller datasets may skip the split and use single held-out subset.
- **Run orchestration**: extend `scripts/eval/run_benchmark.py` to accept cross-dataset
  configs; no framework changes expected.

---

## Go/no-go gates per finding

For each v2 finding that tier-2 can validate or break:

| v2 finding | Tier-2 break criterion | Action if broken |
|------------|------------------------|------------------|
| `gemini-2.5-flash-lite` = balanced default | Loses on 2+ of 3 datasets | Re-tier provider picks per-domain |
| `qwen3.5:9b bundled` = local top | Loses on meeting or short-dialog | Keep as podcast-default; document domain constraint |
| Hybrid < standalone qwen3.5:9b | Flips on short dialog (BART MAP helps on short) | Restore hybrid as niche for short-dialog workloads |
| SummLlama3.2-3B standalone viable | Breaks on non-podcast | Tighten "Tier 2.5" claim to podcast-only |
| Anthropic bundled > non-bundled | Stops holding | Document as podcast-specific; no code change |

---

## Estimated budget

- **Silvers**: 3 datasets × 2 tracks = 6 silver sets. ~$10-20 in Sonnet-4.6 credits.
- **Champion runs**: 6 champions × 2 tracks × 3 datasets × 2 splits = 72 runs. Most are
  fast-cheap API calls (DeepSeek, Gemini); ~$5-15 total API. Ollama/SummLlama on local
  hardware; few hours of wall-clock.
- **Judging**: dual-judge across 72 runs. ~$5-10 in judge API costs.
- **Wall-clock**: 1-2 focused days (datasets + silvers + runs + scoring + report).
- **Total $**: under $50, probably under $30.

---

## Deliverables

1. **New report**: `docs/guides/eval-reports/EVAL_TIER2_CROSSDATASET_2026_<MM>.md`
   - Structure mirrors held-out v2 report
   - Per-dataset matrix for the 6 champions
   - "Did v2 findings generalise?" table
2. **Updated provider guide**: call out which picks are **general** vs **domain-specific**
3. **Updated config_constants.py**: if defaults change per domain, add domain-aware
   selection (or document that picks are podcast-tuned)
4. **LoRA go/no-go decision** in [LORA_HYBRID_PIPELINE_PLAN.md](LORA_HYBRID_PIPELINE_PLAN.md)
   based on tier-2 results

---

## Open questions to pin down before starting

- Do we want one **consolidated multi-dataset** silver generation, or per-dataset silvers?
  (Probably per-dataset so each dataset can evolve independently.)
- Is the "same champion ported across datasets" assumption right, or does some champion
  need per-dataset prompt tuning? (Defer decision until first dataset run shows numbers.)
- Judge choice: keep gpt-4o-mini + claude-haiku-4-5? Probably yes — changing judges
  mid-programme breaks comparability with v2 numbers.
- Should we also sneak in a **multi-run averaging** pilot (N=3) on the top-2 champions?
  (Likely yes — this is where the open item from v2 finally gets addressed.)
