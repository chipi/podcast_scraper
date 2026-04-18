# Tier-2 Cross-Dataset Validation Plan

Next phase after v2 closure. v2 gave us a trusted matrix on **one** corpus
(`curated_5feeds_benchmark_v2`, 5 podcast feeds). Tier-2 asks: **do these findings hold
on different corpora?**

**Phase 2.1 (next session): QMSum first.** Plan below is concrete around QMSum; DialogSum
and additional podcast corpus deferred to Phase 2.2 / 2.3 based on what QMSum shows.

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

## Phase 2.1 — QMSum (next session, half-day)

**Dataset:** QMSum — Query-based Multi-domain Meeting Summarization benchmark
(Zhong et al., NAACL 2021).

- **Source (primary):** `pszemraj/qmsum-cleaned` on Hugging Face —
  [huggingface.co/datasets/pszemraj/qmsum-cleaned](https://huggingface.co/datasets/pszemraj/qmsum-cleaned)
  — pre-cleaned transcripts + reference summaries, one `datasets.load_dataset(...)` call.
- **Source (raw):** `microsoft/QMSum` —
  [github.com/Yale-LILY/QMSum](https://github.com/Yale-LILY/QMSum) — query-focused variant
  if needed; more setup, not recommended for Phase 2.1.
- **Paper:** [arxiv.org/abs/2104.05938](https://arxiv.org/abs/2104.05938) (canonical citation).
- **Size:** 232 meetings total; sample **20 meetings** for Phase 2.1 (mirrors v2 held-out
  scale of 5 for fast iteration, plus dev margin).
- **Shape:** academic + product meetings with multiple speakers; transcripts typically
  1-10k words (our podcast transcripts are 5-8k — similar enough to avoid re-tuning plumbing).
- **Why QMSum first:** (a) **human-written gold references** ship with the dataset — no
  silver generation needed, saves $10-20 and ~2 hrs; (b) closest shape analogue to podcasts
  (spoken, long-form); (c) structural contrast (multi-speaker dialog vs podcast monologue)
  is the useful cross-domain signal.

### Champions to port (Phase 2.1 only — keep it tight)

Three champions, not six. Pick the load-bearing ones and treat it as a "does the ordering
hold?" test, not a full re-measurement.

| Pick | Champion | Rationale |
|------|----------|-----------|
| 1 | `gemini-2.5-flash-lite` non-bundled | v2 balanced default (Pareto-optimal cloud pick) |
| 2 | `qwen3.5:9b` bundled | v2 local top; tests whether bundled-is-fine holds off-podcast |
| 3 | `DISLab/SummLlama3.2-3B` standalone | Tier 2.5; tests whether DPO-on-rubric-axes generalises beyond its training distribution |

Three champions × 2 tracks (bullets + paragraph) = **6 runs total** for Phase 2.1. All
numbers go into a first tier-2 report.

Explicitly **deferred to Phase 2.2+** (after we see if the ordering holds):

- DeepSeek non-bundled (quality-first) — only revisit if champions 1-3 reveal a cell where
  DeepSeek might matter.
- Anthropic Haiku 4.5 bundled — only revisit if the "bundled > non-bundled on Anthropic"
  finding becomes contested.
- bart-led pure ML — sanity reference, add if we have time at end of Phase 2.1.
- All hybrid pipelines — demoted under v2, don't retest unless a specific hypothesis calls
  for it.

### The one gotcha: reference length asymmetry

**QMSum's gold references are short — typically 100-300 words.** Our v2 champion prompts
produce 300-600-word summaries. This will depress ROUGE-L on first run (10-20 pp lower
than v2), **even if semantic quality is fine**.

Handling:

- **Do not panic at absolute scores.** Compare relative ordering between champions
  (which is the actual cross-dataset question).
- If judge_mean stays above ~0.75 while ROUGE-L tanks, the finding is "our summaries are
  longer than QMSum refs but as good" — expected and documentable.
- **Optional mitigation:** add a per-dataset length constraint to the paragraph prompt
  (`"in 150-250 words"`), re-score. Only if first run looks uninterpretable.

### Concrete tomorrow-session checklist

```text
1. Load + materialize QMSum sample                              ~20 min
   - `datasets.load_dataset("pszemraj/qmsum-cleaned")`
   - sample 20 meetings (random seed for reproducibility)
   - write to `data/eval/materialized/qmsum_phase21_v1/`
   - generate `meta.json` matching v2 format (so existing harness loads it)
   - reference summaries become `data/eval/references/gold/qmsum_phase21_gold/`
     (new kind: gold instead of silver; scorer already supports generic references)

2. Port champion prompts to QMSum                               ~15 min
   - No changes for Phase 2.1 — ported verbatim from v2
   - Create 3 configs under `data/eval/configs/summarization/qmsum/` and
     `data/eval/configs/summarization_bullets/qmsum/`

3. Run champions                                                ~2-3 hrs wall-clock
   - gemini-2.5-flash-lite non-bundled (bullets + paragraph): ~5 min each via API
   - qwen3.5:9b bundled (single call produces both): ~15 min
   - SummLlama3.2-3B standalone paragraph: ~60 min; bullets: ~60 min

4. Score against QMSum gold                                     ~30 min
   - `score_run()` with gold reference path
   - Run dual-judge for each of the 6 runs (~10 min each)
   - Compute blended 0.70·ROUGE-L + 0.30·judge_mean

5. Write up                                                     ~45 min
   - New file: `docs/guides/eval-reports/EVAL_TIER2_QMSUM_2026_04.md`
   - One table: 3 champions × 2 tracks, with v2 held-out numbers side-by-side for delta
   - "Did ordering hold?" verdict + next-phase decision
```

**Total wall-clock: ~4-5 hours.** Most of that is SummLlama (the only local MPS workload
with material per-episode latency).

### Decision points after Phase 2.1

- **Ordering holds** (same champion ranks #1 on QMSum as v2): high confidence in v2 picks;
  move to Phase 2.2 = Spotify Podcast Dataset (or alternative podcast corpus).
- **Ordering flips** (e.g., SummLlama drops below Gemini on QMSum): dig in — is it DPO
  distribution mismatch? Length asymmetry artifact? Structural (multi-speaker)?
- **All scores collapse** (all three <0.35 blended): prompt-format issue; likely need
  per-dataset prompt tuning, which changes the scope of tier-2.

---

## Phase 2.2 — Spotify Podcast Dataset or alternative podcast corpus

**Primary candidate:** Spotify Podcast Dataset (TREC 2020, ~100k episodes).

- Exact domain match — actual podcasts with creator-written episode descriptions.
- **Access-gated and possibly offline** since ~2023. Deep search in progress for mirrors.
- If found: sample 20-30 episodes, same 3 champions.
- If not found: source an alternative podcast corpus (Lex Fridman archive, Huberman Lab,
  or curated academic podcast dataset) + generate Sonnet-4.6 silvers (~$10).

**MeetingBank (dropped):** investigated `huuuyeah/MeetingBank` — turns out to be
per-agenda-item (median 60-word gold summaries, not full-meeting summaries). Structural
mismatch with our 300-600 word full-transcript pipeline. Not useful for tier-2.

## Phase 2.3 — Spotify Podcast Dataset (access-gated, exact domain match)

**Dataset:** Spotify Podcast Dataset (TREC 2020 podcast summarization track).

- ~100k episodes with creator-written episode descriptions as gold summaries.
- **Exact domain match** — actual podcasts, the closest possible test to our production use case.
- **Access-gated:** requires a request (details TBD — researching access process separately).
- **Gotcha:** gold "summaries" are episode descriptions (short, promotional, 50-200 words),
  not analytical summaries. Judges may be more useful than ROUGE here.
- **Trigger to run:** submit access request now; use the dataset when/if access is granted,
  likely after Phase 2.2.

## Deferred / skip

- **DialogSum** (`knkarthick/dialogsum`) — 300-2k char dialogues. Too short for our 5-40k
  char pipeline. Stress-tests the wrong dimension (length, not domain). Skip unless a
  specific hypothesis demands short-input testing.
- **AMI Corpus** (`edinburghcstr/ami`) — 137 design meetings. Overlaps QMSum's domain.
  Small. Skip unless QMSum results are inconclusive and we need a tiebreaker.
- **MediaSum** (`ccdv/mediasum`) — NPR/CNN interviews. Gold refs are 1-2 sentences — too
  short for meaningful ROUGE against our 300-600w output. Skip.
- **XSum / CNN-DM** — news articles, not spoken.
- **ArXiv-summarization** — academic papers. Wrong modality.
- **SummScreen / ScreenSum** — TV/movie transcripts. Narrower value than MeetingBank.

---

## Full champion shortlist (reference — expand beyond Phase 2.1 as needed)

We have **a lot** of v2 champions. Cross-dataset measurement doesn't need to repeat all
100+ cells. Pick the **load-bearing ones** — if they replicate, the whole matrix probably
replicates. If they break, we know what to look at next.

Phase 2.1 uses only the first three. The rest are for Phase 2.2+ or specific hypothesis
testing:

1. **Balanced default** — `gemini-2.5-flash-lite` non-bundled bullets + paragraph ✅ Phase 2.1
2. **Local top** — `qwen3.5:9b` bundled (bullets + paragraph) ✅ Phase 2.1
3. **Local no-daemon** — `DISLab/SummLlama3.2-3B` standalone (bullets + paragraph) ✅ Phase 2.1
4. **Quality first** — `deepseek-chat` non-bundled bullets + paragraph — Phase 2.2+
5. **Bundled alternative** — `claude-haiku-4-5` bundled (single-call) — Phase 2.2+
6. **Pure-ML floor** — `ml_bart_led_autoresearch_v1` paragraph (sanity reference) — optional

Full Phase 2.1+2.2 upper bound: 6 champions × 2 tracks × 2-3 datasets = **24-36 cells**.

Explicitly **not** in tier-2:

- All non-winning Ollama models (v1 left them as reference; v2 is the authoritative test)
- All the cheaper-tier variants that didn't win their cell (gpt-4o-mini, mistral-medium
  non-bundled etc.)
- Q4_K_M variants (null result on v2; no new information to gain)

---

## Methodology (reuse v2 framework where possible)

- **Framework**: same v2 harness (dual-judge, fraction-based contestation, Efficiency
  rubric, blended `0.70 * ROUGE-L + 0.30 * judge_mean`). No changes.
- **References**:
  - **QMSum (Phase 2.1):** human-written gold references ship with the dataset — no silver
    generation needed. Scorer's `reference_paths` accepts arbitrary reference dirs; create
    `data/eval/references/gold/qmsum_phase21_gold/` with one ref per meeting.
  - **DialogSum (Phase 2.2):** also has human gold — same path, no silver needed.
  - **Second podcast corpus (Phase 2.3, if run):** needs a Sonnet-4.6 silver. ~$5-10 credits.
- **Dev/held-out split:** for small Phase 2.1 samples (20 meetings), treat whole sample as
  held-out — we're not iterating against QMSum, just measuring. Full split only if a
  campaign expands past ~50 episodes.
- **Run orchestration:** existing harness loads datasets from `data/eval/materialized/<id>/`.
  Materialize QMSum into that shape and no framework changes are needed.

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

### Phase 2.1 only (QMSum)

- **References**: gold ships with dataset — $0.
- **Champion runs**: 3 champions × 2 tracks × 20 meetings = 6 runs. Gemini API ~$0.50;
  Ollama qwen3.5:9b local; SummLlama local (MPS). Total cloud API: **~$1**.
- **Judging**: dual-judge (gpt-4o-mini + claude-haiku-4.5) × 6 runs × 20 eps. Total: **~$3-5**.
- **Wall-clock**: **4-5 hours** focused (see Phase 2.1 checklist above).
- **Phase 2.1 total $: under $10.**

### Phase 2.2+ (if run — DialogSum, additional podcast corpus)

- DialogSum has gold refs, same cost profile as Phase 2.1: ~$10 each.
- Second podcast corpus needs silvers (~$10) + runs (~$10) = ~$20.
- **Full tier-2 program (all phases): under $50, probably under $30.**

---

## Deliverables

### After Phase 2.1 (QMSum)

1. **First tier-2 report**: `docs/guides/eval-reports/EVAL_TIER2_QMSUM_2026_04.md`
   - Table: 3 champions × 2 tracks, QMSum scores vs v2 held-out side-by-side
   - "Did ordering hold?" verdict
   - Decision: go to Phase 2.2 (DialogSum), branch into prompt-per-dataset, or widen the champion set
2. **Note in provider guide**: add a "cross-dataset validation" subsection citing
   EVAL_TIER2_QMSUM, whether v2 picks replicated on meetings
3. **Update `TIER2_CROSSDATASET_PLAN.md`**: mark Phase 2.1 done, annotate Phase 2.2 scope
   based on what 2.1 revealed

### After full tier-2 program

1. **Consolidated tier-2 report**: `EVAL_TIER2_CROSSDATASET_2026_<MM>.md`
   - Structure mirrors held-out v2 report
   - Per-dataset matrix for all evaluated champions
   - "Did v2 findings generalise?" table, classified as: general / podcast-specific /
     domain-broken
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
