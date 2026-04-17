# LoRA Fine-Tuning — Hybrid Pipeline Revival Plan

WIP note. Learning project with a specific goal: use LoRA fine-tuning to try to rescue the
hybrid_ml pipeline so it competes with top cloud/local LLMs under v2 framework.

**Target unlock**: hybrid goes from v1's ROUGE-L 21% → target 45-55% (if it works). Would make
a fully-local, fully-deterministic, zero-API pipeline viable for production deployments where
that matters.

## ⚠️ Status 2026-04-16: GATED ON TIER-2, NOT ACTIVE

v2 closed (see [held-out v2 eval report](../guides/eval-reports/EVAL_HELDOUT_V2_2026_04.md)).
What we learned changes the premise of this plan:

1. **Hybrid's v2 ceiling is visible.** hybrid bart+llama3.2:3b = 0.430, hybrid bart+qwen3.5:9b
   = 0.448 held-out paragraph. Standalone qwen3.5:9b bundled = 0.509. **BART MAP hurts capable
   REDUCE stages more than it helps them** — the premise that "lift the hybrid" is the right
   unit to optimize needs rechecking.
2. **SummLlama3.2-3B proves DPO-on-rubric-axes works for 3B models.** Same Llama-3.2-3B base
   jumped 0.270 → 0.485 held-out paragraph (+80%) via DPO alone. That's a cheaper, more
   direct lift than LoRA on the hybrid.
3. **Tier-2 cross-dataset validation is the real next gate.** If SummLlama / qwen3.5:9b
   defaults still win on QMSum / DialogSum / other podcast datasets, LoRA is solving a
   problem we don't have. If gaps open up, LoRA target may be the standalone base, not the
   hybrid pipeline at all.

**Don't start this plan until tier-2 measurement is complete.** Revisit after tier-2 with
two questions: (a) is there still a gap worth closing? (b) is the hybrid pipeline still the
right fine-tuning surface, or is the answer now "LoRA on SummLlama-style standalone"?

Everything below this line is the original plan, retained for reference.

**Relation to other WIP**: paired with [`AUTORESEARCH_V2_NEXT_STEPS.md`](AUTORESEARCH_V2_NEXT_STEPS.md)
(v2-closed header + ML/hybrid v2 base numbers).

---

## Why this specifically

The hybrid pipeline (`ml_hybrid_bart_llama32_3b_autoresearch_v1`) has two fine-tunable components:

```text
Transcript (~11k chars, chunked)
    ↓
[BART MAP stage]   — per-chunk structured extraction
    ↓
chunk summaries
    ↓
[Llama3.2:3b REDUCE stage]   — combine into final summary
    ↓
Final summary (paragraph or bullets)
```

Two independent fine-tuning targets. Either one could be LoRA-tuned. Together, they define
the quality ceiling of the hybrid path.

Cloud LLMs (single-call) don't have these stages — they can't be "fine-tuned" without
provider-side training access. The hybrid is the only pipeline where local fine-tuning can
meaningfully move the needle.

---

## Why LoRA, not full fine-tuning

LoRA (Low-Rank Adaptation) trades generality for efficiency. Instead of updating all model
parameters, it freezes the base model and trains small **adapter matrices** added to specific
layers (typically attention projections).

Key properties:

- **Tiny trainable parameter count** (~0.1-1% of the base model)
- **Small checkpoints** (~50-500MB vs the full model's GB)
- **Fast training** (minutes to hours, not days)
- **Composable** — can keep the base model and swap adapters
- **Mergeable** — adapter weights can be folded back into the base model for deployment

For our hardware (M4 48GB RAM, no CUDA) LoRA is the only practical option:

- Full fine-tuning a 3B model needs ~24GB of optimizer state alone — tight on 48GB with
  other processes running
- QLoRA (4-bit + LoRA) is the usual memory-efficient path elsewhere, but **bitsandbytes
  (the 4-bit library) doesn't support Apple MPS**. So standard QLoRA isn't available on Mac
- MLX-native LoRA + fp16 base model: the Apple Silicon sweet spot

---

## Tooling options on M4

### Apple MLX (recommended for this project)

- Apple's native ML framework, built specifically for Apple Silicon
- First-class LoRA support via `mlx-lm`
- Faster than PyTorch MPS for many operations on M-series chips
- Growing ecosystem but smaller than PyTorch

```bash
pip install mlx-lm
# Example invocation (one line, for reference):
python -m mlx_lm.lora --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --train --data ./data/training --iters 500 --lora-layers 16
```

### HuggingFace PEFT + PyTorch MPS

- More familiar tooling if you've used transformers before
- Works on MPS but slower than MLX for attention ops
- Required for BART family (MLX has fewer non-LLM models packaged)

```python
from peft import LoraConfig, get_peft_model, TaskType
config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(base_model, config)
# then standard HF Trainer
```

### torchtune

- PyTorch-native fine-tuning library, newer and well-integrated
- Works on MPS
- Probably the best PyTorch-path option

### Recommendation

- **BART LoRA** (seq2seq): HuggingFace PEFT (BART isn't MLX-packaged)
- **Llama3.2:3b LoRA** (causal LM): MLX-LM (faster on Apple Silicon, better LLM tooling)

---

## The data problem (this is the real bottleneck)

LoRA is efficient but needs training data. Our situation:

- 10 dev episodes × 2 tracks = **20 unique (transcript, silver-summary) pairs**
- 5 held-out episodes: **off-limits for training** (destroys held-out validity)
- 20 examples is below typical LoRA threshold — expect underwhelming results

### Data generation plan (do this FIRST)

Before any LoRA experiment, generate more silvers:

1. **Source more podcast RSS feeds** — pick 5-10 podcasts not in our eval set. Diverse
   domains (tech, business, science, sports, interviews).
2. **Scrape + transcribe** — use existing podcast_scraper pipeline. Transcribe with
   Whisper. Target 100-200 new episodes.
3. **Generate silvers with Sonnet 4.6** in bundled mode (our silver methodology).
   Cost: ~$0.01/episode × 200 = ~$2 total.
4. **Validate quality spot-check** — skim 10 generated silvers, confirm they match style
   we expect. Discard any with JSON errors or obvious garbage.
5. **Format as training pairs**:

   ```json
   {"input": "<cleaned transcript>", "output": "<silver summary>"}
   ```

**Result**: 200+ training pairs, zero held-out contamination, genuine dataset.

### Why this data-generation step is the biggest unlock

Even without LoRA, 200 silvers give you:

- Proper train/val/test splits for any model-tuning work
- Multi-shot few-shot prompting data (could retrieve most-similar silvers at inference)
- Dataset to evaluate *future* silver methodology changes against

It pays off regardless of LoRA outcome.

---

## Phase plan for hybrid pipeline LoRA

### Pre-phase 0: Pick the right base models FIRST

**Important realisation from v2 data**: the hybrid's default REDUCE (llama3.2:3b, 0.501)
is the **weakest of the 7B+ class we've evaluated**. It was chosen in v1 because it was
the smallest available Ollama model, not because it was optimal.

Before investing any training time, swap the base to a stronger starting point. Empirical
LoRA pattern: stronger base → stronger fine-tune. LoRA'd 3B won't reach un-tuned 9B.

### REDUCE base candidates (by v2 bullets held-out)

| Candidate | v2 score | Size | Why consider |
|-----------|:--------:|:----:|--------------|
| **qwen3.5:9b** | **0.580** | 9B | Highest local quality; excellent LoRA support; likely ceiling 0.60+ |
| mistral:7b | 0.526 | 7B | Surprise v2 winner at 7B; huge LoRA ecosystem; balanced pick |
| llama3.1:8b | 0.518 | 8B | Well-supported in LoRA literature; solid mid-option |
| llama3.2:3b (current v1 default) | 0.501 | 3B | Smallest; keep only if memory-constrained |
| phi3:mini | 0.475 | 3.8B | Skip — weak base, less LoRA community support |

### MAP base candidates

Current `facebook/bart-large-cnn` is CNN-news tuned, 1024-token input limit. Same
"pick the right base before LoRA-tuning it" logic applies. See
`AUTORESEARCH_V2_NEXT_STEPS.md` §2 for alternatives. Don't LoRA-tune BART if you're about
to swap it.

### Resulting combinations to evaluate

| # | MAP | REDUCE | Rough target | Resource cost |
|---|-----|--------|:------------:|:-------------:|
| 1 | bart-large-cnn (LoRA'd) | llama3.2:3b (LoRA'd) | 0.45 | Lowest |
| 2 | long-t5 (LoRA'd) | mistral:7b (LoRA'd) | 0.55 | Medium |
| 3 | long-t5 (LoRA'd) | qwen3.5:9b (LoRA'd) | 0.60+ | Highest |
| 4 | long-t5 (un-tuned) | qwen3.5:9b (un-tuned) | baseline | Medium |

**Run combo #4 first as Phase 0.** It might already be competitive, in which case LoRA
becomes optional. Combo #3 is the ambitious target. Combos #1-2 are fallbacks if
memory/training time constrains.

### Phase 0: Baseline with right base models

**Goal**: establish un-tuned hybrid numbers with the RIGHT base models under v2 framework.
Not with v1's llama3.2:3b — with the chosen base (combo #4 above).

- Set up hybrid config with long-t5 MAP + qwen3.5:9b REDUCE
- Apply v2 champion REDUCE prompt
- Run on dev + held-out × bullets + paragraph
- Record: ROUGE-L, final score, latency per stage (MAP vs REDUCE)
- Identify bottleneck: is MAP extraction or REDUCE synthesis the weak link?

**Output**: baseline numbers with modern bases + bottleneck identification. Already
meaningful — this alone may lift the hybrid pipeline substantially without any training.

### Phase 1: Data generation

**Goal**: 200+ (transcript, silver) training pairs.

- Pick 5-10 podcast RSS feeds outside the eval set
- Scrape + transcribe 200 episodes via existing pipeline
- Generate silvers with Sonnet 4.6 bundled mode
- Format + split: 160 train / 20 val / 20 eval-held-out (separate from framework held-out)
- **Decision point**: sanity-check silver quality before committing to training

**Time estimate**: 1 day (mostly wall-clock for scrapes + Whisper; human time ~2 hours)

### Phase 2: LoRA on whichever stage bottlenecks

Based on Phase 0 bottleneck identification, target that stage first.

**Path A — Fine-tune BART MAP** (if extraction is the weak link):

- Base: `facebook/bart-large-cnn` (current) or swap to `google/long-t5-tglobal-large` first
- Training objective: (transcript chunk → mini-summary-of-chunk)
- Tool: HuggingFace PEFT
- LoRA config: r=16, alpha=32, target q_proj + v_proj
- Epochs: 3-5, learning rate 1e-4
- Time: 30-60 min per training run

**Path B — Fine-tune Llama3.2:3b REDUCE** (if synthesis is the weak link):

- Base: Llama3.2:3b (via MLX)
- Training objective: (chunk-summaries concatenated → final silver-style summary)
- Tool: mlx-lm
- LoRA config: r=16, alpha=32, default target modules
- Epochs: 3-5, learning rate 1e-4
- Time: 1-2 hours per training run

**Evaluate each run**:

1. Eval loss on val set (catch overfitting)
2. Run through v2 framework on dev + held-out
3. Compare to baseline from Phase 0

### Phase 3: Iterate or combine

Depending on Phase 2 results:

- **If one stage LoRA worked**: try LoRA on the other stage. Combine.
- **If neither moved the needle**: data is the bottleneck. Either abandon or generate
  more silvers (500+).
- **If a stage started overfitting quickly**: lower learning rate, fewer epochs,
  higher LoRA dropout.

### Phase 4: Package + deploy

If the final hybrid matches or beats local LLM baselines:

- Merge LoRA adapters into base weights
- Create Ollama Modelfile for the Llama REDUCE stage (custom model name)
- Publish BART fine-tune locally (HF cache)
- Update `ml_hybrid_*_autoresearch_v2.yaml` config
- Re-run v2 framework on final pipeline
- Update AI provider guide if hybrid now competes

---

## Success criteria

**Minimum viable win**:

- Hybrid ROUGE-L held-out improves ≥ +5pp vs Phase 0 baseline
- Quality now matches or exceeds llama3.2:3b standalone non-bundled (0.501 bullets)
- Held-out numbers generalise (dev→held-out delta < ±5%)

**Real win**:

- Hybrid final score matches qwen3.5:9b (0.580) on bullets
- Deterministic, zero-API, no Ollama required
- Becomes the recommended "privacy-first + reproducibility-first" pick in the guide

**No-win scenarios** (and their learnings):

- 20-200 examples just isn't enough signal → document as "LoRA viable only with 1000+
  examples" finding. Informs future decisions.
- Training succeeds but v2 numbers don't move → suggests the bottleneck is model
  capability, not fine-tuning potential. Hybrid design has a quality ceiling independent
  of data.
- Overfitting dominates → small-sample LoRA isn't the right tool. Few-shot in-context
  with retrieval beats small LoRA at this data scale.

---

## Risks and mitigations

### Risk: 20-200 examples is insufficient

**Probability**: high. Mitigation: Phase 1 data generation. If you skip Phase 1 and
just use our 20 silvers, expect disappointment.

### Risk: Overfitting

**Probability**: high with small data. Mitigations:

- High LoRA dropout (0.1-0.2)
- Fewer epochs (3 instead of 10)
- Validation loss early stopping
- Keep lora rank modest (r=8-16, not 64)

### Risk: Apple MLX ecosystem gaps

**Probability**: medium. Mitigations:

- Prefer MLX for LLM path (best M-series tooling)
- Fall back to PyTorch MPS (slower but broader support)
- For BART specifically, use PyTorch + PEFT — MLX doesn't have encoder-decoder models well

### Risk: Training data quality

**Probability**: low (Sonnet 4.6 silvers are reliable). Mitigations:

- Spot-check 10 silvers before committing to 200
- Filter silvers where the JSON failed or contained errors
- Validate output length distribution matches existing dev silvers

### Risk: Ends up not worth it vs just using qwen3.5:9b

**Probability**: medium-high. Mitigation: accept it as a learning outcome. The real
project value is "do I now know how to LoRA fine-tune?" — that knowledge is transferable
even if this specific pipeline doesn't win.

---

## Reading list / learning resources

**LoRA theory**:

- Original paper: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- Explains why low-rank adaptation works: the update matrices during fine-tuning are
  empirically low-rank, so decomposing into small matrices is lossless in practice

**Apple Silicon tooling**:

- MLX docs: `https://ml-explore.github.io/mlx/`
- MLX-LM LoRA example: `https://github.com/ml-explore/mlx-examples/tree/main/lora`

**HuggingFace path**:

- PEFT library: `https://huggingface.co/docs/peft`
- BART fine-tuning tutorials are plentiful; focus on the PEFT + LoRA variants

**Practical hyperparameter guides**:

- "LoRA Hyperparameters: A Practical Guide" — search for recent blog posts; the
  state of the art at r=8-16, alpha=2×r, dropout 0.05-0.1, learning rate 1e-4 to 3e-4

**Data requirements**:

- LoRA papers typically use 1k-10k examples. Below 100 is widely considered
  "prompt engineering is probably better"

---

## Relation to v2 framework (what changes for evaluation)

Good news: **the v2 framework supports this out of the box**. A LoRA-fine-tuned BART
or custom-Ollama-model is just another model variant. Run the same autoresearch configs,
same silvers, same judges.

What gets added:

- New config `ml_hybrid_bart_lora_llama32_3b_lora_v1.yaml` once fine-tuning is done
- New column in eval report under ML track
- Compare to Phase 0 baseline (un-tuned hybrid v2)
- If successful, gets an entry in the AI provider guide

No framework changes required. This is the cleanest possible test: v2 framework validates
whether LoRA genuinely helped, on held-out content the LoRA adapters were never trained on.

---

## What NOT to do

- **Don't use held-out (`curated_5feeds_benchmark_v2`) for training.** It contaminates
  the entire v2 framework. Training data must come from entirely different episodes.
- **Don't train on the 20 existing dev silvers only.** Insufficient signal. Either do
  Phase 1 data generation or don't bother.
- **Don't over-tune hyperparameters** on small data. Ludicrously easy to overfit.
  Pick reasonable defaults and iterate on data/model, not hyperparams.
- **Don't skip Phase 0 baseline.** Without un-tuned hybrid numbers you can't claim LoRA
  "helped." Compare against actual measurements, not memory of v1 numbers.
- **Don't scope creep into training the cloud providers' models.** You can't. This is
  specifically about local models.
- **Don't try full fine-tuning instead of LoRA.** 48GB RAM + no CUDA makes this tight.
  LoRA is the right tool for this hardware.

---

## Open questions

- **Should we experiment with alternative BART-family models first?** Covered in
  `AUTORESEARCH_V2_NEXT_STEPS.md` §2. Doing that first might change which base model we
  LoRA-tune in Phase 2.
- **Is MLX or PyTorch MPS faster for Llama3.2:3b LoRA on M4?** Not sure — would benchmark
  both with identical config on 10 training steps before committing.
- **Can we use public summarisation datasets for pre-training warm-up?** SamSum, XSum,
  DialogSum — different domain but might help adapt the model to the general task
  before fine-tuning on our silvers. Adds complexity.
- **What's the realistic quality ceiling for hybrid under v2?** Untuned local llama3.2:3b
  non-bundled hits 0.501 bullets. With perfect fine-tuning, could hybrid approach 0.60?
  Unclear without running the experiment.
- **At what data size does LoRA "start working"?** Empirically ~100 examples for style
  adaptation, ~500 for capability changes, ~1000+ for new task learning.

---

## Estimated total effort

- Phase 0 (baseline): 2 hours (covered in v2 next-steps note)
- Phase 1 (data): 1 day wall-clock, 3-4 hours hands-on
- Phase 2 (first LoRA experiment): 1-2 days iterating
- Phase 3 (tune + combine): 2-3 days
- Phase 4 (package + deploy): 0.5 day

**Total: ~1 week of focused work** if committing to it end-to-end. Or scope it into a few
weekends over a month.

First concrete question to answer: **can Phase 0 hybrid v2 baseline actually benefit from
LoRA at all?** If the bottleneck turns out to be BART's 1024-token input (truncation),
LoRA won't fix that. You'd need a different MAP model (long-t5, pegasus-big_patent) first.

---

## If this succeeds, it's quietly important

If LoRA-tuned hybrid matches qwen3.5:9b quality:

- Proves the local deployment story can be built without any Ollama dependency
- Makes fully air-gapped deployment realistic (no Ollama server process, no OpenAI-compat
  layer, just a Python process with HF models)
- Transfers the learning to other tasks — speaker detection, GIL/KG, cleaning could all
  benefit from similar treatment

Even if it doesn't fully close the gap, the data-generation pipeline (Phase 1) is useful
for every future model improvement. That alone might be worth the investment.
