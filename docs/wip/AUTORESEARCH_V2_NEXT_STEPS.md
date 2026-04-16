# Autoresearch v2 — Next Steps (Master Plan)

Consolidating plans and open questions after the v2 provider-matrix sweep. Not a spec — a
prioritised plan for the next 1-3 work sessions.

**Latest commit state:** `3e6655d` on `feat/eval-benchmark-v2` (not yet pushed).

**Related docs:**

- [`docs/guides/eval-reports/EVAL_HELDOUT_V2_2026_04.md`](../guides/eval-reports/EVAL_HELDOUT_V2_2026_04.md) — authoritative eval report
- [`docs/rfc/RFC-073-autoresearch-v2-framework.md`](../rfc/RFC-073-autoresearch-v2-framework.md) — framework spec
- [`LORA_HYBRID_PIPELINE_PLAN.md`](LORA_HYBRID_PIPELINE_PLAN.md) — companion WIP: LoRA
  fine-tuning plan for hybrid_ml pipeline revival (gated on §3 outcome)

---

## Where we finished

**Completed under v2 framework:**

- Dataset split: `curated_5feeds_dev_v1` (10 ep) + `curated_5feeds_benchmark_v2` (5 ep, held-out)
- Silvers: 4 new Sonnet 4.6 silvers across both datasets × both tracks
- Seed plumbing: OpenAI `seed` wired through Config / Params / factory / provider
- Fraction-based contestation (40% threshold) + Efficiency rubric + JSON prose extraction
- Champion prompts ported across all providers (zero per-provider tuning)
- **6 cloud providers × 4 cells** (bundled/non-bundled × bullets/paragraph) on dev + held-out
- **11 Ollama models non-bundled** on dev + held-out + **3 Ollama models bundled**
- Compound analysis (quality × latency × cost) + Pareto frontier + recommended option order

**Headline:**

- Quality winner non-bundled: DeepSeek (bullets 0.586, paragraph 0.541)
- Quality winner bundled: Anthropic Haiku 4.5 (bullets 0.552, paragraph 0.548)
- Compound default: **Gemini 2.0-flash** (0.562, 2.0s, $0.00035/ep — Pareto-optimal)
- Local champion: qwen3.5:9b (0.580, free, 33s/ep)

---

## Tomorrow's candidate work (priority-ordered)

### 1. Model operating mode audit — highest leverage, low cost

**Why first:** we've been running all models with generic champion prompts. Every model
has specific requirements that may or may not be honoured. Findings here may invalidate
tests from other plans.

**Per-model documentation target** (one-page each):

| Dimension | What to document |
|-----------|------------------|
| Base vs fine-tuned | e.g. BART-large-CNN is CNN-news tuned; domain mismatch |
| Chat template | Gemma2 can't take system messages separately; phi3 uses ChatML; etc. |
| Context window | phi3:mini = 4k → may be truncating; most others 8k-128k |
| Optimal temperature | Some models (Mistral, older Llama) prefer 0.1-0.3 for summarisation |
| System prompt handling | Some ignore, some strict-require |
| Task specialisation | bart-large-cnn vs xsum vs pegasus vs flan-t5 — different strengths |

**Models to audit:**

- Cloud: gpt-4o, claude-haiku-4-5, gemini-2.0-flash, deepseek-chat, mistral-small-latest, grok-3-mini
- Local (11 Ollama): llama3.2:3b, llama3.1:8b, qwen3.5:{9b,27b,35b}, qwen2.5:7b,
  mistral:7b, mistral-nemo:12b, mistral-small3.2, gemma2:9b, phi3:mini
- ML: bart-large-cnn, LED, hybrid's REDUCE stage (llama3.2:3b)

### §1a: Ollama provider corrections (audit findings, 2026-04-16)

Initial pass of the audit surfaced three issues, ranked by leverage. These are
**framework-level** fixes — they affect all 11 local model numbers, not just per-model
quirks. Blocks §3 (ML/hybrid v2) meaningfully, since re-measuring hybrid REDUCE baselines
on broken Ollama is wasted work.

### Findings from deep research pass (2026-04-16, post-correction)

After my initial audit made outdated claims, a dedicated research pass returned evidence
from Ollama 0.19.0 release notes + 2025-2026 community guides. Refined findings:

**Finding #1: `num_ctx` — VRAM-tiered defaults, but silent truncation still real**

- Ollama 0.19.0 picks defaults by VRAM tier: <24GB→4k, 24-48GB→32k, ≥48GB→256k
- On our 48GB M4, default is ~32k — explains why our earlier probes didn't show 2048
- **But silent truncation still happens** when prompt+output exceeds set limit
- gemma2 still structurally capped at 8192 (model spec, not Ollama)
- **Recommendation**: explicit `num_ctx=32768` per request (current code sets 8192 — bump it)

**Finding #2: Qwen3.5 template IS a real bug**

- Ollama-shipped qwen3.5:9b/27b/35b have `TEMPLATE {{ .Prompt }}` — no ChatML role tokens
- OpenAI-compat layer merges system+user as flat text, loses `<|im_start|>`/`<|im_end|>` structure
- Model sees "System: ...\nUser: ..." prose instead of ChatML-tagged messages
- Qwen3.5 instruction-following on system-prompt-heavy tasks is measurably degraded
- Referenced bug reports: ollama/ollama#10980, huggingface Qwen3.5-27B discussion #28
- **Recommendation**: custom Modelfile with proper ChatML template for qwen3.5:9b at minimum

**Finding #3: Q4_0 quant upgrades are free quality**

- Three of our models are on Q4_0, which is strictly worse than Q4_K_M at same size:
  - phi3:mini (Q4_0)
  - gemma2:9b (Q4_0)
  - mistral-nemo:12b (Q4_0)
- **Recommendation**: repull as `<model>:<size>-instruct-q4_K_M` tags. No memory/latency cost,
  measurable quality. Likely explains phi3:mini's 0.475 bullets and gemma2:9b's 0.492.
- Small models (phi3, llama3.2:3b) could additionally go Q6_K/Q8_0 for marginal further gain,
  but Q4_K_M → Q6_K is task-dependent (do later if worth it).

**Finding #4: Per-model sampler quirks**

- `mistral-small3.2` should run at **temp=0.15**, not 0.0 (Mistral's own guidance; we've been
  forcing 0.0 for our whole matrix)
- `mistral-nemo:12b` needs `repetition_penalty=1.05` — known to get repetitive
- `gemma2:9b` has a known runaway-generation bug (ollama/ollama#5341) — needs `num_predict`
  + stop tokens. May partially explain held-out paragraph quality issues.
- `phi3:mini`: stick to 4k variant (128k variant produces gibberish > 4k tokens)
- `llama3.1:8b`: Meta defaults temp=0.8, we use 0.0 ✓ (correct for bullet lists)

**Finding #5: Ollama MLX backend preview available on 0.19.0**

- Apple Silicon native backend, ~1.6x prefill / 2x decode on M4
- Currently only ships MLX-optimised weights for `qwen3.5:35b-a3b-coding-nvfp4`
- Sampler is tuned for code, needs override for summarisation
- Maybe worth exploring later; not a priority

**Verified OK (no change needed):**

- Gemma2 system-prompt merging works correctly
- phi3:mini template is proper ChatML
- llama3.2:3b / llama3.1:8b / qwen2.5:7b / mistral:7b / mistral-small3.2 templates all
  OK with current OpenAI-compat layer

**Our existing v2 local numbers are still basically correct** (not truncated as I feared).
Findings #3 and #4 could lift specific models; #2 could lift qwen3.5 family. #1 is
defensive.

**Bug #2: Qwen3.5 template is literally `{{ .Prompt }}`**

- All three Qwen3.5 variants (9b, 27b, 35b) ship with a no-op template — no ChatML
  role tokens, no system-prompt structure
- Ollama's OpenAI-compat layer concatenates system+user as raw text
- Qwen3.5 was trained with ChatML format (`<|im_start|>...`)
- qwen3.5:9b still scored 0.580 despite this — suggests Qwen is robust to wrong template
- **Expected fix impact**: unclear, but worth a ~2 hour investigation. Build custom
  Modelfile for qwen3.5:9b with correct ChatML template, re-run 1 cell, see if lift.

**Opportunity #3: quantization upgrade for small models**

- All 11 models currently at Q4 (mostly Q4_K_M). Defaults pulled by Ollama.
- M4 48GB has plenty of headroom for Q8_0 on everything up to ~13B
- Small models (3B, 7B) are more sensitive to precision loss; benefit most from Q8
- Pull `Q8_0` / `F16` variants of: phi3:mini, llama3.2:3b, mistral:7b
- Re-run 3 × 4 cells = 12 cells
- Expected lift: 1-5% per small model (smaller = more sensitive to quant)

**Not bugs (verified OK):**

- **Gemma2 system-prompt merging**: template correctly merges system→user with a space
  separator. Gemma2's native format; nothing to fix.
- **phi3:mini template**: proper ChatML-style (`<|system|>`, `<|user|>`, `<|assistant|>`,
  `<|end|>`). Fine.

### Expected cloud-side audit items (not yet checked)

- Provider-level max_tokens / response_format settings per cloud backend
- ML models (bart-large-cnn, LED) — chunking behaviour, max-input-tokens, decoding params
- Default temperature mismatches:
  - mistral-small3.2 default temp=0.15 (we override to 0 ✓)
  - qwen3.5 family default temp=1, top_p=0.95, top_k=20, presence_penalty=1.5 — our
    temperature=0 override works but other params may still leak through Modelfile defaults

### 2. Summarisation-specialist model swap for ML track

**Why:** BART-large-CNN is fine-tuned on CNN/DailyMail news. 500-800 word articles, formal
style. Podcasts are 10k-word conversational content. Structural mismatch.

**Candidates to swap in (already on HF, no training required):**

- `google/long-t5-tglobal-large` — explicit long-context (up to 16k), summarisation specialist
- `philschmid/flan-t5-base-samsum` — instruction-tuned, dialogue-summary specialist
- `google/pegasus-large` — Pegasus original
- `facebook/bart-large-xsum` — extreme-abstractive baseline

**Plan:** swap one alternative into the ML pipeline, run 4 cells (bullets + paragraph × dev +
held-out), compare to v2 BART+LED baseline. If a specialist beats BART, it becomes the
ML-track backbone for the ML/hybrid v2 runs.

### 3. ML + hybrid_ml under v2 framework — **start with better base models, not v1 defaults**

**Why:** ADR-073 closed these under v1 methodology (contaminated smoke/benchmark, binary-OR
contestation, Conciseness rubric). Need v2 numbers to know where "old school" actually lands.

**Critical insight from v2 data:** v1's hybrid used `llama3.2:3b` for REDUCE because that
was the smallest Ollama model at the time — not because it was optimal. The v2 matrix shows
llama3.2:3b (0.501) is **the weakest of the 7B+ class**. Running hybrid under v2 with the
v1 base is measuring the wrong thing.

**Right order: pick the best bases, THEN measure.**

| Stage | v1 default | Better v2-informed choice | v2 evidence |
|-------|-----------|---------------------------|-------------|
| MAP | `bart-large-cnn` | `google/long-t5-tglobal-large` (from #2) | Long-context, summarisation specialist, not CNN-news-tuned |
| REDUCE | `llama3.2:3b` | `qwen3.5:9b` or `mistral:7b` | qwen3.5:9b scored 0.580 non-bundled (vs llama3.2:3b at 0.501); mistral:7b (0.526) is a balanced middle pick |

**Scope:**

- Pure ML (MAP-only, no REDUCE): long-t5 → 4 cells
- Hybrid (MAP + LLM REDUCE): long-t5 + qwen3.5:9b → 4 cells
- Optionally, for comparison: hybrid with mistral:7b REDUCE → 4 cells (if time)
- v1 configs (`ml_bart_led_autoresearch_v1`, `ml_hybrid_bart_llama32_3b_autoresearch_v1`):
  keep as-is per reproducibility policy — don't re-run them under v2. Create new v2 configs
  with the better bases.

**Expected v2 numbers (with modernised bases):**

- Pure long-t5 ML: 0.35-0.45 range (big jump from v1's bart-led 0.20 if long-t5's long
  context genuinely helps on 10k-word podcasts).
- Hybrid long-t5 + qwen3.5:9b: 0.50-0.60 range (un-tuned). Could be genuinely competitive
  with top local LLM (qwen3.5:9b standalone at 0.580) since REDUCE gets structured input,
  not raw transcript.

**Real question:** can modernised-base hybrid match or beat qwen3.5:9b standalone? If yes,
LoRA becomes optional rather than necessary. If no, LoRA is where the remaining gap lives —
see [`LORA_HYBRID_PIPELINE_PLAN.md`](LORA_HYBRID_PIPELINE_PLAN.md).

**Where hybrid still wins regardless:**

- Fully deterministic (real `temperature=0`, no API variance)
- Zero-API-key / single-binary deploy
- CPU-only / edge devices
- Predictable latency (no API rate-limits / cold start)

### 4. Unexplored low-cost / low-latency API model variants

**Hypothesis:** current picks (gpt-4o, haiku-4.5, gemini-2.0-flash, deepseek-chat, mistral-small,
grok-3-mini) are one model per vendor. Cheaper or newer variants from same vendors might be
Pareto-optimal and we missed them.

**Phase 1 (highest-leverage candidates):**

| Candidate | Why interesting | Against |
|-----------|----------------|---------|
| `gpt-4o-mini` | $0.15/$0.60 vs gpt-4o's $2.50/$10 (16× cheaper). If within 5% quality, replaces gpt-4o. | — |
| `gemini-2.5-flash-lite` | Gemini cheap-tier on 2.5 generation; no thinking-config issue | — |
| `mistral-medium` | Mid-tier Mistral we haven't tested; may close gap to DeepSeek | Likely more expensive than `small` |

**Phase 2 (generation upgrades if SDK/API ready):**

- `gemini-2.5-flash` — pending `google-genai` SDK upgrade for `thinking_budget` field
- `grok-4` — if reasonably priced vs grok-3-mini

**Phase 3 (specialised, only if Phase 1 didn't find a new champion):**

- `deepseek-reasoner` — reasoning model; different latency/quality tradeoff
- `gpt-5-mini` or `gpt-5-nano` if GPT-5 generation available

**Deliberately skip:**

- Flagship tiers (gpt-5, sonnet-4.6 as candidate, opus, gemini-2.5-pro) — expensive and
  unlikely to beat DeepSeek on quality-per-dollar
- Reasoning models (o1, o3) — slow, expensive, not summarisation-optimised
- Third-party hosted open models (Groq, Together.ai for Llama) — different
  provider complicates the matrix

### 5. LoRA fine-tuning on silvers — separate plan (deferred, multi-session project)

Dedicated plan lives in [`LORA_HYBRID_PIPELINE_PLAN.md`](LORA_HYBRID_PIPELINE_PLAN.md).
Summary of the framing:

- **Target**: lift the hybrid pipeline from v2 baseline to genuinely competitive status
  (potentially matching qwen3.5:9b standalone). The hybrid is the only path with two
  fine-tunable stages; cloud LLMs can't be trained, pure local LLMs are a dead end
  relative to qwen3.5:9b.
- **Feasibility on M4 48GB**: yes for 3B-9B models, via MLX (LLMs) + PyTorch PEFT (BART).
  No QLoRA because bitsandbytes doesn't support Apple MPS.
- **Real bottleneck**: 20 silver examples is below LoRA threshold. Pre-phase: generate
  200+ new silvers using Sonnet 4.6 on non-eval podcasts (~$2 in credits, 1 day).
- **Condition to start**: Phase 0 baseline from §3 must show remaining gap after bases
  are modernised. If long-t5 + qwen3.5:9b un-tuned hybrid already hits 0.55+, LoRA may
  not be worth the investment.
- **Multi-session**: realistic 1 week of focused work (data gen + training infra +
  experiments + evaluation).

**Explicitly NOT tomorrow's work.** Gated on §3 results showing a gap worth closing.

---

## Suggested tomorrow sequencing

Chain findings rather than running in parallel. Each step's result may reshape the next.

### Critical dependency: §2 → §3

Don't evaluate ML/hybrid v2 before picking the right bases. §2 picks the MAP base; §3 should
use that choice. Similarly, §3 uses the v2 data to pick the REDUCE base (qwen3.5:9b or
mistral:7b, not v1's llama3.2:3b).

### Tomorrow's realistic plan

Audit findings (§1a) changed the priority ordering — num_ctx fix blocks everything
downstream. Fix it before any further Ollama or ML/hybrid evaluation.

1. **Fix #1 (~1h incl. re-run): num_ctx=8192 in Ollama provider**
   - Single-line fix in provider + thread through via cfg / params
   - Re-run 11 local models × 4 non-bundled cells (~45 min wall-clock)
   - Update matrix with corrected numbers
   - **Decision point**: if local rankings shift meaningfully, pause to re-think
     REDUCE base pick for §3 (qwen3.5:9b may no longer be the clear winner)
2. **Fix #2 (~1h): quantization upgrade for small models**
   - Pull Q8_0 or F16 variants of phi3:mini, llama3.2:3b, mistral:7b
   - Re-run 3 × 4 non-bundled cells (~30 min)
   - Small-models-only; skip large-model quant upgrades (diminishing returns)
3. **Fix #3 (optional, ~2h): Qwen3.5 ChatML template**
   - Build custom Modelfile with proper ChatML for qwen3.5:9b only (the current winner)
   - Re-run 4 non-bundled cells, compare to post-Fix-#1 numbers
   - Skip if Fix #1 already changed rankings enough that Qwen isn't the focus
4. **(remainder of day) Proceed with original §2 → §3 plan**
   - Summarisation-specialist MAP swap (long-t5)
   - ML + hybrid v2 with corrected-numbers-informed REDUCE base (likely still qwen3.5:9b
     but verify post-Fix #1)
5. **Decision point after §3**: modernised hybrid vs qwen3.5:9b standalone, informs LoRA
   go/no-go (see [`LORA_HYBRID_PIPELINE_PLAN.md`](LORA_HYBRID_PIPELINE_PLAN.md))
6. **Optional: new API variants** (#4 phase 1)
   - gpt-4o-mini, gemini-2.5-flash-lite, mistral-medium
   - Only if steps 1-4 finish with time remaining

Sequencing rationale: fix #1 universally affects local numbers, so doing it first means
every downstream comparison (§2, §3, LoRA base pick) uses accurate data. Doing it later
wastes any measurement taken before it. Fix #2 and #3 are additive lifts on top.

---

## Open questions worth revisiting

- **Does Ollama's OpenAI-compat layer preserve each model's native chat template?** If
  not, we've been using a generic template everywhere — a subtle fix with potentially
  wide impact across all 11 local models.
- **Should we grow the held-out dataset?** 5 episodes → ±5% noise. Would 10 held-out
  episodes be worth sourcing? Already in RFC-073 §Future Work.
- **Is the dev/held-out split contaminated for the DEV dataset due to iteration?**
  Champion prompts were ported, not tuned per-provider. But the prompt structure was
  developed on OpenAI's dev set. The exact amount of leakage is small but non-zero.
- **Multi-run averaging vs larger held-out:** both reduce noise. Pick based on
  per-experiment cost vs infrastructure change.

---

## What NOT to do next session

- **Don't run any Ollama evaluation before fixing `num_ctx`** — current numbers are
  measured on 45%-truncated transcripts. Fix first, then measure.
- **Don't evaluate ML/hybrid with v1 base models** (BART-large-cnn + llama3.2:3b).
  Pick the v2-informed bases first (long-t5 + qwen3.5:9b-post-num_ctx-fix). Evaluating
  with v1 defaults is measuring the wrong question.
- **Don't do multi-run averaging yet** — bigger infra change, not the highest leverage
- **Don't tune per-provider prompts** — diminishing returns relative to model-specific
  fixes (which the audit will surface)
- **Don't add more Ollama models** — we've saturated the local-model value
- **Don't introduce third-party hosted open models** (Groq, Together, etc.) — muddies
  the matrix without clear quality signal
- **Don't jump into LoRA fine-tuning** — gated on §3 result; see
  [`LORA_HYBRID_PIPELINE_PLAN.md`](LORA_HYBRID_PIPELINE_PLAN.md). Multi-session project,
  not tomorrow's work.
- **Don't re-run v1 runs under v2 just to compare** — keep old artifacts as-is per
  reproducibility policy. Create new v2 configs for the modernised pipelines.
