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

**Expected quick fixes:**

1. **phi3:mini context truncation** — verify 4k token limit isn't silently cutting our
   ~3.8k total (prompt + transcript + output). If yes, either bump via Ollama params or
   accept it as the floor.
2. **Gemma2 system-prompt handling** — verify Ollama's OpenAI-compat layer is merging
   our system prompt into the user message correctly (Gemma2 schema doesn't accept
   separate system). Current score 0.492 — may be a handling issue, not model ceiling.
3. **Chat template verification** per Ollama model — the `template` field in Modelfile
   should be inspected for each installed model.

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

1. **Morning (2-3h): Model operating mode audit** (#1)
   - Read model cards for all 11 Ollama + BART + LED
   - One-page-per-model reference table (chat template, context window, temperature,
     specialisation, chat-template handling)
   - Flag any "we're using it wrong" bugs
2. **Late morning (~1h): Quick fixes from audit** (#1 fix)
   - phi3 context truncation, gemma2 system-prompt merge, etc.
   - Re-run only the affected cells
3. **Early afternoon (~1.5h): ML-track alternative MAP backbone** (#2)
   - Swap one summariser specialist (long-t5 recommended), 4 cells
   - This picks the MAP base for step 4
4. **Mid-late afternoon (~2h): ML + hybrid v2 with modernised bases** (#3)
   - Pure ML: long-t5 MAP (from step 3) on 4 cells
   - Hybrid: long-t5 MAP + **qwen3.5:9b REDUCE** (not v1's llama3.2:3b!) on 4 cells
   - Apply v2 champion REDUCE prompt
   - Record: ROUGE-L, final score, latency per stage
5. **Decision point after step 4**: does modernised-base hybrid match qwen3.5:9b standalone?
   - **Yes** → ML/hybrid revived; update guide; next session can pursue new API variants
     (#4) or LoRA (#5)
   - **No (gap remains)** → gap quantified; informs whether LoRA in #5 is worth the cost
6. **Late afternoon (optional): new API variants** (#4 phase 1)
   - gpt-4o-mini, gemini-2.5-flash-lite, mistral-medium
   - Update matrix if any lands on Pareto frontier

Steps 1-4 alone justify a full day. Step 5 is optional. LoRA (#5 in §5) is explicitly
NOT part of this day — it's gated on step 5 showing a gap worth closing.

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

- **Don't evaluate ML/hybrid with v1 base models** (BART-large-cnn + llama3.2:3b).
  Pick the v2-informed bases first (long-t5 + qwen3.5:9b). Evaluating with v1 defaults
  is measuring the wrong question.
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
