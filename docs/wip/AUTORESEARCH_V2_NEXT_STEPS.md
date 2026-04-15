# Autoresearch v2 — Next Steps WIP

Scratch note consolidating plans and open questions after the v2 provider-matrix sweep.
Not a spec — a dump of intentions and hypotheses for the next work session.

**Latest commit state:** `28f3222` on `feat/eval-benchmark-v2` (not yet pushed).
**Authoritative eval report:** [`docs/guides/eval-reports/EVAL_HELDOUT_V2_2026_04.md`](../guides/eval-reports/EVAL_HELDOUT_V2_2026_04.md)
**Framework RFC:** [RFC-073](../rfc/RFC-073-autoresearch-v2-framework.md)

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

### 3. ML + hybrid_ml under v2 framework

**Why:** ADR-073 closed these under v1 methodology (contaminated smoke/benchmark, binary-OR
contestation, Conciseness rubric). Need v2 numbers to know where "old school" actually lands.

**Scope:**

- `ml_bart_led_autoresearch_v1` (or specialist swap winner from #2) on dev + held-out ×
  bullets + paragraph = 4 cells
- `ml_hybrid_bart_llama32_3b_autoresearch_v1` — same 4 cells. Apply v2 champion prompt
  to the Llama3.2:3b REDUCE stage.

**Expected v2 numbers:**

- Pure BART+LED: 0.25-0.35 range. Will clearly land behind top LLMs, but maybe higher
  than v1's 0.20 once Efficiency rubric stops penalising length and contestation stops
  flipping to ROUGE-only.
- Hybrid with champion REDUCE prompt: 0.40-0.50. Llama3.2:3b non-bundled alone was 0.501
  bullets — hybrid combines BART structure with Llama REDUCE.

**Real question:** is there a niche where ML/hybrid still wins? Candidates:

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

### 5. LoRA fine-tuning on silver references (deferred — bigger project)

**Not a one-day task.** Raised here because it's the biggest potential quality lift,
especially for small local models.

**Idea:** fine-tune small models (BART, Llama3.2:3b, phi3:mini) using LoRA on the 20
silver references we have. Could lift those models from 0.20-0.50 toward 0.55-0.65.

**Why deferred:**

- Needs HF Trainer setup + hyperparameter search
- Needs train/val split of the silvers (and we just built dev/held-out — reusing silvers
  for training would contaminate the held-out)
- Apple Silicon MPS training path untested; may need real GPU
- Multi-day investigation, not one session
- Returns diminish relative to "just use qwen3.5:9b" as the local default

**Condition to revisit:** if a specific deployment constraint requires small models
(edge device, <1GB memory, etc.) and quality matters.

---

## Suggested tomorrow sequencing

Chain findings rather than doing things in parallel. Each step may invalidate/reshape
the next.

1. **Morning (2-3h): Model operating mode audit** (#1)
   - One-page-per-model reference table
   - Flag any "we're using it wrong" bugs
2. **Late morning (~1h): Quick fixes from audit** (#1 fix)
   - phi3 context, gemma2 system-prompt merge, etc.
   - Re-run only the affected cells
3. **Early afternoon (~1h): ML-track alternative backbone** (#2)
   - Swap one summariser specialist, 4 cells
   - Pick winning backbone for step 4
4. **Mid afternoon (~2h): ML + hybrid v2 runs** (#3)
   - Use winning backbone from step 3
   - Apply champion REDUCE prompt to hybrid
5. **Late afternoon (optional): new API variants** (#4 phase 1)
   - gpt-4o-mini, gemini-2.5-flash-lite, mistral-medium
   - Update matrix if any lands on Pareto frontier

If steps 1-4 take the full day, defer step 5 to next session.

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

- **Don't do multi-run averaging yet** — bigger infra change, not the highest leverage
- **Don't tune per-provider prompts** — diminishing returns relative to model-specific
  fixes (which the audit will surface)
- **Don't add more Ollama models** — we've saturated the local-model value
- **Don't introduce third-party hosted open models** (Groq, Together, etc.) — muddies
  the matrix without clear quality signal
- **Don't jump into LoRA fine-tuning** — defer as per #5
- **Don't re-run v1 runs under v2 just to compare** — keep old artifacts as-is per
  reproducibility policy
