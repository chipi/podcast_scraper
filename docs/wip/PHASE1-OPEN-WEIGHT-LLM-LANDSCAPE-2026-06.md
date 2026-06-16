# Phase 1: Open-Weight LLM Landscape (≤35B total params), as of 2026-06

**Status:** Phase 1 (analysis) complete. Phase 2 (eval) is tracked at
**[#1016](https://github.com/chipi/podcast_scraper/issues/1016)**;
this doc is the canonical record of how we arrived at the tier-1
shortlist.

**Issue:** #928 reframe (parent #928 itself is closed; this reframes its
Cell C re-baseline scope into a broader landscape analysis +
candidate-selection question). Phase 2 execution: see #1016.
**Target hardware:** NVIDIA Blackwell GB10, 128 GB unified memory.
**Use case:** Three pipeline stages on DGX — summary, GI (insight +
quote extraction with structured JSON output), KG (Topic/Entity/relation
extraction with structured JSON output). Long-form podcast audio
(~10-130 min episodes), transcripts up to ~14K tokens.
**Hard constraint:** ≤35B total params. ≥32K context length (long
episodes don't fit a 4K-context model). Open-weight (Apache-2.0, MIT,
Llama Community, or similar — no CC-BY-NC / research-only).

---

## TL;DR — tier-1 shortlist (decided 2026-06-16)

| # | Candidate | Class | Why it's on the list | Pre-eval adapter work |
|---|-----------|-------|----------------------|------------------------|
| 1 | **Qwen3.5:35b (Ollama Q4_K_M)** | instruct-only | **Incumbent.** Current `cloud_with_dgx_*` summary champion (Sonnet 5.00 / GPT-5.4 4.90 in EVAL_FINALE_SMOKE_V2_2026_06). The "did we improve?" anchor. | None |
| 2 | **Qwen3-30B-A3B-Instruct-2507** | hybrid (reasoning-off toggle) | Live on the autoresearch vLLM (`:8003`). 30B/3B-active MoE = cheap to serve. Apache-2.0, 256K context. The "newer Qwen vintage" challenger to #1. | Light: `enable_thinking=False` + honor `generation_config.json` sampling defaults |
| 3 | **DeepSeek-R1-Distill-Qwen-32B** | pure reasoning | The reasoning bet. Existing #961 prompt fix lifts mean 3.25→4.05; still 0.95 points behind qwen3.5:35b. Question: with proper optimization phase, can it close the gap? Tests whether reasoning beats instruct on GI/KG specifically (those have structured-output requirements that might benefit from controlled reasoning). | **Heavy**: prompt rewrite suppressing "Okay, so I need to..." prose; sampling tune; possibly post-processor for stray reasoning |
| 4 | **Magistral Small 1.2 (24B)** | reasoning-native | Mistral's reasoning play. Same harness shape as #3; different model family. Apache-2.0. Useful for "is reasoning the variable, or is it model family?" question alongside #3. | Heavy (mirrors #3 pattern) |
| 5 | **Mistral Small 4 (24B, multimodal)** | instruct | Newer Mistral instruct, never seen on our matrix. Apache-2.0. The "instruct-only Mistral" counterpart to #4 — controls for "is reasoning the differentiator within Mistral?" | Light |
| 6 | **gemini-2.5-flash-lite** | cloud anchor | "Do we even need DGX?" baseline. The cloud-side reference cost+latency point. | None (production-tuned) |

**Not on the list, with reasons** (further detail in §"What we ruled out"):

- 70B+ candidates (Llama-3.3-70B, DeepSeek-R1-70B, Hermes-3-70B) — out
  of scope per ≤35B cap.
- Gemma 3 27B — written off in EVAL_SMOKE_V2_DGX_REFRESH_2026_06 even
  after Gemma-native prompt tuning. Multimodal-tuning bias hurts text
  summary.
- Phi-4 14B — Microsoft itself flags weak IFEval; rank-13 of 22 on
  Opus rescoring (RougeL 0.240).
- gpt-oss 20B — rank-18, ~14% below qwen3.5:35b baseline.
- Llama 3.1 8B — paragraph contestation on every held-out episode.
- Hermes-3 8B — already production-default for laptop tier (`local.yaml`);
  we know what it does. Not the slot we need to fill.
- Mistral-small 3.2 — known-stable, but the newer Mistral Small 4 dominates
  it on the same evaluation if the vendor's headlines are right; cheaper
  to just test the newer one.
- Yi 1.5, OLMo 2, Cohere CommandR, Falcon, Reka Flash 3, NVIDIA Nemotron
  Nano, IBM Granite — never tested internally; some have license blocks
  (CommandR is CC-BY-NC). Could be follow-up "wildcard" candidates if
  the tier-1 sweep doesn't surface a clear winner.

## Reasoning-model handling taxonomy

Each model class requires different harness handling — applying one
template + sampling config across all six candidates would penalize
some unfairly and let others coast.

| Class | Tier-1 members | Harness adapter pattern |
|-------|----------------|--------------------------|
| **Instruct-only** | Qwen3.5:35b, Mistral Small 4, gemini-2.5-flash-lite | Standard prompt template, vendor-recommended sampling. Bread-and-butter case. |
| **Hybrid (reasoning toggle)** | Qwen3-30B-A3B-Instruct-2507 | Must explicitly set `enable_thinking=False` (Qwen3 toggle). Honor `generation_config.json` sampling defaults from the model card. Monitor for leak in edge cases (toggle has been seen to "stick" wrong on certain prompt patterns in Qwen3.5 family). |
| **Pure reasoning** | DeepSeek-R1-Distill-Qwen-32B, Magistral Small 1.2 | Prompt template suppressing reasoning prose ("Okay, so I need to..."); we have one for R1 at `src/podcast_scraper/prompts/vllm/r1_distill_32b/summarization/` (lifted mean 3.25→4.05; #961). Same shape needs porting for Magistral. Likely also need post-processor to strip stray `<think>` blocks if model leaks despite prompt. Sampling tuned to vendor's stated reasoning defaults (typically temp 0.6, top_p 0.95). |

**Why this matters for the eval:** the "what's the best model" question
is not separable from "with what harness setup." The fair comparison
requires per-model optimization first, then comparison with all configs
locked.

## Per-stage hypothesis — where the routing-mix question lives

Operator explicitly wants per-stage assessment to evaluate whether a
mixed-routing setup (e.g. summary on Qwen, GI on a reasoner, KG on a
smaller model) is better than single-model routing. Hypotheses going
into the eval:

| Stage | Why a particular class might win | Tier-1 candidates expected to lead |
|-------|----------------------------------|------------------------------------|
| **Summary** | Known-shape task, short structured prose output. Generalist instruct models tend to win; reasoning prose actively hurts. | Qwen3.5:35b (incumbent), Qwen3-30B-A3B-Instruct-2507, Mistral Small 4, gemini-2.5-flash-lite |
| **GI** (insight + quote tuples in JSON) | Two sub-tasks: (a) extract insight from evidence (some reasoning helps), (b) emit clean structured JSON (reasoning models sometimes break JSON). Could go either way. | The reasoners (#3, #4) might surprise us if they hold JSON. Qwen3-30B-A3B-Instruct-2507's hybrid mode is interesting here. |
| **KG** (Topic/Entity/relation in JSON) | More structural, less reasoning-needed. Recall + precision on entity extraction matters; clean JSON output matters more. | Instruct-only candidates expected to lead. |

If the eval confirms different winners per stage, the routing
recommendation lands as "mixed per-stage" (probably via
`*_provider` per-stage config in profiles, which already exists in
the codebase). If one model wins everywhere, the recommendation is
"swap from qwen3.5:35b to <winner>".

## Methodology (Phase 2 scope outline)

### Phase 2a — per-candidate optimization (sequential, dedicated DGX per model)

For each candidate in turn (in the order below):

1. **Set harness adapter** — prompt template, sampling, reasoning toggle.
   For reasoning models, port the prompt-engineering pattern from the
   #961 R1 fix.
2. **Smoke 2 episodes** (1 short ~10 min, 1 long ~30 min from the v2
   dev set). Verify clean output for all 3 stages: no reasoning leak,
   valid JSON for GI/KG, no truncation.
3. **Tune** prompts if quality looks visibly off — 1-2 iterations max.
4. **Lock the config** + commit to
   `src/podcast_scraper/prompts/<provider>/<model_id>/{summary,gi,kg}/`
   so the comparison phase is reproducible.
5. **Time budget per model:** ~half-day. If a candidate needs more to
   look reasonable, it's downgraded from tier-1.

### Phase 2b — formal comparison (all 6 candidates with locked configs)

- 10 episodes from the v2 dev set (`curated_5feeds_dev_v1`).
- Per-stage scoring: summary + GI + KG separately.
- Sonnet 4.6 silver + GPT-5.4 cross-check (the existing finale judge
  setup; matches EVAL_FINALE_SMOKE_V2_2026_06's shape).
- Per-stage per-candidate results table.
- Verdict per stage: best candidate, second-best, and the routing
  recommendation (single-model vs mixed).

### Execution order (gated by user)

1. **Qwen3.5:35b** (incumbent) — first, to sanity-check the harness
   produces the known-champion numbers. If the harness shape is off
   somewhere, this is where we'd see it.
2. **Qwen3-30B-A3B-Instruct-2507** — the headline challenger. Already
   on DGX (live vLLM), so zero pull cost.
3. **gemini-2.5-flash-lite** (cloud anchor) — fastest to run; gets the
   "is DGX even worth it" baseline established early.
4. **DeepSeek-R1-Distill-Qwen-32B** — heaviest adapter work. Reuses the
   #961 prompt fix as a starting point. Already on DGX
   (`deepseek-r1:32b` in `ollama list`).
5. **Magistral Small 1.2** — second reasoning model. Adapter pattern
   established by #4 above ports here. **NEEDS DOWNLOAD** (~12-15 GB
   at Q4_K_M); not currently on DGX.
6. **Mistral Small 4** — controls for "is reasoning the differentiator
   within Mistral?" alongside #5. **NEEDS DOWNLOAD** (~14-16 GB at
   Q4_K_M); not currently on DGX.

### Budget breakdown (operator cap: $50)

| Component | per-judgment cost | total |
|-----------|--------------------|-------|
| Sonnet 4.6 silver | ~$0.10 / ep / stage | 6 cand × 3 stages × 10 ep × $0.10 = **$18** |
| GPT-5.4 cross-check | ~$0.10 / ep / stage | same = **$18** |
| Gemini predictions (cloud anchor only) | ~$0.001 / call | trivial |
| Optimization-phase smokes (2 ep × 6 candidates × judges) | ~$0.40 / candidate | ~$2 |
| Re-run buffer | | $5 |
| **Total estimated** | | **~$43** |

Inside the $50 cap with margin for one judgment re-run on the closest
verdict pair.

### Downloads needed

Only 2 of the 6 candidates need a HuggingFace pull:

- Magistral Small 1.2 (`mistralai/Magistral-Small-1.2` or the Ollama tag
  if available). ~12-15 GB at Q4_K_M.
- Mistral Small 4 (`mistralai/Mistral-Small-4-...`). ~14-16 GB at
  Q4_K_M.

Combined: ~30 GB on DGX. DGX currently has ~2.6T free per #948 probe.
Trivial impact on disk; minutes-to-tens-of-minutes on a typical home
network.

### What's NOT in Phase 2

- 70B+ candidates (re-stated for clarity).
- Mistral-small 3.2 even though it's already on DGX (Mistral Small 4
  supersedes it).
- Yi 1.5 / OLMo / CommandR / Falcon / Reka / Nemotron / Granite. Held
  for a possible "wildcard sweep" follow-up if the tier-1 result is
  close or surprising.
- Latency-under-burst comparison (separate operational question; the
  per-stage quality verdict doesn't need it).
- Multi-episode-length stratification of the quality eval. 10 episodes
  from the dev set is enough for the verdict at this sample size.

---

# Reference appendix — full landscape (informational)

The sections below are research-grade family-by-family coverage of
viable ≤35B open-weight LLMs as of 2026-06. They informed the tier-1
shortlist above. Numbers are vendor-reported unless noted; where a
score is not findable in public sources, it's marked `n/a` rather than
guessed.

---

## 1. Qwen (Alibaba)

The most active ≤35B story at Qwen across 2025–2026 spans three generations:
Qwen3, Qwen3.5, and the very recent Qwen3.6.

- **Qwen3-30B-A3B-Instruct-2507** — 30.5B total / 3.3B active (MoE).
  Released **2025-07-29**. Apache-2.0. Native 256K context. Quants commonly
  distributed: **bf16, FP8, AWQ, GGUF (Q4_K_M / Q5_K_M / Q8_0)**. Strong on
  IFEval and WritingBench alignment; benchmarks consistently published on
  AIME, ZebraLogic, MultiPL-E, LiveCodeBench. Specific IFEval/MMLU numbers
  not surfaced in a single canonical source; treat as "competitive with
  Qwen2.5-32B-Instruct or better." Strength: cheap to serve (3.3B active),
  excellent structured output. Weakness: MoE routing requires recent vLLM
  (≥0.10 or vLLM `25.x`/`26.05` tag).
- **Qwen3-32B (dense, Instruct)** — 32B dense. Released **2025-04-28**.
  Apache-2.0. Base MMLU-Pro 65.5. Quants: bf16, FP8, AWQ, GGUF. Strength:
  dense → simpler serving, good IFEval. Weakness: heavier compute vs. the
  A3B variant for comparable quality.
- **Qwen3.5-27B / Qwen3.5-35B-A3B** — released **2026-02**. Apache-2.0.
  Qwen3.5 is the 2026 successor wave; 35B-A3B is at the edge of the budget.
- **Qwen3.6-27B / Qwen3.6-35B-A3B** — released **2026-04-22**. Apache-2.0.
  Qwen3.6-35B-A3B outperforms dense Qwen3.5-27B on agentic coding/reasoning;
  benchmark tables on HF model cards.

Sources: [Qwen3-30B-A3B-Instruct-2507 HF card](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507),
[Qwen3 technical report](https://arxiv.org/pdf/2505.09388),
[Qwen3.6 blog](https://qwen.ai/blog?id=qwen3.6-35b-a3b).

---

## 2. DeepSeek

V3 and R1 main checkpoints are >35B; only the R1 distillations fit our budget.

- **DeepSeek-R1-Distill-Qwen-32B** — 32.8B dense (Qwen2.5-32B base).
  Released **2025-01-20**. **MIT** (weights), commercial use allowed.
  Strong reasoning posture inherited from R1 traces; MMLU figure of 90.8%
  belongs to full R1, not the distill — distill is materially lower.
  Quants: bf16, FP8, AWQ, GGUF (Q4_K_M widely available, plus bnb-4bit).
  Strength: best-in-class open reasoner at 32B. Weakness: leaks `<think>`
  reasoning prose unless system prompt is firm; sampling defaults matter.
- **DeepSeek-R1-Distill-Llama-8B** — 8B dense (Llama-3.1-8B base).
  Released **2025-01-20**. Weights MIT, base inherits Llama-3.1 community
  license. Quants: bf16, GGUF Q4_K_M / Q5_K_M / Q8_0. Strength: cheap reasoner.
  Weakness: same `<think>` leakage; smaller distill is noticeably less reliable.

Sources: [DeepSeek-R1 paper](https://arxiv.org/html/2501.12948v1),
[R1-Distill-Qwen-32B HF](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B),
[R1-Distill-Llama-8B HF](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B).

---

## 3. Meta Llama

Within the ≤35B budget, Llama 3.1 8B remains the practical Meta option;
Llama 3.2 added vision but didn't ship a new ≤35B text-only flagship beyond 8B,
and Llama 3.3 only published a 70B Instruct variant — no Llama 3.3 8B exists
as of 2026-06.

- **Llama-3.1-8B-Instruct** — 8B dense. Released **2024-07-23**.
  Llama 3.1 Community License (commercial OK with attribution + AUP).
  IFEval ≈ 80.4. 128K context, 8-language multilingual, native tool use.
  Quants: bf16, FP8, AWQ, GPTQ, GGUF Q4_K_M / Q5_K_M / Q8_0 ubiquitous.
  Strength: huge ecosystem support; the most-tooled 8B base on the planet.
  Weakness: smaller scale than the budget allows — leaves headroom unused
  on a 128GB GB10. For summarization/IFEval-heavy work, often outclassed by
  20–32B contemporaries.

Sources: [Llama 3.1 8B Instruct on Artificial Analysis](https://artificialanalysis.ai/models/llama-3-1-instruct-8b),
[MLPerf adds Llama 3.1 8B](https://mlcommons.org/2025/10/training-llama-3-1-8b/).

---

## 4. Google Gemma

- **Gemma-3-27B-IT** — 27.4B dense. Released **2025-03-12**. Gemma license
  (commercial-friendly, custom). 128K context, multimodal (image+text),
  multilingual. **MMLU 78.6** (vs. Gemma-2-27B 75.2), MMLU-Pro 67.5,
  LiveCodeBench 29.7, Bird-SQL 54.4. IFEval not published prominently.
  LMSYS Arena Elo 1339 at launch (top-10). Quants: bf16, FP8, AWQ, GGUF
  (Q4_0 / Q4_K_M / Q5_K_M / Q8_0) all officially or community-distributed.
  Strength: strong general-knowledge + multimodal in a single ≤35B slot;
  excellent for summarization. Weakness: Gemma license is bespoke (not
  Apache); attention pattern (interleaved local/global) needs current
  serving stacks.
- **Gemma-2-27B** — older sibling; still maintained but superseded for new
  workloads.

Sources: [Gemma 3 27B Artificial Analysis](https://artificialanalysis.ai/models/gemma-3-27b),
[Gemma 3 tech report](https://arxiv.org/html/2503.19786v1),
[HF blog: Welcome Gemma 3](https://huggingface.co/blog/gemma3).

---

## 5. Mistral

- **Mistral-Small-3.2-24B-Instruct-2506** — 24B dense. Released **2025-06**.
  Apache-2.0. 128K context. **MMLU 80.5** (essentially flat vs. 3.1),
  HumanEval+ 92.9, MATH 69.4; internal instruction-following accuracy moved
  from 82.75 → 84.78 (vendor metric, not strict IFEval). Quants: bf16, FP8,
  AWQ, GGUF widely available. Strength: improved function calling, cleanest
  tool-use story under 32B. Weakness: vendor doesn't publish IFEval, so
  cross-vendor instruction-following comparison requires harness re-run.
- **Mistral-Nemo-12B-Instruct-2407** — 12B dense. Released **2024-07-18**.
  Apache-2.0. MMLU 68.0 (5-shot). 128K context, multilingual (~11 langs),
  Tekken tokenizer. Quants: bf16, FP8, AWQ, GGUF. Strength: still the best
  Apache-licensed ~12B for general use, beats Gemma 2 9B and Llama 3 8B on
  MMLU/Winogrande. Weakness: aging; surpassed by Mistral-Small-3.2 in
  pure capability.

Sources: [Mistral Small 3.2 OpenRouter](https://openrouter.ai/mistralai/mistral-small-3.2-24b-instruct),
[Mistral NeMo announcement](https://mistral.ai/news/mistral-nemo/),
[Mistral Nemo docs](https://docs.mistral.ai/models/mistral-nemo-12b-24-07).

---

## 6. Microsoft Phi

- **Phi-4** — 14B dense. Released **2024-12-12**, weights re-released under
  **MIT** in January 2025. **MMLU 84.8**, HumanEval 82.6. Strong on
  GSM8K/MATH. Quants: bf16, FP8, AWQ, GGUF Q4_K_M / Q8_0. Strength:
  punches above weight class on math/code/MMLU. **Weakness: notably weak
  on IFEval** — strict instruction-following is the published soft spot.
  Treat as a reasoning/knowledge model, not a controllability model.
- **Phi-4-mini-Instruct** — 3.8B dense. Released **2025-02-01**. MIT.
  Edge/on-device focus. Strength: usable at the bottom of the budget.
  Weakness: 3.8B is far below the GB10 headroom; only relevant as a
  speculative-decoding draft.
- **Phi-4-Reasoning / Phi-4-Reasoning-Plus** — 14B reasoning variants, MIT,
  released mid-2025; relevant if reasoning-style traces are wanted at
  14B scale.

Sources: [microsoft/phi-4 HF](https://huggingface.co/microsoft/phi-4),
[Phi-4 tech report](https://arxiv.org/pdf/2412.08905),
[The Decoder: Phi-4 MIT release](https://the-decoder.com/microsoft-releases-full-phi-4-model-with-weights-under-mit-license/).

---

## 7. OpenAI gpt-oss

- **gpt-oss-20b** — 20.9B total (MoE; ~3.6B active per vendor card).
  Released **2025-08-05**. **Apache-2.0**. **MMLU 85.3**, **IFEval ≈ 69.5**,
  AIME 2025 98.7 (with tool use). 128K context. Quants: bf16, FP8, AWQ,
  GGUF (Q4_K_M, Q8_0); reference checkpoint is MXFP4-trained. Strength:
  reasoning trace quality near o3-mini, runs in 16 GB. Weakness: IFEval
  middling vs. peers — instruction-following is not where it shines;
  reasoning preset can leak `analysis` channels if harness ignores
  Harmony format.

Sources: [Introducing gpt-oss](https://openai.com/index/introducing-gpt-oss/),
[gpt-oss model card PDF](https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf),
[GPT-OSS-20B llm-stats](https://llm-stats.com/models/gpt-oss-20b).

---

## 8. NousResearch Hermes

- **Hermes-4-14B** — 14B dense, fine-tune of **Qwen3-14B base** (not Llama
  3.1 as some 14B/70B/405B writeups claim; the 70B/405B siblings are Llama).
  Released **2025-08-27**. **Apache-2.0** (Qwen3 base license). **MMLU 88.4**,
  **IFEval (Loose) 78.7**. Hybrid `<think>` toggle, ~5M-sample / ~60B-token
  post-training corpus. Quants: bf16, FP8 (official Nous release), GGUF
  (Q4_K_M / Q5_K_M / Q8_0 via bartowski / Mungert / NikolayKozloff).
  Strength: format-faithful outputs, schema/JSON, structured reasoning;
  one of the highest IFEval-on-14B scores in the open ecosystem. Weakness:
  hybrid-mode prompts must explicitly opt into / out of `<think>` or
  you get inconsistent latency and verbosity.

Sources: [Hermes 4 Technical Report](https://nousresearch.com/wp-content/uploads/2025/08/Hermes_4_Technical_Report.pdf),
[Hermes-4-14B HF card](https://huggingface.co/NousResearch/Hermes-4-14B),
[MarkTechPost coverage](https://www.marktechpost.com/2025/08/27/nous-research-team-releases-hermes-4-a-family-of-open-weight-ai-models-with-hybrid-reasoning/).

---

## 9. Allen AI OLMo

- **OLMo-2-32B-Instruct** (0325 release) — 32B dense. Released **2025-03-13**.
  **Apache-2.0**. Fully open: weights + data + training code + checkpoints.
  Trained on 6T tokens, post-trained with Tulu 3.1. First fully-open model
  to beat GPT-3.5-Turbo and GPT-4o-mini on a multi-skill benchmark suite.
  Specific MMLU/IFEval numbers for 32B not surfaced in a single canonical
  page from this search; benchmark tables live in HF SFT card and AI2
  release notes. Quants: bf16, GGUF (Q4_K_M, Q8_0 via community / Ollama),
  AWQ less common. Strength: full reproducibility — the only model in this
  list where the training corpus is also open. Weakness: serving ecosystem
  is thinner than Qwen/Llama; fewer pre-baked production templates.
- OLMo-2-13B / OLMo-2-7B also released earlier (Nov 2024).

Sources: [AI2 OLMo release notes](https://allenai.org/olmo/release-notes),
[2 OLMo 2 Furious paper](https://arxiv.org/pdf/2501.00656),
[MarkTechPost OLMo 32B coverage](https://www.marktechpost.com/2025/03/14/allen-institute-for-ai-ai2-releases-olmo-32b-a-fully-open-model-to-beat-gpt-3-5-and-gpt-4o-mini-on-a-suite-of-multi-skill-benchmarks/).

---

## 10. 01.AI Yi

- **Yi-1.5-34B-Chat** — 34B dense. Released **2024-05** (Yi-1.5 wave).
  **Apache-2.0**. MMLU 77.1. Strong CMMLU / BBH / GSM8K. Quants: bf16,
  GGUF (Q4_K_M / Q5_K_M / Q8_0), AWQ. Strength: bilingual EN+ZH, useful
  if any Chinese summarization is in scope. Weakness: nothing newer from
  01.AI in 2025–2026 within ≤35B at meaningful quality; the family is
  effectively frozen vs. Qwen3/Gemma3.
- **Yi-1.5-9B** — same wave, useful as a draft/speculative model.

Sources: [01-ai/Yi GitHub](https://github.com/01-ai/yi),
[Yi-34B HF](https://huggingface.co/01-ai/Yi-34B),
[Yi paper](https://arxiv.org/pdf/2403.04652).

---

## 11. Cohere CommandR

- **c4ai-command-r-v01** — 35B dense (right at our budget ceiling).
  Released **2024-03**. **CC-BY-NC-4.0** + Acceptable Use Policy → research-only
  for the open weights; commercial use requires Cohere API. 128K context,
  strong RAG / tool-use / multilingual generation in 10 languages.
  Surpasses Llama-70B-Chat / Mixtral / GPT-3.5-Turbo on multilingual MMLU
  and FLORES per the vendor card. Quants: bf16, AWQ, GGUF Q4_K_M / Q5_K_M.
  Strength: built for RAG; verified citations natively. **Weakness: the
  CC-BY-NC license is a hard blocker for any commercial deployment** —
  if production matters, Command-R is academic-only.
- **Command-R7B** — 8B-class, December 2024, same CC-BY-NC posture.

Sources: [c4ai-command-r-v01 HF](https://huggingface.co/CohereLabs/c4ai-command-r-v01),
[Command R+ tech overview (Ruder)](https://www.ruder.io/command-r/),
[Cohere Command R7B blog](https://cohere.com/blog/command-r7b).

---

## 12. TII Falcon

- **Falcon3-10B-Instruct** — 10B dense. Released **2024-12** (Falcon 3 wave).
  **TII Falcon-LLM License 2.0** (permissive, commercial OK, Apache-2.0-derived
  with AUP). Trained on 14T tokens. Base MMLU 73.1, MMLU-Pro 42.5.
  Quants: bf16, GGUF (Q4_K_M, Q8_0), AWQ. Strength: solid 10B Apache-style
  license; Arabic capability added in follow-up. Weakness: ecosystem
  attention has shifted to Qwen3/Gemma3 — fewer downstream tunes.
- **Falcon-2-11B** — 11B dense, May 2024, same TII license family.
  Surpassed Llama-3-8B at release, tied with Gemma-7B. Superseded by
  Falcon 3 for new workloads.

Sources: [Falcon 3 blog](https://falcon-lm.github.io/blog/falcon-3/),
[Falcon3-10B-Instruct HF](https://huggingface.co/tiiuae/Falcon3-10B-Instruct),
[Falcon 2 announcement](https://falconllm.tii.ae/falcon-2.html).

---

## Discoveries (notable families outside the original list)

- **NVIDIA Nemotron Nano 2 / Nemotron 3 Nano** — Nemotron-Nano-2 (Sep 2025,
  ~12B hybrid Mamba-Transformer); Nemotron 3 Nano (late 2025/early 2026,
  30B-class with MoE/hybrid). NVIDIA Open Model License (commercial OK).
  MMLU 77.2 → 77.5 (Nano-2 → Nano-V3 line); leads on harder MMLU variants
  (MMLU-Pro), trails on vanilla MMLU. Quants: bf16, FP8, GGUF, plus
  NVIDIA's own TRT-LLM artifacts. Strength: hybrid arch → lower memory
  growth on long context, native-fit for GB10. Weakness: license is
  vendor-specific (not Apache/MIT). Sources:
  [Nemotron Nano 2 paper](https://arxiv.org/pdf/2508.14444),
  [Nemotron 3 Nano launch](https://llm-stats.com/blog/research/nemotron-3-nano-launch).
- **IBM Granite 3.3 / 4.1** — Granite-3.3-8B-Base released **2025-04-16**,
  Apache-2.0. HumanEval 89.7, AIME 2024 81.2, HellaSwag 80.1. Granite-4.1
  (April 2026) reports MMLU 73.8 / IFEval 87.1 for the 8B-Instruct. Strength:
  Apache + enterprise red-team posture + 512K context on 4.1. Weakness:
  benchmark ceiling lower than Qwen3/Gemma3 at comparable size. Sources:
  [Granite 3.3 base llm-stats](https://llm-stats.com/models/granite-3.3-8b-base),
  [Granite 4.1 announcement](https://www.creativeainews.com/articles/ibm-granite-4-1-open-llm-512k-context-coding/).
- **Reka Flash 3** — 21B dense reasoning model, **Apache-2.0**, open-sourced
  **2025-03-11**. Trained from scratch with synthetic SFT + RLOO. MMLU-Pro
  65.0, competitive with Gemini Pro / GPT-3.5 on language+vision. Quants:
  bf16, GGUF. Strength: from-scratch open reasoner at a non-mainstream
  21B size; useful for cost-quality middle ground. Weakness: smaller
  ecosystem, fewer downstream tunes. Sources:
  [Reka Flash 3 announcement](https://reka.ai/news/introducing-reka-flash),
  [RekaAI/reka-flash-3 HF](https://huggingface.co/RekaAI/reka-flash-3).
- **xAI Grok (open weights)** — Grok-1 (314B MoE, Mar 2024, Apache-2.0) is
  over budget; no ≤35B Grok open release found as of 2026-06. Skip.

---

## Old Synthesis paragraph (historical — superseded by the tier-1
## shortlist at the top of this doc)

The external-research agent's first-pass synthesis suggested a different
five-model shortlist optimized purely on published-benchmark posture:
Qwen3-30B-A3B-Instruct-2507, Hermes-4-14B, Gemma-3-27B-IT,
Mistral-Small-3.2-24B, and OLMo-2-32B-Instruct. After the operator
applied internal-eval state (Gemma 3 written off post-tuning, Hermes-3
already production-default for laptop) and the reasoning-model handling
taxonomy, the tier-1 shortlist at the top of this doc is what we
actually plan to eval. The "wildcards" the external synthesis surfaced
(OLMo-2, Hermes-4, Nemotron Nano, Granite 4.1, Reka Flash 3) are
follow-up candidates if the tier-1 sweep doesn't surface a clear
winner.
