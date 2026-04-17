# ML Model Research — Podcast Summarization, April 2026

Research-agent output + commentary. Shortlist of HuggingFace models worth empirically
testing on our M4/48GB Apple Silicon setup for podcast-transcript summarization.

**Context**: after v2 eval (2026-04-15/16) showed hybrid bart+llama3.2:3b at 0.430 and
hybrid bart+qwen3.5:9b at 0.448 on held-out paragraph, while standalone qwen3.5:9b
bundled hits 0.509 — i.e. hybrid loses to standalone when REDUCE is capable. This
research asks whether any **specialist HF model** can change that story.

**Top-line finding**: one genuinely structural candidate exists.

---

## Top pick: `DISLab/SummLlama3.2-3B` 🎯

- Llama-3.2-3B-Instruct base, DPO-tuned on FeedSum dataset
- DPO training axes: **faithfulness, completeness, conciseness**
- These map 1:1 to our v2 judge rubric dimensions (Coverage, Accuracy, Efficiency)
- Paper: "Learning to Summarize from LLM-generated Feedback" (Song et al., 2024)
  — the 8B sibling beats Llama-3-70B-Instruct on human-preferred summary axes
- HF: `DISLab/SummLlama3.2-3B`
- Size: ~6.5GB on disk, ~7-8GB working set fp16
- Context: 128K (inherits Llama-3.2)
- MPS: Llama-3 arch, works on torch ≥2.4 / transformers ≥4.45

**Why this is interesting**: no other open-weights summarization model is aligned to the
exact quality axes our judges measure. Structural advantage.

---

## Full shortlist (ranked by expected payoff)

| Rank | Model | Role | Size | Notes |
|------|-------|------|------|-------|
| 1 | `DISLab/SummLlama3.2-3B` | REDUCE | ~7GB fp16 | DPO on our rubric axes; biggest expected delta |
| 2 | `Qwen/Qwen3-4B-Instruct-2507` | REDUCE fallback | ~8GB fp16 | Jul 2025 refresh; generalist |
| 3 | `pszemraj/long-t5-tglobal-xl-16384-book-summary` | REDUCE | ~12GB fp16 | 30× scale-up of current choice |
| 4 | `pszemraj/pegasus-x-large-book-summary` | REDUCE | ~2GB fp16 | Efficient long-context, 16k input |
| 5 | `philschmid/flan-t5-base-samsum` | MAP | ~1GB fp16 | Dialogue-tuned, 1024-token window |

**Also worth considering** (niche candidates):

- `mikeadimech/pegasus-qmsum-meeting-summarization` — Pegasus-X fine-tuned on meeting dialogue (AMI/ICSI), closest public proxy to podcast conversation
- `jasonmcaffee/flan-t5-large-samsum` — larger SamSum-tuned variant (780M) if base-size undershoots

---

## Skip list (and why)

- `paulowoicho/t5-podcast-summarisation` — known bug: emits hashtags + promotional text (trained on Spotify creator descriptions, not actual summaries). Documented defect on card.
- Any `-8bit` variants (`pszemraj/*-8bit`) — `bitsandbytes`, CUDA-only, won't load on MPS
- Apple Foundation Models — no HF weights, gated/API-only
- 30B+ models (Qwen3-30B-A3B, GPT-OSS-120B, GLM-4.5V) — recreate Ollama stack in slower runtime, no gain
- `Qwen3-4B-Thinking-2507` — emits `<think>` traces that need parsing; no upside for summarization
- `prithivMLmods/Llama-Chat-Summary-3.2-3B` — overlaps SummLlama3.2 but with less rigorous alignment story (no public eval on our axes)

---

## MPS operational notes

Critical setup details for running on M4:

- **PyTorch ≥2.4** required (fixes non-contiguous `addcmul_`/`addcdiv_` MPS kernel bug). Ideally ≥2.6.
- **Use `torch.float16`**, not bfloat16 (bf16 works but measurably slower on M-series).
- **`PYTORCH_ENABLE_MPS_FALLBACK=1`** as safety net for any unimplemented ops during `.generate()`.
- LongT5 / Pegasus-X use pure PyTorch attention — no custom kernel concerns.
- Avoid `bitsandbytes`, `auto-gptq`, `awq` unless fp16 sibling exists (CUDA-only).

---

## Empirical test order

Following the agent's recommendation:

1. **SummLlama3.2-3B REDUCE** — swap REDUCE from llama3.2:3b → SummLlama3.2-3B in hybrid pipeline. If this reaches or exceeds qwen3.5:9b standalone's 0.509, hybrid+ML is back in the picture with a genuinely specialist model.
2. **long-t5-tglobal-xl REDUCE** — isolates "current model but 30× larger" effect vs #1's "different architecture entirely" effect.
3. **flan-t5-base-samsum MAP** with best REDUCE from #1/#2 — dialogue-tuned MAP vs current BART.
4. **Qwen3-4B-Instruct-2507** as fallback REDUCE if SummLlama underperforms.
5. **pegasus-qmsum** if dialogue register specifically feels wrong.

---

## Expected outcomes (commit to predictions before running)

| Scenario | Expected held-out paragraph | Action |
|----------|:---------------------------:|--------|
| SummLlama3.2 hybrid ≥ 0.55 | — | **New Tier 3 pick**; hybrid revived |
| SummLlama3.2 hybrid 0.48-0.55 | — | Competitive with standalone qwen3.5:9b; keep documented |
| SummLlama3.2 hybrid < 0.48 | — | Another null result; ML/hybrid stays dead |
| SummLlama3.2 standalone (non-hybrid) on full transcript | — | **This is the real question** — could be better than in-hybrid |

**Note**: the interesting experiment may be **SummLlama3.2 as a standalone summarizer** (not in hybrid), feeding it the full transcript up to its 128k context. If it can handle our 10-40k char inputs directly without MAP preprocessing, it becomes a completely new category — ML-style local deployment without needing Ollama.

---

## Downloads required

Total ~20GB for the top 3 candidates:

- SummLlama3.2-3B: ~6.5GB
- long-t5-tglobal-xl: ~12GB
- flan-t5-base-samsum: ~1GB

On home wifi, manageable background download while planning configs.

---

## Source list (for further reading)

- [DISLab/SummLlama3.2-3B](https://huggingface.co/DISLab/SummLlama3.2-3B)
- [Learning to Summarize from LLM-generated Feedback (arXiv)](https://arxiv.org/html/2410.13116v2)
- [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [pszemraj/long-t5-tglobal-xl-16384-book-summary](https://huggingface.co/pszemraj/long-t5-tglobal-xl-16384-book-summary)
- [pszemraj/pegasus-x-large-book-summary](https://huggingface.co/pszemraj/pegasus-x-large-book-summary)
- [mikeadimech/pegasus-qmsum-meeting-summarization](https://huggingface.co/mikeadimech/pegasus-qmsum-meeting-summarization)
- [philschmid/flan-t5-base-samsum](https://huggingface.co/philschmid/flan-t5-base-samsum)
- [Elana Simon — the MPS bug that taught me PyTorch (2025)](https://elanapearl.github.io/blog/2025/the-bug-that-taught-me-pytorch/)
- [Comparative Study of PEGASUS, BART, T5 (MDPI 2025)](https://www.mdpi.com/1999-5903/17/9/389)
