# Evaluation Report: Smoke v1 (March 2026)

> **First full provider sweep** — 6 cloud APIs + 11 Ollama local models vs silver
> GPT-4o reference.

| Field | Value |
| ----- | ----- |
| **Date** | March 2026 |
| **Dataset** | `curated_5feeds_smoke_v1` (5 episodes, 5 feeds) |
| **Reference** | `silver_gpt4o_smoke_v1` (GPT-4o summaries) |
| **Schema** | `metrics_summarization_v2` |
| **Configs** | `data/eval/configs/llm_*_smoke_v1.yaml` |
| **Run data** | `data/eval/runs/llm_*_smoke_v1/metrics.json` |

For metric definitions and interpretation guidance, see the
[Evaluation Reports index](index.md).

---

## Key takeaways

1. **OpenAI GPT-4o** scores highest across all metrics — expected, since the silver
   reference is also GPT-4o. Use non-OpenAI comparisons for a fairer picture.
2. **Gemini** (`gemini-2.0-flash`) is the **best non-OpenAI cloud provider** on
   ROUGE-L (33.3%) and embedding similarity (87.3%), with the fastest cloud latency
   (2.7s/ep).
3. **Mistral API** (`mistral-small-latest`) is a close second on ROUGE-L (32.5%) and
   the fastest cloud provider at 2.8s/ep.
4. **On Ollama**, **Mistral Small 3.2** and **Qwen 2.5:32b** **tie** at 38.4%
   ROUGE-L — both outperform every cloud provider except OpenAI. Mistral Small 3.2
   leads on ROUGE-1, BLEU, and embedding similarity.
5. **Qwen 3.5:9b** (with `reasoning_effort: none`) is the best smaller Qwen 3.5 model
   (~30% ROUGE-L, strong 85.2% embedding).
6. **phi3:mini** has the highest embedding similarity among Ollama models (86.9%) but
   extreme verbosity (189% coverage, 173% WER) — check output length before using.

---

## Full metrics table

All providers, sorted by category (cloud first, then Ollama), then by ROUGE-L
descending within each category.

| Run | Latency/ep | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | Embed | Coverage | WER |
| --- | ---------- | ------- | ------- | ------- | ---- | ----- | -------- | --- |
| llm_openai_smoke_v1 | 15.4s | 77.2% | 54.0% | 58.8% | 50.0% | 92.7% | 100.1% | 61.5% |
| llm_gemini_smoke_v1 | 2.7s | 61.3% | 23.4% | 33.3% | 16.3% | 87.3% | 78.1% | 83.8% |
| llm_mistral_smoke_v1 | 2.8s | 59.7% | 23.8% | 32.5% | 17.2% | 84.8% | 99.5% | 92.5% |
| llm_grok_smoke_v1 | 13.2s | 59.0% | 21.9% | 29.5% | 15.5% | 85.4% | 90.9% | 89.1% |
| llm_anthropic_smoke_v1 | 4.8s | 58.8% | 23.6% | 29.4% | 16.9% | 81.8% | 100.7% | 93.2% |
| llm_deepseek_smoke_v1 | 14.2s | 60.5% | 22.3% | 26.3% | 16.1% | 85.0% | 100.7% | 98.5% |
| llm_ollama_qwen25_32b_smoke_v1 | 54.8s | 65.6% | 30.8% | 38.4% | 22.0% | 85.2% | 78.4% | 81.3% |
| llm_ollama_mistral_small3_2_smoke_v1 | 48.6s | 69.4% | 33.0% | 38.4% | 27.8% | 85.8% | 91.4% | 83.2% |
| llm_ollama_mistral_7b_smoke_v1 | 17.4s | 61.4% | 24.8% | 32.8% | 18.9% | 80.4% | 97.0% | 92.2% |
| llm_ollama_mistral_nemo_12b_smoke_v1 | 22.8s | 59.2% | 23.4% | 30.7% | 16.0% | 82.5% | 112.9% | 102.9% |
| llm_ollama_qwen35_9b_smoke_v1 | 21.9s | 60.1% | 23.2% | 30.3% | 16.4% | 85.2% | 102.7% | 93.9% |
| llm_ollama_qwen35_27b_smoke_v1 | 81.5s | 61.2% | 24.8% | 30.2% | 17.2% | 82.4% | 128.5% | 113.1% |
| llm_ollama_qwen35_35b_smoke_v1 | 23.7s | 61.1% | 23.7% | 30.1% | 17.5% | 80.8% | 116.4% | 108.9% |
| llm_ollama_llama31_8b_smoke_v1 | 16.4s | 59.6% | 25.7% | 28.8% | 20.0% | 78.8% | 110.0% | 104.4% |
| llm_ollama_qwen25_7b_smoke_v1 | 12.1s | 63.0% | 24.1% | 28.3% | 17.7% | 84.9% | 95.6% | 95.3% |
| llm_ollama_gemma2_9b_smoke_v1 | 19.6s | 58.2% | 19.0% | 26.7% | 13.1% | 82.8% | 89.4% | 89.2% |
| llm_ollama_phi3_mini_smoke_v1 | 17.2s | 53.0% | 17.2% | 22.6% | 9.6% | 86.9% | 189.1% | 172.9% |

---

## Cloud LLMs (ranked by ROUGE-L)

| Rank | Provider | Model (eval config) | ROUGE-L | Embed | Latency | Note |
| ---- | -------- | ------------------- | ------- | ----- | ------- | ---- |
| 1 | **OpenAI** | GPT-4o | **58.8%** | **92.7%** | 15.4s | Same family as silver reference |
| 2 | **Gemini** | gemini-2.0-flash | 33.3% | 87.3% | **2.7s** | Best non-OpenAI ROUGE-L + embed |
| 3 | Mistral | mistral-small-latest | 32.5% | 84.8% | 2.8s | Very fast; strong ROUGE-L |
| 4 | Grok | grok-3-mini | 29.5% | 85.4% | 13.2s | Higher latency than Gemini/Mistral |
| 5 | Anthropic | claude-haiku-4-5 | 29.4% | 81.8% | 4.8s | Lowest embed in cloud set |
| 6 | DeepSeek | (API default) | 26.3% | 85.0% | 14.2s | Lowest ROUGE-L; solid embed |

---

## Local Ollama (ranked by ROUGE-L)

Hardware and Ollama builds vary; re-run on your machine for latency-based decisions.

| Run | ROUGE-L | Embed | Latency | Note |
| --- | ------- | ----- | ------- | ---- |
| llm_ollama_qwen25_32b_smoke_v1 | **38.4%** | 85.2% | 54.8s | Ties top ROUGE-L; `qwen2.5:32b` |
| llm_ollama_mistral_small3_2_smoke_v1 | **38.4%** | **85.8%** | 48.6s | Ties top; best ROUGE-1/BLEU/embed; `mistral-small3.2:latest` |
| llm_ollama_mistral_7b_smoke_v1 | 32.8% | 80.4% | 17.4s | Strong for 7B; `mistral:7b` |
| llm_ollama_mistral_nemo_12b_smoke_v1 | 30.7% | 82.5% | 22.8s | Good balance; `mistral-nemo:12b` |
| llm_ollama_qwen35_9b_smoke_v1 | 30.3% | 85.2% | 21.9s | Best Qwen 3.5 size/quality; use `reasoning_effort: none` |
| llm_ollama_qwen35_27b_smoke_v1 | 30.2% | 82.4% | 81.5s | Similar to 9B/35B; slowest in block |
| llm_ollama_qwen35_35b_smoke_v1 | 30.1% | 80.8% | 23.7s | Similar ROUGE-L; faster than 27B |
| llm_ollama_llama31_8b_smoke_v1 | 28.8% | 78.8% | 16.4s | `llama3.1:8b` |
| llm_ollama_qwen25_7b_smoke_v1 | 28.3% | 84.9% | 12.1s | Fastest Ollama row; `qwen2.5:7b` |
| llm_ollama_gemma2_9b_smoke_v1 | 26.7% | 82.8% | 19.6s | `gemma2:9b` |
| llm_ollama_phi3_mini_smoke_v1 | 22.6% | 86.9% | 17.2s | High embed but extreme verbosity (189% coverage) |

---

## Model IDs (acceptance-tested)

Use these model IDs in eval/acceptance configs to avoid API errors:

| Provider | Recommended model ID | Deprecated / not found |
| -------- | -------------------- | ---------------------- |
| Anthropic | `claude-haiku-4-5` | `claude-3-5-haiku-20241022` (404 deprecated) |
| Gemini | `gemini-2.0-flash` | — |
| Grok | `grok-3-mini` | `grok-2` (400 model not found) |

Eval configs: `data/eval/configs/llm_*_smoke_v1.yaml`. Acceptance configs:
`config/acceptance/full/acceptance_planet_money_*.yaml`.

---

## Observations and notes

- **Qwen 3.5 family (9b/27b/35b):** All three sizes cluster tightly around 30%
  ROUGE-L. The 9B model is the best value — similar quality at a fraction of the
  latency of 27B. Use `reasoning_effort: none` in the Ollama path for Qwen 3.5.
- **Mistral family:** Mistral Small 3.2 (local) outperforms the Mistral API
  (`mistral-small-latest`) on ROUGE-L (38.4% vs 32.5%), likely because the local model
  runs without the API's token limits and timeout constraints.
- **phi3:mini:** Despite having the highest embedding similarity (86.9%) among Ollama
  models, its 189% coverage ratio and 173% WER indicate severe verbosity. Inspect
  output length before using in production.
- **Silver reference bias:** OpenAI's 58.8% ROUGE-L and 92.7% embedding similarity
  reflect family similarity, not necessarily superior quality. For a fairer comparison,
  focus on the non-OpenAI rows.

---

## How to reproduce

```bash
# Cloud providers (requires API keys)
make experiment-run CONFIG=data/eval/configs/llm_openai_smoke_v1.yaml \
     REFERENCE=silver_gpt4o_smoke_v1

# Ollama (requires ollama serve + model pulled)
make experiment-run CONFIG=data/eval/configs/llm_ollama_qwen25_32b_smoke_v1.yaml \
     REFERENCE=silver_gpt4o_smoke_v1
```

---

## Related

- [Evaluation Reports index](index.md) — methodology and metric definitions
- [AI Provider Comparison Guide](../AI_PROVIDER_COMPARISON_GUIDE.md) — decision guide
  using these results
- [Provider Deep Dives](../PROVIDER_DEEP_DIVES.md) — per-provider reference cards with
  links to measured performance
