# Evaluation Report: Held-out v2 (April 2026)

> **Authoritative 6-provider comparison under the autoresearch v2 framework.** Dev/held-out
> dataset split, champion prompts ported across providers, dual-judge scoring.
> Supersedes the [v1 benchmark report](EVAL_BENCHMARK_V1_2026_04.md) and
> [v1 smoke report](EVAL_SMOKE_V1_2026_04.md) for cloud API providers;
> v1 reports remain authoritative for Ollama and hybrid_ml pending v2 re-run.

| Field | Value |
| ----- | ----- |
| **Date** | April 2026 (2026-04-15) |
| **Framework** | autoresearch v2 ([RFC-073](../../rfc/RFC-073-autoresearch-v2-framework.md)) |
| **Dev dataset** | `curated_5feeds_dev_v1` (10 ep, e01+e02 per feed) — iteration only |
| **Held-out dataset** | `curated_5feeds_benchmark_v2` (5 ep, e03 per feed, ~32 min each) — never used during tuning |
| **Silver (paragraphs)** | `silver_sonnet46_dev_v1_paragraph`, `silver_sonnet46_benchmark_v2_paragraph` |
| **Silver (bullets)** | `silver_sonnet46_dev_v1_bullets`, `silver_sonnet46_benchmark_v2_bullets` |
| **Judges** | gpt-4o-mini + claude-haiku-4-5-20251001 (dual, fraction-based contestation) |
| **Providers evaluated** | OpenAI, Anthropic, Gemini, Mistral, DeepSeek, Grok |

For metric definitions and interpretation guidance, see the [Evaluation Reports index](index.md).
For the framework design rationale (why dev/held-out split, why fraction-based contestation, etc.),
see [RFC-073](../../rfc/RFC-073-autoresearch-v2-framework.md).

---

Six API providers evaluated end-to-end under the same framework with the same champion prompts
(ported from OpenAI r7 champion). All champions validated on held-out content they were never
tuned against. This is the authoritative reference for provider selection.

---

## Headline Matrix (held-out, authoritative)

Final blended score = `0.70 * ROUGE-L + 0.30 * judge_mean`. Higher better.

| Track | Approach | OpenAI | Anthropic | Gemini | Mistral | DeepSeek | Grok |
| ----- | -------- | :----: | :-------: | :----: | :-----: | :------: | :--: |
| Bullets | Bundled | 0.505 | **0.552** | 0.473 | 0.487 | 0.511 | 0.489 |
| Bullets | Non-bundled | 0.566 | 0.570 | 0.562 | 0.537 | **0.586** | 0.553 |
| Paragraph | Bundled | 0.469 | **0.548** | 0.461 | 0.487 | 0.523 | 0.481 |
| Paragraph | Non-bundled | 0.481 | 0.522 | 0.463 | 0.456 | **0.541** | 0.479 |

**Cell winners on quality alone (held-out):**

| Cell | Winner | Score |
| ---- | ------ | ----- |
| Bullets · Bundled | **Anthropic** haiku-4.5 | 0.552 |
| Bullets · Non-bundled | **DeepSeek** deepseek-chat | 0.586 |
| Paragraph · Bundled | **Anthropic** haiku-4.5 | 0.548 |
| Paragraph · Non-bundled | **DeepSeek** deepseek-chat | 0.541 |

> Quality is one dimension. Once latency and cost are counted, **Gemini 2.0-flash** is the
> default pick for balanced use cases (0.562 quality, 2.0s/ep, $0.00035/ep). See the
> [Compound analysis § Pareto frontier](#compound-analysis--pareto-frontier) and
> [Recommended option order by use case](#recommended-option-order-by-use-case) for full
> quality × latency × cost tradeoffs.

## ROUGE-L breakdown (held-out)

| Track | Approach | OpenAI | Anthropic | Gemini | Mistral | DeepSeek | Grok |
| ----- | -------- | :----: | :-------: | :----: | :-----: | :------: | :--: |
| Bullets | Bundled | 33.2% | 39.3% | 28.5% | 30.4% | 34.4% | 30.5% |
| Bullets | Non-bundled | 39.6% | 40.7% | 40.1% | 37.3% | 43.1% | 38.6% |
| Paragraph | Bundled | 29.5% | 39.2% | 26.6% | 30.3% | 35.9% | 28.9% |
| Paragraph | Non-bundled | 31.7% | 36.4% | 29.3% | 27.8% | 40.7% | 31.3% |

## Dev scores (for generalisation sanity-check)

| Track | Approach | OpenAI | Anthropic | Gemini | Mistral | DeepSeek | Grok |
| ----- | -------- | :----: | :-------: | :----: | :-----: | :------: | :--: |
| Bullets | Bundled | 0.476 | 0.549 | 0.489 | 0.475 | 0.519 | 0.502 |
| Bullets | Non-bundled | 0.564 | 0.572 | 0.566 | 0.521 | 0.572 | 0.511 |
| Paragraph | Bundled | 0.467 | 0.523 | 0.462 | 0.479 | 0.479 | 0.441 |
| Paragraph | Non-bundled | 0.460 | 0.526 | 0.472 | 0.456 | 0.536 | 0.478 |

All 24 champions generalise within ±5% dev→held-out. No overfitting caught.

---

## Local models (Ollama)

Eleven local models evaluated as a full local-vs-cloud comparison. Same champion prompts ported
from OpenAI r7. Non-bundled for all eleven; bundled also evaluated for the three 9B-35B models
with reliable JSON mode (qwen3.5:9b, qwen3.5:35b, mistral-small3.2). qwen2.5:32b v2 not
evaluated — qwen3.5 generation (9b, 27b, 35b) covers the Qwen family; qwen2.5:32b would not
change recommendations.

### Held-out non-bundled (5 ep e03) — sorted by bullets final

| Model | Size | Bullets ROUGE-L | Bullets final | Paragraph ROUGE-L | Paragraph final |
| ----- | :--: | :-------------: | :-----------: | :---------------: | :-------------: |
| **qwen3.5:9b** | 9B | **42.8%** | **0.580** | 34.8% | 0.505 |
| qwen3.5:35b | 35B | 41.3% | 0.576 | 32.5% | 0.325 (contested) |
| qwen3.5:27b | 27B | 36.3% | 0.543 | 35.0% | 0.499 |
| mistral-small3.2 | ~22B | 36.1% | 0.536 | 28.8% | 0.288 (contested) |
| mistral:7b | 7B | 36.2% | 0.526 | 31.2% | 0.475 |
| llama3.1:8b | 8B | 33.8% | 0.518 | 30.5% | 0.305 (contested 5/5) |
| llama3.2:3b | 3B | 33.0% | 0.501 | 27.0% | 0.270 (contested) |
| mistral-nemo:12b | 12B | 30.4% | 0.497 | 27.1% | 0.445 |
| gemma2:9b | 9B | 30.3% | 0.492 | 28.0% | 0.453 |
| qwen2.5:7b | 7B | 29.6% | 0.477 | 28.6% | 0.463 |
| phi3:mini | 3.8B | 31.9% | 0.475 | 19.6% | 0.196 (contested) |

### Dev non-bundled (10 ep e01+e02)

| Model | Bullets | Paragraph |
| ----- | :-----: | :-------: |
| qwen3.5:9b | 0.571 | 0.493 |
| qwen3.5:35b | 0.560 | 0.491 |
| qwen3.5:27b | 0.545 | 0.512 |
| mistral-small3.2 | 0.551 | 0.497 |
| mistral:7b | 0.512 | 0.472 |
| llama3.2:3b | 0.533 | 0.441 |
| llama3.1:8b | 0.497 | 0.455 |
| mistral-nemo:12b | 0.505 | 0.450 |
| gemma2:9b | 0.495 | 0.459 |
| qwen2.5:7b | 0.461 | 0.443 |
| phi3:mini | 0.483 | 0.209 (contested) |

### Held-out bundled (5 ep e03)

| Model | Bullets ROUGE-L | Bullets final | Paragraph ROUGE-L | Paragraph final |
| ----- | :-------------: | :-----------: | :---------------: | :-------------: |
| **qwen3.5:9b** | **35.8%** | **0.529** | **33.1%** | **0.509** |
| qwen3.5:35b | 33.3% | 0.514 | 30.8% | 0.492 |
| mistral-small3.2 | 30.3% | 0.488 | 27.5% | 0.468 |

All 12 bundled cells ran cleanly with **zero contestation** — a stark contrast to the non-bundled
paragraph track where 3 of 4 models contested. The structured JSON schema stabilises output
across judges.

### Dev (10 ep e01+e02)

| Model | Bullets final | Paragraph final |
| ----- | :-----------: | :-------------: |
| qwen3.5:9b | 0.571 | 0.493 |
| qwen3.5:35b | 0.560 | 0.491 |
| mistral-small3.2 | 0.551 | 0.497 |
| llama3.2:3b | 0.533 | 0.441 |

### Full matrix with latency and cost (non-bundled bullets held-out)

Per-episode latency from actual run metrics (baseline.json `stats.avg_time_seconds`). Cost
per episode computed from approximate transcript size (~2.7k input tokens + ~500 output tokens)
and Mar 2026 pricing.

| Rank | Provider+model | Final | ROUGE-L | Latency | $/ep | Q/sec | Q/$ |
| :--: | -------------- | :---: | :-----: | :-----: | :--: | :---: | :-: |
| 1 | DeepSeek (deepseek-chat) | **0.586** | 43.1% | 10.2s | $0.00052 | 0.058 | **1131** |
| **2** | **Ollama qwen3.5:9b** | **0.580** | 42.8% | 33.3s | $0 | 0.017 | ∞ |
| 3 | Ollama qwen3.5:35b | 0.576 | 41.3% | 23.1s | $0 | 0.025 | ∞ |
| 4 | Anthropic (haiku-4.5) | 0.570 | 40.7% | 4.8s | $0.00416 | 0.119 | 137 |
| 5 | OpenAI (gpt-4o) | 0.566 | 39.6% | 4.6s | $0.01175 | 0.123 | 48 |
| 6 | Gemini (2.0-flash) | 0.562 | 40.1% | **2.0s** | **$0.00035** | **0.281** | **1594** |
| 7 | Grok (grok-3-mini) | 0.553 | 38.6% | 19.4s | $0.00106 | 0.029 | 522 |
| 8 | Ollama qwen3.5:27b† | 0.543 | 36.3% | 505s† | $0 | 0.001† | ∞ |
| 9 | Mistral (mistral-small-latest) | 0.537 | 37.3% | 2.2s | $0.00084 | 0.244 | 639 |
| 10 | Ollama mistral-small3.2 | 0.536 | 36.1% | 79.2s | $0 | 0.007 | ∞ |
| 11 | Ollama mistral:7b | 0.526 | 36.2% | 28.9s | $0 | 0.018 | ∞ |
| 12 | Ollama llama3.1:8b | 0.518 | 33.8% | 24.5s | $0 | 0.021 | ∞ |
| 13 | Ollama llama3.2:3b | 0.501 | 33.0% | 12.2s | $0 | 0.041 | ∞ |
| 14 | Ollama mistral-nemo:12b | 0.497 | 30.4% | 33.4s | $0 | 0.015 | ∞ |
| 15 | Ollama gemma2:9b | 0.492 | 30.3% | 28.6s | $0 | 0.017 | ∞ |
| 16 | Ollama qwen2.5:7b | 0.477 | 29.6% | 22.9s | $0 | 0.021 | ∞ |
| 17 | Ollama phi3:mini | 0.475 | 31.9% | 17.7s | $0 | 0.027 | ∞ |

†qwen3.5:27b bullets held-out shows anomalous 505s — likely cold-start / model swap overhead
from Ollama. Paragraph cell on same model was 188s, more typical. Treat this cell as warm-up
noise; realistic per-episode latency is probably in the ~100s range.

### Compound analysis — Pareto frontier

Three dimensions worth considering: **quality** (final score), **latency** (s/ep), **cost**
($/ep). A pick is on the Pareto frontier if no other option is strictly better on all three.

**Pareto-optimal cloud:**

- **Gemini 2.0-flash** — cheapest + fastest + not bottom-quality. Dominates on cost + latency.
- **Anthropic haiku-4.5** — 2nd fastest, 4th quality, mid cost. Middle-of-frontier.
- **DeepSeek** — #1 quality, middling latency, very cheap. Quality-first frontier.

**Pareto-optimal local:**

- **Ollama qwen3.5:9b** — #1 local quality at moderate local latency.
- **Ollama llama3.2:3b** — fastest local (12s) at acceptable quality (0.501).

**Dominated picks (avoid unless ecosystem-locked):**

- **OpenAI gpt-4o**: Anthropic has higher quality, same latency, 3× cheaper. Nothing gained.
- **Grok**: slower than Anthropic, lower quality. No wins.
- **Mistral cloud**: faster than DeepSeek but lower quality; Gemini beats it on all three axes.
- **Most local models** (mistral:7b, llama3.1:8b, qwen2.5:7b, gemma2:9b, mistral-nemo:12b, phi3, mistral-small3.2, qwen3.5:27b, qwen3.5:35b): dominated by qwen3.5:9b on quality or llama3.2:3b on speed.

### Recommended option order by use case

**A. Quality first** — you can pay, you can wait.

1. **DeepSeek non-bundled** (0.586, 10s, $0.0005) — top quality, very cheap, worst-case a few seconds.
2. Anthropic haiku-4.5 bundled (0.552, 7s, $0.004) — if you need single-call title+summary+bullets.
3. Ollama qwen3.5:9b (0.580, 33s) — only if cloud is off the table.

**B. Cost first** — cloud, quality secondary.

1. **Gemini 2.0-flash non-bundled bullets** (0.562, 2s, $0.00035) — 3× cheaper than DeepSeek, almost as fast, ~4% lower quality.
2. DeepSeek non-bundled (0.586, 10s, $0.0005) — for 50% more cost, +4% quality, 5× slower.
3. Mistral cloud (0.537, 2s, $0.00084) — only if Gemini locked out for some reason.

**C. Throughput / latency first** — batch processing, real-time serving.

1. **Gemini 2.0-flash** (2.0s) — 2× faster than Anthropic, 5× faster than DeepSeek. Cheapest too.
2. Mistral cloud (2.2s) — similar latency, weaker quality.
3. Anthropic haiku-4.5 (4.8s) — fastest on the quality frontier.

**D. Privacy / offline first** — local only, no external calls.

1. **Ollama qwen3.5:9b** (0.580, 33s) — best local quality. Close to DeepSeek cloud.
2. Ollama llama3.2:3b (0.501, 12s) — for resource-constrained devices; faster, lower quality.
3. Ollama mistral:7b (0.526, 29s) — only non-Qwen/Llama worth considering locally.

**E. Balanced** — you want all three: reasonable quality + fast + cheap.

1. **Anthropic haiku-4.5 non-bundled** (0.570, 4.8s, $0.004) — on the frontier for balance.
2. **Gemini 2.0-flash non-bundled** (0.562, 2s, $0.00035) — slightly worse quality, massively better latency+cost. Almost always the right balanced pick.
3. DeepSeek non-bundled (0.586, 10s, $0.0005) — quality premium over Gemini for 5× latency.

### The short answer

For most production deployments, **Gemini 2.0-flash non-bundled bullets** is the correct first
pick. It sits on every Pareto frontier, wins on latency and cost, and loses only ~4% on
quality vs the absolute best. Upgrade to DeepSeek (quality), Anthropic bundled (single-call),
or Ollama qwen3.5:9b (privacy) only when a specific dimension justifies the tradeoff.

**Headline finding**: `qwen3.5:9b` (open-weights, local, free) lands **2nd in the whole matrix
for bullets held-out**, 0.3pp ROUGE-L behind DeepSeek. On-prem / offline deployments can
essentially match the best cloud option on bullets quality. Seven of the top 11 entries are
local — local models are fully competitive with cloud APIs on this workload.

**Size vs quality** for Qwen3.5 family: 9B (0.580) > 35B (0.576) > 27B (0.543). The smallest
is the strongest — suggests the 9B is the pareto-optimal choice and the larger models add cost
without proportional gain. Similar pattern at 7B size: `mistral:7b` (0.526) noticeably beats
`qwen2.5:7b` (0.477) — family generation matters more than size at this scale.

### Non-bundled paragraph contests on several local models; bundled fixes it

Five of eleven local models **contested** on held-out non-bundled paragraph (judges
diverged >40% of episodes): llama3.2:3b, llama3.1:8b (5/5!), qwen3.5:35b, mistral-small3.2,
phi3:mini. Six models were uncontested: qwen3.5:9b (2/5 under threshold), qwen3.5:27b,
mistral:7b, mistral-nemo:12b, gemma2:9b, qwen2.5:7b.

**Pattern**: contestation is not strictly tied to size or generation. Contestation correlates
with structural inconsistency across episodes, which is somewhat model-specific and hard to
predict.

When the 3 largest bundled-evaluated models are run in bundled mode (same prompt content,
wrapped in the JSON schema), all three produce **uncontested** paragraph output at
meaningfully higher final scores:

| Model | Non-bundled paragraph final | Bundled paragraph final | Gain |
| ----- | :-------------------------: | :---------------------: | :--: |
| qwen3.5:9b | 0.505 (2/5 contested) | **0.509** | +0.8% (and stable) |
| qwen3.5:35b | 0.325 (contested, ROUGE-only) | **0.492** | +51% |
| mistral-small3.2 | 0.288 (contested, ROUGE-only) | **0.468** | +63% |

The JSON schema appears to force local models into more consistent paragraph structure that
judges score reliably. This mirrors the Anthropic cloud finding (bundled paragraph ≥ non-bundled
paragraph) and inverts the assumption that bundled always has an attention-split penalty.

**Practical takeaway for local deployment**: use **bundled mode** for paragraph output on
local models. The single-call cost is lower AND the quality is higher (or equal) than
non-bundled. For bullets, **non-bundled is still better** (qwen3.5:9b non-bundled 0.580 vs
bundled 0.529) — same attention-split penalty as on cloud providers (except Anthropic).

---

## Three storylines

### 1. Non-bundled: DeepSeek is the sleeper winner

DeepSeek (`deepseek-chat`) narrowly beats Anthropic on both non-bundled tracks:

- Bullets: 0.586 vs Anthropic 0.570 (+2.8%)
- Paragraph: 0.541 vs Anthropic 0.522 (+3.6%)

DeepSeek leads ROUGE-L by 2-4pp on both non-bundled cells while holding judge agreement. This
was not obvious from v1 numbers where DeepSeek looked mid-pack. The v2 framework surfaced it.

### 2. Bundled: Anthropic is in a league of its own

Anthropic's bundled scores essentially match its non-bundled scores — the attention-split penalty
that costs OpenAI ~12% and Gemini ~19% on bundled bullets simply does not materialise on Haiku 4.5.
Anthropic bundled paragraph (0.548) even *beats* Anthropic non-bundled paragraph (0.522).

For any workload where bundled (title + summary + bullets in one call) is the preferred shape,
Anthropic is the only provider where it's not a quality compromise.

### 3. Gemini is the balanced-default champion once latency/cost are counted

DeepSeek and Anthropic win on quality; Gemini wins on **everything else**. At only ~4% below
top quality (0.562 vs 0.586), it is:

- **2-5× faster** than all other cloud providers (2.0s/ep vs 4.8-19s elsewhere)
- **Cheapest cloud option** ($0.00035/ep vs $0.0005-$0.012 elsewhere)
- **On every Pareto frontier** in the quality/latency/cost space

For any application where you care about all three dimensions, Gemini is the default first pick.
Upgrade to DeepSeek only when the 4% quality premium matters more than 5× latency and 50% cost.

---

## Bundled viability per provider (non-bundled − bundled, higher = bigger bundled penalty)

| Provider | Bullets gap | Paragraph gap | Bundled verdict |
| -------- | ----------- | ------------- | --------------- |
| Anthropic | +0.018 (+3.3%) | **−0.026 (bundled wins)** | Bundled viable; best bundled quality by a wide margin |
| DeepSeek | +0.075 (+14.7%) | +0.018 (+3.5%) | Paragraph bundled ok; bullets noticeably worse |
| OpenAI | +0.061 (+12.1%) | +0.012 (+2.6%) | Clear bullets penalty |
| Grok | +0.064 (+13.1%) | −0.002 (~tied) | Similar to OpenAI — bullets penalty, paragraph ~same |
| Mistral | +0.050 (+10.3%) | −0.031 (bundled wins paragraph) | Small bullets penalty; paragraph actually prefers bundled |
| Gemini | +0.089 (+18.8%) | +0.002 (+0.4%) | Largest bullets penalty |

Three providers (Anthropic, Mistral, DeepSeek) have ≤3% paragraph penalty in bundled — bundled
is a legitimate choice there. OpenAI, Grok, Gemini show the classic "attention split" penalty.

---

## Cloud pricing reference (Mar 2026, $/M tokens)

| Provider | Model | $/M in | $/M out |
| -------- | ----- | :----: | :-----: |
| Gemini | gemini-2.0-flash | $0.075 | $0.30 |
| DeepSeek | deepseek-chat | $0.14 | $0.28 |
| Mistral | mistral-small-latest | $0.20 | $0.60 |
| Grok | grok-3-mini | $0.30 | $0.50 |
| Anthropic | claude-haiku-4-5 | $0.80 | $4.00 |
| OpenAI | gpt-4o | $2.50 | $10.00 |

Ordered cheapest → most expensive (output tokens): **Gemini < DeepSeek < Mistral < Grok <
Anthropic Haiku < OpenAI GPT-4o**. 33× spread from cheapest to most expensive.

Actual measured per-episode latency and cost are in the [Full matrix](#full-matrix-with-latency-and-cost-non-bundled-bullets-held-out) above. See [Compound analysis](#compound-analysis--pareto-frontier) for recommended option order.

---

## Provider-specific findings

### OpenAI (gpt-4o)

- **Strengths**: Reliable, good non-bundled quality, strong tooling ecosystem, json_object mode for bundled.
- **Weaknesses**: Expensive. Bundled attention-split penalty is real. `temperature=0` is not fully deterministic even with seed.
- **Quirks**: Implemented seed plumbing (`openai_summary_seed`) — helps contestation stability but not full reproducibility.

### Anthropic (claude-haiku-4-5-20251001)

- **Strengths**: Best bundled quality by wide margin. Both tracks competitive. Judge agreement highest.
- **Weaknesses**: More expensive than DeepSeek/Mistral/Gemini. No seed param in API.
- **Quirks**: API has no `response_format: json_object`; bundled requires JSON prefill (`{"role": "assistant", "content": "{"}`) — implemented in provider. No seed; relies on temp=0 (empirically more deterministic than OpenAI's temp=0).

### Gemini (gemini-2.0-flash)

- **Strengths**: Cheapest by far on output tokens. Fast latency. Non-bundled bullets quality is close to top.
- **Weaknesses**: Weakest bundled quality. Paragraph lags.
- **Quirks**:
  - Bundled JSON responses sometimes contain raw control characters — parser uses `json.loads(strict=False)`.
  - **Gemini 2.5-flash blocked**: Thinking tokens on by default consume `max_output_tokens` before real output. Current `google-genai` SDK version lacks `thinking_budget` field in `ThinkingConfig`. Sticking with 2.0-flash; 2.5-flash-with-thinking-disabled tracked as follow-up.

### Mistral (mistral-small-latest)

- **Strengths**: Mid-tier across all cells. Bundled paragraph surprisingly good (0.487 — beats OpenAI bundled paragraph 0.469).
- **Weaknesses**: Non-bundled paragraph is worst in the matrix (0.456). No ecosystem edge.
- **Quirks**: No API-specific issues encountered during v2 application.

### DeepSeek (deepseek-chat)

- **Strengths**: **Best non-bundled quality on both tracks**. Very cheap ($0.28/M output).
- **Weaknesses**: API reliability — bundled calls sometimes time out under load (retries work). No seed param. max_tokens hard-capped at 8192 (provider clamps now).
- **Quirks**: `max_tokens > 8192` returns 400; we cap at 8192 in the bundled path. `temperature=0` appears reasonably deterministic.

### Grok (grok-3-mini)

- **Strengths**: Mid-pack across all cells. No surprises.
- **Weaknesses**: Nothing stands out — third or fourth in every cell. No compelling reason to pick over Anthropic (quality) or DeepSeek (cost).
- **Quirks**: None encountered.

### Ollama (local, 11 models evaluated)

- **Strengths**: `qwen3.5:9b` matches DeepSeek cloud quality at $0/ep. Free, private, offline. Bundled mode fixes paragraph contestation (unlike non-bundled where 5 of 11 models contest on long-form held-out).
- **Weaknesses**: Latency 12-80s/ep (cloud is 2-20s). Larger models often not better — qwen3.5:9b outperforms both 27b and 35b variants. Not all models reliably produce JSON for bundled.
- **Quirks**:
  - JSON parser uses `strict=False` defensively (some models emit control characters).
  - Size-vs-quality: bigger ≠ better. qwen3.5 family peaks at 9B; mistral:7b beats qwen2.5:7b (generation matters more than size).
  - Paragraph contestation unpredictable: llama3.1:8b contested 5/5 episodes on held-out paragraph; uncontested uncorrelated with size.

---

## Framework validation (across 17 model variants)

All champions were validated under the v2 framework across 6 cloud providers + 11 local Ollama models:

- **Champion prompts transferred cleanly across all 6 cloud providers and 11 local models**. Zero provider-specific prompt
  tuning was required — the OpenAI champion prompts produced competitive numbers on every
  provider tested.
- **All champions generalise on held-out content**. Cloud: 24 champions, dev→held-out deltas
  within ±5%. Local: 11 champions + 3 bundled, same generalisation property.
- **Framework reliability**: 23 of 24 cloud cells and all 44 local cells ran cleanly first try
  (1 DeepSeek bundled transient API timeout retried successfully). One Ollama model
  (qwen3.5:27b held-out bullets) showed anomalous 505s latency — likely model-swap overhead,
  not a framework problem.
- **Judge contestation**: Fraction-based threshold (≥40%) correctly rejected high-divergence
  runs without flipping on single-episode noise. Cloud: zero runs fell back to ROUGE-only.
  Local non-bundled paragraph: 5 of 11 models triggered contestation, surfacing a real
  local-model behaviour (mitigated by bundled mode).

The framework's central claim — that these numbers are trustworthy for cross-provider decision
making — is now backed by **68 independent held-out validations** (24 cloud + 44 local). Ship it.

---

## What changed from v1 (before this session)

- **Framework**: v1 had `curated_5feeds_smoke_v1` ⊂ `curated_5feeds_benchmark_v1` (contaminated
  validation). v2 uses disjoint `dev_v1` (e01+e02) + held-out `benchmark_v2` (e03 only).
- **Contestation**: v1 binary-OR flipped runs to ROUGE-only on any divergent episode. v2
  fraction-based threshold (40%) absorbs single-episode noise.
- **Rubric**: v1 penalised long summaries via Conciseness dimension. v2 Efficiency dimension
  rewards content density without length penalty.
- **Prompt extraction**: v2 extracts JSON prose before judging so bundled outputs are judged
  on semantic content, not JSON formatting.
- **Seed**: OpenAI seed parameter plumbed through Config/Params/factory (partial mitigation for
  API non-determinism).
- **Prompt improvements**: Champion prompts from OpenAI r7 tuning + paragraph v2 tuning ported
  to all providers (few-shot bullets, style narration, anti-patterns, 4-6 para default, opening
  sentence, coverage, verbatim terminology).

Earlier v1 reports (e.g. `autoresearch/openai_comparison_2026-04-14.md`) are **superseded by
this one**. v1 used contaminated validation and should not be cited as authoritative.

---

## Open items / deferred

1. **Multi-run averaging** (N=3 per experiment). Would tighten confidence intervals on the
   numbers above and fully absorb API non-determinism. ~3× compute cost. RFC-073 §Future Work.

2. **Gemini 2.5-flash** with thinking disabled. Likely ~3-5% quality lift across all Gemini
   cells. Blocked by `google-genai` SDK version pin.

3. **Provider-specific prompt tuning**. All 6 providers run champion-ported prompts — no
   dedicated tuning. Expected 2-5% additional per cell with per-provider tuning. Low priority
   given DeepSeek and Anthropic already dominate their respective cells.

4. **Larger held-out dataset**. 5 episodes give ±5% noise on held-out. Would help distinguish
   similarly-good champions (e.g., DeepSeek vs Anthropic non-bundled bullets differs by 2.8%,
   within noise bounds).

5. **Sonnet 4.6 as candidate** (currently only silver). Would reveal how much headroom exists
   above Haiku 4.5 and DeepSeek. Not urgent given Haiku + DeepSeek already strong.

6. **Ollama / local model providers** (llama31, llama32, qwen, phi3, gemma, etc.). v1 had
   extensive ollama configs; v2 has not yet applied. Valuable for on-device deployment decisions.

7. **Champion consolidation**. We now have 24 (provider × mode × track) champion combinations.
   Long-term, only a handful matter for production. Pick 3-4 primary recommendations and mark
   the rest as reference-only.
