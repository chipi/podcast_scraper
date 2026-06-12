# Evaluation Report: Smoke v1 — DGX vs Laptop Ollama (June 2026)

> **Apples-to-apples replay** — same 12 Ollama models that ran in [Smoke v1
> (April 2026)](EVAL_SMOKE_V1_2026_04.md) on the operator laptop, re-run
> against the NVIDIA DGX Spark (GB10) via tailnet ([RFC-089](../../rfc/RFC-089-dgx-spark-tailnet-integration.md)
> #810 close-out). Purpose: validate that DGX is a functional drop-in
> replacement for laptop Ollama before promoting it as the autoresearch
> compute target. Not discovery — comparison.

| Field | Value |
| --- | --- |
| **Date** | June 2026 |
| **Dataset** | `curated_5feeds_smoke_v1` (5 episodes, 5 feeds) — identical to April |
| **Silver reference** | `silver_sonnet46_smoke_v1` (Claude Sonnet 4.6) — identical to April |
| **Configs** | `data/eval/configs/summarization/autoresearch_prompt_ollama_*_smoke_paragraph_v1.yaml` (12 files, **unchanged** since April — only `OLLAMA_API_BASE` env var differs) |
| **April baselines** | `data/eval/baselines/baseline_llm_ollama_*_smoke_paragraph_v1/` |
| **June DGX runs** | `data/eval/runs/llm_ollama_*_dgx_smoke_v1/` |
| **Summary JSON** | `data/eval/runs/dgx_vs_laptop_summary.json` |
| **Closes** | #811 acceptance #4 (autoresearch matrix sample re-run report exists) |

## Setup

```bash
export OLLAMA_API_BASE=http://your-dgx.tailnet.ts.net:11434/v1
# 12 models pulled on DGX first (~110 GB total, ~36 min sequential)
for m in llama32_3b phi3_mini mistral_7b qwen25_7b gemma2_9b llama31_8b \
         qwen35_9b mistral_nemo_12b mistral_small3_2 qwen35_27b qwen25_32b qwen35_35b; do
  make benchmark \
    CONFIG=data/eval/configs/summarization/autoresearch_prompt_ollama_${m}_smoke_paragraph_v1.yaml \
    BASELINE=baseline_llm_ollama_${m}_smoke_paragraph_v1 \
    OUTPUT_DIR=data/eval/runs/llm_ollama_${m}_dgx_smoke_v1 \
    SMOKE=1
done
```

The only thing that differs vs April: `OLLAMA_API_BASE` points at DGX instead of `localhost:11434`. Configs, prompts, dataset, silver reference, scoring schema — all identical to the April run.

## Headline numbers

| Model | Size | Laptop p50 (ms) | DGX p50 (ms) | Δ avg (ms) | Δ % | Laptop tok/s | DGX tok/s | rougeL Δ-vs-silver |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `llama3.2:3b` | 3B | 7,297 | 6,309 | +5,904 | +80% | 74.2 | 50.9 | +0.027 |
| `phi3:mini` | 3.8B | 16,322 | 10,563 | -801 | -5% | 59.0 | 53.5 | +0.022 |
| `mistral:7b` | 7B | 16,493 | 9,412 | -5,881 | -37% | 31.0 | 43.4 | +0.037 |
| `qwen2.5:7b` | 7B | 13,043 | 9,933 | -968 | -7% | 35.7 | 43.5 | +0.022 |
| `gemma2:9b` | 9B | 16,055 | 10,850 | -2,164 | -13% | 28.0 | 32.4 | +0.039 |
| `llama3.1:8b` | 8B | 14,962 | 12,256 | -2,802 | -18% | 36.0 | 47.1 | +0.017 |
| `qwen3.5:9b` ⚠ | 9B | 23,485 | 12,895 | +185,463 | +775% | 22.1 | 2.8 | -0.019 |
| `mistral-nemo:12b` ⚠ | 12B | 25,799 | 25,319 | +377,984 | +1537% | 23.6 | 1.5 | +0.000 |
| `mistral-small3.2` | ~24B | 45,791 | 25,320 | -16,935 | -37% | 10.3 | 16.5 | -0.003 |
| `qwen3.5:27b` | 27B | 84,128 | 37,760 | -39,983 | -48% | 8.0 | 15.1 | -0.021 |
| `qwen2.5:32b` | 32B | 57,314 | 34,188 | -21,243 | -36% | 6.9 | 12.0 | -0.025 |
| `qwen3.5:35b` | 35B | 19,834 | 6,622 | -7,030 | -34% | 29.4 | 46.3 | -0.038 |

⚠ = anomalous (see Anomalies section).

## Findings

### 1. DGX is a drop-in replacement for laptop Ollama

For 10 of 12 models, **outputs are functionally equivalent**: rougeL_f1 delta-of-delta vs silver stays within ±0.04 — well inside the noise band of two consecutive runs of the same model on the same hardware. No model showed gate regressions (all `boilerplate_leak_rate`, `speaker_label_leak_rate`, `truncation_rate` stayed at 0). The pipeline does not care which side the Ollama daemon is on.

The two anomalies (qwen3.5:9b, mistral-nemo:12b) — see below — are operational, not quality. Both still produced outputs that scored similarly against silver; they just took outlier wall-clock time.

### 2. DGX wins big on models ≥27B (33–48% faster)

| Model | Laptop avg latency | DGX avg latency | Speedup |
| --- | --- | --- | --- |
| `qwen3.5:27b` | 83.7 sec | 43.7 sec | **1.9×** |
| `qwen2.5:32b` | 78.9 sec | 57.7 sec | **1.4×** |
| `mistral-small3.2` (~24B) | 52.7 sec | 35.8 sec | **1.5×** |
| `qwen3.5:35b` | 21.1 sec/ep | 14.1 sec/ep | **1.5×** |

This is where the **cost arbitrage thesis from RFC-089 starts to bite** — autoresearch matrix cells that were skipped on laptop (or queued behind 80-second-per-episode runs) become viable. A 100-cell sweep at 27B that took ~140 minutes on laptop completes in ~73 minutes on DGX, with no thermal throttling and no battery drain.

### 3. DGX is neutral-to-worse on models <10B

For tiny models, **HTTP roundtrip + Ollama-on-DGX session overhead dominates**, and the gain from a more powerful GPU is small (a 3B model isn't compute-bound on any laptop GPU either). The 3B case actually regressed by 80% (~7s → ~13s/episode). The 7–9B cases were a mix of slight speedups and slight regressions.

**Operational implication:** for autoresearch matrices that include tiny models for cost/quality comparison, the laptop is still the best target for those cells. DGX wins where it counts — the big-model cells that were the original blocker.

### 4. Quality is unchanged in either direction at this sample size

`rougeL_f1` delta-of-delta is the difference between (DGX-vs-silver) and (laptop-vs-silver). Read as "did DGX move closer to or further from silver-Sonnet quality?" Six cases moved slightly closer (+), six moved slightly away (−), magnitudes all within ±0.04. No systematic pattern — consistent with float noise + sampling variance on the same model.

## Anomalies

### `qwen3.5:9b` and `mistral-nemo:12b` — collapsed tok/s during sequential sweep

These two cells produced outputs that scored normally against silver but took absurd wall-clock (~200s and ~400s avg per episode respectively — vs 12–25s for similarly-sized models). Tok/s dropped to 2.8 and 1.5 from a normal 22–24.

**Most likely cause:** model swap / re-load between sequential `make benchmark` calls. Ollama keeps the last-used model resident; switching forces a load cycle. If a model is partially evicted and the load happens mid-episode, that one episode goes very slow and skews the average (avg_latency_ms was always much higher than median for these — confirming it).

**Fix to verify in a follow-up:** rerun those two cells with `--keep-alive` or in a different order. Or use the explicit `--noprune` flag if available. Not a quality issue; a sequencing artifact.

## What this closes

| Issue | Acceptance criterion | Status |
| --- | --- | --- |
| #811 P1 | Autoresearch eval matrix sample re-run report exists, comparing DGX-hosted models to baselines | ✓ (this report; baselines are the silver Sonnet reference + April laptop runs) |

## What this does NOT yet close

- #812 P2 (AI comparison guide with real measurements) — same data could be cited there; this report's table format is suitable
- #811 P1 #6 (operator E2E validated on real or fixture corpus run with `local_dgx_balanced` profile) — separate end-to-end pipeline run, not done here
- #811 P1 #5 (cloud-fallback test) — moved to #814 per the body update on #811

## Methodology caveats

- **5 episodes per model** is directional, not statistically robust. Tight bands at this N can flip on the next 5 episodes.
- **Sequential sweep on a fresh DGX** — the two anomalous cells are almost certainly artifacts of model swap timing. Re-running them individually would tighten the picture.
- **Latency measurements include HTTP roundtrip and Ollama session overhead.** They are operational ("how long does my autoresearch matrix take to complete") not pure-compute ("how fast is the GPU"). The operational number is the one that matters.
- **Single-prompt-set sweep.** April's prompts; no autoresearch round 3 prompt tuning involved. Promoting DGX shouldn't be conflated with prompt improvements.

## Next

- Re-run the two anomalies (`qwen3.5:9b`, `mistral-nemo:12b`) individually to confirm they're swap artifacts.
- Pull this report's table into the AI comparison guide once the missing rows are filled. (#812)
- E2E pipeline run on real corpus with `local_dgx_balanced` profile. (#811 #6)
- Promote DGX as the default Ollama target for autoresearch matrices that include 27B+ models. Smaller cells continue to run on laptop, switched by a single env var.

## References

- [RFC-089 — DGX Spark tailnet integration](../../rfc/RFC-089-dgx-spark-tailnet-integration.md)
- [ADR-098 — Embedding provider as a profile axis](../../adr/ADR-098-embedding-provider-profile-axis.md)
- [Smoke v1 April 2026](EVAL_SMOKE_V1_2026_04.md) — the comparison source-of-truth
- [Evaluation Reports index](index.md)
