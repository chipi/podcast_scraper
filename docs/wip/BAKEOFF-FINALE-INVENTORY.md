# Bake-off finale — work inventory + iterate-vs-proceed decision (living doc)

Started 2026-08-05, mid deepseek 100-ep run. Accumulates everything the 100-ep finale
surfaces to **fix / improve / auto-research**, tagged by scope so we know what must happen
**once, before qwen** (provider-agnostic) vs what only moves one model's number.

Status: **deepseek 100-ep in progress** (~45/105 at first write). Updated as feeds land.

## The methodological guardrail (read first)
**qwen must be judged on the SAME pipeline as the deepseek number we compare it against.**
If we fix the pipeline between the deepseek run and the qwen run, the head-to-head is
confounded — we'd be comparing models AND pipeline versions at once. So provider-agnostic
fixes are applied **once**, and whichever models we compare are (re-)run on that same fixed
pipeline. This directly shapes the iterate-vs-proceed decision below.

## Inventory

Legend — Scope: **PA** provider-agnostic · **DS** deepseek-specific · **INFRA** tooling/eval.
Kind: FIX / IMPROVE / RESEARCH.

| # | Scope | Kind | Item | Evidence (100-ep run) | Priority |
|---|---|---|---|---|---|
| PA1 | PA | FIX | **Cleaning-stage length truncation.** Transcript-cleaning output hits its token cap on long/talk-heavy episodes (`finish_reason=length`). Cleaning may be dropping tail content. | 18 guardrails in first 45 eps, **all** `stage=cleaning reason=length`; concentrated on interview feeds (O'Shaughnessy, Sarah Guo, Sam Altman eps) | HIGH |
| PA2 | PA | IMPROVE/RESEARCH | **Topics is the weak dimension.** Canonical topic-label quality is the closest-fought dim vs 2.4 and below the 9-ep bake-off. Likely an extraction-prompt/canonicalization issue, not model-specific. | topics 7.72 (bake-off 8.22); vs 2.4 only 23-5 with 11 ties; nvidia-ai topics 6.6 | HIGH |
| PA3 | INFRA | FIX | **LiteLLM cost observability.** Gateway reports `estimated_cost_usd: 0.0` in local cost events; real cost only in LiteLLM SpendLogs. Wire per-call cost back so $/ep is visible per-run without SpendLogs reconciliation. | rolling cost had to be token×price estimated ($0.0037/ep) | MED |
| PA4 | INFRA | IMPROVE | **Productionize the rolling eval harness.** The per-feed+cumulative blind-A/B judge (top-12, cached) is currently a scratchpad script; make it a committed, reusable eval tool (all episodes, not hardcoded 0001). | built this session for the 100-ep run | MED |
| PA5 | INFRA | FIX | **Finale-corpus build tooling.** Building the v2.5 copy from relabel-fixed by hand is error-prone — two bugs bit this session (cp trailing-slash flattening; `--profile path` vs name silently ignored → summarization disabled). Add a make target / guarded script. | both bugs caught only by monitoring | MED |
| DS1 | DS? | RESEARCH | **nvidia-ai feed weakness.** deepseek summary 7.2 / topics 6.6 and its only episode losses on NVIDIA's technical/product-dense feed. Reclassify to PA if qwen is also weak there. | nvidia-ai 4-1 (only feed with a deepseek loss) at n=5 | MED |
| DS2 | DS | WATCH | **Summary below bake-off.** 8.18 vs 8.44 at 39 eps — watch whether it stabilizes or is a real scale regression. | cumulative summary 8.18 | LOW |
| AR1 | PA | RESEARCH | **Topics canonicalization auto-research loop.** The weak, provider-agnostic dimension is a natural RFC-057-style prompt-tuning target (silver+judge from a disjoint vendor). | follows PA2 | LATER |
| AR2 | PA | RESEARCH | **Per-feed adaptive config.** Technical vs interview feeds want different cleaning/summary caps + insight counts; the per-feed spread suggests one global config is suboptimal. | feed spread: nvidia vs nopriors/acast | LATER |

## DECIDED PLAN (operator, 2026-08-05) — stabilize deepseek first, then head-to-head

Chosen sequence (Option B + a small-sample optimization loop):

1. **Finish deepseek 100-ep** (running) → reassess **all** opportunities into this inventory.
2. **Apply all fixes** — PA1–PA5 + anything else the full run surfaces.
3. **Optimize on a SMALL sample** — tune each fix on a few episodes (fast loop), not the full
   100 per tweak.
4. **New deepseek 100-ep iteration** → validate the fixes actually moved the numbers.
5. **Iterate 2–4** until the deepseek result is one we're happy with (stable, clean, topics up).
6. **Then qwen 100-ep head-to-head** — run on the SAME fixed pipeline as the accepted deepseek
   iteration, so models are the only variable.

This honours the methodological guardrail: the pipeline is frozen between the accepted deepseek
iteration and the qwen run, so the head-to-head compares models, not pipeline versions.

## FUTURE track (post-finale) — native direct providers vs OpenRouter serving path

Operator, 2026-08-05: **does not touch the deepseek-vs-qwen finale plan above.** Additional
validation *after* the finale picks a model — about the SERVING PATH, not the model choice.

- **NV1 · sync the existing native deepseek provider** (`providers/deepseek/deepseek_provider.py`)
  with everything done here (reasoning-off, GI/insight params, cleaning caps, PA fixes, native
  profile mirroring `bakeoff_litellm_deepseek_flash`) → run the 100-ep the same way, **direct to
  DeepSeek's API (no OpenRouter)**, judge the same way. Compare **results AND cost** vs the
  OpenRouter run.
- **NV2 · build a native qwen provider** (DashScope/Alibaba) after the qwen finale; same
  direct-vs-OpenRouter comparison.
- **Motivation (operator):** OpenRouter routes to different hosting companies at non-fixed prices;
  for a *chosen* model, go direct (fixed price, vendor reference serving). Keep OpenRouter as the
  **research/discovery** platform (breadth: any model instantly).
- **Critical caveat to validate, not assume:** "same model name" ≠ "same outputs." OpenRouter
  backends may serve a different quantization (often FP8), context window, sampling defaults, or
  checkpoint than the vendor's own API. So the native run must be **judged**, not just costed —
  quality parity is a finding, not a given. This is exactly why the operator wants the comparison.

## Rolling snapshot (updated during the run)
- 39/105 judged: summary 8.18 vs 6.49 (37-2) · insights **8.44 vs 5.87 (39-0)** · topics
  7.72 vs 7.08 (23-5, 11t) · overall 38-1 · ~$0.0037/ep (token-estimate).
