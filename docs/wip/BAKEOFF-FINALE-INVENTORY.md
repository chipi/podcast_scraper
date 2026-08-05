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

## 2nd deepseek 100-ep run (all PA fixes, gateway podcast-flash-0731, isolated key) — 2026-08-05
Corpus: v2.5-deepseek-fixed-100ep. Judged 87/105 (7 full feeds + flightcast) at report time.

Scorecard (rolling judge, Opus 4.8, vs 2.4):
- summary 8.30 / insights 8.32 / topics 7.79, overall **84-3**. Cost ~**$0.0050/ep** (isolated key, via PA3).
- **PA1 chunked cleaning: WIN** — 0 cleaning-truncations across 88 eps (pre-fix 43/88).
- **Budget isolation (dedicated OpenRouter key via gateway): WIN** — 0 budget-403; run completes
  (pre-fix DIED at 88/105 on the shared-key weekly cap).
- **PA3 cost observability: WORKS** through the gateway (real $/ep, was $0).
- **PA4 rolling judge / PA5 corpus-build script: shipped**, used to run + judge this.
- **PA2 topic specificity: NULL** — topics 7.79 ≈ pre-fix 7.78. No measurable lift; the topics
  dimension needs a non-prompt lever (DGX embeddings / structural), not prompt tuning. Like the
  earlier dead-end levers #1228/#1192.
- **PA-6 (NEW) summarization_timeout=300 too low on the gateway** for the longest feeds
  (flightcast/WSJ 100k+ char episodes): the extra gateway hop pushes the summary map-reduce past
  300s → per-episode timeout → degraded summary (GI still runs; flightcast still scored 8.4, so
  graceful). Slows the tail badly. Next pass: raise summarization_timeout for the gateway path or
  optimize the reduce for very long transcripts.

## FINAL — 2nd deepseek 100-ep (all fixes, gateway/isolated-key), 2026-08-05

**Completed 105/105** (the pre-fix run died at 88 on the shared-OpenRouter budget cap). Judged vs
the 2.4 baseline (Opus 4.8, blind A/B, top-12):

| Dim | ds-fixed | 2.4 | Win |
|---|---|---|---|
| Summary | 8.31 | 6.31 | 102-3 |
| Insights | 8.33 | 5.89 | 103-2 |
| Topics | 7.81 | 7.16 | 63-13-29t |
| Overall | | | **102-3** |

- **Cost: $0.497 total = $0.0047/ep** (real OpenRouter/Novita cost, PA3-captured — cheaper than the
  earlier $0.0092 token-estimate).
- Every feed wins decisively; nvidia-ai recovered to 11-1.

### Fix outcomes (honest)
- **PA1 chunked cleaning — WIN.** 0 cleaning-truncations (was 43/88); base-level, all siblings.
- **Budget isolation — WIN.** 0 budget-403 (dedicated podcast OpenRouter route via `podcast-flash-0731`).
- **PA3 cost capture — WIN.** Real $/ep now visible; base-level.
- **PA-6 summarization timeout — WIN.** 300→600s; validated on the 3 longest eps (0 timeouts;
  summaries in 61/156/184s — the 12 main-run timeouts were transient gateway spikes, not
  deterministic). Profile knob.
- **PA2 topic specificity — NULL.** Topics 7.81 ≈ pre-fix 7.78. No measurable lift at scale.
  Recommendation: revert (dead-end like #1228/#1192); topics already beats 2.4, real gains need a
  non-prompt lever (low priority). **Pending operator's revert nod.**

### Next (operator-agreed)
Commit checkpoint → resolve the reasoning-on/off confound → **deepseek-DIRECT 100-ep** (native
provider, not OpenRouter) to compare quality / speed(tokens-sec) / cost / reliability → build the
**qwen native sibling** in parallel while it monitors. Base fixes (PA1/PA3) propagate to all
siblings automatically.

## HEAD-TO-HEAD — deepseek-v4-flash vs qwen3.7-flash (100-ep, fixed pipeline), 2026-08-05

Both via the gateway→OpenRouter, reasoning-off, ALL PA fixes, judged vs 2.4 (Opus 4.8, blind A/B,
top-12). deepseek = full 105; qwen = 89 eps judged (tail identical trend).

| Dim | deepseek | qwen3.7 | Δ (deepseek−qwen) |
|---|---|---|---|
| Summary | 8.31 | 7.81 | +0.50 |
| Insights | 8.33 | 7.94 | +0.39 |
| Topics | 7.81 | 7.40 | +0.41 |
| Overall vs 2.4 | **102-3** | 85-4 | both dominate 2.4 |
| Cost/ep | $0.0047 | **$0.0025** | qwen ~half |

**deepseek wins QUALITY on all three dimensions (+0.4–0.5 each); qwen wins COST (~half).** Both
crush 2.4. This **contradicts the 9-ep bake-off** (where qwen was the "value winner") — at 100-ep
scale on the fixed pipeline deepseek is clearly the stronger model, qwen the cheaper one.

### Recommendation: default to **deepseek-v4-flash**
At this price level the cost gap is negligible in absolute terms (~$0.23 per 100 eps; ~$2.30 per
1000), while deepseek's quality edge is consistent and material (+0.4–0.5 per dimension, 102-3 vs
85-4). deepseek's quality dominates unless the corpus scales to 10k+ eps where the per-ep cost
compounds meaningfully — then qwen3.7-flash is the value fallback (still beats 2.4 decisively).
Neither is close to Gemini/2.4 on the downside; both are safe replacements for the v2.4 Gemini LLM.

## DEPLOYMENT-TARGETED PROFILES (operator, 2026-08-05)

The winner is **deployment-dependent** — deepseek-v4-flash cannot be served on the DGX, qwen can:

- **`cloud_openrouter`** (NEW) — cloud production via the LiteLLM gateway → OpenRouter. Model =
  deepseek-v4-flash (`podcast-flash-0731`), the finale CLOUD winner (102-3 vs 2.4, $0.0047/ep). All
  PA fixes, reasoning-off, PA-6 timeout 600. Provider-agnostic name (model is a config choice).
  Registered in `profile_sets.py` (cloud tier) + `enrichment:` block added.
- **`prod_dgx_full`** (UNCHANGED) — DGX-local production via vLLM. Model = the DGX-served qwen
  (`Qwen3-30B-A3B`), the best qwen the hardware can run. Stays as-is.

Backlog: research whether a **better qwen** is now available to serve on the DGX (task filed).
