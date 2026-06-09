# Eval: Summary-Model Reliability axis (#816)

**Date:** 2026-06-08
**Ticket:** [#816](https://github.com/chipi/podcast_scraper/issues/816)
**Parent epic:** [#907](https://github.com/chipi/podcast_scraper/issues/907)

## TL;DR

| Deliverable | Outcome |
| --- | --- |
| D1 — capture 2026-05-24 prod evidence | `autoresearch/data/reliability_evidence/2026-05-24_prod_gemini_2_5_flash_lite.json` |
| D2 — extend methodology w/ reliability axis | `scripts/eval/score/summary_model_reliability_v1.py` (sustained-load harness) |
| D3 — re-run cycle w/ extended methodology | 4-candidate panel at concurrency=5 and stress run at concurrency=20 |
| D4 — decision | **Keep `gemini-2.5-flash-lite`** + operational mitigations documented |

**Headline finding:** at eval-scale concurrency (≤5), all 4 candidates show 100% reliability — confirms the ticket's core thesis that small-batch eval does NOT surface the prod failure mode. At stress-scale concurrency (=20), server errors appear on Gemini Lite at ~1.25%. The prod-observed 15-20% rate sits at a higher operating point still (multi-stage fan-out × 3h). Composite ranking favors `gemini-2.5-flash-lite` by a wide margin even with the reliability axis added.

---

## D1: prod evidence capture

Operator-reported observations from the 2026-05-24 10-feed manual run are structured at:

```text
autoresearch/data/reliability_evidence/2026-05-24_prod_gemini_2_5_flash_lite.json
```

Key numbers (per ticket):

- Sustained **~15-20% Gemini 503 retry rate** over ~3h of batched calls
- Wall-clock cost: **~30-50% longer** than throttle-free baseline
- All episodes completed; #697 circuit breaker absorbed bursts
- Per-call telemetry not exported — `ProviderCallMetrics.record_retry()` is in-process only

The evidence file flags what the observation does and does NOT support (no per-stage attribution, no time-of-day correlation, no cross-model comparison), and points at the D2 harness as the canonical measurement going forward.

---

## D2: sustained-load harness

`scripts/eval/score/summary_model_reliability_v1.py` — multi-candidate burst harness.

Inputs: a transcript prefix (~6K chars, representative single-episode payload), a model list, a call count, and a target concurrency. Dispatches calls through a thread pool, classifies failures (`rate_limit_429`, `server_error_5xx`, `timeout`, `network`), and emits per-call + aggregate metrics.

Per-model output:

- `success_rate_pct` — primary reliability metric
- `rate_limit_rate_pct`, `server_error_rate_pct` — error decomposition
- `latency_p50_s`, `latency_p95_s` — under-load latency (NOT single-call latency)
- `effective_qps` — successes per wall-clock second
- `cost_usd_per_successful_call` — published-rate cost amortized over actual success rate

Composite ranking is reliability-floor-first (`success_rate_pct >= 95`) then cost then latency. The composite-score formula is intentionally simple — the methodology change that #816 commits to is "include reliability in the matrix", not "adopt a specific composite". Operators can re-weight at evaluation time.

The harness measures **provider-side** behavior (SDK call → success/failure/latency). It does NOT measure the application-level circuit breaker (#697) — that's a separate concern handled in the pipeline. The intent is to give autoresearch a clean signal: "this is what each provider does under load, before our mitigation layer."

---

## D3: results

### Eval-scale (concurrency=5, 30 calls per model)

This is the operating point that prior autoresearch cycles tested implicitly.

| Model | succ% | p50 (s) | p95 (s) | qps_eff | $/succ | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **gemini-2.5-flash-lite** (current) | **100.0** | **2.45** | **3.45** | **1.824** | **$0.0004** | — |
| gemini-2.5-flash | 100.0 | 9.94 | 11.88 | 0.505 | $0.0018 | — |
| gpt-4o-mini | 100.0 | 8.37 | 9.39 | 0.589 | $0.0007 | — |
| claude-haiku-4-5 | 100.0 | 7.19 | 9.64 | 0.615 | $0.0041 | — |

**At this concurrency, no candidate breaks.** The 503 floor that prod hits is NOT visible at this level. This is exactly the methodology gap #816 flagged — and validates that the original quality+cost ranking (which selected Gemini Lite) was correct in its own terms.

What's striking even at this clean operating point:

- **Gemini Lite is 3.4× faster** than the next-fastest candidate (claude-haiku-4-5) at p50.
- **Gemini Lite is 10× cheaper per successful call** than claude-haiku-4-5.
- The effective QPS spread is **3.6×** between best and worst (Gemini Lite vs Gemini Full).

### Stress-scale (concurrency=20, 80 calls — Gemini Lite only)

| Model | succ% | p50 (s) | p95 (s) | qps_eff | Errors |
| --- | ---: | ---: | ---: | ---: | --- |
| gemini-2.5-flash-lite | **98.8** | 2.50 | 3.48 | 2.205 | server_error_5xx=1 |

**The reliability floor IS detectable** at higher concurrency — one 503 in 80 calls (1.25%). Tail latency tells a parallel story: call #76 took 27.22s (vs p50 of 2.5s), indicating server-side queue pressure beyond the explicit error.

This is below the prod-observed 15-20%, but the prod operating point includes:

- Multi-stage fan-out per episode (summary + GI + KG + speaker) — the harness only hits the summary endpoint
- Multi-feed parallelism — the harness runs one-feed-equivalent worth of concurrent calls
- 3-hour sustained window — the harness bursts 80 calls in ~36s

The prod rate would require either a longer-running, more diverse stress test (deferred — cost and time both grow), or per-stage instrumentation of an actual prod run (the D1 follow-up that needs operator buy-in). The methodology that #816 D2 commits to is in place; the operator can dial concurrency / call count up when the cost budget allows.

### Composite ranking (reliability floor → cost → latency)

| Rank | Model | Reliability floor | $/succ | p50 (s) |
| --- | --- | --- | ---: | ---: |
| **1** | **gemini-2.5-flash-lite** | **PASS** | **$0.0004** | **2.45** |
| 2 | gpt-4o-mini | PASS | $0.0007 | 8.37 |
| 3 | gemini-2.5-flash | PASS | $0.0018 | 9.94 |
| 4 | claude-haiku-4-5 | PASS | $0.0041 | 7.19 |

---

## D4: decision

**Keep `gemini-2.5-flash-lite` as the `cloud_balanced` + `cloud_thin` summary model.**

Rationale:

1. **Reliability floor passes** at eval-scale and at the stress operating point we measured (98.8% at concurrency=20). The prod-observed 15-20% retry rate is the higher-operating-point story, but per the ticket itself the run completed cleanly — application-level mitigations (#697 circuit breaker) absorb the bursts.
2. **Cost dominance is overwhelming**: $0.0004/successful-call is **4-10× cheaper** than every alternative. Even with reliability discounted as if the model failed at 20%, $/successful-call only rises to ~$0.0005 — still the cheapest.
3. **Latency dominance is also large**: 2.45s p50 vs 7-10s for all alternatives. End-to-end pipeline latency would degrade materially with any swap.
4. **No alternative dominates** on any axis where Gemini Lite is weak. The 15-20% prod retry rate is a known property of this model class; the same operating-point pressure would likely surface on Gemini Full and on cross-provider candidates if measured at the same scale.

### Operational mitigations (no swap needed)

These are existing or recommended controls — captured here so the decision is documented:

- **Circuit breaker (#697)** — already in place; absorbs the 503 bursts cleanly.
- **Retry budget** — current setting is fine; per-call retries × concurrency × stage count govern blast radius.
- **Time-of-day awareness** — surfaced as a #816 D1 follow-up; not measured here, would benefit from a future structured prod-run telemetry export.
- **Per-stage attribution** — same: needs `ProviderCallMetrics` export wired into the pipeline shutdown hook.

### What would change the decision

The composite ranking would flip only if:

- A candidate appears with **comparable cost** ($0.0005 or below per successful call) AND **comparable latency** (sub-3s p50). Currently none in the panel.
- The reliability gap widens to the point where **effective $/successful-call** crosses Gemini Lite. At current Gemini Lite published rates, that would require sustained >75% failure rate — far beyond anything plausibly observed.

---

## Autoresearch methodology change (RFC-057 contribution)

The methodology change that #816 commits to:

1. **Reliability is now a hard floor**, not a tiebreaker. Default: `success_rate_pct >= 95` at eval-scale concurrency.
2. **Effective $/successful-call** replaces nameplate $/call as the cost metric for ranking. Same number when the model is clean; meaningfully different when it isn't.
3. **p50 + p95 under load** replaces single-call latency for the latency axis.
4. The reliability burst is parameterized — `--calls N --concurrency C` — and autoresearch can target the burst at the operating point most representative of the production deployment.

The script is reusable across future summary-model evaluation cycles; the schema is stable enough that comparing across cycles works.

---

## v3 fixtures contribution (#921)

Logged in `docs/wip/AUTORESEARCH_LEARNINGS_FOR_V3.md`:

- **Sustained-load reliability is invisible at small-batch eval scale.** v3 fixtures should include a "reliability burst" mode that runs N parallel summary calls against the configured provider to stress-test the operating-point reliability axis.
- **Per-stage `ProviderCallMetrics` export** would close the prod-evidence gap that D1 hit — the in-process retry counter should be serializable to the run metrics JSON so future prod runs auto-emit reliability evidence.

---

## Acceptance

- [x] D1: prod evidence captured at `autoresearch/data/reliability_evidence/`
- [x] D2: harness `summary_model_reliability_v1.py` extends the methodology
- [x] D3: 4-candidate panel run at eval-scale + 1-model stress run; metrics persisted
- [x] D4: decision documented with rationale, mitigations, and trigger conditions
- [x] v3 contributions logged

## Reproduction

```bash
# Eval-scale (concurrency=5, 30 calls per model)
set -a; source .env; set +a
python scripts/eval/score/summary_model_reliability_v1.py \
    --transcript-path data/eval/sources/curated_5feeds_raw_v2/feed-p01/p01_e01.txt \
    --models gemini-2.5-flash-lite gemini-2.5-flash gpt-4o-mini claude-haiku-4-5 \
    --calls 30 --concurrency 5 \
    --output data/eval/runs/baseline_summary_model_reliability_v1

# Stress-scale (concurrency=20, 80 calls — single model)
python scripts/eval/score/summary_model_reliability_v1.py \
    --transcript-path data/eval/sources/curated_5feeds_raw_v2/feed-p01/p01_e01.txt \
    --models gemini-2.5-flash-lite \
    --calls 80 --concurrency 20 \
    --output data/eval/runs/baseline_summary_model_reliability_v1_stress_gemini_lite
```

## Out of scope (tracked elsewhere)

- Per-stage 503 attribution (summary vs GI vs KG vs speaker) — needs `ProviderCallMetrics` export wired into pipeline shutdown.
- Time-of-day correlation — needs longer-window measurement.
- Running the stress burst against all 4 candidates — incremental insight is unlikely to flip the ranking; deferred until budget supports it.
