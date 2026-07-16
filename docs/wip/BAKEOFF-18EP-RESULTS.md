# Multi-provider medium-tier bake-off — 18-episode prod-v3 corpus

**Status:** FINAL for the medium tier — 8 clean arms scored. `claude-sonnet-5` and `gpt-5.5` were
DROPPED (both are newest-premium reasoning models that returned empty content on bundled evidence
calls; sonnet-5 also hit the Anthropic spend cap, gpt-5.5 also ran away to ~3,500 calls/episode
before the money fuse existed). Flagships CANCELLED (operator: not paying flagship prices while the
pipeline was token-inefficient — since fixed, see "Bundled-quote fix" below).
**Dataset:** `prod_v3_adr110_18ep_v1` — 18 episodes, 2 from each of 9 shows (bugs cluster by show
format, so the corpus is spread rather than Hard Fork-heavy).
**Harness:** identical for every arm — same transcripts, pinned `gi_insight_temperature: 0.0`,
chunked extraction at 30000 chars, value gate min-tier 2, **grok-4.3 as the single pinned judge**
(vendor-disjoint from every candidate, #939). Only the summariser LLM varies, so this measures the
MODEL. Eval and production resolve to the same knobs (ADR-112).

This is a MAP of where each provider sits on the same real corpus and pipeline — not a contest to
crown a winner.

## Knowledge scorecard (medium tier)

| arm | model | surf/ep | total/ep | lost% | redund% | mean_nli* | ground% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemini | gemini-2.5-flash-lite | **22.9** | 31.3 | 27% | 23% | 0.847 | 100% |
| mistral | mistral-medium-latest | 21.3 | 30.6 | 30% | 17% | 0.860 | 100% |
| openai-mini | gpt-5.4-mini | 20.9 | 27.6 | 24% | 14% | 0.807 | 100% |
| openai-nano | gpt-5.4-nano | 17.4 | 23.1 | 25% | 11% | 0.675 | 100% |
| deepseek | deepseek-v4-flash | 17.1 | 23.2 | 26% | 8% | 0.842 | 100% |
| qwen | qwen3.5:35b (local) | 16.3 | 21.9 | 26% | 8% | 0.775 | 100% |
| anthropic-sonnet46 | claude-sonnet-4-6 | 16.1 | 23.2 | 31% | 13% | 0.690 | 100% |
| anthropic-haiku | claude-haiku-4-5 | 15.3 | 22.2 | 31% | 8% | 0.757 | 100% |
| anthropic-sonnet5 | claude-sonnet-5 | **DROPPED** | | | | | |
| gpt-5.5 | gpt-5.5 | **DROPPED** | | | | | |

**sonnet-5 was SKIPPED (operator decision).** Its first run was contaminated by the bundled-quote
bug (empty content on 10-insight batches → grounding dropped → a fake-low 9.4 surf/ep), and the
re-run with the fix crashed at episode 11 when the Anthropic account hit its spend cap. The cap has
since been raised, but the Anthropic family is already represented by sonnet-4-6 (16.1) and haiku
(15.3) — both underperforming on this task — so a clean sonnet-5 number was judged not worth the
spend. The bundled fix WAS verified on sonnet-5's re-run (0 empty-content failures) before the cap
hit, so the fix is not in question. Re-runnable in one command if wanted.

sonnet-4-6 and haiku completed cleanly and are verified not contaminated. The grok judge is xAI, not
Anthropic, so the value gate and every other arm are unaffected by the earlier Anthropic block.

**gpt-5.5 was DROPPED — and it triggered a money-guardrail fix.** Same empty-content root cause as
sonnet-5, but in the bundled *entailment* path (not quote extraction). When that call returned empty,
grounding fell back to scoring (insight, quote) pairs one at a time, and nothing bounded the loop —
it ran to **~3,500 live API calls on a single 69-insight episode over an hour** on an expensive
model, producing `0 grounded` (garbage). Fix: a hard cap of 200 per-pair entailment calls per
episode (`MAX_PER_PAIR_ENTAILMENT_FALLBACK_CALLS`); past it, remaining pairs score ungrounded and the
run logs a loud ERROR. A runaway now bounds at ~200 and shouts instead of burning silently.
`tests/unit/podcast_scraper/gi/test_grounding_call_cap.py`.

Both newest premium models (sonnet-5, gpt-5.5) are dropped: same empty-content class, both cost real
money, both judged not worth chasing when the Anthropic and OpenAI families are already represented
by cleaner, cheaper arms.

Columns:

- **surf/ep** — distinct insights a reader is actually shown (surfaceable, deduped at 0.65). The
  honest cross-model knowledge count.
- **total/ep** — distinct incl. non-surfaceable (extracted but unshowable).
- **lost%** — distinct knowledge lost because a voice could not be NAMED (unnamed-voice defect).
- **redund%** — internal restatement in the raw surfaceable output.
- **mean_nli\*** — average entailment of shipped quotes → insights. **SELF-SCORED** (each model
  grounds with itself), so it is NOT a clean cross-model quality ranking — a lenient self-grader
  scores its own evidence high. Read it as "is this arm's evidence internally consistent", not
  "is model A better than model B". Fixing this to a single pinned NLI judge is future work.
- **ground%** — insights with ≥1 grounded quote.

## What the numbers say (so far)

1. **gemini-2.5-flash-lite leads on quantity** (22.9 surfaceable distinct/ep) but is the most
   redundant (23% internal restatement — the filler the gate and dedupe exist to trim).

2. **mistral-medium is the strongest all-rounder** — 21.3 surfaceable (a hair behind gemini), with
   lower redundancy (17%). A mid-tier model matching the incumbent on knowledge.

3. **gpt-5.4-mini is solid** (20.9, 14% redundancy). gpt-5.4-nano drops to 17.4 — the size step
   costs ~3.5 insights/ep.

4. **claude-sonnet-4-6 underperforms for a premium model** — 16.1 surfaceable, down with qwen at the
   bottom of the completed set, despite being one of the most expensive models here. Premium price
   did not buy more surfaceable knowledge on this task. (haiku / sonnet-5 pending may change the
   Anthropic picture.)

5. **deepseek-v4-flash and qwen are the leanest** — fewest insights (17.1 / 16.3) but cleanest
   output (8% redundancy each). They extract less and restate less.

6. **The unnamed-voice defect costs 24–31% of distinct knowledge on EVERY model.** `lost%` barely
   moves across seven very different models because it is a property of the diarization/roster
   pipeline, not the LLM. This is the single biggest product lever and no model choice touches it.

## Cost & speed (added post-scoring)

Two axes the knowledge scorecard alone hides: **what each arm costs** and **how long it takes**.
Cost provenance is labelled per row — this matters, because token×list-price OVER-reads badly for
cache-heavy providers (deepseek logs 19M input tokens → $0.35/ep at list price, but the dashboard
truth is $0.048/ep, 7× lower, because most input is cache-hit). Trust the source column.

| model | surf/ep | time/ep | $/ep | $/distinct-item | cost source |
| --- | --- | --- | --- | --- | --- |
| gemini-2.5-flash-lite | 22.9 | 5.1 min | ~$0.10 | $0.0044 | est (low cache → trustworthy) |
| mistral-medium | 21.3 | 3.1 min | $0.15–0.54 | ~$0.016 | uncertain (12× log undercount) |
| gpt-5.4-mini | 20.9 | **1.4 min** | **$0.096** | $0.0046 | **dashboard ✓** |
| gpt-5.4-nano | 17.4 | 1.4 min | **$0.024** | **$0.0014** | **dashboard ✓** |
| deepseek-v4-flash | 17.1 | 18.6→**6.7** min¹ | $0.048 | $0.0028 | **dashboard ✓** |
| deepseek-chat (V3.2, non-reasoning) | *pending²* | **1.1 min**² | ~$0.05² | — | re-run DONE |
| qwen3.5:35b (DGX ollama) | 16.3 | **1.8 min**⁴ | **$0**⁴ | $0 | DGX GPU (electricity only) |
| claude-sonnet-4-6 → 4-5³ | 16.1 → *pending³* | 4.1 → **5.9** min³ | ~$0.76³ | $0.037 | est (no cache) |
| claude-haiku-4-5 | 15.3 | 2.6 min | ~$0.15 | $0.0098 | est |

**Re-runs complete (2026-07-15).** All three finished; timings + call counts are final, distinct-
knowledge (surf/ep) re-scoring via the grok-4.3 judge is the one remaining step.

¹ deepseek-v4-flash: original **18.6 min/ep was the pre-chunking-fix run** (the "5 hours"). The
post-fix re-run measures **6.7 min/ep** with **~40 calls/ep** (down from ~84 — the chunking fix
halved the call storm, confirmed on live data). The residual 6.7 min is inherent to v4-flash being a
**reasoning model** (~15s/call thinking), not the bug. Knowledge score unchanged (coverage identical).

² **deepseek-chat (V3.2), the NON-reasoning DeepSeek — the standout of the re-runs.** Same provider,
identical pricing, but it answers directly (no reasoning phase): **1.1 min/ep — 6× faster than
v4-flash's 6.7** — at **~20 calls/ep** (fewer than v4-flash's 40; no reasoning-budget empty-content
fallbacks) and ~$0.055/ep. It sidesteps the exact fragility v4-flash needed patched. Its distinct-
knowledge score vs v4-flash's 17.1 is the open question the judge pass answers — but on speed+cost it
already wins decisively, and arguably should have been the DeepSeek pick from the start.

³ **claude-sonnet-4-6 was the WRONG model** — the requested model was claude-sonnet-4-5, which never
actually ran (see the Anthropic mislabel note below). The 16.1 / 4.1-min row is that mislabeled arm.
The corrected **claude-sonnet-4-5** re-run is DONE: **5.9 min/ep**, and its own (pre-fix) cost
instrumentation logged **~$0.76/ep** — confirming Anthropic is the most expensive medium on the
board by a wide margin. Its distinct-knowledge surf/ep replaces the 16.1 after the judge pass.

**OpenAI arms — itemized from the usage dashboard (14 Jul, operator-pulled).** Confirms the fuse's
worth: the one un-fused model (gpt-5.5) burned more than every other arm on the board combined.

| OpenAI arm | input | cache-in | output | total | /18 ep |
| --- | --- | --- | --- | --- | --- |
| gpt-5.4-mini | $0.79 | $0.126 | $0.813 | **$1.73** | $0.096/ep |
| gpt-5.4-nano | $0.189 | $0.04 | $0.20 | **$0.43** | $0.024/ep |
| gpt-5.5 (DROPPED, runaway) | $10.8 | $3.49 | $24.20 | **$38.49** | — |

**What cost + speed change about the conclusion:**

- **gpt-5.4-mini is the standout all-rounder** once cost is visible — near-top knowledge (20.9, ~tied
  with mistral), **fastest on the board** (1.4 min/ep), and cheap ($0.096/ep, ground-truth). Best
  knowledge×speed×cost balance. The knowledge-only ranking hid this because its tokens weren't logged.
- **gpt-5.4-nano is the efficiency king** — $0.0014 per distinct item, ~2× cheaper than deepseek and
  ~26× cheaper than sonnet-4-6, still 17.4 knowledge, 1.4 min. Cheapest cloud arm, period.
- **sonnet-4-6 is confirmed worst value** — $0.037/item for *less* knowledge (16.1) than the arms a
  fraction of its price. Premium price bought neither more knowledge nor speed.
- **gemini (incumbent)** keeps the raw-knowledge crown and is cheap per item, but is slowest-but-
  deepseek and most redundant (23%).
- **The gpt-5.5 runaway ($38.49, 96% of the day's OpenAI spend)** is the canonical fuse case: one
  un-bounded model out-spent seven full arms. This is why ADR-113 exists.

## Anthropic model mislabel — the "sonnet-4-6" arm was the wrong model, and it went uncaught

The arm reported as **claude-sonnet-4-6 (16.1 surf/ep)** ran a model that was **never requested** — the
ask was **claude-sonnet-4-5**. `claude-sonnet-4-6` is not on any pricing/known-model list, and the
Anthropic dashboard shows the requested sonnet-4-5 got **~12k tokens today = it never ran** as an arm,
while sonnet-5 ran **twice**. The 2.934M tokens the "4-6" arm burned appear folded into the sonnet-5
total, so it may have been **served as sonnet-5**. Unresolved pending a dashboard line-item check;
either way the 16.1 row is NOT the sonnet-4-5 result that was asked for. Corrected sonnet-4-5 arm is
in flight (footnote ³).

**Why it went uncaught — and the fix.** Nothing validated the model id: the string flowed config →
API → 18 finished episodes → this scoreboard, and the mismatch only surfaced when the operator read the
dashboard by hand. Closed with a model-governance layer (5 guards, config-driven allowlist in
`config/known_models.yaml`): (1) unknown cloud models are **rejected at config/profile load** before any
spend, with a suggestion; (2) providers now verify the **served** model against the requested one and
log a loud `llm_model_substitution` on a mismatch; (3) served-model **provenance** on every cost event;
(4) the fictional id fails closed (the #932 `sonnet46` judge that also defaults to it is flagged, not
blindly renamed); (5) an offline **reconciliation** util replays a run's logs to catch served≠requested
drift automatically. `claude-sonnet-4-6` will now hard-stop any new arm/profile that names it.

## Provider integration bugs found & fixed (the point of going one-by-one)

Every newest-generation model broke the pipeline in a different way. All fixed, regression-tested,
and live-verified — no hacks.

| provider | model | bug | fix |
| --- | --- | --- | --- |
| DeepSeek | v4-flash | reasoning model spent the 10-token entailment budget on reasoning → empty content → 0 grounded quotes on every episode | reasoning-aware token headroom (reasoning models only); `test_deepseek_reasoning_budget.py` |
| Anthropic | sonnet-5, opus-4-8 | Claude 5 gen dropped `temperature`; request 400'd, crashed at episode 1 | self-healing `_messages_create` strips temperature for known-deprecated models + learns new ones; `test_anthropic_temperature_deprecated.py` |
| OpenAI | gpt-5.x | renamed `max_tokens` → `max_completion_tokens`; evidence calls 400'd → 0 grounded | central `_token_kwarg` at every call site; `test_openai_token_kwarg.py` |
| OpenAI | gpt-5.5 | fixed `temperature` at default (1); rejects 0 → crashed at episode 1 | self-healing `_chat_create` strips non-default temperature + learns new models |
| Anthropic | haiku-4-5 | verbose; summary exceeded 4096 tokens → truncation guardrail crashed the arm | summary budget → 8192 (does not bias insights: those come from the transcript, not the summary) |

## Two problems to fix (analyzed while the batch ran; fix AFTER this report)

### 1. Cost/token instrumentation is inconsistent — my log-based cost metric is UNRELIABLE

Empirical `llm_cost` events per provider across all runs:

| provider | cost events | reality |
| --- | --- | --- |
| gemini | 4144 | accurate |
| deepseek | 1520 | accurate (matched dashboard: ~1600 req, 18M in) |
| anthropic | 552 | accurate |
| mistral | 182 | **12× undercount** — logs summarization only, or captures wrong token counts (510K logged vs 6M dashboard) |
| openai | **0** | not instrumented on the eval path — $/ep recovered only from the operator's usage dashboard (mini $1.73, nano $0.43, gpt-5.5 runaway $38.49) |
| grok (judge) | **0** | not instrumented |
| ollama | 0 | local/free — N/A |

Consequences, stated plainly:

- My `$` estimates were wrong (stale `pricing_assumptions.yaml`, no cache-hit modeling) — **retracted**.
- My "Mistral 38× more efficient" was an artifact of Mistral's 12× under-logging — **retracted**.
  Real gap vs deepseek is ~3× tokens.
- **Provider dashboards are the only trustworthy $/token source today.** Their usage APIs need
  admin-tier keys we don't have (our keys are `sk-proj-` / `sk-ant-api03-`, missing `api.usage.read`),
  so I cannot pull spend programmatically. DeepSeek `/user/balance` is the one exception ($8.55 left).
- The only signal my logs rank reliably is **request count**.

**Fix:** wire `emit_llm_cost_event` uniformly across all 7 providers at every call site, with correct
token capture, so cost tracking is self-contained and we never need a dashboard again.

### 2. Bundled quote extraction collapsed to per-insight — FIXED

**Root cause (not what it first looked like).** It was not entailment. `extract_quotes_bundled` sent
EVERY insight plus the transcript in ONE call. The response is ~256 tokens/insight, so a 50-insight
episode overran the 8192-token cap and came back as **truncated JSON on deepseek**, timed out
**server-side on gemini (504)**, and **client-side on mistral** — and on **sonnet-5 it returned empty
content** (thinks on the big 10-insight task, exhausts the budget). The single call then raised, and
the WHOLE episode fell back to one `extract_quotes` call PER insight — ~80 calls/episode instead of
~10, the 8× blow-up behind DeepSeek's "5 hours".

**This corrupted results, not just speed.** Where BOTH the bundled call and the per-insight fallback
came back empty — **sonnet-5** — no quotes were produced, and `gi_require_grounding` dropped the
ungrounded insights, so sonnet-5's first run scored an artificially low 9.4 surf/ep. Providers whose
per-insight fallback worked (gemini, deepseek, mistral, openai) kept correct knowledge but paid the
call tax. sonnet-4-6 and haiku were verified unaffected.

**Fix (landed, tested).** Chunk `extract_quotes_bundled` (default 10 insights) at the single pipeline
call site, and **bisect any chunk that still fails** — a reasoning model whose 10-insight payload
truncates gets retried at 5, 2, 1, shrinking only where a provider needs it. Only a size-1 chunk that
still fails drops to the per-insight path. Verified on the largest (90-insight) episode:

| provider | before | after | coverage |
| --- | --- | --- | --- |
| deepseek | 90 per-insight calls | **25** chunked | 90/90 |
| gemini | 90 per-insight calls | **13** chunked | 90/90 |

Zero per-insight fallbacks. `tests/unit/podcast_scraper/gi/test_bundled_quote_chunking.py`
(5 tests, mutation-checked); `QUOTE_BUNDLE_CHUNK_SIZE` + bisection in `gi/pipeline.py`.

## Ordering (operator-locked)

1. Finish the 3 remaining medium arms → **this report** (in progress).
2. Fix cost instrumentation (Problem 1) + bundled-entailment (Problem 2).
3. ONLY THEN: Ollama/DGX local wave (free).
4. Flagships remain cancelled; revisit only if efficiency fixes change the cost math.
