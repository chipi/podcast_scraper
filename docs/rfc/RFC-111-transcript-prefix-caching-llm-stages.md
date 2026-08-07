# RFC-111: Transcript-Prefix Caching for LLM Stages

- **Status**: Proposed (2026-08-07)
- **Authors**: Marko Dragoljevic (chipi), Claude
- **Stakeholders**: Operator (sign-off), providers / GI / cost-observability maintainers
- **Related RFCs**: `docs/rfc/RFC-109-per-episode-observability-manifest.md` (measure-in-prod / reprocess model this most benefits)
- **Related work (this session)**: `scripts/eval/gateway_spend.py` (real cost source), the value-gate/`_LLM_PROVIDERS` fix, B4 grounding cost-emit
- **Anchors**: [#1482](https://github.com/chipi/podcast_scraper/issues/1482) (tracking issue)

> **One-line:** the pipeline re-sends the cleaned transcript to ~5 LLM stages per episode; every
> provider we use (except Gemini's OpenAI-compat endpoint) caches an identical **leading token
> prefix** at ~0.1× price, but our current message layout puts the stage-specific *system* prompt
> first, so the transcript never caches. Move the transcript to the front → ~70–99% cache hit,
> ~4–5× input-cost reduction, and a step-change for **reprocessing** (same transcript across runs).

## 1. Context / problem

The LLM stages (summary, GI insight-generation, quote-extraction, entailment, KG, and the value-gate
judge) each send the **cleaned transcript** as input. Measured on a 9-episode diverse sample this
session: **~71K input tokens/episode**, dominated by the transcript being re-sent ~5–6×. Input
tokens are the majority of LLM cost (a transcript is ~13K tokens; the completion is a few hundred).

Two structural facts make this a large, cheap win:

1. **The transcript is identical across the stages of an episode.** Cleaning runs once; every
   downstream stage consumes the same cleaned text.
2. **Reprocessing re-runs the same frozen transcripts repeatedly** — the bakeoffs, the v2.5 finale,
   the naming arc, and the RFC-109 measure-in-prod reprocess model all re-run LLM stages over a
   frozen corpus. The transcript is identical *across whole runs*, and provider prefix caches last
   hours, so re-runs within the window hit the cache too.

### What prefix caching actually is (correctness note)

The cache key is the **literal token prefix** of the request — the longest identical run of tokens
from position 0, matched byte/token-exactly. It caches the model's **internal compute (KV state) of
those leading tokens**, never the *output*. Every request still generates a fresh, correct answer
for its full input. **Two different transcripts differ at token 0 → zero shared prefix → no cache
hit and no cross-contamination.** Same transcript + different stage instructions → the transcript is
the shared prefix (hit), instructions/task diverge after it (miss). Caching changes cost/latency,
never correctness.

## 2. Empirical findings (this session — the basis for the decision)

Probe: two calls sharing a long transcript-first prefix, differing only in the trailing task; read
each provider's cache-hit token field (`prompt_cache_hit_tokens` / `prompt_tokens_details.cached_tokens`
/ `cache_read_input_tokens`).

| Provider (route) | Cache hit, transcript-FIRST layout |
| --- | --- |
| deepseek-native (api.deepseek.com) | **98%** |
| **gateway → OpenRouter → Novita (deepseek)** | **96%** |
| openai | **99%** |
| qwen-DashScope (official) | **69%** |
| **gateway → OpenRouter → Novita (qwen)** | **69%** |
| gemini (OpenAI-compat endpoint) | **0%** — needs Gemini's *native* `cached_content` |

Layout matters and is the whole crux:

| Message layout | Cache hit (deepseek) |
| --- | --- |
| **`system = TRANSCRIPT + stage-instructions`, `user = task`** | **98%** |
| current pipeline: `system = stage-instructions`, `user = TRANSCRIPT + task` | **0%** (transcript sits after the per-stage divergence) |
| leading `user = TRANSCRIPT`, then `system`, then `user = task` | **0%** (role reordering breaks the prefix) |

**Conclusion:** the transcript must be the **leading block of the system prompt**, identical across
stages, with stage instructions after it and the task in the user message.

## 3. Decision (proposed)

Restructure LLM-stage message assembly so the cleaned transcript is the leading, stage-invariant
prefix of the system prompt. Concretely, per provider family:

- **Auto-prefix-cache providers** (deepseek, qwen, litellm/gateway, openai, grok, mistral, ollama,
  vllm): the *layout alone* enables caching — no extra API fields. Build
  `system = <transcript block> + <separator> + <stage system instructions>`, `user = <task>`, with
  the transcript removed from the user template (no duplication).
- **Anthropic**: automatic prefix caching is opt-in — add explicit `cache_control:
  {type: "ephemeral"}` on the transcript block.
- **Gemini**: the OpenAI-compat endpoint does not cache (0%); use Gemini's native **`cached_content`**
  handle — create a cache for the transcript once per episode, reference it across stages.

Gate behind a `cache_transcript_prefix` config flag (default **on** for supporting providers, **off**
falls back to today's exact layout — zero behaviour change).

## 4. Consequences

**Positive**

- ~4–5× input-cost reduction on the transcript-heavy stages; the dominant, provider-independent cost
  lever the RFC-111 investigation surfaced.
- Step-change for **reprocessing** (identical transcript across runs).
- Faster (cache-hit tokens are served faster than recompute).

**Negative / risk**

- Moving the transcript in the prompt **changes the prompt** → can shift output quality. This is the
  "heart surgery" risk and is why §6 mandates a quality-parity validation before rollout.
- Anthropic and Gemini need provider-specific code paths (not just a reorder).
- Caching is **best-effort** (a cold cache, an evicted prefix, or a non-caching backend simply pays
  full price — never wrong, just not cheaper). Cost numbers are *reductions*, not guarantees.

**Neutral**

- No correctness change (cache reuses compute, not answers — §1).
- Value-gate judge is a distinct vendor (ADR/#939); its transcript is the *insights listing*, not
  the episode transcript, so it caches independently (lower value; out of scope for phase 1).

## 5. Alternatives considered

- **Keep the transcript in the user prompt, add a stable leading system block.** Rejected: the
  transcript still sits after the per-stage divergence → 0% (probed).
- **Response/answer caching.** Rejected: different stages need different answers; this is a compute
  cache, not a response cache.
- **Shrink the transcript** (truncation / retrieval-scoped windows per insight). Complementary, not
  alternative — reduces the base token count; can layer on top later.
- **Do nothing / accept the cost.** Rejected: it is the single largest cost lever, and reprocessing
  (a core workflow) pays it repeatedly.

## 6. Test plan (the surgery is safe only if this passes)

Layered — cheap-and-deterministic first, paid live-validation gated. Numbered continuously so §7 and
the tracking issue can cite a specific test; the `[tag]` marks the layer (unit / integration / live /
regression).

1. **[unit] Layout builder** — given `(transcript, system_instructions, task)` the cacheable builder
   emits `system = transcript + SEP + instructions`, `user = task`; assert the transcript is the
   leading prefix and is **not duplicated** in the user message.
2. **[unit] Flag off = legacy** — with `cache_transcript_prefix=false` the builder emits today's
   exact layout (transcript in user). Backward-compat guard.
3. **[unit] Content invariance** — the information sent (transcript + instructions + task) is
   identical to legacy, only reordered; assert no content is dropped (the model must see exactly what
   it saw before). The core quality safety net.
4. **[unit] Provider shaping** — Anthropic attaches `cache_control` to the transcript block; Gemini
   routes to `cached_content`; auto-cache providers attach nothing extra.
5. **[integration] Transcript is message[0]** — leading content for a cacheable-provider stage
   (assert against the E2E mock's recorded request), across summary + at least one grounding stage.
6. **[integration] Cache-field plumbing** — the provider reads the cache-hit token field from the
   response usage and records it into cost telemetry (savings observable via the B4 cost events).
7. **[live] Cache-hit assertion** — two stages on the same transcript via a real provider → the 2nd
   reports `cache_hit > 0`; run for deepseek, qwen, and the gateway route.
8. **[live] QUALITY-PARITY (the gate)** — full pipeline on N=5–9 diverse episodes, caching **on vs
   off**, judged blind A/B by Opus across summary/insights/topics; **must be within sampling noise**
   (same information → same output modulo temperature). If quality moves, the RFC does not ship as-is.
9. **[live] Cost measurement** — `gateway_spend.py` SpendLogs before/after on the same episodes →
   confirm the real input-cost drop, per provider.
10. **[regression] Layout guard** — assert transcript-first is used whenever `cache_transcript_prefix`
    is on; a future prompt-template edit that puts instructions first (killing the cache) fails.
11. **[regression] Cross-provider coverage** — the layout is applied for every auto-cache provider
    (parallels the ADR-144 sibling parity guards added this session).

## 7. Rollout (staged, each gated by §6)

1. **Phase 1** — base `_cacheable_transcript_prefix` mechanism + `cache_transcript_prefix` flag +
   the **summary** stage; §6.1/§6.2 green; §6.3 (7,8,9) on a few episodes. Ship only if quality holds.
2. **Phase 2** — remaining auto-cache stages (GI, quote, entail, KG).
3. **Phase 3** — Anthropic explicit `cache_control`.
4. **Phase 4** — Gemini native `cached_content`.

Each phase is independently revertable via the flag.

## 8. Open questions

- Gemini native cache TTL/handle lifecycle vs the per-episode processing model — cost of creating a
  cache handle per episode vs the saving.
- Whether the value-gate judge (distinct vendor) is worth a phase-5 prefix pass.
- Exact separator/format for the transcript block so it is byte-stable across stages (any per-stage
  interpolation before the transcript would break the prefix — the block must be assembled
  identically every time).
