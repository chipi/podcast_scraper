# HANDOVER — LLM prompt-caching (RFC-111) + reprocessing kickoff (2026-08-07)

> Written to hand a FRESH session the full context with zero prior memory. Branch:
> `feat/naming-arc-and-corpus-prep`. **67 unpushed commits, clean tree. DO NOT PUSH** — the handover
> travels on the branch. Today's 8 commits are the top of the log (RFC-111 down to the qwen sibling).

## 0. What the next session does (operator's stated intent)

1. **Implement RFC-111** (transcript-prefix caching for the LLM stages) — the big cost lever.
   Phase 1 first (base mechanism + summary stage), gated by a live quality-parity check.
2. **Kick off reprocessing** to actually bank the caching win (the whole point: reprocessing re-runs
   the same frozen transcripts, so the cache hits hard).
3. **opp#2 — re-derive the finale ranking** (see §4): the v2.5 finale is now known to be measured
   wrong on BOTH axes, so its model ranking needs redoing once caching + the fixes are in.

Start empty-brained; everything you need is below or committed.

## 1. Session state / commits (today, top of `git log origin/main..HEAD`)

| Commit | What |
|---|---|
| `a39c6e93` | RFC-111 transcript-prefix caching + issue #1482 (design only, NOT implemented) |
| `f0aa67a9` | B6b — guard thin siblings reuse a parity-covered prompt namespace |
| `39365c11` | B4 — grounding calls emit `llm_cost` events even when unpriced |
| `475d668e` | B7 — warn at init when a reasoning model has thinking LEFT ON |
| `d629f0b7` | **value gate `_LLM_PROVIDERS` fix** (+ A3 docs, B5 attribution, B6a AST, B1/B2 gateway_spend) |
| `aba69cc4` | real gateway cost source (`gateway_spend.py`) + native-vs-OR bakeoff profiles + litellm pricing |
| `9fd7d289` | cloud_qwen → OFFICIAL Alibaba DashScope (not DeepInfra) |
| `2d4ab3f2` | native Qwen provider (ADR-144 sibling) + cloud_qwen + qwen cost pricing |

`.env` is **untracked** (gitignored) and holds real secrets — see §5.

## 2. RFC-111 — the headline feature (read `docs/rfc/RFC-111-...md`, issue #1482)

**Problem:** the cleaned transcript is re-sent to ~5 LLM stages/episode (~71K input tok/ep). Every
provider we use caches an identical **leading token prefix** at ~0.1× price — EXCEPT the current
message layout puts each stage's *system* prompt first, so the transcript (in the user prompt, after)
never caches. **Probed: 0% today.**

**Fix:** make the cleaned transcript the **leading block of the system prompt** (identical across
stages), instructions after, task in user. Empirically (this session, live-probed):

| Layout | cache hit (deepseek) |
|---|---|
| `system = TRANSCRIPT + stage-instructions`, `user = task` | **98%** |
| current: `system = instructions`, `user = TRANSCRIPT + task` | **0%** |
| leading `user=TRANSCRIPT`, then system, then user | **0%** (role reorder breaks prefix) |

**Cross-provider (transcript-first layout, live):** deepseek-native **98%**, gateway→Novita(deepseek)
**96%**, openai **99%**, qwen-DashScope **69%**, gateway→Novita(qwen) **69%**, **gemini OpenAI-compat
0%** (needs Gemini's *native* `cached_content`).

**Correctness (settled a real operator worry):** prefix caching keys on the LITERAL token prefix and
reuses the model's COMPUTE (KV state), never the output. Two different transcripts differ at token 0
→ zero shared prefix → no hit AND no cross-contamination. Same transcript + different instructions →
transcript is the shared prefix (hit), instructions diverge after it (miss). **Caching never changes
the answer, only cost/latency.**

**Implementation phases (each revertable by a `cache_transcript_prefix` flag, default on):**

1. base `_cacheable_transcript_prefix` mechanism + flag + **summary** stage; unit+mock tests; then
   the live gate on 5–9 eps. Ship only if quality holds.
2. remaining auto-cache stages (GI / quote / entail / KG).
3. Anthropic explicit `cache_control` on the transcript block.
4. Gemini native `cached_content`.

**Message construction lives in** `src/podcast_scraper/providers/openai/openai_provider.py` — every
stage builds `[{"role":"system", ...}, {"role":"user", ...}]` (grep `"role": "system"`). The
transcript is baked into the USER prompt via `render_prompt(...)`; the refactor must move it out of
the user template into the leading system position WITHOUT duplicating it. The prompt templates are
under `src/podcast_scraper/prompts/` (`openai/summarization/...`, `openai/insight_extraction/...`,
`shared/kg_graph_extraction/...`). Thin siblings (deepseek/qwen/litellm/vllm) all default to the
`openai/` prompt namespace (so a change there covers them).

**Test plan** is RFC-111 §6 — layered unit → mock → **live quality-parity GATE** (caching on vs off,
blind Opus A/B, must be within noise, else it does NOT ship) → cost proof via `gateway_spend.py`.

**Re-verify the cache probe any time** (the test-7 basis) — minimal reproduction:
```python
from openai import OpenAI
c = OpenAI(api_key=<DEEPSEEK_API_KEY>, base_url="https://api.deepseek.com")
T = ("Host: reliable software for small teams. " * 400)
def hit(msgs):
    r = c.chat.completions.create(model="deepseek-chat", messages=msgs, max_tokens=12,
        temperature=0, extra_body={"thinking": {"type": "disabled"}})
    return r.usage.prompt_tokens, getattr(r.usage, "prompt_cache_hit_tokens", 0)
# transcript-first system prefix -> 2nd call ~98% hit:
hit([{"role":"system","content": T+"\nYou are a SUMMARIZER."}, {"role":"user","content":"Summarize."}])
hit([{"role":"system","content": T+"\nYou are a QUOTE EXTRACTOR."}, {"role":"user","content":"Extract."}])
```
(cache-hit token field varies by provider: `prompt_cache_hit_tokens` / `prompt_tokens_details.cached_tokens` / `cache_read_input_tokens`.)

## 3. CRITICAL gotchas the fresh session MUST know

### 3a. The value gate was SILENTLY OFF for the whole finale (biggest landmine)
`resolve_value_gate` / `_extractor_can_judge` gate on membership in `_LLM_PROVIDERS`
(`src/podcast_scraper/providers/ml/model_registry.py`). It had `deepseek`/`vllm` but NOT `litellm`
or `qwen`. So any litellm/qwen-routed run **silently skipped the GI value gate (fail-open, no error)**.
The ENTIRE v2.5 finale ran via `litellm` → its `gi_value_gate_enabled: true` was a **no-op** → its
insights were **unfiltered**. Fixed in `d629f0b7` (added litellm+qwen). **Implication:** `cloud_openrouter`
(prod, litellm) now ACTUALLY runs the gate with an **anthropic** vendor-disjoint judge → needs
`ANTHROPIC_API_KEY` at run time and will now drop low-tier insights (different output than before).

### 3b. The finale cost numbers were WRONG (~2–3× under-counted)
The finale computed cost as token×price from the app's `llm_cost` events, but (i) there was NO
`litellm:` pricing block → every gateway call logged $0, and (ii) grounding calls didn't emit cost
events at all. Real cost comes from the **LiteLLM gateway SpendLogs**, not the app. Use
`scripts/eval/gateway_spend.py` (reads `LITELLM_MASTER_KEY` from `.env`). Verified real cost this
session (my key, gateway-authoritative):

| model | REAL $/ep | finale claimed |
|---|---|---|
| deepseek-v4-flash (OpenRouter) | **$0.0136** | $0.0047 (2.9× low) |
| qwen3.7-flash (OpenRouter) | **$0.0052** | $0.0025 (2× low) |
| native-direct deepseek (DeepSeek list rate) | $0.0254 | — |
| gemini-2.5-flash-lite (rate × tokens) | ~$0.0124 | (baseline) |

**Corrected conclusions:** native-direct is NOT cheaper (~1.9× the OR route — but that was itself an
artifact: the native arm ran the value gate and the OR arm didn't, so they did different work; on
equal work the routes are ~cost-parity). deepseek-OR ≈ Gemini on cost (not the win the finale
implied). qwen still cheapest. **All of this is why opp#2 (re-derive) is needed.**

### 3c. Reasoning-off params (live-verified per provider)
- **DeepSeek native (api.deepseek.com):** `deepseek_extra_body: {reasoning_effort: none}` OR
  `{thinking: {type: disabled}}` (documented form) disable reasoning. `enable_thinking` /
  `reasoning:{enabled:false}` are IGNORED by DeepSeek's own API (those are OpenRouter/vLLM shapes).
  A reasoning model with thinking LEFT ON starves JSON stages → empty content → episode fails (B7
  guard warns now).
- **Qwen DashScope:** `qwen_extra_body: {enable_thinking: false}` (top-level). qwen3.7-flash is the
  non-thinking flash tier.
- **OpenRouter (via litellm):** `litellm_extra_body: {reasoning: {enabled: false}}`.

### 3d. Native providers + profiles (this session)
- **DeepSeek native:** `bakeoff_deepseek_native_flash.yaml` → api.deepseek.com, `deepseek-v4-flash`,
  reasoning off. `DeepSeekProvider` is a thin `OpenAICompatibleProvider` sibling.
- **Qwen native:** `cloud_qwen.yaml` + `bakeoff_qwen_native_dashscope.yaml` → OFFICIAL Alibaba
  DashScope (`${DASHSCOPE_API_BASE}` = the workspace endpoint in `.env`), `qwen3.7-flash`. `QwenProvider`
  is a vllm-style sibling.
- **DashScope IS the official Qwen API** (not DeepInfra) — the operator was firm on this. The
  workspace endpoint (`ws-...maas.aliyuncs.com/compatible-mode/v1`) is in `.env` as `DASHSCOPE_API_BASE`,
  kept out of the committed profile via env expansion.

### 3e. Comparison / reprocess harness (for opp#2 + validation runs)
- **Frozen corpus (NEVER modify, only copy FROM):** `.test_outputs/manual/prod-v2.4-relabel-fixed`
  (9 feeds, ~10–12 eps each = 105 total).
- **Build a 1-ep/feed copy:** the repo's `scripts/eval/build_finale_corpus.sh <name> 1` has a BUG —
  `set -e` + bash numeric compare treats zero-padded index `0008`/`0009` as invalid octal and ABORTS
  mid-build (only the first 4 feeds copied). Workaround this session: a base-10-safe Python builder
  (was in scratchpad; re-create or fix the shell script). Copies must also include `feeds.spec.yaml`
  + `corpus_manifest.json`.
- **Run a stage over a copy (relabel_only re-runs naming→cleaning→summary→GI→KG in place):**
  ```
  DEEPSEEK_API_KEY=... .venv/bin/python -m podcast_scraper.cli \
    --config config/profiles/<profile>.yaml \
    --feeds-spec <corpus>/feeds.spec.yaml --output-dir <corpus> \
    --pipeline-stage relabel_only --reprocess-existing-only
  ```
  (NOT `enrich_only` — that path doesn't load the on-disk transcript; `relabel_only` intercepts in
  the transcription stage and loads it. The CLI does NOT auto-load `.env` — export keys inline.)
- **Judge:** `scripts/eval/rolling_assess.py` (blind Opus 4.8 A/B, summary/insights/topics vs a
  BASE corpus). Adapt `BASE=<other-arm>/feeds` + `NEWRUN="run_<today>"` for a head-to-head. It reads
  `AUTORESEARCH_JUDGE_ANTHROPIC_API_KEY` from `.env`.
- **Serial only** — concurrent pipeline runs OOM the mac-mini/laptop.

### 3f. Homelab / gateway access
- Gateway: `http://homelab:4001/v1`. Virtual key `LITELLM_API_KEY` (proj-podcast-bakeoff, NOT admin).
  Admin/master key `LITELLM_MASTER_KEY` in `.env` (operator pasted it; = the homepage admin password).
- SSH: `ssh homelab-claude` (user `claude` on Mac-mini.local). **docker is permission-denied for
  claude**, and the deployed litellm `.env` (under `markodragoljevic`'s home) is unreadable as claude.
  So gateway admin only via the master key against the API — which `gateway_spend.py` does.

### 3g. The 8 fixes this session (so you don't redo them)
value-gate registration (d629f0b7) · A3 reasoning-off docs · B5 grounding attribution (10 sites) ·
B6a AST base-check + B6b prompt-namespace guard · B1/B2 gateway_spend hardening · B7 thinking-on
guard · B4 grounding cost-emit. All tested + committed.

## 4. opp#2 — re-derive the finale ranking (pending)

The finale ranked models with (a) the value gate SILENTLY OFF (§3a) and (b) grounding-blind,
under-counted cost (§3b). Both are fixed now. To redo it honestly: re-run the finale arms (value gate
now runs → filtered insights; grounding cost now emits) on the frozen corpus, judge with
`rolling_assess.py`, and take REAL cost from `gateway_spend.py` SpendLogs (per-key scoped so it's not
contaminated). Best done AFTER RFC-111 caching lands (cheaper re-runs). Scope (which models, ep count)
is the operator's call.

## 5. `.env` keys available (gitignored, present on this machine)

`DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY` + `DASHSCOPE_API_BASE` (official Qwen), `LITELLM_API_KEY`
(gateway virtual), `LITELLM_MASTER_KEY` (gateway admin — real-cost queries), `OPENAI_API_KEY`,
`GEMINI_API_KEY`, `ANTHROPIC_API_KEY` (+ `AUTORESEARCH_JUDGE_ANTHROPIC_API_KEY` for the judge),
`GROK/MISTRAL/DEEPGRAM`, `HF_TOKEN`. **No DeepInfra key** (and we don't want one — DashScope is the
official Qwen path). Rotate `LITELLM_MASTER_KEY` if the branch is ever shared, though `.env` is never
committed.

## 6. Key file reference

- `docs/rfc/RFC-111-transcript-prefix-caching-llm-stages.md` — the design + test plan (issue #1482).
- `scripts/eval/gateway_spend.py` — REAL per-model cost from LiteLLM SpendLogs (the source of truth).
- `src/podcast_scraper/providers/openai/openai_provider.py` — base transport + all stage message
  construction (the RFC-111 refactor target).
- `src/podcast_scraper/providers/ml/model_registry.py` — `_LLM_PROVIDERS` / `resolve_value_gate`.
- `src/podcast_scraper/utils/provider_metrics.py` — `apply_gil_evidence_llm_call_metrics` (B4).
- `src/podcast_scraper/config.py` — `deepseek_extra_body` / `qwen_extra_body` / provider fields.
- Profiles: `cloud_qwen`, `cloud_openrouter`, `bakeoff_deepseek_native_flash`,
  `bakeoff_qwen_native_dashscope`, `bakeoff_litellm_{deepseek,qwen37}_*`.
- `config/pricing_assumptions.yaml` — qwen + litellm blocks (ESTIMATE rates, marked verify).

## 7. First actions for the fresh session

1. Read RFC-111 + this file. Confirm branch + clean tree (`git status`).
2. RFC-111 Phase 1: base `_cacheable_transcript_prefix` + flag + summary stage + unit/mock tests
   (RFC §6 tests 1–6), THEN the live gate (tests 7–9) on a few episodes. Ship phase 1 only if the
   quality-parity gate passes.
3. Then reprocessing to bank the win; then opp#2.
