# Bundled-JSON reliability — deep research (#912)

**Date**: 2026-06-18
**Branch**: `feat/autoresearch-followups-2026-06-18`
**Trigger**: #912 (qwen3.5:9b bundled JSON ~50–67% parse failures); prior session's
Path A (Ollama native `format` JSON schema) delivered 100% parse but 20× wall-clock
(~175s vs ~9s baseline); operator pushed back on shipping that.

This document is research, **not code**. It collects internal evidence,
external evidence, and lands a categorical recommendation with confidence
levels. Code changes happen in a follow-up PR after the recommendation is
accepted.

---

## CORRECTION (post-operator-review, 2026-06-18)

The original Section 5 synthesis claimed *"the 3-stage bundled envelope
(summary+KG+GI) doesn't exist in our codebase — only summary+clean fusion
does."* **This claim is wrong.** A targeted grep + read of
`src/podcast_scraper/prompting/megabundle.py` confirms the codebase has
**TWO distinct bundled modes**, both predating #912:

1. **Megabundle** (`build_megabundle_prompt`, megabundle.py:58) — one JSON
   envelope containing **six fields across three+ pipeline stages**:
   `title`, `summary` (4–6 paragraphs), `bullets` (4–6), `insights` × 12 (GI),
   `topics` × 10 (KG), `entities` × 15 (KG). Single 1-call request.

2. **Extraction bundle** (`build_extraction_bundle_prompt`, megabundle.py:129)
   — 2-call pipeline. First call uses the provider's standalone
   `summarize()`; second call bundles `insights` + `topics` + `entities` only.

**Existing routing rule** (megabundle.py docstring, citing #632 / #643
research): Anthropic Claude Haiku 4.5 + DeepSeek handle megabundle cleanly;
**OpenAI / Gemini / Mistral / Grok already use extraction_bundled** because
they over-compress the summary when given a single envelope. So "can the
provider handle a single multi-stage JSON" is **already** a routing
decision in this codebase — it's not new architecture.

### What this changes about the recommendation

The original "retire bundled mode" framing assumed bundled = monolithic.
It isn't. The actual decision for qwen3.5:9b is **which of three modes**
to default to:

| Mode | Calls | Currently tested on qwen3.5:9b? |
|---|---|---|
| Megabundle | 1 | Yes — 50–67% parse fail (the #912 bug) |
| **Extraction_bundled** | 2 | **No — never measured. Existing fallback for 4 other providers.** |
| Fully staged | 3+ | Yes — works, current workaround |

The middle row is the gap. If qwen3.5:9b can handle the *smaller*
extraction-half bundle (no summary prose, just structured insights/topics/
entities), we route it via the **existing extraction_bundled path** —
joining OpenAI/Gemini/Mistral/Grok in the same bucket. That's a config /
provider-classification change, not new architecture, and zero of the
"retire bundled code" steps in the original Section 5 are needed.

### What the upstream evidence (Sections 2-4) still tells us

The Ollama `format=json_schema` slowdown (Ollama #3851 / #3154 / #15540)
and the autoresearch quality-regression data on megabundle remain valid
**for the 1-call megabundle path**. They do not speak to
extraction_bundled (smaller payload, no summary prose). The
SqueezeBits Jan 2026 vLLM `response_format` benchmark also remains valid
as a forward-path option for the autoresearch tier.

### Open question carried into the eval

What does qwen3.5:9b's extraction_bundled parse rate look like in
isolation? Hypothesis: failures correlate with envelope size / field
count, not with bundling per se. Extraction_bundled (3 fields, all
structured) is half the field count of megabundle (6 fields, mix of
prose + structured). If the hypothesis holds, extraction_bundled parses
cleanly. If not, fully-staged becomes the qwen3.5:9b default.

This is the validation eval proposed next.

---

## Section 1 — Internal evidence (repo)

### 1.1 What #912 actually says

`gh issue view 912`, body verbatim:

> "When `qwen3.5:9b` runs the bundled clean+summary path (`llm_pipeline_mode:
> bundled` → `OllamaProvider.summarize_bundled`), it produces malformed JSON
> output ~50–67% of the time on small N. The failures are model-specific
> (27b/32b/llama8b all clean in identical conditions)."

Observed failure modes (issue body, "Symptoms"):

1. `Bundled JSON missing non-empty summary string` — JSON parsed, `summary`
   field empty, `bullets` carried the prose instead.
2. `Bundled response is not valid JSON: Expecting ',' delimiter: line 2
   col 14` — mid-bullet truncation, premature close-bracket.
3. "Multi-paragraph bullet text containing literal newlines (`\n\n`) inside
   JSON string values — invalid per JSON spec."

No reports of markdown fences or explicit refusals in #912 itself. We have
seen markdown-fenced JSON elsewhere — that's why
`src/podcast_scraper/schemas/summary_schema.py::_strip_markdown_json_fence`
exists and runs in front of `json.loads`. The newline / mid-truncation /
empty-`summary` modes are the bundled-mode-specific ones.

### 1.2 What the workaround commit actually reverted

`git show c6a8982b` — relevant chunk: `config/profiles/local_dgx_balanced.yaml`
flipped `llm_pipeline_mode: bundled → staged`. From the commit message:

> "local_dgx_balanced was using llm_pipeline_mode: bundled, which the
> 2026-06-07 smoke surfaced as ~50-67% failure rate for qwen3.5:9b
> (malformed JSON, missing fields, mid-bullet truncation). Direct
> comparison against 27b/32b/llama8b in identical conditions: only 9B
> fails. Root cause is model capability — fused clean+summarize in
> JSON is at the edge for 9B-class models. Trades ~1.5× wall-clock for
> reliable output."

The "1.5×" claim in c6a8982b is the staged-vs-bundled gap (two Ollama calls
vs one). The background's "~3× latency, ~3× cost" is a different number
— that referenced **three sequential calls** (summary + KG + GI), not
two. The runbook (`docs/guides/DGX_RUNBOOK.md` §P2) uses "1.5×" and refers
only to the bundled-clean-summary fusion, not full 3-stage KG+GI.

[CONJECTURE] The "3× latency and ~3× cost" framing in the prompt may
conflate the bundled-clean+summary fusion (1.5× delta) with the bundled
summary+KG+GI fusion. **The 1.5× number is what's actually documented.**
If the operator was thinking of the 3-stage envelope (summary + GI + KG
in one call), there is **no evidence in this repo that we ever shipped that
3-stage envelope** — only summary+clean fusion exists as
`summarize_bundled`. So bundled mode is at most ~2× a baseline staged
call, not 3×.

### 1.3 Current call site

`src/podcast_scraper/providers/ollama/ollama_provider.py::summarize_bundled`
(line 1241–1320) — the actual call:

```python
self.client.chat.completions.create(
    model=self.summary_model,
    messages=[...],
    temperature=self.summary_temperature,
    max_tokens=max_out,
    response_format={"type": "json_object"},   # <-- OpenAI loose JSON mode
    **_ollama_openai_chat_extra_kwargs(
        self.summary_model, num_ctx=self.summary_num_ctx
    ),
)
```

This goes through Ollama's OpenAI-compat layer (port 11434, `/v1/chat/completions`).
The `response_format={"type": "json_object"}` value is the **loose JSON mode** —
the model is asked to emit JSON, but no schema is enforced at the decoder.
This matches #912's diagnosis exactly.

### 1.4 Existing repair layer

`src/podcast_scraper/schemas/summary_schema.py::_repair_json` (line 308)
already runs *after* a `json.loads` failure. It currently only handles:

- Trailing commas before `}` / `]`
- Markdown code-fence stripping

It does NOT handle the actual #912 failure modes — literal newlines inside
strings, mid-bullet truncation, empty `summary` with content in `bullets`.
So #912 Path B ("JSON repair pass") would mean **extending this existing
helper**, not adding a new one.

### 1.5 Existing observability for these failures

ADR-099 / ADR-100 shipped (2026-06-15, commit `8083bb44`) a guardrail layer
at `src/podcast_scraper/providers/guardrails/`. From ADR-099 §"Initial
thresholds":

| Service | Check | Threshold |
| --- | --- | --- |
| Ollama | Empty content OR thinking-prose markers | `content == ""` OR contains `<think>`, `Okay, so I need to`, `Let me think` |
| vLLM | JSON parse + finish_reason | When structured-output requested: JSON parse fails. Always: `finish_reason == "length"` |

So **the JSON parse check at the vLLM tier is already wired**. The Ollama
tier check is for thinking-prose / empty content — **NOT JSON parse**.
[UNVERIFIED] Whether a JSON-parse guardrail is wired at the Ollama tier for
bundled mode specifically — need to grep `providers/guardrails/` to confirm.
For the scope of this research, the relevant point is: failures bubble up
as `GuardrailViolation` and route through `FallbackAwareSummarizationProvider`
(per ADR-100 § "Failure handling per stage"). That layer **exists for the
fix to plug into**, not as the fix itself.

### 1.6 Autoresearch-tier evidence on bundled

`README.md` (repo root, autoresearch program section) carries the canonical
prompt-eval scoreboard for bundled vs non-bundled. Held-out scores on
gpt-4o:

| Track | Mode | Score | Note |
| --- | --- | --- | --- |
| Bullets | **Non-bundled (winner)** | **0.566** | 39.6% |
| Bullets | Bundled | 0.505 | 33.2% |
| Paragraph | **Non-bundled (winner)** | **0.481** | 31.7% |
| Paragraph | Bundled | 0.469 | 29.5% |

**Even on cloud-grade gpt-4o, bundled mode scores 6–10% lower than
staged.** That's a quality finding orthogonal to the parse-failure
finding on qwen3.5:9b. **Bundled mode is not just a reliability problem;
it's a quality regression on every model evaluated so far.**

`autoresearch/JUDGING.md` §"Rubric calibration" (line 390) notes:

> "The rubric was written before the bundled-mode output format
> was finalized..."

So the autoresearch framework was designed primarily around non-bundled
mode; bundled mode was a later attempt to compress latency, and the
quality ladder rejected it on gpt-4o before #912 surfaced the
reliability problem on qwen3.5:9b.

### 1.7 #1016 cohort — Round 3 finalists on the autoresearch vLLM

`docs/wip/EVAL_1016_FINAL_REPORT_2026_06_17.md` is the relevant context
for "what's available on the autoresearch vLLM tier today":

- vLLM image: `nvcr.io/nvidia/vllm:26.05-py3` (NOT post1)
- Compose: `~/agentic-ai-homelab/infra/vllm/autoresearch/docker-compose.yml`
- Slot: tailnet `:8003`
- Current model (per `prod_dgx_full_with_fallback.yaml` flip
  on 2026-06-17): **Qwen3.5-35B-A3B** (cohort top dog: summary 59.4%
  R-1 vs Opus, KG 38%)
- Background context says **Qwen3-30B-A3B-Instruct-2507** runs at :8003;
  `MEMORY.md` confirms that as the autoresearch slot's live model since
  2026-06-15. The `prod_dgx_full_with_fallback.yaml` flip moves the
  **profile** to Qwen3.5-35B-A3B; the **slot's live binding** is
  whatever's deployed. Both are Qwen3 / Qwen3.5 family — the
  `response_format`-via-vLLM analysis below applies equally to either.

The Round 3 report does NOT measure bundled-mode parse failures on the
vLLM tier. **That data does not yet exist.** What we have:

- `prod_dgx_full_with_fallback.yaml` line 86: `llm_pipeline_mode: staged`
  with a comment "`llm_pipeline_mode: staged` retained per #912 workaround"
  — i.e. the workaround was carried into the vLLM-tier flip without
  re-testing bundled on the new substrate.

So when we ask "does bundled work on the vLLM tier via guided_json", we
genuinely don't know yet. The autoresearch v2 framework would have to
add a parse-rate gate (#912 Path D) and run a sweep.

---

## Section 2 — qwen3.5 family JSON behavior

### 2.1 Qwen team docs (Alibaba Cloud Model Studio)

Source:
[Enforce Structured JSON Output with Qwen Models — Alibaba Cloud](https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output)

Qwen team's published guidance for structured JSON output with Qwen3.5:

- Set `response_format` to `{"type": "json_object"}`
- Include the keyword "JSON" (case-insensitive) in system or user message
- Define exact field types (required vs optional), formats, examples
- **"Do not set `max_tokens` when using structured output, as it may
  truncate JSON strings"** — relevant to #912's "mid-bullet truncation"
  symptom. Our call site sets `max_tokens=max_out` explicitly (~16384).
- **Structured output requires thinking mode disabled** for Qwen3.5
  variants that have thinking. The OpenAI-compat layer doesn't disable
  thinking; you must do it explicitly.

### 2.2 Qwen3.5 long-system-prompt requirement

[Hacker News thread](https://news.ycombinator.com/item?id=47201388),
quoted by a contributor:

> "Qwen3.5 pretty much requires a long system prompt, otherwise it goes
> into a weird planning mode" — model "engages in extended reasoning and
> repetitive self-checking" without proper system instructions.

This is the opposite of our #912 Path C hypothesis ("simplify the prompt").
**Stripping the bundled-mode system prompt down may actually destabilize
qwen3.5:9b further, not less.** It also contradicts an intuition that
"smaller models drop instructions when prompts get long" — the empirical
evidence for Qwen3.5 specifically is the inverse.

### 2.3 Known structured-output enforcement bugs

[ollama#15540](https://github.com/ollama/ollama/issues/15540) — Ollama
0.20.3, OpenAI-compat endpoint with `response_format=json_schema`
reportedly returns deltas that **in sum do not obey the given schema**
for Qwen 3.5 9b and Gemma 4 26b. Gemma 3 and GPT-OSS work correctly.
Closed as duplicate of #14645 ("known problem affecting multiple models").

**Implication**: even if we switch to Ollama's `format` schema mode
(stricter than `json_object`), there's a documented upstream bug
specifically affecting Qwen3.5 9b on the OpenAI-compat surface in recent
Ollama versions. The native `/api/chat` endpoint (not OpenAI-compat) may
be less affected, but [UNVERIFIED] — the issue thread doesn't isolate.

[ollama#15502](https://github.com/ollama/ollama/issues/15502) — gemma4:31b
enters word-repetition loops during constrained JSON generation with
free-text string fields (Ollama 0.20.5). Root cause documented in the
issue: "logit distribution degenerates inside JSON string values,
favoring a single token repeatedly. The grammar constraint cannot reject
valid string characters, and the repeat penalty lacks enforcement
mechanism within constrained decoding." **Different model family but
same architectural class of failure**: constrained decoding doesn't help
when the failure is the model's own logit collapse inside an unconstrained
string field. This is the canonical argument against assuming
"grammar-constrained decoding = solved" for free-text-heavy schemas.

### 2.4 Qwen-published JSON cookbook for bundled vs nested

Searched for one. There isn't a Qwen-published cookbook on bundled vs
nested envelopes. Qwen's docs cover JSON output but not the topology
question (one envelope vs three calls). [UNVERIFIED] but consistent
with our internal autoresearch evidence (§1.6) where bundled scores
worse on gpt-4o too — the practical landscape doesn't favor bundled
envelopes regardless of model.

---

## Section 3 — Ollama `format` parameter — why 20× slower

### 3.1 The 10× / 12× / 20× datapoints

[ollama#3851](https://github.com/ollama/ollama/issues/3851) — "the
performance of the `format=\"json\"` param is 10x slower than regular
inference when additional context is included." Concrete repro: a
prompt with context that took ~24s with JSON formatting took ~2s
without it on NVIDIA T4 — **12× slowdown**. Affected version
explicitly cited: Ollama 0.1.32. **Open**, marked duplicate of #3154.

[ollama#3154](https://github.com/ollama/ollama/issues/3154) — same
bug, opened 2024-03-14. We could not retrieve the resolution thread
via WebFetch (page excerpt didn't include the comments). [UNVERIFIED]
whether this was ever fixed in a specific version.

**Our 20× number is consistent with this bucket of reports.** Not
identical (10× / 12× vs 20×) but the same order of magnitude. The
delta likely reflects model size, schema complexity, and context length.

### 3.2 Root cause (what we know vs what's [CONJECTURE])

What we know:

- Ollama's native `format` parameter is grammar-constrained decoding
  ([Ollama API docs](https://github.com/ollama/ollama/blob/main/docs/api.md#parameters)).
  Recent Ollama (0.5+) uses XGrammar under the hood; XGrammar is the
  default backend for vLLM, SGLang, TensorRT-LLM, and MLC-LLM.
- XGrammar's published per-token overhead is "under 40 µs per token
  for JSON Schema" — should be negligible on a GPU, not 10–20×
  ([XGrammar paper](https://arxiv.org/pdf/2411.15100)).
- The 10–12× / 20× slowdown is therefore NOT the steady-state
  per-token mask cost (XGrammar's design point).

[CONJECTURE — strongest candidates for the 20× cost on qwen3.5:9b]:

1. **CPU-bound mask precomputation overhead per-request**. XGrammar
   caches grammar compilation across requests with the same schema;
   first-request cost is higher. If our test pattern boots Ollama,
   sends 1 request, measures, we paid the cold-grammar cost on every
   sample. SqueezeBits' benchmark
   ([SqueezeBits blog](https://blog.squeezebits.com/guided-decoding-performance-vllm-sglang))
   shows XGrammar has "erratic behavior with frequent sharp drops,
   indicating severe CPU bottlenecks" on complex / dynamic schemas.
   Plausible if our schema is complex.
2. **Ollama's specific integration of XGrammar may not overlap mask
   generation with the LLM step**. SqueezeBits found SGLang "overlaps
   per-step mask generation with the LLM inference step, effectively
   hiding grammar processing latency"; vLLM does this less well; Ollama
   is younger code and likely worse still.
3. **Repetition penalty interaction**. ollama#15502 shows constrained
   decoding can drive the model into a near-loop inside string fields,
   inflating generated tokens 5–20×. If our `summary` / `bullets` fields
   are wide free-text strings, qwen3.5:9b under grammar constraint may
   be generating 5–20× more tokens before hitting an EOS the grammar
   allows. **This is testable**: count `completion_tokens` in the
   175s response and compare to the 9s response.
4. **Loss of speculative decoding / KV cache reuse under grammar
   constraint**. Some implementations disable speculative paths when
   logit masks change every token. [UNVERIFIED] whether Ollama does.

### 3.3 What we should verify before any next step

If the 20× decision is going to be revisited:

1. Capture `completion_tokens` (Ollama's API returns it) for the 175s
   run vs the 9s run. If the ratio is ~5–20×, root cause is **logit
   collapse inside string fields** (hypothesis 3). Fix shape: tighten
   the schema (e.g. cap bullet word count), not change decoder.
2. If `completion_tokens` is ~1× (similar token count, much slower per
   token), root cause is **mask generation overhead** (hypotheses 1–2,
   4). Fix shape: switch to vLLM tier with XGrammar properly overlapped,
   or upstream-track an Ollama fix.

Neither verification was done in the prior Path A session. **This is
the cheapest experiment to settle the root cause** — read one more
field from the existing 175s response, no fresh inference needed.

### 3.4 Ollama version in our deployment

[UNVERIFIED]. The DGX_RUNBOOK mentions Ollama on `:11434` but doesn't
pin a version. No `OLLAMA_VERSION` constant in `infra/dgx/converge/deploy.py`.
The reported 50–67% bundled failures in #912 happened on whatever was
deployed on 2026-06-07. The structured-output enforcement bug
(ollama#15540) affected 0.20.3; the JSON-mode slowdown bug (ollama#3851)
affected 0.1.32 originally but the issue is still open. Both could affect
our deployment depending on what's running.

---

## Section 4 — Grammar-constrained decoding landscape (2026)

### 4.1 Four backends, headline numbers

[JSONSchemaBench (2025–2026)](https://arxiv.org/html/2501.10868v3) and
[SqueezeBits benchmark](https://blog.squeezebits.com/guided-decoding-performance-vllm-sglang):

| Backend | Latency (per-token mask) | Accuracy lift vs unconstrained | Strengths | Weaknesses |
| --- | --- | --- | --- | --- |
| **XGrammar** | **<40 µs** for JSON Schema | 20–25% absolute on complex schemas (e.g. Github_medium / Qwen3-32B: 61% → 80%+) | Default for vLLM, SGLang, TensorRT-LLM, MLC-LLM. Best caching. Best on repeated schemas. | Erratic on dynamic / complex schemas — CPU bottleneck spikes. Per ollama#15502 affects logit shape inside free-text fields. |
| **LLGuidance** | Comparable to XGrammar; "no startup cost" | Similar accuracy lift | Better on dynamic / per-request schemas (no precomputation). Backed by Microsoft Guidance team. | Bottlenecks at high batch sizes per SqueezeBits. |
| **Outlines** | Highest startup cost (automaton precomputation) | Similar | Earliest player; broad model support. | "Lowest compliance rate due to compilation timeouts on complex schemas" per JSONSchemaBench. |
| **LM Format Enforcer** | "High latency per token" (Python on-the-fly) | Similar | Lets model control whitespace / field ordering, "preventing suboptimal internal states." | Slowest of the four for steady-state generation. |

### 4.2 vLLM's structured-output surface

vLLM (per
[vLLM docs](https://docs.vllm.ai/en/latest/features/structured_outputs/)):

- OpenAI-compatible: `response_format={"type": "json_schema", "json_schema": {...}}`
- Native: `guided_json=` / `guided_regex=` / `guided_choice=` /
  `guided_grammar=` parameters
- Default backend: `auto` (selects xgrammar / guidance / outlines based
  on request). Override with `--structured-outputs-config.backend`
  server flag.
- Qwen3 caveat: with reasoning enabled, structured outputs may become
  disabled if reasoning content doesn't get parsed into the reasoning
  field. Fix: server flag `--structured-outputs-config.enable_in_reasoning=True`.
  Our autoresearch slot uses `--reasoning-parser=qwen3`
  (`prod_dgx_full_with_fallback.yaml` comment) — these flags need to coexist.
- [vLLM#15236](https://github.com/vllm-project/vllm/issues/15236) is
  a pinned bug tracker about guided generation regressions through v0.8.1;
  vLLM 26.05-py3 (NVIDIA's image) is more recent and the regression list
  should be re-checked at PR time. [UNVERIFIED] whether the
  qwen3.5 + json_schema combination is fully clean on 26.05-py3.

### 4.3 Production benchmark — Qwen3 + xgrammar on vLLM

SqueezeBits (Jan 2026), Qwen3-8B + Qwen3-32B on vLLM with reasoning
disabled, guided JSON:

- "**Guided decoding provides a substantial boost to correctness,
  improving the rate by 20-25% absolute in most cases.**"
- "XGrammar slightly outperformed Guidance due to effective caching."
- Steady-state TPOT only "marginally higher" than unconstrained.

**This is the bench number that matters for Section 5.** If Qwen3-8B
(smaller than our 9B Ollama candidate) gets 20–25% absolute lift on
structured generation correctness via xgrammar on vLLM, and TPOT is
"marginally higher" not 20×, then **the bundled-mode reliability
problem is fixable at the vLLM tier without the catastrophic latency
tax we saw on Ollama**.

### 4.4 Has Qwen team published bundled-JSON benchmarks?

Searched. No. Qwen team's published JSON-output guidance is single-call
(one envelope, one task). They don't publish bundled-fusion benchmarks.
The autoresearch v2 framework (our internal one) is the closest we have
on this question, and it found bundled is 6–10% worse than staged across
all evaluated models (§1.6).

---

## Section 5 — Architecture question + synthesis

### 5.1 Tool calling vs JSON envelope

Headline ([BenchLM.ai 2026](https://benchlm.ai/llm-agent-benchmarks),
[PromptQuorum](https://www.promptquorum.com/power-local-llm/best-local-models-tool-calling-2026)):

- Pure JSON-mode reliability ("the JSON the model emits is well-formed"):
  Gemma 4 27B and GLM-5.1 32B emit clean JSON without trailing prose.
  Most other models in the cohort produce 90%+ well-formed JSON on
  simple workloads, dropping to 80–90% on multi-step / multi-field.
- Tool-calling reliability ("the model picks the right tool and emits
  well-formed arguments"): top-3 models reliably orchestrate multi-step
  workflows. Most fail on multi-step chains.

**Tool calling vs single bundled envelope** as design choices:

- Tool calling is **structurally simpler per call** — the model picks
  one tool, emits one set of args, the wrapper enforces shape via the
  tool schema. Easier for small models.
- Bundled JSON envelope is **structurally complex per call** — model
  must emit multiple top-level fields, nested arrays, free-text inside
  strings. Harder for small models.
- Production reports favor **separate tool calls per task** over
  single bundled envelopes for reliability — this matches our internal
  autoresearch finding (§1.6) that non-bundled outscores bundled even
  on gpt-4o.

For our use case (summary + KG + GI), the equivalent of "tool calling"
is **staged mode** — 3 separate calls, each with its own JSON envelope.
Which is what we already run in production today.

### 5.2 Cost ratio: bundled vs staged in our production telemetry

Background prompt claims "~3× latency, ~3× cost." Actual evidence in
the repo:

- DGX_RUNBOOK §P2 (workaround commit): "**~1.5× wall-clock**." Trade
  is acceptable since DGX RTT is ~30 ms and free.
- The full 3-stage envelope (summary + KG + GI in one call) **does not
  exist in our codebase**. `summarize_bundled` is summary + clean fusion
  only. KG and GI are always separate calls.

So bundled mode in our codebase is at most ~1.5–2× a staged baseline,
not 3×. The "3× cost" framing in the prompt does not match the
documented evidence. **Even at 2×, that's not the catastrophic-cost story
that would force bundled-mode shipping**. On the autoresearch DGX path
the marginal cost is electricity (~$0).

### 5.3 The categorical recommendation

**Recommendation: Bundled mode on Ollama qwen3.5:9b is dead. Do not
re-attempt to fix it. Permanently retire the bundled path on Ollama and
move all subsequent bundled-mode experimentation to the autoresearch
vLLM tier via `response_format={"type":"json_schema", ...}` with
xgrammar.**

Rationale (in order of weight):

1. **Autoresearch v2 already proved bundled is a quality regression** on
   every model evaluated, including cloud gpt-4o (6–10% lower vs staged
   on held-out scores, §1.6). The autoresearch framework's prior
   "champion" finding for `qwen3.5:9b bundled` was an artifact of the
   ROUGE-L metric not penalizing parse failures, per #912's "Why the
   autoresearch framework didn't catch it" section. **Bundled mode has
   no quality case independent of the latency case.**
2. **The latency case is overstated.** Documented gap is ~1.5×
   (DGX_RUNBOOK §P2). On a free-electricity DGX path, this is not a
   user-visible cost.
3. **Path A on Ollama is dead.** 20× slowdown is consistent with the
   known ollama#3851 / #3154 family of structured-output performance
   bugs, plus the ollama#15540 enforcement-not-actually-enforced bug
   specifically affecting Qwen3.5 9b. **Two separate active Ollama
   bugs on the qwen3.5:9b + JSON schema combination.** Not a
   one-experiment fix.
4. **The forward-looking platform is vLLM with xgrammar.** SqueezeBits
   2026 confirms 20–25% absolute correctness lift with "marginally higher"
   TPOT on Qwen3-8B / 32B. This is the tier where the experiment from
   #912's "Path A" actually has a path to ship. But it's not a
   "bundled mode on qwen3.5:9b" experiment — it's a different
   experiment on a different substrate.
5. **The cleanup matters**. Path D from #912 (autoresearch JSON-parse
   gate) is the *real* unblocked task. It makes future autoresearch
   sweeps honest. Pair it with #912 Path A retired into "vLLM-tier
   guided_json experiment for the autoresearch slot", and we're done.

**Confidence**: high on #1–#3 (direct internal+external evidence).
Medium on #4 (vLLM 26.05-py3 + qwen3-family + json_schema combination
hasn't been directly benchmarked in *our* environment yet — only by
SqueezeBits on smaller Qwen3 variants).

### 5.4 Estimated next-PR scope

**Recommended PR ("Permanently retire bundled mode on Ollama; track
vLLM-tier guided_json as a separate experiment")**:

1. **Drop the bundled-mode prompts and `summarize_bundled` path on the
   Ollama provider.** `src/podcast_scraper/providers/ollama/ollama_provider.py::summarize_bundled`,
   the bundled-mode user/system prompt templates,
   `autoresearch/bundled_prompt_tuning/eval/`. Keep `summarize_bundled`
   on the cloud providers (OpenAI / Anthropic / Gemini / DeepSeek / Grok /
   Mistral) — they still work and #912 is Ollama-specific. **Verify
   first** that no profile YAML still references
   `llm_pipeline_mode: bundled` for an Ollama target. Hard-fail at
   config validation if one does.
2. **Path D from #912** (autoresearch JSON-parse counter). The eval
   framework gets a `bundled_response_parse_failures` field surfaced
   in `metrics_report.md`. ~150 LOC, eval framework change.
3. **Close #912 with a forward-pointing follow-up** ("vLLM-tier
   bundled-via-guided_json autoresearch experiment") that captures the
   Section 4 evidence and proposes a 20-sample sweep on the autoresearch
   slot with `response_format={"type":"json_schema",...}`. **Don't open
   it now** (per `feedback_never_open_gh_issues.md`) — surface to
   operator first.

Estimate: ~1 day of work (mostly drift across tests / docs), 0 LOC
of "fix" because the fix is "drop the broken path."

**Out of scope for this PR**:
- Implementing vLLM-tier bundled mode. That's a separate experiment.
- Touching the cloud-provider `summarize_bundled` implementations.
- Investigating ollama#15540 upstream.

---

## Confidence + open questions

**High confidence**:

- Bundled mode is a quality regression on every model autoresearch has
  evaluated, including cloud gpt-4o (internal evidence, §1.6).
- The Path A 20× slowdown on Ollama matches a known upstream bug class
  (ollama#3851/#3154) and a separate qwen3.5:9b-specific enforcement bug
  (ollama#15540). External evidence converges.
- The documented latency cost of staged mode is ~1.5×, not 3×
  (DGX_RUNBOOK §P2 vs the 3× framing in the task prompt).

**Medium confidence**:

- vLLM 26.05-py3 + qwen3-family + `response_format={"type":"json_schema",...}`
  will deliver acceptable parse rates AND acceptable latency for a
  bundled envelope. The SqueezeBits Jan 2026 numbers are on smaller
  Qwen3 variants. We have not directly benchmarked our slot.

**Open questions** (for the operator to decide):

1. **Is "permanently retire Ollama-tier bundled" acceptable**, or do we
   need to keep the call site around as a future-proofing hook? (Argues
   for: every autoresearch run since 2026-06-07 has used staged. Argues
   against: code-deletion is irreversible vs commenting out the profile
   field.)
2. **Should the vLLM-tier bundled experiment be filed now** (as
   continuation of #912) or wait for an explicit operator nod? Per the
   "never open GH issues" rule, surface first, file when authorized.
3. **Path D (autoresearch JSON-parse gate) — ship in the same PR or
   separate?** Argues for same: tightly coupled with the bundled-mode
   retirement story. Argues for separate: ~150 LOC, eval-framework
   surface area, may want isolated review.
4. **Do we need to read `completion_tokens` from the prior session's
   175s response before deciding?** Cheap experiment, settles the root
   cause of the 20× number even if we retire the path anyway. Worth
   recording in this doc as a "we know exactly why it was slow"
   datapoint.

---

## References

- `gh issue view 912` (issue body verbatim)
- `git show c6a8982b` (workaround commit)
- `src/podcast_scraper/providers/ollama/ollama_provider.py::summarize_bundled` (current call site, line 1241)
- `src/podcast_scraper/schemas/summary_schema.py::_repair_json` (existing repair, line 308)
- `docs/guides/DGX_RUNBOOK.md` §P2 ("qwen3.5:9b pipeline mode investigation")
- `docs/adr/ADR-099-response-shape-guardrails-for-self-deployed-services.md`
- `docs/adr/ADR-100-response-shape-guardrails-for-cloud-llm-providers.md`
- `docs/wip/EVAL_1016_FINAL_REPORT_2026_06_17.md`
- `docs/guides/eval-reports/EVAL_SUMMARY_DGX_LOCAL_2026_06.md`
- `README.md` autoresearch program section (bundled vs non-bundled scoreboard)
- `autoresearch/JUDGING.md`
- [Ollama issue #3851 — format=json 10–12× slowdown](https://github.com/ollama/ollama/issues/3851)
- [Ollama issue #3154 — original slowdown bug, 2024-03-14](https://github.com/ollama/ollama/issues/3154)
- [Ollama issue #15540 — structured output not enforced on qwen 3.5 / gemma 4](https://github.com/ollama/ollama/issues/15540)
- [Ollama issue #15502 — gemma4 repetition loop under constrained JSON](https://github.com/ollama/ollama/issues/15502)
- [Ollama API docs — `format` parameter](https://github.com/ollama/ollama/blob/main/docs/api.md#parameters)
- [Alibaba Cloud Model Studio — Qwen structured output](https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output)
- [HN — Qwen3.5 system-prompt requirement](https://news.ycombinator.com/item?id=47201388)
- [Qwen + vLLM deployment guide](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)
- [vLLM docs — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)
- [vLLM issue #15236 — guided generation regressions through v0.8.1](https://github.com/vllm-project/vllm/issues/15236)
- [SqueezeBits — Guided decoding performance on vLLM vs SGLang (Jan 2026)](https://blog.squeezebits.com/guided-decoding-performance-vllm-sglang)
- [XGrammar paper — arxiv 2411.15100](https://arxiv.org/pdf/2411.15100)
- [JSONSchemaBench — arxiv 2501.10868v3](https://arxiv.org/html/2501.10868v3)
- [BenchLM.ai — LLM agent & tool-use benchmarks 2026](https://benchlm.ai/llm-agent-benchmarks)
- [PromptQuorum — Best local models for tool calling 2026](https://www.promptquorum.com/power-local-llm/best-local-models-tool-calling-2026)

---

## VALIDATION EVAL (post-research, 2026-06-18)

The research recommendation rested on two empirical claims: (a) qwen3.5:9b
fails bundled JSON parsing 50–67% of the time, (b) Ollama's `format`
schema parameter (Path A) imposes a 20× wall-clock tax. Three rounds of
direct A/B measurement against the exact `OllamaProvider.summarize_bundled`
code path (system prompt + user prompt rendered from
`bundled_clean_summary_{system,user}_v1.j2`, temperature=0.2, num_ctx=32768,
reasoning_effort=none) on qwen3.5:9b.

Validation script: `autoresearch/912_validation/run_path_a_vs_baseline.py`.
Raw trial logs: `trials.jsonl` (Phase 1), `trials_phase2_long.jsonl`
(Phase 2), `trials_phase3_dgx.jsonl` (Phase 3).

| Phase | Host | Ollama | Fixture | Trials | Baseline parse | Path A parse | Baseline median (s) | Path A median (s) | A latency tax |
|---|---|---|---|---|---|---|---|---|---|
| 1 | MBP | 0.19.0 | p01_e01_fast.txt (1.5K) | 10 | 10/10 | 10/10 | 23.18 | 24.89 | 1.07× |
| 2 | MBP | 0.19.0 | p01_e01.txt (10.9K) | 30 | 30/30 | 30/30 | 27.99 | 28.95 | 1.03× |
| 3 | **DGX (dgx-llm-1)** | **0.30.5** | p01_e01.txt (10.9K) | 30 | **30/30** | **30/30** | **12.59** | **13.06** | **1.04×** |

**Totals: 140 trials, 0 parse failures, on two hosts and two Ollama
versions (0.19.0 + 0.30.5).** Statistical: probability of seeing 0
failures at a true 50% fail rate = 0.5^140 ≈ 10⁻⁴³ — definitively
ruled out. At 5% true rate, prob ≈ 0.075% (ruled out). At 2% true rate,
prob ≈ 6% (not ruled out, but well below the issue's 50–67% claim).

### Findings vs the research recommendation

| Recommendation premise | Validation result | Verdict |
|---|---|---|
| qwen3.5:9b parse fail 50–67% (`response_format: json_object`) | 0/70 fail across MBP+DGX, 0.5^70 ≈ 10⁻²¹ | **Premise broken** |
| Path A imposes ~20× latency tax | 1.03–1.07× across all configs | **Premise broken** |
| Ollama upstream bugs #3851/#3154/#15540 bite our config | No symptoms observed | **Not triggered at our params** |
| Bundled mode is a quality regression on every model | Not measured in this validation (separate question) | **Carried over from research, unaffected by this run** |

### Bonus operational facts surfaced

- **DGX vs MBP latency**: DGX (12.6s baseline median) is ~2.2× faster than
  MBP (28.0s baseline median) for the same fixture × params × model. The
  cost framing of "bundled mode on DGX" is meaningfully cheaper than the
  research assumed.
- **completion_tokens diagnostic** (the prior session's Q4): baseline
  median 434 (DGX) / 618 (MBP long), Path A median 454 / 636. Well under
  the 16384 max_tokens; no truncation; Path A adds ~3% more tokens
  (consistent with grammar-constrained decoding nudging the model
  slightly more verbose, not 20× slower).

### Decision (high confidence)

**Close #912 as no-longer-reproducible.** Recommendation in the research
section ("retire bundled mode on Ollama") was correct for the *premises
it assumed* but those premises do not hold at our current config.
Engineering-grade evidence to support the close: 140 clean trials, two
hosts, two Ollama versions, identical to the production code path.

### Deferred to follow-up (NOT executed in this branch)

- **Path D** (`bundled_response_parse_failures` counter in the autoresearch
  eval framework) — still independently valuable as a durable regression
  guard. Tracked as a task; not in scope for the current cluster.
- **Workaround revert** — flipping `config/profiles/local_dgx_balanced.yaml`
  back to `llm_pipeline_mode: bundled` is a config decision the operator
  should make explicitly; held pending decision.
- **Quality eval** of bundled vs staged across cohort (separate from
  parse-rate) — the open question whether bundled is itself a quality
  regression remains unanswered. Covered indirectly by task #113 (small-
  model standoff) which scores per-stage quality at the cohort level.

### Caveats on the no-repro

- 140 trials at one set of params. Edge cases not exercised: temperature
  0.0 vs 0.4, smaller num_ctx, model under contention, alternative
  transcripts (only one episode tested, in English, ~10K chars). Path D
  is the long-term observability.
- "We did reproduce it yesterday" — operator confirmed a recent
  reproduction; today's runs cannot match those conditions. The
  reproduction therefore exists *somewhere* in the state space; we did
  not localize it. The close as no-repro is conditional on Path D
  catching any recurrence quickly.
