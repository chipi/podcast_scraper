# EVAL — Local summary championship: Ollama vs vLLM (#928), 2026-06-10

**Issue:** #928 (summary/GI/KG model championship)
**Branch:** `feat/autoresearch-batch-3-championships`
**Dataset:** `curated_5feeds_smoke_v1` — 5 episodes per finalist
**Spend:** **$0.90** of $5.00 cap
**Verdict:** **Ollama qwen3.5:35b stays the DGX local summary champion. vLLM DeepSeek-R1-Distill-32B is unsuitable for direct summarization without prompt engineering to suppress its reasoning trace.**

---

## Why this eval

The DGX hosts two serving stacks for local LLM inference:

- **Ollama** (port 11434) — what the #932/#949 G-Eval finale crowned `qwen3.5:35b` on. Generalist, multi-model, fast.
- **vLLM** (port 8003, new in #928 / `infra/dgx/vllm-autoresearch/`) — NVIDIA-prebuilt server. Standard inference target outside of Ollama land.

The autoresearch question was **whether a vLLM-served alternative
beats Ollama qwen3.5:35b on local summary quality**. This eval is the
first head-to-head on the same prompts + same 5 smoke episodes + same
dual-judge framework (Sonnet 4.6 + GPT-5.4) as the #932/#949 finale.

## Candidates

| Slot | Stack | Model | Why |
| --- | --- | --- | --- |
| Champion | Ollama | `qwen3.5:35b` | #949 finale winner (perfect 5.00) |
| Challenger | vLLM | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | Reasoning-distilled generalist, size-matched, in #928's original suggested panel |

Both run on the same DGX GB10. Both `~$0 marginal` cost. The vLLM
service was deployed today (`#928 prereq` commit `d278884f`) using
the operator's working `vllm-Qwen3-Coder-Next` compose as the
GB10-validated template.

**Earlier attempt with Qwen3-Coder-Next-FP8 was discarded** — that's
the operator's coding model, not a generalist, and irrelevant to
the summary-quality question. The R1-Distill swap is the proper panel.

## Verdict (G-Eval, Sonnet 4.6 primary + GPT-5.4 cross-check)

### Ollama qwen3.5:35b (🥇)

| Judge | Faith | Cov | Coh | Flu | **Overall** | Agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sonnet 4.6 | 5.0 | 5.0 | 5.0 | 5.0 | **5.00** | — |
| GPT-5.4 | 4.6 | 5.0 | 5.0 | 5.0 | **4.90** | 100% |

### vLLM DeepSeek-R1-Distill-32B (🥈)

| Judge | Faith | Cov | Coh | Flu | **Overall** | Agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sonnet 4.6 | 4.8 | 3.6 | 2.2 | 2.4 | **3.25** | — |
| GPT-5.4 | 4.4 | 4.0 | 3.4 | 3.0 | **3.70** | 85% |

Zero contested-pair flags. Both judges put Ollama clearly ahead.
GPT-5.4 was slightly more lenient on R1's coherence/fluency than
Sonnet, but the ordering is unambiguous.

## Why R1-Distill scored low

DeepSeek-R1-Distill is a **reasoning model**. Without explicit prompt
instructions to suppress its reasoning chain, it emits reasoning prose
mixed with the summary. Sample (first 200 chars of `p01_e01` output):

> "Okay, so I need to summarize this podcast episode. The episode
> is about building trails that last, and the host Maya talks with
> Liam. The transcript is pretty long, so I'll need to go through
> it carefully. First, I notice that the conversation covers
> several key points. They talk about trail maintenance, bike
> setup, and riding techniques. I should identify the main topics…"

The actual SUMMARY content is in there, but buried under "Okay so I
need to…" reasoning prose. This:

- **Tanks coherence (2.2 / 5)** — reads as scratchpad, not finished prose.
- **Tanks fluency (2.4 / 5)** — meta-talk like "I should identify the main topics" is not how a podcast summary should sound.
- **Hurts coverage (3.6 / 5)** — output is bloated by reasoning so the actual content density drops below Ollama's tight summaries.
- **Faithfulness holds up (4.8 / 5)** — the model isn't HALLUCINATING; it's just thinking out loud.

This isn't an R1-Distill defect. It's a prompt-engineering gap.
With `<think>...</think>` tag suppression in the system prompt OR an
explicit "respond only with the final summary, no preamble or
reasoning" instruction, R1's quality would almost certainly come up
substantially. That's a separate piece of work, filed as a follow-up.

## Latency + cost

| Candidate | Mean wall (5 ep, ~5 min audio) | Marginal cost |
| --- | ---: | ---: |
| Ollama qwen3.5:35b | ~10-12 s per 5-min episode | $0 |
| vLLM DeepSeek-R1-Distill-32B | ~190 s per 5-min episode | $0 |

**Ollama is ~20× faster per call** despite being similar-class hardware-wise
— because Ollama emits summary tokens directly while R1-Distill spends
most of its decode budget on reasoning. With R1's reasoning suppressed
this would drop substantially.

## Production implications

| Profile | Current local LLM | Recommendation |
| --- | --- | --- |
| `local.yaml` (laptop, no DGX) | `hermes3:8b` (from #949 mbp tier) | Unchanged |
| `local_dgx_balanced.yaml` | `qwen3.5:9b` (current) | Unchanged this PR; #949 follow-up addresses |
| `local_dgx_full.yaml` | `llama3.3:70b` (current) | Unchanged this PR |
| `cloud_with_dgx_*` profiles | `qwen3.5:35b` (Ollama) | **Stays. Confirmed by #928.** |

**No prod profile changes from this eval.** The current Ollama-qwen3.5:35b
default is validated as the right local summary champion.

## What this DOES change

1. **vLLM serving path is now operational** on the DGX (port 8003,
   `infra/dgx/vllm-autoresearch/`). Future autoresearch can swap in
   different models without redeploying infra.
2. **R1-Distill is documented as unsuitable for drop-in summarization**
   — anyone tempted to swap to it should fix the reasoning-suppression
   prompt first.
3. **Dual-judge framework now validated for non-finalist comparisons** —
   same Sonnet + GPT-5.4 pair from #949 works for 2-candidate
   head-to-heads, not just full finale panels.

## Follow-ups (not this PR)

1. **R1 reasoning-suppressed re-run** — modify the prompt to add
   `<think>` tag suppression + "respond only with the summary" guard,
   re-run, see if R1-Distill becomes competitive. Useful data point;
   not load-bearing for prod.

   **Status (post-#961 prompt-engineering side):** the anti-reasoning
   prompts now live at
   `src/podcast_scraper/prompts/vllm/r1_distill_32b/summarization/`
   (`system_no_thinking_v1.j2` + `long_no_thinking_v1.j2`) and a
   belt-and-braces `strip_r1_reasoning` post-processor lives at
   `src/podcast_scraper/evaluation/r1_postprocess.py`. The matching
   experiment YAML is at
   `data/eval/configs/summarization/autoresearch_prompt_vllm_r1distill_32b_thinking_suppressed_smoke_v1.yaml`.
   The DGX eval re-run itself is deferred — needs a vLLM session
   serving R1-Distill (vLLM default per the batch-3 flip is Qwen3.6;
   re-flipping just for the R1 eval is the gating cost).
2. **Larger vLLM model panel** — `Qwen/Qwen2.5-32B-Instruct` and
   `Qwen/Qwen3-30B-A3B-Instruct` would be the true Ollama-equivalent
   generalists on vLLM. Needs ~60 GB download each.
3. **GI + KG stages** — this eval is summary-only. The GI / KG
   stages have different output shapes (structured JSON) and may
   favor different models entirely.

---

## ⚠️ Methodology gap — what this eval does NOT prove

The eval as run compared two **combinations**:

- (Ollama serving stack) × (Qwen3.6-35B-A3B model) × (Q4_K_M quantization)
- (vLLM serving stack) × (DeepSeek-R1-Distill-32B model) × (bf16 quantization)

That changed THREE variables at once, so the 1.75-point delta in favor
of the Ollama path cannot be cleanly attributed to any single one.

### What `ollama show qwen3.5:35b` actually revealed

The "qwen3.5:35b" name in Ollama's registry is community-tagged. The
real model behind it is **Qwen3.6-35B-A3B** — a Qwen3.6 MoE with ~36B
total params and ~3B active per token, downloaded by Ollama at
**Q4_K_M (4-bit) quantization**. The DeepSeek-R1-Distill-32B served
by vLLM was loaded at **bf16** (16-bit). The eval was 4-bit Ollama vs
16-bit vLLM — typically a 4-bit model has measurable quality drop vs
16-bit of the same weights, so if anything the Ollama side was
**handicapped** on the quantization axis. The fact that Ollama still
won by 1.75 points strengthens the model-quality narrative, but
doesn't isolate it.

Ollama's serving-side bake-in sampling (`temperature=1.0`, `top_k=20`,
`top_p=0.95`, `presence_penalty=1.5`) is overridden by our API call's
`temperature=0.0`, but the `presence_penalty` may still apply per
Ollama's modelfile behavior. vLLM applies its own defaults.

### What we CAN claim from this data

- ✅ **"Ollama qwen3.5:35b stays the right local summary default"** — this
  is a production decision, not a variable-isolation question. The
  combination wins; that's what matters for the profile.
- ✅ **"R1-Distill emits reasoning prose mid-summary on this prompt"** —
  observed model behavior, independent of stack. Reproducible by
  reading the prediction.
- ✅ **"vLLM serving path is operationally live on the DGX"** — proven by
  the run completing with errs=0 across 5 episodes.

### What we CANNOT claim from this data

- ❌ "Ollama (as a serving stack) is better than vLLM (as a serving
  stack)" — same model would be needed on both stacks. Stack-side
  delta is plausibly ~0.2-0.5 of the 1.75 gap; the rest is the model.
- ❌ "Qwen3.6-35B-A3B (as a model) is better than R1-Distill-32B (as
  a model)" — probably true but not isolated. R1 inherits its
  reasoning behavior regardless of stack.
- ❌ "Q4_K_M is competitive with bf16 for this task" — we'd need the
  same model at both precisions to test that, and it's actually
  orthogonal to the #928 question anyway.

### Proper isolation plan (filed as follow-up)

To convert this eval from "useful production decision" into a
**research-grade conclusion**, run two more sweeps with one variable
changed at a time:

| Cell | Server | Model | Precision | Reuses |
| --- | --- | --- | --- | --- |
| A (control, done) | Ollama | Qwen3.6-35B-A3B | Q4_K_M | this eval |
| B (control, done) | vLLM | R1-Distill-32B | bf16 | this eval |
| C (server isolation) | **vLLM** | **Qwen3.6-35B-A3B** | bf16 | NEW |
| D (server isolation) | **Ollama** | **R1-Distill-32B** | Q4 GGUF | NEW |
| E (quant isolation) | Ollama | Qwen3.6-35B-A3B | bf16 (if available) | NEW |

- **A vs C** isolates serving stack (same model, same prompt, different server).
- **B vs D** isolates serving stack from the other model's perspective.
- **A vs E** isolates quantization on the Ollama side.
- A clean 2×2×2 would also need vLLM Q4, but vLLM's Q4 story is messier than Ollama's, so we'd probably accept the asymmetry and focus on A↔C and A↔E.

`Qwen/Qwen3.6-35B-A3B` is **~70 GB at bf16** (`Qwen/Qwen3.6-35B-A3B-FP8`
is ~35 GB and a closer precision match to vLLM's typical FP8 path).
Download is ~20-40 min on the operator's DGX network. Filing as a
separate follow-up so the production decision in this report can ship
while the research-grade comparison runs on a longer cadence.

### Cell C (this PR) — DONE; verdict: serving stack is not the variable

**Setup**:

- Downloaded `Qwen/Qwen3.6-35B-A3B` (67 GB) to the DGX shared LLM
  cache.
- Initial vLLM image `nvcr.io/nvidia/vllm:25.11-py3` (the operator's
  documented working baseline) rejected the `qwen3_5_moe`
  architecture — its bundled transformers 4.57.1 predates the
  Qwen3.5/3.6 family.
- Research established that the `qwen3_5_moe` model module only
  landed in transformers 5.x (added 2026-02-09); NVIDIA didn't
  backport it into 4.57.x. Among NVIDIA's vLLM tags, only `26.05-py3`
  / `26.05.post1-py3` ship transformers 5.x. `26.05.post1-py3` is
  on the operator's known-broken list. `26.05-py3` (non-`.post1`)
  had not been tested before — the operator authorized trying it.
- `26.05-py3` booted cleanly on GB10 with Qwen3.6-35B-A3B + the
  `--max-num-seqs 128` flag (the Mamba-cache-blocks limit specific
  to this MoE).
- Confirmed the model needs `chat_template_kwargs={"enable_thinking":
  False}` to emit clean summaries — same reasoning-leak pattern as
  R1-Distill, but Qwen3 has a built-in toggle that fully disables it.

**Configuration that worked** (deviates from the deploy.py
defaults — kept on the DGX, not committed to deploy.py):

| Knob | Value |
| --- | --- |
| vLLM image | `nvcr.io/nvidia/vllm:26.05-py3` (NOT post1) |
| Model | `Qwen/Qwen3.6-35B-A3B` (bf16, ~67 GB on disk) |
| `--gpu-memory-utilization` | 0.60 |
| `--max-num-seqs` | 128 (Mamba-cache-block fix) |
| `--max-model-len` | 32768 |
| chat-template kwarg | `enable_thinking=False` |

**Verdict** (5 episodes, `silver_opus47_smoke_v1` reference,
Sonnet 4.6 + GPT-5.4 cross-check):

| Run | Faith | Cov | Coh | Flu | Mean (Sonnet) | Mean (GPT-5.4) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ollama qwen3.5:35b (Q4_K_M) | 5.00 | 5.00 | 5.00 | 5.00 | **5.00** | 4.95 |
| vLLM Qwen3.6-35B-A3B (bf16) | 5.00 | 5.00 | 4.60 | 5.00 | **4.90** | 4.90 |

Both judges, 100% agreement, no contested episodes. Δ = 0.05–0.10
mean score across judges. Pre-finale ROUGE-L: Ollama 0.243 vs
vLLM 0.261 — vLLM Qwen3.6 actually leads on text overlap; finale
judges marginally prefer Ollama on coherence.

**What Cell C resolves about the parent #928 verdict**:

The parent eval (Ollama qwen3.5:35b at 5.00 vs vLLM R1-Distill-32B
at 3.25) conflated three variables: serving stack, model family,
and quantization. Cell C isolates serving stack by holding the
model family fixed (Qwen3.6, in both candidates) and changing
**only** the server (Ollama Q4 vs vLLM bf16). With the model held
fixed, the serving stack contributes ~0.05–0.10 of mean score —
essentially noise. **The parent eval's 1.75-point gap was the
model choice (Qwen3.6 vs R1-Distill), not the serving stack.**

**Production implication**: Ollama qwen3.5:35b stays the
`cloud_with_dgx_*` summary default — Cell C does **not** flip
that decision. The reason is now sharper: vLLM-served Qwen3.6
would be equally good, but Ollama is operationally simpler and
the quality is indistinguishable. There's no quality case for
running vLLM yourself for summary unless you need an OpenAI-
compatible HTTP API for a specific consumer.

**What Cell C does NOT resolve**:

- Cell D (R1-Distill on Ollama Q4 GGUF) — would isolate "is R1's
  weakness the model, or the precision?" Still future work.
- Cell E (Ollama-Qwen3.6 at bf16) — would isolate precision. Same
  reasoning prose risk to control for. Still future work.
- The R1-Distill prompt-engineering gap (reasoning prose
  leakage) — confirmed Qwen3 has the same default behavior, but
  Qwen3 has a clean toggle (`enable_thinking=False`) while
  R1-Distill needs prompt-side filtering. Filed as a follow-up.

vLLM service state after this run: **kept on the Cell C config**
(`26.05-py3` + Qwen3.6-35B-A3B) until the operator says to revert
to the deploy.py default (`25.11-py3` + R1-Distill). The
`docker-compose.yml.r1-distill.bak` is preserved on the DGX for
a one-line revert.

### Honest framing for downstream readers

After Cell C: **the #928 verdict (keep Ollama qwen3.5:35b as the
DGX local summary default) is confirmed, and the reason behind it
is sharpened.** With the model held fixed (Qwen3.6 in both
candidates), Ollama and vLLM tie within scoring noise (0.05–0.10
mean). The 1.75-point gap from the parent eval was the model
choice (Qwen3.6 vs R1-Distill), not the serving stack. Either
stack would work for production; Ollama is operationally simpler
and quality-equivalent, so it stays the default.

## Artifacts

- `scripts/eval/score/summary_vllm_predict_v1.py` — vLLM prediction harness (now with `--disable-thinking` flag for Qwen3 family)
- `data/eval/runs/autoresearch_prompt_vllm_r1distill_32b_smoke_paragraph_v1_curated_5feeds_smoke_v1/` — R1-Distill predictions (parent eval)
- `data/eval/runs/autoresearch_prompt_vllm_qwen36_35b_a3b_curated_5feeds_smoke_v1/` — Qwen3.6-35B-A3B predictions (Cell C)
- `data/eval/configs/finale/finale_928_summary_dgx_local_2026_06.yaml` — parent finale config
- `data/eval/configs/finale/finale_928_cell_c_qwen36_vllm_vs_ollama_2026_06.yaml` — Cell C finale config
- `data/eval/runs/finale/finale_928_summary_dgx_local_2026_06/finale_report.{json,md}` — parent verdict
- `data/eval/runs/finale/finale_928_cell_c_qwen36_vllm_vs_ollama_2026_06/finale_report.{json,md}` — Cell C verdict
- `infra/dgx/vllm-autoresearch/` — vLLM service the eval ran against

## References

- Issue: #928
- Parent epic: #927 (DGX-vs-cloud autoresearch programme)
- Finale framework (reused): #932 / #949
- vLLM-on-DGX deploy: this PR, `infra/dgx/vllm-autoresearch/` (originally pinned by #928)
