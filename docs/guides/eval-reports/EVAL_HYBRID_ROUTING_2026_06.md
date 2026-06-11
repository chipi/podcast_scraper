# EVAL — Hybrid routing synthesis (#931), 2026-06-10

**Issue:** #931 (synthesize #928 / #929 / #930 into a routing decision)
**Branch:** `feat/autoresearch-batch-3-championships`
**Status:** **Synthesis-in-progress.** Locks in the framing the user articulated during the batch ("we have options, not all users have DGX — we keep options and explore") and pins the per-profile decisions on what the underlying evals concluded.

---

## TL;DR

Three production profile shapes, three different best-of-class
recommendations. Not "X wins, kill the others" — instead "here's the
operating point each profile occupies, what we measured, and where the
tradeoffs land."

| Profile shape | Whisper | Diarize | Summary | Net change from before this PR |
| --- | --- | --- | --- | --- |
| `local.yaml` (laptop, no DGX) | openai-whisper / MPS | pyannote / MPS | hermes3:8b (Ollama on CPU) | None — confirmed |
| `cloud_with_dgx_*` (DGX-equipped) | **openai-whisper / `:8002` — promoted after temperature-bug fix this PR** | pyannote / `:8001` | Ollama qwen3.5:35b on `:11434` | Transcription default flips to DGX `:8002` (was: MPS/cloud per pre-Cell-C draft) |
| `cloud_*` (no DGX, cloud-OK) | cloud Whisper API | Gemini speech (current — unmeasured this batch) | gemini-2.5-flash-lite | None — not re-litigated |

After late-batch findings landed, two defaults DO flip:

- **`cloud_with_dgx_*` transcription** → `whisper-openai` on
  `dgx:8002` (after the temperature-schedule fix this PR).
- **vLLM autoresearch service default** → `26.05-py3 +
  Qwen3.6-35B-A3B` (after Cell C confirmed parity with Ollama
  on the same model family).

The `local.yaml` and `cloud_*` shapes stay as-is — evidence base
under each profile improved, no default change needed.

## What each underlying eval concluded

### Diarization (#930)

**3-way: pyannote on Apple MPS vs NVIDIA CUDA (DGX) vs pure CPU.**

- MPS and CUDA are **essentially tied** at ~23 s for a ~5-min episode
  (~13× realtime). Same model, same numerics-up-to-noise.
- Pure CPU is **~17× slower** than either GPU path (~415 s).
- Speaker-count is contaminated by single-voice v2 TTS (`#934`); can't
  draw quality conclusions from this fixture set until distinct
  voices land.

**Net effect on routing**: pyannote in-process is fine for any
profile that runs on a GPU host (Apple Silicon counts). Route to DGX
when the host has no GPU, OR when you want to keep load off the
laptop.

Full report: `docs/guides/eval-reports/EVAL_DIARIZATION_DGX_VS_CLOUD_2026_06.md`

### Summary (#928, with methodology caveat)

**Head-to-head: Ollama qwen3.5:35b vs vLLM DeepSeek-R1-Distill-32B.**

- Ollama 5.00 / 5.00 / 5.00 / 5.00 = **5.00 mean** (Sonnet) /
  **4.90** (GPT-5.4 cross-check, 100% agreement, no contested flags)
- vLLM R1-Distill 4.80 / 3.60 / 2.20 / 2.40 = **3.25 mean** (Sonnet) /
  **3.70** (GPT-5.4, 85% agreement, no contested flags)
- R1-Distill emits **reasoning prose mid-summary** (e.g., "Okay, so I
  need to summarize this podcast episode…"). That's a prompt-
  engineering gap, not a model defect. Fixable in a follow-up; not
  load-bearing for this PR.

**Cell C (proper isolation, done in this PR)**: vLLM-served
Qwen3.6-35B-A3B (bf16) on `nvcr.io/nvidia/vllm:26.05-py3` tied
with Ollama-served qwen3.5:35b (Q4_K_M) within scoring noise —
Sonnet 4.6 mean 4.90 vs 5.00, GPT-5.4 cross-check 4.90 vs 4.95.
**The 1.75-point gap in the parent eval was the model choice
(Qwen3.6 vs R1-Distill), not the serving stack.**

**Net effect on routing**: keep Ollama qwen3.5:35b as the
`cloud_with_dgx_*` summary default. The reason is now operational
(Ollama is simpler to manage), not quality (both stacks are
indistinguishable when serving the same model family).

Full report: `docs/guides/eval-reports/EVAL_SUMMARY_DGX_LOCAL_2026_06.md`

### Transcription (#929 — done; 4-way, post-fix)

**4-way: openai-whisper on Apple MPS vs DGX `:8002` (fixed) vs pure
CPU vs DGX faster-whisper on `:8000` (speaches image).**

- **MPS** (laptop): WER 0.096 mean, 1.6× realtime (warm 2.6-3.8×).
  Clean local default.
- **Pure CPU** (M4 Pro, 5 episodes): WER 0.137 mean, 2.34× realtime.
  Quality 30% behind MPS, still production-viable. **MPS is only
  ~1.4× faster than CPU on this workload, not 17×** — the
  diarization extrapolation didn't hold for openai-whisper.
- **DGX openai-whisper on `:8002`, AFTER the bug fix this PR**:
  WER 0.102 mean (within 6% of MPS, inside scoring noise), **4.56×
  realtime** (~3× faster than MPS, ~2× faster than CPU). **New
  production winner for DGX-equipped profiles.**
- **DGX faster-whisper (speaches image, `:8000`)**: still broken —
  empty output on 4/5 episodes, hallucinations on the 5th. Separate
  bug from the openai-whisper one (lives in speaches container
  config). Filed as follow-up; NOT a production candidate.

**The temperature-schedule bug**: pre-fix, DGX openai-whisper was
producing WER 3.20 mean (hallucinations with 2-9× extra words). The
synthesis-pre-Cell-C version of this report recommended NOT routing
to DGX whisper because of those numbers. **Root cause was a
config bug** in `infra/dgx/whisper-server/app.py` (forced
`temperature=0.0` scalar disabled openai-whisper's built-in
fallback schedule, which is what saves long audio from
autoregressive runaway). Fixed in this PR. Re-run confirmed clean.

**DGX-over-Tailscale hang pattern**: The post-fix sweep hit
intermittent HTTP-response hangs (server returns 200 OK, body
stuck mid-transit over Tailscale). Worked around per-episode for
this eval. Filed as **#956** — applies to every DGX consumer
(whisper, pyannote, vLLM, future agents), not just whisper.

**Net effect on routing**:

- Laptop runs (`local.yaml`) → MPS openai-whisper. **No change.**
- DGX-equipped runs (`cloud_with_dgx_*`) → **route transcription to
  `whisper-openai` on `:8002` (the fixed container)**. Quality
  matches MPS within noise, ~3× faster realtime. This **flips** the
  earlier (pre-Cell-C) recommendation that suggested keeping
  transcription off the DGX.
- Cloud Whisper API stays the `cloud_*` default; not retested this
  batch.

Full report: `docs/guides/eval-reports/EVAL_TRANSCRIPTION_3WAY_2026_06.md`

## Recommended profile defaults (after this batch)

### `local.yaml` (laptop / privacy / airgapped) — **no change**

- Transcription: openai-whisper on MPS auto-select (already the case)
- Diarization: pyannote in-process on MPS auto-select (already the case)
- Summary: hermes3:8b on Ollama (#949 finale verdict, already the
  case)

Validated: this profile delivers usable end-to-end performance on
the operator's MacBook without any DGX dependency. Numbers:
~1.6× realtime whisper (5-ep mean; warm runs 2.6-3.8×), ~13×
realtime pyannote, ~50× realtime hermes3:8b summary. Bonus finding
this batch: openai-whisper is also 1.5× realtime on **pure CPU**
on M4 Pro, so MPS isn't load-bearing — the fallback path stays
production-viable.

### `cloud_with_dgx_*` (DGX-equipped, cost-conscious) — **transcription flips to DGX after the temperature-bug fix**

- Transcription: **openai-whisper at `dgx:8002`** (the fixed
  container, this PR). WER 0.102 mean (within noise of MPS 0.096),
  4.56× realtime mean — the fastest GPU-accelerated path
  available. `speaches/faster-whisper` at `dgx:8000` is **NOT**
  the path; that container has a separate unresolved bug (#952
  remains gated on the speaches root-cause).
- Diarization: pyannote at `dgx:8001` (`#926`)
- Summary: Ollama qwen3.5:35b at `dgx:11434`

**Operational caveats**:

- The earlier "GPU contention with vLLM degrades whisper" claim
  was wrong. The DGX whisper container produced byte-identical
  bad output regardless of vLLM running — it was a config bug in
  our `app.py` (forced `temperature=0.0` disabled openai-whisper's
  fallback schedule). Fix is in this PR. After the fix, no
  contention sensitivity observed.
- **DGX-over-Tailscale HTTP hangs (`#956`)**: long-blocking calls
  from the laptop to the DGX occasionally lose the response
  mid-transit (server returns 200 OK, body stuck). Reproducible.
  Affects every DGX client, not just whisper. Production
  consumers should design around this — either async job
  pattern, or the shared timeout/retry/keepalive layer landed
  in `#956`. For the eval, the workaround was per-call timeout
  combined with per-episode invocation.

### `cloud_*` (no DGX, cloud-OK) — **NOT retested this batch**

- Transcription: cloud Whisper API
- Diarization: `speaker_detector_provider: gemini` (current —
  unmeasured this batch since no Gemini speech provider exists in
  the repo; filed)
- Summary: gemini-2.5-flash-lite (per `#816`)

Out of scope for this PR. The cloud paths were validated previously
and this batch didn't relitigate them. They stay the documented
choice for users without local GPU.

## What this PR DOES change (after the late-batch findings)

- **`cloud_with_dgx_*` transcription default**: flips to
  `whisper-openai` on `dgx:8002` (the fixed container). Earlier
  drafts of this report had kept transcription off the DGX
  because of the temperature-bug results.
- **vLLM autoresearch default**: flips from
  `25.11-py3 + DeepSeek-R1-Distill-32B` to
  `26.05-py3 + Qwen3.6-35B-A3B + --max-num-seqs 128`. The Cell C
  comparison (this PR) showed Qwen3.6-served-by-vLLM ties Ollama-
  qwen3.5:35b within scoring noise. R1-Distill kept as a
  one-line revert (`docker-compose.yml.r1-distill.bak` preserved
  on the DGX).

## What this PR does NOT change

- `local.yaml` profile defaults — confirmed unchanged.
- `cloud_*` profiles — not relitigated; cloud Whisper API + Gemini
  flash-lite + `speaker_detector_provider: gemini` stay the
  documented choice.
- Ollama qwen3.5:35b remains the summary default for
  `cloud_with_dgx_*` (Cell C confirmed: Ollama and vLLM tie when
  serving the same model family; Ollama is operationally simpler).
- The faster-whisper container (`dgx:8000`, speaches image) stays
  out of the production routing — separate bug, separate fix.
  `#952` remains the engine-comparison gate, blocked on that fix.

## Follow-ups (filed)

| # | What | Why |
| --- | --- | --- |
| `#946` (in flight) | Whisper client resilience: duration-scaled timeout + single-flight | Should now consume `#956`'s shared resilience layer instead of bespoke patterns |
| `#952` | Validate faster-whisper WER vs openai-whisper on real podcasts | Blocked on speaches `compute_type=default` root-cause (the empty-output bug surfaced this batch) |
| `#953` | openai-whisper DGX service (deployed earlier, **temperature-schedule bug fixed this PR**) | The deploy was correct; the API contract had a bad default. App.py fix in this PR. |
| `#956` (filed this batch) | DGX-over-Tailscale client resilience (shared timeout/retry/keepalive layer for every DGX consumer) | Long-blocking HTTP over Tailscale loses responses mid-transit; hit during the post-fix #929 sweep |
| `#954` (filed today) | Diarization client resilience analog to `#946` | Same operational gap surfaced during this batch |
| `#934` | Distinct voices in v2 fixtures | Required to verdict pyannote speaker-count accuracy |
| #928 Cell D / E | Quant isolation (R1-Distill on Ollama Q4 GGUF, and Ollama-Qwen3.6 at bf16) | Cell C resolved the serving-stack question; remaining isolated variable is precision |
| Gemini speaker provider deploy | Wire `cloud_*`'s declared `speaker_detector_provider: gemini` so #930 can run the 3rd candidate | Unblocks full 3-candidate diarization championship |
| R1 reasoning-suppressed prompt | Modify the summary prompt to strip `<think>` and require summary-only output | Determines whether R1-Distill is competitive when not reasoning-as-output |

## Honest framing for the operator

This batch **moved the evidence base forward** more than it changed
recommendations. The DGX is now properly diagnosed for whisper
contention; the vLLM serving stack is live for future autoresearch
without redeploys; the summary champion is validated; the laptop
diarization story is documented. Where defaults stay put, it's
because the evidence either confirmed the current choice or wasn't
strong enough to flip it.

The "we keep options and explore" stance the user articulated mid-
batch is reflected in this synthesis: every profile shape has a
documented operating point with measured numbers, and no
recommendation says "kill the alternatives." That's the right shape
for a research output.

## Artifacts

- `docs/guides/eval-reports/EVAL_DIARIZATION_DGX_VS_CLOUD_2026_06.md`
- `docs/guides/eval-reports/EVAL_SUMMARY_DGX_LOCAL_2026_06.md`
- `docs/guides/eval-reports/EVAL_TRANSCRIPTION_3WAY_2026_06.md`
- `scripts/eval/score/diarization_dgx_vs_cloud_v1.py`
- `scripts/eval/score/whisper_dgx_vs_cloud_v1.py` (with `LOCAL_*_DEVICE` env overrides)
- `scripts/eval/score/summary_vllm_predict_v1.py`
- `infra/dgx/whisper-server/` (#953 deploy)
- `infra/dgx/vllm-autoresearch/` (#928 prereq deploy)

## References

- Issues: #931 (this synthesis), #928, #929, #930
- Parent epic: #927 (DGX-vs-cloud autoresearch programme)
- Prior local-LLM finale: #932 / #949
- Related infra: #953 (openai-whisper), #948 (speaches investigation),
  #946 (whisper client resilience), #954 (diarization client resilience),
  #952 (faster-whisper validation), #934 (distinct-voice fixtures)
