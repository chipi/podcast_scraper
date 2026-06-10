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
| `cloud_with_dgx_*` (DGX-equipped) | openai-whisper / `:8002` (with operational caveat) | pyannote / `:8001` | Ollama qwen3.5:35b on `:11434` | None — confirmed |
| `cloud_*` (no DGX, cloud-OK) | cloud Whisper API | Gemini speech (current — unmeasured this batch) | gemini-2.5-flash-lite | None — not re-litigated |

The "no change" net is honest: the **evidence base under each
profile improved**, but no profile-default shifts. That's the right
outcome for a research batch — measure what's there, document the
operating points, change defaults only when evidence demands it.

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

**Methodology limitation made explicit**: the eval compared
combinations, not isolated variables. The cell-C cleanup
(`Qwen/Qwen3.6-35B-A3B` served by vLLM at bf16, head-to-head against
both Ollama-Qwen3.6 and vLLM-R1) is filed as the proper-isolation
follow-up. Download is in flight on the DGX at this writing.

**Net effect on routing**: keep Ollama qwen3.5:35b as the
`cloud_with_dgx_*` summary default. The follow-up could shift this
in either direction.

Full report: `docs/guides/eval-reports/EVAL_SUMMARY_DGX_LOCAL_2026_06.md`

### Transcription (#929 partial)

**3-way: openai-whisper on Apple MPS vs DGX-CUDA `:8002` vs pure CPU.**

- MPS clean: 1.6× realtime mean (with 13-min cold start dragging
  the mean; warm runs are 2.6-3.8× realtime), WER 0.10 mean on v2
  fixtures.
- DGX CUDA under concurrent vLLM load: WER spiked to 8.21 on one
  episode (13,136 hyp words for a 1,519-word reference — model
  started hallucinating after the first segment, consistent with
  CPU-fallback under GPU pressure / the same auto-unload-then-reload
  regression as `#948`'s speaches).
- Pure CPU (1 episode): 1.5× realtime, WER 0.109 — basically tied
  with MPS on warm runs. **MPS only ~1.8× faster than CPU on warm,
  not the 17× extrapolated from diarization.** Means CPU fallback
  for laptop `local.yaml` is production-viable when MPS is busy.

**Speaches/faster-whisper engine comparison stays out of scope**
per `#952` — that gate is what unlocks the proper 4-way (MPS vs
CUDA vs CPU vs ctranslate2-CUDA) eval.

**Net effect on routing**:

- Laptop runs → MPS openai-whisper. Stable, fast, simple.
- DGX runs → openai-whisper on `:8002` **with the operational caveat**
  that GPU contention with concurrent vLLM produces hallucinations.
  Either dedicate DGX to whisper during transcription windows, or
  add single-flight + duration-scaled timeout (`#946` pattern) to
  the client. Filed as a follow-up.
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

### `cloud_with_dgx_*` (DGX-equipped, cost-conscious) — **no change but caveat documented**

- Transcription: openai-whisper at `dgx:8002` (or speaches at
  `dgx:8000` when `#952` validates it)
- Diarization: pyannote at `dgx:8001` (`#926`)
- Summary: Ollama qwen3.5:35b at `dgx:11434`

Validated for non-contended GPU use. **Operational caveat for prod**:
running heavy concurrent vLLM autoresearch workloads on the same
GB10 degrades the whisper container quality (per #929 partial). Either
schedule transcription windows when vLLM is idle, or apply the
single-flight + duration-scaled-timeout pattern from `#946`.

### `cloud_*` (no DGX, cloud-OK) — **NOT retested this batch**

- Transcription: cloud Whisper API
- Diarization: `speaker_detector_provider: gemini` (current —
  unmeasured this batch since no Gemini speech provider exists in
  the repo; filed)
- Summary: gemini-2.5-flash-lite (per `#816`)

Out of scope for this PR. The cloud paths were validated previously
and this batch didn't relitigate them. They stay the documented
choice for users without local GPU.

## What this PR DOES NOT change

- No prod profile defaults flip.
- `cloud_*` profiles aren't relitigated.
- The faster-whisper-vs-openai-whisper engine question stays at `#952`.

## Follow-ups (filed)

| # | What | Why |
| --- | --- | --- |
| `#946` (in flight) | Whisper client resilience: duration-scaled timeout + single-flight | Required for DGX whisper to be production-grade under shared-GPU load |
| `#952` | Validate faster-whisper WER vs openai-whisper on real podcasts | Unblocks the 4-way transcription comparison |
| `#953` | openai-whisper DGX service (delivered this PR) | — |
| `#954` (filed today) | Diarization client resilience analog to `#946` | Same operational gap surfaced during this batch |
| `#934` | Distinct voices in v2 fixtures | Required to verdict pyannote speaker-count accuracy |
| #928 isolation follow-up | Cell C re-eval: Qwen3.6-35B-A3B on vLLM at bf16, vs same on Ollama at Q4 | Proper variable isolation for the serving-stack question. Download in flight. |
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
