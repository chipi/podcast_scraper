# Next session plan — picking up after autoresearch batch-3 (PR #966 merged)

## Where you are now

- PR #966 merged on `main` (commit `458b4f53`) — autoresearch batch-3
  championships (#928 / #929 / #930 / #931), DGX whisper bug fix, vLLM
  default flip to `26.05-py3 + Qwen3.6-35B-A3B`.
- Autoresearch programme (epic #927) is essentially closed. What's left
  is operationalization + follow-up evals — all filed as issues.
- DGX in clean idle state: whisper-openai + pyannote warm; vLLM +
  faster-whisper stopped. ~12.9 GB / 122 GB GPU.

## The 9 open follow-up issues filed this batch

| # | Title | Rough scope | Blockers |
| --- | --- | --- | --- |
| **#956** | DGX-over-Tailscale client resilience (shared timeout/retry/keepalive layer for every DGX consumer) | Large — touches every DGX client | None |
| **#957** | speaches/faster-whisper container produces empty output 4/5 episodes | Small-medium — likely a `WHISPER__COMPUTE_TYPE` config fix | None |
| **#958** | #928 Cell D + Cell E (R1-Distill on Ollama Q4; Qwen3.6 on Ollama bf16) — closes the quantization isolation | Medium — download + 2 finale runs | None |
| **#959** | Real-podcast (90-min) WER + summary quality validation | Large — multi-hour eval | Better after #957 (4-way) + #960 (clean vLLM backend) |
| **#960** | vLLM as a first-class backend in `autoresearch_track_a.py` | Small-medium — extend `openai` backend with `base_url` override | None |
| **#961** | R1-Distill summary prompt — strip reasoning preamble | Small — prompt rewrite + one re-eval | None |
| **#962** | Deploy Gemini speaker-detector provider | Medium-large — new provider class + cloud_* wiring | None |
| **#963** | Re-test DGX whisper under concurrent vLLM contention now that the bug is fixed | Small — 15 min unattended re-run | None |
| **#964** | Wave audio hardening umbrella (Groups A-I from `docs/wip/AUDIO-WAVES-HARDENING-AUDIT.md`) | Large — multiple sub-PRs | Independent of autoresearch path |

Plus **#965** (test fixtures v2 rebuild) was filed and immediately
closed — wrong target; the real v3 audio piece is tracked by **#934**.

## Suggested orderings (pick whichever fits the energy)

### Option A — "quick wins first, then validation"

1. **#963** (~30 min) — re-test DGX whisper contention. Validates the
   temperature fix is robust under load. Either confirms "fix is
   complete" or surfaces a follow-up. Either result is small + useful.
2. **#957** (~half day) — speaches root cause. Cheapest unblocker
   for #952 and the proper 4-way story.
3. **#960** (~half day) — vLLM first-class backend. Cleans the
   `autoresearch_track_a.py` path so future sweeps don't need the
   standalone script workaround.
4. **#958** (~half day) — Cell D / Cell E. Closes the methodology
   matrix. Methodologically nicest with #960 already landed.
5. **#961** (~half day) — R1 prompt fix. Independent; do whenever.
6. **#959** (multi-day) — real-podcast validation. Best done AFTER
   #957 + #960 land so the 4-way is honest.
7. **#962** (multi-day) — Gemini speaker provider. Unblocks cloud_*
   diarization. Can run in parallel with anything.
8. **#956** (multi-day) — Tailscale resilience layer. Big surface
   area; consider whether the other agent on #946 picks this up.

### Option B — "biggest blockers first"

1. **#956** — every DGX client benefits. Bigger but earlier payoff
   in operational stability.
2. **#957** — once the resilience layer is in, also fix the speaches
   bug, then #952 unblocks.
3. **#960** — clean vLLM backend.
4. **#958, #959, #961, #962, #963** in any order.

### Option C — "step away from autoresearch, do production / hardening"

1. **#964** (Wave audio hardening, Groups A-I) — large independent
   chunk. Per the WIP audit doc.
2. Resume autoresearch follow-ups after that lands.

## Where to find the supporting context (for the next session)

- **Eval reports from this batch** (the full quantitative story):
  - `docs/guides/eval-reports/EVAL_DIARIZATION_DGX_VS_CLOUD_2026_06.md`
  - `docs/guides/eval-reports/EVAL_SUMMARY_DGX_LOCAL_2026_06.md`
  - `docs/guides/eval-reports/EVAL_TRANSCRIPTION_3WAY_2026_06.md`
  - `docs/guides/eval-reports/EVAL_HYBRID_ROUTING_2026_06.md`
- **v3 fixture learnings** (where future autoresearch work should
  contribute findings):
  - `docs/wip/AUTORESEARCH_LEARNINGS_FOR_V3.md` (Batch-3 section at
    the end has this batch's contributions)
- **vLLM service contract** (consumers must pass
  `chat_template_kwargs={enable_thinking: false}` for Qwen3 family):
  - `infra/dgx/vllm-autoresearch/README.md`
- **The rebase-before-push rule** (added this batch):
  - `AGENTS.md` "Always rebase before pushing a feature branch"
  - `.ai-coding-guidelines-quick.md` PR Push Workflow item 3
  - `.ai-coding-guidelines.md` PR Push Workflow item 7+8
  - Memory: `feedback_rebase_before_push.md`

## How to start the new session

Paste this into the new session:

> Pick up from `docs/wip/NEXT_SESSION_PLAN.md`. I want to discuss the
> ordering of the 9 open follow-up issues before we start any work.

The new session will load MEMORY.md automatically (which now has the
rebase rule + everything else from this session's feedback memories).
The plan file is small enough to read up-front and gives you the full
decision space without needing to dig through eval reports.
