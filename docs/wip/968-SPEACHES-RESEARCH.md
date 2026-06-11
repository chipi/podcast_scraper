# #968 deep-research findings — speaches/faster-whisper on GB10 Blackwell

**Date:** 2026-06-11
**Status:** research only, no code changes yet
**Workflow run id:** `wf_922f1cd4-43a` (106 sub-agents, 5 search angles, adversarial-verified)

The four investigation threads from #968 are not equally promising. After
fan-out search + adversarial verification, here's the ranking with concrete
next actions.

---

## TL;DR — what to actually do, in priority order

1. **Thread B (highest leverage)** — patch speaches' router to pass a
   tuple temperature `(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)` instead of the
   scalar `0.0`. **This is the same bug class we fixed in our sibling
   openai-whisper container (#929).** Confirmed by reading the upstream
   speaches source. Small patch; likely unlocks 30×-ish quality on the
   pathological episodes.
2. **Thread C (high leverage, low effort)** — pin faster-whisper >= the
   PR #1297 merge (2025-08-06) that fixed batched-VAD silence retention.
   Check what speaches:latest-cuda bundles; upgrade if older. Test
   `WHISPER__VAD_FILTER=false` as an A/B isolation.
3. **Thread A (bad news, needs upstream)** — GB10 is `sm_121`, not
   `sm_100` (DGX/B200) or `sm_120` (RTX 50). CTranslate2 has zero
   `sm_121` enablement through v4.8.0. Bumping ctranslate2 alone WILL
   NOT unlock bf16/fp16 on our hardware. The realistic path is:
   - File a CTranslate2 issue requesting sm_121 enablement (template:
     existing PR #1937 for sm_120).
   - Until that lands, **only int8 and float32 are safe on GB10**.
   - openai-whisper via torch stays the production transcription path
     (it doesn't go through ctranslate2's stricter "is X efficient on
     this SM" gate).
4. **Thread D (skip)** — `Systran/faster-whisper-large-v3` is produced
   by a documented standard conversion; no quality regression vs.
   self-conversion has been reported. Not worth the effort.

---

## Thread A — CTranslate2 Blackwell support status

### TL;DR
**CTranslate2 has NO explicit GB10/sm_121 support through v4.8.0
(2026-06-06).** The only Blackwell change is v4.6.2 PR #1937 which
DISABLES INT8 on sm_120 (consumer RTX 50-series) with a float16
fallback. v4.6.3 (2026-01-06) adds CUDA 12.8 build support but no
per-SM compute-path enablement followed.

### Evidence
- v4.6.2 (2025-12-05): `Disable INT8 for sm120 - Blackwell GPUs (#1937)`.
- v4.6.3 (2026-01-06): `Support for CUDA 12.8 (#1937, #1940)`.
- v4.8.0 (2026-06-06): zero Blackwell/SM_100/SM_120/SM_121/GB10/GB200/B100/B200 references.
- PR #1937 gate is `device_prop.major == 12 && device_prop.minor == 0`
  — **GB10 is `sm_121` so this gate does NOT fire on our hardware.**
- DGX Spark / GB10 confirmed as compute capability 12.1 / sm_121
  (NVIDIA Developer Forums). Architecturally distinct from sm_100
  (B200/GB200) and sm_120 (RTX 50).
- sm_121 needs an explicit code path — sm_100 support would NOT
  automatically cover GB10.

### The runtime error explained
`ValueError: ... target device or backend do not support efficient X
computation` is a STRICTER explicit-request gate than the documented
silent compute-type fallback table. Explicit
`WHISPER__COMPUTE_TYPE=float16/bfloat16/int8_bfloat16` raises
ValueError; `default`/inferred compute types silently fall back — and
on GB10 fall back to a broken path producing empty/runaway transcripts.

### Recommended next action
- File an issue on `OpenNMT/CTranslate2` requesting sm_121 enablement,
  templated after PR #1937. Likely a multi-week wait.
- Accept that on GB10/sm_121, only `int8` and `float32` are currently
  safe. bf16/fp16/int8_bf16 will keep ValueErroring until upstream
  ships sm_121 in the `gpu_supports_*` checks.
- openai-whisper via torch remains the production path (different
  efficiency gate, different code).

### Sources
- https://github.com/OpenNMT/CTranslate2/blob/master/CHANGELOG.md
- https://github.com/OpenNMT/CTranslate2/pull/1937
- https://github.com/OpenNMT/CTranslate2/issues/1865
- https://forums.developer.nvidia.com/t/dgx-spark-gb10-cuda-compute-capability
- https://github.com/vllm-project/vllm/issues/36821
- https://github.com/vllm-project/vllm/issues/43906
- https://developer.nvidia.com/blog/cuda-toolkit-12-8-delivers-nvidia-blackwell-support/

---

## Thread B — speaches temperature handling (HIGHEST LEVERAGE)

### TL;DR
**Confirmed.** speaches passes a SCALAR temperature (default `0.0`)
end-to-end into faster-whisper, disabling the documented
`(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)` compression-ratio/logprob rescue path.
**This is the exact #929-class bug we already fixed in our sibling
openai-whisper service.**

### Evidence (verbatim source read)
- `src/speaches/routers/stt.py`: declares
  `temperature: Annotated[float, Form()] = 0.0` (scalar) at both
  `transcribe_file` and `translate_file`. Forwards unchanged into
  `TranscriptionRequest(... temperature=temperature ...)`.
- `src/speaches/executors/whisper.py`: 3 call sites (non-streaming
  L232-245, streaming L258-269, translation) all call
  `whisper_model.transcribe(..., temperature=request.temperature, ...)`
  with no tuple wrapping or retry logic.
- `faster_whisper/transcribe.py` default:
  `Union[float, List[float], Tuple[float, ...]] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`
- Internal normalization:
  `temperatures = (temperature if isinstance(temperature, (list, tuple)) else [temperature])`
  → a scalar `0.0` becomes `[0.0]` with NO fallback.
- Docstring: the tuple is "successively used upon failures according
  to either `compression_ratio_threshold` or `log_prob_threshold`" —
  the rescue path for autoregressive runaway.

### Why this maps to OUR symptoms
- Whisper large-v3 with deterministic `temperature=0.0` is known to
  enter repetitive non-speech-token loops, producing either empty
  transcripts (post-processor filters non-speech tokens) or runaway
  hallucinations.
- **Precisely matches our observed 4/5 empty + 1 hallucinated 11k-word
  episode pattern.**
- Compression ratios documented at 14.17 / 11.58 (vs ~2.0 normal) in
  upstream Whisper discussion #2420.

### Recommended next action
Patch speaches to construct a tuple before calling transcribe. Two
options:

**Option 1 — fork-and-patch (fastest)**
Build our own speaches image with `routers/stt.py` patched:
```python
temperatures = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) if temperature == 0.0 else temperature
# Then forward `temperatures` into transcribe()
```
Ship in `infra/dgx/converge/deploy.py` as a build step or
volume-mount the patched router.

**Option 2 — upstream PR**
File against `speaches-ai/speaches` mirroring our #929 fix in
`infra/dgx/whisper-server/app.py` (commit `6df41b31`). Likely
accepted; the fix is small and the bug is well-known. Higher latency
to land.

### Sources
- https://github.com/speaches-ai/speaches/blob/master/src/speaches/routers/stt.py
- https://github.com/speaches-ai/speaches/blob/master/src/speaches/executors/whisper.py
- https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py
- https://github.com/openai/whisper/discussions/2420
- https://github.com/openai/whisper/discussions/679
- https://arxiv.org/abs/2501.11378 (Calm-Whisper)
- https://arxiv.org/abs/2505.12969

---

## Thread C — Silero VAD + int8 (HIGH LEVERAGE)

### TL;DR
**Pre-PR #1297 (merged 2025-08-06, MahmoudAshraf97), batched-pipeline
VAD only segmented and left inter-segment silence in the 30s GPU
input chunks**, which the PR author explicitly describes as "prone to
hallucinations." Plausibly compounds with Thread B's temperature bug
to produce the bimodal episode-level WER (0.053 / 0.595) on int8.

### Evidence (verbatim PR description)
> "The current VAD implementation in batched transcription is only used
> for segmentation, silence is only removed at segments boundaries, for
> example, if we have a speech segment from 1 to 3 and another from 9
> to 10, the resulting segment will be from 1 to 10, including a large
> silence period from 3 to 9 which is prone to hallucinations."

### Why this matters
Long inter-segment silence + scalar `temperature=0.0` + int8 weight
noise on an un-enumerated SM is a **triple failure mode.** The
episodes that landed cleanly (p01_e01 = WER 0.053) may simply have
denser speech distribution; the bad ones (p05_e01 = WER 0.595) may
have music-led intros / long pauses that hit the silence-retention
bug.

### Recommended next action
- Identify the faster-whisper version bundled in
  `ghcr.io/speaches-ai/speaches:latest-cuda` (exec into container:
  `pip show faster-whisper`).
- If `< 1.1.x` (i.e., pre PR #1297 merge of 2025-08-06), upgrade.
- A/B test `WHISPER__VAD_FILTER=false` to isolate VAD contribution
  from temperature contribution. Order:
  1. Fix Thread B (temperature) first.
  2. Re-test p05_e01 — if WER drops below 0.20, ship.
  3. If still bimodal, then chase Thread C (VAD config / version).

### Sources
- https://github.com/SYSTRAN/faster-whisper/pull/1297

---

## Thread D — Self-converting whisper-large-v3 (LOW YIELD, SKIP)

### TL;DR
The standard `Systran/faster-whisper-large-v3` weights were converted
with `ct2-transformers-converter --quantization float16` (documented
on the model card). **No verified quality regression vs.
self-conversion** surfaced in any of the research. Lowest-yield thread.

### Recommended next action
Skip. Revisit only if Threads B + C don't close the WER gap.

### Sources
- https://huggingface.co/Systran/faster-whisper-large-v3

---

## Bonus — Blackwell community recipes

### TL;DR
**No public recipe exists for a working speaches/faster-whisper-server
stack on GB10/DGX Spark or any Blackwell datacenter SKU.** Community
is actively requesting sm_121 enablement across the ML stack (vLLM,
flash-attention, CUTLASS).

### Adversarial finding
The agent initially surfaced a "Mekopa/whisperx-blackwell" project as
a candidate recipe; adversarial verification (3-vote) refuted its
existence (0-3 vote). Real result: no such project.

### Recommended path on GB10 today
1. Keep using torch-based openai-whisper (already at 4.6× realtime,
   WER 0.102) for production.
2. After Thread B + C fixes, faster-whisper at int8 should reach
   parity-ish quality but stay slower (likely 1.5-2× realtime).
3. File the CTranslate2 sm_121 enablement issue (Thread A) and wait.
4. Don't promote speaches above openai-whisper in production routing
   until both quality variance AND speed gap close.

### Sources
- https://github.com/vllm-project/vllm/issues/36821
- https://github.com/vllm-project/vllm/issues/43906
- https://github.com/Dao-AILab/flash-attention/issues/1969
- https://github.com/bidual/awesome-dgx-spark

---

## What this means for #968's acceptance bar

#968 set: mean WER ≤ 0.15, ≥3× realtime, max ep WER ≤ 2× mean.

Realistic projection after the research:

- **Thread B (temperature fix) alone** likely closes the quality
  variance — moves p05 from 0.595 to something sub-0.10. Mean WER
  probably lands in the 0.05-0.10 range, hitting the WER bar.
- **Thread C (VAD upgrade)** adds robustness to harder audio shapes.
  Belt-and-braces.
- **Speed bar (≥3× realtime) is the harder one** — without bf16
  unlocking via CTranslate2 sm_121 support, we're stuck at int8's
  ~1.5× realtime. Thread A is the gate; until upstream lands sm_121
  enablement, the speed bar is unreachable.
- **Practical reframe**: ship Thread B + C in this PR family, declare
  partial-acceptance (quality bar met, speed bar parked on Thread A),
  keep openai-whisper as production primary.
