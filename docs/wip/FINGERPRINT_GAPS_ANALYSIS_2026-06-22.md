# Fingerprint gaps analysis (2026-06-22)

> **STATUS: CLOSED.** All 7 gap-closure commits landed in
> PR #1036 (`b1ef7046..a31b4e9a`): generation_params for GI/KG,
> backing_model_id + base_url, task_pipeline + inference_args/image,
> runtime.inference_target, podcast_scraper_config, dataset_content_hash
> (replaces the original "upstream summary provenance" framing with a
> stronger general dataset hash), and fingerprint_version 2.0 + full-dict
> hash. Tracking issue #1074 filed and closed the same day after audit
> confirmed pre-merged closure. Retro tool at
> `scripts/eval/fingerprint/refingerprint_from_run.py` handles old
> artifacts.

Operator-flagged concern after RFC-097 v2 merge (#1039): *"we did not
notice when we did GI/KG from bullets vs summary, and also I think we
need to fingerprint models with all their env setup as part of it
since each param change is a meaningful change possibly."*

This doc audits today's fingerprint generation, identifies concrete
gaps that allow materially different runs to look identical, and
proposes a work batch to close them.

## What today's fingerprint captures

Today's `fingerprint.json` (per `data/eval/runs/<run>/fingerprint.json`):

- `run_context`: baseline_id, dataset_id, git commit/branch/dirty
- `provider`: provider_type, library, library_version
- `pipeline.stages.main.model`:
  `provider_type`, `framework`, **`model_name`** (e.g. `"autoresearch"`,
  the vLLM served-model-name alias), `model_revision`,
  `tokenizer_name`, `tokenizer_revision`, `endpoint`
- `pipeline.stages.main.generation_params`: dict
- `preprocessing`: profile_id + version + per-step booleans
- `chunking`: strategy, word/token sizes, overlap, boundary
- `environment`: python_version, os
- `runtime`: device, device_name, backend, inference_backend
- `prompts.user` + `prompts.system`: name + file + **sha256** + params

That's a solid skeleton. The gaps are real and significant.

## Headline bug — `generation_params: {}` for GI/KG

`scripts/eval/data/materialize_baseline.py:833-836`:

```python
elif experiment_config.task in ("grounded_insights", "knowledge_graph"):
    generation_params = {}
    map_generation_params = {}
    reduce_generation_params = {}
```

**For every GI and KG run we have ever produced, the fingerprint
contains an empty generation_params block.** Magistral KG at temp=0.7,
top_p=0.95, max_length=4096 has the same `generation_params: {}` as
Qwen3.5-35B KG at temp=0.0, top_p=1.0, max_length=800. They are
genuinely different experiments; their fingerprints say otherwise.

The fix is a one-block change to mirror the
`(anthropic, mistral, grok, deepseek, gemini, ollama)` branch
(line 851-857) for GI/KG too.

## Backing-model identity hidden behind `served-model-name: autoresearch`

`fingerprint.pipeline.stages.main.model.model_name = "autoresearch"`
on every vLLM run. That's the **alias** the container declares; the
actual backing model (`Qwen/Qwen3.5-35B-A3B`,
`mistralai/Magistral-Small-2509`, `NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4`,
etc.) only lives in the `vllm serve` command flag and the homelab
compose. **Two fingerprints labeled `model: "autoresearch"` may be
against entirely different LLMs.**

The fix: when `backend.base_url` points at a DGX vLLM, call
`GET <base_url>/v1/models` at run start and capture the `root` field
of the returned model entry (which IS the HF id). Add that as
`pipeline.stages.main.model.backing_model_id`.

## vLLM server flags absent

The chunk-7 workaround sweep proved that these flags are the
difference between a model running at 5 tok/s and 50 tok/s, or
between booting at all and failing in <10s:

- `--gpu-memory-utilization`
- `--max-model-len`
- `--max-num-batched-tokens`
- `--max-num-seqs`
- `--enforce-eager`
- `--reasoning-parser` (mistral / qwen3 / deepseek_r1)
- `--tokenizer-mode mistral --config-format mistral --load-format mistral`
- `--language-model-only`
- `--tool-call-parser`
- `--trust-remote-code`

**None of these are in any fingerprint.** Two runs against
`autoresearch` could have used wildly different vLLM configs and you
can't tell from `fingerprint.json`.

The fix: when `backend.base_url` is a vLLM endpoint, capture the
served container's command-line args. vLLM exposes
`GET /v1/configs` (or equivalent — needs verification) or the
container `Cmd` field can be SSH'd. Either way, the flags landed on
the server become part of `pipeline.stages.main.vllm_flags`.

## Container image version absent

Chunks 1-7 saw multimodal/MoE models behave differently on
`nvcr.io/nvidia/vllm:26.05-py3` vs the old `25.11-py3` image. The
fingerprint records neither.

The fix: add `runtime.inference_image` populated from the same
SSH/inspect probe.

## Postprocessor + `*_extraction_src` absent

Eval YAML configs commonly carry:

- `prompts.postprocessor` — e.g. `strip_r1_reasoning`,
  `decode_r1_byte_level`. Each rewrites the model output before
  scoring; different postprocessors → genuinely different artifacts.
- `params.kg_extraction_src` / `params.gi_insight_src` — `provider`
  vs `eval_stub`. The stub branch bypasses the LLM entirely. Two runs
  with stub vs provider would have identical fingerprints today.
- `params.gi_max_insights` — caps emitted insights per episode.
  Affects coverage scores. Absent.

The fix: add a `task_pipeline` section that captures these per-task
knobs from `experiment_config.params` + `experiment_config.prompts`.

## Bullets vs paragraph — provenance of upstream summary not threaded

When a GI or KG run uses a summary as input, today's fingerprint
captures the GI/KG prompt + the GI/KG generation params (well, it
WILL once the empty-dict bug is fixed) — but NOT the identity of the
upstream summary it was fed. Two GI runs that look identical in
fingerprint could have been fed `bullets_json_v1` summaries OR
`long_v1` paragraph summaries; both `summarization` tasks; user can't
tell from the GI fingerprint.

The fix: add `upstream` section to the GI/KG fingerprint linking the
upstream summary run's fingerprint hash + the summary file's sha256.
Lifecycle: when the experiment runner consumes a summary, it logs the
upstream provenance into the GI/KG fingerprint at run start.

## `endpoint` + `base_url` mismatch hides target DGX

`fingerprint.pipeline.stages.main.model.endpoint` is recorded as
`"chat.completions"` for vLLM. But the actual `base_url` is
`http://dgx-llm-1.tail6d0ed4.ts.net:8003/v1` (or a different DGX if
the operator runs multiple). Different target servers can carry
different models; the fingerprint doesn't disambiguate.

The fix: capture `pipeline.stages.main.model.base_url` verbatim.

## Generic device confusion: laptop vs DGX

`runtime.device: "mps"` records the LAPTOP that issued the API call.
The inference ran on DGX-GB10. Two runs from different laptops would
look like different devices despite identical inference. Two runs
from the same laptop hitting different DGX boxes would look like
identical devices.

The fix: split `runtime.driver_device` (where the orchestrator ran)
from `runtime.inference_device` (the actual GPU that produced the
tokens, from the vLLM `/metrics` GPU info).

## Library version coverage

We capture `provider.provider_library_version` (e.g. `openai 2.15.0`).
We DON'T capture the **`vllm` server** version, **`torch`** on the
inference server, **flashinfer/triton** kernel versions, **CUDA
driver** version. Phase 2c had a documented case where a `torch.compile`
fix landed in a newer image and changed behavior.

The fix: capture vLLM's `GET /metrics` or `GET /version` output and
fold the server-side stack versions into `runtime.inference_stack`.

## Hash should reflect everything

Today's `compute_hash()` in `ProviderFingerprint` derives from
`model_name + version + hash + device + device_name + precision +
package + commit`. It does NOT include generation_params, prompts,
preprocessing profile, chunking, postprocessor, vLLM flags, backing
model — i.e. the things actually being audited.

The fix: rebuild the hash function to walk the entire fingerprint
dict (sorted JSON), so any difference anywhere produces a different
hash. This is also a contract: changing the hash function bumps
`fingerprint_version`.

## Proposed work batch (7 commits, bisectable)

1. **Fix the `generation_params: {}` bug for GI/KG** — one block in
   `materialize_baseline.py`; mirror the `(anthropic, mistral, ...)`
   branch. Lowest-effort, biggest visibility fix. Backfill: stamp a
   note in existing fingerprints' README so old artifacts are
   recognized as having the gap.
2. **Capture backing model id + base_url** — when `backend.base_url`
   is set, probe `GET /v1/models` at run start; record `root` as
   `backing_model_id`; record `base_url` verbatim.
3. **Capture vLLM server flags** — SSH or HTTP probe to inspect the
   `vllm serve` cmdline; record under `runtime.inference_args`.
4. **Capture container image + inference stack** — `nvcr.io/...vllm:X`,
   plus `torch` / `flashinfer` / kernel versions from the server.
5. **Capture task-pipeline knobs** — postprocessor, `kg_extraction_src`,
   `gi_insight_src`, `gi_max_insights`, plus any reasoning-parser /
   prompt-format flags wired via `params`.
6. **Thread upstream summary provenance to GI/KG fingerprints** — link
   the upstream summary run's `fingerprint_hash` + summary file sha256
   under `upstream` in GI/KG fingerprints.
7. **Strengthen `compute_hash`** to walk the full fingerprint dict.
   Bump `fingerprint_version` to `2.0`. Update any pinning / equality
   tests.

## Test plan

- New unit tests per gap: assert each newly-captured field is
  populated for a representative (KG, GI, summarization) pair.
- Equality regression: assert two intentionally-different runs (e.g.
  temp=0.0 vs temp=0.7 on the same model) produce different hashes.
- Backward-compat: keep `fingerprint_version: "1.0"` reads working
  for old artifacts (read-only). New writes are `"2.0"`.

## Out of scope (defer)

- Re-fingerprinting historical runs. Old artifacts are old; we don't
  retroactively edit them. The `EVAL_1016_*` reports already document
  which configs each run was under.
- Building a UI to diff two fingerprints. CLI `diff` works fine.
- Hardware fingerprinting beyond device name (no GPU serial, no
  thermal state).

## Why this matters now (before DeepSeek decision)

Operator's pending decision on DeepSeek-V2-Lite-Chat (cohort floor:
keep or drop) is exactly the kind of choice that hinges on whether
two fingerprints agree. If DeepSeek's apparent floor performance is
partly a config artifact (the BPE postprocessor wiring gap was
exactly this — `decode_r1_byte_level` was applied to summary text but
NOT to GI/KG node.label per `EVAL_1016_FINAL_REPORT_2026_06_17.md`),
we want fingerprints precise enough to surface that. With today's
gaps, "DeepSeek scored 1.5%" could mean "the model is weak", or "the
postprocessor was misconfigured for this task", or "the prompt
shifted between runs without us noticing." Closing the fingerprint
gaps gives us the tool to tell those apart.
