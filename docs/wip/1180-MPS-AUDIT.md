# MPS exclusive mode audit (#1180 follow-up)

**Question:** does `should_serialize_mps` fire when it doesn't need to, silently
disabling audio↔LLM overlap?

**Verdict:** the structure is correct. One real edge case — the
`torch.backends.mps.is_available()` fallback in `_both_providers_use_mps` —
can over-serialize on Apple Silicon when the summary provider's models are
unloaded at check time. Made observable via a `WARNING` line rather than
removed, because removing it would trade "silent under-overlap" for
"occasional MPS OOM crashes." Fix landed in this branch.

## What the code does

`workflow/orchestration.py`:

```
should_serialize_mps = False
if cfg.mps_exclusive:                            # default True
    should_serialize_mps = _both_providers_use_mps(cfg, tx_prov, sum_prov)
```

`_both_providers_use_mps` returns True only when BOTH stages resolve to MPS.
For each stage, it checks in order:

1. Stage-level device override (`cfg.transcription_device` /
   `cfg.summarization_device`) — precise.
2. Provider auto-detect (`_detect_whisper_device()` / `_map_model.device`) —
   precise when models are loaded.
3. `cfg.summary_device` fallback (summarization only).
4. `torch.backends.mps.is_available()` fallback (summarization only) — the
   only step that can lie.

## The edge case (only one silent failure mode found)

**Setup:**

- `cfg.summarization_device` unset.
- `cfg.summary_device` unset.
- Summary provider is a local ML provider whose `_map_model` and
  `_reduce_model` are `None` / unloaded at check time (models load lazily
  from many code paths).
- Host is Apple Silicon (`torch.backends.mps.is_available() == True`).
- Transcription resolves to MPS (Whisper on MPS).

**Old behavior:** `summarization_uses_mps = True` via the fallback →
`should_serialize_mps = True` → the processing thread waits for ALL
transcription to complete before starting LLM work → operator sees a low
`processing_overlap_ratio` with no log line explaining why.

**Why the fallback exists:** if the provider then actually resolves to MPS
(the common case on Apple Silicon), disabling serialization would risk
MPS OOM crashes when Whisper and the summarizer are both live. The fallback
is deliberately safe.

**Why it's a problem:** on a run where the provider will resolve to CPU
(explicit config in a profile the check can't see, low-memory hardware, or
future changes to MLProvider defaults), we over-serialize for no benefit and
the operator has no signal.

## Fix landed in this branch

The two fallback branches (main path and exception path) in
`_both_providers_use_mps` now emit a `WARNING` line whenever they resort to
`torch.backends.mps.is_available()`:

```
MPS-exclusive check: summarization provider models unloaded at setup time;
falling back to torch.backends.mps.is_available()=True. If the provider
will actually run on CPU, this over-serializes and disables audio↔LLM overlap.
Set cfg.summarization_device or cfg.summary_device explicitly, or
cfg.mps_exclusive=False, to override. See #1180 audit.
```

Operators seeing a low `processing_overlap_ratio` on a Mac can now grep for
`MPS-exclusive check:` in the run log and take action.

New test: `TestMPSExclusiveFallbackWarning::test_fallback_warns_when_summary_models_unloaded`
in `tests/integration/workflow/test_mps_exclusive_integration.py`. Asserts
the warning fires when MPS is available; asserts a clean False otherwise so
the test runs on both Mac and Linux CI.

## What was NOT changed (and why)

- **The fallback itself.** Removing it would trade "silent over-serialize"
  for "occasional MPS OOM crash." Safe > overlap-optimal.
- **Transcription-side fallback.** The transcription check errs toward
  "assume not MPS" on failure (favoring overlap). Different asymmetry but
  compatible with the operator's expectation.
- **`cfg.mps_exclusive` default.** Stays True. Opt-in "no I know my provider
  is MPS-safe" is the right knob to expose to operators; changing the
  default would break configs that rely on the safety.

## Non-goals

- Not changing default parallelism knobs.
- Not touching the transcription-side MPS check (its fallback is already
  compatible with the "err toward overlap" side of the trade).
- Not extending the audit to CUDA — CUDA contention is a separate concern
  the current code does not model.
