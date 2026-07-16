# ADR-111: The eval and the pipeline are two toys built from ONE set of bricks

**Status:** Proposed
**Date:** 2026-07-14
**Deciders:** Marko
**Related:** ADR-110, ADR-109, #1169

## Context

The eval does not *use* the pipeline. It *re-implements* it.

`scripts/eval/experiment/run_experiment.py` builds its own `Config`, wires its own providers, and
calls `gi.pipeline.build_artifact` with its own arguments. `workflow/metadata_generation.py` does the
same thing, differently. Two assemblies of the same bricks, maintained by hand.

So every brick has to be plugged in **twice**, and each one that gets missed fails **silently** —
because the missing value falls back to a default, and the default is almost always "off".

Found in a single day, all of them this exact shape:

| brick | how the eval's copy diverged | how it failed |
| --- | --- | --- |
| diarized segments | never passed to `build_artifact` | every voice gate **dead in eval**, live in prod |
| `gi_max_insights` | eval 50, prod 12 | the head-to-head measured a cap prod never runs |
| value gate | eval **on**, prod **off** | the judge we spent two days fixing never ran in production |
| evidence align | eval had to **re-implement** it | `model_copy` skips validators, so a cell summarised with qwen and ground with the local stack — one run produced 513 insights and **zero** quotes |
| LLM grounding | prod declared `bundled` but ground with `transformers` | the bundled setting did nothing |

### The anti-pattern has a name: the allowlist

A `Config` is not built by validation. It is built by **hand-copying keys**, in at least three
separate places:

* `cli.py` — a literal tuple of ~32 field names forwarded from the profile YAML to `Config`. A key
  not on the list is dropped by `--config profile.yaml`.
* `evaluation/eval_gi_kg_runtime.py` — ~23 `p.get("...")` calls mapping an experiment's `params:`
  onto a `Config`. A key not mapped never reaches the run.
* the experiment YAML itself — a third vocabulary again.

A key nobody copies **does not error**. It silently takes its default. `gi_value_gate_enabled`
defaults to `False`, so forgetting to forward it does not break a run — it turns the judge off, and
the run looks fine.

Every one of the failures above is the same bug, in a different allowlist.

## Decision

**One brick, two toys.**

1. **ONE assembly function.** `gi.build_episode_artifact(cfg, inputs) -> artifact` is the only way an
   episode becomes an artifact. `metadata_generation.py` calls it. `run_experiment.py` calls it.
   Neither re-wires providers, gates, segments or grounding. If a stage needs a new input, it is
   added to `inputs` once and both callers get it — or neither does.

2. **ONE config resolver.** A `Config` is produced by **validation**, never by hand-copied keys:

   ```python
   resolve_config(profile=..., yaml=..., overrides={...}) -> Config
   ```

   It builds the full mapping and hands it to `Config.model_validate`, so every validator runs —
   including the evidence auto-align that `model_copy` skips. **Delete all three allowlists.** An
   unknown key becomes a loud error, not a silent default.

3. **The eval's `params:` are Config overrides, not a private vocabulary.** An experiment YAML names
   real `Config` fields. No translation layer, so no key can be lost in translation.

4. **A parity assertion that reads the real thing.** `TestEvalMatchesProduction` compares *configs*
   today, which is why it never noticed that the eval passed no `transcript_segments` at all. It must
   assert on the resolved **inputs to `build_episode_artifact`**, not on a config dict.

## Alternatives considered

### A. Keep two assemblies, add tests that they agree

This is what we have. `TestEvalMatchesProduction` exists and passed all day while the eval ran a
pipeline with every voice gate dead. A test that compares two hand-maintained lists cannot notice a
brick that is missing from **both** lists — and the failure mode is a silent default, not a crash.
**Rejected: it has already been tried and it already failed.**

### B. Make the eval call the production workflow end-to-end

Tempting, and wrong. The eval must be able to hold the transcript fixed and vary only the model —
that is the whole point of a controlled arm. Running the full workflow would re-transcribe and
re-diarize, which is both expensive and the thing we are deliberately holding constant.
**Rejected: the eval needs the same BRICKS, not the same TOY.**

### C. A shared "GI runtime" object that both construct

A third thing to keep in sync with two others. **Rejected.**

## Consequences

**Good.** A stage cannot be live in one and dead in the other, because there is only one wiring. A
config key cannot be silently dropped, because nothing copies keys. The eval measures the product.

**Cost.** `run_experiment.py` is old and load-bearing; the GI branch alone is ~200 lines of bespoke
wiring. This is a refactor with real risk, and it must land behind the existing eval fixtures
(`test_run_experiment_gi_kg_stub`) with before/after artifacts compared on a frozen dataset.

**Not addressed here.** The KG branch of `run_experiment.py` has the same shape and almost certainly
the same bugs — it was never audited. It should be, and it should land in the same refactor rather
than becoming the fourth allowlist.
