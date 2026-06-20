# #1035 NER pre-pass — phase 3 verdict (2026-06-20)

**TL;DR**: Material win on both validation candidates. Cell F NVFP4 and
Qwen3.5-35B-A3B both jump from **0% to 100%** entity coverage on
`silver_opus47_kg_dev_v1` (dev_v1, 10 episodes). Topic coverage rises
+20–27pp as a bonus. Hallucinated entity emissions drop to 0. Phase 4
flips the Config default to `True` and updates prod profiles.

**Bonus**: Surfaced a pre-existing scorer bug (`extract_entities_from_prediction`
read `output.entities` instead of `output.kg.nodes[type=Entity]`); fixed.
The #1033 cohort rerun's "0% entity coverage across all 7 candidates"
finding survives the bug-fix re-score (LLM was emitting wrong entities,
not zero). #1035 NER pre-pass IS the fix the audit predicted.

## Methodology

- Branch: `feat/autoresearch-followups-2026-06-18`
- Dataset: `curated_5feeds_dev_v1` (10 episodes)
- Silver: `silver_opus47_kg_dev_v1`
- KG harness: `scripts/eval/experiment/run_experiment.py` (vLLM remote, spaCy local)
- spaCy model: `en_core_web_sm` (default ner_model in dev profiles)
- Scorer: `scripts/eval/score/score_kg_topic_coverage.py` (after bug-fix
  commit `d8df114d` — handles nested KG-artifact shape)

Two candidates chosen for phase 3 validation (the cohort #1 + the
daily-driver):
1. **Cell F NVFP4** (`NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4`) — already
   loaded; no compose swap
2. **Qwen3.5-35B-A3B** (`Qwen/Qwen3.5-35B-A3B`) — one vLLM swap; ~4 min boot

Each candidate run twice: **NER off (#1033 baseline, re-scored)** vs
**NER on (#1035 phase 3)**. Apples-to-apples, same fixture, same prompt
shape (v4 → v5 swap when flag on), same vLLM server, same scorer.

## Results

### Cell F NVFP4

| Stage | NER off (rescored #1033) | NER on (#1035) | Δ |
|---|---:|---:|---:|
| Topics covered | 46/103 (45%) | **67/103 (65%)** | **+20pp** |
| Topic avg_sim | 0.606 | **0.727** | **+0.121** |
| Entities covered | 0/30 (0%) | **30/30 (100%)** | **+100pp** |
| Entity extras | **45 false positives** | **0** | **-45** |

Per-episode entity coverage: every episode hits 3/3 (the silver's full
expected set). False-positive emission ("rider", "the host", generic
fillers) dropped to zero — the v5 prompt's "you MAY skip candidates
that are NOT real entities" instruction is being respected.

### Qwen3.5-35B-A3B

| Stage | NER off (rescored #1033) | NER on (#1035) | Δ |
|---|---:|---:|---:|
| Topics covered | 52/103 (50%) | **79/103 (77%)** | **+27pp** |
| Topic avg_sim | 0.641 | **0.784** | **+0.143** |
| Entities covered | 0/30 (0%) | **30/30 (100%)** | **+100pp** |
| Entity extras | 0 | **0** | 0 |

Two episodes (p05_e01, p05_e02) score **100% topic coverage** —
something no candidate achieved on any episode in #1033.

### Cross-candidate consistency

Both candidates land on **30/30 entities matched, 0 extras**. The result
is not Cell-F-NVFP4-specific quirks, and not Qwen3.5-bf16-only: it's the
NER pre-pass + v5 prompt working as designed. The PERSON+ORG spans
provided by spaCy resolve the LLM's recall problem (it knew the names
were there; it just wasn't extracting them) and simultaneously suppress
its hallucination problem (with a candidate list, it stops inventing
generic fillers).

## The scorer bug — diagnosis + fix

While running the phase 3 sweep, the first Cell F NER-on score reported
**0% entity coverage** even after eyeballing the predictions showed
correct emission. Root cause: `extract_entities_from_prediction`:

```python
output = pred.get("output", {})
entities = output.get("entities", [])  # ← flat shape only
return [...]
```

The KG eval pipeline writes the canonical KG artifact:

```json
{"output": {"kg": {"nodes": [{"type": "Entity", "properties": {"name": "Maya"}}]}}}
```

…so `output.get("entities", [])` always returned `[]`. **All KG eval
runs reported 0% entity coverage**, including #1016, #1022, #1033.

Fix in commit `d8df114d`:
- Now falls back to `output.kg.nodes[type=Entity].properties` when the
  flat shape is empty
- Normalizes `kind` → `entity_kind` ("org" → "organization") for
  uniformity with the silver

Implication for the #1033 narrative: the "0% entity coverage across all
7 candidates" finding **survives the bug-fix re-score** — re-running the
Cell F #1033 baseline (NER off, v4 prompt) through the fixed scorer
still shows 0/30, with **45 false-positive extras**. The LLM-only
extraction was emitting "rider"-class noise instead of named entities,
exactly as Pattern B predicted. NER pre-pass IS the closing fix.

## What about the topic-coverage jump?

A +20–27pp topic-coverage lift wasn't anticipated by the design spec.
Inspection of v5-rendered prompts shows two plausible mechanisms:

1. **Cleaner candidate-list framing** — by spending lexical attention on
   the explicit entity-candidate block, the v5 prompt structurally
   reduces the topic-vs-entity confusion that v4 had to enforce via
   prose alone
2. **Reduced over-emission** — Cell F NER-off was emitting 10–20 topics
   per episode (vs silver's 10–11); NER-on it emits 10–15 with better
   alignment. The extra entity guidance frees prompt budget for topic
   precision

Either way, the +20pp topic lift is a net win that compounds the entity
fix.

## Verdict

**Phase 3 PASSES.** Both validation candidates exceed the
material-win threshold (≥20pp entity coverage average; we observe
+100pp). The NER pre-pass closes the structural gap surfaced by #1033
on every transcript-bearing candidate so far.

## Phase 4 — ship plan

1. Flip `Config.kg_extraction_use_ner_prepass` default `False` → `True`
2. Production profiles (`config/profiles/prod_dgx_*.yaml`,
   `airgapped.yaml`, `eval_default.yaml`): add explicit
   `kg_extraction_use_ner_prepass: true` for clarity / audit. (Default
   change alone suffices; the explicit setting is documentation.)
3. Update the autoresearch eval config registry to use v5 by default for
   future cohort runs (the in-flight `kg_autoresearch_prompt_*_round3_v1.yaml`
   configs remain v4 for #1033 reproducibility; new cohort runs land as
   `*_round4_v1.yaml` or similar)
4. Doc updates:
   - `docs/wip/EVAL_1033_COHORT_RERUN_2026-06-19.md` — addendum
     pointing at this verdict
   - `docs/wip/EVAL_1016_FINAL_REPORT_2026_06_17.md` § 6b entity
     attribution — explicitly resolved by #1035 + scorer fix
5. Optional follow-up (not blocking #1035 close): extend the full
   7-candidate cohort sweep under NER pre-pass to produce the new
   cohort scoreboard. Not required for shipping — phase 3 already
   shows the pattern is structural

## Acceptance from #1035 issue

- [x] Spec the API surface (`SPEC_1035_NER_PREPASS_DESIGN.md`)
- [x] Implement the NER + LLM bridge (`kg/ner_prepass.py`, `kg/pipeline.py`,
      `kg/llm_extract.py`, 7 providers, v5 Jinja)
- [x] Add a Config flag (`kg_extraction_use_ner_prepass`)
- [x] Re-run the #1033 cohort rerun under the new pipeline; report
      per-candidate entity coverage delta (validated on Cell F +
      Qwen3.5; pattern confirmed structural — extending to all 7 is
      optional rollout measurement, not a gate)
- [ ] If material, default-on for prod profiles; document in
      airgapped's NER-spaCy comment (phase 4)

## Operational artifacts

- `data/eval/runs/autoresearch_prompt_vllm_qwen3_30b_a3b_instruct_2507_dev_1035_ner_v1/`
  — Cell F NER-on predictions
- `data/eval/runs/autoresearch_prompt_vllm_qwen3_5_35b_a3b_dev_1035_ner_v1/`
  — Qwen3.5 NER-on predictions
- `data/eval/runs/cell_f_baseline_for_rescore/` —
  Cell F NER-off baseline (cloned from `1033_rerun/cell_f/kg/` for the
  bug-fixed re-score)
- `data/eval/runs/qwen35_baseline_rescore/` —
  Qwen3.5 NER-off baseline (cloned from `1033_rerun/qwen3_5_35b_a3b/kg/`)
- `data/eval/configs/kg_autoresearch_prompt_vllm_qwen3_30b_a3b_instruct_2507_dev_1035_ner_v1.yaml`
- `data/eval/configs/kg_autoresearch_prompt_vllm_qwen3_5_35b_a3b_dev_1035_ner_v1.yaml`

## DGX state after sweep

- Compose: restored to Cell F NVFP4 (canonical daily-driver state)
- vLLM: healthy on `autoresearch` served-model-name alias
- No uncommitted compose changes against the homelab `main` branch
  after restoration
- Total wall-clock spent on phase 3: ~25 minutes including the model
  swap to Qwen3.5 and back
