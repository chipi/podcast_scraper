# Corpus reprocess — the canonical arc plan (v2 → v3, toward 1000+ episodes)

Status: **Active**. Created 2026-07-20. Updated **2026-08-02** — reconciled: the
former `PLAN-v2.5-corpus-and-naming-arc-handover.md` is folded in here (the 2.5
corpus is the current stage of *this* plan, not a separate north star). This is
the single canonical arc doc.

Umbrella sequencing plan. Not authoritative for component detail — the component
specs (`CORPUS-V4-FIXTURE-LADDER.md`, the issue bodies, `autoresearch/JUDGING.md`)
are. This doc carries the ordering and the gates. The durable **methodology**
(single-variable validation, reprocess-once economics, the judge-panel parity gate)
lives permanently in **ADR-143** (`docs/adr/ADR-143-corpus-reprocess-methodology.md`) —
this plan is the ephemeral sequencing on top of it, and survives only as long as the arc.

## North star

Reprocess the corpus **v2 → v3** using a **fully-local (DGX) pipeline**, and expand
from the current ~90–100-episode corpus to **500–1000 episodes across 20–30 podcasts**
(with 10k as the eventual horizon). The 2.5 corpus — v2.4 pipeline/quality with the
**Gemini** naming/summarization LLM replaced by a **DGX-local model at judge-panel
parity** — is the final single-variable step before the frozen scale run.

## The incremental single-variable validation arc (prod-v2.x, ~90 eps)

Before the frozen scale run (Phase 4), each producing-model / input change is
**validated in isolation** on the existing ~90-episode corpus: change exactly ONE
variable, let the whole cascade re-run, and compare that version against its
predecessor. The stick is **parity with the prior version, not ultimate truth** —
each step only has to be "not worse than what we had," because the previous corpus
is the thing we already accepted. (Same discipline as the deepgram freeze arbiter in v2.2.)

| Version | Single variable changed | Status |
| --- | --- | --- |
| **v2.1** | speaker **naming** (on frozen deepgram diarization) | done |
| **v2.2** | **diarization** → pyannote community-1 (DGX-local), full ~90-ep corpus | **MERGED — PR #1335** (Closes #1188, #1290, #1292–#1296, #1321, #1329–#1331) |
| **v2.3** | **ASR** → faster-whisper turbo (DGX-local) + ADR-123 coverage failover | **MERGED** (bundled into #1355) — turbo primary, coverage failover retargeted large-v3→**MOSS** (#1273). Open follow-up **#1273 TODO 1** (int8-vs-fp16 serving test) BLOCKED on DGX SSH — [doc](1273-largev3-int8-vs-fp16-BLOCKED.md); NOT on the critical path. |
| **v2.4** | **GI/KG optimizations** — remove `GI_MAX_INSIGHTS_CEILING` (#1191) + KG Voice-node (#1220) + naming-4 speaker labeling + per-episode observability manifest | **MERGED — PR #1355** (squash `b24608fa`; Closes #1191, #1220, #1337, #1358–#1364) |
| **v2.5** | **LLM** → Gemini replaced by a DGX-hosted local model (the gateway to a **fully-local** pipeline) | **CURRENT FRONT** — see the v2.5 stage detail below |

**Why this ordering / why NOT bundle:** each variable's downstream deltas must be
attributable to *that* variable. Folding a GI/KG **schema** change (#1191/#1220) into
the ASR step (v2.3) would make a GI delta un-attributable — so the schema work was its
own gate (v2.4). DGX-LLM (v2.5) lands **after** the GI/KG shape is settled, so we swap
the LLM against the final artifact shape, and v2.4 banks the optimization gains before
that swap. The scale run then reprocesses the **frozen combination** once (Phase 4).

### Measurement per step — cost-aware, no needless re-transcription

The expensive layer is **transcription** (OpenAI Whisper, ~$0.50/ep). Everything
downstream — diarization, naming, GI, KG, summary — is free (DGX) or cents (gemini).
So we never re-transcribe to "confirm" a baseline; the prior corpus on disk **is** the
baseline. Per step:

- **Deterministic layers = the primary verdict, no baseline needed.** Transcript WER
  vs the prior version; **speaker-roster parity** (names + roles) — mostly deterministic,
  the star signal, same metric as the v2.2 community-1↔deepgram comparison.
- **Noisy layers (GI/KG/summary) get a cheap noise floor.** pyannote and gemini both
  drift run-to-run, so establish the floor by re-running only the **downstream cascade**
  on the prior version's **existing transcript** (`rediarize_only`: reuse the paid
  transcript, re-diarize + re-name + re-enrich) — free DGX + ~$1/100 eps of gemini, **no
  re-transcription**. Then the real signal = (vX − vX−1) **above** (vX−1 − vX−1′).

## v2.5 stage detail — the current front (Gemini → DGX-local LLM swap)

The full-corpus diarization run is **done** (v2.2 community-1 across the ~90-ep corpus,
PR #1335) — the corpus already carries uniform RTTM / speaker counts, so that is off the
plate. The remaining single-variable step is the **naming/summarization LLM swap**.

### Decisions locked (operator, 2026-07-30)

1. **2.5 model = decided by the loop.** Do NOT pre-commit the DGX model. The
   autoresearch prompt-tuning loop (stage D below) bakes off candidates (Cohere Command /
   Qwen3-30B-A3B / DeepSeek — all in the roster) and picks the winner. "column-based
   command-free" ≈ Cohere Command, but it's a candidate, not a given.
2. **2.5 ship gate = judge-panel parity — ONLY (operator, 2026-08-02).** A cross-vendor
   judge panel scores the DGX output vs the Gemini baseline; 2.5 ships only at statistical
   parity. Panel must be **disjoint-vendor** (silver + judge from a vendor NOT in the cohort —
   `feedback_silver_judge_vendor_bias`), **scalar mode** (not pairwise —
   `feedback_scalar_over_pairwise_for_judge_trust`), and the score parser must strip
   `</think>` (`feedback_judge_reasoning_block_parsing`). **Human-GT / #1189 is NOT part of the
   parity gate** — the judge panel is the sole ship signal; #1189 stays only the Phase-3
   reprocess acceptance gate (a separate role).
3. **Prompt-tuning precedes the swap.** Tune the naming/resolver LLM query per-provider
   (incl. DGX) BEFORE swapping, because the swap itself forces more prompt fine-tuning.
4. **Sequencing = 2.5 first, THEN go-live (operator, 2026-08-02).** Single-track focus on the
   corpus arc; freeze the 2.5 corpus before returning to the go-live ops tail (CF WAF, alerts,
   RBAC). Go-live is largely done and has no corpus-pipeline dependency, so it waits, not blocks.
5. **DGX serving = the original plan (operator, 2026-08-02).** v2.5 swaps ONLY the naming/
   summarization **LLM** (Gemini → a DGX-local Llama-class model). That model is served on the
   **autoresearch vLLM (`:8003`)**, brought up with `ssh ops@dgx-llm-1
   /usr/local/bin/gpu-mode-swap.sh research` (the GPU sits `free`/idle until then — it was never
   "gone"; an earlier probe as the wrong SSH user misreported that). The **supporting services**
   (faster-whisper ASR `:8000`, pyannote diarization `:8001`, MOSS ASR-failover `:8004`) are fixed —
   they run every reprocess but are NOT under research in v2.5. Full topology + access:
   `docs/architecture/DGX_SERVING.md` → homelab SSOT `agentic-ai-homelab/infra/dgx/README.md`.

### The v2.5 sub-sequence

- **D — Autoresearch prompt-tuning + bake-off (the pivot, BEFORE the swap).** Stand up
  the bake-off in the autoresearch harness (`autoresearch/`, `JUDGING.md`,
  `bundled_prompt_tuning/`). **Serving:** bring the autoresearch vLLM up on `:8003` via
  `ssh ops@dgx-llm-1 /usr/local/bin/gpu-mode-swap.sh research` (GPU sits idle until then — see
  decision 5 + `docs/architecture/DGX_SERVING.md`). Serve the local LLM candidate set (Cohere
  Command, Qwen3-30B-A3B, DeepSeek — the Llama-class replacements for Gemini); deep-research each
  model's knobs (`feedback_deep_research_per_model`) and update
  `autoresearch/PER_MODEL_OPTIMAL_PARAMS.md`. Metric = judge-panel parity vs the 2.4-Gemini
  baseline (the naming-4 relabel defect/named metrics from #1355 + the panel). Output:
  (a) winning DGX model, (b) tuned per-provider prompts, (c) recorded parity result.
  **This is critical-path gap #1.**
- **E — Provider swap (Gemini → DGX-local).** Wire the winning model via the existing DGX
  profiles: `prod_dgx_full`, `cloud_with_dgx_primary`
  (`config/profiles/`). Expect iterative prompt fine-tuning as real output diverges from
  the bake-off. Re-validate with the naming-4 harness + judge panel each iteration.
- **F — Freeze the 2.5 corpus.** Full reprocess under the DGX-local profile on the ~90-ep
  corpus. Confirm judge-panel parity with 2.4 holds (ship gate). Commit all; write the
  2.5 ADR (decision + alternatives + parity evidence).

DGX access reminders: `gpu-mode-swap.sh research` (NEVER `code`/coder-next —
`feedback_never_use_coder_next`); ACL permits only :8003 / :11434
(`project_dgx_tailscale_acl`); model size class = TOTAL params, A3B active ≠ substitute
(`feedback_model_size_class`).

## The organizing principle: reprocess-once economics

A 500–1000-episode run is expensive and slow (at large-v3's measured 7.8× realtime,
1000 eps ≈ 4 GPU-days; 10k ≈ 40 days). Anything that changes:

- the **stored artifact shape** (KG/GI schema),
- the **input text** (cleaning), or
- the **model that produces it** (ASR/diarization/LLM)

must be locked **before** the run, or the whole corpus is reprocessed twice. So the
"next cut" is not "the most issues" — it is **everything that would otherwise force a
second full rebuild**. The 2.5 LLM swap is the last such variable.

## v4 fixtures as a growing harness (the key sequencing decision)

`#1189` (golden fixtures v4) is the acceptance gate for the reprocess. Its own thesis:
every real trap case came from a human reading real output, not from a test — so freezing
the full ladder early encodes an incomplete understanding and then churns on every new case.

Resolution — split v4 into **container vs contents**:

- **Now (cheap):** build only the v4 *harness* — the fixture format, the §G
  metadata-vs-conversation contract, and 2–3 seed cases from bugs already known.
- **During the arc:** every bug found drops in as a new fixture row (repro-first /
  matrix-row rule). The arc *authors* v4 as a side effect.
- **Late:** freeze the full 12-case ladder once feedback saturates → the frozen set is
  the reprocess gate.

**Decided 2026-07-20 — scaffold thickness:** build the harness **as big as needed**, not
artificially lean. Phase 0 delivers the fixture schema (§G contract) + a loader + a runner
that drives the *shipped* roster path (never re-implemented) + the perturbation mechanism +
at least one real seed case (Hard Fork ep1) proving the harness end-to-end.

## Phased sequence

| Phase | Work | Purpose |
| --- | --- | --- |
| **0** | branch (done) + v4 **harness** scaffold (`#1189` container only) | the feedback sink |
| **v2.1–v2.4** | naming → diarization → ASR → GI/KG schema, each validated in isolation | **DONE** (#1335, #1355) |
| **v2.5** | LLM swap Gemini → DGX-local, stages D→E→F above | **CURRENT** — gated by judge-panel parity |
| **3** | freeze v4 (`#1189` full 12-case ladder) | the acceptance gate |
| **4** | reprocess v2 → v3 @ 500–1000 eps, 20–30 podcasts, fully-local | gated by frozen v4 |

## Bucket A — lock before the reprocess

Ordered by the sequence above. Each, if done *after* v3 ships, means re-running the corpus.

| Issue | What | Why before | Impact |
| --- | --- | --- | --- |
| `#1189` | Golden fixtures v4 — one per show, real pyannote turns + feed metadata + hand-labelled truth (12 trap cases). | The ruler. Build it (as a growing harness) before you cut. | cheap, no GPU |
| **v2.5 LLM swap** | DGX-local naming/summarization model at Gemini parity (stages D→E→F). | The LLM is the last producing-variable; swapping it after v3 ships = full rebuild. | DGX GPU (bake-off) + parity gate |
| `#630` | The expansion *vehicle* — source 20–30 podcasts / 500–1000 eps. Sourcing runs in parallel with the LLM swap. | Cannot pick the corpus shape after transcribing. | sourcing effort |

The former Bucket-A schema items (`#1191`, `#1220`, `#1188`) and the ASR/diar deathmatch
(`#1178`/`#1179`) all **landed** in v2.2–v2.4; they are no longer open locks.

## Bucket B — decide before, lower artifact risk

| Issue | Call |
| --- | --- |
| `#630` | The expansion vehicle (also in Bucket A). Scale target to 20–30 podcasts / 500–1000 eps. |
| `#102` | Golden data set for transcripts — overlaps `#1189` truth-labelling; fold in, don't run separately. |
| `#1192` | Speaker recall for the ~113 unknown panel tail. Trades precision ("a wrong name is worse than no name"); **validated NEGATIVE this session** — defer to v3.1 unless measured precision-safe. |

## Bucket C — explicitly NOT this cut

Stated with equal weight so the exclusion is a decision, not silence.

- **LoRA / fine-tuning:** `#629`, `#631`. Autoresearch programme is closed; LoRA out of scope. Do not reopen inside this cut.
- **Go-live / public-edge / security:** `#911`, `#1062`, `#1063`, `#1158`–`#1166`, `#801`, `#806`, `#840`, `#1162`. Separate arc on `production` (Goal-1). Own track. (Now largely live — player + operator on `main`.)
- **Viewer / UX / frontend:** `#627`, `#1168`, `#1208`, `#1209`, `#1210`, `#1211`, `#1214`, `#1219`. Consume the corpus; do not gate producing it.
- **Housekeeping / tech-debt:** `#18`, `#216`, `#255`, `#333`, `#372`, `#426`, `#436`, `#447`, `#538`, `#860`, `#976`, `#1028`, `#1142`, `#1143`, `#1222`.

## Follow-on: viewer-perf pass (after v3 exists)

Not reprocess blockers, but the 1000-episode corpus makes them bite. Do this pass once
v3 is built:

1. **`#1219` — graph-v3 KG-second-wave forces a full cytoscape rebuild (~2500–3000 ms).**
   The full-rebuild cost scales with node count (measured ~3.1 s on prod-v2's ~1,157
   nodes) — at 1000 episodes the graph is far larger, so this graduates from
   nice-to-have toward "the graph view is unusable on the full corpus." **First item of
   the viewer-perf pass.** Its cleaner fix (canonicalise ids on wave 1) needs a consumer
   audit of every raw-id reader; do that audit *after* the v3 artifact shape settles.
2. Remaining viewer/UX items (`#1211`, `#1208`, `#1209`, `#1210`, `#1214`, `#1168`).

## Decisions

- **Scaffold thickness** — DECIDED (2026-07-20): as big as needed. See the v4-harness section.
- **Scope of the cut** — DECIDED (2026-07-20): **full Bucket A** (one rebuild, born correct).
- **2.5 model** — decided by the bake-off loop (locked decision 1 above), not pre-committed.
- **2.5 ship gate** — disjoint-vendor scalar judge-panel parity vs Gemini (locked decision 2).
- **Expansion sourcing** — OPEN: which 20–30 podcasts (feeds `#630` must source).

## Open decisions (for the fresh session to resolve)

- Does go-live have to precede 2.5, or run in parallel? (Go-live is largely done; likely parallel.)
- Does the parity gate use judge-panel ONLY, or judge-panel + human-GT (`#1189`) as a second signal?
- Corpus-growth strategy: target size (500 / 1000 / path to 10k?) and curated-overlap-vs-broad-ingest — undecided (`ONBOARDING-SHOWS-FOR-ENRICHER-VALUE.md`, feeds Corpus Scout `PRD-037`/`RFC-088`).
