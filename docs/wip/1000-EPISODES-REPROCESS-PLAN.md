# 1000-episode reprocess — readiness plan (v2 → v3 at scale)

Branch: `feat/1000-episodes`. Status: **Active (plan)**. Created 2026-07-20.

Umbrella sequencing plan for the next arc. Not authoritative — component specs
(`CORPUS-V4-FIXTURE-LADDER.md`, `1191-ROUTE-AND-TAG-PLAN.md`, the issue bodies) are.

## North star

Reprocess the corpus **v2 → v3** using other LLMs, and expand from the 10-episode
v3 pilot to **500–1000 episodes across 20–30 podcasts**.

## The organizing principle: reprocess-once economics

A 500–1000-episode run is expensive and slow (at large-v3's measured 7.8× realtime,
1000 eps ≈ 4 GPU-days; 10k ≈ 40 days). Anything that changes:

- the **stored artifact shape** (KG/GI schema),
- the **input text** (cleaning), or
- the **model that produces it** (ASR/diarization)

must be locked **before** the run, or the whole corpus is reprocessed twice. So the
"next cut" is not "the most issues" — it is **everything that would otherwise force a
second full rebuild**.

## v4 fixtures as a growing harness (the key sequencing decision)

`#1189` (golden fixtures v4) is the acceptance gate for the reprocess. But its own
thesis is that every real trap case came from a human reading real output, not from a
test — so freezing the full ladder early encodes an incomplete understanding and then
churns on every new case.

Resolution — split v4 into **container vs contents**:

- **Now (cheap):** build only the v4 *harness* — the fixture format, the §G
  metadata-vs-conversation contract, and 2–3 seed cases from bugs already known. This
  is the *sink* that captures feedback, not "building v4 early".
- **During B and A:** every bug found drops in as a new fixture row (repro-first /
  matrix-row rule). B and A *author* v4 as a side effect.
- **Late:** freeze the full 12-case ladder once feedback saturates → the frozen set is
  the reprocess gate.

**Decided 2026-07-20 — scaffold thickness:** build the harness **as big as needed**, not
artificially lean. Phase 0 delivers the fixture schema (§G contract) + a loader + a runner
that drives the *shipped* roster path (never re-implemented — §A/§D/§F0 lesson) + the
perturbation mechanism + at least one real seed case (Hard Fork ep1) proving the harness
end-to-end.

## Phased sequence

| Phase | Work | Purpose |
| --- | --- | --- |
| **0** | branch (done) + v4 **harness** scaffold (`#1189` container only) | the feedback sink |
| **1 · B** | `#1178` turbo → `#1179` ASR/diar table on the real 10 eps; `#630` source 20–30 feeds | pick the ASR/diar model + generate real cases to read |
| **2 · A** | `#1188` cleaning → `#1191` GI route-and-tag → `#1220` Voice node | each new bug → a fixture row; lock the artifact shape in ONE rebuild |
| **3** | freeze v4 (`#1189` full 12-case ladder) | the acceptance gate |
| **4** | reprocess v2 → v3 @ 500–1000 eps, 20–30 podcasts, other LLMs | gated by frozen v4 |

## Bucket A — lock before the reprocess

Ordered by the sequence above. Each, if done *after* v3 ships, means re-running the corpus.

| Issue | What | Why before | Impact |
| --- | --- | --- | --- |
| `#1189` | Golden fixtures v4 — one per show, real pyannote turns + feed metadata + hand-labelled truth (12 trap cases). | The ruler. Build it (as a growing harness) before you cut. | cheap, no GPU |
| `#1178` → `#1179` | ASR/diar deathmatch. `#1178` turbo = 1-line config, ~5× faster (1000 eps ~0.8d vs 4d). `#1179` = full 5-stack table → default-profile decision. | Cannot pick the transcription model after transcribing 1000 eps. Highest leverage on the board. | `#1178` cheap; `#1179` ~DGX-days. Absorbs the transcription half of `#972`. |
| `#1191` | GI route-and-tag — remove the `GI_MAX_INSIGHTS_CEILING = 50` truncation. | A cap baked into the pipeline is baked into the corpus; reprocess with it on → v3 born truncated. | GI + KG schema change + migration |
| `#1220` | KG Voice-node (write-side) — retype unnamed diarization voices. | Requires a KG schema bump (2.0→2.1) + corpus rebuild anyway; batch into the same v3 run. | KG schema change + migration |
| `#1188` | Cleaning misses host-read cross-promos (The Athletic ad survives 10/10). | Cleaning runs at reprocess time and feeds speaker attribution; else the ad + speaker-set pollution carry into every v3 episode. | input-text quality, no schema change |

## Bucket B — decide before, lower artifact risk

| Issue | Call |
| --- | --- |
| `#630` | The expansion *vehicle*. Scale its target to 20–30 podcasts / 500–1000 eps. Sourcing runs in parallel with A. |
| `#102` | Golden data set for transcripts — overlaps `#1189` truth-labelling; fold in, don't run separately. |
| `#1192` | Speaker recall for the ~113 unknown panel tail. Trades precision ("a wrong name is worse than no name"); defer to v3.1 unless measured precision-safe. |
| `#972` | Real-podcast full sweep — transcription half superseded by `#1179`; keep only the summary-backend comparison if still wanted. |

## Bucket C — explicitly NOT this cut

Stated with equal weight so the exclusion is a decision, not silence.

- **LoRA / fine-tuning:** `#629`, `#631`. Autoresearch programme is closed; LoRA out of scope. Do not reopen inside this cut.
- **Go-live / public-edge / security:** `#911`, `#1062`, `#1063`, `#1158`–`#1166`, `#801`, `#806`, `#840`, `#1162`. Separate arc on `production` (Goal-1). Own track.
- **Viewer / UX / frontend:** `#627`, `#1168`, `#1208`, `#1209`, `#1210`, `#1211`, `#1214`, `#1219`. Consume the corpus; do not gate producing it.
- **Housekeeping / tech-debt:** `#18`, `#216`, `#255`, `#333`, `#372`, `#426`, `#436`, `#447`, `#538`, `#860`, `#976`, `#1028`, `#1142`, `#1143`, `#1222`.

## Follow-on: viewer-perf pass (after v3 exists)

Not reprocess blockers, but the 1000-episode corpus makes them bite. Do this pass once
v3 is built:

1. **`#1219` — graph-v3 KG-second-wave forces a full cytoscape rebuild (~2500–3000 ms).**
   The GI→KG merger re-prefixes node ids (`g:`/`k:`, `__unified_ep__`) on wave 2, so cy
   is destroyed + rebuilt instead of taking the fast path. Pure frontend
   (`web/gi-kg-viewer/`); before-vs-after the reprocess costs the same, so it is
   correctly out of the prep cut. **But:** the full-rebuild cost scales with node count
   (measured ~3.1 s on prod-v2's ~1,157 nodes) — at 1000 episodes the graph is far
   larger, so this graduates from nice-to-have toward "the graph view is unusable on the
   full corpus." **First item of the viewer-perf pass.** Ordering note: its cleaner fix
   (Design 1 — canonicalise ids on wave 1) needs a consumer audit of every raw-id reader
   (enricher artifacts, search index, deep-links); do that audit *after* the v3 artifact
   shape settles under `#1191`/`#1220`.
2. Remaining viewer/UX items (`#1211`, `#1208`, `#1209`, `#1210`, `#1214`, `#1168`).

## Decisions

- **Scaffold thickness** — DECIDED (2026-07-20): as big as needed. See the v4-harness
  section above.
- **Scope of the cut** — DECIDED (2026-07-20): **full Bucket A** (one rebuild, born
  correct). Lean A rejected — reprocessing 1000 eps twice is not worth deferring the
  schema epics.
- **Expansion sourcing** — OPEN: which 20–30 podcasts (feeds `#630` must source).
