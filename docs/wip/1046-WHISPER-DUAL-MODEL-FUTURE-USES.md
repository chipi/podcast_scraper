# #1046 — Whisper dual-model machinery: future uses (beyond skip-deep)

**Created**: 2026-06-23.
**Sibling docs**:
- `1046-WHISPER-MULTI-MODEL-DESIGN.md` — the original design + the
  measurement passes that drove the skip-deep-gate rejection.
  Removed 2026-06-24 (closeable per WIP_README); git history retains
  the content at commit `c48b01b1` and earlier.
- Measurement artifacts: `data/eval/runs/1046-measurement-pass-2/`

## Why this doc exists

The original #1046 design proposed a **skip-deep gate**: run `small.en`
first; if the sniff transcript is content-light by NER count, *skip*
the `large-v3` deep pass and ship the sniff transcript as canonical.

The 32-episode measurement pass on the fixture corpus (see sibling doc
§ 13) showed that this gate is the **wrong optimisation under the
operator's stated "best intelligence extraction" goal**:

- ~12% wallclock saved on the corpus (modest)
- **9% false-negative rate** — 1 in 11 deep-worthy episodes silently
  gets the lossy `small.en` transcript, corrupting downstream NER /
  KG / summarisation
- Quality goal trumps wallclock; the 9% FN cost is unacceptable

**Decision**: skip-deep gate is **rejected** for any prod profile.
The orchestrator code shipped (commit `9f75049f`) stays in-tree as
opt-in plumbing — it works correctly — but no prod profile enables it.

However, the per-call `model_override` machinery and the fact that
both models coexist on DGX (3.26 GiB resident, ~3% of unified memory)
**are useful for other things** that DO align with the intelligence
goal. This doc parks five such uses so a future session can pick them
up without re-deriving the framing.

## What machinery is already in tree (and what it can do)

After commits `27d65fb4` + `9f75049f`:

- `TailnetDgxWhisperTranscriptionProvider.transcribe_with_segments`
  takes a `model_override: str | None` kwarg. When set, the call uses
  that HF repo ID instead of `cfg.dgx_whisper_model` for THIS call
  only.
- `src/podcast_scraper/workflow/sniff_gate.py` is a workflow-level
  orchestrator that calls the provider with overrides. It currently
  implements only the skip-deep path; the same module is a natural
  home for the alternatives below.
- Speaches container on DGX has both `Systran/faster-whisper-small.en`
  and `Systran/faster-whisper-large-v3` loaded (~3.26 GiB total).
  Per-request model selection via `/v1/audio/transcriptions` works.
- pyinfra deploy (commit `9f75049f` step 1a) chowns the HF cache to
  uid 1000 so speaches can install additional models on-demand
  (~9.5 s download per new model on first use).
- spaCy NER + `en_core_web_sm` are loadable in the workflow process
  (`sniff_gate._load_nlp`).

So the *capability* exists; nothing below requires new infra. Each
option is a workflow-orchestration change plus tests.

## Measurement data already in hand (from the rejected gate's run)

- `T_small / T_large` ratio: **geomean 4.98×**, range **1.5× – 8.0×**
  across 32 mixed-length episodes
- Both models coexist in 3.26 GiB / 121.7 GiB (autoresearch + pyannote
  coexistence comfortable)
- 32 small.en transcripts + 32 large-v3 transcripts of the fixture
  corpus saved at `data/eval/runs/1046-measurement-pass-2/transcripts_*/`
  — usable as cached inputs for ANY follow-up offline experiment
  without re-running DGX

The corpus-level entity counts on (source / small / large) for these
32 episodes are in
`data/eval/runs/1046-measurement-pass-2/analysis.csv`. The 5%/100%
agreement signal between small.en and large-v3 entities is already
captured.

## The 5 options

Each one is a *use* of the dual-model machinery that serves the
intelligence-extraction goal rather than the wallclock goal. Listed
cheapest → highest-value-but-most-complex.

---

### Option 1 — Dual-pass reconciliation

**Pitch**: Run *both* models on every episode. Compare the two
transcripts. Flag segments where they disagree as "review-worthy" so
operators can sanity-check (or so downstream LLM-based extraction
prompts can be made aware of low-confidence regions).

**Mechanism**:
1. Sniff (`small.en`) and deep (`large-v3`) both run on every episode.
   These can run **in parallel** within the speaches container (single
   GPU, but speaches serialises requests internally → effectively
   sequential at the container level, but the *workflow* doesn't
   block).
2. Align the two transcripts segment-by-segment by start-time
   (speaches returns `verbose_json` with per-segment timestamps).
3. For each aligned pair compute a similarity score (token overlap or
   Levenshtein-on-words). Below a threshold → mark the segment as
   `dual_disagreement=True` in the transcript output.
4. Downstream stages (LLM extraction, screenplay generation, KG
   building) read the flag and can weight / verify accordingly.

**Cost**:
- Wallclock: **+T_small per episode** (small.en runs in addition to
  the existing large-v3). Measured: 25–75s extra on medium-to-large
  episodes; ~3s on short clips.
- Code: workflow-level diff to `sniff_gate.py` (or a new module
  `dual_pass.py`), plus a segment-alignment helper. ~150 LOC + tests.
- Storage: 2× transcript size — both go to artifacts.

**Quality impact**: **strictly better** than current single-pass. The
"disagreement" signal is a hint to downstream stages that lets them
hedge or verify. Best signal for catching Whisper hallucinations
(both models hallucinate, but rarely the same hallucination).

**Open questions**:
- What disagreement threshold marks a segment as "review-worthy"?
  (Levenshtein-on-words ratio < 0.7? operator preference TBD)
- Does the downstream LLM extraction *actually* use the flag, or do
  we need to change the prompt template too? (#1035-adjacent — worth
  designing the flag schema first)

**Effort estimate**: ~1 day. Segment alignment is well-trodden;
nothing exotic.

---

### Option 2 — Confidence-weighted NER

**Pitch**: Entities the NER pipeline finds in *both* `small.en` and
`large-v3` transcripts are high-confidence. Entities only in `large-v3`
are lower-confidence (model dependence) and merit extra verification.
Feed the confidence into the KG so the operator can filter by
trust-level.

**Mechanism**:
1. Run both models on every episode (same as Option 1).
2. Run spaCy NER on both transcripts.
3. Per entity: if it appears in both → `confidence_tier="A"`. If
   only in large-v3 → `confidence_tier="B"`. (If only in small.en →
   `confidence_tier="ignore"` — small.en finding something large
   missed is almost always small.en hallucination.)
4. Write the confidence_tier into the KG node properties.
5. KG queries / viewer can filter by tier.

**Cost**:
- Wallclock: same as Option 1 (+T_small per episode) — Option 2 is
  a *consumer* of Option 1's output, so it composes for free if you
  already paid the dual-pass cost.
- Code: NER pipeline already runs (RFC-097 wired this). Need to:
  (a) feed it both transcripts, (b) thread confidence_tier through
  the KG schema. ~200 LOC + KG migration for the new property +
  tests.
- KG schema: new node property on Person/Organization. Backwards-
  compatible if defaulted.

**Quality impact**: **stronger KG**. You can now query "show me only
the entities that BOTH models agree exist" → near-zero false positives
in the KG. Or "show me the entities that ONLY large-v3 surfaced" →
candidates for human review.

**Open questions**:
- What about repeated entities at different positions? (e.g. "Maya"
  appears in segment 12 (both) and segment 47 (only large-v3) — is
  THAT instance tier-B?) Per-instance vs aggregated-per-entity
  decision.
- Does the KG viewer need a UI affordance for the tiers? (chip color,
  filter toggle, etc.)

**Effort estimate**: ~1.5 days on top of Option 1. KG schema change
is small but needs migration plan.

---

### Option 3 — Sniff-driven NER pre-pass (#1035 territory)

**Pitch**: Use the cheap `small.en` transcript JUST for entity
discovery to hand the downstream LLM extraction stages a *hint*
about which entities to expect. The LLM still works on the
`large-v3` transcript as the substrate, but its prompt now includes
"these entities probably appear: ..." which improves recall.

**Mechanism**:
1. Run `small.en` first.
2. Run spaCy NER on the sniff transcript.
3. Pass the entity list as context into the LLM extraction prompt
   (e.g. `metadata_generation`, `gi_pipeline`) when the deep transcript
   arrives.
4. The LLM uses this as a "look out for these" hint, NOT ground truth.
5. The deep transcript is the canonical input.

**Cost**:
- Wallclock: **+T_small per episode** (like Options 1+2).
- Code: workflow-level addition to `metadata_generation` and/or
  `gi_pipeline`. Need a prompt-template change that accepts a
  `expected_entities: list[str]` field.
- Tests: prompt-injection-safe (the entity hint is user-controlled
  but bounded — small.en isn't a prompt-injection attacker but
  source audio could in theory contain text that becomes hint text).

**Quality impact**: depends entirely on the LLM. Reasonable estimate:
**+5-15% recall** on entity extraction in long episodes (LLM has a
short context-aware sniff list). Less impact on short episodes (LLM
already attends to the whole transcript).

**Open questions**:
- Overlaps with **#1035 (NER pre-pass)** — that issue already proposed
  a similar mechanism using NER on the deep transcript. Sub-question:
  is `small.en` NER good enough for the pre-pass to skip the deep-NER
  step (and save the deep-NER compute), or does it just augment?
- How to prompt-inject-safely thread the entity list. Probably just
  passes-as-context, but should be reviewed before shipping.

**Effort estimate**: ~2 days. Most of the cost is in measuring
whether the recall gain justifies the latency penalty.

**Note**: this option is closest to what `cfg.dgx_whisper_sniff_model`
was *implicitly* designed for. If you re-enable the Config knob for
this purpose, the meaning shifts from "gate criterion" to "pre-pass
model" — worth a Config-knob rename when this lands.

---

### Option 4 — Speculative pipeline (streaming UX win)

**Pitch**: Start downstream stages (NER, screenplay, KG insertion)
on the `small.en` transcript *immediately* — UI shows preliminary
results in seconds. Swap in the `large-v3` transcript when it
finishes, re-running affected stages.

**Mechanism**:
1. Workflow orchestrator dispatches `small.en` first (~3-75s per
   episode).
2. As soon as the sniff transcript lands, downstream stages
   (preliminary NER, draft screenplay, draft summary) start
   processing.
3. UI shows "transcript draft" / "preliminary results" with a
   "deep pass running..." indicator.
4. When `large-v3` finishes (5-30 minutes later for long episodes),
   re-run the same downstream stages with the deep transcript.
5. UI updates with final results; preliminary results either
   replaced silently or marked as "now superseded".

**Cost**:
- Wallclock: net same. **Critical-path UX time-to-first-display:
  ~5× faster.**
- Code: large workflow refactor — stages need to be re-runnable
  with different inputs and merge-able outputs. Probably the
  hardest of the 5 options.
- UI: viewer needs a "draft → final" state machine and update flow.
- Idempotency: every downstream stage needs to be safely re-runnable
  on the same episode. Several already are (transcription is
  idempotent); some aren't yet (KG insert dedups are best-effort).

**Quality impact**: same final quality as current; **dramatically
better perceived latency** for the operator. Episode appears in the
UI seconds after submission instead of minutes-to-hours.

**Open questions**:
- Is the UX win worth the workflow refactor cost?
- What does the "preliminary results superseded" UI look like?
  (Toast? Inline diff? Silent replace?)
- KG-insert idempotency: do we need a "preliminary → final" marker
  in the graph itself so consumers know which is which?

**Effort estimate**: ~5 days minimum, mostly in the stage-re-runnable
refactor and the viewer UI.

---

### Option 5 — Cross-model dispatch by domain

**Pitch**: Different audio characteristics route to different models.
English-only fast podcast → `small.en` is fine. Mixed-language /
non-English → `large-v3-multilingual` (HF repo: `Systran/faster-whisper-large-v3`,
loaded). Music-heavy / noisy → `medium` model with VAD pre-filter
turned up. This is NOT a quality-vs-cost trade — it's "pick the right
tool for the audio".

**Mechanism**:
1. Add a *very* cheap audio classifier upstream of transcription:
   language ID (already in the speaches output) + maybe genre /
   speech-vs-music ratio (could use a lightweight classifier or
   even just the language-detection confidence as a proxy).
2. Workflow picks `model_override` based on the classification.
3. New Config knob: `dgx_whisper_per_audio_model_dispatch: bool`
   (default False); when True, the dispatch logic runs.

**Cost**:
- Wallclock: ~+50ms-1s (language ID is fast on most encoders).
- Code: small workflow addition. ~50 LOC + tests.
- Infra: language ID model may need to be loaded — speaches does it
  via Whisper itself, so probably no additional load.

**Quality impact**: depends entirely on whether your corpus mixes
domains. If 100% English podcasts, this is a no-op. If 20%
non-English, the dispatch could *recover* episodes that currently
fail Whisper transcription silently.

**Open questions**:
- Does our corpus actually mix domains enough to justify this?
- What's the failure mode of `large-v3` on the non-English subset
  today? (Worth measuring on a small sample first.)
- Could compose with Option 1 — dual-pass with cross-model dispatch
  could be a really strong signal for difficult episodes.

**Effort estimate**: ~1-2 days. Mostly bounded by how rigorous the
domain classifier needs to be.

---

## Recommended sequencing (if any of these get prioritised)

If pursuing in this order:

1. **Option 1 first** (dual-pass reconciliation, ~1 day). Cheap, low
   risk, immediate quality signal. Reuses the existing transcripts
   sitting in `data/eval/runs/1046-measurement-pass-2/` for an offline
   prototype before committing to dual-passing in prod.

2. **Option 2 on top of 1** (~1.5 days additional). Free composition;
   just needs the KG schema change. The KG ends up stronger with no
   additional wallclock cost beyond Option 1's.

3. **Options 3 and 5 in parallel** (~2 + ~2 days, independent). Both
   add scope. Option 5 needs corpus characterisation first to know
   if it's worth it.

4. **Option 4 last** (5+ days). High-value UX win but big workflow
   refactor; ship after the other options de-risk the "both models
   running in prod" pattern.

**My recommendation when you next pick this up**: prototype Option 1
offline using the already-saved transcripts at
`data/eval/runs/1046-measurement-pass-2/transcripts_*/`. Measure the
disagreement-segment distribution. If most segments agree, Option 1
gives little upside; if disagreement is concentrated in domain words
(like `Singletrack` / `berm` / `braking` from measurement pass 1),
it's a strong quality signal worth the wallclock.

## Anti-patterns to avoid

- **Don't re-introduce the skip-deep gate** as part of any of these.
  All 5 options run BOTH models. The intelligence goal doesn't
  tolerate skipping the deep pass.
- **Don't conflate Option 3 with the rejected #1046 gate.** They
  share machinery but answer different questions: skip-deep was
  *replacement* (sniff INSTEAD of deep); Option 3 is *augmentation*
  (sniff IN ADDITION to deep, hint into LLM).
- **Don't move forward on Option 4 without idempotency audit** of
  all downstream stages. A non-idempotent stage will produce a
  mess when the preliminary results get re-computed.

## Status

This doc is **planning / parking** material. Nothing here is queued
or in-flight. Each option is independent; can be picked up in any
order. The dual-model machinery in tree stays unchanged regardless.
