# #1046 — DGX Whisper multi-model serving — design doc

**Issue**: [#1046](../../issues/1046) — split from [#910](../../issues/910) subscope 4.
**Status**: Design phase. Per #1046 acceptance criteria, **a design doc
must land before any service code does**. This is that doc.
**Created**: 2026-06-22.

The acceptance bar (verbatim from #1046):

> - [ ] Design doc with the (r, latency, accuracy) measured estimate
>   before any service code lands.
> - [ ] Per-request model selection working end-to-end against the
>   test-prod corpus.

This doc satisfies the first bar in plan form (the empirical
measurements need DGX time to collect — section 3 specifies the
methodology). It does NOT yet propose service code; that's gated
on the operator approving the design + the measurements coming back
in the green.

## 1. Today's state

`faster-whisper-server` (`podcast-speaches:0.1.0`) runs on DGX with
`Systran/faster-whisper-large-v3` loaded (verified 2026-06-22 via
`curl :8000/v1/models`). The service is OpenAI-compatible — its
`/v1/audio/transcriptions` endpoint already accepts a `model` query
parameter, so per-request model selection is **API-supported today**
(the model just has to be loaded in the container's model cache).

Pipeline-side:
- `src/podcast_scraper/config.py` has both `whisper_model` (default
  `base.en`) and `dgx_whisper_model` knobs.
- `src/podcast_scraper/transcription/factory.py` reads
  `params.model_name` per-call. There's no plumbing for a
  "different model on retry / sniff-pass" — every episode uses one
  model.

## 2. The sniff-pass cost model

**Workflow**:
1. Transcribe episode with `small.en` (~10× real-time on DGX).
2. Run topic-detection on the small-transcript.
3. If detection surfaces signal → transcribe again with `large-v3`
   for the deep pipeline.
4. Else → keep the small transcript (or skip entirely).

**Cost equation** (per-episode mean transcription time):

```text
without_gate(N) = N × T_large
with_gate(N, r) = N × T_small + r × N × T_large
```

Where:

- `N` = number of episodes
- `T_small` = small.en transcription time
- `T_large` = large-v3 transcription time
- `r` = sniff-hit rate (fraction of episodes that pass the gate)

**Break-even** when `with_gate == without_gate`:

```text
r* = 1 − T_small / T_large
```

For typical Whisper latency ratios on GB10:

- `large-v3`: ~0.5× real-time (5min episode → ~2.5min transcription)
- `small.en`: ~0.05× real-time (5min episode → ~15s transcription)
- ratio `T_small / T_large` ≈ 0.10
- **break-even `r* ≈ 0.9`** — the gate only saves time if **<90%
  of episodes pass** it. Above 90%, the gate is pure overhead.

**Net savings** (when `r < r*`):

```text
savings_pct = 1 − (T_small + r × T_large) / T_large
            = 1 − (0.1 + r)
```

For example, `r = 0.5` → 40% savings; `r = 0.3` → 60% savings;
`r = 0.1` → 80% savings.

**Important caveat**: this cost model ignores cold-load time. If
`small.en` is loaded on demand per-request, the ~1s model-load
overhead dominates short transcriptions. The design assumes BOTH
models stay resident in container memory (see section 4 budget).

## 3. Methodology for measuring `r` empirically

We need three measurements before any service code lands:

### 3a. Measure `T_small` and `T_large` on real audio

Use 10 representative episodes from the prod-v2 corpus (varied
durations: 5min / 30min / 60min episodes). Transcribe each with both
models. Report median, p90, p99 wall-clock per duration bucket.

Sketch script (operator-runnable on DGX):

```bash
# On DGX, with both models cached in the speaches container:
for ep in p01_e01 p02_e01 p03_e01 p04_e01 p05_e01 \
          p01_e02 p02_e02 p03_e02 p04_e02 p05_e02; do
  for model in Systran/faster-whisper-small.en Systran/faster-whisper-large-v3; do
    start=$(date +%s)
    curl -s -F "file=@/audio/${ep}.mp3" \
         -F "model=${model}" \
         http://localhost:8000/v1/audio/transcriptions > /dev/null
    end=$(date +%s)
    echo "$ep,$model,$((end - start))"
  done
done | tee /tmp/whisper_latency_${date_iso}.csv
```

### 3b. Measure `r` (sniff-hit rate) on real corpus

This needs the **sniff-pass topic-detection logic**, which doesn't
exist yet. The design question is: **what counts as a "hit"?**

Two candidate signals (pick one or AND/OR them):

- **Topic-classifier signal**: run a cheap classifier (e.g. spaCy NER
  density, or a small transformer) over the `small.en` transcript.
  Hit = ≥N named entities OR ≥X domain-specific topic terms.
- **Length / dialogue signal**: hit = `small.en` transcript has
  ≥N speakers identified by speaker-detection.

For the design doc, we pre-measure the candidate `r` against the
prod-v2 corpus with the chosen signal:

```bash
# On a laptop with cached small.en transcripts:
for ep in <99 prod-v2 episodes>; do
  python -c "
    from podcast_scraper.kg.ner_prepass import extract_kg_ner_hints
    import spacy
    nlp = spacy.load('en_core_web_sm')
    text = open(f'small.en/{ep}.txt').read()
    hints = extract_kg_ner_hints(text, nlp, max_candidates=40)
    print(f'$ep,{len(hints)}')
  "
done | awk -F, '{print $1, $2 >= 5 ? "HIT" : "SKIP"}'
```

Then `r = HIT_count / 99`. Sanity-check across episode types
(podcasts vs interviews vs solo monologues).

**If `r ≥ 0.9`**: the gate is pure overhead — abandon the sniff
pass and either (a) ship single-model with the better model, or
(b) reconsider the gate criterion.

**If `r ≤ 0.5`**: clear win, proceed to per-request model selection
implementation.

**If `0.5 < r < 0.9`**: marginal — depends on whether the latency
ratio holds at the corpus mean, and whether `small.en` quality
suffices for the downstream topic-detection step. Run the
measurement; let the data decide.

### 3c. Validate `small.en` quality for the gate decision

The gate's correctness depends on `small.en` being good enough to
classify topics. Methodology:

1. Transcribe 10 episodes with `large-v3` → "gold" transcripts.
2. Transcribe same 10 with `small.en`.
3. Run the gate criterion on both transcripts.
4. Confusion matrix: (`small.en` hit, `large-v3` hit) — count
   agreements.
5. Acceptance: ≥85% agreement (false-negative + false-positive
   combined ≤15%) is the floor. Lower → small.en is too coarse.

## 4. GPU memory budget — coexistence with autoresearch

Per the issue body and `DGX_RUNBOOK.md § GB10 unified memory`:

- GB10 unified memory: 128 GiB total.
- `large-v3`: ~3 GiB.
- `small.en`: ~0.5 GiB.
- pyannote (`Up 10 days`): ~3 GiB resident.
- autoresearch vLLM (Qwen3-30B-A3B-FP4): ~50-60 GiB during eval sweeps.

With both Whisper models resident: ~3.5 GiB total for Whisper.
Combined with autoresearch peak (~60 GiB) + pyannote (3 GiB) +
Ollama buffer (~10 GiB residual) → ~76 GiB used, ~52 GiB headroom.
**Coexistence is structurally fine.**

The risk is **contention timing**: when an autoresearch sweep
is mid-run and a transcription request lands, the small.en hot path
must complete without OOM-evicting the autoresearch model. Per
`gpu-mode-swap.sh` discipline, autoresearch sweeps are operator-
scheduled — they don't run concurrent with prod transcription. So
this contention is **manageable by scheduling**, not a hard tech
constraint.

**Measurement**: during 3a, observe `dcgm-exporter` (`:9400`) GPU
memory metrics. Verify both Whisper models stay resident (no
unload-on-LRU thrashing) across the 20-call sequence.

## 5. Per-request model selection — design sketch

### 5a. Server side (speaches container)

speaches already supports multi-model. The only config change is
the model cache directive on the container:

```yaml
# agentic-ai-homelab/infra/dgx/speaches/docker-compose.yml
services:
  faster-whisper:
    environment:
      # Pre-load BOTH models so the first sniff-pass request
      # doesn't pay the ~1s cold-load cost.
      WHISPER_MODELS: "Systran/faster-whisper-small.en,Systran/faster-whisper-large-v3"
```

(Exact env-var name depends on speaches version — verify at apply
time. If the version doesn't support multi-model preload, a startup
script that fires two transcription calls with each model name
warms the cache.)

### 5b. Pipeline side — Config knob + factory plumbing

Two new Config fields (already prefixed `dgx_`):

```python
# src/podcast_scraper/config.py
dgx_whisper_sniff_model: str = Field(
    default="",  # "" disables the sniff pass entirely
    alias="dgx_whisper_sniff_model",
    description=(
        "When set, transcribe the episode with this Whisper model "
        "first, then re-transcribe with `dgx_whisper_model` only if "
        "the sniff-pass gate fires. Empty disables (single-model)."
    ),
)
dgx_whisper_sniff_gate_min_entities: int = Field(
    default=5,
    alias="dgx_whisper_sniff_gate_min_entities",
    description=(
        "spaCy NER entity count required on the sniff transcript to "
        "trigger the deep transcription pass. Below this → skip."
    ),
)
```

Wire-through in `transcription/factory.py`:

```python
# Pseudocode — gated path
def transcribe_with_gate(audio_path, params):
    sniff_model = cfg.dgx_whisper_sniff_model
    if not sniff_model:
        return transcribe(audio_path, model=cfg.dgx_whisper_model)
    sniff_text = transcribe(audio_path, model=sniff_model)
    if gate_fires(sniff_text, cfg.dgx_whisper_sniff_gate_min_entities):
        return transcribe(audio_path, model=cfg.dgx_whisper_model)
    # gate didn't fire — keep the sniff transcript
    return sniff_text  # marked as 'sniff_only' in metadata
```

Default behaviour: `dgx_whisper_sniff_model=""` → no gate, every
episode goes straight to the configured model. Same as today.
Operator opts into the gate by setting the env var.

### 5c. Artifact provenance

The pipeline metadata records WHICH model produced each transcript:

- `transcription.model_used: "Systran/faster-whisper-large-v3"`
  (with the gate name + version)
- `transcription.gate_path: "deep" | "sniff_only"` — which branch
  the gate took.

Downstream consumers (silver eval, quality dashboards) can filter on
`gate_path` to compare quality between the two paths.

### 5d. Fallback

When the gate decision is uncertain (e.g. spaCy fails), default to
the deep path — never silently skip the deep transcription on a
classifier error. The gate is opt-IN deep-skipping; it's not allowed
to opt-OUT deep transcription on a flake.

## 6. Operator runbook stub (to land alongside the service code)

When the gate ships, `docs/guides/DGX_RUNBOOK.md` gets:

- A "Whisper sniff-pass gate (#1046)" section with the env var
  conventions + how to disable for a one-off corpus.
- The fallback contract (gate failure → deep path, never silent
  skip).
- How to verify the gate fired on a known episode (curl `/v1/audio/
  transcriptions` with `model=small.en`, observe metadata
  `gate_path=sniff_only` if the entity count is below threshold).

## 7. Acceptance milestones

In order:

1. **Section-3 measurements collected** — `(T_small, T_large, r,
   small.en gate accuracy)` filed back into this doc with real
   numbers. Operator-attended on DGX (~1h of compute).
2. **Decision gate**: if `r ≥ 0.9` OR `gate accuracy < 85%`, **stop
   here** — the design isn't viable. Close #1046 with the measurement
   data as the artifact.
3. **Service code lands** — Config knobs + factory plumbing + tests.
4. **Two-model serving validated under load** — eval harness runs a
   batch of 25 episodes through the gate path; assert (a) no GPU
   memory thrash, (b) gate decisions match the offline measurement.
5. **Operator runbook addition** — gate on/off recipe + fallback.

## 8. Open questions for operator

- **Q1**: What's the right gate criterion? (NER density vs speaker
  count vs domain-keyword match vs combination). The methodology
  scaffolds NER density as a starting point but the operator may
  want a different signal.
- **Q2**: When the gate doesn't fire, is the sniff-only transcript
  **shipped** (saved to artifacts as the canonical transcript), or
  **discarded** (the episode is treated as not-transcribed)?
  Different downstream impacts.
- **Q3**: Is there a different first-pass strategy worth measuring
  alongside (e.g. transcribe the first 60s only, decide from that)?

These don't block this design doc but DO block the service code.
Operator decides before milestone 3.

## 9. Why this isn't a "just ship it" project

Per the issue body: "Acceptance: Design doc with the (r, latency,
accuracy) measured estimate before any service code lands." That's
the operator's discipline guarding against an obvious anti-pattern
— ship the feature, find out empirically that it doesn't save
anything (or worse, regresses quality). The design doc + measurement
pass forces the cost-benefit conversation BEFORE the engineering.

Estimated effort post-greenlight:
- Section-3 measurements: ~1h on DGX (operator-attended).
- Service code (Config + factory + tests): ~half-day.
- Validation under load + runbook: ~half-day.
- **Total**: ~1-2 days, gated on the section-3 numbers.

## 10. What this doc explicitly does NOT do

- Ship gate ORCHESTRATION code in this branch. Per acceptance criteria
  the orchestrator (call sniff → run gate → conditionally call deep)
  waits for measurements.
- Predict whether the gate will pay off. The empirical measurements
  are the answer; this doc just specifies how to collect them.
- Pre-commit to a specific gate criterion (Q1 above is open).

## 11. What HAS shipped — provider plumbing (2026-06-22)

The PROVIDER-side knobs landed so the orchestrator drops in cleanly
once measurements come back. Behaviour is unchanged today
(``dgx_whisper_sniff_model`` defaults to ``""`` which disables the
sniff path entirely; single-model behaviour preserved).

### Config knobs (off by default)

- ``cfg.dgx_whisper_sniff_model: str`` — empty string disables; set
  to e.g. ``Systran/faster-whisper-small.en`` to make the override
  available.
- ``cfg.dgx_whisper_sniff_gate_min_entities: int`` — default 5
  (placeholder until measurement-pass pins the optimal value).

### Provider — per-call model override

``TailnetDgxWhisperTranscriptionProvider.transcribe_with_segments``
now accepts an optional ``model_override: str | None`` parameter.
When set, the multipart form's ``model`` field carries the override
(not the configured default). Threaded through
``_transcribe_with_fallback`` → ``_transcribe_dgx`` so the
health-check substring also matches the effective model.

### Provenance

The returned dict now includes a ``model_used`` field with the
effective model id. Downstream artifact writers can record which
model produced each transcript for ``gate_path`` accounting
(``sniff_only`` vs ``deep`` once the orchestrator lands).

### Tests

Five new cases in
``tests/unit/podcast_scraper/providers/test_tailnet_dgx_whisper.py``:

- override propagates to the multipart form's ``model`` field
- None override falls back to ``cfg.dgx_whisper_model``
- empty-string override (the disabled-sniff default) falls back too
- public ``transcribe_with_segments`` threads override end-to-end +
  records ``model_used`` provenance
- default path records the default model in ``model_used``

### What's still missing for #1046 acceptance

- The 3 measurement passes from section 3 (operator-attended ~1h on
  DGX).
- Operator decisions on the 3 open questions in section 8.
- The gate orchestrator itself (workflow-level helper) — wires the
  Config knobs to the provider's ``model_override`` parameter.
- The compose change on DGX to preload both Whisper models (so the
  first sniff call doesn't pay ~1s cold-load).
- Operator runbook section (when the gate ships).
