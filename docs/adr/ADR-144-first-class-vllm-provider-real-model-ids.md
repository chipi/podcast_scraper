# ADR-144: First-class `vllm` provider — real model ids + fail-closed served-model verification

- **Status**: Accepted
- **Date**: 2026-08-02
- **Authors**: Marko Dragoljevic, Claude (Opus 4.8)
- **Related**: [ADR-044](ADR-044-local-llm-backend-abstraction.md) (local LLM backend
  abstraction), [ADR-143](ADR-143-corpus-reprocess-methodology.md) (reproducible
  single-variable corpus arc — the reason this matters now), [ADR-124](ADR-124-model-governance-registry-sanctioned.md)
  (registry-sanctioned model governance); agentic-ai-homelab `infra/vllm/autoresearch/`
  (the compose that owns `--served-model-name`)

## Context

DGX-served models reach the pipeline over vLLM's OpenAI-compatible HTTP API on `:8003`. Rather
than a first-class provider, that path was bolted onto the existing `openai` provider: a
StageOption sets `provider="openai"` with the endpoint pointed at vLLM. The homelab starts vLLM
with `--served-model-name autoresearch` — a *stable alias* so repo config need not change when
the slot's weights swap — so every DGX StageOption sets `model="autoresearch"`, and the real
weights survive only in a comment / `extra_settings.underlying_hf_model`.

Three concrete defects (verified against the tree, branch `feat/naming-arc-and-corpus-prep`):

1. **The registry does not govern the wire today — renaming it alone changes nothing.** The
   provider consumes `openai_api_base` / `openai_summary_model`, and in the DGX profiles those
   are **hand-authored above the `registry-materialized — do not hand-edit` divider**
   (`config/profiles/prod_dgx_full.yaml`).
   The materializer emits `summary_model` / `summary_endpoint`
   (`model_registry.py` resolver,
   `_emit`*), but `summary_endpoint` is not a Config field and `materialize_profiles.py`
   deliberately drops non-Config resolver keys; the materialized `summary_model: autoresearch`
   is **decorative** — read only by metadata provenance
   (`workflow/metadata_generation.py`).
   So the wire model + endpoint are governed by **nothing**.
2. **The alias hides identity → reproducibility is violated.** Even the hand-authored key reads
   `autoresearch`, so **you cannot read a profile and know what produced the corpus**; the only
   pin is a human loading the right compose. Directly violates ADR-143's single-variable rule.
3. **The abstraction is asymmetric.** Ollama options self-describe (`provider="ollama"`,
   `model="qwen3.5:35b"`); vLLM options name a *transport* (`openai`) and an *alias*
   (`autoresearch`). The layer types by wire protocol, not serving stack, so vLLM has no identity.

`autoresearch` is the historical name of the single GB10 vLLM slot (born for judge/scoring runs),
leaking into production model identity. The slot is GPU-mode-swapped to serve whatever weights are
loaded — sometimes a judge, sometimes the production naming/summarization model. The v2.5 corpus
arc (Gemini → DGX-local LLM swap) makes a reproducible, self-describing, **governed** representation
a prerequisite (**Step 0**), not a nicety.

## Decision

**Make the serving stack a first-class provider dimension. Name + govern real models on the wire.
Verify the served model fail-closed at init. Take every DGX LLM stage local together. Delete
`autoresearch` from production config via a phased migration.**

1. **New `vllm` provider — a distinct sibling of `openai`, not a subclass.** With vLLM we serve a
   wide family of *non-OpenAI* open models (Qwen, DeepSeek, Llama, …); `openai` is reserved for
   OpenAI-native models. They share only a wire protocol, so the OpenAI-compatible transport is
   factored into a common base (`OpenAICompatibleProvider`) that **both** `OpenAIProvider` and
   `VLLMProvider` extend as siblings. `VLLMProvider` owns a `vllm_*` config namespace.
   **Sequencing (the extraction is a real refactor, not a lift):** `OpenAIProvider` is ~2,780
   lines with the `openai_*` namespace + `provider="openai"` telemetry baked into nearly every
   method, plus OpenAI-native heuristics (`sk-` key check, `o1/o3/gpt-5` temperature rules). Land
   the **config-namespace parameterization** (`self._ns` → `f"{ns}_summary_model"`, threaded
   `provider_name` telemetry) as its **own zero-behaviour-change commit with existing tests green**,
   *then* introduce `VLLMProvider`.
2. **Registry governs the wire with real ids (closes defect 1).** Add Config fields
   `vllm_api_base`, `vllm_summary_model`, `vllm_speaker_model`, `vllm_api_key_env`; add them to
   `REGISTRY_GOVERNED_FIELDS`; add `_emit_summary_model` / `_emit_speaker_model`-style routing
   (precedent: `_emit_transcription_model`) so the materializer writes the **provider-consumed**
   keys, killing the hand-authored block. **Materialize the endpoint in *template* form**
   (`http://${DGX_TAILNET_HOST:-…}:8003/v1`), never `resolve_endpoint()` output (which bakes the
   materializer-runner's env or a fail-fast sentinel into the YAML). DGX StageOptions become
   `provider="vllm"`, `model="<real HF id>"` (e.g. `NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4`);
   `underlying_hf_model` + `autoresearch` are removed.
3. **Every producing LLM call goes DGX-local — one model choice across the whole series (closes
   B1, the corpus-corrupting one).** *Single-variable* (ADR-143) means **one LLM choice applied
   across every producing call**, not one call: the single DGX model replaces Gemini for
   **summary, naming/labeling (`ner`), GI insight, KG extraction, and quote/entailment
   grounding** — no cloud LLM is left in a DGX-local corpus. Summary, GI-insight, and KG already
   ride the summary provider (`gi_insight_source` / `kg_extraction_source: provider`), so they
   follow automatically; the stages that must be **explicitly flipped** in the DGX profiles are
   `speaker_detector_provider` (naming, today `gemini`), `quote_extraction_provider`, and
   `entailment_provider` (grounding, today `openai`). Left unflipped, the grounding stages build a
   *separate* `OpenAIProvider` with no `openai_api_base` → **api.openai.com + `gpt-4o-mini`**
   silently produces the "DGX-local" grounding
   (`gi/deps.py` instance-reuse breaks under the split). Add
   `vllm` to the evidence-provider enum + `GIL_EVIDENCE_ALIGN_SUMMARY_PROVIDERS` + the `deps.py`
   match; **add a test that a vllm-summary profile builds zero cloud-OpenAI clients.**
   **The in-pipeline value-gate *judge* also goes local** (operator, 2026-08-03 — see the As-built
   amendment; this REVERSES the earlier "cross-vendor cloud ≥Sonnet judge" decision). A DGX profile
   must consume nothing from the internet, so the value gate self-grades with the **same local
   model** as the extractor (`vllm`/`ollama` join `_LOCAL_ONLY_LLM`; no cloud judge is pinned). The
   #939 self-grading leniency is the accepted cost of airgapping — the gate still trims the clear
   filler, just conservatively; a distinct **second local** judge (still airgapped) is a future
   autoresearch evaluation point. Because one model now drives many stages, the *parity* gate (the
   separate autoresearch scorer, not this in-pipeline gate) must evaluate **per stage /
   multi-perspective** (summary, naming, GI, KG), not summary alone.
4. **Fail-closed verification at provider init (closes B3).** The cost-telemetry `served_model`
   capture never fires on the DGX path (the summary cost event is gated on `cost>0` and the vLLM
   slot is priced `$0`). Instead, on `VLLMProvider` construction, `GET /v1/models` and assert the
   configured id ∈ the served-id set — reusing the existing pattern in
   `onboard_model_smoke.py` and the normalization in
   `verify_served_model` (casefold + dated-suffix tolerance; **do not strip the org prefix**).
   Raise a config error on mismatch. A cheap per-response check may follow, but the startup
   set-membership check is the robust seam.
5. **Pricing + governance renamed in the same commit (closes B4).** A CI guard fails PRs that
   reference a model with no pricing row. Add a `vllm` section (real ids, `$0`) to **both**
   `config/pricing_assumptions.yaml` and `src/podcast_scraper/data/pricing_assumptions.yaml`, and
   to both `known_models.yaml` copies; state the governance decision for local weights (whether
   `vllm` joins `governed_providers`); remove `autoresearch` from the `openai` lists.
6. **Naming (`ner`) becomes provider-symmetric (S4).** `speaker_detector_provider` accepts `vllm`;
   add a `vllm_speaker_model` field + emission (the OpenAI speaker path reads `openai_speaker_model`,
   *not* `ner_model`, so a bare StageOption reaches nothing) and a `vllm_speaker_detector`
   StageOption pointing at the DGX real model. Representation-only until the naming bake-off picks
   the model.

**Full enum/dispatch surface (S2)** — every site that must learn `vllm`:
`_validate_summary_provider`, `_validate_evidence_providers`, and the `speaker_detector_provider` Literal (`config.py`);
`summarization/factory.py` (+ its error string); `speaker_detectors/factory.py`; the two
`llm_providers` sets in `workflow/helpers.py`; `_cleaning_model_for_summary_provider`
(with `vllm_cleaning_model` defaulting to the **served** model, not `gpt-4o-mini`);
`resolve_value_gate(sm.provider)`; the metadata-provenance branch (so the sidecar records the model —
non-negotiable for a reproducibility ADR); run-suffix naming (cosmetic).

### Migration plan (phased — mandatory, S3)

A hard cutover breaks reruns forever: frozen `data/eval/configs/` (never-mutate) and live sweep
drivers / judges (`evaluation/judges/vllm_chat.py`) reference `autoresearch`.

- **Phase A.** Homelab serves **both** names `[<real-id>, autoresearch]`, **real id first**
  (`/v1/models data[0].id` is the first entry; the init check keys off set-membership so order is
  safe, but old smoke configs asserting `data[0]` will fail during transition — expected).
- **Phase B.** Land this repo's change requesting the real id; migrate **live** (non-frozen) eval
  tooling off the alias (`onboard_model_smoke.py` default, sweep drivers, judges).
- **Phase C.** Drop the alias. Frozen `data/eval/configs/` referencing `autoresearch` are
  **accepted as retired** (never-mutate → not rewritten); new real-id configs are created
  downstream. We do **not** keep serving the alias to preserve old reruns.
- **Unverified, verify on the DGX first:** what `response.model` echoes under multi-name serving on
  the pinned `vllm:26.05` build. If per-call equality is chosen it must match empirically; the
  startup set-membership check does not depend on it.

## Consequences

**Positive:** a profile is self-describing **and** governed — `summary_provider: vllm` +
`vllm_summary_model: <real id>` is the wire truth, enforced (fail-closed) not trusted; grounding
stays local (no silent cloud leak); `vllm`/`ollama` are symmetric stacks; `openai` stops
double-serving as a covert vLLM shim.

**Negative:** large surface — a namespace refactor of a 2,780-line provider, ~9 dispatch/enum
sites, 4 pricing/governance files, endpoint-materialization plumbing, and the GI grounding flip.
Cross-repo **phased** migration (homelab + repo, three phases); losing the floating alias means a
homelab model swap now requires a matching registry/profile edit (the point, but friction); frozen
eval configs pin `autoresearch`, so the alias can only be *retired for live tooling*, not deleted.

**Neutral:** the `OpenAICompatibleProvider` extraction is a config-namespace parameterization
threaded through every `cfg` read, sequenced as its own no-behaviour-change commit; the existing
non-DGX `openai` (cloud) path is untouched.

## Alternatives considered

- **Keep `provider="openai"`, only replace the `autoresearch` string with the real id.** Less
  code — but keeps the transport-vs-stack conflation the operator rejected; the YAML still lies
  about the stack, and (per defect 1) the string is hand-authored, so it still isn't governed.
- **Keep the floating alias, record the real id in a comment + verify at runtime.** The wire still
  cannot name the model and a reader still cannot trust the profile. Rejected — the convenience is
  exactly what broke reproducibility.
- **A generic `openai_compatible` provider keyed by base_url** (covers vLLM + Ollama-OpenAI + cloud
  in one branch). Erases the stack identity the operator wants surfaced; muddies per-stack model
  governance. Rejected.
- **Composition instead of a shared base** (a `vllm` class translating `vllm_*`→`openai_*` into an
  internal transport). Materially cheaper than the base extraction, but contradicts the locked
  "common base, siblings" decision. Noted as a cost, not re-litigated.

## References

- `model_registry.py` — `StageOption`,
  DGX options, resolver/`_emit`*, `REGISTRY_GOVERNED_FIELDS`, `resolve_endpoint`
- `openai_provider.py` — the
  transport `VLLMProvider` extraction is factored from; OpenAI-native heuristics to keep out of the base
- `summarization/factory.py`,
  `speaker_detectors/factory.py` — provider dispatch
- `gi/deps.py` — the quote/entailment instance-reuse that B1 turns on
- `config.py` — provider enums, env-presence validators, `vllm_*` fields
- `config/` + `src/podcast_scraper/data/` — `pricing_assumptions.yaml`, `known_models.yaml` (both copies)
- `workflow/helpers.py`, `workflow/metadata_generation.py`, `scripts/eval/onboard_model_smoke.py`,
  `evaluation/judges/vllm_chat.py`, `materialize_profiles.py`
- agentic-ai-homelab `infra/vllm/autoresearch/docker-compose.yml` — owns `--served-model-name`

## As-built amendment (2026-08-03)

Implemented across commits (ADR `56fa7bc4` → B3 `763ab4d0`). Two operator decisions during
implementation extended/changed the plan:

1. **Fully airgapped DGX profiles.** Beyond swapping the producing LLMs, the operator required the
   DGX profiles to consume **nothing** from the internet — every LLM stage AND every fallback is
   DGX-local. As built (prod_dgx_full, eval_default):
   - summary / naming / GI / KG / quote / entailment → `vllm` (real Qwen id);
   - summary fallback → **DGX-local ollama** (`:11434`), not cloud gemini;
   - transcription → DGX-whisper + local in-process whisper + MOSS coverage failover (**no cloud
     Whisper**); diarization → local pyannote (**no cloud deepgram**);
   - a test asserts all three load with zero cloud API keys and hold no cloud provider anywhere.
2. **Value gate self-grades local** — reverses the §3 cross-vendor-cloud-judge decision (see §3).

**Ollama symmetry (now FULL).** `vllm` and `ollama` are the two DGX-local serving stacks and are
fully symmetric: both first-class providers, both self-grade the value gate (`_LOCAL_ONLY_LLM`),
both price at `$0`, neither cloud-governed, ollama is the airgapped summary fallback, AND ollama's
wire config (`ollama_summary_model` / `ollama_speaker_model` / `ollama_api_base`) is now
registry-governed + materialized like vllm's — whether ollama is the primary (experiment_dgx_only)
or the airgapped summary fallback (`_emit_fallback_chains` emits the fallback ollama option's wire
config). The blocker that had deferred this — an LLM naming StageOption's `model` being a **spaCy**
id, not the LLM tag — is resolved by splitting the two: `model` = the spaCy `ner_model`, and the
LLM tag lives in `extra_settings['speaker_llm_model']` → `{ns}_speaker_model`, identically for vllm
and ollama. **This also fixed a latent bug**: the vllm naming option had been leaking its Qwen HF
id into `ner_model` (a spaCy field) — `spacy.load()` would have crashed if the entity stage ran.

**Phased homelab migration (S3) — NOT YET DONE.** The registry/profiles now request the real HF id
on the wire; the homelab vLLM must serve under that `--served-model-name` (dual-name transition)
before a live run passes the B3 check. Prepared as a handoff, not deployed — see the session
handover in `docs/wip/`.
