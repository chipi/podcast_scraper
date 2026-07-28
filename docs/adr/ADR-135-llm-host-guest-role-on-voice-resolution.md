# ADR-135: LLM host/guest role determination on the existing voice-resolution call

- **Status**: Accepted
- **Date**: 2026-07-28
- **Authors**: Marko Dragoljevic
- **Related issues**: v2.3.1/v2.3.2 pilot host/guest gaps; F1/F2 seat-fill fix (roster `stated_non_host_voices`)
- **Related work**: ADR-110 (metadata-first speaker resolution), #876 (a wrong name is worse than none), #1169/#1170 (role confusion), #1226 (merged-cluster host cap)
- **Design source**: this ADR (operator proposal 2026-07-28 + advisor review)

## Context & Problem Statement

The v2.3.1 pilot surfaced host/guest gaps the deterministic roster cannot close, because they need
knowledge the metadata does not carry:

1. **No metadata host anchor** — rotating/narrator-host shows (Planet Money) state no fixed hosts, so
   the show's own hosts (Robert Smith, Brittany Luce) are labeled guests. Heuristics have no anchor.
2. **Guest naming miss** — a stated guest (Latent Space / Eiso Kant) is in `metadata_named` but never
   self-introduces and is never bound to a voice; the two hosts aren't in `known_hosts` either → all
   voices unresolved.
3. **Role ambiguity** — a present voice whose host-vs-guest role the cues can't decide (`role=unknown`).

The deterministic fixes (F1/F2, ADR-110) are correct and stay in charge. What is missing is a
judgement call — *given the title, description, and the intro, which voice is the host and which is
the guest* — that only a language model can make from unstructured framing.

The pipeline **already runs one LLM call per episode** to match diarized voices to metadata names
(`resolve_voices_from_conversation` → `build_resolution_prompt`, ADR-110). It answers *"which of
these named people is each voice?"* from a **closed candidate list**, and it is **provider-agnostic**
(dispatched via `completion_fn_for(provider).complete_text`; 7 LLM providers implement it, local
non-LLM summarizers return `None` and keep the deterministic path). It does **not** produce a role,
and it is **not** shown the episode title, description, or the intro.

## Decision

**Extend that existing call to also return host/guest role — one call, flat cost — and feed it three
role-bearing inputs it does not see today.** No new LLM round-trip; no new provider method.

### Payload additions (the "query")

The single prompt (`build_resolution_prompt`) gains, alongside the existing closed candidate list +
per-voice speech samples:

1. **Episode title**
2. **Episode description**
3. **The labeled, cleaned intro** — the first ~500 words of the diarized transcript with speaker ids
   (`SPEAKER_00: Welcome to…`), **restricted to real voices** (ads / cameos / commercials removed).
   This is the highest-signal input for role: the intro is where "I'm your host X, my guest is Y" is
   said, tied to the voices we are classifying. The LLM is asked only about **real voices** — never a
   cameo/commercial/ad, on which it must abstain anyway.

### Cleaning: one classifier, computed once (the ordering fix)

The LLM call runs *before* the roster (it feeds the roster's naming), so it cannot consume the
roster's cleaning. Rather than **replicate** the roster's cameo/commercial rule at the LLM site (the
rejected first cut), the deterministic real-vs-noise classification is **extracted into one function,
`roster.classify_voices(diarization, ad_intervals, …)`**, computed once right after diarization and
consumed by **both** the LLM call and the roster:

- **ad** = `_edge_ad_voices` + the cross-episode recurring-ad strategy (#1188);
- **cameo** = under `CAMEO_MAX_TALK_S` total talk; **commercial** = mostly inside `ad_intervals`;
- **real** = everything else.

This is possible because ad/cameo/commercial are all deterministic from `(diarization, ad_intervals)`
— naming is *not* required (only the finer person/unknown/unidentified split is, and that stays in
the roster). The roster now receives the classification (`cleaning=`) and uses `cleaning.ad` for its
ad set and `cleaning.commercial/cameo` for its voice-typing instead of recomputing them, so "which
voices are noise" is defined in exactly **one place** and the LLM call and the roster can never
disagree. Standalone/relabel callers that pass no `cleaning` get the identical computation inline
(behaviour-neutral).

### Output schema

Per voice, `{ "name": "<from closed list> | null", "role": "host | guest | null" }`. `name` keeps the
closed-list discipline unchanged (the model matches, never authors). `role` is a new field; `null` is
correct and expected.

### Guardrails (in code, never the prompt — #876)

The LLM role is advisory input the roster **verifies**, mirroring the existing `_refuted_by_third_person`
enforcement. It may only:

- **veto** a positional host guess (a voice the LLM calls `guest` joins `conv_guests` in the roster's
  seat-fill exclusions), and
- **anchor** an empty host pool (on a no-stated-host show, a voice the LLM calls `host` may seat).

It may **never** unseat a voice that self-introduced as a stated host, promote a `_refuted`/`conv_guests`
voice, or push the host count past what the feed stated. Role vocabulary is closed to `{host, guest, null}`;
anything else is discarded and logged, exactly like an invented name.

### Provider rollout

Gemini is the POC (the pilot profile's `summary_provider`), riding the existing dispatch with zero new
provider code. Because all 7 LLM providers already implement `complete_text`, rollout is **validation,
not reimplementation**: confirm each model's role JSON parses (`_parse` tolerates fenced/`<think>`
preambles and both the legacy string and new object forms).

## Alternatives considered

- **A separate LLM host/guest call.** Rejected — doubles per-episode LLM calls for no benefit; the
  existing call already has the voices, the candidates, and the retrieved passages in context.
- **Replace the heuristic host classifier with the LLM.** Rejected — airgapped/spaCy profiles have no
  completion endpoint; the deterministic path must stay complete and correct on its own.
- **Prompt-only guardrails.** Rejected — a prompt is not an enforcement mechanism (#876). Role limits
  are enforced in code.

## Consequences

- **Positive**: closes the pilot's category-1/2/3 gaps; flat cost; multi-provider on day one;
  metadata-first preserved (role is a conversation claim, not a WHO the model authors).
- **Negative / risk**: a model may flip a correct heuristic result — bounded by the veto/anchor-only
  guardrails and the closed role vocabulary. Non-determinism is contained to the LLM path; airgapped
  runs are unchanged and CI stays deterministic (no real LLM in CI; canned-completion unit tests).
- **Non-goal**: naming genuinely anonymous crowd/field voices (pilot category 4). The call must
  **abstain** on those — forcing a name is the exact #876 failure.
