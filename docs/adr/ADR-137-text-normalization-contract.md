# ADR-137: Text normalization contract (match / display / raw) — a pipeline-wide core primitive

- **Status**: Proposed
- **Date**: 2026-07-29
- **Authors**: Marko Dragoljevic
- **Related**: labeling failures on The Daily / WSJ / lowercase-turbo episodes (prod-v2.4-100ep);
  ADR-110 (name resolution against a closed metadata list), ADR-128 (ASR-mangled-name canonicalizer),
  ADR-135 (additive LLM role resolution).
- **Supersedes/absorbs**: the ad-hoc per-site casing handling scattered across ~40 modules.

## Context & Problem Statement

Text arrives from many producers (turbo ASR, openai-whisper ASR, feed metadata, LLM output) in
**inconsistent surface form** — most visibly **casing**: DGX turbo emits lowercase, openai-whisper
truecases, feeds are Title Case. Every downstream matching/recognition component then makes its own
assumption about that surface, so **the same input resolves in one component and not another, and a
given component resolves or fails depending on which ASR happened to run.**

Measured, current:

- **Name recognition is capitalization-dependent by construction.** `_NAME` in
  `speaker_detectors/hosts.py` is `(?-i:[A-Z][\w'-]+(\s+[A-Z][\w'-]+)+)` — IGNORECASE is explicitly
  **disabled** for the name capture so it won't match lowercase phrases. Consequence: on lowercase
  turbo episodes the self-intro and host-introduces-guest regexes capture **nothing**. Corpus
  evidence: lowercase episodes produced **0 self_intro names** (n=13) vs 1.47/ep on truecased.
- **~40 modules do their own `.lower()/.casefold()/IGNORECASE`** — `gi/*`, `kg/*`,
  `identity/resolver.py`, `identity/slugify.py`, `cleaning/*`, `enrichment/*`. There is **no single
  source of truth** for "what is the comparable form of this text", so entity identity, quote
  binding, search, and speaker naming can and do disagree.

Casing is an **ASR/producer artifact, not reliable signal**. Depending on it is the bug.

## Decision

Adopt a **text-normalization contract as a core pipeline primitive**. It applies **everywhere text
is matched or recognized — not only the three labeling failures that surfaced it, but every
component, including the ones that already work.** The contract has three parts.

### 1. Three canonical forms

| Form | Definition | Who holds it |
| --- | --- | --- |
| **raw** | verbatim producer output + provenance; never mutated | storage of record; the source others derive from |
| **display** | human-readable: original/restored case + punctuation | player transcript, screenplay labels, any human surface |
| **match** | `NFKD` → strip combining marks (diacritic-fold "Gómez"→"gomez") → casefold → normalize quotes/hyphens → collapse whitespace | **every recognition / matching / dedup / indexing path** |

### 2. Force-at-your-own-boundary rule

**Every stage normalizes its own INPUT to the form it needs; it never trusts the surface the
previous stage emitted.** A matching stage casts to **match-form at entry**, unconditionally.
Producers remain free to emit any surface — the burden is on the consumer, at the boundary. This is
the "interface" the pipeline is missing: casing (and unicode/whitespace/punctuation drift) can no
longer change a downstream result.

### 3. One SSOT

`normalize_for_match(text)` (+ `normalize_name_for_match(name)` for the name-token variant) live in
**one module** (`text_normalization.py`). Every matching site calls it; **no matching path keeps its
own `.lower()`/`casefold`**. `identity/slugify.py` becomes a thin caller (its NFKD+lowercase is a
stricter ID-form built on top).

### 4. Provider boundary — providers do NOT lowercase; they emit raw + declare their surface

A tempting alternative is to lowercase at each **provider's exit** ("every ASR returns lowercase").
**Rejected — it is the wrong boundary and it is lossy.** Lowercasing on the way *out* destroys the
truecased form permanently, and two consumers need it:

- **Display** — the player transcript / screenplay must be readable. openai-whisper *gives* us good
  case + punctuation; folding it away at the provider degrades the player for no gain. (Turbo
  already emits lowercase — that is a display *deficit* to potentially fix by *restoring* case, the
  opposite direction, not a reason to drag the others down.)
- **The no-metadata discovery fallback** (below) — capitalization is its only signal.

The contract's premise is **fold at the consumer's boundary, never trust the producer**, so what a
provider emits does not matter to correctness. Therefore the provider-boundary rule is:

1. Providers emit **raw/verbatim** text (+ provenance) — they do **not** normalize casing.
2. Providers **declare their surface form** (`lowercase-unpunctuated` | `truecased-punctuated`) so
   display logic can decide whether to truecase-restore. *(Deferred — not yet implemented; no
   provider declares a surface form today. Consumers fold to match-form at their own entry
   regardless, so this is a display-restore optimization, not a correctness dependency.)*
3. **No consumer relies on a provider's casing.** Matching folds to match-form at its own entry;
   this is what makes the pipeline robust to a heterogeneous provider fleet (turbo lowercase,
   openai truecased) without any provider change.

This gives cross-provider consistency **without** the information loss of exit-lowercasing.

## Key consequence: match-form is case-blind, so name-finding must be metadata-anchored

Capitalization is doing **two** jobs today; only one survives:

- **Matching** ("is this *known* name spoken here?") — already case-insensitive; pure win under the
  contract.
- **Discovery** ("find a name I *don't* know from raw text") — today the *only* signal that
  "rich gelfond" is a name and not the adjective "rich" is the capital R.

Under match-form there are no capitals, so **discovery anchors on the metadata candidate list**: a
voice is named by matching its spoken tokens against the names the episode metadata already stated
(ADR-110's closed list), **case-blind and fuzzy** (nickname table + ADR-128 phonetic/edit-distance).
This is already how the labeling business is supposed to work — you may only assign a *stated* name.
Capitalization-based discovery survives **only** as a fallback where truecased text exists and no
metadata does (a no-metadata network show), behind the existing ordinary-word guards
(`looks_like_a_person_name`, `is_plausible_mononym`) — which must themselves run case-blind.

## Adopters (the pipeline-wide scope)

End state: **every matching boundary uses the contract; zero ad-hoc casefold in matching paths.**

- **Speaker recognition** — *first adopter (this change):* `providers/ml/diarization/roster.py`
  (self-intro, host-introduces-guest cues, canonicalizer), building its case-blind match-forms from
  the `speaker_detectors/hosts.py` cue vocabulary. *Not yet adopted (follow-up):*
  `speaker_detectors/resolution.py` (mention retrieval / refutation) still uses its own `.lower()`.
- **GI** — `gi/speakers.py`, `gi/grounding.py` (quote → speaker binding), `gi/filters.py`.
- **KG** — `kg/entity_clusters.py`, `kg/ner_prepass.py`/`ner_postpass.py`, `kg/filters.py`.
- **Identity** — `identity/resolver.py`, `identity/slugify.py`.
- **Search** — indexing + query normalization.
- **Cleaning / boilerplate** — `cleaning/commercial/detector.py`, ad/boilerplate text matching.

## Rollout (scope discipline — not a big-bang rewrite)

1. **Recognition first (this change):** introduce `normalize_for_match`, adopt it in `roster.py`
   (with the `hosts.py` cue vocabulary), make name discovery metadata-anchored + case-blind, add the
   nickname table, extend narrated-desk cue vocabulary. Fixes The Daily / WSJ / the 13 lowercase
   episodes. `resolution.py` adoption is a follow-up step, not part of this change.
2. **Then GI/KG/identity:** swap each layer's ad-hoc normalization for the SSOT, one layer per
   change, each behind a regression check that named/entity output is unchanged on already-good
   episodes.
3. **Then search/cleaning.**

Each step deletes ad-hoc casefolds as it lands; the contract is "done" when a grep for
`.lower()/.casefold()` in matching paths returns only the SSOT.

## Testing: the seam consistency invariant

The contract is enforced by a **case-invariance test across every recognition seam**: the same
content, fed **truecased** and **lowercased**, must produce the **same** output. This is the
regression guard that keeps a future ASR/provider change (or a new seam) from silently reintroducing
the casing dependency. Seams under test:

- `extract_self_introduced_host`, `_self_intros_by_voice`
- `_voice_named_by_the_introduction` (cue-first + name-first)
- `_canonicalize_to_stated_name` / the canonicalizer

Seams **to be tested at adoption** (not covered yet — resolution.py has not adopted the SSOT):
`resolution.py` — `retrieve_mentions`, `_introduces_itself_as`, `_refuted_by_third_person`.

**The one documented exception:** the invariant holds **when metadata is present** (a stated
candidate list anchors the match). The **no-metadata discovery** fallback *cannot* be case-invariant
— with no candidate list, capitalization is the only signal that a token run is a name, so lowercase
genuinely carries less information. That seam is tested to be *stable* (truecased still works,
lowercase degrades gracefully to "unnamed", never to a wrong name), not case-identical. This is the
direct consequence of §"match-form is case-blind", stated so no future reader mistakes it for a gap.

## Alternatives considered

1. **Truecase/restore capitalization on ASR output.** Rejected: model-dependent and fragile, adds a
   failure surface, and does nothing for the cross-layer inconsistency (each layer still casefolds
   its own way).
2. **Leave each layer ad-hoc (status quo).** This is the bug — layers disagree, results depend on
   the ASR.
3. **Only make the regexes `IGNORECASE`.** Insufficient: the `(?-i:[A-Z])` name capture is
   deliberately case-bound and would match every lowercase phrase if naively relaxed. Case-blindness
   *requires* the metadata-anchored discovery change, not just a flag flip.

## Consequences

- **Robust:** recognition/matching is ASR-agnostic — "which ASR ran" stops changing outcomes.
- **Consistent:** speaker naming, GI quote binding, KG entity dedup, and search share one comparable
  form and can no longer disagree.
- **Simpler:** one primitive replaces ~40 ad-hoc casefolds.
- **Cost:** requires the metadata-anchored discovery change in recognition + keeping the ordinary-word
  guards working case-blind; and a disciplined multi-step rollout so other layers migrate without
  regressing.

## Non-Goals

- **Not** truecasing or punctuation-restoring the *display* transcript — the display form keeps its
  readable case; only the match-form is folded.
- **Not** changing WHAT any name/entity resolves to — this is about the comparable form, not the
  meaning.
- **Not** a single-PR rewrite of all ~40 sites — recognition lands first; the rest migrate
  incrementally per the rollout.
