# P3 Consolidation — execution plan (PRD-041 / RFC-101, epic #1113)

- **Status**: Planning (P2 Capture shipped on `feat/consumer-remember`; this details P3).
- **Hard dependency**: the **Enrichment Layer (RFC-088 / Epic #1101)**. As of this writing #1101 is
  **complete in the `podcast_scraper-FUTURE` worktree** (all 9 chunks) and its PR is in flight. **P3
  coding starts once #1101 lands on `main`**, immediately after a `git fetch && rebase origin/main`.
- **Premise** (per operator): *something is probably already in — unify and build incrementally; do
  not rebuild.* This plan therefore leads with "what already exists" before any new code.

> **Reconcile-on-rebase note.** The envelope artefact layout, the `/api/corpus/enrichments*`
> reader routes, and the signal names below were read from the FUTURE worktree (RFC-088 as shipped).
> They are stable but **must be re-verified against `main` on rebase** before coding — if a field or
> route name shifted in review, update §3 and the affected child issues.

---

## 1. What already exists — we unify on it (no rebuild)

| Capability | Where it already lives | P3 reuse |
| --- | --- | --- |
| Highlights / notes / saved insights (per-user files) | P2: `app_user_state.py`, `/api/app/highlights*`, `/api/app/notes*` | The capture half of the personal corpus — recall + resurfacing read these directly. |
| Interests (cluster / `topic:` / `person:` tokens) | P2/3.5: `app_user_state.get_interests`, `rank_discover` | The seed of FR5 interest profile — extend, don't replace. |
| Listen log + playback positions | `listen_events.jsonl`, `playback.json` | Derive the **heard set** (≥30% played ∪ captured) from these. |
| Hybrid retrieval (filtered search) | RFC-090 | Recall = hybrid search **filtered to the heard set**. |
| Relational traversals (person/topic/who-said) | RFC-094 / `app_relational.py` | Connections = these traversals **scoped to the heard set**. |
| Canonical identity (one node across episodes) | RFC-072 | Unifies a guest/topic across the user's episodes. |
| Entity cards + topic clusters | P2 epic-3: `EntityCardBody.vue`, RFC-102 | Reuse for the connections UI (scoped per-user). |
| **Enrichment envelopes** | RFC-088: corpus-scope `enrichments/<id>.json`; episode-scope `metadata/enrichments/<stem>.<id>.json`; envelope = `{enricher_id, schema_version, …signals}` | Recall/resurfacing/connections **consume** these read-only (ADR-104 boundary — never recompute). |
| **User-facing envelope readers** | RFC-088 chunk 6: `routes/corpus_enrichments.py` → `GET /api/corpus/enrichments`, `/api/corpus/enrichments/{id}`, `/api/corpus/episode/enrichments/{id}` | P3's consumer surface **wraps these loaders** + adds heard-set scoping (don't re-read files). |

Available enrichment signals (consume as-is): `topic_cooccurrence`, `topic_similarity`,
`temporal_velocity`, `grounding_rate`, `nli_contradiction` (+ RFC-097 chunk-9 core `RELATED_TO`
edges, already traversable via the relational layer).

---

## 2. The gap P3 fills

The shipped `corpus_enrichments` routes are **global / corpus-scope** (`/api/corpus/*`), not under the
consumer namespace and not scoped to a user. P3 adds the **per-user projection**: `/api/app/*`
endpoints that (a) call the *same* envelope loaders, (b) scope to the user's heard set, and (c) shape
the signals for consumer surfaces (recall, connections, resurfacing, "Your Week"). No new artefacts,
no recompute — a read-time filtered view (RFC-101 §2 decision: projection, not a rebuilt graph).

---

## 3. Consumer enrichment read surface (design — sub-epic *a*)

Thin `/api/app/*` projections over the shipped loaders, all auth-gated, all heard-set-scoped:

| Method | Path | Shape | Built on |
| --- | --- | --- | --- |
| `GET` | `/api/app/episodes/{slug}/enrichment` | Per-episode signals for an episode in the user's set: co-occurring topics, similar threads, contradictions touching this episode, grounding_rate. `404`/empty when the episode isn't in the heard set. | `get_episode_enrichment` loader |
| `GET` | `/api/app/corpus/enrichment` | Corpus-scope signals **filtered to the heard set**: `temporal_velocity` for topics the user has heard ("trending in your corpus"), `topic_similarity` peers among heard topics. | `get_corpus_enrichment` + `list_corpus_enrichments` |

Design rules: read-only (ADR-104); **no request-time LLM** (D6); **zero-coverage honesty** (if the
user has heard nothing on a signal, say so — never widen to the global corpus); reuse the envelope
Pydantic models from RFC-088 where they exist (extend `schemas.py` only for the per-user wrapper).

---

## 4. Child decomposition (#1120–#1126)

Maps the REMEMBER-half-scope sub-epics (a–f) + a closing test/docs issue onto the seven children.
**Confirm each scope against the live GH issue body on resume** (titles set in the planning session).

| Issue | Sub-epic | Scope | Unifies with | Tests |
| --- | --- | --- | --- | --- |
| **#1120** | a — consumer enrichment read surface | `/api/app/episodes/{slug}/enrichment` + `/api/app/corpus/enrichment` over the shipped loaders, heard-set-scoped; per-user wrapper schemas | `corpus_enrichments.py` loaders | unit (scoping), integration (route over fixture corpus + envelopes), zero-coverage |
| **#1121** | b — heard set + corpus projection | `heard_set(user)` deriver (≥30% played ∪ any capture) from `listen_events`/`playback`/highlights; read-time projection helper unifying highlights+insights+notes+heard-episode artifacts via canonical identity | `app_user_state`, RFC-072 | unit (set derivation, A≠B isolation) |
| **#1122** | c — grounded recall | `POST /api/app/corpus/recall {q}` → grouped grounded set (hybrid RFC-090 filtered to heard set + relational + user highlights), enriched with co-occurrence/similarity/contradiction; **no LLM**; zero-coverage message. Consumer recall UI. | RFC-090/094, #1120/#1121 | unit (group/rank/coverage), integration (multi-user, no-LLM assert), e2e |
| **#1123** | d — cross-episode connections | `GET /api/app/corpus/person/{id}` + `/corpus/topic/{id}` (RFC-094 traversals scoped to heard set) → "you also heard `<guest>` discuss this in …"; connections UI reusing entity-card machinery | RFC-094, RFC-102 entity cards | unit (scoping), integration, e2e |
| **#1124** | e — spaced resurfacing + reflection | `GET /api/app/resurfacing` (read-time due-item ladder 2d/1w/1mo/3mo on highlight `created_at`/`last_surfaced`), reflection prompt + one-tap re-listen, pacing controls (per-user settings); in-app inbox/digest UI; `temporal_velocity` informs what to resurface | highlights store, `temporal_velocity` | unit (due-ladder), integration, e2e |
| **#1125** | f — interest profile evolution | Aggregate topic/person frequencies from captures+history into the interest model; cross-reference enrichment topic signals; extend the shipped interest-token model (flag-gated personalised ordering, off by default) | P2/3.5 interests, `rank_discover` | unit (aggregation), integration |
| **#1126** | tests + docs | App-activity e2e over an extended committed corpus (heard/captured + enrichment fixture); reconcile PRD-041 + RFC-101 to as-shipped; `Closes #1120…#1126` + `Part of #1113` in the PR body | committed `app-validation-corpus` | e2e, docs (mkdocs strict) |

---

## 5. Sequencing

```text
[RFC-088 #1101 lands on main]  →  git fetch && rebase origin/main  (the agreed break point)
        │
   #1121 heard set + projection ─┐
   #1120 enrichment read surface ─┼─► #1122 recall ─► #1123 connections
        │                         │                    │
        └─────────────────────────┴──► #1124 resurfacing ──► #1125 interest profile ──► #1126 e2e+docs
```

- **#1120 + #1121 are the foundation** — everything else reads through them. They can proceed in
  parallel the moment the rebase is clean.
- **#1122–#1125** are the user-facing slices; each is independently shippable (backend → UI →
  tests), same loop discipline as P2 (commit per issue, mid + end sweeps).
- **#1126** closes the phase: the app-activity e2e the operator deferred in P2 (simulate per-user
  activity over a committed corpus) finally lands here, where there's per-user corpus to exercise.

## 6. Cross-cutting guardrails (carry over from P2 + RFC-101)

- **D6 — no request-time LLM.** Recall/connections/resurfacing are extractive/verbatim retrieval.
  CI must assert no LLM/network call (per the no-LLM-in-CI rule).
- **ADR-104 boundary.** Enrichment envelopes are read-only; P3 never recomputes a signal.
- **Scope = the user's experience.** Recall/connections cite only heard/captured episodes — never the
  global corpus. Zero-coverage is an honest "nothing in your corpus yet."
- **Per-user files, projection at read time.** No per-user graph in v2.7 (materialise later only if
  read latency proves it — RFC-101 Open Q1).
- **Same test pyramid** as P2: unit (logic) → integration (`/api/app/*` over fixtures, multi-user
  isolation) → e2e (committed corpus). Extend the committed `app-validation-corpus`, never invent a
  bespoke one.
- **PR hygiene.** The Remember PR body lists `Closes #1114…#1126` (delivered) + `Part of #1112`/
  `#1113` — squash erases commit mentions, the body is the only carrier.
