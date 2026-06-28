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

## 3. Consumer enrichment read surface (design — issue #1121)

Thin `/api/app/*` projections over the shipped loaders, all auth-gated, all heard-set-scoped:

| Method | Path | Shape | Built on |
| --- | --- | --- | --- |
| `GET` | `/api/app/episodes/{slug}/enrichment` | Per-episode signals for an episode in the user's set: co-occurring topics, similar threads, contradictions touching this episode, grounding_rate. `404`/empty when the episode isn't in the heard set. | `get_episode_enrichment` loader |
| `GET` | `/api/app/corpus/enrichment` | Corpus-scope signals **filtered to the heard set**: `temporal_velocity` for topics the user has heard ("trending in your corpus"), `topic_similarity` peers among heard topics. | `get_corpus_enrichment` + `list_corpus_enrichments` |

Design rules: read-only (ADR-104); **no request-time LLM** (D6); **zero-coverage honesty** (if the
user has heard nothing on a signal, say so — never widen to the global corpus); reuse the envelope
Pydantic models from RFC-088 where they exist (extend `schemas.py` only for the per-user wrapper).

---

## 4. Child decomposition (#1120–#1126 — verified against live GH issues)

The real issues split **backend (#1120–#1123, `fastapi`) → UI (#1124–#1125, `ui/ux`) → e2e+docs
(#1126, `documentation`)**, each carrying its own `Depends on` in the issue body. Every child is
blocked on the RFC-088 rebase. Verb in **bold** is the issue's own framing (Extend / New / Reuse) —
the unify-don't-rebuild signal.

| Issue | Label | Scope (as written) | Extends / unifies | Depends on | Tests |
| --- | --- | --- | --- | --- | --- |
| **#1120** | fastapi | **Extend** hybrid search: derive the `heard∪captured` set; add `scope=mine` to `/api/app/search` + relational; grounded recall over the scoped set (no request-time LLM, D6); honest zero-coverage | `app_search.py` (its docstring already anticipates "scoped to the user's library once auth + library land") | rebase | unit (scoping) + integration |
| **#1121** | fastapi | **New** consumer enrichment read surface: `GET /api/app/episodes/{slug}/enrichment` (+ corpus-scope) exposing co-occurrence / similarity / temporal-velocity / NLI-contradiction + `RELATED_TO` edges, read-only (ADR-104) | wraps the shipped `corpus_enrichments.py` envelope loaders (§3) | rebase | integration over an enrichment fixture |
| **#1122** | fastapi | **Extend** person/topic endpoints with a "your corpus" lens (a guest/topic across heard episodes) + enrichment connections ("you also heard X discuss this in …") | `app_relational.py`, #1121 signals | #1120, #1121 | integration |
| **#1123** | fastapi | **Extend** interests with implicit profile signals from captures + history (beside explicit follows); **new** spaced-resurfacing selection over highlights/heard (schedule + pacing) | interests / `app_user_state`; `temporal_velocity` informs selection | #1120 | unit (selection/schedule) + integration |
| **#1124** | ui/ux | **Reuse** the search UI / result cards for a scoped **Recall** mode ("what have I learned about X") → grouped grounded results (insights/quotes/highlights) with jump-to-moment; honest zero-coverage message | `SearchView` / result cards | #1120 | vitest |
| **#1125** | ui/ux | **Extend** `EntityCardBody` with a "your corpus" toggle; **new** in-app digest/inbox surfacing past highlights + a reflection prompt + one-tap jump + pacing controls (frequency/pause/dismiss) | `EntityCardBody.vue`; resurfacing API (#1123) | #1122, #1123 | vitest |
| **#1126** | documentation | e2e (recall/connections/resurfacing) on the `app-validation-corpus` extended with enrichment + a heard/captured fixture; promote PRD-041 + RFC-101; add the endpoints to PLATFORM_API / HTTP_API; `make docs` + `lint-markdown` green | committed `app-validation-corpus` | #1124, #1125 | e2e + docs |

---

## 5. Sequencing (the issues' own dependency graph)

```text
[RFC-088 #1101 lands on main]  →  git fetch && rebase origin/main  (the agreed break point)
        │
  #1120 scope + scoped recall (be) ─┬─────────────► #1122 connections (be) ─┐
  #1121 enrichment read surface (be) ┘                                      ├─► #1125 connections+
        │                                                                   │   resurfacing UI ─► #1126
  #1120 ─► #1123 interest + resurfacing-selection (be) ────────────────────┘
  #1120 ─► #1124 Recall surface UI ───────────────────────────────────────────────────────► #1126
```

- **#1120 ∥ #1121** are the backend foundation — start both the moment the rebase is clean.
- Then **#1122, #1123, #1124** unblock (1122 also needs 1121); **#1125** needs both 1122 + 1123.
- **#1126** closes the phase: the per-user-activity e2e deferred in P2 finally lands here (there's now
  a personal corpus to exercise), plus the PRD-041/RFC-101 promotion + API-doc updates.
- Same loop discipline as P2: one commit per issue, mid- and end-of-phase sweeps.

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
