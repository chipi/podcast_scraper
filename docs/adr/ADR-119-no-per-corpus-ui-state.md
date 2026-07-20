# ADR-119: No per-corpus UI state — user-scoped features live in USERPREFS-1

- **Status**: Accepted
- **Date**: 2026-07-20
- **Authors**: Marko
- **Related RFCs**:
  - [RFC-107](../rfc/RFC-107-search-v3-query-workspace.md) — Search v3 (first consumer; retracts the per-corpus JSON store)
- **Related Documents**:
  - `docs/wip/USERPREFS-1.md` — server-backed per-user preferences (shipped)
  - `web/gi-kg-viewer/src/stores/userPreferences.ts` — client store

## Context & Problem

The viewer today reads/writes a corpus at a filesystem path (`ps_corpus_path` in `localStorage`). That path is a **transient** substrate — we are migrating to a database at some point, at which point "corpus" becomes an identifier over the DB, not a directory. Any UI feature whose persistence is scoped to the corpus root (`<corpus_root>/.viewer/*.json`) will be:

1. Invalidated at the migration boundary — the files don't come along.
2. Non-portable across devices — a user on device A cannot see the same state on device B, even against the same logical corpus.
3. Non-personal — two operators sharing a corpus share all state, whether they want to or not.

Concretely, this ADR was triggered by Search v3 §S7 (saved queries), which I initially spec'd as `<corpus_root>/.viewer/saved_queries.json`. That is wrong on all three counts above.

Meanwhile, USERPREFS-1 (shipped 2026-06 / 2026-07 via #1213 / #1215) provides exactly the right substrate: a server-backed, per-user, cross-device, silent-degrade preferences store with a free-form JSON payload, PATCH-shallow-merge semantics, per-tab BroadcastChannel sync, and a client Pinia store (`useUserPreferencesStore`) with `.get<T>(key)`, `.set(key, value)`, `.setMany({})`, and `.resetToDefaults()`.

## Decision

**No UI feature may persist state scoped to the corpus.** For each state-persistence need, choose exactly one of:

- **Per-user (default for anything personal)** — via USERPREFS-1 (`/api/app/preferences`). Requires the user to be signed in; degrades silently offline / unauth (the feature-store keeps a `localStorage` mirror as fallback).
- **Per-corpus / corpus-analytics (only for things about the corpus itself, not about a specific user)** — via existing corpus-side artifacts (e.g. `search/query_log` writes an append-only stream used by Dashboard's QueryActivityChart). This stays because it is corpus telemetry, not user state; when we migrate to a DB it moves with the corpus record.
- **Device-local (only for things that are genuinely device-specific)** — via `localStorage`. Today's `ps_corpus_path` is the canonical example — the *filesystem path* is a device-scoped detail that should not roam.

The default when in doubt is **per-user via USERPREFS-1**.

## Consequences

**Positive**

- Every user-facing feature survives the corpus-→-DB migration by construction.
- Cross-device portability by default; no per-feature retrofit.
- Silent-degrade offline / unauth is already solved once by USERPREFS-1.
- A single store, a single API, a single failure mode to reason about.

**Negative**

- Features that would want to be "per-corpus" (e.g. "recent queries for THIS corpus") can either:
  - Store per-user with a `corpusId` field on each entry (works today, works after DB migration).
  - Store as corpus telemetry (Dashboard-only, not personal recall).

- USERPREFS-1 has a free-form JSON payload with no schema at the server. Adding a new feature key requires disciplined namespacing (e.g. `search.savedQueries`, `search.recentQueries`) — no server release, but no server validation either. Feature stores must own their own shape validation on hydrate.

**Neutral**

- `ps_corpus_path` remains device-local (documented in USERPREFS-1.md as intentional).
- `search/query_log` remains per-corpus (it is corpus telemetry, not user state).

## Alternatives Considered

1. **Per-corpus JSON at `<corpus_root>/.viewer/*.json`.** Rejected — invalidated by the DB migration; non-portable; non-personal. (This ADR is the retraction.)
2. **A new database table per feature.** Rejected — USERPREFS-1 already exists, is battle-tested for graph-lenses / theme / lp.interests / lp.audioSyncOffsets / corpusLensPreset, and adding a feature to it is a client-side namespace addition with no server work. A new table per feature is server-work + migration + admin surface for zero user-visible gain.
3. **Cookie-scoped preferences.** Rejected — cookie payload budget too small; cookies can't hold a saved-queries list; no per-tab sync story.

## Implementation notes

- **Namespace convention:** feature keys are dotted paths, `search.savedQueries`, `search.recentQueries`, `graphLenses` (already shipped), etc. Feature stores own their sub-payload shape.
- **Hydration:** feature stores must accept the case where USERPREFS is `available: false` and read from a `localStorage` mirror. New features may skip the mirror if they're allowed to be blank when unauth.
- **Server:** no new endpoints for USERPREFS-1 consumers — `/api/app/preferences` handles all of it. Additive query params on unrelated endpoints (e.g. Dashboard analytics) still allowed.

## References

- [RFC-107 §8 — Saved queries + query history](../rfc/RFC-107-search-v3-query-workspace.md) (USERPREFS-1 backed)
- `docs/wip/USERPREFS-1.md` — the shipped USERPREFS-1 mechanism.
- Retracted design: `<corpus_root>/.viewer/saved_queries.json` (never implemented).
