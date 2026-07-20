# USERPREFS-1 — Cross-device user preferences

Shipped on `feat/graph-v3` (bundled per operator direction) — the graph-v3 arc
was the trigger, but the mechanism generalises across every viewer store that
currently persists to `localStorage`.

## The problem

Every UI opinion-state today lives in `localStorage` — graph lens flags, theme,
panel collapse, corpus path, hint dismissals. That means a user who signs in
from a second device sees factory defaults. Every setting is a per-browser
secret.

## The shape

- **Server**: one free-form JSON object per user, file-backed under the same
  per-user overlay the rest of `app_user_state` already uses
  (`<data_dir>/users/<user_id>/preferences.json`). Same FileLock semantics.
  No schema — the server round-trips whatever the client writes. Adding a
  new preference key never requires a server release.
- **Client**: a Pinia store (`useUserPreferencesStore`) that hydrates once
  at mount, exposes `get<T>(key)` / `set(key, value)` / `setMany({...})`.
  `set(key, null)` deletes the key server-side. Every mutation writes
  through immediately (local ref updates synchronously, network PATCH fires
  fire-and-forget) so multi-tab races stay bounded to top-level keys.
- **Feature-store adopters**: existing stores keep their `localStorage`
  mirror as the offline / unauthenticated fallback, PLUS write-through to
  `userPreferences.set(featureKey, {...})` on every mutation, PLUS apply
  server payload back onto local refs once `userPreferences.hydrated`
  flips true. An `applyingRemote` guard prevents echo loops.

## API contract

```
GET    /api/app/preferences          → { preferences: {…} }         (401 if unauth)
PUT    /api/app/preferences          → replaces the whole payload
PATCH  /api/app/preferences          → shallow-merges top-level keys
                                       (null value deletes the key)
```

Auth via existing session cookie / `get_current_user`. No new schema, no
migration script, no version field on the payload.

## Failure modes are silent by design

- **Unauthenticated / offline**: server returns 401 or the fetch throws;
  store flips `available` to false permanently for the session and no
  further round-trips fire. Feature stores keep working from their
  `localStorage` fallback.
- **Corrupted `preferences.json` on disk**: server returns `{}`; client
  falls back to local defaults on every unset key.
- **Multi-tab races**: PATCH is a shallow-merge on top-level keys, so
  two tabs writing `graphLenses` and `theme` don't clobber each other.
  Two tabs writing the SAME top-level key follow last-writer-wins —
  the client owns any deeper conflict-resolution (none needed today).

## Adopted stores

Pilot: `useGraphLensesStore` — all 8 lens flags now sync cross-device.
The store keeps its `ps_graph_lenses` localStorage mirror; server value
overrides on hydrate.

Not yet adopted (localStorage-only for now — one-liners each when
someone wants them):

- `useThemeStore` (`gi-kg-viewer-theme`)
- Panel collapse (`ps_left_panel_open`, `ps_right_panel_open`, `ps_graph_bottom_bar_collapsed`, `ps_graph_theme_legend_collapsed`)
- `ps_corpus_path` — user's chosen corpus footer input
- `ps_graph_hints_seen` — one-time gesture-overlay dismissal

Migration is per-store and additive (add ~10 lines to the store's watcher
+ 1 line to hydrate on mount). No breaking change for any consumer.

## Tests

- `tests/unit/podcast_scraper/server/test_app_user_preferences.py` — 9 unit
  tests for the file-backed store (roundtrip, patch merge, null-deletes-key,
  per-user isolation, malformed-file recovery).
- `tests/integration/server/test_app_user_preferences_routes.py` — 7 integration
  tests over the FastAPI TestClient (auth-required, GET-default-empty, PUT
  replace, PATCH merge, PATCH null delete, cross-request persistence,
  arbitrary-JSON round-trip).
- `web/gi-kg-viewer/src/stores/userPreferences.test.ts` — 10 client-store
  tests (starts empty, hydrate happy-path, hydrate marks unavailable on
  null, hydrate idempotent, set writes through, null delete, permanent
  unavailable on server failure, setMany batches with null normalisation,
  get typed values).
- `web/gi-kg-viewer/src/stores/graphLenses.test.ts` — unchanged public
  contract; API wrappers mocked so pilot integration doesn't fire real
  network calls.

Total delta: **+26 tests**, 3 test files.

## Live-verified

Manual roundtrip against `make serve` + prod-v2 corpus + the mock OAuth
provider (default dev seed):

1. Load viewer, hydrate empty (fresh user, no `preferences.json` yet).
2. Toggle `velocityHalo` false in the Lenses popover → payload PATCHed to
   `/api/app/preferences` with `{graphLenses: {…, velocityHalo: false, …}}`.
3. Direct `curl /api/app/preferences` returns the same payload.
4. Full page reload → hydrate returns the persisted payload → server value
   (`velocityHalo: false`) overrides the localStorage `true` mirror.

The cross-device story then reduces to "second device also authenticates
under the same OAuth identity" — the store just fetches its state.

## #1213 / #1215 adoption wave (2026-07-19)

Broadened USERPREFS-1 beyond the initial graphLenses pilot to the whole
consumer + operator surface. Split into two work items on one branch:

- **#1213 — learning-player consumer surface.**
  - New `web/learning-player/src/stores/userPreferences.ts` (stripped-
    down mirror of gi-kg-viewer's store, `fetch`-based, silent-degrade,
    no BroadcastChannel).
  - App-root hydrate call in `web/learning-player/src/main.ts`.
  - Adopted keys: `lp.interests.dismissed` (HomeView), nested
    `lp.audioSyncOffsets = { slug: offset }` (PlayerView adjustSync /
    resetSync).
  - Commit: `24fec52f`.

- **#1215 — gi-kg-viewer operator surface.**
  - `corpusLens` store write-throughs the `activePreset` string
    (`'all' | '7' | '30' | '90'`) under key `corpusLensPreset`.
    Persisting the preset (not the calculated YYYY-MM-DD) makes
    "Last 7 days" stay today-relative across devices. Numeric legacy
    writes tolerated via `parsePreset`.
  - `useUserPreferencesStore.resetToDefaults()` — PUTs `{}` to
    `/api/app/preferences`, clears the in-memory map, broadcasts null
    for each cleared key so other tabs' consumers reset too.
    Silent-degrade on failure (marks store unavailable, local clear
    stands). See the docstring for the "reset just section X" pattern
    (iterate keys → `setMany(nulls)`).
  - Tests: `corpusLens.test.ts` +3 (write-through, echo-suppressed
    hydrate, numeric-legacy normalisation) → 16/16;
    `userPreferences.test.ts` +3 (reset happy-path w/ broadcast,
    silent-degrade on null, skip-network-when-unavailable) → 17/17.
  - Not yet adopted here (deferred, tracked in graph-v3 follow-ups):
    `graphFilters.allowedTypes` (per-corpus state, complex shape),
    graph bottom-bar time-scale, and the reset-to-defaults UI control
    itself (backend is landed; the button is a separate UX pass).

## Not shipped (documented follow-ups)

- **Per-user-preferences migration of the other localStorage keys.**
  Each is a small write-through addition to its own store; batched into
  one PR when someone wants cross-device parity for those too.
- **Import/export UX** — the server accepts PUT `/api/app/preferences`
  with any payload, so an "export from device A / import onto device B"
  UI would be pure client work. Deferred.
- **Conflict resolution for concurrent multi-tab writes on the same key**
  — last-writer-wins per top-level key is fine for today's flat payload;
  if a future adopter has a nested payload with independent writes to
  different sub-keys, wrap in an explicit `setMany` + read-modify-write
  loop or bump to the merge-on-server pattern.
