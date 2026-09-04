# `src/stores/` — Pinia stores

Per-user state and playback. Each store owns one slice, wraps its `/api/app/*` endpoints, and is the
only place a component should reach for that data.

## The stores

| Store | Owns | Endpoint | Reach for it when |
| --- | --- | --- | --- |
| [`player.ts`](player.ts) | The `<audio>` element itself, transport, rate, MediaSession | — | Anything about what is playing, or how |
| [`auth.ts`](auth.ts) | The signed-in user | `GET /me` | You need to know who, or whether |
| [`queue.ts`](queue.ts) | Ordered episode slugs, auto-advance source | `GET/PUT /queue` | Up-next, reorder, play-next |
| [`capture.ts`](capture.ts) | Highlights + notes — moments, spans, insights | `/highlights`, `/notes` | The learning loop's write side |
| [`library.ts`](library.ts) | Shows the user follows | `/library` | Follow / unfollow, "Your shows" |
| [`favorites.ts`](favorites.ts) | Saved episodes + insights | `/favorites` | The heart control |
| [`interests.ts`](interests.ts) | Topic interests that shape ranking | `/interests` | Personalization, the interests picker |
| [`savedQueries.ts`](savedQueries.ts) | Ring buffer of recent searches | `/saved-queries` | Search history surfaces |
| [`userPreferences.ts`](userPreferences.ts) | Cross-device prefs (USERPREFS-1) | `/preferences` | Anything that must survive a device switch |
| [`downloads.ts`](downloads.ts) | Episodes downloaded to **this device** | — (device-local) | Offline listening, storage used |

## Four conventions that span every store

**1. `ensureLoaded()` is the entry point, not `load()`.** It no-ops once loaded and shares a single
in-flight promise, so N components mounting at once produce one request. Call `load()` only to force
a refetch.

**2. Writes are optimistic and swallow their failures.** The store flips local state, fires the
request, and reverts on rejection — deliberately never throwing, so `void store.toggle()` in a
template handler cannot raise an unhandled rejection. **The consequence matters:** a caller cannot
tell success from failure unless the action *returns* it. `capture.*` returns `boolean` for exactly
this reason ([S8](../../../../docs/wip/2026-08-13-player-overhaul-observations.md)) — callers were
announcing "Saved" to screen readers after failed POSTs. If you add a write whose outcome the user
can perceive, return the outcome.

**3. Every per-user write must be gated.** Signed out, an ungated call 401s, the store swallows it,
and the control flips then silently reverts — which reads to the user as their own action failing.
Route writes through [`useSignInGate`](../composables/README.md). Enforced per call site by
[`__checks__/auth-gate.test.ts`](../__checks__/auth-gate.test.ts), which also derives its list of
"per-user writes" from the store sources — so **adding an async action here fails that test until
you decide** whether it needs a gate.

**4. Stores must not import the queue-from-player direction, or the API for data they don't own.**
`player.ts` deliberately knows nothing about the queue or the API; auto-advance is resolved by the
shell (`App.vue`) and handed in via `setAdvanceResolver`. Keeping playback free of data-fetching is
why the element can outlive every view.

## `player.ts` is different — read this before touching it

It owns a **body-attached `<audio>` element created once and never unmounted** (#1587). Before that,
`PlayerView` rendered the element, so navigating away stopped playback.

Consequences that are easy to break:

- `load()` **no-ops for the already-loaded slug**, so returning to the player does not restart it.
- Every path into `load()` carries title/artwork, because auto-advance runs with **no view mounted** —
  without it the mini-player shows "Loading…" and the lock screen shows the previous episode.
- The advance resolver is **async and called at `ended`**, not cached at load, so mid-listen queue
  edits take effect.
- A view must not call `resetForLoad()` for the episode already playing — it wipes transport state
  the store will not restore.

## Related

- [`../composables/README.md`](../composables/README.md) — the gate and section-state primitives
- [`../services/api.ts`](../services/api.ts) — the HTTP layer these wrap
- [`../../e2e/E2E_SURFACE_MAP.md`](../../e2e/E2E_SURFACE_MAP.md) — behaviours asserted in the browser
- [RFC-099](../../../../docs/rfc/RFC-099-learning-platform-consumer-client.md) §4 queue · [PRD-040](../../../../docs/prd/PRD-040-capture.md) capture

## Known gaps

- **`favorites.ts` has no test** — direct or indirect. It is the only store with none.
- **`library.ts` has no direct test**; it is exercised through
  [`../components/ShowTile.test.ts`](../components/ShowTile.test.ts).
