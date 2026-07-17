# USERPREFS-1 — conflict resolution design

Companion note to `USERPREFS-1.md`. Documents what "conflict" actually means
for the preferences payload today, what the current strategy handles, what
it misses, and the ladder of options if we ever need to climb it.

## What "conflict" means here

A conflict is any state where two independent writers commit incompatible
changes to the same preference without knowing about each other. On a
free-form key/value store, that reduces to two shapes:

1. **Same top-level key, two values.** Tab A sets `theme=dark` while tab B
   sets `theme=light`. This is the only realistic conflict on today's
   flat payload — every consumer store owns a distinct top-level key
   (`graphLenses`, `theme`, `leftPanelOpen`, …), so cross-store conflicts
   don't exist.
2. **Same top-level key, nested divergent edits.** Tab A sets
   `graphLenses.velocityHalo=true` while tab B sets
   `graphLenses.bridgeRing=false`. Both tabs PATCH the whole
   `graphLenses` object because that's the granularity the store writes
   at → one whole payload clobbers the other. This is a real gap for the
   `graphLenses` shape today.

## Current strategy (what we ship)

Three layers, cheapest to most expensive:

| Layer | Where | Wins when |
|---|---|---|
| 1. **BroadcastChannel** cross-tab sync | `stores/userPreferences.ts` (`ps_user_preferences_sync`) | Same browser, another tab, alive right now. Zero server round-trip, zero latency. |
| 2. **PATCH shallow-merge** on top-level keys | server `app_user_preferences.patch_preferences` | Different browsers / devices writing to *different* top-level keys concurrently. |
| 3. **Last-writer-wins per top-level key** (implicit) | server FileLock serialises + client accepts server truth on next hydrate | Different browsers writing to the *same* top-level key concurrently. |

## What today's design handles

- **Same-browser cross-tab** for any key: BroadcastChannel propagates each
  `set()` synchronously to sibling tabs; both stay in sync without waiting
  for a server round-trip. Verified end-to-end + tested.
- **Cross-device convergence** on eventual reload: even without pushed
  updates, the next page-load hydrate reads server truth and clobbers the
  local mirror. Users see the write eventually.
- **Server-side write serialisation**: `FileLock` on
  `preferences.json` guarantees two simultaneous PATCH requests don't
  interleave the read-modify-write, so shape (2) conflicts still leave a
  valid JSON on disk — one whole nested object replaces the other cleanly,
  no partial merge.

## What today's design misses

1. **Real-time cross-device push.** Tab A on laptop, tab B on phone. A
   change on the laptop doesn't reach the phone until the phone
   reloads. This is the classic "cross-device sync feels magical" gap.
2. **Nested-key concurrent edits under one top-level object** (shape 2
   above). If two tabs edit different `graphLenses` sub-flags at the
   same time, one wipes the other's edit until the loser's tab
   re-reads and re-writes.
3. **No versioning / lost-update detection.** Two devices can PATCH
   the same key concurrently and neither notices the collision — the
   later PATCH silently wins.
4. **No offline write queue.** If the client is offline when the user
   toggles, the local mirror updates + BroadcastChannel fires + the
   PATCH silently fails + we mark `available=false` for the session.
   The server never learns about the toggle unless the user reloads
   (fresh page-load re-hydrates and can pick up the localStorage
   fallback via feature-store re-emission). Recovery is unreliable.

## Design ladder — when to reach for each

Ordered by cost vs the problem it solves. Do NOT climb the ladder until
the tier below is genuinely inadequate for a real user complaint.

### Rung 1 — BroadcastChannel (SHIPPED)

Cheapest cross-tab sync. Handles almost everything users notice in
same-browser scenarios. Cost: ~30 lines in the store; already tested.

### Rung 2 — Server-Sent Events push per user

Server holds one SSE stream per authenticated session; broadcasts every
PATCH to all connected sessions for that user. Adds cross-device
real-time sync while staying HTTP-only (no WebSocket infra).

- **Server work**: FastAPI endpoint yielding `text/event-stream` from
  a per-user asyncio queue; PATCH publishes to the queue after the
  file write.
- **Client work**: `EventSource` opened after hydrate; on message,
  apply the same shape as BroadcastChannel's incoming handler.
- **Cost**: ~1 day. Route + fanout + reconnect logic + tests.
- **When**: user complaint "I toggled X on my phone but my laptop
  didn't pick it up". None today.

### Rung 3 — Per-key ETag + optimistic concurrency

Each preference key gets a monotonic version number. PATCH must send
the last-seen version; server rejects with 409 if the stored version
moved on. Client on 409 refetches, replays user intent, re-PATCHes.

- **Server work**: track per-key version in `preferences.json` (e.g.
  `{value, version}` shape) + `If-Match` header enforcement.
- **Client work**: cache last-seen version per key; on 409, refetch +
  merge + retry once.
- **Cost**: ~2 days including tests + migration of stored files.
- **When**: user complaint "the two devices are fighting each other,
  I keep having to reload". None today; unlikely on flat scalar keys.

### Rung 4 — CRDT-style merge for nested payloads

For genuinely concurrent nested writes (shape 2 above), a CRDT (LWW-map
or OR-map) lets both writes land losslessly without coordination.
Complexity: significant. Real payload savings only when the user is
actively toggling multiple sub-flags on multiple devices at once —
essentially theoretical for this app.

- **Cost**: ~1 week. Library choice (Yjs / Automerge) + serialisation
  contract + migration + full test suite.
- **When**: probably never, given the app's actual usage patterns.
  Documented for completeness. If it becomes necessary, the right
  intermediate move is Rung 3.5 — split the nested payload into more
  top-level keys (one per sub-flag) — before adopting a CRDT.

### Rung 3.5 — Split nested payloads into per-key top-level

An escape hatch that avoids Rungs 3/4 entirely: instead of storing
`graphLenses: {velocityHalo, bridgeRing, …}` as one payload, store
each flag as its own top-level key (`graphLensVelocityHalo`,
`graphLensBridgeRing`, …). PATCH shallow-merge then handles them
correctly per-key.

- **Cost**: ~1 hour per feature store — no server change, no protocol
  change. Small churn.
- **Downside**: ~9 keys in the preferences payload where there used
  to be 1; slightly more chatty on network but negligible.
- **When**: as soon as shape (2) becomes an actual user pain point.

## Offline write queue

Orthogonal to sync direction, but worth capturing here since it
compounds every rung's failure mode.

Today: a failed PATCH marks `available=false` for the session; that's a
permanent silence. Better:

1. Queue failed writes in `localStorage` under a `pending_prefs_patches`
   key with `{timestamp, key, value}`.
2. Retry on next `available=true` transition (network came back / user
   re-authenticated).
3. Flush + clear on success.

Complexity: ~half a day. Worth building when we see a real offline user
population, not before.

## Concrete recommendation

**Keep the current design. Do NOT climb the ladder yet.**

The only concrete shape-2 gap today is `graphLenses` — 8 flags in one
object. Realistic conflict scenario: user opens two tabs, changes 3
different lens flags in one tab and 2 in the other within the same
second. Whichever tab PATCH'd last wins for the whole object; other
tab's changes vanish on next reload. Painful, but rare enough that
Rung 3.5 (split into 8 top-level keys) is the correct fix if we ever
observe it.

Real-time cross-device push (Rung 2) is worth building the day someone
credibly complains "I keep having to reload my second device". Nothing
today suggests that day.

## Test coverage of the shipped strategy

`web/gi-kg-viewer/src/stores/userPreferences.test.ts` covers:

- BroadcastChannel: post on `set()` (own tab writes reach siblings).
- Incoming broadcast updates local state without a network call.
- Incoming null deletes the key locally, other keys preserved.
- Own broadcast is ignored (senderId echo suppression).

Server-side: FileLock serialisation is exercised by the existing
integration test `test_preferences_survives_across_requests_per_user`
(two sequential clients same user), and by the unit test
`test_users_are_isolated` (two users, independent state). No test yet
for a *simultaneous* two-request race — the FileLock 5-second timeout
would make that flaky in CI. If we build Rung 3, the version-mismatch
test becomes the canonical concurrent-write regression guard.
