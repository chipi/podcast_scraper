# RFC-100: Audio Bridge Subsystem

- **Status**: Draft
- **Authors**: Marko
- **Stakeholders**: Server API, Core Pipeline, Consumer App, Legal/Licensing
- **Related PRDs**:
  - `docs/prd/PRD-035-learning-platform.md` (Principle 4 — bridge, never rehost)
  - `docs/prd/PRD-039-player.md`
- **Related RFCs**:
  - `docs/rfc/RFC-096-audio-pipeline-separation-and-viewer-media.md` (today's local-byte serving)
  - `docs/rfc/RFC-098-learning-platform-foundation.md` (consumer API that references the source)
  - `docs/rfc/RFC-099-learning-platform-consumer-client.md` (player that plays it)
- **Related Documents**:
  - `docs/wip/player/SERVER-SIDE-GAP-ANALYSIS.md` (gap G5)

## Abstract

This RFC defines a **new, standalone subsystem** that lets the consumer player stream each episode from its
**original host** while we store only derived artifacts (PRD-035 Principle 4 — *bridge, never rehost*). It
resolves an episode to a fresh, playable origin enclosure URL, validates it, and hands it to the client.
Today the server only serves **local** audio bytes (`/api/corpus/media`, RFC-096) — the opposite of a
bridge — so this is complete new work.

**Architecture Alignment:** Keeps internal pipeline audio (`media/`) strictly operator-internal; the
consumer never reaches it. Mounts under the consumer API namespace (RFC-098) as its own module.

## Problem Statement

Principle 4 forbids rehosting third-party audio. But (G5) the platform has **no concept of an origin
enclosure URL at request time** — the pipeline downloads audio to `media/` for transcription, and the
server only streams those local bytes. Origin URLs also rot: feeds use tracking-prefix redirects, signed
URLs that expire, and hosts that block hotlinking or lack permissive CORS / HTTPS. A naive "store the URL,
return it" is insufficient; we need a subsystem that resolves, refreshes, and validates playable URLs, with
a no-store fallback only where a host forces it.

**Use Cases:**

1. **Direct play**: the player gets a fresh origin URL and streams it directly (no bytes through us).
2. **Stale/expiring URL**: a previously-stored URL 404s/expires → the subsystem re-resolves from the feed.
3. **Hostile origin**: mixed-content/CORS/hotlink blocks direct play → optional no-store pass-through.

## Goals

1. **Persist the origin enclosure URL** per episode at scrape time (pipeline addition + backfill).
2. **Resolve** an episode (by slug) to a *fresh, playable* origin URL: `GET /api/app/episodes/{slug}/audio-source`.
3. **Validate & refresh**: handle redirects/tracking prefixes, HEAD-check, re-resolve expiring URLs.
4. **Optional no-store pass-through proxy** — only when an origin blocks direct play; stream range requests,
   **store nothing**.
5. **Hard isolation**: consumers can never reach internal `media/` bytes.

## Constraints & Assumptions

**Constraints:**

- **Never store or transcode** third-party audio for delivery (Principle 4). The proxy, when used, is
  pass-through only — no disk, no cache of bytes.
- No real network in CI; resolution/proxy are tested against fixtures + stubbed HTTP.
- Internal `media/` stays operator-only (RFC-096 `/api/corpus/media` is not exposed on `/api/app/*`).

**Assumptions:**

- The episode's enclosure URL is available in the source feed at scrape time (verify; small pipeline add if
  not — see Open Questions).
- Most hosts permit direct cross-origin playback (industry norm for podcast enclosures).

## Design & Implementation

### 1. Capture origin URL at scrape time

The pipeline already reads the RSS enclosure to download audio. Persist that enclosure URL (and any feed
playback hints) into episode metadata + the `episode_slugs`/episode record (RFC-098 G4). A one-shot backfill
populates the existing corpus from feed metadata where available.

### 2. Resolution endpoint

```text
GET /api/app/episodes/{slug}/audio-source
-> 200 {
     "url": "https://cdn.host/ep123.mp3",   # fresh, playable
     "mime": "audio/mpeg",
     "duration_seconds": 4823,
     "strategy": "direct" | "proxy",
     "expires_at": "..."                      # if the resolved URL is signed/short-lived
   }
```

- **direct** (default): return the validated origin URL; client plays it.
- **proxy**: return a URL pointing at our pass-through endpoint (below), chosen only when direct is known to
  fail for this host.

### 3. Resolution & freshness

- Follow tracking-prefix/redirect chains to the terminal media URL; HEAD-validate (status + content-type).
- If the stored URL fails or is past `expires_at`, **re-resolve from the live feed** before returning.
- A small, bounded in-memory resolution cache (URL + validation result, short TTL) avoids re-HEADing on
  every play — caches *metadata only*, never bytes.

### 4. Optional no-store pass-through proxy

```text
GET /api/app/episodes/{slug}/audio-stream   # only when strategy == "proxy"
```

- Streams the origin response through, **forwarding HTTP Range** for seek; holds nothing on disk and caches
  no bytes. Used for: HTTP-only enclosures on an HTTPS app (mixed content), CORS-blocking hosts, or hotlink
  protection that a direct client request can't satisfy.
- This is a deliberate, documented softening of "never proxy" — it still **never stores** a copy. Default is
  direct; proxy is per-episode/per-host opt-in.

### 5. Isolation

The operator `/api/corpus/media` route is **not** mounted on the consumer API. The consumer audio module
only ever returns origin/proxy URLs. Internal `media/` remains a reprocessing asset (droppable).

## Key Decisions

1. **Bridge, not rehost — a dedicated subsystem**
   - **Decision**: resolve origin URLs at request time; never serve stored copies to users.
   - **Rationale**: Principle 4; legal + cost cleanliness; the operator's `media/` bytes are reprocessing
     data, not a delivery source.
2. **Direct by default, no-store proxy as fallback**
   - **Decision**: prefer client-direct playback; pass-through proxy only when a host blocks it.
   - **Rationale**: minimal bandwidth/latency; "never store" stays absolute, "never proxy" softens only
     under necessity — documented.
3. **Re-resolve expiring URLs from the feed**
   - **Decision**: treat stored URLs as hints, validate/refresh on read.
   - **Rationale**: signed/expiring CDN URLs are common; stale URLs must self-heal.

## Alternatives Considered

1. **Serve our stored `media/` bytes to users** — Rejected: that *is* rehosting (Principle 4) and a
   licensing/cost problem.
2. **Always proxy** — Rejected: bandwidth + latency + edges toward "serving audio"; only needed for hostile
   hosts.
3. **Store the enclosure URL once, return verbatim** — Rejected: ignores redirects, expiry, and hostile
   hosts; would break playback intermittently (G5).

## Testing Strategy

**Test Coverage:**

- **Unit**: redirect/tracking-prefix resolution; expiry detection; strategy selection (direct vs proxy).
- **Integration**: `audio-source` against fixture feeds + **stubbed HTTP** (200, 302 chain, 404→re-resolve,
  expiring URL); proxy streams Range correctly and writes nothing to disk; `/api/corpus/media` is **not**
  reachable on `/api/app/*`.
- **No real network** in CI.

**Test Organization:** `tests/integration/app_api/test_audio_bridge.py`; HTTP stub fixtures; assert
zero-byte persistence in proxy mode.

## Rollout & Monitoring

- **P0/P1**: capture + backfill origin URLs; ship direct resolution; add proxy fallback as hosts require.
- **Monitoring**: resolution success rate, re-resolve rate, proxy-fallback rate per host, audio start
  failures. A rising proxy rate flags hostile hosts.
- **Success**: episodes play from origin with reliable seek; nothing third-party is stored for delivery.

## Open Questions

1. **Confirm enclosure persistence** — is the origin URL captured today, or is a small pipeline add needed?
   (Flagged in the gap analysis; verify before P1.)
2. Proxy resource limits (max concurrent pass-through streams) and where it runs in prod hosting (RFC-082).
3. Analytics/affiliate prefixes some hosts require — preserve them on direct play?

## References

- **Related PRDs**: `docs/prd/PRD-039-player.md`, `docs/prd/PRD-035-learning-platform.md`
- **Related RFCs**: `docs/rfc/RFC-096-audio-pipeline-separation-and-viewer-media.md`, `docs/rfc/RFC-098-learning-platform-foundation.md`
- **Source Code**: pipeline enclosure capture; new `server/app_api/` audio module
- **Analysis**: `docs/wip/player/SERVER-SIDE-GAP-ANALYSIS.md` (G5)
