# ADR-146: "Your Week" is a primary in-app surface, with a sidecar scheduler and route-local enrichment

- **Status**: Accepted — 2026-08-07. Implemented on `feat/your-week` (#1412).
- **Date**: 2026-08-07
- **Authors**: Marko Dragoljevic, Claude (Opus 4.8)
- **Related**: [ADR-145](ADR-145-channel-agnostic-outbox-seam.md) (the outbox seam this reuses), [ADR-144](ADR-144-self-hosted-delivery-queue-outsourced-last-mile.md) (the delivery worker), [ADR-116](ADR-116-privilege-split-public-control-api.md) (app_only player), [RFC-110](../rfc/RFC-110-outbound-delivery-and-seam.md), [RFC-101](../rfc/RFC-101-personal-knowledge-corpus.md), [PRD-046](../prd/PRD-046-delivery-and-curation.md)
- **Tracking**: epic #1413, infra slice #1412

## Context

The personal digest ("Your Week") shipped as **email-only** (RFC-110): the app assembles a rollup
(revisit + new-in-follows + trending-in-your-corpus) and enqueues a `DeliveryEnvelope`; the homelab
worker renders + sends it. Two problems surfaced once it was live:

1. **The capability was gated behind email.** Turning the digest email off (a legitimate choice) lost
   the whole thing — there was no in-app view of the rollup. That inverts the value: recommendations
   are the winning surface (the Spotify pattern), and they belong *in the app*.
2. **It never auto-fired on the player.** The player runs `PODCAST_SERVE_APP_ONLY=1` (ADR-116), which
   force-disables the in-process job scheduler (`app.py`: `enable_jobs_api = False`). The committed
   `config/player/viewer_operator.player.yaml` scheduled-job was therefore **inert** — nothing read
   it, and no digest ever fired autonomously.

Also, "digest" was overloaded: it already names the **operator** corpus-wide feature (the operator
viewer's Digest tab / corpus digest), and was being reused for the personal one.

## Decision

1. **"Your Week" is a primary in-app surface.** New route `GET /api/app/your-week` serves the rollup
   synchronously to the signed-in player user; the email becomes the *edge* for when you don't visit.
   The home page shows it as the first personalized block (compact rail / full sections, a synced
   per-user layout preference).

2. **The in-app view is decoupled from email consent.** Showing a user their *own* data in-app needs
   no outbound-comms consent. `comms.digest.enabled` governs **only** the email/push delivery; the
   in-app view is always available. (Delivery stays consent-gated — unchanged.)

3. **One assembler, two surfaces.** The route reuses `app_digest_personal.assemble_digest_payload`
   verbatim — the single source of truth — so in-app and email never drift.

4. **Enrichment is route-local, never in the envelope.** The route adds `image_url` (episode/show
   artwork for the card backdrop) and backfills `episode_title` for topic-centric trending items. The
   `DeliveryEnvelope` schema keeps `additionalProperties: false` on items, which walls these off: the
   assembler MUST NOT write them into the envelope, and the worker renders from `graph_refs`, not art.

5. **Scheduling for the app_only player is a compose sidecar**, not the in-process scheduler. A tiny
   `digest-scheduler` container (same image, `network_mode: none`, `user 1000:1000`, read-only,
   cap-drop-all, heartbeat healthcheck) wakes on an aligned interval and calls
   `enqueue_due_digests`. This keeps the player least-privilege (ADR-116) and is reproducible on a
   fresh box (it comes up with `docker compose`); the inert `viewer_operator.player.yaml` is deleted.
   The appdata stores are multi-process-safe (per-envelope `FileLock` + atomic rename), so a second
   writer is safe.

6. **Naming**: **Digest** = the operator corpus-wide feature; **Your Week** = the personal/consumer
   feature (user-facing name + `your-week-digest.v1` template). New code uses `your_week`.

## Consequences

- The digest survives an email opt-out; the in-app surface is the primary experience.
- The digest auto-fires on the player without granting it scheduler/jobs privileges.
- In-app and email cannot drift (shared assembler); the envelope contract stays frozen (ADR-145).
- A second appdata writer (the sidecar) exists — acceptable because the stores are multi-process-safe.
- **Follow-ups (known, not blocking):** the route does a second `build_catalog_rows` scan for
  enrichment (fine at current corpus size; thread the catalog through the assembler if it grows); a
  homelab-side dead-man alert (no enqueue in >8 days while ≥1 user consents) still to wire; the
  deep-rename of the internal `app_digest_personal` module + the `your-week-digest.v1` template string
  is deferred (it touches the homelab worker's vendored schema — a coordinated cross-repo change).

## Alternatives considered

- **In-process scheduler on the player** — rejected: app_only disables it by design (ADR-116);
  re-enabling would punch a hole in the least-privilege posture for a scheduling concern.
- **Host systemd timer** — rejected: the deploy user's sudo is scoped to `systemctl restart caddy`
  only; it cannot install units. The sidecar needs no host privileges.
- **Add `image_url` to the shared assembler / envelope** — rejected: it would change the frozen seam
  contract + the homelab worker's vendored schema for an in-app-only rendering concern.
