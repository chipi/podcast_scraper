# Notifications & Delivery Guide

How Podcast Scraper notifies users — digest emails, push nudges, native iOS
push, the daily recap, and the in-app inbox. This is the end-to-end map: who
produces a notification, how consent gates it, how it crosses the app↔infra
seam, and how the last-mile worker actually sends it.

> **Design authority:** [ADR-144](../adr/ADR-144-self-hosted-delivery-queue-outsourced-last-mile.md)
> (self-hosted queue, outsourced last-mile), [ADR-145](../adr/ADR-145-channel-agnostic-outbox-seam.md)
> (channel-agnostic outbox seam), [RFC-110](../rfc/RFC-110-outbound-delivery-and-seam.md)
> (full spec), [RFC-122](../rfc/RFC-122-post-episode-recap.md) (daily recap).
>
> **Where the code lives:** the *producer* half (subscriptions, consent,
> generation, the outbox seam) is in this repo under
> `src/podcast_scraper/server/`. The *delivery worker* (last-mile sender)
> currently lives in the **homelab infra repo** at `agentic-ai-homelab/infra/delivery/`.
> Its packaging is being reevaluated in
> [podcast_scraper#2077](https://github.com/chipi/podcast_scraper/issues/2077) —
> that only moves *where the files live*, not how any of this works.

## The shape in one picture

The system is a **two-stage, channel-agnostic boundary**. The app produces
structured envelopes into an outbox; a stateless worker drains, renders, and
sends them, then reports terminal status back so the app can suppress bad
recipients.

```text
 PRODUCER (this repo, learning-player API)          SEAM             LAST-MILE (homelab worker)
 ─────────────────────────────────────────   ─────────────────   ──────────────────────────────
 subscription  ─┐                             GET  /internal/      pull pending ──▶ render template
 consent matrix ─┼─▶ hourly scheduler ──▶     outbox/pending?         │              (Jinja, per channel)
 finished today ─┘   assemble + enqueue       channel=email|push      │                    │
                      DeliveryEnvelope ──▶  ┌─ outbox store ──────────▶│              send via:
                      (idempotent id)       │  (file per envelope)     │               • Resend  (email)
                                            │                          │               • Web Push (VAPID)
 consent write-back ◀── _suppress() ◀───────┴─ POST /internal/  ◀──────┘               • APNs    (native iOS)
 (bounce/complaint/unsub)                      outbox/{id}/status  ◀── terminal status ──┘
```

**Key properties**

- **Channel-agnostic + frozen contract.** The envelope `payload` is structured
  JSON (never HTML). All rendering and protocol detail lives on the worker
  side. The schema is versioned (`schema_version: "1"`).
- **Idempotent everywhere.** Every envelope has a deterministic `id`
  (period-keyed), so re-running the scheduler or re-posting a status is a no-op.
- **Consent stays with the app.** The worker never reads or writes the consent
  store; the app filters at enqueue *and* re-checks at pull, and applies
  suppression from delivery feedback.
- **Tailnet-only, egress-only.** The seam is a tailnet-internal HTTP API gated
  by a shared token; the worker only makes outbound calls (Resend / push over
  443).

---

## Part 1 — Producing notifications (this repo)

### 1.1 Subscriptions

A user opts in to push per device. Two kinds share one endpoint and one store.

**Web Push (browser, VAPID)** — `web/learning-player/src/composables/usePushSubscription.ts`
(`enablePush()`):

1. `Notification.requestPermission()`
2. `GET /api/app/push/vapid-key` → the server's `APP_VAPID_PUBLIC_KEY` (503 if unset)
3. `pushManager.subscribe({ applicationServerKey })`
4. `POST /api/app/push/subscribe` with the W3C `PushSubscription` (endpoint + keys)

**Native iOS (Capacitor, APNs)** — `enablePushNative()` (the #2068/#2072 work):

1. `PushNotifications.requestPermissions()` (Capacitor plugin)
2. listen for the `registration` event → the APNs **device token**
3. `POST /api/app/push/subscribe` with
   `{ endpoint: "apns://<token>", kind: "apns", platform, token }`

Both land in the same per-user store
(`src/podcast_scraper/server/app_push_store.py` → `<data_dir>/users/<id>/push_subscriptions.json`),
deduped on the `endpoint` string. The `PushSubscription` schema is
`extra="allow"`, which is why the APNs `kind`/`platform`/`token` fields ride
through without a schema change. Removing the last subscription auto-disables
push everywhere for that user.

> **Why native APNs at all?** An iOS Capacitor WKWebView has no ServiceWorker /
> PushManager, so Web Push silently fails there. Native APNs (device token →
> `apns://` subscription) is the iOS path; the worker sends it over APNs HTTP/2.

### 1.2 Consent — a type × channel matrix

`src/podcast_scraper/server/app_comms_store.py` →
`<data_dir>/users/<id>/comms.json`.

- **Types:** `digest`, `daily_recap`, `new_episodes`, `product`
- **Channels:** `email` (opt-in, default **off**), `push` (opt-in, default
  **off**), `in_app` (default **on**)
- **Schedules:** `digest_schedule` (`cadence` weekly|daily, `day_of_week`,
  `hour`, `paused`) and `daily_recap_schedule` (`hour`, `paused`)
- **Timezone:** IANA string; all "due" checks use the user's **local** hour
  (empty = UTC fallback)
- **`unsubscribe_ref`:** an opaque per-user string minted on first save, used
  by one-click unsubscribe links

Managed via `GET`/`PUT /api/app/comms`. `channel_enabled(comms, type, channel)`
is the boolean gate enforced everywhere an envelope is generated.

### 1.3 Generation — the four templates

Every generator assembles a **structured, graph-carrying** payload (each item
carries `graph_refs` + a `deep_link`; bridge-only, no audio) and returns an
envelope only if consent is met and there is content.

| Template | Type | Channel | Source | Cadence |
| -------- | ---- | ------- | ------ | ------- |
| `your-week-digest.v1` | `digest` | email | `app_digest_personal.py` (revisit + new-in-follows/interests + trending) | weekly or daily |
| `resurface-nudge.v1` | `digest` | push | `app_digest_personal.py` (one **per subscription**) | daily |
| `daily-recap.v1` | `daily_recap` | email | `app_digest_daily_recap.py` (episodes you finished today) | daily |
| `recommendations-digest.v1` | `digest` | email | `app_digest_recommendations.py` (discovery only) | monthly |

**Consent gates** (examples):

- Email digest: `digest.email` enabled · not paused · `email_verified`
- Push nudge: `digest.push` enabled · ≥1 subscription · graph-carrying content
- Daily recap: `daily_recap.email` enabled · not paused · `email_verified` ·
  finished ≥1 episode today (local day)

### 1.4 Scheduling & enqueue

An **hourly** scheduler job (`src/podcast_scraper/server/scheduler.py`,
`kind="digest"`) fans out to the three orchestrators:

```python
enqueue_due_digests(...)          # your-week (email) + resurface (push)
enqueue_due_recommendations(...)  # monthly discovery digest
enqueue_due_daily_recaps(...)     # daily recap
```

Each checks the user's **local** due-slot, then writes envelopes to the outbox
store (`app_outbox_store.py` → `<data_dir>/outbox/<sha256(id)>.json`). The
`id` is period-keyed (`dgst_<period>_<uid>`, `drcp_<YYYYMMDD>_<uid>`, …), so
`enqueue()` is a **no-op if the id already exists** — re-runs are safe.

> **Timezone note (not-yet-per-user at fire time):** the app enqueues at a fixed
> UTC hour today; per-user local timezone at the schedule level is tracked
> separately. See the recap/digest scheduler for current behavior.

---

## Part 2 — The seam (app ↔ infra)

The delivery worker pulls from a small, token-gated HTTP API served by the
learning-player API. Routes: `src/podcast_scraper/server/routes/internal_outbox.py`.

### 2.1 Auth

Every `/internal/outbox/*` call presents `X-Internal-Token`, compared against
`app.state.internal_outbox_token` (from the `INTERNAL_OUTBOX_TOKEN` env var,
`app.py`). **Unset → 503** (the endpoints are hard-disabled), **mismatch →
401**. The worker's matching value is `PODCAST_INTERNAL_OUTBOX_TOKEN` (same
secret, a different variable name on the worker side).

### 2.2 Endpoints

**`GET /internal/outbox/pending?channel=email|push&limit=N`** → `{ envelopes: [...] }`

- pending-only, oldest-first, capped at `limit`
- **re-checks *current* consent** (live `comms`, not the frozen
  `consent_snapshot`) — a user who unsubscribed or paused since enqueue is
  dropped here
- **excludes past-`expires_at`** envelopes (no stale flush after an outage)

**`POST /internal/outbox/{id}/status`** with `{ status, detail? }`

- terminal statuses: `delivered`, `bounced`, `complaint`, `suppressed`, `failed`
- **idempotent per id** — a repeated status returns the stored one, no-op
- **suppression write-back:** on `bounced`/`complaint`/`suppressed`, the app
  flips the matching consent cell (`_suppress()` → `set_channel(..., False)`;
  a bad push endpoint → `disable_push_everywhere`). This is how the app stops
  producing for a dead recipient without the worker ever touching consent.

### 2.3 The envelope

`DeliveryEnvelope` (schema: `docs/api/delivery-envelope.schema.json`, golden
fixtures under `tests/fixtures/delivery/`, contract test
`tests/unit/server/test_delivery_envelope_contract.py`):

```jsonc
{
  "schema_version": "1",
  "id": "dgst_2026W37_u123",       // idempotency key (period + user)
  "user_id": "u123",
  "channel": "email",               // email | push
  "type": "digest",                 // informational
  "template": "your-week-digest.v1",
  "recipient": { "email": "…", "email_verified": true },
                                    // or { "push_subscription": <W3C sub> } / apns sub
  "consent_snapshot": { "digest_enabled": true, "cadence": "weekly",
                        "unsubscribe_ref": "…" },   // informational (live consent re-checked at pull)
  "payload": { "sections": [ /* graph-carrying items, NOT HTML */ ] },
  "not_before": "…", "expires_at": "…", "created_at": "…"
}
```

---

## Part 3 — Delivering (the homelab worker)

`agentic-ai-homelab/infra/delivery/` — a standalone Python package with no
`podcast_scraper` dependency; it consumes only the **vendored** seam schema.
See that repo's `README.md` and `HANDOVER-homelab.md`.

### 3.1 Three long-running loops

`delivery-email`, `delivery-push`, `delivery-events` (one worker per tenant per
channel). Each loop: `GET …/pending?channel=…` → render the `template` (Jinja)
→ send → `POST …/status`.

- **email** → **Resend** HTTP API (443, reputation relay). Templates:
  `your-week-digest`, `daily-recap`, `recommendations-digest`.
- **push** → dispatched by subscription `kind`:
  - `webpush` → self-hosted VAPID / RFC 8291 POST to the subscription endpoint
  - `apns` → **APNs** token-based ES256-JWT over HTTP/2 (native iOS). Prod uses
    the production APNs host (`apns_sandbox: false`); dev-signed builds need a
    sandbox tenant (see §4).
- **events** → polls Resend's events API (cursor-based) and posts
  `bounced`/`complaint` back to the seam.

### 3.2 Tenants

`tenants.yaml` in the worker: one worker set per tenant.

- **`podcast`** — prod. Drains the prod player-API outbox over the tailnet;
  APNs `apns_sandbox: false`.
- **`podcast-dev`** — dev twin. Its outbox URL is env-overridable
  (`PODCAST_DEV_OUTBOX_URL`) so it follows wherever you run the dev API;
  APNs `apns_sandbox: true` for dev-signed device builds.

### 3.3 Reliability & observability

- **Retry + dead-letter:** transient failures retry with backoff; `failed` is
  terminal (dead-lettered after N).
- **Idempotency:** keyed on envelope `id` end-to-end.
- **o11y:** all three channels emit **logs** (JSONL → VictoriaLogs), **metrics**
  (Prometheus `/metrics` → VictoriaMetrics, labeled by channel), and **traces**
  (OTEL spans → VictoriaTraces, `service.name=delivery-worker`). Errors →
  GlitchTip.

---

## Part 4 — Operating it

### 4.1 Local dev testing (email + push, without TestFlight)

The dev API (`make serve-api`, port `:8000`) serves `/internal/outbox`. It now
binds `0.0.0.0` by default, so it's reachable over the tailnet with no extra
flags:

```bash
# laptop: run the dev API (token comes from .env → INTERNAL_OUTBOX_TOKEN)
make serve-app-dev

# mini: point the delivery worker's podcast-dev tenant at your laptop, once
~/agentic-ai-homelab/infra/delivery/bin/dev-outbox.sh laptop
```

Then a dev-signed iOS build registers its **sandbox** APNs token, a digest
enqueues, and the `podcast-dev` worker (sandbox) delivers to your device.
Email needs nothing extra — the `podcast-dev` email worker sends via Resend
(real mail from the shared `mail_from`, so use safe recipients).

### 4.2 Prod

- **Server:** deploy the app (token-registration + enqueue code) to the prod
  VPS via the normal player deploy.
- **iOS:** a **TestFlight / App Store** build → production APNs tokens (matches
  the prod `podcast` tenant's `apns_sandbox: false`).
- **Worker:** already runs on the homelab mini for the `podcast` tenant.

### 4.3 Config / env (both sides)

| Where | Var | Purpose |
| ------ | --- | ------- |
| app | `INTERNAL_OUTBOX_TOKEN` | gate `/internal/outbox/*` (503 unset / 401 mismatch) |
| app | `APP_VAPID_PUBLIC_KEY` | browser Web Push subscription |
| worker | `PODCAST_INTERNAL_OUTBOX_TOKEN` | same value as the app's token |
| worker | `RESEND_API_KEY` | email last-mile |
| worker | `PODCAST_VAPID_PRIVATE_KEY` | Web Push signing (pairs the app's public key) |
| worker | `PODCAST_APNS_AUTH_KEY` (+ key/team/bundle ids in `tenants.yaml`) | native iOS APNs |
| worker | `PODCAST_DEV_OUTBOX_URL` | dev: point `podcast-dev` at the active dev API |

---

## In-app inbox (the `in_app` channel)

Separate from the outbound worker: `app_notifications_store.py` writes to
`<data_dir>/users/<id>/notifications.json` (bounded, newest-first), gated by the
same `in_app` consent cell. Read via `GET /api/app/notifications` +
`.../read` / `.../read-all`. This never leaves the app — no seam, no worker.

## One-click unsubscribe

Emails carry an RFC 8058 `List-Unsubscribe-Post` header and a link
`…/api/app/comms/unsubscribe?ref=<unsubscribe_ref>&type=<type>`. `GET` renders
a confirm page (no mutation, guards email prefetch); `POST` resolves the ref →
user and disables that one email type. Type-aware, so unsubscribing from the
daily recap doesn't touch the weekly digest.

---

## See also

- [ADR-144 — Self-hosted delivery queue, outsourced last-mile](../adr/ADR-144-self-hosted-delivery-queue-outsourced-last-mile.md)
- [ADR-145 — Channel-agnostic outbox seam](../adr/ADR-145-channel-agnostic-outbox-seam.md)
- [RFC-110 — Outbound delivery and seam](../rfc/RFC-110-outbound-delivery-and-seam.md)
- [RFC-122 — Post-episode recap](../rfc/RFC-122-post-episode-recap.md)
- [Development Guide](DEVELOPMENT_GUIDE.md) — `make serve` / `serve-api`
- [Server Guide](SERVER_GUIDE.md) — the `/api/*` surface
- [podcast_scraper#2077](https://github.com/chipi/podcast_scraper/issues/2077) —
  reevaluating where the delivery worker's code lives
