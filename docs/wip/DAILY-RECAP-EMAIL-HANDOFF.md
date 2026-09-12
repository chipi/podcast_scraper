# Daily recap email — delivery-worker handoff (#2039 / RFC-122)

The app (this repo) now assembles + enqueues the daily post-episode recap as a `DeliveryEnvelope`.
The **delivery worker (#1412, separate repo)** must render + send it. This is the contract.

## What the app already does

- **Assembles** the recap (`app_digest_daily_recap.assemble_daily_recap_payload`) from the episodes
  a user **finished today** (UTC day; per-user timezone is a tracked follow-up), 0 → no envelope.
- **Enqueues** an email envelope to the outbox (`type: "daily_recap"`, `template: "daily-recap.v1"`),
  gated on `types.daily_recap.email` opt-in + `daily_recap_schedule.paused` + a verified email, at the
  `daily_recap_schedule.hour` (UTC) slot, once per day (idempotent per-day envelope id `drcp_…`).
- Extends the committed seam schema (`docs/api/delivery-envelope.schema.json`): `type` gains
  `daily_recap`, `template` gains `daily-recap.v1`.

## What the worker must do

1. **Drain** `daily-recap.v1` email envelopes from `/internal/outbox/pending?channel=email` (already
   generic — no change needed) and report status as for other templates.
2. **Render** `daily-recap.v1` from the payload (below), **adaptively**:
   - `count == 1` → the FULL recap (key points + signature quote + top insights + topic chips +
     storyline links).
   - `count > 1` → a COMPACT per-episode stack (title + show + signature quote + up to 2 key points
     + an open link).
   - Visual spec / reference markup: **`docs/wip/daily-recap-email.html`** (email-safe tables +
     inline styles, light background, `closelistening.` wordmark + gold accent `#b6791f`).
3. **Deep links**: each item's `deep_link` is app-relative (`/player/<slug>`); prefix with the app
   origin (`https://closelistening.app`).
4. **One-click unsubscribe (RFC-8058)**: build from the envelope's **top-level `type`** + the
   `consent_snapshot.unsubscribe_ref`:
   - `List-Unsubscribe: <https://closelistening.app/api/app/comms/unsubscribe?ref=<ref>&type=daily_recap>, <mailto:…>`
   - `List-Unsubscribe-Post: List-Unsubscribe=One-Click`
   - The `&type=daily_recap` is REQUIRED — without it the unsub flips the weekly digest instead.

## Payload shape (`daily-recap.v1`)

```jsonc
{
  "day": "2026-09-11",          // UTC day
  "count": 3,
  "episodes": [
    {
      "slug": "feed_ep",
      "title": "…",
      "podcast_title": "…",       // show
      "artwork_url": "…" | null,  // app-hosted large art, or null
      "key_points": ["…", "…"],   // summary bullets, ≤3 (compact view shows ≤2)
      "signature_quote": { "text": "…", "speaker": "…" | null } | null,
      "insights": ["…", "…"],     // top salience insights, ≤3 (full view only)
      "topics": [{ "id": "topic:…", "label": "…" }],       // chips → /topic/:id
      "storylines": [{ "id": "topic:…", "label": "…" }],   // → /storyline/:id (anchor topic id)
      "deep_link": "/player/feed_ep"
    }
  ]
}
```

Bridge-only: text + KG refs + artwork; there is never an audio URL in the payload.

## Open follow-up

Sends fire at a fixed UTC hour for every recipient; an end-of-day recap should land at the
recipient's local end of day. Adding a per-user timezone + an hourly fan-out is tracked separately
(see the timezone discussion item) and is NOT part of this cut.
