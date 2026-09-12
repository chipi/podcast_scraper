# Daily recap email — deploy notes (#2039 / RFC-122)

The daily post-episode recap now works **end-to-end in this repo**: assemble → enqueue → render →
send. The renderer + delivery worker were built in-repo (no separate service needed) but still
consume ONLY the committed outbox seam, so they could move to a standalone infra service unchanged.

## The pipeline (all in-repo)

- **Assemble** — `app_digest_daily_recap.assemble_daily_recap_payload`: the episodes a user
  **finished today** (`listening.finished_at`, UTC day; per-user timezone is a tracked follow-up),
  0 → no envelope.
- **Enqueue** — `daily-recap.v1` email envelope to the outbox, gated on `types.daily_recap.email`
  opt-in + `daily_recap_schedule.paused` + a verified email, at the `daily_recap_schedule.hour`
  (UTC) slot, once per day (idempotent per-day id `drcp_…`).
- **Render** — `app_email_render.render_email`: adaptive (full for 1 episode, compact stack for
  many), HTML-escaped, deep links absolutized. Visual spec: `docs/wip/daily-recap-email.html`.
- **Send** — `app_delivery_worker.deliver_pending_emails` drains the outbox
  (`list_pending`/`record_status`), builds the type-aware RFC-8058 List-Unsubscribe header
  (`…/comms/unsubscribe?ref=<ref>&type=daily_recap`), sends via Resend's REST API
  (`app_email_send`, httpx — no new dep), records `delivered`. Wired into the hourly digest cron
  (enqueue → drain in one fire) + a CLI (`python -m podcast_scraper.server.app_delivery_worker`).

## To actually send on deploy (the ONLY remaining step)

Safe-by-default: with no key the worker **dry-runs** (renders + logs, sends nothing). To go live,
set in the deploy env:

- `RESEND_API_KEY` — the Resend API key (secret; never committed).
- `EMAIL_FROM` — e.g. `closelistening <recap@mail.closelistening.app>` (optional; sensible default).
- `APP_ORIGIN` — e.g. `https://closelistening.app` (optional; defaults to it).

And in Resend itself: verify the `mail.closelistening.app` sending domain (SPF/DKIM). Nothing else.

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
