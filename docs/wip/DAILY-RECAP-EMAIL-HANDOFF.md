# Daily recap email — status (#2039 / RFC-122)

Built end-to-end across both repos. The app (this repo) assembles + enqueues the envelope; the
homelab delivery worker renders + sends it (ADR-144: self-hosted outbox queue, Resend last-mile).

## App side (this repo) — done

- **Assembles** (`app_digest_daily_recap.assemble_daily_recap_payload`) from the episodes a user
  **finished today** (`listening.finished_at`, UTC day; per-user timezone is a tracked follow-up),
  0 → no envelope.
- **Enqueues** a `daily-recap.v1` / `type: daily_recap` email envelope, gated on
  `types.daily_recap.email` opt-in + `daily_recap_schedule.paused` + a verified email, at the
  `daily_recap_schedule.hour` (UTC) slot, once per day (idempotent per-day id `drcp_…`).
- Extends the committed seam schema (`docs/api/delivery-envelope.schema.json`): `type` +=
  `daily_recap`, `template` += `daily-recap.v1`.

## Worker side — done (agentic-ai-homelab, commit e1ea3a5, NOT yet deployed)

`~/Projects/agentic-ai-homelab/infra/delivery/`:

- `delivery/templates/podcast/email/daily-recap.v1.{subject,html}.j2` — the DARK Close-Listening
  brand (same shell as your-week), adaptive (full for 1 episode, compact stack for many).
- Also added `recommendations-digest.v1.{subject,html}.j2` — that monthly digest was enqueued but
  the worker had NO template, so it was silently never sending. Now it renders.
- `render.py`: type-aware one-click unsubscribe (`…/comms/unsubscribe?ref=<ref>&type=<type>`) so a
  daily_recap unsub disables the recap, not the weekly digest; `new_in_interests` section label;
  monthly `_period_label`. `envelope.py` parses the new `type`.
- Vendored seam schema synced from the app + golden fixtures for both new emails. Worker suite: 36
  passed.

> ⚠️ The earlier `docs/wip/daily-recap-email.html` light/gold mock was OFF-BRAND — the real emails
> are dark (`#0b0e14`, "Close Listening"). Deleted; the worker `.j2` templates are the source of truth.

## The only remaining step: DEPLOY the worker

`RESEND_API_KEY` is already set (weekly digests send today). Deploy the updated homelab worker so it
picks up the two new templates + the synced schema. Nothing else — no app deploy is required for the
worker change.

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
