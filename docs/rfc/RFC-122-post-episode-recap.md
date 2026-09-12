# RFC-122: Post-episode recap (in-app panel + daily listening email)

- **Status**: Draft
- **Authors**: Marko Dragoljevic
- **Stakeholders**: Player / consumer app; delivery (comms) owner
- **Related RFCs**:
  - `docs/rfc/RFC-099-learning-platform-consumer-client.md` — the consumer client; the "more like
    this" / similar-episodes engine (#1084) the recap reuses lives under it
  - `docs/rfc/RFC-120-login-first-lure-landing.md` — the recap is a signed-in surface
  - `docs/rfc/RFC-119-holistic-collections.md` — "save from the recap" reuses collections
- **Related UX specs**:
  - `docs/uxs/UXS-014-interaction-patterns.md` (new §Post-episode recap)
- **Related issues**: #2038 (in-app panel), #2039 (daily recap email)

## Abstract

When an episode ends, the player just stops. This RFC proposes a **post-episode recap** that turns
the completion moment into reinforcement: an **in-app panel** (#2038) that appears in the player's
footprint the instant an episode finishes, and a **daily listening email** (#2039) that batches the
day's listens as a delayed nudge. Both render the **same recap model** — summary key points, top
insights, one signature quote, and "listen more like this" — assembled once from artifacts we
already produce (GI insights, the episode summary, the RFC-099 similar-episodes engine). It is
bridge-only (transcript-derived text + KG metadata + artwork, never audio) and reuses the existing
delivery plumbing for the email; almost no new intelligence, mostly assembly + two surfaces.

**Architecture Alignment:** a thin read-projection over existing GI / summary / KG artifacts + a new
render target for the existing outbox — no new data pipeline, consistent with the consumer-app
read-only projection model (RFC-098) and the OG-card assembly pattern (#2036, `server/og/build.py`).

## Problem Statement

The end of an episode is the highest-intent moment we have — the listener just invested 30–60
minutes — and today we waste it: playback stops and nothing reinforces what they heard or offers a
next step. We already derive an episode's summary, its salience-ranked insights, its KG people/
topics, and a similar-episodes ranking, but that value sits behind taps the just-finished listener
won't make. Retention and "did I actually learn something" both suffer.

Separately, we have no re-engagement touch after a session: a listener who finishes an episode gets
no reminder of what mattered and no reason to come back tomorrow.

**Use Cases:**

1. **Reinforce the listen**: an episode ends; the panel shows "your notes" (key points + top
   insights + one signature quote) so the listener leaves with the gist consolidated.
2. **Surface the next listen**: the same panel offers a mini-grid of similar episodes, one tap to
   keep going.
3. **Delayed nudge**: that evening, a daily digest email recaps the day's episode(s) — key points,
   top insights, a quote, and next recommendations — a first re-engagement touch.

## Goals

1. **Reinforcement surface at completion**: an in-app recap panel in the player's footprint, keyed
   on episode-finished, dismissible.
2. **One shared recap model**: assemble summary + insights + signature quote + similar-episodes once
   (`recap_view`), consumed by both the panel and the email — no divergence between surfaces.
3. **Daily listening email**: a per-user digest of the day's completed episodes, on the existing
   delivery stack, respecting notification preferences + one-click unsubscribe.
4. **Reuse, don't rebuild**: GI insights (`insights_from_gi`), the summary fields, the RFC-099
   similar engine (`run_similar_episodes`), and the comms/outbox pipeline — no new artifacts.
5. **Bridge-only**: text + KG metadata + artwork only; never audio.

## Constraints & Assumptions

**Constraints:**

- Signed-in only (RFC-120 login-first); anonymous sessions get no recap/email.
- Bridge-only (PRD-035 Principle 4): the recap and email carry transcript-derived text + KG metadata
  + artwork, never a stream URL or audio.
- The email must honour notification preferences and carry RFC-8058 one-click unsubscribe.
- The recap build must not block the request path (assemble off the event loop; same lesson as
  #2036 `spa.py`).

**Assumptions:**

- "Finished" is derivable from the existing playback/completed state (client progress + the
  completed store, PL.6).
- The per-user "what did they listen to today" set is derivable from the same heard/completed signal
  the corpus-scope lens already uses.
- The email is a new template + trigger on the existing outbox/Resend delivery worker (#1412 /
  #1415), not new infrastructure.

## Design & Implementation

### 1. The shared recap model (`recap_view`)

A read projection, `recap_for_episode(root, slug) -> Recap`, assembled from artifacts we already
have — the single source both surfaces render:

- **Key points** — the episode `summary_bullets` (fall back to `summary_text`).
- **Top insights** — `insights_from_gi(gi, limit=N)` (salience-ranked; the same projection the
  episode/topic surfaces use).
- **Signature quote** — the strongest attributed quote: `insights[0].quotes[0]` with its speaker
  (same selection as the #2036 card's topic quote), shown verbatim with attribution.
- **More like this** — `run_similar_episodes(...)` (RFC-099), the same ranking that feeds the
  episode-related rail; capped to a small N for the mini-grid.

Best-effort per field (a thin corpus drops a field, never fails the recap), mirroring
`server/og/build.py`.

### 2. In-app panel (#2038)

- **Trigger**: episode *finished* = playback reaches the end **or** ≥95% heard (not gated on the
  explicit mark-as-played tap). Fires per-episode.
- **Footprint**: the recap **replaces the player's footprint in place** on the episode page (same
  size), with a clear dismiss back to the finished player. Not a full-screen takeover; not a global
  sheet over the mini-player.
- **Layout**: kicker "You just finished" → episode title → **key points** → **top insights** → the
  **signature quote** (the emotional anchor) → a **"listen more like this"** mini-grid. Save-to-
  collection (RFC-119) available inline. Key people/topics chips are intentionally omitted to keep
  the panel focused.
- **Data**: one call to the recap projection (`GET /api/app/episodes/{slug}/recap`), reusing the
  existing summary/GI/similar reads.

### 3. Daily listening email (#2039)

- **Cadence**: **daily digest first** — one email per user batching the day's completed episodes.
  (A per-episode recap email is a later follow-up, not in this RFC's first cut.)
- **Trigger**: a daily job, per user with ≥1 completed episode that day and email enabled, over the
  existing scheduler; assembles a digest from each episode's `recap_view`.
- **Content**: the day's episode(s) + show, key points, top insights, a signature quote, and a few
  next recommendations — the same recap model, rendered to an email template.
- **Delivery**: the existing outbox → Resend worker (#1412 / #1415), notification-preference gated,
  with RFC-8058 one-click unsubscribe.

### Integration Points

- **Player / completed store (PL.6)**: emits the finished signal that opens the panel; supplies the
  day's listened set for the email.
- **GI / summary / KG projections**: `insights_from_gi`, the catalog summary fields — read-only.
- **RFC-099 similar engine** (`run_similar_episodes`): the "more like this" ranking.
- **Comms / outbox** (#1415 internal outbox, Resend, RFC-8058 unsubscribe, notification prefs): the
  email delivery path.

## Key Decisions

1. **Trigger = finished (end or ≥95%)** — Rationale: catches real completion even when listeners
   skip the outro; not the rarely-tapped mark-as-played.
2. **Panel replaces the player in place, dismissible** — Rationale: matches the YouTube end-screen
   mental model, least disruptive; avoids a global sheet fighting the persistent mini-player.
3. **Recap content = key points + top insights + one signature quote** — Rationale: consolidate the
   gist + one memorable anchor; people/topics chips left out to stay focused.
4. **Email = daily digest first, panel per-episode** — Rationale: immediate reinforcement belongs
   in-app at each completion; the email is a calm once-a-day nudge, not an inbox per episode.
5. **One shared recap model** — Rationale: the panel and the email must never drift; assemble once,
   render twice (the #2036 card lesson).

## Alternatives Considered

1. **Full-screen recap takeover** — Pros: max attention. Cons: interrupts flow, heavy-handed. Rejected.
2. **Per-episode recap email first** — Pros: tightest loop. Cons: inbox noise; a daily digest is the
   calmer default and still reinforces. Deferred to a follow-up.
3. **Rebuild bespoke summary/insight components for the recap** — Cons: duplicates existing surfaces
   and invites drift. Rejected in favour of reusing the projections + components.

## Testing Strategy

**Test Coverage:**
- **Recap projection (unit)**: assembly from a fixture corpus — key points, top insights, signature
  quote selection + attribution, similar-episodes cap; best-effort field-drop on a thin corpus.
- **Panel trigger (client unit)**: finished = end / ≥95% fires; mark-played alone does not; dismiss
  restores the player.
- **Email digest (integration)**: the daily job selects the right per-user set, renders the template,
  respects notification prefs + unsubscribe, and is bridge-only (no audio field).

**Test Organization:** server unit under `tests/unit/podcast_scraper/server`; email/job integration
under `tests/integration/server`; client vitest for the panel.

## Rollout & Monitoring

**Rollout Plan:**
- **Phase 1** — the recap projection + the in-app panel (#2038), behind a flag.
- **Phase 2** — the daily digest email (#2039) on the existing delivery worker, behind a flag +
  notification pref.

**Monitoring:** panel impressions vs episode-finishes; "more like this" click-through; email
open/click and unsubscribe rate.

**Success Criteria:**
1. The panel appears on episode-finish and its "more like this" drives measurable next-listens.
2. The daily email sends to opted-in users, unfurls cleanly, and holds a low unsubscribe rate.

## Non-Goals

- Per-episode recap email (a Phase-2+ follow-up).
- New summarization / insight extraction — the recap consumes existing artifacts only.
- Anonymous recap (signed-in only, per RFC-120).
- Any audio in the recap or email (bridge-only).
