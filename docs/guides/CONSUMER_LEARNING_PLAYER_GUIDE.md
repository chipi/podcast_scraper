# Consumer Learning Player — feature guide

The consumer app (`web/learning-player/`, a Vue 3 PWA) is the end-user **Learning Player**: listen → understand →
**remember**. It is a thin client of the auth-gated `/api/app/*` API (RFC-098), entirely separate
from the operator GI/KG viewer. This guide is the single orientation for the "Remember" half —
**Capture** (PRD-040) and **Consolidation** (PRD-041) — and how they fit together.

Authoritative specs: PRD-040 / PRD-041 (product), RFC-098 / RFC-099 / RFC-101 (design),
`docs/api/PLATFORM_API.md` (every route), UXS-011 (interaction design).

## Principles

- **Grounded, extractive, no request-time LLM (D6).** Every saved/recalled item carries its episode
  slug + timestamp + verbatim quote; recall returns an assembled grounded set, never generated prose.
- **Per-user files, no DB.** All personal state is plain JSON under `<data_dir>/users/<id>/`
  (`highlights.json`, `notes.json`, `resurfacing.json`, … alongside `playback`, `queue`, `favorites`,
  `interests`, `listen_events.jsonl`). Shared corpus artifacts are never mutated.
- **Read-time projection.** The "personal knowledge corpus" is not a stored graph — it is computed on
  each request by scoping the shared corpus to the user's **heard∪captured** set (≥30% played ∪ any
  capture), unified by canonical identity (RFC-072).
- **Auth-gated, additive.** Signed-out users see the read-only app unchanged; every capture control
  and scope toggle appears only when signed in.

## Capture (PRD-040)

Turn a listening moment into a durable, grounded highlight. Three one-tap inline entry points:

- **Mark this moment** — the bookmark in the Player hero captures the current position as a `moment`.
- **Save a transcript line / phrase** — the per-line bookmark in the transcript saves the whole line,
  or — with text selected inside the line — that exact **phrase** (char offsets + quote).
- **Save an insight** — the bookmark on a Knowledge-panel insight keeps the grounded claim.

Highlights take an optional **colour** (fixed palette) and free-text **notes**. The timestamp is the
stable anchor: on re-scrape a highlight **re-anchors** by time and is flagged `drifted` if its text
moved — never silently dropped. Review them in **Library → Highlights** (grouped by episode, colour
filter, jump-to-moment) and **export to Markdown**.

API: `GET/POST/PATCH/DELETE /api/app/highlights`, `…/notes`, `GET /api/app/highlights/export.md`.

## Consolidation (PRD-041)

Turn captures + listening history into recall, connections, and resurfacing — all scoped to the
user's own experience.

- **Recall** — Search gains an **Everything / My corpus** toggle. "My corpus" (`scope=mine`) is
  grounded retrieval over the heard∪captured set ("what have I learned about X"), with honest
  zero-coverage. API: `GET /api/app/search?scope=mine`.
- **Connections** — person/topic entity cards gain an **All / My corpus** lens: the guest/topic
  across the episodes you've heard ("you also heard them in …").
  API: `GET /api/app/persons/{id}?scope=mine`, `…/topics/{id}?scope=mine`.
- **Resurfacing** — the Library **Revisit** inbox resurfaces past highlights on a spaced ladder
  (2d/1w/1mo/3mo, computed on read) with a reflection prompt + one-tap re-listen; Pause/Resume pacing.
  API: `GET /api/app/resurfacing`, `POST /api/app/resurfacing/{id}/surfaced`, `…/settings`.
- **Interest profile** — implicit interests derived from the user's corpus, beside explicit follows.
  API: `GET /api/app/interests/derived`.
- **Enrichment** — the consumer enrichment read surface projects the RFC-088 envelopes (co-occurrence,
  similarity, temporal velocity, contradiction) for the player + recall surfaces, read-only
  (ADR-104). API: `GET /api/app/episodes/{slug}/enrichment`, `GET /api/app/corpus/enrichment`.

## Testing

- **Unit / integration / e2e** mirror the project pyramid. The e2e runs against a **committed,
  deterministically-synthesised** corpus (`tests/fixtures/app-validation-corpus/v3`, built by
  `scripts/build_app_validation_corpus.py` — now carrying RFC-088 enrichment envelopes), served by
  the real API with **no mocks**; per-user state goes to a gitignored `APP_DATA_DIR`.
- CI must never call a real LLM; recall/connections/resurfacing are deterministic + extractive.
- **App CI jobs** (`.github/workflows/python-app.yml`, path-A gated on `web/learning-player/**`):
  `app-unit` (Vitest with coverage gate), `app-e2e` (Playwright + real API), `app-lighthouse`
  (LHCI PWA audit — hard-fails on missing/broken manifest, SW, maskable icon, apple-touch-icon,
  viewport, themed omnibox). See `web/learning-player/lighthouserc.json` for the gates.

## PWA shipping notes

The app is an installable PWA. The install / offline / update path has been hardened against
the specific traps in the shipping guide (`docs/guides/PWA_SHIPPING_GUIDE.md` if present, or
the source in `docs/wip/` before v2.8):

- **Icons.** `icon-192.png`, `icon-512.png`, `maskable-512.png` (Android crop safe-zone),
  `apple-touch-icon-180.png` for iOS. Fixtures live in `web/learning-player/public/`. Missing icons silently
  break Chrome's install prompt and produce a broken glyph on iOS home-screens — regression
  guarded by `web/learning-player/e2e/pwa.spec.ts`.
- **Runtime cache bounds.** Audio is never cached (bridge-never-rehost); artwork
  is `CacheFirst` with 500-entry × 30d expiration; shared GET `/api/app/*` is `SWR` with
  200-entry × 7d expiration; per-user `/me` / `/queue` / `/playback` / `/auth` are excluded.
  All caches are bounded — an unbounded cache eventually gets the whole SW evicted (iOS
  punishes this hardest).
- **Update path.** `registerType: 'prompt'` — a visible "New version available — Reload"
  toast replaces the silent-update stall. See `web/learning-player/src/composables/usePwaUpdate.ts` +
  `PwaUpdateToast.vue`. Update checks fire on tab refocus and every 15 min.
- **Debugging updates.** `window.__buildInfo = { sha, time }` and a
  `console.info('[app] Learning Player build=…')` line at boot give every client a
  visible identity so "the PWA isn't updating" reports carry evidence.
- **Subpath deploys.** Set `APP_BASE=/some-prefix/` at build time; manifest paths, SW scope,
  navigateFallback, and Vue Router base all pick it up. Verified with root and `/app/`.

## Related

- Product: `docs/prd/PRD-040-capture.md`, `docs/prd/PRD-041-consolidation.md`
- Design: `docs/rfc/RFC-098…`, `RFC-099…`, `RFC-101-personal-knowledge-corpus.md`
- API: `docs/api/PLATFORM_API.md` · Interaction: `docs/uxs/UXS-011-consumer-learning-app.md`
- Enrichment substrate: `docs/rfc/RFC-088…`, `docs/guides/ENRICHMENT_LAYER_GUIDE.md`
