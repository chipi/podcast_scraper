# Player backend-blocked work — capability-area waves (2026-09-10)

Fresh consolidation of every item parked "because it needs backend", re-derived
from **user-facing effect** and checked against **actual backend readiness**
(probe 2026-09-10). Supersedes the scattered "Backend/data-dep" notes in
`PLAYER-UX-BACKLOG-2026-09-09.md`. Each area is sized to **open and close as one
themed branch/epic**.

Readiness legend: **READY** = data already returned by an API, UI-only surfacing ·
**SCHEMA** = data computed, needs a small schema/endpoint add · **PIPELINE** = data
not computed, needs ingest/derivation · **BOUNDARY** = collides with a product rule.

---

## Area A — Speaker roles & people identity  ·  readiness: READY  ·  risk: LOW

**Effect:** the app can show *who someone is* on a conversation — host vs guest vs
merely mentioned — on episode people lists, the Browse › People tab, and the player.
Today every person looks identical regardless of role.

**Items:** BE.4, BP.3, PL.2 (surface role); BP.2 (order People by role) — decision.

**Backend readiness:** the role is ALREADY returned.
- `AppEntity.role: str|None` on `/episodes/{slug}/entities` — `schemas.py:250`
- `AppPersonCard.role` + `AppPersonCard.shows[].role` on `/persons/{id}` — `schemas.py:342`

So this is **UI-only**. No endpoint work.

**Risk:** LOW. The only judgment call is BP.2: the People tab is velocity-sorted
(trending); re-ordering by role changes the tab's *meaning* and needs the shared
TrendingSparkChips to allow a role-primary sort. Surfacing the role badge (BE.4/BP.3/
PL.2) carries no risk and can ship without touching sort.

**Close when:** role badge renders on episode people, browse people, and player
speaker lists; unit tests assert host/guest/mentioned rendering; BP.2 decided
(surface-only vs re-sort).

---

## Area B — Show / topic / storyline enrichment surfaces  ·  READY + tiny SCHEMA  ·  risk: LOW

**Effect:** richer detail pages — "recurring guests" on a show, trend sparklines on
entities, storyline momentum — from signals the corpus already computes.

**Items:** SD.3 recurring-guests (#2006); F4.2 opening-act sparkline; BT.4 / F5.3
storyline velocity.

**Backend readiness:**
- recurring_guests: READY — `/podcasts/{feed_id}/signals` → `recurring_guests` (`schemas.py:2460`)
- entity trend-series: READY — `/trending?kind=…` → `AppTrendingEntity.series + velocity` (`schemas.py:493`)
- storyline velocity: PRESENT in the trending response already; `AppStoryline` itself
  lacks the field (`schemas.py:460`). Either read it from `/trending?kind=storyline`
  (UI-only) or add `velocity` to `AppStoryline` (one-field SCHEMA add).

**Risk:** LOW. No new computation; worst case one additive schema field.

**Close when:** show page shows recurring guests; entity cards show the sparkline;
storyline cards show momentum; tests cover the empty-signal fallback.

---

### Area B — OUTCOME (2026-09-10)

- **F4.2 DONE** — topic-card momentum now shows a sparkline (derived from `temporal_velocity`'s
  `monthly_counts` over `window_months`), via a new shared `TrendMomentum` component.
- **BT.4 DONE** — storyline momentum wired on BOTH the Home storylines rail (rail variant) and the
  StorylineView header (badge variant), joined by `thc:` id from `/trending?kind=storyline`;
  best-effort (no badge when a storyline is outside the trending set). The ID coupling I initially
  flagged was a non-issue — both paths key on `graph_compound_parent_id`.
- **SD.3 WON'T-DO** (operator) — the show page already renders `key_people`, a deliberate
  near-equivalent of recurring guests; `recurring_guests` is not surfaced separately.
- Shared `TrendMomentum` (badge = detail idiom, rail = trending-rail idiom) now backs topic + storyline
  momentum identically. Unit 1329 pass; vue-tsc + `make docs` green.

## Area C — Collections depth (Boards)  ·  PARTIAL + PIPELINE  ·  risk: MEDIUM

**Effect:** Boards feel like a real library — collection thumbnails, a big-thumbnail
grid, and notes/boards organized in folders.

**Items:** CO.6 per-collection artwork; CO.3 big-thumbnail grid (depends CO.6);
CO.4 notes-in-folders / nesting.

**Backend readiness:**
- Per-item artwork exists (`CollectionItem.artwork_url`, `schemas.py:1193`); **per-collection**
  artwork does NOT (`Collection` has only id/name/count, `schemas.py:1136`). CO.6 = derive a
  cover from member artwork → a compute + (ideally) a cached field.
- Nesting: MISSING — no parent/child model on `Collection`. CO.4 = a new data-model.

**Risk:** MEDIUM.
- **Perf:** the collections LIST returns no member artwork by design (the backlog flags the
  per-collection detail-fetch cost). Deriving a cover naively = N detail fetches on list render.
  Needs a stored/cached cover, computed on mutation — not a fan-out on read.
- **Data model:** CO.4 nesting evolves the collections store *we just changed for #1*. Pre-launch,
  so **no migration needed** (forward-only, no users) — but it is a real schema evolution; design
  the shape (flat parent_id vs tree) before building.

**Close when:** collections carry a derived cover (computed on add/remove, not on read);
grid view renders covers; nesting model shipped with a decided shape + tests.

---

### Area C — OUTCOME (2026-09-10)

- **CO.6 DONE** — collections now carry a `cover_url`, derived from the first episode/highlight
  member's artwork (via the same `resolve_slug`+`row_to_summary` path favorites use) and **cached on
  the row, recomputed on add/remove** — the list stays one cheap read (no per-render fan-out). The
  item resolver still returns other kinds as-is; topics/people/search/link contribute no cover.
- **CO.3 DONE** — a list⇄grid toggle on Boards (matching the Catalog/Browse idiom); grid tiles show
  the cover (placeholder when none). Grid can't expand a tile in place, so tapping a tile opens the
  board in the familiar list accordion (reuses all open/play logic).
- **CO.4 DEFERRED** (operator) — nesting/folders is feature-sized (folder CRUD + move UI + hierarchy
  render) and wants its own UX design pass; tracked as a separate arc, not built here.
- Backend 40 pass; unit 1330; vue-tsc green.

## Area D — Catalog metadata: podcast category  ·  PIPELINE  ·  risk: MEDIUM

**Effect:** Browse shows by category/genre.

**Items:** BS.1.

**Backend readiness:** MISSING everywhere — `AppPodcastItem` has no category (`schemas.py:1356`)
and RSS ingest does not capture `<category>` / iTunes genre. Needs ingest + a corpus backfill.

**Risk:** MEDIUM. Touches ingestion; feed categories are inconsistent/multi-valued across
publishers (data-quality risk → a category picker can look sparse/wrong). Scope the taxonomy
(raw feed categories vs a normalized set) before building the UI.

**Close when:** ingest captures category, corpus backfilled, `AppPodcastItem.category`
surfaced, Browse filter added; test covers feeds with missing/odd categories.

---

### Area D — OUTCOME (2026-09-10)

Built the full pipeline (operator: "build the full pipeline now" + generic backfill):
- **D1** — `extract_feed_category` parses `<itunes:category>`/`<category>`, threaded through the feed
  block (both the metadata and summarization stage paths) into the serialized metadata.
- **D2** — category flows catalog (`CatalogEpisodeRow` + `aggregate_feeds`) → `AppPodcastItem` →
  `/podcasts`; end-to-end route test asserts present-vs-null.
- **D3** — a category facet on Browse › Shows (renders only when the catalogue carries any).
- **D4** — a **generic** `refresh-feed-metadata` CLI backfill (operator: "generalize it") — re-derive
  the whole feed block from live RSS and patch existing metadata files, no transcription; merge-only,
  idempotent, injectable fetch. Extend `_derive_feed_updates` for any future feed field.

Additive/optional throughout; a corpus with no categories is unchanged. The prod backfill is an
operator-run action post-deploy (`refresh-feed-metadata`), not run against prod here.

## Area E — Profile & account  ·  PARTIAL (new endpoint + auth)  ·  risk: MEDIUM-HIGH

**Effect:** a real profile — display username, OAuth profile photo, an editable bio.

**Items:** Profile batch backend — `/me` fields (username, image, bio) + a profile-update endpoint.

**Backend readiness:** PARTIAL. `/me` returns name/email/role only (`app_auth.py:317`); no image,
bio, or username; no update endpoint; OAuth profile image is not captured at login.

**Risk:** MEDIUM-HIGH — the riskiest "normal" area because it touches **auth and a new write path**:
- OAuth callback change to capture the provider image (auth flow — highest-blast-radius code).
- A new **write** endpoint (profile update): input validation, size limits, rate/abuse.
- Editable **bio** = user-generated content → XSS/moderation surface on any place it renders.

**Close when:** `/me` returns image + username + bio; a validated update endpoint exists;
OAuth image captured at login; bio render is escaped; tests cover validation + the escape.

### Area E — REFINED SCOPE (operator, 2026-09-10)

- **Profile image:** capture the OAuth provider avatar at login (auth-callback change) → `/me`, AND a
  **separate, narrow, dedicated upload endpoint** for a user-supplied image — explicitly NOT folded
  into a general `/me` update. Upload is a file-handling + validation surface (size/type/storage).
- **Username = an IMMUTABLE handle** (like X / Instagram `@handle`): set once, **cannot be changed**.
  Format like a handle (lowercase alnum + underscore, length-bounded). "A future handle" — forward
  identity concept.
- **NO editable bio** (dropped from scope).
- **No general `PATCH /me`** — the only write is the narrow avatar-upload endpoint.

**Handle birth (RESOLVED, operator):** the username is **auto-derived at account creation** from the
OAuth identity (email local-part / name → sanitized to a handle, deduped for uniqueness) — it is
**not chosen during registration** and there is no claim/change flow. Immutable thereafter.

**Open sub-decision (upload):**
- **Upload storage + limits** — where the uploaded image lives (per-user dir?), max size, allowed
  content types, and whether it replaces/overrides the OAuth image on `/me`. Sensible defaults:
  per-user data dir, ≤2 MB, `image/{png,jpeg,webp}`, user upload overrides the OAuth avatar.

**Risk note:** highest-risk area (auth callback + file upload + a handle namespace). Recommended as
its own focused arc with fresh context rather than the tail of a long multi-area session.

---

## Area F — Audio delivery (normalization + quality)  ·  BOUNDARY  ·  risk: HIGH — needs a product ruling

**Effect:** loudness normalization (#2029) and a download/stream quality control (ST.3).

**Items:** #2029 Web-Audio normalization; ST.3 media-quality.

**Backend readiness:** audio is served **bridge/direct** — `/episodes/{slug}/audio-source` returns
the origin enclosure URL as-is; `strategy` is always `"direct"`, and the "no-store proxy stays
deferred" (`app_episodes.py:404`). No quality variants exist (single origin URL, `schemas.py:32`).

**⚠ RISK — HIGH, product boundary (flagging explicitly per your ask):**
- Web-Audio normalization needs the audio to be **same-origin** (a `MediaElementSource` silences
  cross-origin audio without CORS). The only backend ways there are **(a) a same-origin audio proxy**
  (stream origin bytes through our server) or **(b) transcoded variants** for ST.3. **Both rehost
  audio bytes**, which directly contradicts the standing product rule *"AUDIO is bridge-only / never
  rehost"* (memory `project_transcript_vs_audio_hosting`; the endpoint comment keeps the proxy
  deferred for this reason).
- So this area is **not an engineering decision — it's a product/legal one.** Recommend: **do NOT
  build the proxy/transcode path without an explicit ruling.** Client-only alternatives (e.g. a
  WebAudio `GainNode` without `MediaElementSource`, or a perceived-loudness gain applied only when
  the origin already sends CORS headers) are the only non-boundary-crossing options and are limited.

**Close when:** a product ruling is recorded (rehost allowed? scope?). Until then this area stays
**parked with a reason**, not silently dropped.

---

## Proposed open/close order

| Order | Area | Readiness | Risk | Why here |
|------|------|-----------|------|----------|
| 1 | **A — Roles** | READY | LOW | Pure UI on data already returned; high visible payoff. |
| 2 | **B — Enrichment** | READY | LOW | Same — surfaces computed signals; one optional schema field. |
| 3 | **C — Collections depth** | PARTIAL | MED | Needs a cover-derivation design + a nesting shape; no users = no migration. |
| 4 | **D — Category** | PIPELINE | MED | Ingest + backfill; taxonomy decision first. |
| 5 | **E — Profile/account** | PARTIAL | MED-HIGH | Auth + new write path; sequence after the low-risk wins. |
| 6 | **F — Audio** | BOUNDARY | HIGH | **Needs a product ruling before any code** — likely parked. |

A and B are genuinely **UI-only** (the probe confirmed the data is already on the wire) — they were
mis-filed as "backend" in the old backlog. They're the fastest, safest wins and should open first.

## Cross-cutting risk notes
- **No migrations anywhere** — pre-launch, no users/data to preserve (forward-only). Schema changes
  are free to make destructively.
- **The one sharp edge is Area F** (rehosting audio) — a product boundary, not an eng call.
- **Area E** is the only area touching **auth**; treat the OAuth-callback change with extra care and
  keep the profile-update endpoint strictly validated + escaped.
- Security-audit (`pip_audit`) currently fails `ci-ui-full` on dependency CVE advisories + un-auditable
  local packages — pre-existing, unrelated to this work, but it blocks the UI tests in that target
  (run `make test-app`/`test-app-e2e` directly until it's addressed separately).
