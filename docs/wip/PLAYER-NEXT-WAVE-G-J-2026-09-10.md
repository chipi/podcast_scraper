# Player next wave — G–J scope + readiness (2026-09-10)

Four operator-confirmed areas. Readiness from a codebase scout (2026-09-10). This is a plan +
decision doc — nothing built yet.

Legend: **READY** = infra/data exists, mostly surfacing · **SCHEMA** = additive field/endpoint ·
**PIPELINE/EXTERNAL** = net-new data path · **BOUNDARY** = breaks a standing rule / needs a ruling.

---

## G — Person enricher  ·  mixed  ·  risk: HIGH (external data)

**Exists to build on:**
- Person card served by `/persons/{id}` (`AppPersonCard`), role badge already shipped (Area A).

**"Key voices" (was mis-scoped as "cluster people" — operator reframed 2026-09-10 to KEY VOICES,
a prominence/ranking surface, NOT community detection).** Ranking data already computed:
- **per-topic** — `mcp/tools/cil.py:topic_perspective_leaders` (comment: "the closest thing to graph
  centrality the corpus surfaces"). Surfaces the key voices ON a topic → lands on the **topic page**,
  the direct people-analog of the topic storylines/trending already there.
- **per-user** — the one net-new bit: intersect a ranking with the user's followed shows
  (`library.json` by feed_id) / listened episodes → "your key voices" on Home.
- **DECIDED (operator 2026-09-10):** build **per-topic + per-user**; NO corpus-wide rail
  (`top_people` exists but is not in scope).
- Pure surfacing for per-topic; per-user needs a small ranking-∩-library query. Zero external data —
  entirely separate from the Wikipedia boundary below.

**Net-new:**
- **Bio + photo** — `AppPersonCard` is deliberately lean (no bio, no image, `schemas.py:313-363`).
  A **Wikipedia enricher** is the real new work.
- Surfaces: bio/photo on the person page; a small person avatar in front of Trending names; a person
  photo as fallback "artwork" where an episode lacks its own.

**⚠ BOUNDARY (the sharp one):** every enricher today is **deterministic, corpus-internal, zero
external API calls** (`enrichment/enrichers/`, executor in-process) — and CI is airgapped (memory
`feedback_no_llm_in_ci`). A web fetch breaks that. It MUST be an **offline enrichment-time**
step that fetches + caches per person, **fixtured in tests** (never a request-time call, never a
live call in CI).

**DECIDED (operator 2026-09-10): tier = a NEW `EnricherTier.WEB`** (option B, not reuse-ML) — a
dedicated external-fetch tier. Ripples into the resilience policy, admission/profiles, and the UI
tier surface; the CI-airgap holds by keeping WEB out of the airgapped CI profile (verify the
profile membership before building so the airgap genuinely excludes it). Fixtured in tests.

**DECIDED (operator 2026-09-10): build it as a GENERAL web-enricher, with Wikipedia as the first
provider/specialization** — so we can add more web-enrich providers later (same generalize-first
shape as the D backfill). Concretely: a base `WebEnricher` (fetch → parse → cache → emit, with the
offline/fixtured contract enforced once in the base) + a pluggable provider interface; a
`WikipediaProvider` is provider #1. The airgap rule is NOT relaxed by generalizing — the base owns
the "cached, offline, fixtured, never-in-CI" contract for every provider.

**DECIDED — image hosting (operator 2026-09-10): HOST OURSELVES + attribute.** Download at enrichment
time, store in our assets, serve from our own domain, carry CC-BY-SA attribution (author / license /
source_url rendered as a small credit). Robust + cached + no user-IP leak; cost = storage + an
attribution field we must render.

**SHIPPED (2026-09-10) — photo hosting.** Enricher image step downloads + validates the photo
EXACTLY like the avatar (content-type allow-list + magic-byte sniff + 2 MB cap), stores it at
`enrichments/person_images/<slug>.<ext>` + an `{ext,license,artist}` sidecar (cache-on-skip). The
photo's OWN license/author come from Wikipedia `imageinfo` (extmetadata) — **no license → not
hosted** (never store what we can't attribute). Served by `GET /api/app/persons/{id}/photo`
(auth-gated, path-sanitized, nosniff); the person card exposes **only our served route** (never the
raw external URL — no IP leak) + the photo license as a credit. Mock Wikimedia image + imageinfo on
the e2e server; full-cycle e2e asserts stored+served+attributed. **Follow-up:** photos on the
key-voices / Top-voices chips (those endpoints must carry the image url).

**DECIDED — execution model (operator 2026-09-10): a standard enricher, run BOTH ways, exactly like
the others.** It plugs into the existing enrichment framework so it runs (1) **in-pipeline** per new
episode and (2) **corpus-wide batch re-enrich** (the same path other enrichers use today — `reenrich`
/ the batch executor). No bespoke run path. The base `WebEnricher` owns the airgap gate so the
external fetch happens at enrichment time only and is **skipped/fixtured in CI** — it never fires a
live web call in the airgapped suite, even though it lives in the same executor the others do.

---

## H — Recommendations / digest email  ·  READY data + PIPELINE delivery  ·  risk: MEDIUM

**Exists:** comms store (`app_comms_store.py`: `digest` + `push`, weekly cadence), **digest section
builders** (`app_digest_sections.py` — reusable for email + home), trending/storylines data
(`AppTrendingResponse`). "Your Week" is the one existing email.

**Net-new / unknown:** the actual **email template + branding/logo + send trigger** were not found in
`src/` — likely infra-side or a template not yet located. **Must find how "Your Week" actually renders
+ sends** before scoping precisely (do not assume out-of-repo — verify). Then: a **second email type**
(recommendations, trending/rising/storylines unpacked), branding/logo, optional **monthly** cadence,
and backporting any email-only section to Home.

**SHIPPED (2026-09-10) — verified: rendering/sending/branding is INFRA (the #1412 delivery worker),
NOT this repo.** The app only PRODUCES `DeliveryEnvelope`s; the worker renders + sends. So H's
app-side is a second envelope type:
- `app_digest_recommendations`: a **monthly** discovery digest (rising/trending + new-in-interests,
  reusing the shipped `app_digest_sections` builders — graph-carrying, airgap-clean), template
  `recommendations-digest.v1`, gated on the `digest` × `email` cell + schedule `paused` +
  email-verified, per-month idempotent id.
- Wired into the SAME hourly digest cron (`scheduler._spawn` JOB_KIND_DIGEST) via
  `enqueue_due_recommendations`; a monthly-slot gate (1st of month at the user's hour) keeps it monthly.
- `delivery-envelope.schema.json`: +`recommendations-digest.v1` template, +`new_in_interests`
  section kind, +`monthly` cadence, payload→digestPayload; golden + contract test updated.
- **In-app equivalent already exists** (Home Rising/Trending/Storylines tabs) — no Home backport
  needed. **Branding/logo + the email HTML template are the infra worker's** (cross-repo follow-up,
  mirror the new template there). Monthly-vs-weekly is a fixed cadence for this type (not a user knob).

---

## INFRA FOLLOW-UPS (cross-repo — the #1412 delivery worker MUST mirror before shipping)

The delivery seam (`docs/api/delivery-envelope.schema.json`) changed this wave — the infra worker
reads the same schema, so it must be updated **before the first recommendations digest fires** or
those envelopes are undeliverable (worker won't match the template):

- **`recommendations-digest.v1`** template (payload = the digest `sections` shape) → add a Jinja
  template + branding/logo (email HTML lives infra-side).
- **`monthly`** consent cadence value + the optional **`type`** field (`digest`/`new_episodes`/
  `product`) on the envelope — additive; worker may ignore `type` but must accept it.
- **`new_in_interests`** section kind in the digest payload.
- Also I.6: the player deploy must set **`APP_PLAYER_VERSION`** (= the SPA build's `__APP_VERSION__`)
  for the native update prompt, and the person_web images need the corpus `enrichments/person_images/`
  to survive deploys (served by `GET /api/app/persons/{id}/photo`).

## DEFERRED (review findings 2026-09-10 — tracked, not blocking the push)

- **Key-voices scan order** — `sorted(heard)[:200]` is an arbitrary (alphabetical) sample, matching
  the existing `trending_items` pattern; a recency-ranked scan needs playback timestamps → follow-up.
- **`image_url` colon** — `/api/app/persons/person:jane-doe/photo` works (FastAPI captures the
  segment; e2e green) but is unencoded; encode defensively if a downstream encoder ever bites.
- **`image_artist`** persisted + typed but not yet rendered (its value may carry HTML → needs
  sanitization before display).
- **no-license image re-fetch** — a person whose photo can't be licensed re-hits `imageinfo` each
  run (skips aren't cached); cache a "skipped" sidecar once transient-vs-permanent is distinguished.
- **people-images-everywhere** — key-voices / Top-voices chips + show/episode rosters (see
  `PLAYER-PEOPLE-IMAGES-2026-09-10.md`); + **crop-on-upload** for the user avatar.

## I — Notifications framework  ·  READY + SCHEMA  ·  risk: LOW-MEDIUM

**Exists (strong):** WebPush fully implemented — `app_push_store.py` + `usePushSubscription.ts`
(subscribe/unsubscribe, VAPID). Comms store has **per-CHANNEL** toggles (digest/push). Profile
Notifications section exists (`ProfileView.vue:335-406`). Version endpoint exists (`health.py`
`code_version`) + client knows `__APP_VERSION__`.

**Net-new:**
- **Per-TYPE × per-CHANNEL** preference matrix (today it's one toggle per channel) — a `comms.json`
  schema evolution + a Profile UI expansion.
- **"Update available"**: client compares its `__APP_VERSION__` to `/api/health` `code_version` → a
  prompt. Web = reload; **native (Capacitor) = App Store link** (different action — a decision).
  - **RESOLVED (2026-09-10, I.6 shipped).** Fixed per operator: the server now publishes a
    dedicated `player_version` in `/api/health` (from `APP_PLAYER_VERSION`, the released player-app
    version on the SAME scale as `__APP_VERSION__`, distinct from backend `code_version`); the
    client `useAppUpdate` compares like-to-like, native-only (`AppUpdateBanner`); web stays on the
    service worker. Store URL empty pre-launch → informational banner, no dead link. Original
    blocker below, for the record.
  - **(original blocker, 2026-09-10, I.6).** The planned comparison is INVALID as specified:
    `__APP_VERSION__` is the learning-player package version (**1.0.0**); `code_version` is the
    backend `podcast_scraper.__version__` (**2.7.0.dev0**). They are versioned on **independent
    scales**, so a direct compare makes the server permanently "ahead" → a false, never-clearing
    "update available". AND the native action has no target: the app is TestFlight-only pre-launch,
    there is **no published App Store URL** to link to.
  - **Web is already covered** — `PwaUpdateToast` + `usePwaUpdate` (service-worker, content-hash
    based) handle the web reload prompt correctly today. No new work needed there.
  - **To do I.6 correctly needs an operator decision** (see the reprint at the end of this doc):
    (a) `/api/health` publishes a *client*-version signal on the SAME scale as `__APP_VERSION__`
    (e.g. `min_player_version` / `current_player_version`, set by the player deploy) so the client
    compares like-to-like; and (b) a store URL / update channel to send native users to (post-launch,
    or TestFlight in the interim). Until both exist, a native prompt would be a false alarm pointing
    at a dead link — not built.
- An **in-app** notification surface (vs OS push).

**FROZEN CONTRACT (operator 2026-09-10) — the type×channel matrix everything routes through.**
Forward-only rewrite of `comms.json` (no migration; pre-launch, no-backcompat rule):
```jsonc
{
  "types": {
    "digest":       { "email": false, "push": false, "in_app": true },  // Your Week + recs (H)
    "new_episodes": { "email": false, "push": false, "in_app": true },  // followed-show alerts (J)
    "product":      { "email": false, "push": false, "in_app": true }   // app updates / announce (I)
  },
  "digest_schedule": { "cadence": "weekly", "day_of_week": 6, "hour": 13, "paused": false },
  "unsubscribe_ref": "…"
}
```
- **3 types** (digest / new_episodes / product); adding a type = one registry entry.
- **3 channels** — `email`, `push` (OS-level, app-closed, VAPID, opt-in default OFF),
  `in_app` (in-app inbox, app-open, no OS grant, default ON). Push ≠ in-app: push reaches you away,
  in-app is what's waiting when you return. **All three confirmed IN.**
- **in-app inbox is the one net-new store** — a per-user notifications list + a bell UI.
- Scheduling (cadence/day/hour/paused) sits OUTSIDE the matrix in `digest_schedule` (email-delivery
  specifics, not per-channel). Old `push` nudges fold into `digest.push`.

---

## J — New-episode alerts  ·  SCHEMA + query  ·  risk: MEDIUM

**Exists:** followed shows (`library.json` by feed_id); a **corpus revision delta-log**
(`app_corpus_revision.py`) — sequence-numbered add/remove episode events per feed. That's the seed.

**Net-new:** a **per-user last-seen sequence** + a "new episodes in my followed feeds since T" query,
delivered via the Area-I channels/prefs. Ingest is fire-and-forget; the revision log lets us compute
the delta on-demand at send/open time (no ingest hook needed). Ordering care: advance last-seen only
AFTER capturing the delta.

**SHIPPED (2026-09-10) — corrected premise.** The readiness scout mischaracterised
`app_corpus_revision.py` as "add/remove episode events per FEED"; it is per-USER membership
(experienced/saved), not new-episodes-in-followed-feeds. The correct delta already existed:
`app_digest_sections.new_in_follows_items` (recent UNHEARD episodes in followed feeds). J is built on
that:
- `app_new_episode_alerts.sweep_for_user` turns the delta into `new_episodes` in-app notifications,
  **deduped per episode** (`newep:<slug>`) — so no per-user last-seen store is needed; the
  notification store's `dedupe_key` is the idempotency key.
- **Computed on-demand** in `GET /api/app/notifications` (best-effort; a sweep failure never fails
  the read) — alerts appear when the bell is opened.
- **Channels:** in-app is delivered here (gated on the `new_episodes` × `in_app` cell). **Email**
  already reaches the user via the weekly digest's *new-in-follows* section. **Standalone push** for
  this type needs its own delivery envelope/template + worker render → deferred follow-up (not built
  speculatively).

---

## Dependencies + proposed order

- **I (notifications framework) FIRST** — J's alerts and H's second email both deliver *through* I's
  per-type×channel prefs. Building I first gives J and H their opt-in/routing home.
- **J** next — small, high-value, rides I + the revision log.
- **G** — two independent halves: (1) **key voices** per-topic + per-user (zero external data, pure
  surfacing + one library query — buildable anytime) and (2) **bio/photo enricher** (gated on the
  external-data ruling). The key-voices half is NOT blocked by the Wikipedia decision.
- **H** — second email; gated on locating the Your-Week render/send path.

Order: **I → J → G → H** (infra-first, then the two data-dependent ones).

## Decisions needed before building
1. **G external-data ruling** (the boundary): offline cached enricher + fixtures (confirm), and
   **host person images ourselves (download+store+attribute) vs hotlink Wikipedia**.
2. **Order/priority** — accept I→J→G→H or reprioritize.
3. Deferred-until-in-area (sensible defaults, will confirm per area): I's update-prompt native
   behavior; H weekly-only vs +monthly; G key-voices surfacing depth (how many, expandable?).
