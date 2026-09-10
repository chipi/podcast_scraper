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

---

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
  - **BLOCKED (found 2026-09-10, I.6).** The planned comparison is INVALID as specified:
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
