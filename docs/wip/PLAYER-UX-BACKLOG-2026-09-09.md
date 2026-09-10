# Player UX backlog — arcs, slices, order (2026-09-09)

Source: operator note-dump (2026-09-09) on the consumer **learning-player**
(`web/learning-player/`, Vue 3 + Pinia + vue-router + Capacitor). Faithful capture,
de-typo'd, organized into **foundation arcs** (define-once, applied app-wide per UXS-014)
and **per-screen arcs**. Nothing here is started yet. Ordering + open decisions at the end.

Existing patterns this backlog reuses:
- **Follow pill** — text pill `+ Follow` / `✓ Following`, `data-testid="follow-show"`
  (`PodcastView.vue`, `ShowTile.vue`). The operator repeatedly calls this the good pattern.
- **FavoriteButton** — shared heart `♥/♡` round icon button, already the "save anything"
  affordance (`FavoriteButton.vue`, UXS-014), routes signed-out taps to sign-in (#1590).
- **Detail-page "topic" pattern** — the show-detail topic rendering w/ integrated trending
  indicator, which the operator wants as the default detail template.

Backend/data dependencies already tracked (UI renders what exists, does not fix identity):
- Host/guest/mentioned roles: **#1863** (guests default to 'mentioned'), **#1897** (42%
  unattributed), **#1286** (guest voices anonymous), **EPIC-HOST-IDENTIFICATION**.
- `recurring_guests` computed but never rendered: **#2006** (→ SD.3 "number after key people").
- Playback stops on navigation / global audio host + persistent mini-player: **#1587**
  (already an open ui/ux issue; relevant to the Player arc).

---

## FOUNDATION ARCS (define once, apply everywhere)

### F1 — Offline & degraded-mode + loading stability (NFR)
- **F1.1 BUG (hotfix) — FIXED (working tree, uncommitted 2026-09-09).** Cold-start while
  offline showed a **blank screen after splash**; tapping a bottom-nav item recovered it.
  **Root cause (evidence, not theory):** the router guard (`router/index.ts:143`) awaits
  `auth.ensureLoaded()` → `refresh()` → `getMe()`, and `apiFetch` (`api.ts:118`) has **no
  request timeout**, so offline the call hangs until the OS connection timeout. The initial
  navigation stayed pending; `booting` flips false at 1800ms (`App.vue:382`) → splash lifts
  over an empty `<RouterView>` (shell + BottomNav painted). A nav tap re-ran the guard after
  `onMounted`'s `hydrateFromDevice` had set `loaded`, so it resolved. **Fix:** `ensureLoaded`
  now hydrates from the instant device snapshot first and revalidates in the background —
  the first paint never blocks on the network. Repro test in `auth.test.ts` (RED = 2s
  timeout → GREEN = 6ms).
  - **Follow-up F1.1a (surfaced, not done):** `apiFetch`/`getMe` has no timeout — a
    genuinely first, never-online launch (no device snapshot) still hangs the guard on
    `refresh()`. Recommend a scoped timeout on `getMe` (small, clearly correct). A *global*
    fetch timeout is a broader change (could abort slow legit requests) — needs a decision.
    Belongs to F1.2/F1.4.
- **F1.2** App-level online/offline awareness → an explicit **offline mode** that changes
  *what* and *how* each page renders (show / episode / topic / storyline / person).
  Define the degraded-render contract once.
- **F1.3** **Layout stability**: reserve areas / skeletons so data loading causes **no
  layout jumps**, on every page.
- **F1.4** Graceful "can't load this" states per page — never blank, never crash.

### F2 — Favorite standardization ("heart = favorite", one affordance, synced)
**Save ≠ Follow — they are two different actions (operator clarification 2026-09-09):**
- **Save to library = "favorite", always the heart icon `♥/♡`, everywhere.** No pill
  variant. This is the ONE save affordance.
- **Follow (a show) = the existing follow pill**, kept **consistent everywhere it works
  today**. Not a save; do not merge the two, do not replace the episode heart with a pill.

Slices:
- **F2.1** One name everywhere: **"favorite"** (heart). Audit + unify current
  heart/like/save/highlight-save wording to "favorite"; all land in Library / Saved.
- **F2.2** A favorite **syncs across** episode / insight / show / topic / person /
  storyline. Today: episodes + insights → library. Extend show "save as heart" and sync
  the state across every surface.
- **F2.3** Keep the **heart icon** for favorite in every context (resolved — no pill).
- **F2.4** Follow-pill **consistency audit**: ensure Follow renders/behaves identically
  wherever a show can be followed.

### F3 — Standard item action set ("actions everywhere")
- **F3.1** Define the **minimum action set** for a saveable item: favorite, download,
  add-to-queue; **add-to-collection** (detail surfaces); **add-note**.
- **F3.2** One shared `ItemActions`/`EpisodeActions` component placed on **every** episode
  surface. Known gaps to close:
  - MomentumRail — missing download
  - What's New — missing favorite + download
  - Recommended-for-you — missing download + favorite
- **F3.3** **Add-to-queue** today only on the player's current episode → expose on episode
  surfaces generally.
- **F3.4** **Add-to-collection** placement policy: details + player + browse — evaluate
  which surfaces get it.
- **F3.5** RESOLVED: do **not** replace the episode heart with a follow-style pill. Save
  (heart) and Follow (pill) are different actions — see F2.

### F4 — Shared detail-page template (Topic / Person / Storyline aligned to the topic page)
- **F4.1** Header row: **title + primary actions (follow/favorite/note) on the RIGHT**, in
  the same row as the title. **Back button at top.**
- **F4.2** Opening "act": a **detailed sparkline / trend under the title**.
- **F4.3** **Move search further down** the page (repeated on topic / person / storyline).
- **F4.4** Clarify the **"All / My listening"** toggle — label + behavior are unclear today
  (operator themselves unsure what it does). Applies to topic + person.
- **F4.5** Storyline detail opens as half-screen sheet → should open **more at top**, same
  look/feel as the topic page, **back at top**.
- **F4.6** Person detail on the same template.

### F5 — Trending indicator + color-coding as the default
- **F5.1** The show-detail topic rendering w/ **integrated trending indicator** → make it
  the **default** wherever topics/storylines/people render.
- **F5.2** Define + **document color-coding semantics** — currently everything is green
  (all trends positive?). Add a legend / meaning.
- **F5.3** **Trending for storylines** — can we compute/show it?

---

## PER-SCREEN ARCS

### H — Home
- **H.1** (= F1.1 offline blank-screen bug — cross-ref.)
- **H.2** "Your Week" cards are **rectangular** — why; make card shape consistent.
- **H.3** Move **Search lower**, between segments, to declutter monotony and free top space
  (put something under "Continue listening").
- **H.4** Push more **up** besides "For You": a Trending Topics take / **"Rising now"** /
  Storylines (if trending available for them).
- **H.5** **"Jump back in"** section when there are multiple active listens; show all.

### BE — Browse › Episodes
- **BE.1 BUG:** all episodes show "**8 key points**" which is impossible → fix data/display
  (likely a hard-coded/placeholder count).
- **BE.2** **Read more / less** per episode → expand the row to show the full summary.
- **BE.3** Replace the "**key points**" pill with an "**insights**" pill that opens the
  player-style insights popup.
- **BE.4** Under artwork, make clear **who is guest / who is host** (data dep: #1863).
- **BE.5** **View toggle → grid** (apply the shows-grid style).
- **BE.6** **Hide played** episodes.
- **BE.7** **Filter** by downloaded / played / all; **sort** by published date.

### BS — Browse › Shows
- **BS.1** **Browse by category** view (build it).
- **BS.2** **View toggle → list** (apply the episodes-list style).

### BT — Browse › Topics
- **BT.1** Show **top 10**, then "show more" in **increments of 10**.
- **BT.2** Storylines: top 10 + show more/less.
- **BT.3** Make the **storylines list look like the topics list**.
- **BT.4** **Trending of storylines** (ties F5.3).

### BP — Browse › People
- **BP.1** Top 10, then +10 increments.
- **BP.2** Show **guests + mentioned first, then hosts**.
- **BP.3** Guest-vs-mentioned still unclear — data dep, tracked (#1863/#1897/#1286/#2006).
  UI: render roles clearly *when the pipeline provides them*.

### SD — Show detail
- **SD.1** **Save as heart**, synced everywhere (ties F2).
- **SD.2** No show-more/less on topics — **show all always**.
- **SD.3** "Key people": are they only guests? **What is the number after?** (→ #2006
  `recurring_guests`.)
- **SD.4** Move **activity up**; make it more frequent-looking and **colorful**.
- **SD.5** "**What's this show about**" collapsible + nicer.
- **SD.6** Compute **update cadence** → 1-line in details ("updates ~weekly").
- **SD.7** Better **metadata overview**: length, # published, rated.
- **SD.8** **Highlight the latest episode** at top.
- **SD.9** **Hide played** episodes.
- **SD.10** Split into **tabs**: Episodes / About / More like this.
- **SD.11** Use the **same episode-list component** as browse › episodes.

### TD — Topic detail (beyond F4)
- **TD.1** Works like insights: **start from the start**, header row (ties F4).
- **TD.2** Say more about **topic dynamics** (what else can we surface?).
- **TD.3** Clarify: the section **below the topic list** — is it for storyline or topic?
  What is "topic only" here?
- **TD.4** **Similar topics** on top of storyline (topic-specific).
- **TD.5** Episode artwork **top-aligned** in the "discussed episodes" list.
- **TD.6** Bring the **strongest shows** about this topic + recommend them.
- **TD.7** **Add-note** (ties NT).

### SL — Storyline detail (beyond F4)
- **SL.1** Ordered topics like **episode insights in the player**.
- **SL.2** **Top episodes per storyline** + **related people** (like the topic-page bottom).
- **SL.3** **Add-note.**

### PD — Person detail (beyond F4)
- **PD.1** "**Host of <show>**" under the name + **show artwork**.
- **PD.2** More detailed sparkline at top as the opening act (= F4.2).
- **PD.3** Move search down (= F4.3); clarify all/my-listening (= F4.4).
- **PD.4** **Add-note.**

### PL — Player
- **PL.1** Better **metadata overview**: length, # published, rated.
- **PL.2** Clear **who is guest / who is host** (data dep #1863).
- **PL.3** Small buttons **too tight**; right side sits closer to the edge than the left —
  fix spacing symmetry.
- **PL.4** Maybe **3 button sizes** — the outer two-per-side somewhat smaller.
- **PL.5** **Remove the number** from the insights pill.
- **PL.6** Add **"mark as played"** manual action.
- (Relates to #1587 — global audio host / persistent mini-player.)

### IN — Insights
- **IN.1** Key-point **bullets look bad** → better list style.
- **IN.2** Storyline / similar are **underrepresented** in topics.
- **IN.3** Small **per-insight-type filter** on top.
- **IN.4** Clarify the **time label** next to an insight (what is it?).

### NT — Notes (new capability)
- **NT.1** **Add-note** on episodes + everything saveable (highlights already have it).
- **NT.2** **Timestamp** note creation; show the timestamp next to the note.
- **NT.3** **Audio dictation** for a note (mic icon) — native lift.
- **NT.4** **Notes section** in the Collections tab, next to collections.

### CO — Collections
- **CO.1** Replace the **ugly icon** with a follow-style **pill + text**.
- **CO.2** **Last-modified** on a collection.
- **CO.3** Collections as **big thumbnails** (like shows/episodes).
- **CO.4** Organize **notes into collections + folders**.
- **CO.5** **Search + sort** collections / notes.
- **CO.6** **Synthesize collection artwork** from member items' artwork.
- **CO.7** **Rename the tab** (Collections + Notes → something else).

### SR — Search
- **SR.1** **Include notes** in results.
- **SR.2** **View toggle** grid / list (apply show/episode styles).
- **SR.3** **Recent searches** under the search box.
- **SR.4** Fix the **empty-state empty space**.

### ST — Settings
- **ST.1** Group into **sections**.
- **ST.2** Move **Connected agents** from Profile → Settings (also).
- **ST.3** **Download & streaming media quality** settings.

---

## PROGRESS + RESOLVED DECISIONS (2026-09-09, branch `feat/player-ux-overhaul`, unpushed)

**Shipped (each committed + green; `ci-ui-full` still owed before push):**

- F1.1 offline blank-screen bug + hardening (bearer race pre-mount, scoped `getMe` timeout).
- F3 shared `EpisodeActions` row (favourite/download/queue) → adopted on `EpisodeTile`, Home
  What's-new/Recommended, Search. Download self-hides on web everywhere.
- RFC-121 **phase 1** (insight is not a favorite; 422 write-ban; insights bucket deleted) +
  **phase 2** (insight save = the shared `.lp-fav` heart via the capture/highlights path).
- `OverflowMenu` — the one canonical `⋯` menu (teleported, a11y, tested).
- F4.1 entity card: title + actions on one row (topic/person modal + panel + pages).
- TD.5 discussed-episode artwork top-aligned. PL.5 insights pill drops its count.
- SD.2 show description: collapsed preview 5→8 lines, toggle only for ~400+ chars (SD.5 kept).
- CO.1 add-to-collection `variant` — pill on detail headers, icon on dense cards/player toolbar.

**Resolved decisions (operator, 2026-09-09):**

- **PL.6 mark-as-played → a real `completed` flag** (new per-user field + endpoint; NOT reusing
  last-position). Full-stack; NOT yet built — next.
- **F5 trending colour → keep as-is + document.** Trending rails pre-filter to rising, so colour is
  redundant THERE (all green); it is meaningful anywhere direction varies. No product change; green
  ≥1.15 / red ≤0.85 / amber steady (`components/trending.ts`).
- **CO.1** → pill on detail, icon on cards (done).
- **SD.2** → keep the collapse toggle, larger collapsed preview (done); **SD.5 kept** (collapsible).

**Deferred with rationale:** RFC-121 **phase 3** (unified saved list + kind-filter chips) — premature
until phases 4–5 add topic/person/storyline/note saved kinds. **F4.2** sparkline opening-act &
**PD.1** host-show artwork — need entity trend-series / `PersonShow.image` from the backend
(`PersonShow` has no image field today). add-to-collection→overflow fold-in — do it with the
Player/Notes arc when mark-as-played/note join the `⋯`.

**Also shipped (session 2, 2026-09-09):**

- **PL.6 mark-as-played — FULL-STACK, done.** Per-user `completed` slug store + GET/PUT/DELETE
  `/api/app/completed` routes (server); `completed` client store (optimistic, outbox
  `completed.add/remove`); the player `⋯` (OverflowMenu's first real use) carries
  mark/unmark; completed episodes drop out of Continue-listening. Revisit deliberately NOT
  filtered (highlights are a different axis). Follow-up tests owed (continue-filter, `⋯`
  interaction).
- **BT.1/BP.1** trending topics/people: top-10 + `+10` per tap (opt-in `step` on
  TrendingSparkChips; Home untouched). **BT.2** storylines: top-10 + expand toggle.
- **BE.6/BE.7** episode filter: All / Unplayed / Played / With insights (+ Downloaded on
  native); sort-by-published already existed.

**Next (clean, no input needed):** BE.5/BS.2 grid/list view toggles; BE.2 read-more per
episode; BE.3 "insights" pill → popup; BE.1 "8 key points" data bug; then the detail/
storyline/person arcs, Notes feature (NT), Collections features (CO.2–CO.7), Home polish, Settings.

**Owed before push:** `ci-ui-full` (broad testid/i18n/component changes) + rebase on main; push
needs explicit approval. Follow-up unit tests: TrendingSparkChips increment, PlayerView `⋯`,
continue-filter.

## WAVE-3 RE-BASELINE (2026-09-09 session 3, verified against code)

F2 audit + fixes: **F2.2** (storyline favorite) and **F2.4** (shared `FollowButton`) shipped;
**F2.3** done; **F2.1** partial-by-design (favorite wording unified; "highlight" wording persists,
gated on deferred RFC-121 phase 3).

Wave-3 detail pages audited against the actual code — most were already built in earlier sessions.
The backlog's per-item status above was stale.

- **DONE (verified):** SD.1, SD.2, SD.4, SD.5, SD.6, SD.8, SD.9, SD.11; TD.1, TD.3 (sections are
  labeled "similar topics" vs "topics in this storyline"), TD.4, TD.5, TD.7; PD.4; PL.1; IN.1, IN.3,
  IN.4; F4.1, F4.3, F4.4 (toggle reads "All" / "My listening"), F4.6.
- **Open — needs operator input, NOT a blind edit:** SD.10 (tabs = structural reversal of the inline
  design kept for SD.2/SD.5); PL.3/PL.4 (subjective spacing/sizes — the control layout is already
  heavily reworked; needs a specific on-device problem to fix, not a guess); PD.3 "search further
  down" (subjective micro-placement; label part already done).
- **Open — needs a concrete product definition:** TD.2 (what "topic dynamics" to surface), TD.6
  (strongest-shows — likely a backend query), IN.2 ("underrepresented" — target unclear), SD.7
  (length/"rated" — "rated" has no data; avg length only from the loaded page sample).
- **Backend/data-dep (skipped per operator "skip backend"):** SD.3 (#2006 recurring_guests),
  PD.1 (`PersonShow.image` not in the type), BE.4 / BP.3 / PL.2 (host/guest roles, #1863).

Net: Wave 3 is essentially complete. Remaining work is decisions + backend, not clean UI builds.

**Wave 3 CLOSED (session 3, operator decisions applied):**
- **SD.7** DONE — typical episode length ("~48 min avg", median of loaded episodes) on the metadata
  line; "rated" skipped (no data).
- **TD.6** DONE — "Strongest shows on this topic" section (episodes grouped by feed, top 5, >1 show).
- **IN.2** DONE — storyline/similar cluster labels promoted to a clear left-aligned block in the
  insight panel.
- **TD.2** DONE (was already built) — EntitySignals momentum + `TopicConversationArc` on the card.
- **SD.10** WON'T-DO (operator) — keep the inline single-scroll show page.
- **PL.3 / PL.4** WON'T-DO (constraint, verified) — a symmetric flex layout and a smaller edge
  control both overflow the 412px transport row (a sub-44px edge control's `lp-tap` hit box spills
  4px past the edge; `design-invariants.spec` fails). The row is maxed at the 44px minimum. Attempted
  and reverted; documented in `PlayerControls.vue`.
- **Backend/data-dep (open, not UI):** SD.3 (#2006), PD.1 (`PersonShow.image`), PL.2/BE.4/BP.3 (#1863).

Wave 3 is now fully closed except the backend/data-dep items. Next clean UI: Wave 4 (Browse).

**Wave 4 (Browse) audited + built (session 3):**
- **DONE (already built, verified):** BE.1 (data bug fixed — card shows real `summary_bullet_count`),
  BE.2 (read-more), BE.5 (grid/list toggle on Episodes), BE.6 (hide-played), BE.7 (filter+sort),
  BT.1/BT.2 (topics/storylines top-10 + more), BP.1 (people top-10 +10).
- **DONE (built this session):** BS.2 (grid/list toggle on Browse › Shows), BT.3 (storylines list
  restyled to match the topics list — swatch + label + count rows).
- **WON'T-DO (superseded/constraint):** BE.3 (insights popup on browse cards) — the badge was
  deliberately renamed "N insights" → "N key points" because it counts bullets, not insights, and a
  true per-card insight count is intentionally not computed server-side (per-card artifact-load cost,
  schemas.py:104). A popup would re-conflate the two and re-add that cost. Key points already expand
  in place (BE.2).
- **NEEDS OPERATOR DECISION:** BP.2 (order People by role — guests/mentioned first, then hosts):
  role data is client-side, but the People tab is velocity-sorted (trending), and TrendingSparkChips
  re-sorts by velocity; role-primary ordering changes the tab's meaning + needs a shared-component
  change. BP.3 (guest-vs-mentioned clarity — small, could ride along).
- **BACKEND/DATA-DEP:** BS.1 (Podcast has no category field), BE.4 (#1863 roles), BT.4 (Storyline
  has no velocity field → no trending).

Wave 4 clean-UI work is complete. Remaining: BP.2/BP.3 (decision) + backend.

## PROPOSED ORDER (waves)

- **Wave 0 — Hotfix:** F1.1 offline blank-screen (broken app; repro-first).
- **Wave 1 — Affordance foundation:** F2 + F3 together (favorite + action set; both are
  card-affordance work, gated by decision Q1).
- **Wave 2 — Structure foundation:** F4 (detail template) + F5 (trending/color) +
  F1.2–F1.4 (degraded mode + skeletons, broad).
- **Wave 3 — Detail pages consume foundations:** SD (heavy), TD, SL, PD, PL, IN.
- **Wave 4 — Browse tabs:** BE, BS, BT, BP (view toggles, pagination, filters, hide-played).
- **Wave 5 — Features:** NT (notes), CO (collections), SR (search; depends on NT for
  "include notes").
- **Wave 6 — Polish:** H (home) + ST (settings).

Each wave = one themed branch, sliced per arc, bisectable per component (per operator's
"bundle across arcs on one branch" rule).

---

## OPEN DECISIONS (gate ordering / scope)

- **~~Q1 — Favorite affordance~~ RESOLVED (2026-09-09):** Save = "favorite" = heart icon
  everywhere. Follow = existing show pill, kept consistent. Separate actions, no merge.
- **Q2 — What leads: RESOLVED** — follow the proposed 7-wave order as written (Wave 0
  offline hotfix first).
- **Q3 — Notes scope: RESOLVED** — **full Notes incl. audio dictation** (native mic +
  notes-in-collections + folders). NT.3 is a Capacitor native feature.
- **Q4 — Target surface: RESOLVED** — **web + native iOS in lockstep**. Every change lands
  on both; native-only items (dictation, media-quality/download) flagged as they arise.

Resolved during their own arc (no gate): F5.2 color legend, F4.4 all/my-listening meaning,
BS.1 category taxonomy, TD.3 topic-vs-storyline section labeling.
