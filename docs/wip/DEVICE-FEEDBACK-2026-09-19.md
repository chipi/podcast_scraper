# Device feedback round — 2026-09-19

Everything the operator reported from the installed app on 2026-09-19, what was done about it, and
what was **not**. Branch `fix/ui-followups-2026-09-18`, three commits.

## What turned out to be a real bug (not a layout preference)

### 1. Storyline rows in Trends were dead on tap

Reported as "storylines do open directly from the topic, but not from the trends".

A storyline has no endpoint of its own — it is read as its most-central member topic's card — so a
trend row needs an **anchor topic id**. The client derived one by joining the trending rows against
`GET /theme-clusters` on `thc:` id. Those two lists never covered the same set and could not:

| | ranked by | member floor |
|---|---|---|
| `/theme-clusters` | size, top-N | ≥ 4 (`DEFAULT_MIN_THEME_MEMBERS`) |
| `/trending?kind=storyline` | momentum | none |

So a storyline that was small, or large but outside the size top-N, was absent from the join — and
the client's `?? entity_id` fallback then handed a `thc:` id to a lookup that resolves a **topic**.
Nothing resolved; the tap did nothing at all.

`/trending` now carries `anchor_topic_id`, resolved server-side from a new `theme_cluster_anchors()`
— every cluster, no floor, no top-N — because the route already knows which clusters it ranked. **No
client fallback:** a row the server could not resolve is genuinely not openable and renders
disabled rather than pretending to go somewhere.

### 2. "all ›" on Discover did nothing

Discover in the tab bar **is** the `browse` route, and browse is the only surface that renders that
header. The RouterLink pointed at the page you were already on, carrying the kind you were already
reading. Vue Router navigated, the query changed, nothing moved.

It is now a button that uncaps the list in place, rendered only when rows are genuinely hidden.

### 3. `tc:` and `thc:` were the same kind to the app

`interestKind()` reported **both** as `'storyline'`. Harmless while the kind only picked a hue;
a false claim the moment each profile pill named itself. They are different objects, and the wire
names invert against the reader-facing ones:

| Wire prefix | Backend name | Built from | Reader-facing name |
|---|---|---|---|
| `thc:` | theme cluster | co-occurrence — topics that keep coming up together | **Storyline** |
| `tc:` | topic cluster | vector similarity — topics that mean similar things | **Theme** |

Settles the `tc:` half of #1603 the way UXS-013 always asked. `--lp-theme` was being worn by two
**storyline** surfaces — the same swap one layer down — so both moved to the accent treatment the
storyline pill uses, and the token now renders the thing it is named after.

## Layout and behaviour

| # | Ask | What changed |
|---|---|---|
| 4 | Transcript: time first, subtle separator, speaker, flush left | The only thing between speaker and time was the green **grounded** marker — present on a paragraph that grounds an insight, absent otherwise — so the line appeared to gain and lose punctuation depending on content. Order is now fixed and the marker is gone from that row; grounded-ness stays signalled on the paragraph (underline + `aria-label`), so removing it does not reduce the state to colour. |
| 5 | Similar topics should not list the current topic | It led the list as a ringed chip and was counted, so the heading disagreed with what was under it. Both fixed. |
| 6 | Conversation-over-time chart up beside the sparkline | Was at the foot of the page below the episode list — two time-series about one topic at opposite ends of a long scroll. |
| 7 | Perspectives up under Top voices | Same people, named. |
| 8 | Person page: episodes below related topics | A frequent guest put dozens of episode rows between the reader and two rows of pills. |
| 9 | Cap "discussed in N episodes" at 10, page by 10 | New shared `EntityEpisodeList` replaces four uncapped lists (topic / storyline / person / org), which had also drifted — three said "newest first", the storyline did not. |
| 10 | Storyline should say "newest first" too | Carried by the shared component. |
| 11 | Discover: episodes load-more should match the shows one | Full-width bar, not a pill. |
| 12 | Recommended: cap at 4 + show more | Four, then four more. Fetch raised from the api default of 6 to 12 so the control is worth the tap. **No "see all"** — nothing in the app is a recommendations page, and inventing a route to satisfy the shape of a link would be worse. |
| 13 | Strongest shows: add artwork, compact list | `EpisodeRow`'s 40px thumbnail. The feed artwork was already in the payload; the computed dropped it. |
| 14 | Filters on one row (Following / Saved / Notes) | All four strips are one shared `TypeFilterBar` — `flex-wrap` out, horizontal scroll with hidden scrollbar in. |
| 15 | Saved: mute on the colour row, with a separator | Swatches scroll; mute and clear pin right behind a hairline. |
| 16 | Notes filters: counts per kind | Counted over **all** notes, not the search-filtered set — a chip whose number changed as you typed would answer a different question from the one it asks. |
| 17 | Boards: counts on "Your collections" / "Your notes" | Same `lp-kicker` slot Saved and Following use. |
| 18 | Saved: cap shows and episodes at 5, page by 5 | Was 10+10 for episodes and **no cap at all** for shows. |
| 19 | Saved: drop the duplicated highlight count | The page said "Highlights 12" then "12 highlights" one line below. The heading keeps the tally; the export row keeps the formats. |
| 20 | Profile: tell storyline / theme / topic / person apart | Each pill names its kind in a mono kicker as well as carrying a hue. Hue alone was never going to carry it — the three knowledge-layer colours sit close in value by design — and carried nothing for a colour-blind reader. |
| 21 | Queue reachable without playing | Its only entrances were the full player and the mini-player, and the mini-player only exists while something is loaded. Added beside Resume on Home. |
| 22 | Add-to-collection: mark boards the item is already in | `GET /collections` takes an optional `contains_kind`/`contains_ref` pair. `contains` is **NULL** when unasked, so "not in it" stays distinguishable from "never checked". |
| 23 | Export PDF: brand it | Both printable documents carry a header naming the app plus the show, and a footer linking home. The page **body stays light** on purpose: the app's canvas is navy and printing that empties a cartridge for a background nobody asked for. |

## Answers to questions asked, not work items

**Theoretical maximum of recommendations.** The route ceiling is **25** (`top_k`, `ge=1, le=25`);
the similarity merge caps at **50**. The client was asking for the api default of **6** and then
slicing to 8 — a dead slice, since only 6 ever arrived. Now fetches 12.

**Can one item be in multiple collections?** **Yes in the storage, no in the UI.** Membership is
`{collection_id: [items]}` — many-to-many — and `add_item` appends to one list without touching the
others. But the popup closes 800ms after a pick, so a second board needs re-opening it. Nothing was
changed here: the operator asked to clarify, not to fix.

**The changing highlight colour.** Not a leak. `App.vue` watches the playing episode's artwork,
extracts its dominant colour and writes `--lp-accent` on the root element. In the `broadcast`
direction `--lp-topic` / `--lp-person` / `--lp-theme` / `--lp-grounded` are all aliased to it, which
is why the whole knowledge layer follows. Operator elected to keep it and live with it a while.

## Screenshots

`docs/wip/feedback-2026-09-19/` — one per item, cropped to the element that changed, captured by
`e2e/design/feedback-2026-09-19.design.spec.ts` against the fixture corpus at 375px.

Two things the screenshots caught that no test did:

- **The strongest-shows rows rendered broken images.** `topShows` is built from an episode list, so
  the only show-level image available was `feed_image_url` — the feed-hosted original, which points
  off-origin and is routinely unreachable. Our stored copy existed but was not exposed per-episode:
  `artwork_url` prefers the EPISODE's own image when it has one, so reusing it would hand a show row
  whichever episode came first and call that the show's cover. `AppEpisodeSummary` now carries
  `feed_artwork_url`.
- **"Show 8 more" revealed four.** The label counted what remained while the handler adds one page.
  The arithmetic was correct under both readings, so nothing was red.

And one process note worth keeping: the first three capture runs were wrong because
`playwright.design.config.ts` sets `reuseExistingServer: true` and I had overlapping runs, so an API
server started BEFORE the `feed_artwork_url` fix stayed alive on :8011 and kept serving the old
payload. I spent four tool calls theorising about URL resolution before checking what was actually
listening on the port. One run at a time; check the server's start time before doubting the code.

## NOT done / NOT verified

- **Multi-collection add is still UI-blocked.** Deliberate — clarification was requested, not a fix.
- **The `"theme" / "similar"` i18n pair and the Knowledge Panel lead-in are untouched.** This round
  settled the INTERESTS vocabulary only; the rest of #1603 stands.
- **Where semantic clusters belong in the product** remains open (#1603, #1595).
- **PDF opening in-app, and the Markdown filename.** Reported the same day, NOT addressed in this
  round — only the styling half of that message was. The in-app viewer still shows the coming-soon
  gate, and the Markdown file is still named from the slug rather than the episode title.
- **Nothing here is verified on device.** Everything is verified by the unit suites and by
  screenshots from the design harness against the fixture corpus. The reported bugs were diagnosed
  from code and pinned with tests; none was reproduced on the phone first.
- **#2120** (a download reports complete with nothing on disk) is still open and undiagnosed.
- **#2121** still needs device verification.
- **No e2e run.** The design harness produces screenshots, which prove rendering, not behaviour.
