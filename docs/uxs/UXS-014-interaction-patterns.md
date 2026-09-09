# UXS-014: Interaction patterns (consumer)

- **Status**: Active (foundational — applies to every consumer surface)
- **PRD**: `docs/prd/PRD-043-knowledge-layer.md` (and all consumer PRDs)
- **RFC**: `docs/rfc/RFC-102-knowledge-clusters-entity-cards.md`
- **Inherits**: UXS-011 (Editorial Bold tokens, `--lp-*`), UXS-012 (Home), UXS-013 (knowledge).

This UXS is the **shared contract for how the consumer app behaves and is styled across surfaces**.
It exists because the operator demanded strict consistency: *define patterns and styles once, apply
them app-wide; never restyle the same affordance per page.* New UXS docs and components **conform to
this spec** — they do not re-invent navigation, layering, or saving.

## Surfaces — the four kinds and when to use each

| Surface | What | Examples |
| ------- | ---- | -------- |
| **Page** | A route; URL-addressable destination | Home, Search, Catalog, Player, **Library** |
| **Panel** | Persistent, in-layout region; not modal | Insights panel beside the Player |
| **Modal** | One dimmed backdrop; teleported to `<body>` | Interests picker, entity card **from Search** |
| **Sheet** | The mobile form of a panel/modal (bottom, drag-handle) | Insights on mobile |

## Core rule — never stack two dimmed layers

- **Drilling deeper *inside a panel* replaces the panel's content in place** with a `‹ Back`
  (a back-stack), never a new overlay. Example: tapping a person/topic chip in Insights swaps the
  panel to the entity card; `‹ Back` returns to the insight list.
- **A modal opens only from a page-level surface**, never on top of a panel/sheet. So the entity
  card is *in-panel* from Insights but a *modal* from Search (a page).
- **At most one backdrop on screen.** If you would dim a second layer, use replace-in-place instead.

### Entry-point → surface map

| You tap … | … here | Result |
| --------- | ------ | ------ |
| Person/topic chip | Insights **panel** | Replace-in-panel (entity card, `‹ Back`) |
| Entity match | Search **page** | Modal entity card |
| "Set interests" | Home **page** | Modal picker |
| A "see all" link | any | Navigate to a **page** |

## Layering mechanics

- Every overlay **`Teleport`s to `<body>`** so it covers the viewport and escapes any clipped or
  transformed ancestor (a panel sheet uses `overflow-hidden`/offsets that otherwise clip a nested
  `position: fixed`).
- **z-scale:** panel `z-40`, modal `z-50`, mobile backdrop `z-30`.

## Headers & navigation

- **Header order is `‹ Back` (own row) → kicker → title.** The entity card header mirrors the
  episode-detail masthead exactly — back never crammed beside the kicker/name.
- **Navigation reads differently from content labels.** A `‹ Back` control is muted (`.lp-nav`),
  never accented; navigation is not a thing you act on. The two must contrast.

## Shared style classes — define once, use everywhere

Recurring affordances are **single classes in `web/learning-player/src/style.css`**, not per-element Tailwind
hand-rolled on each page. Adding a one-off `class="text-muted …"` for one of these is a regression.

| Class | Role |
| ----- | ---- |
| `.lp-kicker` | Editorial eyebrow / content label (mono, muted, uppercase) — e.g. a show name. The instrument voice: measured metadata, never interactive. (#2013) |
| `.lp-section` | Section/region heading (calm display heading) — **never** the kicker, so a section title can't be mistaken for a show name |
| `.lp-speaker` | Speaker attribution in transcript / quotes (muted, normal-case) — distinct from the kicker |
| `.lp-nav` | Back / navigation control (muted; distinct from content) |
| `.lp-fav` | Favorite (heart) toggle; `.lp-fav--on` = saved. The only toggle that spends the accent on hover/press. |

When a new recurring treatment appears, add one class and reuse it — do not copy styles between
pages.

> **Amended #2013 — The kicker is NOT accent.** The table previously read `.lp-kicker` as "(accent,
> uppercase)". The prior design shipped with the kicker accented on 55 call sites across 25 components,
> so orange became the app's label colour, and when everything is accented, nothing is. A blind design
> critic's finding ("the accent has lost its meaning") was confirmed by measured analysis (158 accent
> usages, 30 decorative). The kicker is a label you cannot tap — it carries metadata about content
> (show name, duration, timestamp) using the instrument voice (mono, letterspaced caps, muted), the
> same treatment every measured value carries. This is enforced by `src/__checks__/accent-discipline.test.ts`.
> The accent now means "you can act on this" only — it is spent on focus rings, exclusive-choice toggles
> in their selected state, and toggle affordances in hover/pressed states. See UXS-011 decision #2013.

**Show names never truncate — where the layout has room.** In a full-width row, a list item, or a
header, a podcast/show name **wraps to the next line** rather than ellipsising. (Episode titles may
still clamp; show names do not.)

> **Scoped #1604.** The rule as originally written was unqualified, and it is **incompatible with
> uniform grid rows**: in a fixed-width tile, a name that wraps freely makes the row as tall as its
> longest member, which is the exact defect #1584 was filed to fix. Something has to bound the
> label.
>
> So the rule now holds where width is elastic, and in **fixed-width grid or rail tiles** a show
> name may clamp — but only with a **reserved height** (so rows stay uniform) and a `title`
> attribute (so the full name stays reachable). See `ShowTile.vue`.
>
> This is recorded because I broke the rule before scoping it: #1584 added `truncate` to
> Recommended's show kicker to stop it wrapping and undoing the reserved height, resolving the
> conflict silently in the code. That is the behaviour the drift audit exists to prevent, so the
> conflict is written down here instead.

## Drill-in navigation

- Drilling deeper is **replace-in-place with a `‹ Back` stack**; closing returns to the prior view
  in the **same** surface (no layer added or removed).
- The shared body (e.g. `EntityCardBody`) is rendered **inline** in a panel and **wrapped in the
  modal** from a page — one component, two presentations (`variant`), so they cannot drift.

## Dismissal & accessibility (every modal)

- Dismiss via **ESC**, **backdrop tap**, and an explicit control — all three.
- `role="dialog"` + `aria-modal`, a **focus trap**, **initial focus**, and **restore focus on
  close**. In-panel replacements move focus to the new heading instead of trapping.

## Tab strips and option groups (#1594 item 7)

`Tabs.vue` — the only tab strip. Seven hand-written ones preceded it and none was complete; the two
rules every copy missed are the two that are invisible unless you are already navigating by keyboard.

**Which pattern.** The question is not what it looks like, it is what it controls:

- switches **between distinct panels** → `pattern="tabs"`: `role="tablist"`/`tab`/`tabpanel`,
  `aria-selected`, and each tab's `aria-controls` naming its panel. Library, Browse, Home discovery.
- **re-parameterises one region** → `pattern="radio"`: `role="radiogroup"`/`radio`, `aria-checked`,
  and no `aria-controls` at all. Search scope, entity-card corpus scope, trend window, Your Week
  layout. These were all marked up as tablists, and none of them had a panel to point at — a
  `role="tab"` whose `aria-controls` names nothing is a dangling promise, worse than the missing
  linkage it would have replaced.

**Roving tabindex** (both patterns). Exactly one option is in the page tab order; the arrows move
between them and selection follows focus. Plain buttons are *usable* — you can Tab to each one — so
nothing looks broken; it just costs a five-tab strip five Tab presses instead of one, and the arrow
keys do nothing. Wraps at both ends; Home/End jump to the extremes.

**Tab ↔ panel ids** come from `tabId()`/`panelAttrs()` in `components/tabs.ts`, so both ends of the
pair are generated from one prefix and cannot silently disagree. A panel carries `tabindex="0"`: one
holding no focusable element of its own is a dead end for a keyboard user.

**Three visual variants** (`underline`, `segment`, `pill`) are kept on purpose — an underline is a
page-level section switcher, a pill a compact in-card control. One component, not one appearance.

**`.lp-segment-option` is styled from the ARIA state**, and matches BOTH `aria-selected='true'` and
`aria-checked='true'`. Keying it off one attribute is how the Your Week preference kept working and
quietly stopped looking selected when it became the radiogroup it should always have been.

`src/__checks__/tabs-single-implementation.test.ts` fails the build on an eighth hand-rolled strip.

## Folding a long panel (`CollapsibleSection`)

The Knowledge Panel's spine — Summary, Key points, Topics & People, Insights, More like this — is
long: ~8 key points of ~200 characters, and up to 36 insight rows. Folding is how you reach the part
you came for without scrolling past everything else.

- **Native `<details>`.** Keyboard operation, the disclosure role and the expanded-state
  announcement come from the element. Rebuilding those with a div and a ref is where a11y bugs live.
- **Open by default, always.** Collapsing by default hides the substance behind a tap nobody asked
  for; the panel's job is to show it. Folding is an escape hatch, not the resting state.
- **The count rides in the header** — `Insights · 8`. A folded section must still say what it holds,
  or folding costs you the knowledge that it exists.
- **The Summary never folds.** It is the reason the panel was opened, and it is one paragraph:
  folding it saves nothing and hides the one thing everyone wants.
- **State is per USER, not per episode** (`lp.kp.<key>`). "Don't show me related episodes" is a
  preference about the panel; keying it per episode would ask the same question on every episode.
- **Storage failure falls back to OPEN.** A preference we cannot persist is not a reason to hide
  content.

## Cards vs tiles — match the shape to the container

Two components, and the choice is not stylistic:

- **`EpisodeCard`** is a horizontal ROW: artwork column, text column beside it. Correct in a
  vertical list, where the row is as wide as the page — Podcast, Queue, the Queue panel's
  recently-played.
- **`EpisodeTile`** stacks: artwork on top at full slot width, then actions, then a full-width
  title clamped to three lines. Correct in a horizontal RAIL, where each slot is narrow.

**Putting a row card in a rail slot is the failure this rule exists for.** "More like this" did
exactly that: the text column got ~100px of a 224px slot, one real title wrapped to eight lines, the
slot grew to ~800px tall, and the action row — positioned against the card's top-right — floated
over the artwork. Nothing errored; it just looked broken and wasted most of the vertical space.

**A narrow slot drops things, and says so.** No summary: at 176px a truncated fragment is the shape
of a summary rather than one, and the title earns the space. Actions are the shared **minimum row**
(`EpisodeActions` — favourite, download, queue; see "Item actions" below), not a per-tile subset:
three 32px targets sit at a non-overlapping `gap-3` pitch across 176px. Add-to-collection is NOT in
the row — it is a detail/overflow action. (This supersedes the earlier "two actions, not four" tile
rule, which predated the shared action row.)

**Actions go below the artwork in a tile.** `ShowTile` overlays a single follow button deliberately
and that works for one; two icons over episode art is crowding.

## Item actions — the standard set, overflow, and per-surface context

The minimum action set was hand-rolled per surface, so rails carried only favourite+queue, Home's
What's-new / Recommended were missing favourite and download, and add-to-queue lived only on the
player. This section is the single contract; components conform, they do not re-decide per page.

**Save ≠ Follow — two different actions.**

- **Favorite = save to Library.** ONE affordance, the `.lp-fav` heart, everywhere an item can be
  saved (episode, and any saveable entity). Never a pill, never a second glyph. All saves land in
  Library › Saved.
- **Follow = subscribe to a *show* (or interest token).** The follow **pill** (`+ Follow` /
  `✓ Following`), rendered/behaving identically wherever it appears. It is not a save; the two are
  never merged and the episode heart is never swapped for a follow pill.

**The shared minimum row (`EpisodeActions`).** Every episode surface shows favourite · download ·
queue via the one component. Download self-hides on web (`DownloadButton` is native-only), so the
row is favourite+queue on the web PWA and all three on native — parity, not a per-surface omission.

**Overflow (`⋯`) where space is tight.** Primary actions sit inline; anything that does not fit is
pulled into a `⋯` menu — one extra tap, never a dropped capability. Secondary/detail actions live
there by default (add-to-collection, add-note, share, mark-as-played). Roomy surfaces (detail rows,
the player) may inline more before overflowing; dense tiles/rails inline the primaries only and
overflow the rest.

**Per-surface context — a surface never shows the "add-to-X" action for the X it already is.** That
action inverts to a remove or drops. Everything below follows from that one principle plus the
density rule.

| Surface | Favorite | Queue | Download | `⋯` overflow |
| ------- | -------- | ----- | -------- | ------------ |
| Home rails / Browse / Search / detail episode-lists | add | add | native | add-to-collection, add-note, share |
| Library › Saved | see **OPEN-1** | add | native | remove |
| Queue | add | **remove** (inverted — you are in the queue) | native | … |
| Downloaded list | add | add | **downloaded → delete** state | … |
| Collection detail | add | add | native | add-to-collection *for this collection* omitted |
| Player (current episode) | add | **n/a** (it is playing) → **mark-as-played** | native | add-to-collection inline (roomy) |

> **OPEN-1 — RESOLVED (RFC-121): Library favorite = keep the heart, inverted.** On Library the heart
> shows saved-state truth and is **one-tap unfavorite** — not dropped. This matches invert-don't-drop
> (Queue→remove, Downloaded→delete) and satisfies "no *add* on Library". The redundant `⋯ remove` in
> that row is dropped; the confirm-on-authored rule (below) makes one-tap unfavorite safe on noted
> items.
>
> **OPEN-2 — RESOLVED (RFC-121): one "Saved" concept over two identity classes.** "Favorite" and
> "Highlight" become one user-facing concept (the `.lp-fav` heart); the word "Highlight" leaves the
> UI. But **Saved is not one record shape** — it is one concept over **class A** singletons keyed
> `(kind, ref)` (episode/show/topic/person/storyline, toggleable) and **class B** captures keyed by
> `id` (insight/moment/span — today's highlight, kept). A moment cannot live in `(kind, ref)`, so a
> favorite-with-a-moment IS a class-B record. #1593 is preserved, not broken: the insight heart routes
> to the existing capture/highlights write path (re-skin, not re-plumb), and `PUT /favorites` gets a
> **422 on `kind=insight`** so the banned second write-path cannot return. Notes/colour become optional
> extras on any save. Full model, phased plan, and migration (read-layer only, no on-disk migration):
> **`docs/rfc/RFC-121-unified-saved-model.md`**.

## Insight type marks (#2004 item 8)

`InsightTypeMark.vue` — how one insight is told from another in a list that can hold 36 of them.

**Shape first, colour second.** Four SVG marks at one fixed size — diamond (claim), ring
(observation), triangle (recommendation), square (question) — so all four carry the same optical
weight. Text glyphs did not: `◆` and `?` are punctuation and read as typography, at whatever weight
the font gives them. A type outside the closed vocabulary gets a neutral dot, never nothing; an
empty mark column on the one already-unusual row is worse than an unlabelled one.

The set must stay legible in **greyscale** — colour is the second channel and never the only one.
`KnowledgePanel.test.ts` asserts shape-distinctness separately from colour for that reason.

**Colour rides on the mark, never the label.** `--lp-insight-*` alias `--lp-topic`,
`--lp-grounded`, `--lp-warning`, `--lp-person`, so every visual direction adapts them for free
rather than needing four hand-tuned hues each. The label stays mono + muted (`.lp-kicker`), and
none of this spends `--lp-accent`, which means "you can act on this". A direction that collapses
two of those base tokens makes two marks share a hue — survivable precisely because shape carries
the distinction.

**A symbol nobody can decode is decoration.** Each mark carries a `title` describing what the type
MEANS ("Claim — something the speaker asserts as true"), so on a pointer device the meaning is one
hover away. The visible type word carries it everywhere else, which is why the mark itself is
`aria-hidden`: a screen reader should hear "claim", not "diamond claim".

**No second constant mark may precede it.** A green "grounded" dot used to, on every grounded row —
and it rendered on the same condition as that row's `▶ mm:ss` button, so it distinguished nothing
while diluting the mark beside it. That is the failure this pattern exists to prevent, and a test
asserts the type mark is the first element in the row.

## Destructive confirmation (#1594)

`ConfirmDialog.vue` — the one pattern in front of a delete that cannot be undone.

**When it applies.** A delete gets a confirmation when it destroys something the user **authored or
curated** and the app **cannot restore it exactly**: delete a collection, delete a highlight, delete
a note. All three fail the restore test for the same reason — the create endpoints mint a new id, so
an "undo" would produce a different object wearing the same name, with every reference to the
original still broken.

**When it does not.** Removing an item from a collection is a membership row; the item itself
survives and re-adding it is two taps from the same screen. It gets **no** dialog. Confirmations
spent on cheap, reversible actions are how people learn to dismiss them without reading — which
costs exactly the three above.

**Prefer undo where an exact restore IS possible.** Confirmation is the fallback for when it is not.
Do not add a dialog to an action you could simply reverse.

**Mechanics.**

- A native `<dialog>` opened with `showModal()`, so the browser supplies the top layer, focus trap,
  Escape and inert background (same reasoning as the Knowledge Panel, S9).
- **Initial focus is Cancel**, never the destructive button. A dialog that opens with Delete focused
  turns "tap, tap" into a deletion: a step without a decision, which is worse than no dialog because
  the user now believes they are protected.
- The confirm button carries the **verb** ("Delete collection"), never a bare "OK".
- The dialog's own `close` event maps back to a cancel. The browser closes on Escape without telling
  the parent, and a parent that keeps its pending id shows no confirmation on the *next* delete —
  a failure that appears one action after its cause.

## Player hero (artwork zone)

The Player masthead is a **hero**: a fixed-square artwork carrying overlays, so layout height is
constant regardless of content length.

- **Summary** is revealed on demand: hidden by default (clean artwork), it **slides up + fades in on
  hover/focus** over a darker legibility gradient (`from-black/95 via-black/85 to-black/40`, white
  text) so it stays readable even over bright artwork. Always shown on touch (no hover). Full text,
  never clamped — the fixed-square hero stabilises height regardless.
- **Live intelligence** ("Insight now / Speaking now") sits top-left; the **per-episode reach**
  cluster (listeners · opens · Insights + a tiny opens-over-time `Sparkline`) sits top-right. The
  Insights score opens the panel — no duplicate "Ask" entry (Ask lives inside the panel).
- The **Grounded** chip sits up by the date/meta line, **not** floating over the image.

## Saved & Library

- Per-user collections live in **one "Library" hub** (page) with tabs **Following · Saved ·
  Collections · Revisit** (`LibraryView.vue:36`; the `Following` tab's key is `shows`). **Highlights**,
  **Queue** and **Recent** are **not tabs** — Highlights is an `h2` section inside **Saved**; the
  player auto-resumes from the saved position, so recent/played episodes need no separate "resume at"
  affordance and no dedicated tab.

  **Saved's empty state (#1962).** All three Saved sections — Episodes, Insights, Highlights — are
  **conditional**; none renders a heading over nothing. When all three are empty the tab shows **one**
  empty state naming what it holds ("Episodes you favourite, insights you keep, and moments you mark
  all live here"), a muted ghost card showing the shape of a future entry, and the single action a
  person can take about being empty: `Find something to listen to →` (to `catalog`).
  Highlights used to be the only *un*conditional section, so a fresh account met a lone `Highlights`
  heading standing in for a third of the tab and read the tab as redundant. The heading placement is
  unchanged — it is still an `h2` inside Saved, per the paragraph above; only its gating is.

  > **Amended #1599 (source of truth: `LibraryView.test.ts` :82-89).** This spec said **Saved ·
  > Knowledge · Queue · Recent**, then briefly **Saved · Highlights · Collections · Revisit · Queue ·
  > Recent** — neither matched the code. The shipped tab set is the four above. History:
  > `b3cd95d1` (2026-06-28) added a Knowledge tab and the original wording together; `a9705819`
  > (#1141, 2026-07-05) deliberately removed the tab and merged insights back into **Saved**;
  > `Collections` became a first-class tab under RFC-119; `Shows` returned as **Following**. None of
  > those code changes amended this spec, and the only written record was a comment in
  > `LibraryView.test.ts` *asserting* the absence of Knowledge/Queue/Recent — so CI defended the
  > undocumented state against the documented one.
  >
  > The code is kept (it shipped, stuck, and is test-defended); the spec is corrected to match. See
  > #1604 — a decision recorded only in a test comment is a decision that gets lost.

  (The **Following** tab covers followed shows, topics, people and storylines; Home's "Your shows"
  deep-links here via `?tab=shows`. This is #1585 — now **built**, not pending.)
- **One card, every surface.** Catalog, Saved, Queue and Recent all showcase an episode through the
  shared `EpisodeCard` (Queue keeps a slim ↑/↓ reorder rail beside it; the card's own queue toggle
  removes). Hydrated `EpisodeDetail`s are adapted via `summaryFromDetail` so they never drift.
- **Saved** holds favorited **episodes**, plus a **legacy, read-only** Insights section. The backend
  favorites bucket is polymorphic (`AppFavoritesResponse`: `episodes`, `insights`) and both still
  render, but since **#1593 nothing writes insight favourites any more**: an insight had BOTH a
  bookmark (→ Highlights) and a heart (→ Saved › Insights) — same text, two destinations, two places
  to look for it later. **Highlights is the single destination** for insights; it carries colours,
  notes and export. Existing saves stay readable and removable, and that section disappears on its
  own as each user clears theirs. Do not add a new write path.
- Saving is the shared `.lp-fav` heart on **episodes** (episode cards, the player masthead). It is
  no longer on insights.
- Favorites / queue / interests / playback are **per-user files** (no DB). Interests are viewable +
  editable on the **Profile** page (header → user icon).
- **Following an interest** is a one-tap toggle on a person/topic **entity card** (`Follow` /
  `Following`), in addition to the Home cluster picker. The interest list is a mixed token set —
  clusters (`tc:`), topics (`topic:`) and people (`person:`) — and re-ranks "Recommended for you"
  by how many followed tokens an episode matches (flag-gated personalized discovery, PRD-043).

## Listening analytics

Listening stats are computed from per-user files — **no LLM, no DB** — and surface in two places:

- **Profile "Your listening"** (own data): single scores — day streak, episodes, shows, hours — plus
  an opens-over-time `Sparkline`. Derived from the user's playback + listen log.
- **Player per-episode reach** (cross-user, anonymous): distinct listeners, total opens, an
  opens-over-time `Sparkline`, and the grounded-insight count. Aggregated by scanning every user's
  listen log; counts only, never identities. Public (no auth).
- The **listen-events log** (`<data_dir>/users/<id>/listen_events.jsonl`, append-only) is the only
  per-listen history we keep — playback stays last-position-only. The player appends one "open"
  event on mount. `Sparkline` is the single shared mini-chart (`currentColor`) for both surfaces.

## Header navigation

The header uses **icon links with hover/focus tooltips** (`NavIconLink`) — Browse (compass),
Library (book-spines), Profile (user) — never bare emoji; one shared component, labelled by
tooltip. Lists use the shared collapsible **`ListToolbar`** (search · sort · filter, incl.
filter-by-show), not stock inputs.

## Conformance checklist

- [ ] No second backdrop; drilling inside a panel is replace-in-place with `‹ Back`.
- [ ] Overlays `Teleport` to `<body>`; correct z-scale.
- [ ] Back/nav uses `.lp-nav`; content labels use `.lp-kicker`; section headings use `.lp-section`;
      speakers use `.lp-speaker`; saving uses `.lp-fav`.
- [ ] Episodes showcase through the shared `EpisodeCard` on every collection surface.
- [ ] Library tabs are **Following · Saved · Collections · Revisit** (`LibraryView.vue:36`); saved
      insights render as an **Insights** section inside **Saved**, not under a separate Knowledge tab
      (there is none). Highlights/Queue/Recent are not tabs.
- [ ] Following an interest is the one-tap `Follow` / `Following` toggle on a person/topic entity
      card (mixed-token interests: `tc:` / `topic:` / `person:`), alongside the Home cluster picker.
- [ ] The Player hero summary is hover/focus-revealed (slide-up + fade over the legibility
      gradient), always shown on touch; never clamped.
- [ ] Listening analytics surface through the single shared `Sparkline` — Profile "Your listening"
      (own data) and the Player per-episode reach cluster (cross-user, anonymous, public).
- [ ] Header order: `‹ Back` row → kicker → title.
- [ ] Modal a11y: dialog/aria-modal, focus trap, restore focus, ESC + backdrop + control.
- [ ] No per-page restyle of a shared affordance.
