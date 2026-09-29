# Three defects that only exist on the device — 2026-09-27

All three were found by the operator on an iPhone 15 Pro, against builds that passed
**1,702 unit tests, `vue-tsc`, and `mkdocs --strict`**. None is reproducible in jsdom or in
desktop Chromium. That is the thread connecting them, and it is the most useful thing in this
document: see [What this says about the tiers](#what-this-says-about-the-tiers).

Branch `october-fixed`, 10 commits, unpushed. Device is running `e82b1c591`.

---

## 1. Collection names render blank in the add-to-collection menu — UNSOLVED

**Symptom.** Opening the add-to-collection menu shows rows with a visible "✓ Added" and **no board
name**. First reported as "the items are not painted in white", then "invisible collection names
after 2nd popup opening still there". The operator's own framing both times involved opening the
menu a second time.

**Two fixes were shipped and neither worked** (`1c3c40137`). Do not re-apply either reasoning.

### Ruled out — each by measurement, not by reading

| Hypothesis | How it was killed |
| --- | --- |
| The data is empty | Prod `collections.json` holds `AI`, `Investments`, `Tech` — read off the host |
| The wire drops it | `list_collections` spreads `**c`; `Collection.name` is a required pydantic field; `withAbsoluteCovers` spreads |
| The DOM lacks the text | Mounted with the real prod rows: names render on 1st, 2nd AND 3rd open. `getCollections` called 3×, correct every time |
| It is a colour inherited through the teleport | The row now states `text-canvas-foreground` explicitly in both branches — verified present in the installed bundle |
| `text-canvas-foreground` is not a generated class | It is: `.text-canvas-foreground{…color:color-mix(in srgb, var(--lp-canvas-foreground) …)}` |
| `--lp-canvas-foreground` is out of scope in a teleported panel | All theme vars are declared on `:root` |
| The span shrinks to zero width (`min-w-0` + `truncate`) | `flex-1` added, giving a definite basis — verified in the installed bundle |
| The compiled CSS does not paint it | Served the real bundle to Chromium with the panel's exact markup: names paint at 12.9px / 78.3px, full contrast, standalone AND under `lp-sheet-scrim`, with the name span topmost at its own coordinates |
| The phone runs older code | Prod runs `sha-9278574`; the file was unchanged since; and both fixes were grepped out of the installed `App.app` |
| The text is there but invisible | Contrast-stretched the screenshot's name column to full range: **no glyphs at any intensity**. Only the ✓ appears |

### The one observation that would split the remaining space

Long-press and drag across the blank row **on the device**.

- Selection highlights letters → the text exists and something is hiding it.
- Nothing selects → the text is genuinely not in the device DOM, and the cause is upstream of
  everything measured above.

Failing that: Safari → Develop → iPhone → the Close Listening WebView, and read the computed
`color` and `getBoundingClientRect()` of `span.min-w-0.flex-1.truncate`.

**Do not ship a third speculative fix without one of those two.**

---

## 2. "Ask this episode" returns search results, not an answer — DIAGNOSED, UNDECIDED

Not a bug; a design decision that was never made. The control is labelled **Ask** and runs a
**search**: it renders `hit.text` (raw transcript chunks) with `kp.noResults` when empty.

Three separate defects sit on top of it, from one screenshot:

1. **The same chunk was returned twice** — plausibly the chunking work in #2159 (a boundaryless
   transcript became one 44,924-char chunk); overlapping chunks would return near-identical text.
2. **Neither hit carried a timestamp.** `▶ {time}` is `v-if="hitStartSeconds(hit) != null"` and was
   absent on both, so jump-to-moment — the reason to ask an episode at all — was dead.
3. **The keyboard covers the results**, which render below the input.

**The fork, for the operator:** make Ask synthesise (an answer, with those chunks as citations
beneath it), or rename it to what it does. 1 and 2 are worth chasing either way; 3 is layout.
Nothing here has been changed.

---

## 3. Fixed chrome paints mid-list during scroll — NOT DIAGNOSED

The mini-player and bottom nav render **in the middle of the episode list**, with list content
continuing above and below them, during momentum scroll.

Checked and **not** the cause: the usual suspect is a translucent fixed layer, and both bars are
opaque — `MiniPlayer` is `bg-elevated`, `BottomNav` is `bg-canvas`, and `BottomNav` carries a
comment recording that `backdrop-blur` was deliberately removed for WCAG contrast.

Remaining suspicion, unproven: WKWebView compositing the fixed layer at a stale scroll offset.
Both bars position off `env(safe-area-inset-bottom)` inside a `calc()`, which is the only unusual
thing about them. Needs the inspector attached, or a hard-scroll reproduction in the simulator.

---

## What this says about the tiers

Tonight produced three defects that **1,702 green unit tests could not see**, plus one regression I
introduced and three of my own tests passed over:

- **The top layer.** The notes viewer was teleported to `<body>` while the Knowledge panel is
  `showModal()`'d. The top layer paints above the normal layer regardless of z-index, so the viewer
  rendered correctly and invisibly. jsdom implements neither the top layer nor `showModal` stacking,
  so "behind a modal" and "in front of a modal" are the same DOM there. Fixed in `e82b1c591`; the
  guard now asserts *containment in the open dialog*, because that is the part a unit test can see.
- **Hover latching.** iOS applies `:hover` on tap and leaves it applied. A `group-hover` tooltip
  therefore lit on tap and survived the navigation. No jsdom test can express this. Fixed in
  `1c1e04cae`.
- **Real layout.** The Saved colour strip overflowed by 20.4pt and hid a swatch behind a suppressed
  scrollbar. Only measurable against real geometry — it was in fact measured off a screenshot, not
  off a test.

The device-tier handover already names the gap: *"Neither tier runs in CI, so a green PR says
nothing about them."* Tonight is four more data points. The counter-argument in that handover still
stands — putting a flaky tier on required checks just relocates the pain — but the ordering it
proposes (determinism first, then path-filtered nightly) now has a concrete cost attached to the
delay.

**Cheapest thing that would have caught two of the four:** a single device-tier smoke that opens
each modal surface and asserts the thing it opened is hittable, not merely present. XCUITest
distinguishes those two; jsdom cannot.
