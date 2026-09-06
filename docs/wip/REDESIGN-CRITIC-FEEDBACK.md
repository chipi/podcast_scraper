# Critic feedback — compositional findings

Part of [#1941](https://github.com/chipi/podcast_scraper/issues/1941).

Two blind critique rounds: a cold baseline against the shipping app, and a re-score after the
fixture gained realistic cover art. The critic saw only pixels — no code, no brief, no indication
which variant it was judging.

**This file keeps the COMPOSITIONAL findings.** They are direction-independent: they describe what
is on the screen, how much of it, and where it sits. No palette swap fixes any of them, so they
survive whatever visual direction is eventually chosen — and they are the more valuable half of
the feedback for exactly that reason.

> **READ THE NUMBERS IN THIS FILE WITH SUSPICION (#1978).** Working through these findings, three
> separate measurements turned out to be DEVICE pixels reported as CSS pixels — off by the 2.625x
> Pixel 7 scale factor:
>
> | claimed | measured | where |
> | --- | --- | --- |
> | artwork panel "880px tall" | **372pt** | Player |
> | rows "~490px tall" | **212–237pt** | Browse |
> | "~1,200px of unbroken black" | **477pt** | Search |
>
> Two further claims did not reproduce at all: `Sort & filter` "occupies its own ~100px band"
> measures a **26pt collapsed pill**, and "three icon buttons per row" is four. The qualitative
> observations in this file have held up well — the numbers attached to them have not. Measure on
> the rendered page before acting on any figure here.
>
> Of 18 findings worked through, 7 needed work and 11 did not. That is not a criticism of the
> critique: a blind critic judging pixels cannot know that Profile is deliberately a settings page,
> that square cover art has no safe crop, or that a sparse surface is sparse because the account is
> empty. It is an argument for the measure-and-rule step between critique and code.

Scores: baseline **5/10** overall (Player 6, Home 5, Browse 5, Profile 5, Search 4, Library 4).
Re-scored against real artwork: Player **5**, Browse **5**, Profile **4**.

---

## The two systemic findings

**1. `Sign out` occupies the most valuable pixels on every screen.** Top-right on mobile is where
the most-used action belongs; the least-used one is there instead, styled as a bordered pill so it
outweighs every content-level action beneath it. Moving it into Profile reclaims the header on all
six surfaces at once.

**2. Density is unmanaged in both directions at once.** Home and Browse are crammed to the point
of duplicated labels; Search and Library are so empty they read as broken. The same app cannot be
both. The restraint that already exists on Library is what Home needs.

---

## Per surface

### Home — 5

- **Three pitches stack before any content.** Hero card, search field, and the "Personalize your
  Home" prompt all precede the first real thing. The numbered "What's new" rail — the one
  genuinely distinctive component in the app — starts roughly 1,900px down. It should be the
  second thing on screen.
- **The personalize card is the worst-composed object on the page.** Its 1px orange stroke fights
  the solid orange Search button ~40px above it; its title wraps to two lines in a column with
  200px of unused width; "Not now" aligns to neither the button's left nor its centre.
- **"Your shows" tiles say the name twice** — burned into the generated artwork, then repeated as
  a caption directly beneath.

### Player — 6 → 5

- **The artwork zone is ~45% of the viewport and does one job.** It pushes the scrubber, transport
  and insight into the bottom quarter. Two remedies were proposed and they are opposites: cut it
  to a ~40%-height cropped band, or keep it as a blurred colour source behind a full-bleed scrim
  so the insight card can live at real contrast on a real background.
  **Note the tension with UXS-011 §43**, which says this zone should be "a live intelligence
  surface, not decoration — speaking-now, a grounding badge, and the insight surfacing at this
  moment". It carries one of those three. Shrinking it and filling it are both defensible; doing
  neither is not.
- **The transport row is six mismatched shapes in a line:** rounded-square icon button, bare text
  `↺15`, filled circle, bare text `30↻`, circle icon button, pill `1×`. Two have no container at
  all. One geometry for the four secondary controls; let the play button be the only filled shape.
- **The glow behind the play button** is the only decoration on an otherwise disciplined screen and
  drags it toward "AI-generated dark UI".

### Search — 4

- **The empty state leaves ~1,200px of unbroken black** below three chips. Not confident negative
  space — an unfinished page. Raise the block into the vertical centre, or fill it with recent and
  saved searches.
- **The results row carries four controls of four different weights** — orange-ringed input, filled
  Search, outlined Save, ghost bookmark — and the input collapses to under half the row to make
  space. `Save` and the bookmark appear to do the same job.
- **Delete the standalone `Search` button.** The field is the button; submit on return. The freed
  ~200px lets query, scope toggle and save sit on one calm line instead of two crowded ones.

### Browse — 5

- **The accent has lost its meaning.** Orange appears ~60 times in one scroll: every show kicker,
  every insights chip, the tab underline, the active nav icon. When everything is accented, nothing
  is. Let one element per row carry it.
- **Rows are ~490px tall for four lines of text**, so only two fit above the fold. Moving the meta
  line onto the insights-pill baseline gets it to ~340px.
- **Three icon buttons per row, in two different container styles**, competing with the show name
  beside them.
- **`Sort & filter` occupies its own ~100px band** doing one job. It belongs on the tab row's right
  edge.
- **The show name appears twice per row** — in the artwork and as the kicker. Note this is now
  inherent rather than fixable: real podcast covers carry their own name, so the question is
  whether the kicker should repeat it.
- **20+ structurally identical rows with no break** — "a spreadsheet with pictures". One moment of
  hierarchy (a full-width editorial first result, or a divider every six rows) would do more than
  any per-row polish.

### Library — 4

- **85% of the viewport is empty black** with a heading, one sentence, and no action to take. An
  empty state with no CTA is a dead end.
- Proposed: one large muted example highlight card at ~40% opacity as a "this is what will live
  here" ghost, with a `Find something to listen to →` action beneath it.

### Profile — 5 → 4

- **Typographic hierarchy is flat.** `Interest topics`, `Your Week` and `Your activity` are the
  same size and weight, so the page reads as three equal blocks with no sense of what matters.
- **`Your Week` opens with a five-line explanatory paragraph** — settings copy at body-paragraph
  length signals a feature that has not earned its own name.
- **The checkboxes are the only components in the app that look system-default** rather than
  designed.
- **There is no evidence of the person.** No listening minutes, no streak, no shows followed. The
  most personal surface in the app is the least memorable. Push the toggles into the gear icon
  already sitting in the header and give the screen back to the user's own data.

---

## Already fixed

- **Empty reach pill on every episode** below the k-anonymity floor —
  [#1957](https://github.com/chipi/podcast_scraper/issues/1957), fixed on main.
- **Profile's segmented control** built differently from Search's, with no ARIA —
  [#1959](https://github.com/chipi/podcast_scraper/issues/1959), fixed.
- **Overlay legibility on the artwork.** At `bg-canvas/40`–`/80` the illustration read straight
  through the Summary pill and the reach chip, and the `INSIGHT NOW` card showed a backdrop-filter
  halo. All now `/95`. Only visible once the fixture had real artwork.

## Checked and rejected

- **"The artwork panel is ~880px tall, cut it 40%."** Measured: it is `aspect-square` at
  **372 × 372 CSS px**, 40% of viewport height. The 880 was device pixels. Not oversized.
- **"The inactive segment label sits at maybe 3:1."** Measured against the tokens: **6.44:1**
  (selected 6.15:1). Both pass 4.5:1. Taste, not accessibility.
- **"60% empty gradient"** in round one — an artifact of the fixture's synthesised covers, not the
  design. Gone once real artwork was in place.

## Was documented intent — re-opened and changed

Both of these were parked as "the spec says so, so the critic is wrong". Re-read, the critic was
seeing something real in each — not the decision itself, but a consequence of it that no one had
looked at since. Neither change contradicts its UXS section; both make the section true on screen.

- **`Highlights` as an `h2` inside `Saved`** (Library). The spec (UXS-014 §117-121) is kept —
  Highlights stays a folded-in section, the Knowledge tab stays gone. What changed is that it was
  the only UNCONDITIONAL section, so on an empty account it was the whole tab: one heading naming a
  third of what Saved holds, which reads as "this tab is just Highlights, and it is empty". It is
  now conditional like Episodes and Insights, and the tab carries ONE empty state naming all three
  things it holds plus the only action available — `Find something to listen to →`. The critic's
  "empty state with no CTA is a dead end" was right; the heading was never the point.
- **The purple kicker on Home's discover hero.** The `topic` tone is kept (UXS-012 §103). The
  problem was that it was the ONLY topic-coloured element on the screen, so a token that means "this
  is a topic" was carrying no meaning — it read as decoration, which is exactly the critique levelled
  at the orange accent on Browse. Fixed by giving the colour siblings rather than by removing it: a
  row of real topic chips now sits under the search field, drawn from the trending topics Home
  already fetches, and tapping one searches it. That also answers a second finding — the hero asked
  you to search across every episode and then offered an empty box you had to know what to type into.
