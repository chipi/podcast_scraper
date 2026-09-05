# Design hands — Phase 3 divergence (#1950)

Dealt `2026-09-05T04:26:34Z` by `scripts/tools/deal_design_hands.py`, five hands, dark grounds,
structure without replacement.

> **A second deal was generated at 04:26:58Z and discarded unseen.** It was an accident — a
> redirect that re-ran the dealer instead of capturing the output already on screen — but it has to
> be recorded, because "deal again" is exactly how taste re-enters a process built to keep it out.
> A dealer you can re-run until you like the hand is not a dealer. **The 04:26:34Z deal below is the
> one we execute**, and it stands whether or not it is comfortable.
>
> An earlier set of hands (`2026-09-05T04:01:58Z`) was dealt by the FIRST version of the dealer and
> is void: that version burned hands rather than interpretations (a statistical no-op — i.i.d.
> draws are unchanged by discarding some), dealt structure with replacement (three of five hands
> got the same card), had no typography card at all, and left every numeric value to be chosen
> rather than drawn. Five directions were written from it, and all five independently chose
> `radius: 0rem` though only one hand dealt the square card. That convergence is what exposed the
> problem.

---

## The rules attached to these hands

**Every direction must set the font tokens.** They are as value-open as the colours, and the
previous attempt rendered "broadsheet — hierarchy through type alone" and two hands of "type
carries ALL hierarchy" in Inter at shipping weights. A direction that does not touch type is a
recolour.

**Each hand is interpreted three times and two are discarded** — the interpretation closest to the
shipping app, and the one closest to an already-written direction. Each discard gets one line here.
The trained average lives in the execution, not in the deck, so this is where the burn belongs.

**Collisions are not resolved by ignoring a card.** Where two cards cannot both be satisfied
literally, the resolution is written down. Hand 2 is the live example: *"loud — three or more
saturated hues, none of them apologising"* against a dealt **chroma ceiling of 4**, which is
near-grey. Both cannot be true of the palette, so loudness has to come from somewhere that is not
chroma — scale, weight, lightness jumps, or the artwork itself.

**The artwork zone is a per-direction choice** (operator ruling, this session): a direction may
fill it as the live intelligence surface UXS-011 §43 promises, or shrink it to a band. Whichever it
picks is stated in its comment.

---

## Hand 1

```text
structure: Hierarchy by SIZE only — one weight, one colour for all text.
colour:    One hue only. Everything else is a value of it.
posture:   Hairline everything — the thinnest strokes carry all separation.
type:      Display serif for headings against a grotesque body — maximum contrast between the two.
world:     Early-web fan shrine — dense links, tables, personality over polish.
numbers:   ground=dark L*=9%  hue=215deg second=+180deg  chroma<=100
           radius=0rem  density=1.45  motion=1.5
```

## Hand 2

```text
structure: Type carries ALL hierarchy — no rules, dividers, borders or boxes anywhere.
colour:    Loud: three or more saturated hues at once, none of them apologising.
posture:   Generous and unhurried.
type:      Condensed grotesque: Haettenschweiler, Impact, Oswald fallbacks. Tall, tight, loud.
world:     Terminal multiplexer — panes, status bars, keybind hints, no chrome.
numbers:   ground=dark L*=3%  hue=304deg second=+150deg  chroma<=4
           radius=1.0rem  density=1.3  motion=1.0
```

## Hand 3

```text
structure: Each screen has exactly one focal object at twice the size of anything else.
colour:    Every colour desaturated to the edge of grey, distinguished by temperature alone.
posture:   Generous and unhurried.
type:      Serif throughout: ui-serif, Georgia, Charter. Editorial, not app.
world:     Seed catalogue — botanical plates, ornate labels, cream stock.
numbers:   ground=dark L*=11%  hue=271deg second=+120deg  chroma<=100
           radius=0rem  density=0.8  motion=0
```

## Hand 4

```text
structure: Everything is a stack of full-bleed horizontal bands, edge to edge, no gutters.
colour:    The palette is lifted wholesale from the artwork and nothing is fixed but the ground.
posture:   Heavy: thick strokes, blunt shapes, nothing delicate.
type:      Serif throughout: ui-serif, Georgia, Charter. Editorial, not app.
world:     Sports broadcast lower-thirds — bold blocks, live data, aggressive diagonals.
numbers:   ground=dark L*=9%  hue=224deg second=+150deg  chroma<=100
           radius=0.125rem  density=0.8  motion=1.5
```

## Hand 5

```text
structure: Dense information tables; whitespace is a failure, not a feature.
colour:    One hue only. Everything else is a value of it.
posture:   Soft, rounded, physical.
type:      Display serif for headings against a grotesque body — maximum contrast between the two.
world:     Broadsheet newspaper front page — column rules, decks, hierarchy through type alone.
numbers:   ground=dark L*=3%  hue=225deg second=+180deg  chroma<=80
           radius=0.75rem  density=1.45  motion=0
```

---

## Acceptance test, per direction

Applied **as rendered**, not as intended: a direction must differ from every other candidate on at
least three of four axes — palette, typography, posture, and what it does with the artwork zone.

"All square, all desaturated, all Inter" was the failure signature of the previous attempt. If it
reappears, the interpretation layer is still winning and the hands are not the problem.
