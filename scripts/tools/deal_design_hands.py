#!/usr/bin/env python3
"""Deal random constraint hands for the redesign's divergence phase (#1950).

## What this is, and what it is NOT a replacement for

The runbook's seed-string technique is: generate `openssl rand -hex 8`, hand it to the model, and
ask it to derive a bold direction from "patterns" in the string without revealing which. A hex
string carries no design information, so that seed→design mapping is invented by the model. What
the technique actually supplies is two things: a **boldness license** (the framing suppresses the
ask-permission, safe-default mode) and a **blinding commitment device** (the human cannot steer the
model back toward comfort, because the human does not know what was drawn).

This script is that technique **with the confabulation step removed**, not a rejection of it. OS
entropy maps into decisions through fixed tables and numeric ranges, so the entropy reaches the
output instead of being laundered into a post-hoc story. The boldness license is kept — it is real
and it costs nothing.

## What the first version of this file got wrong

Both errors are recorded because both are easy to make again.

**The burn was theatre.** It discarded the first three HANDS and claimed that removed the trained
average. `secrets.choice` draws are i.i.d.: discarding the first three leaves every remaining hand
identically distributed. It changed nothing at all, while reading as rigor. The runbook's "reject
the first three instincts" is about the model's first three *interpretations* — the trained average
lives in the execution layer, not in the deck. Hand-burning is gone; the burn now belongs to the
execution step and is enforced by `EXECUTION_BURN` below, which the author must actually perform.

**The deck was the author's taste one level up.** Every cross-domain card was austere
analog-institutional print — Swiss grid, museum signage, brutalist zine, instrument panel. Nothing
maximal, saturated, soft, playful or digital-native could be dealt, which is the current
model-favoured aesthetic AND what the blind critic's own rubric rewards. A generator tuned to
produce what its judge already likes is self-agreement wearing the costume of validation. The
worlds below are deliberately spread across that axis.

**And the values were still being chosen.** Under five different hands the author picked
`--lp-radius: 0rem` five times out of five, though only one hand dealt the square card.
Concept-level cards constrain the brief while every free variable snaps back to house taste. So the
numbers are dealt too: hue angles, ground lightness, radius base, density and motion all come from
`secrets`, and the author's job is to harmonise within them, not to pick them.

## The hand

* STRUCTURE  — what the layout may and may not use (dealt WITHOUT replacement across candidates,
               because with-replacement dealing gave three of five hands the same defining move and
               produced one direction in three trim levels)
* COLOUR     — what the palette may and may not do
* POSTURE    — a stance, paired with dealt numbers rather than adjectives
* TYPE       — mandatory. Font tokens are as value-open as the colours, and a direction that does
               not touch type is a recolour. Three hands of "type carries ALL hierarchy" were
               previously rendered in Inter at shipping weights.
* WORLD      — a visual world from outside app design
* NUMBERS    — dealt, not chosen

Usage::

    python3 scripts/tools/deal_design_hands.py            # 5 candidates, dark grounds
    python3 scripts/tools/deal_design_hands.py 3 --light  # allow light grounds

`--light` exists only for exploration: the do-not-break list is dark-primary, so a light direction
cannot ship from this exercise and is excluded from the contact sheet by default.
"""

from __future__ import annotations

import secrets
import sys
from datetime import datetime, timezone

STRUCTURAL = [
    "Type carries ALL hierarchy — no rules, dividers, borders or boxes anywhere.",
    "Asymmetric thirds: nothing is centred, nothing is full-width.",
    "Every surface is a card with a visible edge; no bare content on the ground.",
    "Hierarchy by SIZE only — one weight, one colour for all text.",
    "Content bleeds to the screen edge; margins exist only between things, never around them.",
    "Dense information tables; whitespace is a failure, not a feature.",
    "Each screen has exactly one focal object at twice the size of anything else.",
    "Everything is a stack of full-bleed horizontal bands, edge to edge, no gutters.",
]

CHROMATIC = [
    "One hue only. Everything else is a value of it.",
    "No accent colour at all — the interface is achromatic and content supplies every colour.",
    "Two colours that should clash, used at full strength, nothing in between.",
    "Colour means STATE and nothing else — never decoration, never branding.",
    "A near-monochrome field with exactly one saturated element per screen.",
    "Every colour desaturated to the edge of grey, distinguished by temperature alone.",
    "Loud: three or more saturated hues at once, none of them apologising.",
    "The palette is lifted wholesale from the artwork and nothing is fixed but the ground.",
]

POSTURE = [
    "Square and mechanical.",
    "Generous and unhurried.",
    "Packed and instrument-like.",
    "Soft, rounded, physical.",
    "Hairline everything — the thinnest strokes carry all separation.",
    "Heavy: thick strokes, blunt shapes, nothing delicate.",
]

# Mandatory. Stacks only — no webfont may be added (network cost, and the do-not-break list).
TYPE = [
    "Serif throughout: ui-serif, Georgia, Charter. Editorial, not app.",
    "Monospace throughout: ui-monospace, SF Mono. Everything on a grid of characters.",
    "System grotesque, but ONE weight everywhere — hierarchy may not use boldness.",
    "Display serif for headings against a grotesque body — maximum contrast between the two.",
    "Condensed grotesque: Haettenschweiler, Impact, Oswald fallbacks. Tall, tight, loud.",
    "Rounded humanist: Avenir Next, Nunito fallbacks. Soft terminals, open counters.",
    "Monospace for every NUMBER and label, grotesque for prose — machine data vs human words.",
]

WORLD = [
    # analog / institutional
    "Swiss editorial print — grid, Akzidenz, ruthless alignment.",
    "Museum wall signage — small caps, long measure, quiet authority.",
    "Airport and transit wayfinding — enormous type, arrows, no ambiguity at a glance.",
    "Broadsheet newspaper front page — column rules, decks, hierarchy through type alone.",
    "Scientific instrument panel — legends, tick marks, calibrated readouts.",
    # physical objects
    "Cassette-era hi-fi — engraved labels, physical controls, warm metal.",
    "Vinyl record sleeve — the artwork IS the interface, type sits on top of it.",
    "Paperback pulp covers — flat colour blocks, condensed type, no gradients.",
    # loud / maximal — deliberately present, because the first deck had none of this
    "Gig poster / flyposting — screen-printed, overprinted, misregistered, urgent.",
    "Sports broadcast lower-thirds — bold blocks, live data, aggressive diagonals.",
    "Arcade cabinet art — saturated, high-energy, unashamedly synthetic.",
    "Streetwear lookbook — oversized type, cropped imagery, deliberate negative space.",
    # digital-native
    "Terminal multiplexer — panes, status bars, keybind hints, no chrome.",
    "Early-web fan shrine — dense links, tables, personality over polish.",
    "Broadcast teletext — blocky, limited palette, unapologetically low-fi.",
    # soft / domestic
    "Japanese stationery packaging — restraint, tiny type, enormous negative space.",
    "Children's picture book — flat shapes, warm ground, generous scale.",
    "Seed catalogue — botanical plates, ornate labels, cream stock.",
]

EXECUTION_BURN = 3


def deal_numbers(allow_light: bool) -> dict[str, object]:
    """Dealt, not chosen — see the module docstring on the all-zero-radius convergence."""
    light = allow_light and secrets.randbelow(3) == 0
    return {
        "ground": "light" if light else "dark",
        # Lightness of the canvas, as an L* percentage. Two very different bands.
        "ground_L": secrets.randbelow(12) + 88 if light else secrets.randbelow(11) + 3,
        # Primary hue, and a second hue at a dealt angular distance from it.
        "hue_a": secrets.randbelow(360),
        "hue_offset": secrets.choice([15, 30, 60, 90, 120, 150, 180]),
        # Chroma ceiling: how saturated anything is allowed to be, 0-100.
        "chroma_max": secrets.choice([4, 8, 15, 30, 55, 80, 100]),
        # Posture numbers. `radius_base` is the `rounded` step; the ladder scales from it.
        "radius_base_rem": secrets.choice([0, 0, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0]),
        "density": secrets.choice([0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.45]),
        "motion": secrets.choice([0, 0, 0.4, 0.7, 1.0, 1.5, 2.0]),
    }


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    allow_light = "--light" in sys.argv
    count = int(args[0]) if args else 5

    if count > len(STRUCTURAL):
        print(f"at most {len(STRUCTURAL)} hands (structure is dealt without replacement)", file=sys.stderr)
        return 2

    # Without replacement: with-replacement dealing gave 3/5 hands the same structural card.
    structures = list(STRUCTURAL)
    drawn: list[str] = []
    for _ in range(count):
        drawn.append(structures.pop(secrets.randbelow(len(structures))))

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"# Design hands — dealt {stamp}")
    print("#")
    print(f"# Structure is dealt WITHOUT replacement. Numbers are dealt, not chosen.")
    print(f"# Grounds: {'dark or light' if allow_light else 'dark only (light cannot ship — #1941 scope)'}")
    print("#")
    print(f"# THE BURN IS NOT HERE. Discarding dealt hands would be a no-op: the draws are i.i.d.,")
    print(f"# so the surviving hands are identically distributed. The trained average lives in the")
    print(f"# EXECUTION, so each hand must be interpreted {EXECUTION_BURN} times and the interpretation")
    print(f"# closest to the shipping app — or to an already-written direction — discarded. Record")
    print(f"# what was discarded and why, or the burn is theatre again.")
    print()

    for i, structure in enumerate(drawn, start=1):
        n = deal_numbers(allow_light)
        print(f"## Hand {i}")
        print(f"  structure: {structure}")
        print(f"  colour:    {secrets.choice(CHROMATIC)}")
        print(f"  posture:   {secrets.choice(POSTURE)}")
        print(f"  type:      {secrets.choice(TYPE)}")
        print(f"  world:     {secrets.choice(WORLD)}")
        print(f"  numbers:   ground={n['ground']} L*={n['ground_L']}%  hue={n['hue_a']}deg "
              f"second=+{n['hue_offset']}deg  chroma<={n['chroma_max']}")
        print(f"             radius={n['radius_base_rem']}rem  density={n['density']}  motion={n['motion']}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
