#!/usr/bin/env python3
"""Seed ONE episode with every kind of saved capture, for a visual sweep of Library -> Saved.

Not a fixture and not wired into any suite: a throwaway used to put every variant of the highlight
card on one screen at once (operator 2026-09-18, "seed more data ... so I can see in a screenshot
how all options look when they're in 1 episode"). Writes straight into the e2e app-state dir the
local api serves from.

Covers, on a single episode so they stack in one group:
  - moment WITH its captured line (what a capture made today looks like)
  - moment WITHOUT one (a record written before the line was stored — the operator's own case)
  - two spans, now labelled "Quote", one of them anchor-drifted so the badge shows
  - two saved insights, which carry a source_insight_id
  - a spread of colours, so the colour control and the filter strip have something to show
  - notes attached to two of them
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SLUG = "p05-ee8e47b94b"
NOW = 1_789_600_000

# The quotes are REAL lines from this episode's transcript, at the timestamps they actually occur.
#
# They have to be. The server re-anchors every highlight on read (`re_anchor_highlight`): it finds
# the segments overlapping the timestamp and searches their text for the stored quote. Invented
# text is not found, so it is reported `drifted` — which is the feature working, and which made a
# first pass at this seed render "⚠ anchor drifted" on cards that were supposed to look ordinary.
#
# One entry keeps a deliberately wrong quote so the drifted badge still has a specimen on screen,
# and one moment carries no quote at all (`time_only`, and the pre-fix "no captured text" case).
HIGHLIGHTS = [
    # kind, start_ms, quote, speaker, colour, insight_id
    (
        "moment",
        26_000,
        "Index funds are not a strategy — they're the absence of one.",
        "Nora",
        "amber",
        None,
    ),
    ("moment", 47_000, None, "Daniel Cho", "rose", None),
    (
        "span",
        19_000,
        "What does the failure mode look like? How does Vanguard fit into this picture?",
        "Daniel Cho",
        "sky",
        None,
    ),
    (
        "span",
        37_000,
        "This line was never spoken in the episode, so it anchors nowhere.",
        "Nora",
        None,
        None,
    ),
    (
        "insight",
        0,
        "Welcome back to Long Horizon Notes. Today we're talking about index investing,"
        " and I'm joined by Daniel Cho, former bond trader turned index advocate.",
        "Nora",
        "emerald",
        "insight:589d7b4ec978e114",
    ),
    (
        "insight",
        19_000,
        "What does the failure mode look like? How does Vanguard fit into this picture?",
        "Daniel Cho",
        "violet",
        "insight:1860b98400777024",
    ),
]

NOTES = {
    "hl-showcase-1": "This is the line I keep coming back to.",
    "hl-showcase-5": "Worth checking against the 2008 data.",
}


def build() -> tuple[list[dict], list[dict]]:
    highlights = []
    for i, (kind, start, quote, speaker, color, insight_id) in enumerate(HIGHLIGHTS, 1):
        highlights.append(
            {
                "id": f"hl-showcase-{i}",
                "episode_slug": SLUG,
                "kind": kind,
                "start_ms": start,
                "end_ms": start + 6_000 if kind == "span" else None,
                "char_start": None,
                "char_end": None,
                "segment_ids": [],
                "quote_text": quote,
                "speaker": speaker,
                "source_insight_id": insight_id,
                "color": color,
                "created_at": NOW - i * 600,
                # Left null on purpose: the server recomputes it on every read, so anything written
                # here is overwritten. The status you see is the real re-anchor verdict.
                "anchor_status": None,
                "graph_refs": [],
            }
        )
    notes = [
        {
            "id": f"note-showcase-{i}",
            "target": "highlight",
            "target_id": hid,
            "text": text,
            "created_at": NOW - i * 300,
            "updated_at": NOW - i * 300,
        }
        for i, (hid, text) in enumerate(NOTES.items(), 1)
    ]
    return highlights, notes


def main() -> None:
    root = (
        Path(__file__).resolve().parents[2]
        / "web"
        / "learning-player"
        / "e2e"
        / ".app-state"
        / "users"
    )
    if not root.is_dir():
        sys.exit(f"FAIL: no app-state users dir at {root} — run the e2e suite once first")
    highlights, notes = build()
    for user_dir in sorted(root.iterdir()):
        if not user_dir.is_dir():
            continue
        (user_dir / "highlights.json").write_text(
            json.dumps(highlights, indent=2), encoding="utf-8"
        )
        (user_dir / "notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
        print(f"seeded {user_dir.name}: {len(highlights)} highlights, {len(notes)} notes")


if __name__ == "__main__":
    main()
