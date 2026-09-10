#!/usr/bin/env python3
"""Find API fields the app receives and never puts on screen (#2004 item 17).

## Why this exists

Four of the first six fixes in #2004 were the same shape: **the data was already there and the UI
discarded it.**

* `connected_at` — on the wire, already the server's sort key, dropped by the row
* `summary_bullets` — served, rendered by the browse card, dropped by the player's summary panel
* the `401` status — returned, converted into an empty list before any caller could see it
* the sticky/unstuck state — knowable, never asked for, so a conditional inset was applied always

None were "build X". Each was one line of rendering against data already in hand, which is exactly
the class of defect that is invisible in review: nothing is broken, nothing errors, a field simply
never completes its journey to a pixel.

## What this does NOT claim

**An unused field is not a bug.** Most of the output is correct by design:

* fields consumed by stores, services or player logic rather than templates (`Quote.char_start`,
  `Highlight.segment_ids`, `AudioSource.*`) — these are inputs to behaviour, not content
* internal identifiers used for keys and lookups
* fields a surface deliberately omits because a rail is not the place for a list

So this prints CANDIDATES for a human to triage, and the allowlist below records the verdicts so a
second run does not re-litigate them. Treating the raw list as a defect inventory would be
manufacturing work; the point is to make the discarded-data class *visible*, not to render every
field the API happens to send.

Usage::

    python3 scripts/tools/audit_unused_api_fields.py
    python3 scripts/tools/audit_unused_api_fields.py --all   # include triaged/known-internal
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

APP = Path("web/learning-player/src")
TYPES = APP / "services" / "types.ts"

#: Fields whose absence from templates has been checked and is CORRECT, with the reason.
#: Keyed by ``Interface.field`` so a rename re-surfaces the question rather than staying suppressed.
TRIAGED_INTERNAL = {
    # Timing / offset data driving behaviour (seeking, highlighting) rather than being shown.
    "Quote.char_start",
    "Quote.char_end",
    "Quote.end_ms",
    "Highlight.end_ms",
    "Highlight.char_start",
    "Highlight.char_end",
    "Highlight.segment_ids",
    "Highlight.source_insight_id",
    "HighlightCreate.end_ms",
    "HighlightCreate.char_start",
    "HighlightCreate.char_end",
    "HighlightCreate.segment_ids",
    "HighlightCreate.source_insight_id",
    "Note.target_id",
    "NoteCreate.target_id",
    # Audio plumbing: consumed by the source resolver + native download path, never shown.
    "AudioSource.mime",
    "AudioSource.media_id",
    "AudioSource.strategy",
    "AudioSource.resolved_url",
    "AudioSource.verified",
    "AudioSource.content_length",
    # Paging / service-layer plumbing.
    "EpisodesPage.page_size",
    "LibraryItem.feed_url",
    "EntitiesResponse.orgs",
    "CommsSchedule.day_of_week",
    "CommsSchedule.hour",
    "CommsSettings.unsubscribe_ref",
    "McpConnectionConfig.authorization_server",
    "McpConnectionConfig.oauth_enabled",
    "YourWeekResponse.period_label",
    "YourWeekResponse.generated_at",
    "SearchHit.supporting_quotes",
    "SearchHit.source_tier",
}


def interfaces(src: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for m in re.finditer(r"export interface (\w+)\s*\{(.*?)\n\}", src, re.S):
        fields = re.findall(r"^\s{2}(\w+)\??:", m.group(2), re.M)
        if fields:
            out[m.group(1)] = fields
    return out


def main() -> int:
    show_all = "--all" in sys.argv
    if not TYPES.exists():
        print(f"FAIL: {TYPES} not found — run from the repo root", file=sys.stderr)
        return 2

    sources = [
        p
        for p in APP.rglob("*")
        if p.suffix in (".ts", ".vue")
        and p.name != "types.ts"
        and ".test." not in p.name
        and "__checks__" not in str(p)
    ]
    everywhere = "\n".join(p.read_text() for p in sources)
    templates = "\n".join(p.read_text() for p in sources if p.suffix == ".vue")

    rows: list[tuple[str, str, str]] = []
    for iface, fields in sorted(interfaces(TYPES.read_text()).items()):
        for f in fields:
            key = f"{iface}.{f}"
            if key in TRIAGED_INTERNAL and not show_all:
                continue
            word = re.compile(r"\b" + re.escape(f) + r"\b")
            if not word.search(everywhere):
                rows.append((iface, f, "never referenced in app code at all"))
            elif not word.search(templates):
                rows.append((iface, f, "referenced in logic, never in a template"))

    print(
        f"untriaged candidates: {len(rows)}  (allowlisted as internal: {len(TRIAGED_INTERNAL)})\n"
    )
    for iface, f, why in rows:
        print(f"  {iface:32s} .{f:26s} {why}")
    if not rows:
        print("  none — every field is either rendered or explicitly triaged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
