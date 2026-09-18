"""Markdown export for P2 Capture highlights + notes (#1115, PRD-040).

Pure rendering — no IO, no FastAPI. The route layer hydrates episode titles and attaches
notes, then hands a list of ``EpisodeHighlights`` here. Markdown is the only export format in
v1 (REMEMBER-half-scope §4).

What an export carries is CONTENT, not app state (operator 2026-09-18). The capture's own words,
who said them, when it was captured, what it is about — those travel. The scheduling machinery
(resurfacing counts, whether the user muted it) stays in the app: it describes how this product
nags you, which means nothing in a vault you may read years from now in another tool.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

#: Capture kinds in the user's words — the same labels the Saved list shows, so an export reads
#: like the screen it came from rather than exposing `span` as a term of art.
KIND_LABELS: dict[str, str] = {
    "moment": "Marked moment",
    "span": "Quote",
    "insight": "Insight",
}


def _timecode(ms: int | None) -> str:
    """Render milliseconds as ``H:MM:SS`` / ``M:SS`` (blank when unknown)."""
    if ms is None:
        return ""
    total = max(0, int(ms) // 1000)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _captured_on(ts: int | None) -> str:
    """The capture date as ``YYYY-MM-DD`` (blank when unknown or unusable).

    ISO and date-only on purpose: the export is read in other tools and possibly other locales, so
    it sorts lexically and carries no timezone claim the stored epoch cannot back up.
    """
    if ts is None:
        return ""
    try:
        return time.strftime("%Y-%m-%d", time.gmtime(int(ts)))
    except (TypeError, ValueError, OSError):
        return ""


@dataclass
class HighlightLine:
    """One highlight to render, with any attached note texts."""

    kind: str
    start_ms: int | None = None
    quote_text: str | None = None
    speaker: str | None = None
    color: str | None = None
    #: Unix seconds the capture was made. WHEN the user caught something is part of the record —
    #: without it a year of captures has no chronology once it leaves the app.
    created_at: int | None = None
    #: Canonical person/topic labels for this capture (from ``graph_refs``). Plain names, not
    #: wikilinks: this is one flat document, so ``[[…]]`` would render as broken links for anyone
    #: who is not in Obsidian — and the Obsidian export exists for exactly that audience.
    entities: list[str] = field(default_factory=list)
    #: Absolute player URL, positioned at this capture's second. The whole point of exporting a
    #: timestamp: from a note in any tool, one click opens the player exactly where the line was
    #: said. A site-relative path would not do it — outside the app there is no origin to resolve
    #: against (operator 2026-09-18).
    jump_url: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class EpisodeHighlights:
    """All of one episode's highlights, grouped under its heading."""

    slug: str
    title: str | None = None
    show: str | None = None
    #: Absolute player URL for the episode itself.
    url: str | None = None
    highlights: list[HighlightLine] = field(default_factory=list)
    #: Notes attached to the EPISODE rather than to a highlight. The export only ever matched notes
    #: by highlight id, so these silently never appeared — while the endpoint described itself as
    #: exporting highlights "with attached notes". An export that quietly drops the user's writing
    #: is worse than one that does not offer it.
    episode_notes: list[str] = field(default_factory=list)


def _body(h: HighlightLine, stamp: str) -> str:
    """The capture's own words, quoted where they are a quotation.

    Empty when there are none — a moment captured before quote text was stored, or a span whose
    text did not survive. The line still carries its kind label, timestamp and speaker, and the
    old placeholder ("Marked moment") now only repeats the label that already leads the line.
    """
    quote = (h.quote_text or "").strip()
    if not quote:
        return stamp.rstrip()
    return f"{stamp}{quote}" if h.kind == "insight" else f'{stamp}"{quote}"'


def render_highlights_markdown(
    episodes: list[EpisodeHighlights], orphan_notes: list[str] | None = None
) -> str:
    """Render grouped highlights as a Markdown document (stable, deterministic).

    ``orphan_notes`` are the user's notes whose target cannot be placed under an episode heading —
    today, notes on a saved insight, whose target id is an insight rather than an episode. They get
    their own trailing section rather than being dropped: the export must not lose writing just
    because this renderer has nowhere tidy to put it.
    """
    lines: list[str] = ["# My Highlights", ""]
    if not episodes and not orphan_notes:
        lines.append("_No highlights captured yet._")
        return "\n".join(lines) + "\n"
    for ep in episodes:
        heading = ep.title or ep.slug
        if ep.show:
            heading = f"{heading} — {ep.show}"
        lines.append(f"## {heading}")
        lines.append(f"<!-- {ep.slug} -->")
        if ep.url:
            lines.append(f"[Open in player]({ep.url})")
        lines.append("")
        for note in ep.episode_notes:
            note_text = note.strip()
            if note_text:
                lines.append(f"- _note on this episode:_ {note_text}")
        for h in ep.highlights:
            tc = _timecode(h.start_ms)
            # The timecode IS the link when we have one — the reader's next move from a quote is
            # almost always "play me that bit", so it goes on the thing they already look at.
            stamp = ""
            if tc:
                stamp = f"[{tc}]({h.jump_url}) " if h.jump_url else f"[{tc}] "
            # Kind leads, as the kicker does on the card: "Marked moment" and "Quote" are different
            # objects, and a reader three years out has no other way to tell them apart.
            label = KIND_LABELS.get(h.kind, h.kind)
            suffix = []
            if h.speaker:
                suffix.append(f"— {h.speaker}")
            if h.color:
                suffix.append(f"_{h.color}_")
            captured = _captured_on(h.created_at)
            if captured:
                suffix.append(f"· captured {captured}")
            tail = (" " + " ".join(suffix)) if suffix else ""
            lines.append(f"- **{label}** {_body(h, stamp)}{tail}".replace("**  ", "** "))
            if h.entities:
                lines.append(f"  - _about:_ {' · '.join(h.entities)}")
            for note in h.notes:
                note_text = note.strip()
                if note_text:
                    lines.append(f"  - _note:_ {note_text}")
        lines.append("")
    kept = [n.strip() for n in (orphan_notes or []) if n.strip()]
    if kept:
        lines.append("## Other notes")
        lines.append("")
        lines.extend(f"- {n}" for n in kept)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
