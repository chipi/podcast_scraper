"""Markdown export for P2 Capture highlights + notes (#1115, PRD-040).

Pure rendering — no IO, no FastAPI. The route layer hydrates episode titles and attaches
notes, then hands a list of ``EpisodeHighlights`` here. Markdown is the only export format in
v1 (REMEMBER-half-scope §4).
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _timecode(ms: int | None) -> str:
    """Render milliseconds as ``H:MM:SS`` / ``M:SS`` (blank when unknown)."""
    if ms is None:
        return ""
    total = max(0, int(ms) // 1000)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


@dataclass
class HighlightLine:
    """One highlight to render, with any attached note texts."""

    kind: str
    start_ms: int | None = None
    end_ms: int | None = None
    quote_text: str | None = None
    speaker: str | None = None
    color: str | None = None
    anchor_status: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class EpisodeHighlights:
    """All of one episode's highlights, grouped under its heading."""

    slug: str
    title: str | None = None
    show: str | None = None
    highlights: list[HighlightLine] = field(default_factory=list)


def render_highlights_markdown(episodes: list[EpisodeHighlights]) -> str:
    """Render grouped highlights as a Markdown document (stable, deterministic)."""
    lines: list[str] = ["# My Highlights", ""]
    if not episodes:
        lines.append("_No highlights captured yet._")
        return "\n".join(lines) + "\n"
    for ep in episodes:
        heading = ep.title or ep.slug
        if ep.show:
            heading = f"{heading} — {ep.show}"
        lines.append(f"## {heading}")
        lines.append(f"<!-- {ep.slug} -->")
        lines.append("")
        for h in ep.highlights:
            tc = _timecode(h.start_ms)
            stamp = f"[{tc}] " if tc else ""
            if h.kind == "moment":
                body = f"{stamp}Marked moment".rstrip()
            elif h.kind == "insight":
                body = f"{stamp}{(h.quote_text or 'Saved insight').strip()}"
            else:  # span
                quote = (h.quote_text or "").strip()
                body = f'{stamp}"{quote}"' if quote else f"{stamp}Highlighted span".rstrip()
            suffix = []
            if h.speaker:
                suffix.append(f"— {h.speaker}")
            if h.color:
                suffix.append(f"_{h.color}_")
            if h.anchor_status == "drifted":
                suffix.append("⚠ anchor drifted")
            tail = (" " + " ".join(suffix)) if suffix else ""
            lines.append(f"- {body}{tail}")
            for note in h.notes:
                note_text = note.strip()
                if note_text:
                    lines.append(f"  - _note:_ {note_text}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
