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

import html
import os
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


def format_duration(seconds: object) -> str:
    """``416`` -> ``"6 min"``; ``3920`` -> ``"1 h 5 min"``. Blank when unknown.

    Rendered, not raw: ``duration_seconds`` is an integer and a reader wants a length, not a count
    of seconds. The Obsidian frontmatter keeps the integer as well, because frontmatter is queried.
    """
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
        return ""
    total = int(seconds)
    if total <= 0:
        return ""
    h, m = divmod(total // 60, 60)
    if h and m:
        return f"{h} h {m} min"
    if h:
        return f"{h} h"
    return f"{m} min" if m else "under a minute"


def format_note(text: str, created_at: int | None = None, updated_at: int | None = None) -> str:
    """A note with its date: ``(2026-09-13)``, or ``(2026-09-13, edited 2026-09-18)``.

    "Edited" only when it actually differs (operator 2026-09-18). ``updated_at`` equals
    ``created_at`` on a note that was never touched, so printing it unconditionally would label
    every note as edited and make the word meaningless.
    """
    body = text.strip()
    made = captured_on(created_at)
    if not made:
        return body
    edited = captured_on(updated_at)
    when = f"{made}, edited {edited}" if edited and edited != made else made
    return f"({when}) {body}"


def captured_on(ts: int | None) -> str:
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
    #: Pre-formatted note lines (see :func:`format_note`) — text plus when it was written.
    notes: list[str] = field(default_factory=list)


@dataclass
class EpisodeHighlights:
    """All of one episode's highlights, grouped under its heading."""

    slug: str
    title: str | None = None
    show: str | None = None
    #: Absolute player URL for the episode itself.
    url: str | None = None
    #: Episode metadata. Three DISTINCT summary fields, each with its own job in the app, so each
    #: travels (operator 2026-09-18): `summary_title` is a headline and explicitly "not a short
    #: summary" (KnowledgePanel), `summary_text` is the prose the player's Summary button renders,
    #: and `summary_bullets` is the digest that opens the insights panel.
    publish_date: str | None = None
    duration_seconds: int | None = None
    summary_title: str | None = None
    summary_text: str | None = None
    summary_bullets: list[str] = field(default_factory=list)
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
        # Date · length · link on one line: the facts that place an episode, none of them worth a
        # line of their own.
        meta = [x for x in (ep.publish_date, format_duration(ep.duration_seconds)) if x]
        if ep.url:
            meta.append(f"[Open in player]({ep.url})")
        if meta:
            lines.append(" · ".join(meta))
        lines.append("")
        # The three summary fields, in the order the player presents them and each doing its own
        # job — headline, then the prose the Summary button shows, then the digest that opens the
        # insights panel. They are NOT three renderings of one thing: `summary_title` is a headline
        # and, per KnowledgePanel, "is not a short summary".
        if ep.summary_title:
            lines.append(f"**{ep.summary_title.strip()}**")
            lines.append("")
        if ep.summary_text:
            lines.append(ep.summary_text.strip())
            lines.append("")
        bullets = [b.strip() for b in ep.summary_bullets if b and b.strip()]
        if bullets:
            lines.extend(f"- {b}" for b in bullets)
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
            captured = captured_on(h.created_at)
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


# --- printable HTML (the PDF path, operator 2026-09-18) ------------------------------------------
#
# No PDF library. WeasyPrint drags Cairo/Pango into the API image and ReportLab means hand-building
# a layout; the browser already has a good renderer, and "Print -> Save as PDF" is native on every
# platform we ship, including the iOS share sheet. So this emits the same document with a print
# stylesheet and lets the browser convert.

BRAND_NAME = "Close Listening"
BRAND_ACCENT = "#efa843"  # --lp-brand-default; the app's one accent
BRAND_INK = "#080d1b"  # --lp-canvas


def public_origin() -> str:
    """The origin export links point at, e.g. ``https://closelistening.app``.

    Shared by every exporter: the vault writer, the printable capture sheet and the episode
    notes sheet all stamp the same origin, so a reader can tell where a loose PDF came from.

    Configured, NOT derived from the request Host, for a reason specific to this exporter: vault
    note content is HASHED to drive the incremental-export cursor. A host-derived URL would change
    every note's hash the moment the user exported from a different origin (native shell, a tunnel,
    localhost), turning a no-op export into a full rewrite of their vault.

    Links must be ABSOLUTE or they do not work at all: a vault note is read inside Obsidian, where
    ``/episode/x`` resolves against the VAULT, not against any website, and silently dead-ends.
    """
    raw = (os.environ.get("APP_PUBLIC_ORIGIN") or "https://closelistening.app").strip()
    return raw.rstrip("/").splitlines()[0] if raw else "https://closelistening.app"


def brand_header(subtitle: str | None = None) -> str:
    """The masthead band every printed export carries.

    A PDF leaves the app and outlives it — on a desktop, in a vault, attached to an email. Without a
    wordmark it is an anonymous page of quotes and nobody can tell where it came from or go back to
    the source (operator 2026-09-19).
    """
    sub = f'<span class="brandsub">{_e(subtitle)}</span>' if subtitle else ""
    return (
        '<header class="brandbar">'
        f'<span class="brandmark">{_e(BRAND_NAME)}</span>{sub}'
        "</header>"
    )


def brand_footer(origin: str) -> str:
    """Where it came from and how to get back — the thing a shared PDF needs most."""
    return (
        '<footer class="brandfoot">'
        f"Exported from {_e(BRAND_NAME)} · "
        f'<a href="{_e(origin)}">{_e(origin.replace("https://", "").replace("http://", ""))}</a>'
        "</footer>"
    )


_PRINT_CSS = """
  :root { color-scheme: light; }
  body {
    font: 11pt/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Georgia, serif;
    color: #1a1a1a; background: #fff;
    max-width: 44rem; margin: 2rem auto; padding: 0 1.25rem;
  }
  h1 { font-size: 1.9rem; margin: 0 0 .25rem; letter-spacing: -.01em; }
  .sub { color: #666; font-size: .85rem; margin-bottom: 2rem; }
  h2 { font-size: 1.15rem; margin: 2.25rem 0 .2rem; letter-spacing: -.01em; }
  .meta { color: #666; font-size: .8rem; margin: 0 0 .75rem; }
  .meta a { color: #666; }
  .stitle { font-weight: 700; margin: .75rem 0 .35rem; }
  .stext { margin: 0 0 .6rem; }
  ul.bullets { margin: 0 0 1rem; padding-left: 1.1rem; color: #333; }
  ul.bullets li { margin: .2rem 0; }
  .cap { margin: .9rem 0; padding-left: .7rem; border-left: 3px solid #d8d8d8; }
  .cap.amber { border-color: #d99a0b; } .cap.rose { border-color: #d4506e; }
  .cap.sky { border-color: #3d8fd1; }   .cap.emerald { border-color: #2f9e6e; }
  .cap.violet { border-color: #8b5cf6; } .cap.slate { border-color: #64748b; }
  .kind { font-size: .7rem; text-transform: uppercase; letter-spacing: .06em; color: #888; }
  .quote { margin: .15rem 0; }
  .who { color: #555; font-size: .85rem; }
  .about, .note { font-size: .82rem; color: #555; margin: .2rem 0 0 .2rem; }
  .note { color: #333; }
  a { color: #1a1a1a; text-decoration: none; border-bottom: 1px solid #ccc; }

  /* Brand chrome. Deliberately a BAND rather than a dark page: the body stays light because this
     is a document people print, and a navy A4 page is a ruined cartridge. The accent rule and the
     wordmark carry the identity instead. */
  .brandbar {
    display: flex; align-items: baseline; gap: .6rem;
    border-bottom: 2px solid #efa843; padding-bottom: .5rem; margin-bottom: 1.5rem;
  }
  .brandmark {
    font: 600 .72rem/1 ui-monospace, SFMono-Regular, Menlo, monospace;
    letter-spacing: .18em; text-transform: uppercase; color: #080d1b;
  }
  .brandsub { font-size: .72rem; color: #888; letter-spacing: .04em; }
  .brandfoot {
    margin-top: 2.5rem; padding-top: .6rem; border-top: 1px solid #e3e3e3;
    font-size: .72rem; color: #888;
  }
  .brandfoot a { color: #888; border-bottom: none; }

  @media print {
    /* Margins belong to the page box, not the body, or every printed page loses its top margin. */
    @page { margin: 18mm 16mm; }
    body { margin: 0; max-width: none; font-size: 10.5pt; }
    /* An episode's heading must not be the last thing on a page, orphaned from its captures. */
    h2 { break-after: avoid; page-break-after: avoid; }
    .cap { break-inside: avoid; page-break-inside: avoid; }
    /* Print drops the href, so a bare "Open in player" becomes a dead phrase on paper. The URL is
       the whole point of exporting a timestamp, so it is spelled out. */
    .meta a::after { content: " (" attr(href) ")"; font-size: .75em; word-break: break-all; }
    a { border-bottom: none; }
  }
"""


def _e(text: object) -> str:
    """Escape for HTML. Every value here is user- or feed-supplied, so nothing is trusted."""
    return html.escape(str(text or ""), quote=True)


def render_highlights_html(
    episodes: list[EpisodeHighlights], orphan_notes: list[str] | None = None
) -> str:
    """The same export as ``render_highlights_markdown``, styled for printing.

    Built from the identical ``EpisodeHighlights`` structure so the two formats cannot disagree
    about what an export contains.
    """
    out: list[str] = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>My Highlights</title>",
        f"<style>{_PRINT_CSS}</style></head><body>",
        brand_header(),
        "<h1>My Highlights</h1>",
    ]
    if not episodes and not orphan_notes:
        out.append("<p class='sub'>No highlights captured yet.</p>")
        out.append(brand_footer(public_origin()))
        out.append("</body></html>")
        return "\n".join(out)

    total = sum(len(ep.highlights) for ep in episodes)
    out.append(
        f"<p class='sub'>{total} capture{'' if total == 1 else 's'} "
        f"from {len(episodes)} episode{'' if len(episodes) == 1 else 's'}</p>"
    )

    for ep in episodes:
        heading = ep.title or ep.slug
        if ep.show:
            heading = f"{heading} — {ep.show}"
        out.append(f"<h2>{_e(heading)}</h2>")
        meta = [_e(x) for x in (ep.publish_date, format_duration(ep.duration_seconds)) if x]
        if ep.url:
            meta.append(f'<a href="{_e(ep.url)}">Open in player</a>')
        if meta:
            out.append(f"<p class='meta'>{' · '.join(meta)}</p>")
        for note in ep.episode_notes:
            if note.strip():
                out.append(f"<p class='note'>Note on this episode: {_e(note)}</p>")
        if ep.summary_title:
            out.append(f"<p class='stitle'>{_e(ep.summary_title)}</p>")
        if ep.summary_text:
            out.append(f"<p class='stext'>{_e(ep.summary_text)}</p>")
        bullets = [b.strip() for b in ep.summary_bullets if b and b.strip()]
        if bullets:
            out.append("<ul class='bullets'>")
            out += [f"<li>{_e(b)}</li>" for b in bullets]
            out.append("</ul>")

        for h in ep.highlights:
            colour = (h.color or "").strip().lower()
            out.append(f"<div class='cap {_e(colour)}'>")
            label = KIND_LABELS.get(h.kind, h.kind)
            tc = _timecode(h.start_ms)
            head = [f"<span class='kind'>{_e(label)}</span>"]
            if tc:
                head.append(f'<a href="{_e(h.jump_url)}">{_e(tc)}</a>' if h.jump_url else _e(tc))
            captured = captured_on(h.created_at)
            if captured:
                head.append(f"<span class='who'>captured {_e(captured)}</span>")
            out.append(" · ".join(head))
            quote = (h.quote_text or "").strip()
            if quote:
                body = _e(quote) if h.kind == "insight" else f"&ldquo;{_e(quote)}&rdquo;"
                out.append(f"<p class='quote'>{body}</p>")
            if h.speaker:
                out.append(f"<p class='who'>— {_e(h.speaker)}</p>")
            if h.entities:
                out.append(f"<p class='about'>{_e(' · '.join(h.entities))}</p>")
            for note in h.notes:
                if note.strip():
                    out.append(f"<p class='note'>{_e(note)}</p>")
            out.append("</div>")

    kept = [n.strip() for n in (orphan_notes or []) if n.strip()]
    if kept:
        out.append("<h2>Other notes</h2>")
        out += [f"<p class='note'>{_e(n)}</p>" for n in kept]
    out.append(brand_footer(public_origin()))
    out.append("</body></html>")
    return "\n".join(out)
