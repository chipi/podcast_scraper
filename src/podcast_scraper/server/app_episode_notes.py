"""Episode notes — the whole episode as a document you could print after listening.

A different artifact from the highlights export (``app_capture_export``), and deliberately so. That
one answers "what did I save, across everything"; this one answers "what was this episode, and what
did I take from it" — title, summary, key points, who spoke, everything the episode said, and the
user's own captures and notes on it. It is the export of the insights panel rather than of the
Library (operator 2026-09-18).

Nothing is capped. A long interview can run to many pages, and that is the intent: printed notes
are complete or they are a teaser. Measured on the live corpus at the time of writing, an 80-minute
interview yielded 77 grounded insights and a 20-minute news episode yielded 40, so "many pages" is
the normal case, not the edge one.

Pure rendering plus one builder that reads the corpus. Shares every formatting helper with
``app_capture_export`` so the two documents cannot disagree about how a timecode, a duration, a
date or a note is written.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from podcast_scraper.server.app_capture_export import (
    _e,
    _PRINT_CSS,
    _timecode,
    brand_footer,
    brand_header,
    captured_on,
    format_duration,
    KIND_LABELS,
    public_origin,
)


@dataclass
class NoteQuote:
    """One supporting quote under an insight."""

    text: str
    speaker: str | None = None
    start_ms: int | None = None
    jump_url: str | None = None


@dataclass
class NoteInsight:
    """One grounded insight: the distilled claim, plus the lines that support it.

    The claim and its quotes do different jobs and both travel. On a well-extracted episode the
    claim is a distillation and the quotes are the evidence; where extraction has gone wrong they
    can be the same sentence, which the document shows rather than hides — a printed record is
    also the clearest place to SEE that the pipeline produced something poor.
    """

    text: str
    insight_type: str | None = None
    quotes: list[NoteQuote] = field(default_factory=list)


@dataclass
class NoteCapture:
    """One of the user's own captures on this episode."""

    kind: str
    quote_text: str | None = None
    speaker: str | None = None
    color: str | None = None
    start_ms: int | None = None
    created_at: int | None = None
    jump_url: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class EpisodeNotes:
    """Everything one episode is, as a document."""

    slug: str
    title: str
    show: str | None = None
    publish_date: str | None = None
    duration_seconds: int | None = None
    url: str | None = None
    summary_title: str | None = None
    summary_text: str | None = None
    summary_bullets: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    people: list[str] = field(default_factory=list)
    insights: list[NoteInsight] = field(default_factory=list)
    captures: list[NoteCapture] = field(default_factory=list)
    episode_notes: list[str] = field(default_factory=list)


# --- markdown ------------------------------------------------------------------------------------


def render_episode_notes_markdown(doc: EpisodeNotes) -> str:
    """The episode as Markdown. Section order mirrors the player's insights panel."""
    L: list[str] = [f"# {doc.title}", ""]
    meta = [x for x in (doc.show, doc.publish_date, format_duration(doc.duration_seconds)) if x]
    if meta:
        L.append(" · ".join(meta))
    if doc.url:
        L.append(f"[Listen]({doc.url})")
    L.append("")

    if doc.summary_title or doc.summary_text:
        L += ["## Summary", ""]
        if doc.summary_title:
            L += [f"**{doc.summary_title}**", ""]
        if doc.summary_text:
            L += [doc.summary_text, ""]

    if doc.summary_bullets:
        L += ["## Key points", ""] + [f"- {b}" for b in doc.summary_bullets] + [""]

    if doc.people or doc.topics:
        L += ["## Topics & people", ""]
        if doc.people:
            L += [f"**People** — {' · '.join(doc.people)}", ""]
        if doc.topics:
            L += [f"**Topics** — {' · '.join(doc.topics)}", ""]

    if doc.insights:
        L += ["## What was said", "", f"_{len(doc.insights)} grounded insights._", ""]
        for ins in doc.insights:
            L += [f"**{ins.text}**", ""]
            for q in ins.quotes:
                L.append(f"> {q.text}")
                tail = [x for x in (q.speaker, _stamp_md(q)) if x]
                if tail:
                    L.append(f"> — {', '.join(tail)}")
                L.append("")

    if doc.captures:
        L += ["## What I saved", ""]
        for c in doc.captures:
            label = KIND_LABELS.get(c.kind, c.kind)
            stamp = _stamp_md(c)
            quote = (c.quote_text or "").strip()
            body = f'"{quote}"' if quote and c.kind != "insight" else quote
            bits = [x for x in (stamp, body) if x]
            suffix = []
            if c.speaker:
                suffix.append(f"— {c.speaker}")
            if c.color:
                suffix.append(f"_{c.color}_")
            made = captured_on(c.created_at)
            if made:
                suffix.append(f"· captured {made}")
            line = f"- **{label}** {' '.join(bits)}"
            if suffix:
                line += " " + " ".join(suffix)
            L.append(line.replace("  ", " "))
            for n in c.notes:
                L.append(f"  - _note:_ {n}")
        L.append("")

    if doc.episode_notes:
        L += ["## My notes on this episode", ""] + [f"- {n}" for n in doc.episode_notes] + [""]

    return "\n".join(L).rstrip() + "\n"


def _stamp_md(obj: Any) -> str:
    tc = _timecode(getattr(obj, "start_ms", None))
    if not tc:
        return ""
    url = getattr(obj, "jump_url", None)
    return f"[{tc}]({url})" if url else tc


# --- printable HTML ------------------------------------------------------------------------------


def render_episode_notes_html(doc: EpisodeNotes) -> str:
    """The same document, print-styled — reuses the highlights export's stylesheet verbatim."""
    out: list[str] = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_e(doc.title)}</title>",
        f"<style>{_PRINT_CSS}</style></head><body>",
        brand_header(doc.show),
        f"<h1>{_e(doc.title)}</h1>",
    ]
    meta = [_e(x) for x in (doc.show, doc.publish_date, format_duration(doc.duration_seconds)) if x]
    if doc.url:
        meta.append(f'<a href="{_e(doc.url)}">Listen</a>')
    if meta:
        out.append(f"<p class='meta'>{' · '.join(meta)}</p>")

    if doc.summary_title:
        out.append(f"<p class='stitle'>{_e(doc.summary_title)}</p>")
    if doc.summary_text:
        out.append(f"<p class='stext'>{_e(doc.summary_text)}</p>")
    if doc.summary_bullets:
        out.append("<h2>Key points</h2><ul class='bullets'>")
        out += [f"<li>{_e(b)}</li>" for b in doc.summary_bullets]
        out.append("</ul>")

    if doc.people or doc.topics:
        out.append("<h2>Topics &amp; people</h2>")
        if doc.people:
            out.append(
                f"<p class='about'><strong>People</strong> — {_e(' · '.join(doc.people))}</p>"
            )
        if doc.topics:
            out.append(
                f"<p class='about'><strong>Topics</strong> — {_e(' · '.join(doc.topics))}</p>"
            )

    if doc.insights:
        out.append("<h2>What was said</h2>")
        out.append(f"<p class='sub'>{len(doc.insights)} grounded insights.</p>")
        for ins in doc.insights:
            out.append("<div class='cap'>")
            out.append(f"<p class='quote'><strong>{_e(ins.text)}</strong></p>")
            for q in ins.quotes:
                out.append(f"<p class='quote'>&ldquo;{_e(q.text)}&rdquo;</p>")
                bits = []
                if q.speaker:
                    bits.append(_e(q.speaker))
                tc = _timecode(q.start_ms)
                if tc:
                    bits.append(
                        f'<a href="{_e(q.jump_url)}">{_e(tc)}</a>' if q.jump_url else _e(tc)
                    )
                if bits:
                    out.append(f"<p class='who'>— {' · '.join(bits)}</p>")
            out.append("</div>")

    if doc.captures:
        out.append("<h2>What I saved</h2>")
        for c in doc.captures:
            out.append(f"<div class='cap {_e((c.color or '').lower())}'>")
            head = [f"<span class='kind'>{_e(KIND_LABELS.get(c.kind, c.kind))}</span>"]
            tc = _timecode(c.start_ms)
            if tc:
                head.append(f'<a href="{_e(c.jump_url)}">{_e(tc)}</a>' if c.jump_url else _e(tc))
            made = captured_on(c.created_at)
            if made:
                head.append(f"<span class='who'>captured {_e(made)}</span>")
            out.append(" · ".join(head))
            quote = (c.quote_text or "").strip()
            if quote:
                body = _e(quote) if c.kind == "insight" else f"&ldquo;{_e(quote)}&rdquo;"
                out.append(f"<p class='quote'>{body}</p>")
            if c.speaker:
                out.append(f"<p class='who'>— {_e(c.speaker)}</p>")
            for n in c.notes:
                out.append(f"<p class='note'>{_e(n)}</p>")
            out.append("</div>")

    if doc.episode_notes:
        out.append("<h2>My notes on this episode</h2>")
        out += [f"<p class='note'>{_e(n)}</p>" for n in doc.episode_notes]

    out.append(brand_footer(public_origin()))
    out.append("</body></html>")
    return "\n".join(out)


# --- building it from the corpus ------------------------------------------------------------------


def build_episode_notes(
    root: Path,
    slug: str,
    *,
    row: Any,
    gi_artifact: Any,
    kg_artifact: Any,
    highlights: list[dict[str, Any]],
    notes_by_target: dict[str, list[str]],
    episode_url: Any,
) -> EpisodeNotes:
    """Assemble the document from already-loaded corpus artifacts.

    Takes artifacts rather than loading them so the route owns IO and this stays testable without a
    corpus on disk. Every section is independently optional: an episode with no KG still produces
    notes, exactly as the panel still renders.
    """
    from podcast_scraper.server.app_gi_view import insights_from_gi
    from podcast_scraper.server.app_kg_view import entities_from_kg

    doc = EpisodeNotes(
        slug=slug,
        title=str(getattr(row, "episode_title", "") or slug),
        show=getattr(row, "feed_title", None),
        publish_date=getattr(row, "publish_date", None),
        duration_seconds=getattr(row, "duration_seconds", None),
        url=episode_url(slug),
        summary_title=getattr(row, "summary_title", None),
        summary_text=getattr(row, "summary_text", None),
        summary_bullets=list(getattr(row, "summary_bullets", ()) or ()),
    )

    if kg_artifact is not None:
        persons, _orgs, topics = entities_from_kg(kg_artifact)
        doc.people = [p.name for p in persons if getattr(p, "name", None)]
        doc.topics = [t.label for t in topics if getattr(t, "label", None)]

    # No limit: the operator's call is that printed notes are complete or they are a teaser.
    for ins in insights_from_gi(gi_artifact) if gi_artifact is not None else []:
        doc.insights.append(
            NoteInsight(
                text=ins.text,
                insight_type=ins.insight_type,
                quotes=[
                    NoteQuote(
                        text=q.text,
                        speaker=q.speaker,
                        start_ms=q.start_ms,
                        jump_url=episode_url(slug, q.start_ms),
                    )
                    for q in (ins.quotes or [])
                ],
            )
        )

    for h in highlights:
        hid = str(h.get("id") or "")
        doc.captures.append(
            NoteCapture(
                kind=str(h.get("kind", "span")),
                quote_text=h.get("quote_text"),
                speaker=h.get("speaker"),
                color=h.get("color"),
                start_ms=h.get("start_ms"),
                created_at=h.get("created_at"),
                jump_url=episode_url(slug, h.get("start_ms")),
                notes=list(notes_by_target.get(hid, [])),
            )
        )
    doc.episode_notes = list(notes_by_target.get(slug, []))
    return doc
