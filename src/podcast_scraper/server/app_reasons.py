"""Why a recommendation is here — the one shape every Discover rail carries (ADR-163, 2026-10-10).

ADR-163 §4: the "why" is EXTRACTIVE — the listener's own capture and a grounded quote with its
time. Never a sentence the transcript cannot back, so never an insight's own text (an LLM-written
paraphrase) standing in for a quote: a recommendation without a quote carries ``quote=None`` and
the client shows no quote, rather than an invented one.

The server sends the PARTS — what kind of reason, its subject, the capture that seeded it, the quote
and where it is said — and the client writes the words, so the reason line is localised and no
sentence is generated anywhere.

Reason kinds (each rail declares one, ADR-163 §3):

* ``rising_topic`` — a topic rising in the listener's scope; ``subject`` is the topic.
* ``most_engaged`` — what listeners finished, saved and listened to most; ``events`` is the count.
* ``captured`` — seeded by something the listener captured; ``capture`` is it, ``subject`` what
  links them (a topic, person or storyline).
* ``thread`` — new in a storyline the listener follows; ``subject`` is the storyline.
* ``person`` — a speaker the listener keeps hearing; ``subject`` is the person.
* ``another_view`` — a different speaker on a topic the listener engages with; ``subject`` is
  the topic.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal, Optional, Sequence

from podcast_scraper.server.schemas import AppQuote

ReasonKind = Literal["rising_topic", "most_engaged", "captured", "thread", "person", "another_view"]
SubjectKind = Literal["topic", "person", "storyline", "show"]
CaptureKind = Literal["highlight", "note", "save"]

#: A card shows the quote, not the passage: longer quotes are cut (the episode has the whole).
QUOTE_MAX_CHARS = 280


@dataclass(frozen=True)
class ReasonQuote:
    """A quote said in an episode, and where — the grounding of a reason."""

    text: str
    episode_slug: str
    speaker: Optional[str] = None
    start_ms: Optional[int] = None
    insight_id: Optional[str] = None


@dataclass(frozen=True)
class ReasonSubject:
    """What links the recommendation to its reason: a topic, person, storyline or show."""

    kind: SubjectKind
    id: str
    label: str


@dataclass(frozen=True)
class ReasonCapture:
    """The listener's own capture that seeded the recommendation."""

    kind: CaptureKind
    episode_slug: str
    text: Optional[str] = None
    start_ms: Optional[int] = None


@dataclass(frozen=True)
class Reason:
    """Why one recommendation is here."""

    kind: ReasonKind
    quote: Optional[ReasonQuote] = None
    subject: Optional[ReasonSubject] = None
    capture: Optional[ReasonCapture] = None
    events: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for the response model (``AppReason``)."""
        return asdict(self)


def cut_quote(text: str) -> str:
    """Whitespace normalised, cut at ``QUOTE_MAX_CHARS`` with an ellipsis."""
    text = " ".join(text.split())
    return text if len(text) <= QUOTE_MAX_CHARS else text[: QUOTE_MAX_CHARS - 1].rstrip() + "…"


def grounded_quote(
    quotes: Sequence[AppQuote],
    episode_slug: str,
    *,
    speaker: Optional[str] = None,
    insight_id: Optional[str] = None,
) -> Optional[ReasonQuote]:
    """The quote to ground a reason on: the first timed quote with text, else the first with text.

    ``None`` when there is no quote at all — the caller must NOT fall back to the insight's own text
    (ADR-163 §4). ``speaker`` overrides the quote's own (a perspective names the person even where
    the quote node does not).
    """
    with_text = [q for q in quotes if (q.text or "").strip()]
    if not with_text:
        return None
    timed = [q for q in with_text if q.start_ms is not None and q.start_ms >= 0]
    q = timed[0] if timed else with_text[0]
    return ReasonQuote(
        text=cut_quote(q.text),
        episode_slug=episode_slug,
        speaker=speaker or q.speaker,
        start_ms=q.start_ms,
        insight_id=insight_id,
    )


#: A quote supports its insight when at least this share of its content words is in the insight.
#: Pleasantries and ad reads attached to an insight ("Thanks, Sam. Great to be back.") share none.
MIN_SUPPORT = 1 / 3

_WORD = re.compile(r"[a-z0-9']+")


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 3}


def quote_support(quote_text: str, insight_text: str) -> float:
    """The share of the quote's content words (longer than three letters) the insight also uses."""
    words = _content_words(quote_text)
    return len(words & _content_words(insight_text)) / len(words) if words else 0.0


def insights_by_episode(insights: Iterable[Any]) -> list[tuple[str, list[Any]]]:
    """``[(episode slug, its insights)]`` in first-seen order; insights with no episode dropped."""
    groups: dict[str, list[Any]] = {}
    for ins in insights:
        slug = str(getattr(ins, "episode_slug", None) or "")
        if slug:
            groups.setdefault(slug, []).append(ins)
    return list(groups.items())


def best_grounded_quote(
    insights: Iterable[Any],
    episode_slug: str,
    *,
    speaker: Optional[str] = None,
    said_by: Optional[str] = None,
) -> Optional[ReasonQuote]:
    """The quote to ground a reason on, chosen across one episode's ``insights``.

    The quote that best backs one of the episode's insights wins, credited to the insight it backs:
    it must share at least ``MIN_SUPPORT`` of its content words with that insight, and among those
    the most shared words wins (a ratio alone would crown "Before we start."); ties go to the
    earlier quote. An insight's
    quote list can carry a greeting or an ad read that says nothing any insight says, and a quote
    can sit under the neighbouring insight rather than the one it supports. With no backing quote
    anywhere, the first insight's ``grounded_quote`` — the choice before support was measured, so
    no card is lost. ``said_by`` keeps only quotes by that speaker; ``speaker`` overrides the name.
    """
    insights = list(insights)
    claims = [(getattr(i, "id", None), getattr(i, "text", "") or "") for i in insights]
    best: Optional[tuple[AppQuote, Optional[str]]] = None
    best_shared = 0
    fallback: Optional[ReasonQuote] = None
    for ins in insights:
        quotes = [
            q
            for q in getattr(ins, "quotes", None) or []
            if (q.text or "").strip() and (said_by is None or q.speaker == said_by)
        ]
        if fallback is None:
            fallback = grounded_quote(
                quotes, episode_slug, speaker=speaker, insight_id=getattr(ins, "id", None)
            )
        for q in quotes:
            words = _content_words(q.text)
            for iid, claim in claims:
                shared = len(words & _content_words(claim))
                if quote_support(q.text, claim) >= MIN_SUPPORT and shared > best_shared:
                    best, best_shared = (q, iid), shared
    if best is None:
        return fallback
    q, iid = best
    return ReasonQuote(
        text=cut_quote(q.text),
        episode_slug=episode_slug,
        speaker=speaker or q.speaker,
        start_ms=q.start_ms,
        insight_id=iid,
    )
