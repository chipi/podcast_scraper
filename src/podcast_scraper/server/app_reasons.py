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

``outside_habits`` marks the serendipity share (ADR-163 §7): an item from outside the listener's
usual shows and topics, which the client labels as such.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional, Sequence

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
    outside_habits: bool = False

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
