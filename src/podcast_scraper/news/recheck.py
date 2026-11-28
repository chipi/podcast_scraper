"""Re-checking a written news article for a correction: the normaliser, the compare and hold
rule, and the per-item schedule (CRFC-006 Part B, B.2; W2's public half).

Pure code, no network and no rows: the re-check job (``news.recheck_job``, the database
session's since 2026-11-24) fetches the page the way the ingest does and calls :func:`compare`
and :func:`advance`, and writes the version rows. Nothing here runs while every source's re-check
window is 0.

The normalisation rules and the hold share are not designed yet (CRFC-006 B.2 steps 3-4); what is
here is a proposal, and the drift measurement (W2) is what sets the share.
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Optional, Tuple

from . import extract

#: Proposed: re-check days after the first fetch (CRFC-006 B.2 step 1, PRD D4); the window is the
#: last one (7 days, E05N2:19).
DEFAULT_DAYS: Tuple[int, ...] = (1, 2, 4, 7)
#: Proposed, until the drift measurement sets it: a candidate shorter than this share of the
#: served text (both normalised) is held, not written.
DEFAULT_HOLD_SHARE = 0.6
#: A 404 or 410 on this many checks in a row withdraws the item (proposed, R7).
GONE_CHECKS = 2

SAME = "same"
CHANGED = "changed"
HELD_SHORT = "held_short"
HELD_NOT_FULL = "held_not_full"

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def normalise(text: str) -> str:
    """``text`` as the compare sees it: Unicode NFKC, curly quotes straight, a page's furniture
    paragraphs dropped (``extract.strip_boilerplate``), and whitespace folded inside each
    paragraph. Paragraph breaks are kept: a paragraph added is a change, a line re-wrapped is
    not."""
    folded = unicodedata.normalize("NFKC", text or "").translate(_QUOTES)
    return "\n\n".join(extract.paragraphs(extract.strip_boilerplate(folded)))


def normalised_hash(text: str) -> str:
    """The sha256 of :func:`normalise`, hex."""
    return hashlib.sha256(normalise(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Comparison:
    """What a candidate is against the served text: ``SAME``, ``CHANGED`` (a new version), or held
    (``HELD_SHORT``, ``HELD_NOT_FULL``: the served version stays and the hold is counted)."""

    outcome: str
    served_hash: str
    candidate_hash: str
    length_share: float


def compare(
    served_text: str,
    candidate_text: str,
    *,
    url: str,
    hold_share: float = DEFAULT_HOLD_SHARE,
) -> Comparison:
    """Compare a fresh extraction with the served text (B.2 steps 3-4).

    Equal normalised hashes are ``SAME``. A different candidate is held when it is shorter than
    ``hold_share`` of the served text, or when the served text was a full article and the
    candidate is not (a paywall or an index page in its place); otherwise it is ``CHANGED``.
    Ruling out extractor instability (extract once more and compare the two) is the caller's: it
    has the page."""
    served, candidate = normalise(served_text), normalise(candidate_text)
    served_hash = hashlib.sha256(served.encode("utf-8")).hexdigest()
    candidate_hash = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
    share = len(candidate) / len(served) if served else 1.0
    if served_hash == candidate_hash:
        outcome = SAME
    elif share < hold_share:
        outcome = HELD_SHORT
    elif (
        extract.text_status(served, url) == extract.TEXT_FULL
        and extract.text_status(candidate, url) != extract.TEXT_FULL
    ):
        outcome = HELD_NOT_FULL
    else:
        outcome = CHANGED
    return Comparison(outcome, served_hash, candidate_hash, round(share, 3))


#: What a check found: the page's text compared (``Comparison.outcome``), ``gone`` (404 or 410),
#: or ``failed`` (anything else: a timeout, a 5xx, robots.txt refusing it).
GONE = "gone"
FAILED = "failed"

#: What the job does after a check.
RECORD = "record"  # only the check is recorded
WRITE_VERSION = "write_version"
HOLD = "hold"
WITHDRAW = "withdraw"


@dataclass(frozen=True)
class RecheckState:
    """One item's re-check state (CRFC-006 G7: checks done, next due, last status).
    ``gone_in_a_row`` counts consecutive 404/410 answers; ``done`` is set when the window is over or
    the item is
    withdrawn."""

    first_fetched_at: datetime
    checks_done: int = 0
    next_due: Optional[datetime] = None
    last_status: Optional[str] = None
    gone_in_a_row: int = 0
    done: bool = False
    days: Tuple[int, ...] = field(default=DEFAULT_DAYS)


def start(first_fetched_at: datetime, days: Tuple[int, ...] = DEFAULT_DAYS) -> RecheckState:
    """The state of an item just written: due on the first re-check day; an empty ``days`` (a
    window of 0) is done at once."""
    if not days:
        return RecheckState(first_fetched_at, done=True, days=days)
    return RecheckState(
        first_fetched_at, next_due=first_fetched_at + timedelta(days=days[0]), days=days
    )


def advance(state: RecheckState, status: str) -> Tuple[RecheckState, str]:
    """The state after one check that found ``status`` (a ``Comparison.outcome``, ``GONE`` or
    ``FAILED``), and what the job does. ``GONE_CHECKS`` gone answers in a row withdraw it; a
    failed check is retried on the next day of the schedule like any other. After the last day
    the item is done. A done item is never advanced."""
    if state.done:
        raise ValueError("a done item is not re-checked")
    gone = state.gone_in_a_row + 1 if status == GONE else 0
    if gone >= GONE_CHECKS:
        action = WITHDRAW
    elif status == CHANGED:
        action = WRITE_VERSION
    elif status in (HELD_SHORT, HELD_NOT_FULL):
        action = HOLD
    else:
        action = RECORD
    checks = state.checks_done + 1
    done = action == WITHDRAW or checks >= len(state.days)
    next_due = None if done else state.first_fetched_at + timedelta(days=state.days[checks])
    return (
        replace(
            state,
            checks_done=checks,
            next_due=next_due,
            last_status=status,
            gone_in_a_row=gone,
            done=done,
        ),
        action,
    )
