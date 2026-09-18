"""Spaced resurfacing + interest-profile derivation (P3 Consolidation, #1123 / RFC-101 §5–6).

Pure, read-time logic — no scheduler, no background job (RFC-101 decision 3). The route layer reads
the user's highlights + a small per-user resurfacing state and asks these helpers what is **due**
and what the user is implicitly **interested in**, computed on each request.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

# Interval ladder (seconds) — a highlight resurfaces 2d after capture, then 1w, 1mo, 3mo as it is
# seen and dismissed. The index into the ladder is the number of times it has been surfaced.
DAY = 86_400
LADDER_SECONDS: tuple[int, ...] = (2 * DAY, 7 * DAY, 30 * DAY, 90 * DAY)

# Deterministic reflection prompts (no LLM) — chosen per-highlight by a stable hash.
REFLECTION_PROMPTS: tuple[str, ...] = (
    "What still resonates about this?",
    "How does this connect to something else you've heard?",
    "Would you act on this differently now?",
    "What would you tell someone else about this?",
    "Is this still true in your experience?",
)


def reflection_prompt(highlight_id: str) -> str:
    """A stable reflection prompt for a highlight (deterministic; same id → same prompt)."""
    idx = sum(ord(c) for c in highlight_id) % len(REFLECTION_PROMPTS)
    return REFLECTION_PROMPTS[idx]


def select_due(
    highlights: Iterable[dict[str, Any]],
    state: dict[str, dict[str, Any]],
    now: int,
    *,
    ladder: tuple[int, ...] = LADDER_SECONDS,
    paused: bool = False,
    listened_at: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Highlights due to resurface, grouped by episode, most recently engaged episode first.

    A highlight is due when ``now - last_seen >= ladder[surface_count]``, where ``last_seen`` is the
    last time it was surfaced (or its ``created_at`` if never) and ``surface_count`` is how many
    times it has already been shown (capped at the last ladder step). Paused → nothing is due.

    **Ordering: episodes by ``max(listened_at, newest capture)``, newest first** (operator
    2026-09-18). This replaced most-overdue-first, which sounded right and measured worst.

    Why the obvious order was wrong. An unreviewed capture never moves ``last_seen`` off its
    capture date, so it grows more overdue for ever, and an old capture sits on the 90-day rung —
    it comes back fast and re-occupies the top. Sorting by overdue-ness therefore spent every
    session on the same ancient set while new captures queued behind it: the surface RECIRCULATED
    its oldest items instead of draining. Reviewing a FRESH capture instead advances it
    2d → 7d → 30d → 90d, so it leaves for months.

    Simulated over a year of 2 captures/day for a user who opens the tab weekly and answers ten:

    ===============================  ========  ====================
    ordering                         coverage  median age at review
    ===============================  ========  ====================
    most-overdue-first (was)              44%                118 days
    episode by max(listened, capture)     71%                  4 days
    ===============================  ========  ====================

    Coverage is the share of captures surfaced even once in the year; the old order never showed
    the user 406 of their 730 captures.

    ``listened_at`` is ``playback[slug].updated_at`` — when the user last PLAYED that episode, not
    when it was published. Publishing is not listening: keying on publish date collapsed back to
    49% for a listener whose diet is half back-catalogue, because an old episode played today sank
    to the bottom. Re-listening without capturing also counts, since replaying something is renewed
    interest even when it produces no new capture.

    Grouping is by episode because captures are made while listening, so an episode's captures form
    one session's thinking and are worth meeting together.
    """
    if paused:
        return []
    rows = list(highlights)
    # The episode's recency: its newest capture, or when it was last played, whichever is later.
    # Computed over ALL captures rather than only the due ones, so an episode does not jump around
    # as individual captures fall due.
    engaged: dict[str, int] = dict(listened_at or {})
    for h in rows:
        slug = str(h.get("episode_slug") or "")
        try:
            made = int(h.get("created_at") or 0)
        except (TypeError, ValueError):
            made = 0
        if slug and made > engaged.get(slug, 0):
            engaged[slug] = made

    scored: list[tuple[int, dict[str, Any]]] = []
    for h in rows:
        hid = str(h.get("id") or "")
        created = int(h.get("created_at") or 0)
        if not hid or not created:
            continue
        st = state.get(hid, {})
        # RETIRED: kept, but never asked about again (operator 2026-09-18).
        #
        # The ladder had no exit. Reviewing advances a rung and tops out at 90 days, so a
        # capture you have answered five times still returns every quarter; ignoring one
        # leaves `last_seen` at its capture date, so it stays permanently overdue and —
        # since this sorts most-overdue-first — climbs to the TOP for ever. Both paths
        # loop, and the only way out was deleting the capture, which is a different
        # decision: "stop asking me" is not "I no longer want this".
        #
        # Checked before any date maths: a retired highlight is not due, however overdue it looks.
        if isinstance(st, dict) and st.get("retired"):
            continue
        # Defensive: `state` comes off disk, so it may be hand-edited, half-written, or left by an
        # older build. `mark_surfaced` clamps what IT writes, but this is the function that READS,
        # and it trusted the value outright — a non-numeric count raised ValueError and 500'd both
        # /resurfacing and /your-week, while a NEGATIVE one indexed the ladder from the END (Python
        # negative indexing) and silently scheduled on the wrong rung. The quiet wrong answer is
        # the worse of the two: nothing anywhere would have reported it.
        if not isinstance(st, dict):
            st = {}
        try:
            count = max(0, int(st.get("count", 0)))
        except (TypeError, ValueError):
            count = 0
        try:
            last_seen = int(st.get("last_surfaced") or created)
        except (TypeError, ValueError):
            last_seen = created
        interval = ladder[min(count, len(ladder) - 1)]
        overdue = (now - last_seen) - interval
        if overdue >= 0:
            scored.append((overdue, h))
    # Episode recency first, then newest capture within the episode. `overdue` no longer orders
    # anything — it only decides IS-DUE — but it stays in the tuple as the final tie-break so the
    # sort is total and therefore stable across calls.
    scored.sort(
        key=lambda pair: (
            -engaged.get(str(pair[1].get("episode_slug") or ""), 0),
            -int(pair[1].get("created_at") or 0),
            -pair[0],
        )
    )
    return [h for _, h in scored]


# derive_interest_signals() and _interest_token() lived here until 2026-08-17.
#
# They were the SECOND implementation of "what this user is into". The only one now is
# app_user_corpus.derived_interest_counts(), and the token helper is
# app_user_corpus.interest_token().
#
# Why they went: three surfaces each derived this concept their own way and gave three
# different answers for the same user — /discover over the 40 most recently engaged episodes,
# /corpus over sorted(slugs)[:40] (the alphabetical freeze #18 fixed for /discover ONLY), and
# /interests/derived over every episode with no bound at all. That drift is also what produced
# the doubled `topic:topic:` prefix (d390f7b0). Deleting the duplicate is the fix; leaving an
# unused second definition around is how it came back the first time.
