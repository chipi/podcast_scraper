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
) -> list[dict[str, Any]]:
    """Highlights due to resurface, most-overdue first.

    A highlight is due when ``now - last_seen >= ladder[surface_count]``, where ``last_seen`` is the
    last time it was surfaced (or its ``created_at`` if never) and ``surface_count`` is how many
    times it has already been shown (capped at the last ladder step). Paused → nothing is due.
    """
    if paused:
        return []
    scored: list[tuple[int, dict[str, Any]]] = []
    for h in highlights:
        hid = str(h.get("id") or "")
        created = int(h.get("created_at") or 0)
        if not hid or not created:
            continue
        st = state.get(hid, {})
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
    scored.sort(key=lambda pair: pair[0], reverse=True)
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
