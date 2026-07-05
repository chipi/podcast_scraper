"""Spaced resurfacing + interest-profile derivation (P3 Consolidation, #1123 / RFC-101 §5–6).

Pure, read-time logic — no scheduler, no background job (RFC-101 decision 3). The route layer reads
the user's highlights + a small per-user resurfacing state and asks these helpers what is **due**
and what the user is implicitly **interested in**, computed on each request.
"""

from __future__ import annotations

from collections import Counter
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
        count = int(st.get("count", 0))
        last_seen = int(st.get("last_surfaced") or created)
        interval = ladder[min(count, len(ladder) - 1)]
        overdue = (now - last_seen) - interval
        if overdue >= 0:
            scored.append((overdue, h))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [h for _, h in scored]


def derive_interest_signals(
    entities_per_episode: Iterable[tuple[str, str, str]],
    *,
    min_count: int = 1,
) -> list[dict[str, Any]]:
    """Implicit interest tokens from the user's corpus entities (RFC-101 §6).

    ``entities_per_episode`` yields ``(kind, id, label)`` for each person/topic occurrence across
    the user's heard∪captured episodes (one tuple per episode it appears in). Returns ranked tokens
    ``{token, kind, label, count}`` — ``person:<id>`` / ``topic:<id>`` — by descending occurrence,
    so the same guest/topic across several heard episodes ranks highest. These are *implicit*
    signals, surfaced alongside (never overwriting) the user's explicit follows.
    """
    counts: Counter[tuple[str, str]] = Counter()
    labels: dict[tuple[str, str], str] = {}
    for kind, ent_id, label in entities_per_episode:
        if kind not in ("person", "topic") or not ent_id:
            continue
        key = (kind, str(ent_id))
        counts[key] += 1
        labels.setdefault(key, label or str(ent_id))
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        {"token": f"{kind}:{ent_id}", "kind": kind, "label": labels[(kind, ent_id)], "count": n}
        for (kind, ent_id), n in ranked
        if n >= min_count
    ]
