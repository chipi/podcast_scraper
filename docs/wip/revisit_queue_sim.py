"""Model the Revisit queue at one year of real use (operator's numbers, 2026-09-18).

One episode a day for 365 days, two highlights per episode on the conservative side → 730
captures. The question is what the Revisit tab looks like on day 365 under different review
habits, and whether it needs a cap.

Uses the REAL `select_due`, not a re-implementation of the rules, so the answer moves if the
ladder moves. `GET /resurfacing` applies no limit and does not mark anything surfaced by
rendering it, so "due" here IS what the route returns and what the tab draws.

Run: .venv/bin/python docs/wip/revisit_queue_sim.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from podcast_scraper.server.app_resurfacing import DAY, select_due  # noqa: E402

START = 1_700_000_000
YEAR = 365
PER_DAY = 2


def simulate(reviews_per_session: int | None, session_every_days: int, label: str) -> dict:
    """Walk a year day by day.

    `reviews_per_session=None` means "clear the whole queue"; an int caps how many of the most
    overdue the user gets through. `session_every_days` is how often they open the tab at all.
    """
    highlights: list[dict] = []
    state: dict[str, dict] = {}
    depths: list[int] = []

    for day in range(YEAR):
        now = START + day * DAY
        for i in range(PER_DAY):
            hid = f"h{day}_{i}"
            highlights.append({"id": hid, "created_at": now, "kind": "moment"})

        due = select_due(highlights, state, now)
        depths.append(len(due))

        if day % session_every_days == 0:
            take = due if reviews_per_session is None else due[:reviews_per_session]
            for h in take:
                hid = str(h["id"])
                prev = state.get(hid, {})
                state[hid] = {
                    "last_surfaced": now,
                    "count": int(prev.get("count", 0)) + 1,
                }

    final = select_due(highlights, state, START + YEAR * DAY)
    reviewed = sum(1 for v in state.values() if v.get("count"))
    return {
        "label": label,
        "captures": len(highlights),
        "final_depth": len(final),
        "peak_depth": max(depths),
        "median_depth": sorted(depths)[len(depths) // 2],
        "never_reviewed": len(highlights) - reviewed,
        "backlog_pct": round(100 * len(final) / len(highlights)),
    }


SCENARIOS = [
    (None, 1, "Diligent — opens daily, clears the queue"),
    (10, 1, "Realistic — opens daily, gets through 10"),
    (10, 7, "Weekly — opens Sundays, gets through 10"),
    (5, 7, "Light — opens Sundays, gets through 5"),
    (0, 1, "Lapsed — opens it, reviews nothing"),
]

if __name__ == "__main__":
    print(f"{YEAR} days x {PER_DAY} highlights = {YEAR * PER_DAY} captures\n")
    hdr = (
        f"{'scenario':<44} {'peak':>6} {'median':>7} "
        f"{'day 365':>8} {'never rev.':>11} {'backlog':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for reviews, every, label in SCENARIOS:
        r = simulate(reviews, every, label)
        print(
            f"{r['label']:<44} {r['peak_depth']:>6} {r['median_depth']:>7} "
            f"{r['final_depth']:>8} {r['never_reviewed']:>11} {r['backlog_pct']:>7}%"
        )
