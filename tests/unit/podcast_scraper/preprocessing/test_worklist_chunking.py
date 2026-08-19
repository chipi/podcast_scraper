"""A work-list should be splittable by COST, because cost is the unit that matters.

A 3-hour episode costs 3.5x a 51-minute one, so "16 episodes" is not a budget — which is how the
2026-08-18 repair was planned and why nobody could say in advance what it would cost. Splitting by
hand is arithmetic an operator should not be doing at 1am, and getting it wrong is what the
per-run cap then refuses: correctly, but only after the dispatch.

The property these tests defend hardest: an episode whose duration is unknown must never be
folded into a priced batch. It cannot be shown to fit, and quietly including it is exactly how a
"$5 batch" becomes a $20 one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.preprocessing.audit import (
    chunk_ids_by_cost,
    episode_durations_seconds,
    write_work_list,
)

pytestmark = [pytest.mark.unit]

RATE = 0.0043  # $/min — the deepgram nova-3 row the 2026-08-18 bill matched
HOUR = 3600.0


def _durations(**kw):
    return {k: float(v) for k, v in kw.items()}


# -- packing ----------------------------------------------------------------------------------


def test_episodes_are_packed_until_the_budget_would_be_exceeded() -> None:
    # 1 hour = 60 * 0.0043 = $0.258 each; $1.00 fits 3 (0.774), not 4 (1.032)
    ids = [f"e{i}" for i in range(10)]
    chunks, unpriced = chunk_ids_by_cost(
        ids, {e: HOUR for e in ids}, budget_usd=1.0, usd_per_minute=RATE
    )
    assert unpriced == []
    assert all(len(c) == 3 for c in chunks[:-1]), [len(c) for c in chunks]
    assert sum(len(c) for c in chunks) == 10, "no episode may be dropped"


def test_every_chunk_actually_fits_its_budget() -> None:
    """The invariant, checked rather than assumed."""
    ids = [f"e{i}" for i in range(40)]
    durations = {e: (i % 7 + 1) * 1200.0 for i, e in enumerate(ids)}  # 20-140 min
    chunks, _ = chunk_ids_by_cost(ids, durations, budget_usd=5.0, usd_per_minute=RATE)
    for chunk in chunks:
        cost = sum(durations[e] for e in chunk) / 60.0 * RATE
        assert cost <= 5.0 + 1e-9, f"chunk costs ${cost:.2f}: {chunk}"


def test_a_single_episode_costlier_than_the_WHOLE_budget_still_gets_emitted() -> None:
    """Refusing to emit it would silently drop work from the list — worse than a fat batch.

    The operator sees a batch whose stated estimate exceeds the budget and decides.
    """
    chunks, unpriced = chunk_ids_by_cost(
        ["huge"], {"huge": 10 * HOUR}, budget_usd=0.50, usd_per_minute=RATE
    )
    assert chunks == [["huge"]]
    assert unpriced == []


def test_UNPRICEABLE_episodes_are_separated_never_folded_into_a_batch() -> None:
    """The one that matters. An episode that cannot be priced cannot be shown to fit."""
    ids = ["a", "no-duration", "b"]
    chunks, unpriced = chunk_ids_by_cost(
        ids, _durations(a=HOUR, b=HOUR), budget_usd=5.0, usd_per_minute=RATE
    )
    assert unpriced == ["no-duration"]
    assert "no-duration" not in [e for c in chunks for e in c]


def test_the_original_order_is_preserved() -> None:
    """The order is the audit's — it groups related episodes. Re-sorting to pack tighter would
    scramble a list a human reads."""
    ids = ["z", "m", "a"]
    chunks, _ = chunk_ids_by_cost(
        ids, {e: 600.0 for e in ids}, budget_usd=100.0, usd_per_minute=RATE
    )
    assert chunks == [["z", "m", "a"]]


def test_an_empty_list_produces_no_chunks() -> None:
    assert chunk_ids_by_cost([], {}, budget_usd=5.0, usd_per_minute=RATE) == ([], [])


def test_the_rate_is_a_parameter_so_nothing_here_names_a_provider() -> None:
    ids = [f"e{i}" for i in range(4)]
    cheap, _ = chunk_ids_by_cost(ids, {e: HOUR for e in ids}, budget_usd=1.0, usd_per_minute=0.001)
    dear, _ = chunk_ids_by_cost(ids, {e: HOUR for e in ids}, budget_usd=1.0, usd_per_minute=0.10)
    assert len(cheap) < len(dear), "a dearer provider must produce more, smaller batches"


# -- reading durations off the corpus -----------------------------------------------------------


def _write_meta(root: Path, run: str, idx: int, episode_id: str, duration) -> None:
    d = root / run / "metadata"
    d.mkdir(parents=True, exist_ok=True)
    ep = {"episode_id": episode_id, "guid": f"g{idx}", "title": f"Episode {idx}"}
    if duration is not None:
        ep["duration_seconds"] = duration
    (d / f"{idx:04d} - Episode {idx}.metadata.json").write_text(
        json.dumps({"episode": ep, "content": {}}), encoding="utf-8"
    )


def test_durations_are_read_from_on_disk_metadata(tmp_path) -> None:
    _write_meta(tmp_path, "run_a", 1, "ep1", 1800)
    _write_meta(tmp_path, "run_a", 2, "ep2", 3600)
    assert episode_durations_seconds(tmp_path) == {"ep1": 1800.0, "ep2": 3600.0}


@pytest.mark.parametrize("bad", [None, 0, -5, "not-a-number"])
def test_a_missing_or_nonsense_duration_is_ABSENT_not_zero(tmp_path, bad) -> None:
    """Absent reads as "unknown" downstream. A zero would make an expensive episode look free."""
    _write_meta(tmp_path, "run_a", 1, "ep1", bad)
    assert "ep1" not in episode_durations_seconds(tmp_path)


def test_an_unreadable_metadata_file_does_not_break_the_scan(tmp_path) -> None:
    _write_meta(tmp_path, "run_a", 1, "ep1", 1800)
    bad = tmp_path / "run_a" / "metadata" / "0002 - broken.metadata.json"
    bad.write_text("{ not json", encoding="utf-8")
    assert episode_durations_seconds(tmp_path) == {"ep1": 1800.0}


# -- the writer -------------------------------------------------------------------------------


def test_without_a_budget_the_behaviour_is_unchanged(tmp_path, monkeypatch) -> None:
    """Chunking is opt-in; every existing caller must be unaffected."""
    monkeypatch.setattr(
        "podcast_scraper.preprocessing.audit.damaged_episode_ids", lambda _root: ["a", "b"]
    )
    dest = tmp_path / "worklist.txt"
    assert write_work_list(tmp_path, dest) == 2
    body = dest.read_text()
    assert "a\nb" in body
    assert not list(tmp_path.glob("worklist.txt.*")), "no chunk files without a budget"


def test_with_a_budget_it_writes_numbered_batches_that_state_their_estimate(
    tmp_path, monkeypatch
) -> None:
    ids = [f"ep{i}" for i in range(6)]
    for i, e in enumerate(ids, start=1):
        _write_meta(tmp_path, "run_a", i, e, HOUR)
    monkeypatch.setattr(
        "podcast_scraper.preprocessing.audit.damaged_episode_ids", lambda _root: ids
    )

    dest = tmp_path / "worklist.txt"
    assert write_work_list(tmp_path, dest, chunk_budget_usd=1.0, usd_per_minute=RATE) == 6

    parts = sorted(tmp_path.glob("worklist.txt.0*"))
    assert len(parts) == 2, [p.name for p in parts]
    first = parts[0].read_text()
    assert "BATCH 1 of 2" in first
    assert "audio-hours" in first and "est. $" in first
    # every listed id appears exactly once across the batches
    listed = [
        ln for p in parts for ln in p.read_text().splitlines() if ln and not ln.startswith("#")
    ]
    assert sorted(listed) == sorted(ids)


def test_unpriceable_episodes_get_their_OWN_file_that_says_so(tmp_path, monkeypatch) -> None:
    _write_meta(tmp_path, "run_a", 1, "priced", HOUR)
    _write_meta(tmp_path, "run_a", 2, "no-duration", None)
    monkeypatch.setattr(
        "podcast_scraper.preprocessing.audit.damaged_episode_ids",
        lambda _root: ["priced", "no-duration"],
    )
    dest = tmp_path / "worklist.txt"
    write_work_list(tmp_path, dest, chunk_budget_usd=5.0, usd_per_minute=RATE)

    unpriced = (tmp_path / "worklist.txt.unpriced").read_text()
    assert "no-duration" in unpriced
    assert "cost UNKNOWN, not zero" in unpriced
    assert "no-duration" not in (tmp_path / "worklist.txt.001").read_text()
