"""Unit tests for `scripts/eval/score/cleaning_baseline_v2.py` (#903 audit follow-up)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

_PATH = ROOT / "scripts" / "eval" / "score" / "cleaning_baseline_v2.py"
_spec = importlib.util.spec_from_file_location("cleaning_baseline_v2_under_test", _PATH)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

pytestmark = [pytest.mark.unit]


def _ep(eid: str, raw: int, cleaned: int, raw_hits: int, cleaned_hits: int) -> dict:
    return {
        "episode_id": eid,
        "transcript_path": f"data/eval/sources/fake/feed-p01/{eid}.txt",
        "raw_chars": raw,
        "cleaned_chars": cleaned,
        "chars_removed": raw - cleaned,
        "chars_removed_pct": round(100 * (raw - cleaned) / raw, 2) if raw else 0,
        "blocks_detected": 3,
        "sponsor_pattern_hits_raw": raw_hits,
        "sponsor_pattern_hits_cleaned": cleaned_hits,
        "sponsor_pattern_hits_per_pattern_raw": {},
        "sponsor_pattern_hits_per_pattern_cleaned": {},
        "has_sponsor_raw": raw_hits > 0,
        "has_sponsor_cleaned": cleaned_hits > 0,
    }


def test_aggregate_passes_ac_when_clean() -> None:
    """All episodes have raw sponsor content; cleaning removes everything → both ACs pass."""
    eps = [_ep(f"p0{i}_e01", 1000, 900, 5, 0) for i in range(1, 6)]
    agg = _mod.aggregate(eps)
    assert agg["raw_episode_hit_rate"] == 100.0
    assert agg["cleaned_episode_hit_rate"] == 0.0
    assert agg["pattern_hits_retained_pct"] == 0.0
    assert agg["ac_pass"]["raw_episode_hit_rate"] is True
    assert agg["ac_pass"]["cleaned_episode_hit_rate"] is True


def test_aggregate_fails_cleaned_ac_when_residue() -> None:
    """4 of 10 episodes retain sponsor content → 40% cleaned hit-rate, fails AC <5%."""
    eps = [_ep(f"p0{i}_e01", 1000, 900, 5, 0) for i in range(1, 7)]
    eps += [_ep(f"p0{i}_e02", 1000, 900, 5, 1) for i in range(1, 5)]
    agg = _mod.aggregate(eps)
    assert agg["raw_episode_hit_rate"] == 100.0
    assert agg["cleaned_episode_hit_rate"] == 40.0
    assert agg["ac_pass"]["raw_episode_hit_rate"] is True
    assert agg["ac_pass"]["cleaned_episode_hit_rate"] is False


def test_aggregate_fails_raw_ac_when_sparse() -> None:
    """Only 50% of episodes carry sponsor content → fails AC >80%."""
    eps = [_ep("a", 1000, 900, 5, 0), _ep("b", 1000, 1000, 0, 0)]
    agg = _mod.aggregate(eps)
    assert agg["raw_episode_hit_rate"] == 50.0
    assert agg["ac_pass"]["raw_episode_hit_rate"] is False


def test_aggregate_empty_returns_empty_dict() -> None:
    assert _mod.aggregate([]) == {}


def test_count_sponsor_hits_picks_known_pattern() -> None:
    text = (
        "Host: Welcome. Today's episode is brought to you by Linear. " "Visit linear.com/podcast."
    )
    hits = _mod.count_sponsor_hits(text)
    assert hits["total"] >= 2
    assert all(isinstance(v, int) and v > 0 for v in hits["per_pattern"].values())


def test_score_episode_round_trip(tmp_path: Path) -> None:
    """End-to-end on a small synthetic transcript with one closing sponsor block."""
    transcript = tmp_path / "p01_e01.txt"
    transcript.write_text(
        "Host: Maya\nGuest: Liam\n\n"
        "Maya: Today's episode is brought to you by Linear. "
        "Get started at linear.com/podcast.\n"
        "Maya: And finally, a big thank you to our partners at Stripe. "
        "Check out stripe.com.\n"
        "Maya: That's it.\n",
        encoding="utf-8",
    )
    ep = _mod.score_episode(transcript)
    assert ep["episode_id"] == "p01_e01"
    assert ep["has_sponsor_raw"] is True
    # Detector should remove at least one block.
    assert ep["blocks_detected"] >= 1
    assert ep["chars_removed"] > 0
