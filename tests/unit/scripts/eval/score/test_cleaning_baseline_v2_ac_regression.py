"""AC regression: run the cleaning baseline on committed v2 sources, assert thresholds.

The cleaning baseline is fully deterministic (pure regex over committed text via
`CommercialDetector` — no LLM, no embeddings). That makes it a real candidate
for CI-time AC enforcement, unlike the KG/GIL/CIL baselines that depend on
Gemini.

If this test fails, either v2 sources drifted, the detector regressed, or the
SPONSOR_PATTERNS regex set changed in a way that broke recall.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_PATH = ROOT / "scripts" / "eval" / "score" / "cleaning_baseline_v2.py"
_spec = importlib.util.spec_from_file_location("cleaning_baseline_v2_ac_regression", _PATH)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

pytestmark = [pytest.mark.unit]

V2_SOURCES = ROOT / "data" / "eval" / "sources" / "curated_5feeds_raw_v2"


@pytest.mark.skipif(
    not V2_SOURCES.is_dir(),
    reason="v2 sources not present (acceptable in slimmed checkouts)",
)
def test_cleaning_baseline_meets_ac_targets() -> None:
    """raw_episode_hit_rate > 80% AND cleaned_episode_hit_rate < 5% on v2 sources."""
    txt_files = sorted(V2_SOURCES.rglob("*.txt"))
    assert len(txt_files) >= 5, "expected at least one episode per feed"

    episodes = [_mod.score_episode(t) for t in txt_files]
    agg = _mod.aggregate(episodes)

    raw_rate = agg["raw_episode_hit_rate"]
    cleaned_rate = agg["cleaned_episode_hit_rate"]
    retained = agg["pattern_hits_retained_pct"]

    assert raw_rate > 80.0, (
        f"raw_episode_hit_rate dropped to {raw_rate}% (target >80%) — "
        f"v2 sources may have lost sponsor content"
    )
    assert cleaned_rate < 5.0, (
        f"cleaned_episode_hit_rate climbed to {cleaned_rate}% (target <5%) — "
        f"commercial detector recall regressed on v2"
    )
    assert retained < 5.0, (
        f"pattern_hits_retained_pct climbed to {retained}% — "
        f"check SPONSOR_PATTERNS / boundary logic in cleaning/commercial/"
    )
