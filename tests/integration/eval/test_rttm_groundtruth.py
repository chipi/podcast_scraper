"""Guard the per-turn RTTM diarization ground truth (#1170).

Each v3 fixture (except the ffmpeg-truncated ``_fast`` one) carries a co-located
``<name>.rttm`` emitted by ``transcripts_to_mp3.py --rttm-only`` from the
deterministic ``say`` aiff timeline. It is the time-weighted reference for DER,
complementing the count metric (``expected_diarized_voices``).

Asserts, per v3 fixture:
1. the RTTM exists and its sha256 matches the sidecar's ``rttm_sha256`` (drift guard);
2. the RTTM's distinct speaker labels equal ``expected_diarized_voices`` — the
   time-based reference and the count reference agree on how many voices are present;
3. every turn is well-formed (monotonic onsets, positive durations);
4. ``_fast`` has no RTTM and a null ``rttm_sha256`` (it is not a diarization fixture).

Run::

    pytest tests/integration/eval/test_rttm_groundtruth.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[3]
V3_DIR = PROJECT_ROOT / "tests" / "fixtures" / "transcripts" / "v3"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _rttm_turns(path: Path) -> list[tuple[str, float, float]]:
    """Parse ``(label, onset, duration)`` turns from a NIST RTTM."""
    turns: list[tuple[str, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or parts[0] != "SPEAKER":
            continue
        assert len(parts) >= 8, f"malformed RTTM line in {path.name}: {line!r}"
        turns.append((parts[7], float(parts[3]), float(parts[4])))
    return turns


def _sidecars() -> list[Path]:
    cars = sorted(V3_DIR.glob("*.groundtruth.json"))
    assert cars, "no v3 ground-truth sidecars found"
    return cars


@pytest.mark.parametrize(
    "sidecar", _sidecars(), ids=lambda p: p.name.replace(".groundtruth.json", "")
)
def test_rttm_matches_sidecar(sidecar: Path) -> None:
    gt = json.loads(sidecar.read_text(encoding="utf-8"))
    fixture = gt["fixture"]
    rttm = V3_DIR / f"{fixture}.rttm"

    if fixture.endswith("_fast"):
        assert gt["rttm_sha256"] is None, f"{fixture}: _fast must have null rttm_sha256"
        assert not rttm.exists(), f"{fixture}: _fast must not have an RTTM"
        return

    assert rttm.exists(), f"{fixture}: missing RTTM (run transcripts_to_mp3.py --rttm-only)"
    assert gt["rttm_sha256"] == _sha256(
        rttm
    ), f"{fixture}: rttm_sha256 drift — regenerate the sidecar with make_groundtruth.py"

    turns = _rttm_turns(rttm)
    assert turns, f"{fixture}: RTTM has no turns"

    onset = -1.0
    for label, start, dur in turns:
        assert dur > 0, f"{fixture}: non-positive turn duration for {label!r}"
        assert start >= onset, f"{fixture}: non-monotonic onset at {label!r}"
        onset = start

    distinct = {label for label, _, _ in turns}
    assert len(distinct) == gt["expected_diarized_voices"], (
        f"{fixture}: RTTM has {len(distinct)} distinct voices {sorted(distinct)} but "
        f"expected_diarized_voices={gt['expected_diarized_voices']} — the time-based and "
        "count-based references disagree; reconcile before scoring DER"
    )
