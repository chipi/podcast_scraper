"""The governance ratchet: undeclared profile divergence is a build failure.

Three incidents in two days, one class — a Config field hand-authored differently across profile
YAMLs (or default-leaking into some of them) with no registry preset owning the choice:
``gi_value_gate_provider`` (#1874/#1875), ``llm_pipeline_mode`` + ``deepseek_summary_model``
(#1878). Each changed prod behaviour and was discovered only when it misbehaved.

The rule this enforces (operator, 2026-08-30): hand-tuning profiles is legal, but it must be
DECLARED. A field that diverges across profiles is either registry-governed (the drift test owns
it) or signed into ``config/profile-governance-accepted.yaml`` with a reason. The
signing moment is the review moment the incidents never had.

The heavy lifting lives in ``scripts/config/profile_governance_audit.py`` (also a CLI —
``--report`` for the what's-different-what's-the-same view); this test just runs its check mode
so CI ratchets every push.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration]

REPO = Path(__file__).resolve().parents[3]


def test_no_undeclared_profile_divergence() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "config" / "profile_governance_audit.py"),
            "--check",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (
        "profile governance ratchet failed:\n" + proc.stdout[-2000:] + "\n" + proc.stderr[-2000:]
    )
