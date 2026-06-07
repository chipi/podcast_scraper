"""Unit tests for `scripts/eval/data/generate_v2_transcripts.py` (#903 audit follow-up).

Two load-bearing behaviours:

1. ``_stable_seed`` is PYTHONHASHSEED-independent — regenerating fixtures must
   be deterministic across runs (the bug that #903 cascade exposed).
2. ``render_episode`` emits *every* callback in the ``callbacks`` list
   (previous behaviour used ``rng.choice`` and silently dropped extras — that's
   how the Marco Bianchi callback in p05_e02 would get lost).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_PATH = ROOT / "scripts" / "eval" / "data" / "generate_v2_transcripts.py"
_MOD_NAME = "generate_v2_transcripts_under_test"
_spec = importlib.util.spec_from_file_location(_MOD_NAME, _PATH)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
# Register before exec so dataclasses can resolve cls.__module__ via sys.modules.
sys.modules[_MOD_NAME] = _mod
_spec.loader.exec_module(_mod)

pytestmark = [pytest.mark.unit]


def test_stable_seed_deterministic_for_same_input() -> None:
    assert _mod._stable_seed("p01:e01") == _mod._stable_seed("p01:e01")
    assert _mod._stable_seed("p05:e02") == _mod._stable_seed("p05:e02")


def test_stable_seed_differs_for_different_inputs() -> None:
    assert _mod._stable_seed("p01:e01") != _mod._stable_seed("p01:e02")


def test_stable_seed_fits_32_bit() -> None:
    """random.Random ignores >32-bit bits — keep us inside the documented range."""
    for s in ("p01:e01", "p99:e99", "anything"):
        assert 0 <= _mod._stable_seed(s) < 2**32


_SUBPROCESS_PROLOGUE = (
    "import importlib.util, sys; "
    "from pathlib import Path; "
    f"spec = importlib.util.spec_from_file_location('m', {str(_PATH)!r}); "
    "m = importlib.util.module_from_spec(spec); "
    "sys.modules['m'] = m; "  # dataclasses resolve cls.__module__ via sys.modules
    "spec.loader.exec_module(m); "
)


def test_stable_seed_independent_of_pythonhashseed() -> None:
    """Re-import the module under different PYTHONHASHSEED values; same seed."""
    code = _SUBPROCESS_PROLOGUE + "print(m._stable_seed('p01:e01'))"
    env_a = {**os.environ, "PYTHONHASHSEED": "0"}
    env_b = {**os.environ, "PYTHONHASHSEED": "999"}
    a = subprocess.check_output([sys.executable, "-c", code], env=env_a, text=True).strip()
    b = subprocess.check_output([sys.executable, "-c", code], env=env_b, text=True).strip()
    assert a == b
    assert a.isdigit()


def test_render_episode_emits_every_callback() -> None:
    """The Marco Bianchi p05_e02 callback only reaches the transcript if every
    callback in the list renders — `rng.choice` would have silently dropped
    one of the two (Daniel + Marco Bianchi) per p05_e02's spec."""
    podcast = _mod.Podcast(
        pod_id="p99",
        title="Test",
        domain="testing",
        host="Maya",
        guests={"Liam": _mod.Guest("Liam", "test guest", "testing")},
        recurring_orgs=["Linear", "Stripe", "Sentry"],
        episodes=[],
    )
    ep = _mod.Episode(
        ep_id="e01",
        title="Callback Test",
        primary_guest="Liam",
        primary_topic="topic:testing",
        secondary_topics=["topic:reliability"],
        sponsor_brands=["Linear", "Stripe", "Sentry"],
        talking_points=["Trail planning matters."],
        callbacks=[
            "Sentinel callback ALPHA marker.",
            "Sentinel callback BETA marker.",
            "Sentinel callback GAMMA marker.",
        ],
    )
    text = _mod.render_episode(podcast, ep)
    for sentinel in ("ALPHA", "BETA", "GAMMA"):
        assert sentinel in text, f"callback {sentinel} missing from rendered transcript"


def test_render_episode_deterministic_across_pythonhashseed(tmp_path: Path) -> None:
    """Same spec under different PYTHONHASHSEED must produce identical bytes."""
    code = _SUBPROCESS_PROLOGUE + (
        "p = m.Podcast(pod_id='p99', title='T', domain='d', host='Maya', "
        "guests={'Liam': m.Guest('Liam', 'r', 'e')}, "
        "recurring_orgs=['Linear','Stripe','Sentry'], episodes=[]); "
        "ep = m.Episode(ep_id='e01', title='T', primary_guest='Liam', "
        "primary_topic='topic:t', secondary_topics=['topic:r'], "
        "sponsor_brands=['Linear','Stripe','Sentry'], "
        "talking_points=['Trail planning matters.', 'Buoyancy first.']); "
        "sys.stdout.write(m.render_episode(p, ep))"
    )
    env_a = {**os.environ, "PYTHONHASHSEED": "0"}
    env_b = {**os.environ, "PYTHONHASHSEED": "999"}
    a = subprocess.check_output([sys.executable, "-c", code], env=env_a, text=True)
    b = subprocess.check_output([sys.executable, "-c", code], env=env_b, text=True)
    assert a == b
    assert len(a) > 100  # actually rendered something
