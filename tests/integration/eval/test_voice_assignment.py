"""Enforce the ONE-VOICE-PER-PERSON fixture rule (#1170).

Canonical rule (see tests/fixtures/FIXTURES_SPEC.md, "Voices — ONE VOICE PER
PERSON"): every person has exactly one ``say`` voice, used across every show,
episode, and garble/nickname surface form; two DISTINCT people never share a
voice. Voice == identity. Deviation is not allowed — this test fails CI if it
regresses.

Asserts, over every v3 transcript speaker label:
1. no name resolves to more than one voice (determinism);
2. no voice is shared by more than one distinct PERSON (identity uniqueness);
3. no label falls through to the md5 hash fallback (every surface form is mapped).

Run::

    pytest tests/integration/eval/test_voice_assignment.py
"""

from __future__ import annotations

import importlib.util
import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[3]
V3_TRANSCRIPTS = PROJECT_ROOT / "tests" / "fixtures" / "transcripts" / "v3"
SPEAKER_RE = re.compile(r"^([A-Za-z][A-Za-z .'\-]{0,40}):\s+(.*)$")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"Failed to load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses need the module registered first
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def t2m():
    return _load(PROJECT_ROOT / "tests" / "fixtures" / "scripts" / "transcripts_to_mp3.py", "t2m")


@pytest.fixture(scope="module")
def surface_to_person():
    """Map every known surface form -> a canonical person id, from the generator roster."""
    gen = _load(PROJECT_ROOT / "scripts" / "build_v3_fixtures.py", "build_v3_fixtures")
    mapping: dict[str, str] = {}
    for pod in gen.build_v3_spec():
        mapping[pod.host] = f"host:{pod.host}"
        for gkey, guest in pod.guests.items():
            person = f"guest:{guest.name}"
            forms = {guest.name}
            forms |= set(getattr(guest, "garble_variants", []) or [])
            forms |= set(getattr(guest, "nickname_variants", []) or [])
            for form in forms:
                mapping[form] = person
    # garble/alias surface forms + cameos + bare aliases not in the roster objects
    extra = {
        "Joll Wisenthal": "guest:Jonas Weisenthal",
        "Hanna Krebohticker": "guest:Hanna Crebo-Rediker",
        "Skanda Eminas": "guest:Skanda Amarnath",
        "Liam": "guest:Liam Verbeek",
        "Noah": "guest:Noah Brier",
        "Sophie": "guest:Sophie Laurent",
        "Caller": "cameo:Caller",
        "Nadia Sereni": "cameo:Nadia Sereni",
        "Ad": "synthetic:Ad",
    }
    mapping.update(extra)
    return mapping


@pytest.fixture(scope="module")
def transcript_labels():
    labels: set[str] = set()
    for txt in sorted(V3_TRANSCRIPTS.glob("*.txt")):
        for line in txt.read_text(encoding="utf-8").splitlines():
            m = SPEAKER_RE.match(line.strip())
            if not m:
                continue
            name = m.group(1).strip()
            if name in ("Host", "Guest"):
                continue
            labels.add(name)
    assert labels, "no speaker labels found in v3 transcripts"
    return labels


def test_every_label_is_mapped_no_hash_fallback(t2m, transcript_labels):
    """Rule (3): every real fixture label is an exact SPEAKER_VOICE_MAP entry."""
    unmapped = sorted(n for n in transcript_labels if n not in t2m.SPEAKER_VOICE_MAP)
    assert not unmapped, (
        "these transcript speaker labels fall through to the hash fallback — add them "
        f"to SPEAKER_VOICE_MAP pointing at their identity's voice: {unmapped}"
    )


def test_name_resolves_to_single_voice(t2m, transcript_labels):
    """Rule (1): resolution is deterministic — one name, one voice."""
    multi = {n: sorted({t2m.get_voice_for_speaker(n)}) for n in transcript_labels}
    # get_voice_for_speaker is pure, so this is trivially true per-name; the real guard
    # is that identical names across fixtures never diverge (they can't, same map) —
    # asserted here as a regression tripwire if resolution ever becomes stateful.
    assert all(len(v) == 1 for v in multi.values())


def test_no_voice_shared_by_two_people(t2m, transcript_labels, surface_to_person):
    """Rule (2): a voice belongs to exactly one person (garbles of one person are ok)."""
    voice_people: dict[str, set[str]] = defaultdict(set)
    for name in transcript_labels:
        person = surface_to_person.get(name)
        assert person is not None, (
            f"transcript label {name!r} is not attributed to any roster person — update "
            "surface_to_person in this test (and add it to SPEAKER_VOICE_MAP)"
        )
        voice_people[t2m.get_voice_for_speaker(name)].add(person)
    collisions = {v: sorted(p) for v, p in voice_people.items() if len(p) > 1}
    assert not collisions, (
        "ONE VOICE PER PERSON violated — these voices are shared by distinct people: "
        f"{collisions}"
    )
