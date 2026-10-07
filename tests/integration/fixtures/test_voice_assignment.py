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
        # Hand-made non-English shows. `build_v3_spec()` generates the p01-p09 roster only, so
        # people who exist ONLY in a hand-written fixture have to be named here — the same
        # reason the cameos above are. p10 is here too since 2026-10-01: it reused p01's Maya
        # and Liam Verbeek until then, and now has its own Spanish cast like the other four.
        "Lucía Herrera": "host:Lucía Herrera",
        "Javier Benavides": "guest:Javier Benavides",
        "Giulia Ferrara": "host:Giulia Ferrara",
        "Marco Bellini": "guest:Marco Bellini",
        "Camille Dubois": "host:Camille Dubois",
        "Julien Mercier": "guest:Julien Mercier",
        "Katrin Vogel": "host:Katrin Vogel",
        "Stefan Brandt": "guest:Stefan Brandt",
        "Beatriz Antunes": "host:Beatriz Antunes",
        "Rafael Vasconcelos": "guest:Rafael Vasconcelos",
        # The e02/e03 guests, since those ten transcripts became real translations (1538029b4)
        # with their own people — this list was left at the e01 cast.
        "Marta Solís": "guest:Marta Solís",
        "Diego Ferrer": "guest:Diego Ferrer",
        "Chiara Ricci": "guest:Chiara Ricci",
        "Luca Moretti": "guest:Luca Moretti",
        "Élodie Chevalier": "guest:Élodie Chevalier",
        "Mathieu Lefèvre": "guest:Mathieu Lefèvre",
        "Lena Hofmann": "guest:Lena Hofmann",
        "Jonas Richter": "guest:Jonas Richter",
        "Inês Carvalho": "guest:Inês Carvalho",
        "Tiago Moreira": "guest:Tiago Moreira",
    }
    mapping.update(extra)
    return mapping


@pytest.fixture(scope="module")
def labels_by_language(t2m):
    """``{language: {label}}`` — each transcript's labels under ITS OWN declared language.

    The flat ``transcript_labels`` fixture below collapses every transcript together, which was
    fine while every fixture was English. It cannot see the invariant that matters once a voice
    map is per-language: that WITHIN one language no two distinct people share a voice. A
    language-blind check passes trivially — Spanish Maya and English Maya resolve to the same
    voice and are the same person — while saying nothing about whether Spanish Maya and Spanish
    Liam Verbeek collide.
    """
    from collections import defaultdict

    out: dict[str, set[str]] = defaultdict(set)
    for txt in sorted(V3_TRANSCRIPTS.glob("*.txt")):
        raw = txt.read_text(encoding="utf-8")
        language = t2m.transcript_language(raw)
        for line in raw.splitlines():
            m = SPEAKER_RE.match(line.strip())
            if not m:
                continue
            name = m.group(1).strip()
            if name in ("Host", "Guest"):
                continue
            out[language].add(name)
    assert out, "no speaker labels found in v3 transcripts"
    return dict(out)


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
    """Rule (3): every real fixture label resolves from an EXPLICIT map, never the hash fallback.

    Checked against the canonical map AND every per-language map, because that is what
    `get_voice_for_speaker` actually does (language map -> canonical -> first word -> hash).
    While every person appeared in an English show the canonical map alone was a faithful proxy;
    p11..p14 introduced eight people who exist ONLY in a non-English fixture, and against the
    old check they read as unmapped while in fact resolving perfectly.
    """
    explicit = set(t2m.SPEAKER_VOICE_MAP)
    for lang_map in t2m.VOICE_MAPS_BY_LANGUAGE.values():
        explicit |= set(lang_map)
    unmapped = sorted(n for n in transcript_labels if n not in explicit)
    assert not unmapped, (
        "these transcript speaker labels fall through to the hash fallback — add them to "
        f"SPEAKER_VOICE_MAP, or to their language's map, pointing at their voice: {unmapped}"
    )


def test_name_resolves_to_single_voice(t2m, transcript_labels):
    """Rule (1): resolution is deterministic — one name, one voice."""
    multi = {n: sorted({t2m.get_voice_for_speaker(n)}) for n in transcript_labels}
    # get_voice_for_speaker is pure, so this is trivially true per-name; the real guard
    # is that identical names across fixtures never diverge (they can't, same map) —
    # asserted here as a regression tripwire if resolution ever becomes stateful.
    assert all(len(v) == 1 for v in multi.values())


def test_no_voice_shared_by_two_people(t2m, labels_by_language, surface_to_person):
    """Rule (2): a voice belongs to exactly one person (garbles of one person are ok).

    Scoped to ENGLISH labels. This check resolves with no language argument, which is the
    canonical map's domain — and asking it about a person who only ever speaks Italian answers a
    question nobody posed: it would report Giulia Ferrara's ENGLISH voice, which does not exist
    and falls to the hash bucket. The non-English languages are covered, per language and
    including the pitch dimension, by `TestTheRuleHoldsPerLANGUAGE` below.
    """
    voice_people: dict[str, set[str]] = defaultdict(set)
    for name in labels_by_language.get("en", set()):
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


class TestTheRuleHoldsPerLANGUAGE:
    """ONE VOICE PER PERSON, PER LANGUAGE (#2169 / V.6b).

    The map is keyed by (person, language) since the Spanish counterpart fixture landed: the
    canonical map holds each person's ENGLISH voice, and there is no Spanish `Samantha`. The
    original rule's two failure modes are unchanged and both are re-asserted here per language —
    two humans on one voice makes a speaker count unreachable, and one person drifting across
    voices makes identity unstable.
    """

    def test_no_voice_is_shared_within_a_language(self, t2m, labels_by_language, surface_to_person):
        """The invariant the language-blind check cannot see."""
        for language, names in sorted(labels_by_language.items()):
            voice_people: dict[str, set[str]] = defaultdict(set)
            for name in names:
                person = surface_to_person.get(name)
                assert person is not None, f"[{language}] unattributed label {name!r}"
                # The ACOUSTIC identity, not the voice name. Italian and German ship exactly
                # one macOS voice each, so host and guest are the same synthesis at two
                # pitches — `Alice` for both would report a collision that the audio does not
                # have, and a real pyannote run on these files finds three speakers. Same
                # statement the RTTM makes; see `voice_identity` in transcripts_to_mp3.py.
                voice_people[
                    t2m.voice_identity(
                        t2m.get_voice_for_speaker(name, language),
                        t2m.pitch_shift_for(name, language),
                    )
                ].add(person)
            collisions = {v: sorted(p) for v, p in voice_people.items() if len(p) > 1}
            assert (
                not collisions
            ), f"[{language}] ONE VOICE PER PERSON violated — shared voices: {collisions}"

    def test_a_person_has_ONE_voice_per_language(self, t2m, labels_by_language):
        """Resolution is a pure function of (name, language), so a person cannot drift between
        episodes of the same language."""
        for language, names in sorted(labels_by_language.items()):
            for name in sorted(names):
                voices = {t2m.get_voice_for_speaker(name, language) for _ in range(3)}
                assert len(voices) == 1, f"[{language}] {name!r} resolved to {voices}"

    def test_the_spanish_fixture_uses_SPANISH_voices(self, t2m, labels_by_language):
        """The point of the whole extension. Reading Spanish with `Samantha` (en_US) produces
        audio that is neither good Spanish nor good English, and this fixture exists to exercise
        Spanish ASR — so it would test the wrong thing."""
        if "es" not in labels_by_language:
            pytest.skip("no Spanish fixture in this tree")
        spanish_voices = set(t2m.SPANISH_SPEAKER_VOICE_MAP.values())
        for name in sorted(labels_by_language["es"]):
            voice = t2m.get_voice_for_speaker(name, "es")
            if name == "Ad":
                # Zarvox is robotic and locale-free; it is the ad voice in every language.
                assert voice == "Zarvox"
                continue
            assert (
                voice in spanish_voices
            ), f"{name!r} renders Spanish with {voice!r}, which is not a Spanish voice"

    def test_english_resolution_is_UNCHANGED(self, t2m, labels_by_language):
        """The 40-episode English corpus must not have moved. Asserted on the canonical map
        directly, so it fails if the language branch ever shadows the default."""
        for name in sorted(labels_by_language.get("en", ())):
            assert t2m.get_voice_for_speaker(name, "en") == t2m.SPEAKER_VOICE_MAP[name]
            # And the default argument must still mean English.
            assert t2m.get_voice_for_speaker(name) == t2m.SPEAKER_VOICE_MAP[name]

    def test_the_groundtruth_records_the_voice_actually_USED(self, labels_by_language):
        """The sidecar's stated job is to record exactly who sounds like what. It resolved
        language-blind at first, so the Spanish sidecar claimed `Samantha`/`Ralph` while the
        audio had been rendered with `Monica`/`Paulina` — a lie rather than a gap."""
        import json

        gt = V3_TRANSCRIPTS / "p10_e01.groundtruth.json"
        if not gt.is_file():
            pytest.skip("no Spanish fixture in this tree")
        vm = json.loads(gt.read_text(encoding="utf-8"))["voice_map"]
        # Its own Spanish cast since 2026-10-01 — it reused p01's Maya and Liam Verbeek before.
        # `Lucía` is the corpus's first accented speaker name and she is here on purpose: the
        # sidecar's parser was `[A-Za-z]`-only, so she silently stopped being a speaker and this
        # map came back as guest + ad alone, with `expected_diarized_voices: 2` against an RTTM
        # holding 3. An accented name must survive every layer, or "we ingest Spanish" is untrue
        # at the character level.
        assert vm == {
            "Lucía Herrera": "Monica",
            "Javier Benavides": "Paulina",
            "Ad": "Zarvox",
        }, vm


class TestEveryShiftedVoiceIsIntelligible:
    """A pitch factor that separates well can still make the voice unintelligible to ASR (#2187).

    V.6b, the first real ASR run on this audio, read the Italian e01 guest at 92% WER: `Alice@0.4`
    is gibberish to Whisper while `Alice@0.7` is perfect. Every ASR measurement on a destroyed
    voice measures the fixture, not the model. The floor and its measured exceptions live beside
    the table in transcripts_to_mp3.py.
    """

    def test_no_factor_below_the_floor_without_a_measurement(self, t2m):
        below = {
            key: factor
            for key, factor in t2m.VOICE_PITCH_SHIFT.items()
            if factor < t2m.INTELLIGIBLE_MIN_FACTOR and key not in t2m.INTELLIGIBLE_BELOW_FLOOR
        }
        assert not below, (
            f"pitch factors below {t2m.INTELLIGIBLE_MIN_FACTOR} with no intelligibility "
            f"measurement in INTELLIGIBLE_BELOW_FLOOR: {below}"
        )

    def test_every_exception_is_still_in_the_table(self, t2m):
        stale = set(t2m.INTELLIGIBLE_BELOW_FLOOR) - set(t2m.VOICE_PITCH_SHIFT)
        assert (
            not stale
        ), f"INTELLIGIBLE_BELOW_FLOOR names people the table no longer shifts: {stale}"
