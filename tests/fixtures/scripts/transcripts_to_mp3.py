#!/usr/bin/env python3
"""
Generate MP3 files from .txt transcripts using either macOS ``say`` (default,
offline + deterministic) or **Gemini 2.5 multi-speaker TTS** (cloud +
naturalistic, #934).

Per-speaker voice mapping (RFC-059 §2 / issue #111) replaces the prior
binary host/guest scheme so diarization tests (RFC-058) can actually
distinguish speakers in fixture audio.

Backend selection: ``--backend say`` (default, macOS-only) or
``--backend gemini``.

For the ``say`` backend:
    Voice resolution order:
    1. Exact name match in SPEAKER_VOICE_MAP
    2. First-word match (e.g. "Alex Morgan" -> "Alex")
    3. Stable hash-based fallback (md5; deterministic across runs)
    Requires: /usr/bin/say, ffmpeg.

For the ``gemini`` backend:
    Each speaker maps to a prebuilt Gemini voice via
    SPEAKER_GEMINI_VOICE_MAP. Single-call multi-speaker TTS for transcripts
    with ≤ 2 distinct speakers; per-segment fallback for ≥ 3 speakers
    (Gemini multi-speaker mode is capped at 2 voices per call as of 2026).
    Output is non-deterministic — regen drifts byte-by-byte. Requires:
    GEMINI_API_KEY env var, ffmpeg.

Output version is derived from the input transcript's path
(``.../transcripts/v2/...`` -> ``.../audio/v2/...``); explicit
``--output-version`` overrides; otherwise FIXTURES_VERSION is read from
``tests/fixtures/FIXTURES_VERSION``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# Update if you rename podcast IDs or hosts.
PODCAST_HOSTS: dict[str, str] = {
    "p01": "Maya",
    "p02": "Ethan",
    "p03": "Rina",
    "p04": "Leo",
    "p05": "Nora",
    "p06": "TBD",  # Edge cases podcast — v2 may name an actual host
    "p07": "Alex Morgan",  # The Long View - Sustainability
    "p08": "Alex Morgan",  # The Long View - Solar Energy
    "p09": "Alex Morgan",  # The Long View - Biohacking
    # Sesiones de Sendero — the SPANISH counterpart of p01 (#2169 / V.6b). Same conversation,
    # its own people since 2026-10-01 — see SPANISH_SPEAKER_VOICE_MAP for why.
    "p10": "Lucía Herrera",
    # it / fr / de / pt, 2026-10-01. The CONTENT is parallel to p01 and p10 — the same
    # trail-building conversation — so a translation can be compared across five languages
    # against one known meaning. The PEOPLE are not: reusing Maya and Liam would have put one
    # host on seven feeds, and `person:maya` spanning shows is a precision-failure signal the
    # capability audit measures. At three feeds it reads as a finding; at seven it is just noise,
    # and the fixture would be asserting something nobody believes about real podcasts.
    "p11": "Giulia Ferrara",
    "p12": "Camille Dubois",
    "p13": "Katrin Vogel",
    "p14": "Beatriz Antunes",
}

# Per-speaker voice mapping (#1170): ONE voice per PERSON across the whole corpus.
# This is the CANONICAL, NON-NEGOTIABLE rule — see tests/fixtures/FIXTURES_SPEC.md
# ("Voices — ONE VOICE PER PERSON"). Enforced by
# tests/integration/eval/test_voice_assignment.py (deviation fails CI).
# - each show host has a single voice; each guest identity has a single voice, kept
#   across shows (cross-show guests) AND across every garble/nickname surface form;
# - distinct identities NEVER share a voice (incl. same-first-name pairs, e.g. the
#   two Daniels / two Marcos), so voice == identity;
# - accent-matched where a macOS voice exists; en-US overflow spills to other ENGLISH
#   accents (only 9 en_US voices exist); a few non-US guests have no matching-locale
#   voice (Nigerian/Italian/Brazilian) and take the nearest distinct voice.
# Derived deterministically from the scripts/build_v3_fixtures.py roster.
SPEAKER_VOICE_MAP: dict[str, str] = {
    # --- Hosts (one voice per show host) ---
    # Maya @en-US (host)
    "Maya": "Samantha",
    # Ethan @en-US (host)
    "Ethan": "Alex",
    # Rina @en-US (host)
    "Rina": "Allison",
    # Leo @en-GB (host)
    "Leo": "Daniel",
    # Nora @en-US (host)
    "Nora": "Kathy",
    # Cam @en-US (host)
    "Cam": "Fiona",
    # Alex Morgan @en-AU (host)
    "Alex Morgan": "Lee",
    # A. correspondent @en-US (host)
    "A. correspondent": "Karen",
    # Sam @en-US (host)
    "Sam": "Evan",
    # --- Guests (one voice per identity; cross-show guests keep it; garbles fold in) ---
    # Ava Lemoine @fr-CA (guest)
    "Ava Lemoine": "Amélie",
    "Ava Lemonne": "Amélie",
    "Ava Lemoyne": "Amélie",
    # Daniel Cho @en-US (guest)
    "Daniel Cho": "Fred",
    "Daniel Choh": "Fred",
    "Daniel Joh": "Fred",
    # Daniel Olufemi @en-NG (guest)
    "Daniel Olufemi": "Xander",
    "Daniel Olufemy": "Xander",
    "Daniel Olufoemi": "Xander",
    # Dr. Elena Fischer @de-DE (guest)
    "Dr. Elena Fischer": "Anna",
    "Dr. Elena Fischner": "Anna",
    "Dr. Elena Fisher": "Anna",
    # Hanna Crebo-Rediker @en-GB (guest)
    "Hanna Crebo Rediker": "Kate",
    "Hanna Crebo-Rediker": "Kate",
    "Hanna Krebo-Rediker": "Kate",
    "Hanna Krebohticker": "Kate",
    # Jonas Weisenthal @en-US (guest)
    "Joll Wisenthal": "Nathan",
    "Jonas Wassenthal": "Nathan",
    "Jonas Weisenthal": "Nathan",
    "Jonas Wisenthal": "Nathan",
    # Jordan Park @en-US (guest)
    "Jordan Park": "Matilda",
    # Liam Verbeek @en-US (guest)
    "Liam Verbeak": "Ralph",
    "Liam Verbeck": "Ralph",
    "Liam Verbeek": "Ralph",
    # Marco Bianchi @it-IT (guest)
    "Marco Biancchi": "Thomas",
    "Marco Bianchi": "Thomas",
    "Marco Bianci": "Thomas",
    # Marco Silva @pt-BR (guest)
    "Marco Silva": "Jacques",
    "Marco Silvah": "Jacques",
    "Marco Sylva": "Jacques",
    # Noah Brier @en-US (guest)
    "Noah Brier": "Tom",
    "Noah Brier-ah": "Tom",
    "Noah Bryer": "Tom",
    # Priya Nair @en-IN (guest)
    "Priya Naar": "Isha",
    "Priya Nair": "Isha",
    "Priya Nayar": "Isha",
    # Renee Montagne-Park @en-US (guest)
    "Renee Montagne-Park": "Moira",
    "Renee Montague-Park": "Moira",
    # Richard Clarida @en-US (guest)
    "Rich Clarida": "Oliver",
    "Richard Clarida": "Oliver",
    "Richard Claridah": "Oliver",
    # Scott Bessent @en-US (guest)
    "Scott Bessant": "Jamie",
    "Scott Bessent": "Jamie",
    "Scott Bessett": "Jamie",
    # Skanda Amarnath @en-IN (guest)
    "Skanda Amarnath": "Rishi",
    "Skanda Amarnauth": "Rishi",
    "Skanda Eminas": "Rishi",
    # Sophie Laurent @en-US (guest)
    "Sophie Laurent": "Tessa",
    "Sophie Lorent": "Tessa",
    "Sophie Lorenz": "Tessa",
    # Tariq Hassan @ar-EG (guest)
    "Tarek Hassan": "Majed",
    "Tariq Hasaan": "Majed",
    "Tariq Hassan": "Majed",
    # --- Cameos (#1170) ---
    # Caller @ (cameo)
    "Caller": "Veena",
    # Nadia Sereni @ (cameo)
    "Nadia Sereni": "Alice",
    # Bare first-name aliases (standalone smoke fixtures; unambiguous)
    "Liam": "Ralph",
    "Noah": "Tom",
    "Sophie": "Tessa",
    # Synthetic / non-human
    "Ad": "Zarvox",
}

# ONE VOICE PER PERSON, PER LANGUAGE (#2169 / V.6b). The map above is keyed by NAME alone, so
# it cannot express the same person speaking a different language — and `p10` is the Spanish
# counterpart of `p01`: the same show, the same Maya, the same Liam Verbeek.
#
# The original rule ("one voice per PERSON") exists to prevent two failures, both named in
# FIXTURES_SPEC.md: two humans collapsing onto one voice inside an episode, which makes a
# speaker count unreachable, and one person drifting across voices between episodes. Neither is
# about language, and extending the key to (person, language) preserves both — within the
# Spanish episode no two people share a voice, and Maya is Monica in every Spanish episode.
#
# THE ALTERNATIVE WAS WORSE. Reading Spanish text with `Samantha` (en_US) produces audio that is
# neither good Spanish nor good English, and the entire purpose of this fixture is to exercise
# SPANISH ASR — so it would test the wrong thing, and a failure could not be attributed to the
# pipeline rather than to the synthesis.
#
# `Ad` is deliberately absent: `Zarvox` is a robotic mid-roll voice with no locale to match, and
# it stays the ad voice in every language.
#
# LIAM VERBEEK'S VOICE HERE IS A PLACEHOLDER, and the audio is NOT generated from it yet.
# Measured median F0 over the same sentence: Monica 175.8 Hz, Paulina 164.9 Hz — 11 Hz apart with
# standard deviations of 27 and 21, so the distributions overlap almost entirely. The English
# control pair is Samantha 177.8 / Ralph 79.8, i.e. 98 Hz apart. Both default macOS Spanish
# voices are female, so a diarizer would very likely merge them, the episode would have no guest,
# and D-34's payoff (position-bearing insights) would be unmeasurable — failing for a reason that
# has nothing to do with the pipeline. See FIXTURES_SPEC.md; unblocking needs Jorge/Juan/Diego
# installed through System Settings, which has no shell path.
SPANISH_SPEAKER_VOICE_MAP: dict[str, str] = {
    # The show declares <language>es-ES</language>, so the host takes the es_ES voice and the
    # guest the es_MX one.
    #
    # Its OWN people since 2026-10-01, as p11..p14 always had. p10 was first built reusing p01's
    # Maya and Liam Verbeek, which made language the only variable — useful then, wrong to keep:
    # `person:maya` spanning three feeds is a precision-failure signal, and a corpus of English
    # names cannot show how naming handles names belonging to the language being spoken.
    #
    # `Benavides` carries a b AND a v, which are one phoneme in Spanish — the same hazard that
    # turned "Liam Verbeek" into "Liam Berbeek" in a measured ASR run. `Lucía` carries an accent
    # through every layer.
    "Lucía Herrera": "Monica",
    "Javier Benavides": "Paulina",
    # e02 / e03 guests. Same base voice as the e01 guest, but each at its OWN measured pitch
    # factor (VOICE_PITCH_SHIFT): two distinct people never share an acoustic identity, across
    # episodes too (FIXTURES_SPEC.md, #1170). Sharing the e01 guest's exact voice+shift, as the
    # first version of these did, is what test_voice_assignment rejected on main.
    "Marta Solís": "Paulina",
    "Diego Ferrer": "Paulina",
}

#: it / fr / de / pt, 2026-10-01. Measured median F0 for every voice below, each reading its own
#: language, by the same autocorrelation proxy used for Spanish:
#:
#:     it  Alice    181.8 Hz      (the ONLY Italian voice macOS ships)
#:     fr  Amelie   228.6 Hz      fr_CA        fr  Thomas   137.9 Hz   fr_FR
#:     de  Anna     164.9 Hz      (the ONLY German voice)
#:     pt  Luciana  186.0 Hz      pt_BR        pt  Joana    164.9 Hz   pt_PT
#:     (controls: Samantha 177.8, Ralph 85.1, Zarvox 89.9 — the ad voice in every language)
#:
#: FRENCH IS THE ONLY ONE THAT NEEDS NOTHING. Amelie and Thomas are genuinely different voices
#: 90.7 Hz apart, with Thomas a further 48 Hz clear of Zarvox — the English fixture's geometry,
#: for free.
#:
#: Portuguese has two voices but they are 21 Hz apart and both female, so the guest is shifted.
#: Italian and German ship ONE voice each, so host and guest are the same synthesis and the pitch
#: shift is the entire separation — see `VOICE_PITCH_SHIFT`.
ITALIAN_SPEAKER_VOICE_MAP: dict[str, str] = {
    "Giulia Ferrara": "Alice",  # host
    "Marco Bellini": "Alice",  # guest — separated by pitch alone; see VOICE_PITCH_SHIFT
    "Chiara Ricci": "Alice",  # e02 guest
    "Luca Moretti": "Alice",  # e03 guest
}
FRENCH_SPEAKER_VOICE_MAP: dict[str, str] = {
    "Camille Dubois": "Amelie",  # host, fr_CA
    "Julien Mercier": "Thomas",  # guest, fr_FR — a real second voice, no shift needed
    "Élodie Chevalier": "Amelie",  # e02 guest — the host's voice, shifted (VOICE_PITCH_SHIFT)
    "Mathieu Lefèvre": "Thomas",  # e03 guest
}
GERMAN_SPEAKER_VOICE_MAP: dict[str, str] = {
    "Katrin Vogel": "Anna",  # host
    "Stefan Brandt": "Anna",  # guest — separated by pitch alone; see VOICE_PITCH_SHIFT
    "Lena Hofmann": "Anna",  # e02 guest
    "Jonas Richter": "Anna",  # e03 guest
}
PORTUGUESE_SPEAKER_VOICE_MAP: dict[str, str] = {
    "Beatriz Antunes": "Luciana",  # host, pt_BR
    "Rafael Vasconcelos": "Joana",  # guest, pt_PT — female too, so also shifted
    "Inês Carvalho": "Joana",  # e02 guest
    "Tiago Moreira": "Joana",  # e03 guest
}

#: Per-language voice maps, keyed by the primary language subtag. `en` is the default and uses
#: the canonical map above.
VOICE_MAPS_BY_LANGUAGE: dict[str, dict[str, str]] = {
    "es": SPANISH_SPEAKER_VOICE_MAP,
    "it": ITALIAN_SPEAKER_VOICE_MAP,
    "fr": FRENCH_SPEAKER_VOICE_MAP,
    "de": GERMAN_SPEAKER_VOICE_MAP,
    "pt": PORTUGUESE_SPEAKER_VOICE_MAP,
}

#: Post-synthesis pitch shift, per ``(language, speaker)``. A factor below 1.0 lowers the voice.
#:
#: WHY THIS EXISTS, and why it is legitimate rather than a hack. Voices must be far enough apart
#: that a diarizer attributes each passage to the right speaker — the whole point of fixture
#: audio. macOS ships exactly TWO Spanish voices and both are female:
#:
#:     Monica  es_ES  175.8 Hz          <- Maya, the host
#:     Paulina es_MX  164.9 Hz          <- 11 Hz away, sd 27 and 21: the distributions overlap
#:     (English control: Samantha 177.8 / Ralph 79.8 = 98 Hz apart)
#:
#: At 11 Hz a diarizer merges them into one voice, the episode loses its guest, and with it every
#: SPOKEN_BY edge and every position-bearing insight — a fixture that fails for a reason with
#: nothing to do with the pipeline. The additional Spanish voices Apple offers (Jorge, Juan,
#: Diego, all male) install through a path that did not work on this machine: the UI reported
#: success while `mobileassetd` recorded no voice download, every voice-asset directory stayed
#: empty, and both the legacy `say` registry and AVSpeechSynthesizer still saw only two.
#:
#: So the separation is created instead of sourced. Fixture audio is synthetic BY CONSTRUCTION —
#: it is `say` output, and the corpus already tunes its acoustics with `--rate`. Shifting the
#: pitch of synthesised audio is the same kind of knob, and it preserves the Spanish phonemes
#: exactly, because it is the same synthesis resampled rather than a different voice reading
#: Spanish badly.
#:
#: 0.45 is chosen to reproduce the ENGLISH fixture's geometry, measured turn-by-turn from the
#: finished mp3s against each RTTM:
#:
#:     p01_e01 (English)   Samantha 176.4   Ralph 73.0    Zarvox 90.0
#:     p10_e01 (Spanish)   Monica   169.6   shifted 77.1  Zarvox 90.0
#:
#: The first attempt used 0.55, which put the guest at 91.9 Hz — a 78 Hz gain over the host, but
#: **2 Hz from Zarvox**, the ad voice. The English fixture keeps 17 Hz there (Ralph 73 vs Zarvox
#: 90), so 0.55 traded one collision for another. 0.45 lands the guest at 77.1 Hz: ~92 Hz from
#: the host and ~13 Hz from the ad voice, which is the working corpus's own shape.
#:
#: Duration is unchanged (6.87 s -> 6.86 s measured): `atempo` restores the tempo `asetrate`
#: changes. That matters beyond sounding right — the RTTM is built from these aiff durations, so
#: a shift that moved them would make the diarization reference disagree with its own audio.
#:
#: If a real Spanish male voice is ever installed, delete the entry and point
#: ``SPANISH_SPEAKER_VOICE_MAP["Javier Benavides"]`` at it — one line, then regenerate.
#: it / de / pt use 0.40, chosen the same way and from a measured sweep rather than by copying
#: the Spanish 0.45. Each guest, and where it lands against its host and against Zarvox (89.9 Hz,
#: the ad voice in every language):
#:
#:                   base     0.40      0.45      host      -> chosen  gap to host  gap to ad
#:     it Alice     181.8     77.3      86.5     181.8         0.40       104.5        12.6
#:     de Anna      164.9     79.2      84.2     164.9         0.40        85.7        10.7
#:     pt Joana     164.9     78.0      81.4     186.0         0.40       108.0        11.9
#:
#: 0.45 — the Spanish value — is WRONG for all three: it lands them 3.4, 5.7 and 8.5 Hz from
#: Zarvox, reproducing exactly the collision the Spanish comment above records rejecting 0.55
#: for. The right factor is the one that fits the voice, not the one that worked last time.
#:
#: FRENCH e01 IS DELIBERATELY ABSENT. Amelie and Thomas are 90.7 Hz apart on their own, so
#: shifting that pair would be decoration — and a shift that is not needed is a difference between
#: the fixture and real audio for no measured reason. The French e02/e03 guests ARE shifted: with
#: only two voices, a third and fourth person need a shift to be anyone other than Amelie or Thomas.
#:
#: Italian and German are the WEAKEST case in the corpus and worth stating plainly: macOS ships
#: one voice per language, so host and guest are the same synthesis at two pitches, not two
#: voices. Spanish at least had Monica and Paulina. Whether a diarizer separates them is a
#: question for the diarizer, not for this table — measured against the real pyannote service,
#: not assumed.
VOICE_PITCH_SHIFT: dict[tuple[str, str], float] = {
    ("es", "Javier Benavides"): 0.45,
    ("it", "Marco Bellini"): 0.40,
    ("de", "Stefan Brandt"): 0.40,
    ("pt", "Rafael Vasconcelos"): 0.40,
    # The e02 / e03 guests, 2026-10-06: each person its OWN factor, because two distinct people
    # never share an acoustic identity (`voice_identity`), not even in different episodes —
    # the first version reused the e01 guest's factor and broke that rule. Median F0 of the
    # rendered audio (same autocorrelation proxy, each voice reading its own language), with the
    # 25th-75th percentile, against the show's host and Zarvox (~89.5 Hz in every language):
    #
    #   es  host Monica 170 [154-188]  e01 Paulina@0.45 74 [70-80]
    #       Marta  Paulina@0.8  131 [123-143]    Diego  Paulina@0.65 107 [101-116]
    #   it  host Alice  174 [157-195]  e01 Alice@0.4    72 [68-79]
    #       Chiara Alice@0.8    140 [125-155]    Luca   Alice@0.6    105 [95-117]
    #   fr  host Amelie 229 [208-239]  e01 Thomas       133 [118-143]
    #       Élodie Amelie@0.75  170 [155-180]    Mathieu Thomas@0.5  69 [63-73]
    #   de  host Anna   165 [154-178]  e01 Anna@0.4     68 [64-72]
    #       Lena   Anna@0.8     132 [124-143]    Jonas  Anna@0.65    107 [101-115]
    #   pt  host Luciana 191 [178-198] e01 Joana@0.4    69 [65-77]
    #       Inês   Joana@0.85   144 [136-157]    Tiago  Joana@0.65   110 [103-120]
    #
    # Chosen so that every identity in a show has a quartile range clear of every other one's,
    # sits >= 25 Hz from the host and >= 15 Hz from Zarvox (the smallest margin the corpus already
    # accepted: Paulina@0.45 at 74 Hz). Female names took the upper slot, male the lower. NOT yet
    # measured against the real pyannote service.
    ("es", "Marta Solís"): 0.80,
    ("es", "Diego Ferrer"): 0.65,
    ("it", "Chiara Ricci"): 0.80,
    ("it", "Luca Moretti"): 0.60,
    ("fr", "Élodie Chevalier"): 0.75,
    ("fr", "Mathieu Lefèvre"): 0.50,
    ("de", "Lena Hofmann"): 0.80,
    ("de", "Jonas Richter"): 0.65,
    ("pt", "Inês Carvalho"): 0.85,
    ("pt", "Tiago Moreira"): 0.65,
}


def pitch_shift_for(speaker: str, language: str) -> float | None:
    """The post-synthesis pitch factor for this speaker in this language, or ``None``."""
    key = ((language or "en").split("-")[0].lower(), speaker.strip())
    return VOICE_PITCH_SHIFT.get(key)


# Hash-based fallback for speakers not in SPEAKER_VOICE_MAP. Order matters
# for stability — appending is safe; reordering or deleting changes
# previously generated fallback assignments. Picked for clear distinction
# from each other and from the curated voices above.
FALLBACK_VOICES: list[str] = [
    "Albert",
    "Bruce",
    "Junior",
    "Nicky",
    "Shelley",
    "Trinoids",
    "Whisper",
    "Bells",
]

# Gemini 2.5 prebuilt voices per speaker (#934). Voice names are the
# canonical ones the Gemini TTS API recognises in 2026: Kore, Aoede, Puck,
# Charon, Fenrir, Leda, Orus, Zephyr. Each speaker label gets a distinct
# voice for diarization separability — the mapping intentionally varies
# across hosts so the gemini-backed fixtures behave like the say-backed
# ones (different speaker, different voice).
SPEAKER_GEMINI_VOICE_MAP: dict[str, str] = {
    # Hosts
    "Maya": "Kore",  # warm + clear, common host pick
    "Ethan": "Charon",  # lower-pitched male
    "Rina": "Leda",  # bright female
    "Leo": "Fenrir",  # mid male
    "Nora": "Zephyr",  # airy female
    "Alex": "Orus",  # neutral male
    # Guests — picked to contrast against each host they pair with
    "Liam": "Puck",  # higher-energy male (contrast vs Kore on p01)
    "Sophie": "Aoede",
    "Noah": "Charon",
    "Priya": "Leda",
    "Jonas": "Orus",
    "Camila": "Aoede",
    "Marco": "Puck",
    "Hanna": "Leda",
    "Owen": "Fenrir",
    "Ava": "Aoede",
    "Tariq": "Orus",
    "Elise": "Leda",
    "Daniel": "Charon",
    "Isabel": "Aoede",
    "Kasper": "Fenrir",
    # Ad reads — robotic-shaped pick (Charon is the closest to Zarvox-y).
    "Ad": "Charon",
}

GEMINI_FALLBACK_VOICES: list[str] = [
    "Kore",
    "Aoede",
    "Puck",
    "Charon",
    "Fenrir",
    "Leda",
    "Orus",
    "Zephyr",
]

TS_RE = re.compile(r"^\[\s*\d{1,2}:\d{2}(:\d{2})?\s*\]$")
# Names may contain internal periods ("A. correspondent", "Dr. Elena Fischer");
# without '.' those turns fall through to the host voice instead of their own (#1170).
#: ``[^\W\d_]`` is "any Unicode letter" — NOT ``[A-Za-z]``, which was the bug. The class used to
#: be ASCII-only, so ``Marta Solís:`` did not match and the line fell through to
#: ``parse_segments``'s "no Speaker: prefix, attribute it to the host" branch. Every accented
#: guest in the corpus therefore lost their turns to the host: a regenerated ``p10_e02`` came out
#: with 3 RTTM segments where its sibling has 47, one of them 227 seconds long, and the guest
#: voice replaced by the hash-fallback ``Zarvox`` (which is the AD voice). The five ``_e01``
#: episodes escaped only because their GUESTS happen to be pure ASCII — ``Javier Benavides``,
#: ``Marco Bellini`` — while the accented names among them are HOSTS, where falling through to
#: the host voice is a no-op. ``transcripts_to_vtt.py`` already carries a comment about this exact
#: trap; this script never got the fix.
SPEAKER_RE = re.compile(r"^([^\W\d_][\w .'’\-]{0,40}?):\s+(.*)$")
FILE_PREFIX_RE = re.compile(r"^(p\d{2})_", re.IGNORECASE)
VERSION_SEGMENT_RE = re.compile(r"^v\d+$")


def get_voice_for_speaker(name: str, language: str = "en") -> str:
    """Resolve a speaker name to a macOS ``say`` voice.

    Lookup precedence: the language's own map -> exact -> first word -> stable hash fallback.

    ``language`` is the primary subtag of the transcript's ``#fixture-v3: voice=`` annotation.
    A person in a non-English episode resolves through that language's map first, because the
    canonical map holds their ENGLISH voice and there is no Spanish `Samantha` — see
    ``SPANISH_SPEAKER_VOICE_MAP`` for why the key had to become (person, language).

    Falling THROUGH to the canonical map is deliberate rather than an error: ``Ad`` has no
    Spanish entry and should not — ``Zarvox`` is robotic and locale-free, and it is the ad voice
    in every language.
    """
    name = name.strip()
    lang_map = VOICE_MAPS_BY_LANGUAGE.get((language or "en").split("-")[0].lower(), {})
    if name in lang_map:
        return lang_map[name]
    if name in SPEAKER_VOICE_MAP:
        return SPEAKER_VOICE_MAP[name]
    parts = name.split()
    if parts and parts[0] in lang_map:
        return lang_map[parts[0]]
    if parts and parts[0] in SPEAKER_VOICE_MAP:
        return SPEAKER_VOICE_MAP[parts[0]]
    # Stable hash (Python's built-in hash() varies across runs). Not a security
    # boundary — just a deterministic bucket selector for voice assignment.
    digest = hashlib.md5(name.lower().encode("utf-8"), usedforsecurity=False).hexdigest()
    return FALLBACK_VOICES[int(digest, 16) % len(FALLBACK_VOICES)]


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )


def parse_segments(raw: str, host_name: str) -> list[tuple[str, str]]:
    """Parse transcript text into a list of (speaker_name, text) segments.

    Free-form narrative lines (no ``Speaker:`` prefix) are attributed to the
    host. Consecutive segments by the same speaker are merged so we don't
    re-invoke ``say`` for every sentence.
    """
    out: list[tuple[str, str]] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if TS_RE.fullmatch(s):
            continue
        low = s.lower()
        if s.startswith("#") or low.startswith(
            ("podcast:", "episode:", "host:", "guest:", "title:")
        ):
            continue
        if s.startswith("*") and s.endswith("*"):
            continue

        m = SPEAKER_RE.match(s)
        if m:
            speaker = m.group(1).strip()
            text = m.group(2).strip()
            if not text:
                continue
            out.append((speaker, text))
        else:
            out.append((host_name, s))

    # Merge consecutive segments from the same speaker.
    merged: list[tuple[str, str]] = []
    for speaker, text in out:
        if merged and merged[-1][0] == speaker:
            merged[-1] = (speaker, merged[-1][1] + "\n" + text)
        else:
            merged.append((speaker, text))
    return merged


def say_to_aiff(
    text: str,
    out_aiff: Path,
    voice: str,
    rate: int | None,
    pitch: float | None = None,
) -> None:
    cmd = ["say", "-o", str(out_aiff), "-v", voice]
    if rate is not None:
        cmd += ["-r", str(rate)]
    cmd += [text]
    run(cmd)
    if pitch is not None:
        _apply_pitch_shift(out_aiff, pitch)


def _apply_pitch_shift(aiff: Path, factor: float) -> None:
    """Lower (or raise) the pitch of a rendered turn IN PLACE, keeping its duration.

    ``asetrate`` changes pitch and tempo together; ``atempo`` puts the tempo back, so the turn
    lasts as long as it did. That matters beyond sounding right: the RTTM is built from these
    aiff durations, so a shift that changed them would make the diarization reference disagree
    with the audio it describes.

    See ``VOICE_PITCH_SHIFT`` for why a fixture voice is shifted at all.
    """
    import shutil as _shutil

    ff = _shutil.which("ffmpeg") or "/usr/local/bin/ffmpeg"
    tmp = aiff.with_suffix(".shifted.aiff")
    # Read the real sample rate rather than assuming one: `say` picks it per voice.
    probe = subprocess.run(
        [
            ff.replace("ffmpeg", "ffprobe"),
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate",
            "-of",
            "default=nw=1:nk=1",
            str(aiff),
        ],
        capture_output=True,
        text=True,
    )
    try:
        sr = int((probe.stdout or "").strip().splitlines()[0])
    except (ValueError, IndexError):
        sr = 22050
    run(
        [
            ff,
            "-y",
            "-i",
            str(aiff),
            "-af",
            f"asetrate={sr}*{factor},aresample={sr},atempo={1 / factor:.6f}",
            "-ar",
            str(sr),
            "-ac",
            "1",
            str(tmp),
        ]
    )
    tmp.replace(aiff)


# ---------------------------------------------------------------- Gemini TTS


def get_gemini_voice_for_speaker(name: str) -> str:
    """Resolve a speaker name to a prebuilt Gemini voice (Gemini backend)."""
    name = name.strip()
    if name in SPEAKER_GEMINI_VOICE_MAP:
        return SPEAKER_GEMINI_VOICE_MAP[name]
    parts = name.split()
    if parts and parts[0] in SPEAKER_GEMINI_VOICE_MAP:
        return SPEAKER_GEMINI_VOICE_MAP[parts[0]]
    digest = hashlib.md5(name.lower().encode("utf-8"), usedforsecurity=False).hexdigest()
    return GEMINI_FALLBACK_VOICES[int(digest, 16) % len(GEMINI_FALLBACK_VOICES)]


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int, out_wav: Path) -> None:
    """Wrap raw little-endian 16-bit PCM bytes into a WAV container."""
    n_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * n_channels * bits_per_sample // 8
    block_align = n_channels * bits_per_sample // 8
    data_size = len(pcm_bytes)
    fmt_chunk = struct.pack(
        "<4sIHHIIHH",
        b"fmt ",
        16,
        1,  # PCM
        n_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    )
    data_chunk = struct.pack("<4sI", b"data", data_size) + pcm_bytes
    riff_size = 4 + len(fmt_chunk) + len(data_chunk)
    riff_header = struct.pack("<4sI4s", b"RIFF", riff_size, b"WAVE")
    out_wav.write_bytes(riff_header + fmt_chunk + data_chunk)


def _gemini_tts_pcm(
    client: Any, model: str, script: str, speaker_voice_pairs: list[tuple[str, str]]
) -> tuple[bytes, int]:
    """Return (PCM bytes, sample_rate) for a Gemini TTS call.

    ``speaker_voice_pairs`` is (speaker_label, gemini_voice_name). Gemini's
    multi-speaker mode requires *exactly* 2 distinct voices; with 1 voice
    the script falls back to single-speaker mode (the speaker label is
    ignored by the API).
    """
    from google.genai import types

    if len(speaker_voice_pairs) == 2:
        speaker_configs = [
            types.SpeakerVoiceConfig(
                speaker=label,
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice),
                ),
            )
            for label, voice in speaker_voice_pairs
        ]
        speech_config = types.SpeechConfig(
            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=speaker_configs,
            ),
        )
    elif len(speaker_voice_pairs) == 1:
        _, voice = speaker_voice_pairs[0]
        speech_config = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice),
            ),
        )
    else:
        raise ValueError(
            f"Gemini TTS expects 1 or 2 speakers per call, got {len(speaker_voice_pairs)}"
        )

    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=speech_config,
    )
    response = client.models.generate_content(
        model=model,
        contents=script,
        config=config,
    )
    parts = response.candidates[0].content.parts
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and inline.data:
            mime = str(inline.mime_type or "")
            # Mime looks like 'audio/L16;codec=pcm;rate=24000'.
            sample_rate = 24000
            m = re.search(r"rate=(\d+)", mime)
            if m:
                sample_rate = int(m.group(1))
            return inline.data, sample_rate
    raise RuntimeError(
        f"Gemini TTS returned no audio part (got: {[type(p).__name__ for p in parts]})"
    )


def _gemini_render_script(
    segments: list[tuple[str, str]],
) -> tuple[str, list[tuple[str, str]]]:
    """Build a Gemini-friendly TTS script + the speaker→voice list.

    Returns (script_text, speaker_voice_pairs). Speakers in script appear in
    alphabetical order of their first-seen index so the model receives them
    in a stable order.
    """
    distinct: list[str] = []
    for speaker, _ in segments:
        if speaker not in distinct:
            distinct.append(speaker)
    speaker_voice_pairs = [(s, get_gemini_voice_for_speaker(s)) for s in distinct]
    body_lines: list[str] = []
    for speaker, text in segments:
        body_lines.append(f"{speaker}: {text}")
    return "\n\n".join(body_lines), speaker_voice_pairs


def render_segments_via_gemini(
    segments: list[tuple[str, str]],
    out_mp3: Path,
    bitrate: str,
    model: str,
    api_key: str,
) -> None:
    """Render the full transcript via Gemini multi-speaker TTS in chunks.

    Gemini multi-speaker mode is capped at 2 distinct voices per call as of
    2026. Transcripts with more speakers fall back to **single-speaker mode
    per segment** — slower, costlier, and the voice for the same speaker may
    drift across segments. Still functional for fixture generation; flagged
    in the comparison memo (#934).
    """
    from google import genai

    client = genai.Client(api_key=api_key)
    distinct_speakers = {s for s, _ in segments}

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        wav_chunks: list[Path] = []

        if len(distinct_speakers) <= 2:
            script, speaker_voice_pairs = _gemini_render_script(segments)
            pcm_bytes, sample_rate = _gemini_tts_pcm(client, model, script, speaker_voice_pairs)
            wav_path = td_path / "full.wav"
            _pcm_to_wav(pcm_bytes, sample_rate, wav_path)
            wav_chunks.append(wav_path)
        else:
            # > 2 speakers: per-segment single-speaker rendering.
            for idx, (speaker, text) in enumerate(segments, start=1):
                voice = get_gemini_voice_for_speaker(speaker)
                pcm_bytes, sample_rate = _gemini_tts_pcm(client, model, text, [(speaker, voice)])
                wav_path = td_path / f"chunk_{idx:03d}.wav"
                _pcm_to_wav(pcm_bytes, sample_rate, wav_path)
                wav_chunks.append(wav_path)

        # Concat WAV → MP3 via ffmpeg, mirroring the say backend's path.
        playlist = td_path / "playlist.txt"
        playlist.write_text(
            "\n".join(f"file '{p.as_posix()}'" for p in wav_chunks),
            encoding="utf-8",
        )
        run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(playlist),
                "-ac",
                "1",
                "-b:a",
                bitrate,
                str(out_mp3),
            ]
        )


def aiff_duration(path: Path) -> float:
    """Duration (seconds) of a rendered aiff via ffprobe. The ``say`` aiff render is
    byte-deterministic, so these durations reproduce the concatenated mp3's timeline
    exactly — the basis for the free per-turn RTTM ground truth (#1170)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return float(out.stdout.strip())


def voice_identity(voice: str, pitch: float | None) -> str:
    """The ACOUSTIC identity of a rendered turn: the voice, plus the shift when there is one.

    The RTTM used to record the bare ``say`` voice, on the stated grounds that
    one-voice-per-person made it a faithful person identity. That stopped being true the moment
    a language shipped only ONE voice: Italian has just Alice and German just Anna, so host and
    guest are the same synthesis at two pitches, and an RTTM saying `Alice` for both described a
    two-speaker episode that does not exist. A diarization reference that merges two speakers
    would score a correct diarizer as wrong — and the real pyannote service does separate them,
    finding three voices in exactly these files.

    `make_groundtruth.py` already draws this distinction in its own sidecar: "a shifted voice is
    not the voice it came from, and this field is what stops the voice_map from implying it is".
    This is the same statement, made where the diarization reference is written.

    DER solves the optimal label mapping, so the string only has to be DISTINCT per distinct
    acoustic identity — it does not have to be pretty.
    """
    return voice if pitch is None else f"{voice}@{pitch:g}"


def write_rttm(file_id: str, turns: list[tuple[str, float]], out_rttm: Path) -> None:
    """Write a NIST RTTM from ``(speaker_label, duration_s)`` turns in speaking order.

    Onset is the cumulative sum of prior turn durations. The speaker label is the acoustic
    identity from :func:`voice_identity` — the say voice, plus its pitch shift where one was
    applied."""
    lines: list[str] = []
    onset = 0.0
    for label, dur in turns:
        lines.append(f"SPEAKER {file_id} 1 {onset:.3f} {dur:.3f} <NA> <NA> {label} <NA> <NA>")
        onset += dur
    out_rttm.write_text("\n".join(lines) + "\n", encoding="utf-8")


def concat_aiff_to_mp3(aiffs: list[Path], out_mp3: Path, bitrate: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        playlist = Path(f.name)
        for p in aiffs:
            f.write(f"file '{p.as_posix()}'\n")
    try:
        run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(playlist),
                "-ac",
                "1",
                "-b:a",
                bitrate,
                str(out_mp3),
            ]
        )
    finally:
        playlist.unlink(missing_ok=True)


def transcript_language(raw: str) -> str:
    """The primary language subtag of the EPISODE, from its ``#fixture-v3:`` annotation.

    Read from ``host_voice``, falling back to ``voice``. The two are not the same question:
    ``voice`` is the GUEST's accent, and several English episodes carry a guest with a
    non-English one —

        p07_e01  #fixture-v3: voice=de-DE host_voice=en-AU
        p09_e01  #fixture-v3: voice=de-DE host_voice=en-US

    — which are English episodes with a German-accented guest, not German episodes. Reading
    ``voice`` alone called them German, and that was invisible while no German voice map existed.
    Adding one surfaced it as a false collision: `Dr. Elena Fischer` (Anna, her established
    accent-matched voice in an English show) against `Katrin Vogel` (Anna, host of the actual
    German show), reported as two people sharing a voice inside one language when they have
    never been in the same language at all.

    The host's accent is the show's language, because a show is conducted in its host's language
    and guests are the ones who vary. Defaults to ``en``, which is every fixture predating the
    annotation.
    """
    m = re.search(r"^#fixture-v3:.*\bhost_voice=([A-Za-z]{2})", raw, re.MULTILINE)
    if m is None:
        m = re.search(r"^#fixture-v3:\s*voice=([A-Za-z]{2})", raw, re.MULTILINE)
    return (m.group(1) if m else "en").lower()


def render_say_fixture(
    segments: list[tuple[str, str]],
    *,
    stem: str,
    out_mp3: Path,
    rttm_path: Path,
    rate: int | None,
    bitrate: str,
    rttm_only: bool,
    language: str = "en",
) -> None:
    """Render one fixture with the ``say`` backend: one aiff per turn, an exact per-turn
    RTTM from the (deterministic) aiff durations, and — unless ``rttm_only`` — the
    concatenated mp3. The RTTM therefore matches the mp3 timeline turn-for-turn."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        aiffs: list[Path] = []
        turns: list[tuple[str, float]] = []
        for i, (speaker, text) in enumerate(segments, start=1):
            voice = get_voice_for_speaker(speaker, language)
            pitch = pitch_shift_for(speaker, language)
            safe_speaker = re.sub(r"[^A-Za-z0-9_]", "_", speaker)[:24] or "spk"
            out_aiff = td_path / f"{stem}_{i:03d}_{safe_speaker}.aiff"
            say_to_aiff(text.strip(), out_aiff, voice=voice, rate=rate, pitch=pitch)
            aiffs.append(out_aiff)
            turns.append((voice_identity(voice, pitch), aiff_duration(out_aiff)))
        write_rttm(stem, turns, rttm_path)
        if not rttm_only:
            concat_aiff_to_mp3(aiffs, out_mp3, bitrate=bitrate)


def host_for_file(stem: str) -> str:
    """Pick host name from filename prefix: ``p01_e01`` -> ``Maya``."""
    m = FILE_PREFIX_RE.match(stem)
    if not m:
        raise ValueError(f"Filename does not start with 'pXX_': {stem}")
    pid = m.group(1).lower()
    host = PODCAST_HOSTS.get(pid)
    if not host:
        raise ValueError(f"No host mapping for podcast id '{pid}' (file: {stem})")
    return host


def derive_version_from_path(txt: Path) -> str | None:
    """Detect a ``v1``/``v2``/... segment inside the input transcript's path."""
    for part in txt.resolve().parts:
        if VERSION_SEGMENT_RE.fullmatch(part):
            return part
    return None


def read_fixtures_version_file() -> str | None:
    """Read the default fixture version from tests/fixtures/FIXTURES_VERSION."""
    candidate = Path(__file__).resolve().parent.parent / "FIXTURES_VERSION"
    if candidate.exists():
        return candidate.read_text(encoding="utf-8").strip()
    return None


def list_voices() -> int:
    if not shutil.which("say"):
        print("say not found (macOS required)", file=sys.stderr)
        return 2
    result = subprocess.run(["say", "-v", "?"], capture_output=True, text=True)
    print(result.stdout)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "transcripts",
        type=Path,
        nargs="*",
        help="Transcript files (.txt) or folder containing transcripts (searched recursively)",
    )
    ap.add_argument(
        "--output-version",
        default=None,
        help=(
            "Output version segment under tests/fixtures/audio/<version>/. "
            "Default: detect from input path; fall back to FIXTURES_VERSION."
        ),
    )
    ap.add_argument("--rate", type=int, default=145, help="Speech rate for say (default: 145)")
    ap.add_argument("--bitrate", default="64k", help="MP3 bitrate (default: 64k)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing mp3 files")
    ap.add_argument(
        "--rttm-only",
        action="store_true",
        help=(
            "Emit per-turn RTTM ground truth next to each transcript and skip mp3 output "
            "(say backend only). Pure additive artifact — never touches the audio."
        ),
    )
    ap.add_argument(
        "--list-voices",
        action="store_true",
        help="List available macOS 'say' voices and exit",
    )
    ap.add_argument(
        "--backend",
        choices=("say", "gemini"),
        default="say",
        help="TTS backend (default: say). 'gemini' requires GEMINI_API_KEY.",
    )
    ap.add_argument(
        "--gemini-model",
        default="gemini-2.5-flash-preview-tts",
        help="Gemini TTS model id (default: gemini-2.5-flash-preview-tts)",
    )
    args = ap.parse_args()

    if args.list_voices:
        return list_voices()

    if not args.transcripts:
        ap.error("transcripts argument required (unless --list-voices)")

    if args.rttm_only and args.backend != "say":
        ap.error("--rttm-only requires --backend say (per-turn timings need the aiff render)")

    # Collect all transcript files
    txt_files: list[Path] = []
    for path in args.transcripts:
        if path.is_file() and path.suffix == ".txt":
            txt_files.append(path)
        elif path.is_dir():
            txt_files.extend(sorted(path.rglob("*.txt")))
        else:
            print(f"Skipping (not a .txt file or directory): {path}", file=sys.stderr)

    if args.backend == "say":
        if not shutil.which("say"):
            print("say not found (macOS required)", file=sys.stderr)
            return 2
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found", file=sys.stderr)
        return 2

    gemini_api_key: str | None = None
    if args.backend == "gemini":
        gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip() or None
        if not gemini_api_key:
            print("GEMINI_API_KEY required for --backend gemini", file=sys.stderr)
            return 2

    scripts_dir = Path(__file__).resolve().parent
    fixtures_dir = scripts_dir.parent

    if not txt_files:
        print("No .txt transcripts found", file=sys.stderr)
        return 1

    # Output version: per-file (input path), CLI override, then FIXTURES_VERSION.
    cli_version = args.output_version
    fallback_version = read_fixtures_version_file()

    for txt in txt_files:
        if cli_version:
            version = cli_version
        else:
            version = derive_version_from_path(txt) or fallback_version
        if not version:
            print(
                f"Skipping {txt.name}: cannot determine output version "
                "(pass --output-version or place under tests/fixtures/transcripts/<version>/)",
                file=sys.stderr,
            )
            continue

        audio_dir = fixtures_dir / "audio" / version
        out_mp3 = audio_dir / f"{txt.stem}.mp3"
        rttm_path = txt.with_suffix(".rttm")
        if args.rttm_only:
            # `_fast` is an ffmpeg 60s truncation of p01_e01 (not a `say` render), so a
            # re-rendered RTTM would not match its audio — and it is not a diarization
            # eval fixture. Skip it.
            if txt.stem.endswith("_fast"):
                continue
        else:
            audio_dir.mkdir(parents=True, exist_ok=True)
            if out_mp3.exists() and not args.overwrite:
                continue

        raw = txt.read_text(encoding="utf-8").strip()
        if not raw:
            continue

        try:
            host_name = host_for_file(txt.stem)
        except ValueError as e:
            print(f"Skipping {txt.name}: {e}", file=sys.stderr)
            continue

        segments = parse_segments(raw, host_name=host_name)
        if not segments:
            continue

        if args.backend == "gemini":
            assert gemini_api_key is not None  # narrowed above
            render_segments_via_gemini(
                segments,
                out_mp3,
                bitrate=args.bitrate,
                model=args.gemini_model,
                api_key=gemini_api_key,
            )
        else:
            render_say_fixture(
                segments,
                stem=txt.stem,
                out_mp3=out_mp3,
                rttm_path=rttm_path,
                rate=args.rate,
                bitrate=args.bitrate,
                rttm_only=args.rttm_only,
                # The transcript declares its own language, so a fixture cannot be rendered in
                # one its text disagrees with.
                language=transcript_language(raw),
            )

        print(f"Wrote {rttm_path if args.rttm_only else out_mp3}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
