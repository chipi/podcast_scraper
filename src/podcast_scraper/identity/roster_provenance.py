"""Did this episode's speaker list come from the AUDIO or from a guess? (#2070)

``metadata_generation`` decides it in one line::

    speakers = diarized_speakers or _build_speakers_from_detected_names(detected_hosts, ...)

**This is #2065's root cause.** The real diarization roster and the pre-diarization HINT — read
from the feed and the show notes before a second of audio is processed — land in the same field and
are indistinguishable on disk. That is precisely why the graph could be fed the hint for months
while nobody could tell by looking at an artifact: there was nothing in the artifact to look at.

Every downstream reader then treats a guess as evidence. ``_speaker_lists_for_graph``'s own
docstring says the roster is the ONLY source used when it heard the episode — a rule it cannot
enforce, because it cannot tell which it was handed. The m0009 migration has the same problem one
layer down, and so does every coherence check that asks "did this person actually speak".

THE RULE FOR READERS: **unknown provenance is not evidence.** ``unknown`` is its own answer, not a
synonym for ``diarized``. Treating an unlabelled artifact as a measurement is exactly the
assumption that produced #2065, and every artifact currently on disk is unlabelled — so a reader
that cares must decide explicitly what to do about that, rather than having the decision made for
it by a default.

ONE FUNCTION, EVERY CALLER THAT ASKS. :func:`roster_provenance` is the only way to ask, so the
migration and the coherence report cannot drift into two different opinions about what a roster is
— which is the failure mode this whole arc keeps rediscovering. Today that is exactly two callers,
both gating m0009's roster-denies demotion; the pipeline WRITES the label (via
:func:`roster_source`) and has no reason to read it back.

WHAT IS NOT WIRED, so nobody reads a promise into this module that the code does not keep:
:func:`is_measured_roster` and :func:`build_content_speakers_block` have no callers. The first is
the strict reading, kept for the surfaces that will need it once the corpus carries labels; the
second assembles the roster and its label together so a future writer cannot emit one without the
other, and the single writer today predates it. Neither is load-bearing — check before assuming
either is enforcing anything.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Literal, Mapping, Optional

#: Written into ``content`` beside ``speakers``.
SOURCE_KEY = "speakers_source"

#: What a WRITER may record: the roster was resolved from audio, or it is the pre-diarization
#: guess. Narrower than :data:`Provenance` on purpose — ``unknown`` and ``absent`` are answers a
#: READER derives, never values anyone writes down.
RosterSource = Literal["diarized", "hint"]

#: ``diarized`` — resolved from the audio. ``hint`` — the pre-diarization guess from the feed and
#: show notes. ``unknown`` — the artifact predates this field. ``absent`` — no speakers at all.
Provenance = Literal["diarized", "hint", "unknown", "absent"]


def roster_source(speakers: Iterable[Any], *, diarized: bool) -> Optional[RosterSource]:
    """The value to write for a roster that was (or was not) resolved from audio.

    ``None`` when there is no roster: an empty list has no origin to claim, and writing one would
    be inventing provenance for a value that does not exist.
    """
    if not list(speakers or []):
        return None
    return "diarized" if diarized else "hint"


def build_content_speakers_block(
    *, speakers: Any, diarized: bool, num_speakers: Optional[int]
) -> Dict[str, Any]:
    """The ``content`` fragment carrying the roster and its provenance.

    A single place that assembles the two together, so a future caller cannot write the roster and
    forget the label — which would put us back where we started with no way to tell.
    """
    block: Dict[str, Any] = {"speakers": speakers}
    source = roster_source(speakers or [], diarized=diarized)
    if source is not None:
        block[SOURCE_KEY] = source
    if num_speakers is not None:
        block["diarization_num_speakers"] = num_speakers
    return block


def roster_provenance(metadata_payload: Mapping[str, Any]) -> Provenance:
    """Where this artifact's ``content.speakers`` came from.

    ``unknown`` means the artifact predates the field — which is every artifact written before
    #2070. It is deliberately NOT ``diarized``: a reader must handle "I cannot tell" as its own
    case. Silently upgrading unknown to trusted is the assumption this module exists to remove.
    """
    content = metadata_payload.get("content") or {}
    speakers = content.get("speakers") or []
    if not speakers:
        return "absent"
    source = str(content.get(SOURCE_KEY) or "").strip().lower()
    if source in ("diarized", "hint"):
        return source  # type: ignore[return-value]
    return "unknown"


def is_measured_roster(metadata_payload: Mapping[str, Any]) -> bool:
    """True only when the roster is PROVABLY from the audio.

    The strict reading: ``unknown`` returns False. Callers that must still work on the existing
    corpus should say so explicitly at their call site, with the reason, rather than relying on a
    lenient default here.
    """
    return roster_provenance(metadata_payload) == "diarized"
