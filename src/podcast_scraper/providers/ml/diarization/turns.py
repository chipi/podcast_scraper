"""Speaker turns and sentences, built from the offset segments (RFC-123 / slice S1.1).

WHY THIS EXISTS. A translation unit is a group of sentences inside one speaker's turn, so
translation needs a turn structure the transcript does not carry: the screenplay is text, and the
segments sidecar is a flat list. This builds the missing middle layer.

NOTHING READS IT YET. S1.2 writes the artifact; the three consumers RFC-123 describes are all v2.
That is deliberate — the artifact has to exist and be correct before anything depends on it.

TURNS ARE THE SCREENPLAY LINES, BY CONSTRUCTION. ``format_diarized_screenplay_with_offsets``
starts a new line exactly when ``speaker_label`` changes, so "consecutive segments with the same
label" is not a heuristic here — it is the same grouping the text was rendered from, which is what
lets a turn's char span be verified against the rendered text rather than trusted.

THE ONE THING THIS MODULE MUST NOT DO is invent a grouping that disagrees with the text. Every
invariant below exists to make a disagreement fail loudly at build time instead of surfacing as a
quote attributed to the wrong person.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: A backchannel is a very short interjection from another speaker between two turns of the same
#: speaker -- "yeah", "right", "mm-hmm". RFC-123 §2.2: they are FLAGGED, never merged away.
#: Merging would attribute the interjection to whoever spoke around it, which is falsification.
BACKCHANNEL_MAX_SEC = 1.5
BACKCHANNEL_MAX_WORDS = 3

#: Sentence-final punctuation followed by whitespace. The lookbehind keeps the punctuation with
#: the sentence it ends.
_SENT_SPLIT = re.compile(r"(?<=[.?!…])\s+")

#: Abbreviations whose trailing period must NOT end a sentence. Deliberately short: a long list
#: is a maintenance burden that buys little, and a wrong split costs a slightly odd unit boundary
#: rather than a correctness failure. Matched case-insensitively against the token before the dot.
_PROTECTED_ABBREV = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "vs",
        "etc",
        "eg",
        "ie",
        "approx",
        "dept",
        "est",
        "inc",
        "ltd",
        "co",
        "corp",
        "u.s",
        "u.k",
        "no",
    }
)


@dataclass
class Sentence:
    """One sentence inside a turn, with times and a statement of how they were derived."""

    sent_id: str
    char_start: int
    char_end: int
    start_ms: int
    end_ms: int
    #: ``segment_exact`` when both ends coincide with segment boundaries, otherwise
    #: ``segment_interpolated`` -- the times were derived by character proportion within a
    #: segment. A consumer that needs real precision (word-level anchors, V2-F) must be able to
    #: tell these apart rather than discovering it from behaviour.
    timing: str

    def to_dict(self) -> Dict[str, Any]:
        """The on-disk sentence row, `timing` included.

        `timing` travels with every sentence on purpose: it is what tells a consumer whether
        these millisecond bounds are word-anchored or interpolated across the segment. Dropping
        it from the wire format would leave the numbers looking equally precise.
        """
        return {
            "sent_id": self.sent_id,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "timing": self.timing,
        }


@dataclass
class Turn:
    """One speaker's uninterrupted run -- exactly one screenplay line."""

    turn_id: str
    speaker_label: str
    start_ms: int
    end_ms: int
    char_start: int
    char_end: int
    segment_idx: List[int]
    backchannel: bool
    sentences: List[Sentence] = field(default_factory=list)
    speaker: Optional[str] = None
    speaker_role: Optional[str] = None
    voice_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """The on-disk turn row.

        Both fields are emitted, and they follow `.segments.json`: `speaker` is the diarizer's
        voice id (`SPEAKER_12`), `speaker_label` is the label the screenplay line shows — the
        resolved name once naming has run ("Brenda Guillén"), else the voice id again. Checked on
        a named voice, 2026-10-09; this docstring used to say the reverse, from when naming ran
        after translation (D-34, reverted in #2234).
        """
        out: Dict[str, Any] = {
            "turn_id": self.turn_id,
            "speaker_label": self.speaker_label,
            "speaker": self.speaker,
            "speaker_role": self.speaker_role,
            "voice_type": self.voice_type,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "segment_idx": list(self.segment_idx),
            "backchannel": self.backchannel,
            "sentences": [s.to_dict() for s in self.sentences],
        }
        return out


@dataclass
class Turns:
    """The built artifact, minus the provenance S1.2 attaches when it writes."""

    turns: List[Turn] = field(default_factory=list)
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """The whole artifact: a schema version and the turns, in order.

        The version is first and always written, because `turns` sits outside
        `CANONICAL_STAGE_ORDER` (the one acknowledged exception to D-39) and so cannot rely on
        the pipeline's stage contract to describe its shape.
        """
        return {"version": self.version, "turns": [t.to_dict() for t in self.turns]}


class TurnInvariantError(AssertionError):
    """A built turn structure disagrees with the text it was built from.

    Raised rather than logged. A turn whose char span does not match the rendered text yields
    quotes attributed to the wrong speaker, and that is not a degradation a caller can sensibly
    continue past.
    """


def _ms(seconds: Any) -> int:
    return int(round(float(seconds or 0.0) * 1000.0))


def _is_backchannel(segments: Sequence[Dict[str, Any]]) -> bool:
    """A single very short, very few-word segment from one speaker.

    Judged on the TURN, not on its neighbours: a one-word turn is a backchannel whether or not
    the same speaker happens to surround it. The neighbour test in RFC-123's prose describes the
    typical case; using it as the rule would make the flag depend on unrelated turns.
    """
    if len(segments) != 1:
        return False
    seg = segments[0]
    duration = float(seg.get("end") or 0.0) - float(seg.get("start") or 0.0)
    words = len((seg.get("text") or "").split())
    return duration <= BACKCHANNEL_MAX_SEC and 1 <= words <= BACKCHANNEL_MAX_WORDS


def _protected_split(text: str) -> List[Tuple[int, int]]:
    """``(start, end)`` character ranges of sentences within ``text``, relative to it.

    Splits on sentence-final punctuation + whitespace, then re-joins a split whose left side ends
    in a protected abbreviation ("Dr.", "vs.") so "Dr. Fischer said" stays one sentence.
    """
    if not text.strip():
        return []
    pieces: List[Tuple[int, int]] = []
    cursor = 0
    for part in _SENT_SPLIT.split(text):
        if not part:
            continue
        start = text.index(part, cursor)
        pieces.append((start, start + len(part)))
        cursor = start + len(part)

    merged: List[Tuple[int, int]] = []
    for start, end in pieces:
        if merged:
            prev_start, prev_end = merged[-1]
            tail = text[prev_start:prev_end].rstrip()
            last_token = tail.rsplit(None, 1)[-1] if tail else ""
            if last_token.endswith(".") and last_token[:-1].lower() in _PROTECTED_ABBREV:
                merged[-1] = (prev_start, end)
                continue
        merged.append((start, end))
    return merged


def _sentence_times(
    abs_start: int,
    abs_end: int,
    segments: Sequence[Dict[str, Any]],
) -> Tuple[int, int, str]:
    """Times for a char range, from the segments containing it.

    Interpolates by character proportion when an end falls mid-segment, and SAYS SO via the
    returned timing mode -- a caller that needs precision must be able to tell an interpolated
    boundary from a real one.
    """
    exact_start = exact_end = True
    start_ms = end_ms = None

    for seg in segments:
        s_cs, s_ce = int(seg["char_start"]), int(seg["char_end"])
        if s_cs <= abs_start < s_ce or (abs_start == s_cs == s_ce):
            span = max(1, s_ce - s_cs)
            frac = (abs_start - s_cs) / span
            seg_start, seg_end = _ms(seg.get("start")), _ms(seg.get("end"))
            start_ms = seg_start + int(round(frac * (seg_end - seg_start)))
            exact_start = abs_start == s_cs
        if s_cs < abs_end <= s_ce:
            span = max(1, s_ce - s_cs)
            frac = (abs_end - s_cs) / span
            seg_start, seg_end = _ms(seg.get("start")), _ms(seg.get("end"))
            end_ms = seg_start + int(round(frac * (seg_end - seg_start)))
            exact_end = abs_end == s_ce

    if start_ms is None:
        start_ms = _ms(segments[0].get("start"))
        exact_start = False
    if end_ms is None:
        end_ms = _ms(segments[-1].get("end"))
        exact_end = False
    if end_ms < start_ms:
        end_ms = start_ms

    mode = "segment_exact" if (exact_start and exact_end) else "segment_interpolated"
    return start_ms, end_ms, mode


def build_turns(
    offset_segments: Sequence[Dict[str, Any]],
    *,
    screenplay_text: Optional[str] = None,
) -> Turns:
    """Group offset segments into turns with sentences. Pure; no I/O.

    ``offset_segments`` must be what ``format_diarized_screenplay_with_offsets`` returned — its
    second element, carrying ``char_start`` / ``char_end`` into the screenplay it rendered.
    Passing the RAW diarization segments instead produces turns whose char spans point nowhere,
    which is why ``screenplay_text`` exists: supply it and the identity is VERIFIED rather than
    assumed.
    """
    turns = Turns()
    if not offset_segments:
        return turns

    # Group consecutive same-label segments. No sort: the formatter already sorted by start time
    # and the char offsets are into ITS output, so re-sorting here could disagree with the text.
    groups: List[List[int]] = []
    for idx, seg in enumerate(offset_segments):
        label = str(seg.get("speaker_label") or seg.get("speaker") or "SPEAKER")
        if groups and _label_of(offset_segments[groups[-1][-1]]) == label:
            groups[-1].append(idx)
        else:
            groups.append([idx])

    for ordinal, idx_group in enumerate(groups):
        segs = [offset_segments[i] for i in idx_group]
        first, last = segs[0], segs[-1]
        turn = Turn(
            turn_id=f"t{ordinal:04d}",
            speaker_label=_label_of(first),
            start_ms=_ms(first.get("start")),
            end_ms=_ms(last.get("end")),
            char_start=int(first["char_start"]),
            char_end=int(last["char_end"]),
            segment_idx=list(idx_group),
            backchannel=_is_backchannel(segs),
            speaker=first.get("speaker"),
            speaker_role=first.get("speaker_role"),
            voice_type=first.get("voice_type"),
        )

        # Sentences, over the turn's speech text as assembled from its segments. Coalesced
        # segments are joined by a single space in the screenplay, so the same join reproduces
        # the turn's text exactly.
        turn_text = " ".join((s.get("text") or "").strip() for s in segs)
        for s_ordinal, (rel_start, rel_end) in enumerate(_protected_split(turn_text), start=1):
            abs_start = turn.char_start + rel_start
            abs_end = turn.char_start + rel_end
            start_ms, end_ms, mode = _sentence_times(abs_start, abs_end, segs)
            turn.sentences.append(
                Sentence(
                    sent_id=f"{turn.turn_id}.s{s_ordinal:02d}",
                    char_start=abs_start,
                    char_end=abs_end,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    timing=mode,
                )
            )
        turns.turns.append(turn)

    _assert_invariants(turns, offset_segments, screenplay_text)
    return turns


def _label_of(seg: Dict[str, Any]) -> str:
    return str(seg.get("speaker_label") or seg.get("speaker") or "SPEAKER")


def _assert_invariants(
    turns: Turns,
    offset_segments: Sequence[Dict[str, Any]],
    screenplay_text: Optional[str],
) -> None:
    """RFC-123 §2.4. Raises ``TurnInvariantError`` rather than returning a verdict.

    These are not defensive paranoia: each one corresponds to a way a turn structure can silently
    disagree with the text, and every such disagreement ends as a quote attributed to the wrong
    speaker.
    """
    prev_end = -1
    seen_segments: List[int] = []
    for turn in turns.turns:
        if turn.char_start < prev_end:
            raise TurnInvariantError(
                f"{turn.turn_id} starts at {turn.char_start}, before the previous turn ended at "
                f"{prev_end} — turns must be ordered and non-overlapping in char space"
            )
        if turn.char_end < turn.char_start:
            raise TurnInvariantError(f"{turn.turn_id} has char_end < char_start")
        if turn.end_ms < turn.start_ms:
            raise TurnInvariantError(f"{turn.turn_id} has end_ms < start_ms")
        prev_end = turn.char_end
        seen_segments.extend(turn.segment_idx)

        last_sent_end = turn.char_start
        for sent in turn.sentences:
            if sent.char_start < last_sent_end:
                raise TurnInvariantError(
                    f"{sent.sent_id} overlaps the previous sentence in {turn.turn_id}"
                )
            if not (turn.char_start <= sent.char_start <= sent.char_end <= turn.char_end):
                raise TurnInvariantError(
                    f"{sent.sent_id} ({sent.char_start},{sent.char_end}) escapes its turn's span "
                    f"({turn.char_start},{turn.char_end})"
                )
            if sent.end_ms < sent.start_ms:
                raise TurnInvariantError(f"{sent.sent_id} has end_ms < start_ms")
            last_sent_end = sent.char_end

    expected = list(range(len(offset_segments)))
    if sorted(seen_segments) != expected:
        missing = set(expected) - set(seen_segments)
        dupes = [i for i in set(seen_segments) if seen_segments.count(i) > 1]
        raise TurnInvariantError(
            f"every segment must be in exactly one turn; missing={sorted(missing)} "
            f"duplicated={sorted(dupes)}"
        )

    # The identity that makes the char spans trustworthy rather than merely plausible. Only
    # checkable when the caller supplies the text the offsets point into.
    if screenplay_text is not None:
        for turn in turns.turns:
            rendered = screenplay_text[turn.char_start : turn.char_end]
            expected_text = " ".join(
                (offset_segments[i].get("text") or "").strip() for i in turn.segment_idx
            )
            if rendered != expected_text:
                raise TurnInvariantError(
                    f"{turn.turn_id}: screenplay_text[{turn.char_start}:{turn.char_end}] is "
                    f"{rendered[:60]!r} but its segments say {expected_text[:60]!r} — the turn "
                    "grouping disagrees with the rendered text"
                )
