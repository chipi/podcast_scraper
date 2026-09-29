"""Which transcript variant a reader gets, and why (#2170).

An episode has up to four bodies on disk, and picking the wrong one fails SILENTLY:

- ``<base>.txt``           canonical, full timeline, ads included
- ``<base>.adfree.txt``    ads excised — **the space GI's ``char_start`` lives in**
- ``<base>.cleaned.txt``   the summariser's in-memory cleaner output, saved as a byproduct
- plus a ``.segments.json`` sidecar per body

Before this module fifteen readers each derived their own precedence from the single
``transcript_file_path`` stored in metadata, and they disagreed. Most of the disagreement
was correct, which is exactly why it was dangerous:

**Analysis readers need the ad-free body.** GI computes quote ``char_start`` against
``.adfree.txt``. Pair those offsets with the raw body and every span is displaced by the
ads before it — no exception, no error, just text from the wrong place.

**The player needs the raw body.** It streams the ORIGINAL unbridged audio, ads included.
The ad-free segments are minutes shorter, so pairing them with that audio drifts
highlight-follow and tap-to-seek. And a plausible wrong segment is indistinguishable from
a right one, so this fails without a symptom too.

So a single precedence cannot serve both, and that is why resolution takes a
:class:`TranscriptPurpose` rather than being a function of the path alone. Asking for a
transcript without saying what for is the bug this module exists to make unrepresentable.

``docs/wip/2170-TRANSCRIPT-RESOLVER-INVENTORY.md`` lists every reader, its purpose, and the
sites deliberately left resolving their own way.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

ADFREE_SUFFIX = ".adfree"
CLEANED_SUFFIX = ".cleaned"

#: The English-derived variant (RFC-124 / S2.1b). Inserted BEFORE ``.adfree``, so the four bodies
#: an episode can have are ``<base>.txt``, ``<base>.adfree.txt``, ``<base>.en.txt`` and
#: ``<base>.en.adfree.txt`` -- source-language canonical, source-language ad-free, English
#: canonical, English ad-free.
EN_SUFFIX = ".en"

PathLike = Union[str, "os.PathLike[str]"]


class TranscriptPurpose(str, Enum):
    """What the caller is going to DO with the transcript.

    Not a preference and not a tuning knob — the two values select different coordinate
    spaces, and the wrong one is a correctness bug rather than a degradation.
    """

    #: Text whose character offsets GI's ``char_start`` / ``char_end`` index. Ad-free first.
    ANALYSIS = "analysis"

    #: Text and segment times that line up with the original audio. Raw canonical first.
    TIMELINE = "timeline"


def adfree_transcript_relpath(transcript_relpath: str) -> str:
    """``transcripts/01 - ep.txt`` -> ``transcripts/01 - ep.adfree.txt``."""
    base, ext = os.path.splitext(transcript_relpath)
    return f"{base}{ADFREE_SUFFIX}{ext or '.txt'}"


def english_transcript_relpath(transcript_relpath: str) -> str:
    """``transcripts/01 - ep.txt`` -> ``transcripts/01 - ep.en.txt``."""
    base, ext = os.path.splitext(transcript_relpath)
    return f"{base}{EN_SUFFIX}{ext or '.txt'}"


def english_adfree_transcript_relpath(transcript_relpath: str) -> str:
    """``transcripts/01 - ep.txt`` -> ``transcripts/01 - ep.en.adfree.txt``."""
    return adfree_transcript_relpath(english_transcript_relpath(transcript_relpath))


def _cleaned_transcript_relpath(transcript_relpath: str) -> str:
    """``transcripts/01 - ep.txt`` -> ``transcripts/01 - ep.cleaned.txt``."""
    base, ext = os.path.splitext(transcript_relpath)
    return f"{base}{CLEANED_SUFFIX}{ext or '.txt'}"


def _segments_relpath(transcript_relpath: str) -> str:
    """The sidecar beside a transcript body: ``<base>.txt`` -> ``<base>.segments.json``."""
    return f"{os.path.splitext(transcript_relpath)[0]}.segments.json"


def _canonical_relpath(transcript_relpath: str) -> str:
    """Strip a derived suffix so a ``.adfree.txt`` reference resolves like its raw base.

    Metadata always stores the plain ``.txt`` — three separate guards reject the derived
    variants as "the transcript" — but GI's ``transcript_ref`` points at whichever body it
    read, and that ref does come back here.
    """
    rel = (transcript_relpath or "").strip().replace("\\", "/")
    if not rel:
        return ""
    base, ext = os.path.splitext(rel)
    # Strip REPEATEDLY, because the suffixes stack: ``ep1.en.adfree.txt`` has to canonicalize all
    # the way to ``ep1.txt``. Stripping only the outermost one (what this did before the English
    # branch existed) would leave ``ep1.en``, whose candidate list is built off the wrong base and
    # resolves nothing -- the quiet kind of failure, since every candidate simply fails to exist.
    changed = True
    while changed:
        changed = False
        for suffix in (ADFREE_SUFFIX, CLEANED_SUFFIX, EN_SUFFIX):
            if base.lower().endswith(suffix):
                base = base[: -len(suffix)]
                changed = True
    return f"{base}{ext or '.txt'}" if base else rel


def text_relpath_candidates(
    transcript_relpath: str,
    *,
    purpose: TranscriptPurpose,
    include_cleaned: bool = False,
) -> List[str]:
    """The bodies to try, in order, for *purpose*. Pure — no disk access.

    ``include_cleaned`` inserts ``.cleaned.txt`` as a middle candidate. It is not a third
    purpose: the cleaned body is the summariser's byproduct and only one reader wants it
    (the recurrent-host scan, which will take any rendering with the ads gone).

    THE ENGLISH HEAD IS A PURE PREPEND (S2.1b). Each list gains one English candidate at the
    front and its existing tail is untouched, so for an episode with no ``.en.*`` on disk the
    resolved path is exactly what it was before this branch existed. That is why the resolver
    needs no ``multilingual_ingest`` check: the flag gates whether the English files are ever
    PRODUCED, and a candidate that does not exist costs one ``is_file()``.

    ANALYSIS DOES NOT FALL BACK FROM ``.en.adfree.txt`` TO ``.en.txt``. Doing so would put an
    ad-laden English body into the space GI's offsets index. The English artifact set is written
    atomically (S2.5), so the intermediate state does not occur; if it somehow did, the honest
    fallback is the source-language ad-free text, which is at least in the right coordinate space.
    """
    rel = _canonical_relpath(transcript_relpath)
    if not rel:
        return []
    adfree = adfree_transcript_relpath(rel)
    cleaned = [_cleaned_transcript_relpath(rel)] if include_cleaned else []
    if purpose is TranscriptPurpose.ANALYSIS:
        return [english_adfree_transcript_relpath(rel), adfree, *cleaned, rel]
    return [english_transcript_relpath(rel), rel, *cleaned, adfree]


def segments_relpath_candidates(
    transcript_relpath: str,
    *,
    purpose: TranscriptPurpose,
) -> List[str]:
    """The segment sidecars to try, in order, for *purpose*. Pure — no disk access."""
    return [
        _segments_relpath(rel)
        for rel in text_relpath_candidates(transcript_relpath, purpose=purpose)
    ]


def _first_existing(output_dir: PathLike, relpaths: Sequence[str]) -> Optional[Path]:
    root = Path(output_dir)
    for rel in relpaths:
        candidate = root / rel
        if candidate.is_file():
            return candidate
    return None


def resolve_text_path(
    output_dir: PathLike,
    transcript_relpath: str,
    *,
    purpose: TranscriptPurpose,
    include_cleaned: bool = False,
) -> Optional[Path]:
    """The transcript body *purpose* should read, or None if nothing is on disk."""
    return _first_existing(
        output_dir,
        text_relpath_candidates(
            transcript_relpath, purpose=purpose, include_cleaned=include_cleaned
        ),
    )


def resolve_segments_path(
    output_dir: PathLike,
    transcript_relpath: str,
    *,
    purpose: TranscriptPurpose,
) -> Optional[Path]:
    """The segment sidecar *purpose* should read, or None if nothing is on disk."""
    return _first_existing(
        output_dir, segments_relpath_candidates(transcript_relpath, purpose=purpose)
    )


def _read_text(path: PathLike) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""


def _read_json(path: PathLike) -> Optional[Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _segments_list(doc: Any) -> Optional[List[Dict[str, Any]]]:
    """A BARE LIST sidecar, or None.

    Deliberately narrower than ``gi/repair._segments_for``, which also unwraps
    ``{"segments": [...]}``. Accepting the wrapped shape here would hand GI and KG segments
    where they previously got ``None`` — a behaviour change this slice is not allowed to
    make. Widening it is a decision for whoever can say what the wrapped shape means.
    """
    return doc if isinstance(doc, list) else None


@dataclass
class ProcessingTranscript:
    """A resolved transcript: its text, its segments, and WHICH file they came from.

    ``transcript_ref`` is the relpath actually loaded — point quote and viewer references
    at it so a highlight lands where the reader computed it. ``is_adfree`` lets a consumer
    tell whether ad removal already happened or it must excise for itself (an old corpus,
    or ``save_adfree_transcript`` disabled).
    """

    text: str
    segments: Optional[List[Dict[str, Any]]]
    transcript_ref: str
    ad_map: Optional[Dict[str, Any]]
    is_adfree: bool


def load_transcript(
    output_dir: PathLike,
    transcript_relpath: str,
    *,
    purpose: TranscriptPurpose,
) -> ProcessingTranscript:
    """Load the body *purpose* wants, with its own sidecar and (when ad-free) its ad-map.

    THE SEGMENTS ALWAYS COME FROM THE BODY THAT WAS LOADED. Mixing a body with the other
    variant's sidecar is the displacement bug in a different shape, so the sidecar is
    derived from the resolved body rather than resolved independently.
    """
    root = Path(output_dir)
    candidates = text_relpath_candidates(transcript_relpath, purpose=purpose)
    chosen_rel = next((rel for rel in candidates if (root / rel).is_file()), None)

    if chosen_rel is None:
        # Nothing on disk. Report the canonical path as the ref rather than inventing one,
        # and stay non-ad-free so a consumer excises for itself instead of trusting empty.
        return ProcessingTranscript(
            text="",
            segments=None,
            transcript_ref=_canonical_relpath(transcript_relpath),
            ad_map=None,
            is_adfree=False,
        )

    chosen = root / chosen_rel
    base = os.path.splitext(str(chosen))[0]
    is_adfree = os.path.splitext(chosen_rel)[0].lower().endswith(ADFREE_SUFFIX)
    return ProcessingTranscript(
        text=_read_text(chosen),
        segments=_segments_list(_read_json(base + ".segments.json")),
        transcript_ref=chosen_rel,
        ad_map=_read_json(base + ".admap.json") if is_adfree else None,
        is_adfree=is_adfree,
    )


def load_processing_transcript(output_dir: str, transcript_file_path: str) -> ProcessingTranscript:
    """The ad-free base if present, else the raw transcript — i.e. ANALYSIS.

    Kept under its original name because it is the resolver GI and KG already call. Its old
    docstring claimed to be "the single resolver all NLP consumers use", which was never
    true: it had two callers while thirteen other readers resolved independently. It is now
    one spelling of :func:`load_transcript`, and the claim is the module's job to keep.
    """
    return load_transcript(output_dir, transcript_file_path, purpose=TranscriptPurpose.ANALYSIS)
