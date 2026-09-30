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


def is_english_render_relpath(transcript_relpath: str) -> Optional[bool]:
    """Is this relpath one of the derived ENGLISH bodies? ``None`` when there is nothing to read.

    Answers a question about the path in hand, which is the opposite of what the callers that
    needed it were doing: building ``english_transcript_relpath(rel)`` and testing whether THAT
    exists. The suffixes stack, so that construction is only correct when ``rel`` is canonical —
    and the one caller that mattered passed a path that had already been resolved to
    ``ep1.en.adfree.txt``, producing ``ep1.en.adfree.en.txt``, which never exists.

    Measured 2026-09-30, before this existed: for a successfully translated Spanish episode the
    indexer chunked ``transcripts/ep1.en.adfree.txt`` — English text — and labelled the chunks
    ``es``, so the router dropped their embeddings and filed them in the vector-less
    ``segments_nonen`` tier. The episode was findable by neither semantic search nor its own
    language, which is the exact outcome `_indexed_text_language` was written to prevent.

    Matches on the suffix STACK, not on the outermost suffix, because ``.en`` may sit under
    ``.adfree`` (``ep1.en.adfree.txt``) or be outermost (``ep1.en.txt``).
    """
    rel = (transcript_relpath or "").strip().replace("\\", "/")
    if not rel:
        return None
    base = os.path.splitext(rel)[0].lower()
    while True:
        for suffix in (ADFREE_SUFFIX, CLEANED_SUFFIX):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        else:
            break
    return base.endswith(EN_SUFFIX)


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

    ENGLISH IS THE DEFAULT, FOR BOTH PURPOSES — a decision (D-38), not a side effect of the
    order these lines happen to be written in. When an episode has been translated, every
    surface shows English unless it asks otherwise: ``analysis`` because the intelligence layer
    is single-path by construction (D-1), and ``timeline`` — the player's own precedence —
    because a translated episode has full standing (D-37) and serving its source language by
    default would be a per-surface split in everything but name.

    The source language is never destroyed: ``.txt`` and ``.segments.json`` stay canonical, so
    S2.8's ``?lang=`` exposes the source as an explicit choice rather than needing a reprocess.
    To reverse the default, move the English candidate off the front of the relevant list here —
    one line, and the tests that pin it will say so.

    THE ENGLISH HEAD IS A PURE PREPEND (S2.1b). Each list gains one English candidate at the
    front and its existing tail is untouched, so for an episode with no ``.en.*`` on disk the
    resolved path is exactly what it was before this branch existed. That is why the resolver
    needs no ``multilingual_ingest`` check: the flag gates whether the English files are ever
    PRODUCED, and a candidate that does not exist costs one ``is_file()``.

    ANALYSIS DOES NOT FALL BACK FROM ``.en.adfree.txt`` TO ``.en.txt``. Doing so would put an
    ad-laden English body into the space GI's offsets index.

    This docstring used to add "the English artifact set is written atomically, so the
    intermediate state does not occur" — and that sentence became the alibi for a real bug: the
    English ad-free base was governed by ``save_adfree_transcript``, so with that flag off the
    state occurred constantly and ANALYSIS fell through to the SOURCE language. The set is now
    complete-or-withdrawn and ``english_artifacts_present`` names this file explicitly, so the
    claim is enforced rather than asserted. If the state somehow arises anyway, the fallback to
    source-language ad-free text is at least in the right coordinate space.
    """
    rel = _canonical_relpath(transcript_relpath)
    if not rel:
        return []
    adfree = adfree_transcript_relpath(rel)
    cleaned = [_cleaned_transcript_relpath(rel)] if include_cleaned else []
    if purpose is TranscriptPurpose.ANALYSIS:
        return [english_adfree_transcript_relpath(rel), adfree, *cleaned, rel]
    return [english_transcript_relpath(rel), rel, *cleaned, adfree]


def source_language_relpath_candidates(transcript_relpath: str) -> List[str]:
    """The bodies to try for the SOURCE-language layer, in order. Pure — no disk access.

    The complement of :func:`text_relpath_candidates`: every English candidate removed, so this
    resolves the text the episode was actually spoken in even when a translation exists. Needed
    by RFC-124 §6.2's "both layers are indexed" — the search index carries the source-language
    chunks so a query in that language reaches the episode, which the English-first precedence
    cannot provide by construction.

    Ad-free first, matching ANALYSIS, because a source-language ad-free body is the better
    retrieval target when one exists. It usually does NOT: ad excision runs on the English text
    (D-19), so for a translated episode this normally lands on the canonical ``.txt``. Both are
    in the same coordinate space as each other and NEITHER is in the analysis space once a
    translation exists — which is why chunks built from this must be labelled with the source
    language, never indexed as analysis text, and excluded from the quote-offset verifier and
    the transcript lift.
    """
    rel = _canonical_relpath(transcript_relpath)
    if not rel:
        return []
    return [adfree_transcript_relpath(rel), rel]


def resolve_source_language_text_path(
    output_dir: PathLike, transcript_relpath: str
) -> Optional[Path]:
    """The first existing source-language body, or ``None``."""
    return _first_existing(output_dir, source_language_relpath_candidates(transcript_relpath))


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
