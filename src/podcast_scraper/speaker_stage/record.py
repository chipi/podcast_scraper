"""The episode's speaker record: who is on a voice and who is only named, and the graph's hosts
and guests derived from it (moved verbatim from ``workflow.metadata_generation``, E04 S2).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Set, Tuple

from podcast_scraper.graph_id_utils import is_bare_speaker_label
from podcast_scraper.identity.slugify import canonical_person_name
from podcast_scraper.speaker_detectors.hosts import (
    is_publishable_speaker_name,
    looks_like_publisher,
    strip_role_prefix,
)
from podcast_scraper.workflow.metadata_generation import SpeakerInfo
from podcast_scraper.workflow.transcript_resolution import resolve_segments_path, TranscriptPurpose

logger = logging.getLogger(__name__)


def _read_speaker_diagnostics(
    output_dir: Optional[str], transcript_file_path: Optional[str]
) -> Dict[str, Any]:
    """The roster's ``speakers.diagnostics.json`` beside the transcript, or ``{}``.

    It is the explanation of how names were chosen — what was tried, which method named each
    voice. The record reads two things from it: each placed voice's method, and the names the
    roster was given but could not place. Nothing else here depends on it.
    """
    if not (output_dir and transcript_file_path):
        return {}
    base = os.path.splitext(os.path.join(output_dir, transcript_file_path))[0]
    path = f"{base}.speakers.diagnostics.json"
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _self_introduced_names(
    output_dir: Optional[str], transcript_file_path: Optional[str]
) -> List[str]:
    """Names a diarized voice introduced ITSELF with, per the roster's own diagnostics."""
    if not output_dir or not transcript_file_path:
        return []
    diagnostics = _read_speaker_diagnostics(output_dir, transcript_file_path)
    mine = _self_introduced_voices(diagnostics)
    return [
        str(v["resolved_name"])
        for v in diagnostics.get("voices") or []
        if isinstance(v, dict) and v.get("voice") in mine and v.get("resolved_name")
    ]


def _self_introduced_voices(diagnostics: Mapping[str, Any]) -> Set[str]:
    """Voices whose name is their OWN introduction: source ``self_intro``, or an E1d rescue
    (``ladder_rescues``) whose cue was the voice's own "I'm X" (its decision-trace step). The
    rescue keeps its own source so its fires stay countable; the evidence is the same."""
    from podcast_scraper.providers.ml.diarization.ladder_rescues import E1D_SOURCE

    out: Set[str] = set()
    for v in diagnostics.get("voices") or []:
        if isinstance(v, dict) and v.get("voice") and v.get("source") == SELF_INTRO_SOURCE:
            out.add(str(v["voice"]))
    trace = diagnostics.get("decision_trace")
    steps = (trace.get("voices") if isinstance(trace, dict) else None) or {}
    for voice, voice_steps in steps.items():
        for step in voice_steps or []:
            if (
                isinstance(step, dict)
                and step.get("rung") == E1D_SOURCE
                and step.get("decision") == "named"
                and step.get("cue") == SELF_INTRO_SOURCE
            ):
                out.add(str(voice))
    return out


#: The roster source for a name a voice introduced ITSELF with ("I'm Paul Avery") — the one
#: evidence that a name which also begins the show's title is the host, not the show.
SELF_INTRO_SOURCE = "self_intro"


def _clean_record_name(name: Any) -> str:
    """A name as the record stores it: stripped, internal whitespace collapsed."""
    return " ".join(str(name or "").split())


def _unplaced_speakers(
    placed: Sequence[SpeakerInfo],
    *,
    diagnostics: Mapping[str, Any],
    detected_hosts: Optional[List[str]],
    detected_guests: Optional[List[str]],
    feed_title: Optional[str],
    language: Optional[str] = None,
) -> List[SpeakerInfo]:
    """People a source named for this episode whom no voice was matched to (#2075).

    Three sources, in order of authority, each tagged in ``source``:

    * ``feed_statement`` — the hosts the feed states (diagnostics ``tried.known_hosts``). This is
      how a host the roster refused to guess a voice for is kept: The Daily states three hosts, the
      roster could not tell which voice was Natalie Kitroeff, and until now she vanished from every
      field on the episode.
    * ``episode_metadata`` — names the show notes gave the roster that it could not bind
      (diagnostics ``summary.unbound_names``), e.g. a guest corroboration refused.
    * ``hint`` — the pre-listening guess passed to metadata generation. On an episode the roster
      named nobody on, or that was never diarized, this is the only thing anyone stated.

    None of them is a claim that the person spoke. That is the whole point of ``placed: false``.

    Refused here, with the same predicates the placed roster uses, so a name kept out of the
    speaking roster cannot come back through this list: role-word placeholders and bare speaker
    labels, publishers and networks, and the show's own name. A spelling variant of a placed person
    (``Dr. Rafael Prieto-Curiel`` beside ``Rafael Prieto-Curiel``) is the same human and is not
    listed twice.
    """
    from podcast_scraper.providers.ml.diarization.roster import (
        _same_person,
        _same_person_on_one_episode,
    )
    from podcast_scraper.speaker_detectors.hosts import names_the_show
    from podcast_scraper.speaker_detectors.normalization import is_default_speaker_name

    def _one_person_on_this_record(a: str, b: str) -> bool:
        # Every name here is one EPISODE's, so the one-episode respelling rule applies too: the
        # feed's "Hannah Fry" and a voice's "Professor Hannah Frye", or the show notes' "Bernard
        # Leong" and the hint's "Bernard Leung", are one person listed twice otherwise (12
        # episodes, 2026-10-08). `_same_person` stays: across sources a missing "Jr." is no
        # disagreement, which the one-episode rule reads as a father and a son.
        return a.lower() == b.lower() or _same_person(a, b) or _same_person_on_one_episode(a, b)

    tried_raw = diagnostics.get("tried")
    tried: Mapping[str, Any] = tried_raw if isinstance(tried_raw, dict) else {}
    summary_raw = diagnostics.get("summary")
    summary: Mapping[str, Any] = summary_raw if isinstance(summary_raw, dict) else {}

    candidates: List[Tuple[str, str, str]] = []  # (name, role, source)
    for n in tried.get("known_hosts") or []:
        candidates.append((n, "host", "feed_statement"))
    for n in detected_hosts or []:
        candidates.append((n, "host", "hint"))
    for n in summary.get("unbound_names") or []:
        candidates.append((n, "guest", "episode_metadata"))
    for n in detected_guests or []:
        candidates.append((n, "guest", "hint"))

    kept: List[SpeakerInfo] = []
    for raw_name, role, source in candidates:
        name = _clean_record_name(raw_name)
        # The same repair a placed name gets before the gate ("Planet Money's Kenny Malone").
        name = strip_role_prefix(name, language) if name else name
        # Two placeholder predicates, because they cover different shapes: `is_bare_speaker_label`
        # knows `SPEAKER_01` and the role words, `is_default_speaker_name` knows the providers'
        # failure tuple (`unknown_guest_1`). Either alone lets the other through.
        if (
            not name
            or is_bare_speaker_label(name)
            or is_default_speaker_name(name)
            or looks_like_publisher(name, language)
            # The same last gate a PLACED name passes. Unplaced entries were published without it:
            # "The China-Global South Project" on 10 episodes as an unplaced host, read from the
            # feed description (2026-10-02).
            or not is_publishable_speaker_name(name, language=language)
        ):
            continue
        if feed_title and names_the_show(name, feed_title, language):
            continue
        # The record publishes the canonical spelling (#2130: `Ali Ghodsi)`, `Paul Avery, MD`).
        # Only AFTER the filters above, which keep judging the stated string exactly as before.
        name = canonical_person_name(name) or name
        if any(_one_person_on_this_record(name, p.name) for p in placed):
            continue
        if any(_one_person_on_this_record(name, k.name) for k in kept):
            continue
        kept.append(
            SpeakerInfo(
                id=f"unplaced_{len(kept) + 1}",
                name=name,
                role=role,
                placed=False,
                voices=[],
                source=source,
            )
        )
    return kept


def _build_speaker_record(
    output_dir: Optional[str],
    transcript_file_path: Optional[str],
    detected_hosts: Optional[List[str]],
    detected_guests: Optional[List[str]],
    feed_title: Optional[str],
    language: Optional[str] = None,
) -> Tuple[List[SpeakerInfo], Optional[int]]:
    """The episode's speaker record: everyone placed on a voice, then everyone only named (#2075).

    Operator decision 2026-09-17: ONE record per episode, and every surface that shows people is
    written from it. Before this, `speakers` held EITHER the diarized roster OR — when the roster
    named nobody, or the episode was never diarized — the pre-listening guess, indistinguishable in
    the artifact. That second case is how an a16z episode whose transcript names no voice published
    eight hosts in its graph. Now both live in one list and each entry says which it is.

    Returns ``(speakers, num_speakers)``; ``num_speakers`` is the diarizer's voice count, or
    ``None`` when there was no diarization.
    """
    diarized, num_speakers = (
        _build_speakers_from_diarized_segments(
            output_dir, transcript_file_path, detected_guests, language
        )
        if output_dir
        else (None, None)
    )
    placed: List[SpeakerInfo] = list(diarized or [])
    # An ORGANISATION BY CORPUS VOTE is not a placed person, even when a segment label written
    # before #2220 says so: the label is what every re-derive reads back (see
    # `gi.pipeline._label_is_an_organisation`). Its voice stays raw, i.e. unnamed.
    from podcast_scraper.speaker_detectors.entity_kind_votes import KindVotes, votes_for_output_dir

    _votes = votes_for_output_dir(output_dir)
    if isinstance(_votes, KindVotes):
        placed = [sp for sp in placed if not _votes.calls_organisation(sp.name)]
    diagnostics = _read_speaker_diagnostics(output_dir, transcript_file_path)
    method_by_voice: Dict[str, str] = {}
    self_introduced = _self_introduced_voices(diagnostics)
    for v in diagnostics.get("voices") or []:
        if isinstance(v, dict) and v.get("voice") and v.get("source"):
            voice = str(v["voice"])
            method_by_voice[voice] = (
                SELF_INTRO_SOURCE if voice in self_introduced else str(v["source"])
            )
    for sp in placed:
        methods = [method_by_voice[v] for v in sp.voices if v in method_by_voice]
        # One person on two voices (a diarization split): if EITHER voice introduced itself, the
        # person did. Taking the first listed voice's source dropped a self-introduced host whose
        # name is the show's ("The Paul Avery Drive") whenever the other voice was listed first.
        sp.source = (
            SELF_INTRO_SOURCE
            if SELF_INTRO_SOURCE in methods
            else (methods[0] if methods else "roster")
        )
    if feed_title:
        # The SHOW is not a person on it (#2064). The graph already refuses it, so a record that
        # placed it would list a speaker no surface casts. A transcript still labelled with the
        # show's name is then reported by the sync check (LABEL_NOT_PLACED) — a roster to re-run.
        #
        # UNLESS THE VOICE SAID IT. A show is named after its host often enough ("The Paul Avery
        # Drive"), and a title-prefix rule cannot tell that host from "Machine Learning Street".
        # A self-introduction can: measured on prod, Paul Avery's voice named itself on all 40
        # episodes (source `self_intro`) and not one of the real show-name labels did (MLST,
        # Trivium China, Africa Tech Summit, Turkey Book — all `known_hosts` / `llm_resolution`).
        from podcast_scraper.speaker_detectors.hosts import names_the_show

        placed = [
            sp
            for sp in placed
            if sp.source == SELF_INTRO_SOURCE or not names_the_show(sp.name, feed_title, language)
        ]
    unplaced = _unplaced_speakers(
        placed,
        diagnostics=diagnostics,
        detected_hosts=detected_hosts,
        detected_guests=detected_guests,
        feed_title=feed_title,
        language=language,
    )
    return placed + unplaced, num_speakers


def _build_speakers_from_diarized_segments(
    output_dir: str,
    transcript_file_path: Optional[str],
    detected_guests: Optional[List[str]],
    language: Optional[str] = None,
) -> Tuple[Optional[List[SpeakerInfo]], Optional[int]]:
    """Derive ``content.speakers`` + ``diarization_num_speakers`` from saved segments (#876).

    The diarization roster (host + guests + voice count) lives only in-memory during the
    diarize pass; the durable record is the per-segment ``speaker_label`` (a real name when
    the roster resolved it, else ``SPEAKER_xx``) written to the segments sidecar. For network
    feeds the host (e.g. "Patrick O'Shaughnessy") is named by the transcript self-intro, not
    the RSS author, so the pre-diarization ``detected_hosts`` is empty — only the diarized
    segments carry the host. This reads them so ``content.speakers`` includes every named
    voice and ``num_speakers`` reflects the diarizer's voice count.

    Returns ``(speakers, num_speakers)``, or ``(None, None)`` when no diarized segments exist
    (the caller, :func:`_build_speaker_record`, then records only the named-but-unplaced).
    """
    if not transcript_file_path:
        return None, None
    # ANALYSIS: the ad-free sidecar first, the raw one for a pre-#974 corpus (#974, #2170).
    seg_path = resolve_segments_path(
        output_dir, transcript_file_path, purpose=TranscriptPurpose.ANALYSIS
    )
    if seg_path is None:
        return None, None
    try:
        segs = json.loads(seg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    if not isinstance(segs, list) or not segs:
        return None, None

    raw_ids: set[str] = set()
    # Preserve first-appearance order of named voices for stable host/guest ids.
    named_order: List[str] = []
    # Which diarization voices carry each name — the record's `voices` (#2075).
    voices_by_label: dict[str, List[str]] = {}
    # The roster's authoritative per-voice role, persisted on each named segment (first wins). The
    # roster decides host vs guest; the pre-diarization detected_guests hint is only a fallback.
    role_by_label: dict[str, str] = {}
    for s in segs:
        if not isinstance(s, dict):
            continue
        # ``is not None`` (not truthiness): a native diarizer's speaker id can be int 0, which a
        # falsy ``or`` chain would drop — undercounting num_speakers by one.
        _spk = s.get("speaker")
        if _spk is None:
            _spk = s.get("speaker_id")
        raw = str(_spk).strip() if _spk is not None else ""
        if raw:
            raw_ids.add(raw)
        label = str(s.get("speaker_label") or "").strip()
        # #2064: `is_bare_speaker_label` rather than a local `startswith("speaker")` test, which was
        # wrong in BOTH directions. It let the ROLE-WORD placeholders through — when the roster
        # cannot name the host's voice it labels the segments "Host", and that became a named voice,
        # reached `content.speakers`, and shipped as a Person with role="host". On production that
        # is `person:host`: 54 episodes, 1,437 grounded insights, 2nd of 2,907 people. And it
        # rejected "Speaker John Knight", who is a real person. The shared predicate covers numbered
        # labels AND role words, and lives beside the id builder so the two cannot drift.
        if (
            label
            and not is_bare_speaker_label(label)
            and not looks_like_publisher(label, language)  # a publisher is not a person
        ):
            if label not in named_order:
                named_order.append(label)
            if raw and raw not in voices_by_label.setdefault(label, []):
                voices_by_label[label].append(raw)
            seg_role = s.get("speaker_role")
            if label not in role_by_label and seg_role in ("host", "guest"):
                role_by_label[label] = seg_role

    num_speakers = len(raw_ids) or (len(named_order) or None)
    if not named_order:
        return None, num_speakers  # diarized but unnamed → keep num, no named roster

    guest_set = {g.strip().lower() for g in (detected_guests or []) if g and g.strip()}
    speakers: List[SpeakerInfo] = []
    hosts: List[str] = []
    guests: List[str] = []
    for name in named_order:
        # Roster role is authoritative; fall back to the detected_guests hint only when the segment
        # carries no role (legacy sidecars written before the role was persisted).
        role = role_by_label.get(name) or ("guest" if name.lower() in guest_set else "host")
        (guests if role == "guest" else hosts).append(name)
    # Every name here was read off a segment the diarizer attributed to a voice: PLACED (#2075).
    for idx, name in enumerate(hosts):
        sid = "host" if len(hosts) == 1 else f"host_{idx + 1}"
        speakers.append(
            SpeakerInfo(
                id=sid, name=name, role="host", placed=True, voices=voices_by_label.get(name, [])
            )
        )
    for idx, name in enumerate(guests):
        sid = "guest" if len(guests) == 1 else f"guest_{idx + 1}"
        speakers.append(
            SpeakerInfo(
                id=sid, name=name, role="guest", placed=True, voices=voices_by_label.get(name, [])
            )
        )
    return speakers, num_speakers


class _HasNameAndRole(Protocol):
    """What :func:`_speaker_lists_for_graph` actually needs from a speaker (#2062).

    ``SpeakerInfo`` satisfies it, and so does a plain object built from a persisted
    ``content.speakers`` dict — which is how ``enrich-edges`` reuses the same merge rule over
    artifacts on disk instead of growing a second copy that drifts from this one.
    """

    name: str
    role: str


def _speaker_lists_for_graph(
    speakers: Optional[Sequence[_HasNameAndRole]],
    feed_title: Optional[str] = None,
    language: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """``(hosts, guests)`` for the graph: the people a VOICE was matched to, and nobody else.

    THE RULE (operator decision 2026-09-17, #2075): one speaker record per episode, and a person
    becomes a host or guest in the graph only if a diarized voice was matched to them. The record is
    ``content.speakers``; each entry says whether it was ``placed``.

    What this function no longer does, each removed on purpose:

    * **It takes no pre-listening hint.** ``detected_hosts`` / ``detected_guests`` were a guess
      made from the feed and show notes before any audio was heard. Passed to the graph, that guess
      shipped a corpus where 0.6% of Person nodes were ``guest`` and 89.5% ``mentioned`` (#2062),
      and later cast eight a16z hosts on episodes whose transcripts named none of them. People the
      guess names now live in the record as ``placed: false``.
    * **It no longer falls back to the hint when the roster is empty.** An episode that was never
      diarized, or was diarized and named nobody, casts nobody (decision 2026-09-17, which reverses
      the documented fallback). Measured on the production snapshot: 124 no-voice episodes carried
      114 host/guest nodes no voice was matched to.
    * **It no longer injects feed hosts the roster abstained on.** The 2026-09-16 decision kept
      them as "feed-level participants" with role ``host``; it is superseded. They are kept in the
      record as ``placed: false`` instead, where no surface can read them as speakers.

    Pre-1.2.0 artifacts carry no ``placed``. For them every named entry of the roster is taken, as
    before — that roster was the diarization's own output — but neither the fallback nor the
    injection above applies any more, so re-deriving an old artifact can only REMOVE a cast member
    the audio never supported, never add one.

    Order is preserved and duplicates removed case-insensitively; the first spelling wins.
    """
    hosts: List[str] = []
    guests: List[str] = []
    seen: set[str] = set()

    def _is_the_show(name: str) -> bool:
        """The SHOW is not a person on it (#2064), even when the roster already says it is.

        Feed host detection is fixed at four separate entry points, but that only changes what NEW
        ingests produce. Every artifact already on disk whose roster was seeded with the show's own
        name still carries it — 19 episodes in a 279-episode production sample — and a re-run of the
        detector cannot repair them, because the wrong name is baked into `content.speakers`.

        So the graph boundary refuses it too. This is the last point before a string becomes a
        Person node with `role="host"`: followable, rankable among speakers, counted in person
        metrics. Costs one comparison against a title the artifact already carries.
        """
        if not feed_title:
            return False
        from podcast_scraper.speaker_detectors.hosts import names_the_show

        return names_the_show(name, feed_title, language)

    def _take(name: Optional[str], role: str, source: Optional[str] = None) -> None:
        clean = (name or "").strip()
        key = clean.lower()
        if not clean or key in seen:
            return
        if _is_the_show(clean) and source != SELF_INTRO_SOURCE:
            logger.info(
                "speaker %r names the show %r — not publishing it as a %s (#2064)",
                clean,
                feed_title,
                role,
            )
            return
        seen.add(key)
        (guests if role == "guest" else hosts).append(clean)

    # A RAW DIARIZATION LABEL IS NOT A PERSON. An abstained seat arrives as `SPEAKER_00` with
    # `role="host"`; taken literally that publishes the label itself as a host Person node. Same
    # guard as `graph_id_utils.is_bare_speaker_label`, which also covers the role words.
    from podcast_scraper.graph_id_utils import is_bare_speaker_label

    entries = list(speakers or [])
    is_record = any(getattr(sp, "placed", None) is not None for sp in entries)
    for sp in entries:
        if is_record and getattr(sp, "placed", None) is not True:
            continue
        nm = (getattr(sp, "name", "") or "").strip()
        if not nm or is_bare_speaker_label(nm):
            continue
        rl = (getattr(sp, "role", "") or "").strip().lower()
        # A voice with no usable role is treated as a host — matching
        # `_build_speakers_from_diarized_segments`, so the two cannot disagree.
        _take(nm, rl if rl in ("host", "guest") else "host", getattr(sp, "source", None))
    return hosts, guests
