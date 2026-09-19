"""Realistic episode SHAPES driven through the real pipeline, one assertion table (#2062 / #2064).

WHY THIS FILE EXISTS. #2064 was fixed four times — the title pattern, the `<itunes:author>` tag, the
episode-authors fallback, and finally the LLM provider fall-through — and the first three all passed
their unit tests while a fresh ingest still produced ``host='Africa Tech Summit'`` on every episode.
Each test covered a FUNCTION; none covered the PATH an episode takes through them, and the show name
simply arrived by the next route along.

So these drive `generate_episode_metadata` itself, from a diarization segments sidecar to the
persisted artifacts, and assert on what a reader would see. The KG runs in ``metadata_only`` mode,
so there is no LLM and no network: the shapes are cheap enough to keep one per realistic episode
type rather than one per code path.

EACH SHAPE IS A REAL ONE, taken from production or from a fresh DGX ingest, and named as such.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from podcast_scraper.kg.speaker_coherence import check_episode
from podcast_scraper.speaker_detectors.hosts import names_the_show
from podcast_scraper.workflow import metadata_generation as metadata

pytestmark = [pytest.mark.integration]

_tests_dir = Path(__file__).parent.parent.parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))
_spec = importlib.util.spec_from_file_location("parent_conftest", _tests_dir / "conftest.py")
assert _spec is not None and _spec.loader is not None
_pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pc)

TRANSCRIPT_REL = "transcripts/0001 - Episode_Title.txt"


@dataclass
class Shape:
    """One realistic episode shape: what diarization produced, and what must come out."""

    name: str
    #: ``(speaker_label, raw_voice, role_or_None, line)`` per turn, in order.
    turns: List[Tuple[str, str, str | None, str]]
    #: the pre-diarization hint the pipeline is handed
    hint_hosts: List[str] = field(default_factory=list)
    hint_guests: List[str] = field(default_factory=list)
    expect_hosts: List[str] = field(default_factory=list)
    expect_guests: List[str] = field(default_factory=list)
    #: names that must NOT appear with a speaker role
    forbid_speakers: List[str] = field(default_factory=list)
    #: overrides for the RSS feed — title/authors, so a show-as-author feed can be driven whole
    feed: Dict[str, Any] = field(default_factory=dict)
    #: when False, no segments sidecar is written: the episode was never diarized
    with_roster: bool = True


SHAPES = [
    Shape(
        name="ordinary interview — host and guest both named",
        turns=[
            ("Ryan Knutson", "SPEAKER_00", "host", "Welcome back to the show today."),
            ("Heather Haddon", "SPEAKER_01", "guest", "Thanks for having me on."),
            ("Ryan Knutson", "SPEAKER_00", "host", "Tell me what you found."),
            ("Heather Haddon", "SPEAKER_01", "guest", "The margin structure is the story."),
        ],
        expect_hosts=["Ryan Knutson"],
        expect_guests=["Heather Haddon"],
    ),
    Shape(
        name="stand-in interviewer — the feed's hosts are absent (#2061)",
        turns=[
            ("Lane Florsheim", "SPEAKER_00", "host", "In My Monday Morning, we sit down."),
            ("Twiggy Lawson", "SPEAKER_01", "guest", "I was sixteen when it started."),
            ("Lane Florsheim", "SPEAKER_00", "host", "And what changed after that?"),
        ],
        hint_hosts=["Ryan Knutson", "Jessica Mendoza"],
        expect_hosts=["Lane Florsheim"],
        expect_guests=["Twiggy Lawson"],
        forbid_speakers=["Ryan Knutson", "Jessica Mendoza"],
    ),
    Shape(
        name="panel — three named voices, nobody collapses",
        turns=[
            ("Eric Olander", "SPEAKER_00", "host", "Welcome to the discussion."),
            ("Jorge Heine", "SPEAKER_01", "guest", "The trade picture changed in 2020."),
            ("Lizzie Lee", "SPEAKER_02", "guest", "And the investment flows followed it."),
            ("Eric Olander", "SPEAKER_00", "host", "Say more about the flows."),
            ("Lizzie Lee", "SPEAKER_02", "guest", "They doubled inside eighteen months."),
        ],
        expect_hosts=["Eric Olander"],
        expect_guests=["Jorge Heine", "Lizzie Lee"],
    ),
    Shape(
        name="mixed markers — an unnamed voice must not inherit the host's name (#2062)",
        turns=[
            ("Kevin Roose", "SPEAKER_00", "host", "Welcome back to the programme."),
            ("SPEAKER_01", "SPEAKER_01", None, "I think the margin structure matters most."),
            ("Kevin Roose", "SPEAKER_00", "host", "Why is that the case?"),
            ("SPEAKER_01", "SPEAKER_01", None, "Because it compounds with scale."),
        ],
        expect_hosts=["Kevin Roose"],
        forbid_speakers=["SPEAKER_01"],
    ),
    Shape(
        name="the SHOW already sits in the roster as host — the real #2064 artifact shape",
        turns=[
            # Verbatim shape of a freshly ingested Africa Tech Summit episode: feed host detection
            # seeded the show's own name, so the ROSTER labelled a voice with it and stamped
            # role=host. Every artifact on disk with this defect looks like this, and neither the
            # feed-detection fix nor a re-run of it can repair them — so the graph boundary has to
            # refuse the name too.
            ("Africa Tech Summit", "SPEAKER_00", "host", "Welcome to the summit podcast."),
            ("Mukami Wairaina", "SPEAKER_01", "guest", "The agritech funding gap is the story."),
            ("Nixon Kanali", "SPEAKER_02", "guest", "And the policy side moved with it."),
        ],
        feed={"title": "Africa Tech Summit Podcast", "authors": ["Africa Tech Summit"]},
        expect_guests=["Mukami Wairaina", "Nixon Kanali"],
        forbid_speakers=["Africa Tech Summit"],
    ),
    Shape(
        name="mononym guest — a single-token name is still the guest (#1685/#2062)",
        turns=[
            ("Lane Florsheim", "SPEAKER_00", "host", "Today we talk about the sixties."),
            ("Twiggy", "SPEAKER_01", "guest", "I was sixteen when it started."),
            ("Lane Florsheim", "SPEAKER_00", "host", "And the photographs?"),
        ],
        expect_hosts=["Lane Florsheim"],
        expect_guests=["Twiggy"],
    ),
    Shape(
        name="no roster at all — never diarized, so nobody is invented",
        turns=[
            ("Kevin Roose", "SPEAKER_00", "host", "Welcome back to the programme."),
            ("Casey Newton", "SPEAKER_01", "guest", "The deal still stands."),
        ],
        with_roster=False,
        hint_hosts=["Kevin Roose"],
        hint_guests=["Casey Newton"],
        # REVERSED by operator decision 2026-09-17 (#2075). This shape used to pin "with no roster
        # the hint is used wholesale". But the hint is a guess made before any audio was heard, and
        # an episode with no diarization has no voice to match anyone to — so nobody is cast. The
        # names are kept in the speaker record as placed: false; they must not hold a speaking
        # role in the graph. Measured on the production snapshot: 124 such episodes carried 114
        # host/guest graph nodes nobody had matched to a voice.
        expect_hosts=[],
        expect_guests=[],
        forbid_speakers=["Kevin Roose", "Casey Newton"],
    ),
    Shape(
        name="monologue — one voice, no invented guest",
        turns=[
            ("Sarah Guo", "SPEAKER_00", "host", "This week I want to talk about compute."),
            ("Sarah Guo", "SPEAKER_00", "host", "The cost curve has not bent yet."),
        ],
        expect_hosts=["Sarah Guo"],
        expect_guests=[],
    ),
]


def _write_corpus(tdir: Path, shape: Shape) -> None:
    (tdir / "transcripts").mkdir(parents=True, exist_ok=True)
    text = "".join(f"{label}: {line}\n" for label, _raw, _role, line in shape.turns)
    (tdir / TRANSCRIPT_REL).write_text(text, encoding="utf-8")
    segs: List[Dict[str, Any]] = []
    for i, (label, raw, role, line) in enumerate(shape.turns):
        seg: Dict[str, Any] = {
            "start": float(i),
            "end": float(i + 1),
            "text": line,
            "speaker": raw,
            "speaker_label": label,
        }
        if role:
            seg["speaker_role"] = role
        segs.append(seg)
    if not shape.with_roster:
        # No sidecar: diarization never ran, so there is no roster and the pre-diarization hint is
        # all the pipeline has. It must still not invent anybody.
        return
    base = str(tdir / TRANSCRIPT_REL)[: -len(".txt")]
    Path(base + ".segments.json").write_text(json.dumps(segs), encoding="utf-8")


def _run(tdir: Path, shape: Shape) -> Tuple[dict, dict]:
    cfg = _pc.create_test_config(
        output_dir=str(tdir),
        generate_metadata=True,
        metadata_format="json",
        generate_kg=True,
        kg_extraction_source="metadata_only",
    )
    path = metadata.generate_episode_metadata(
        feed=_pc.create_test_feed(**shape.feed),
        episode=_pc.create_test_episode(),
        feed_url=_pc.TEST_FEED_URL,
        cfg=cfg,
        output_dir=str(tdir),
        run_suffix=None,
        transcript_file_path=TRANSCRIPT_REL,
        transcript_source="whisper_transcription",
        whisper_model="base",
        detected_hosts=list(shape.hint_hosts),
        detected_guests=list(shape.hint_guests),
    )
    assert path, f"{shape.name}: no metadata written"
    meta = json.loads(Path(path).read_text(encoding="utf-8"))
    kg_files = list(tdir.rglob("*.kg.json"))
    assert kg_files, f"{shape.name}: no KG artifact written"
    return meta, json.loads(kg_files[0].read_text(encoding="utf-8"))


def _roles(kg: dict) -> Dict[str, str]:
    return {
        str((n.get("properties") or {}).get("name") or ""): str(
            (n.get("properties") or {}).get("role") or ""
        )
        for n in (kg.get("nodes") or [])
        if n.get("type") == "Person"
    }


@pytest.fixture(params=SHAPES, ids=lambda s: s.name)
def shape_run(request, tmp_path: Path):
    shape: Shape = request.param
    tdir = tmp_path / "out"
    tdir.mkdir()
    _write_corpus(tdir, shape)
    meta, kg = _run(tdir, shape)
    return shape, meta, kg


class TestEveryShapeProducesACoherentEpisode:
    def test_the_expected_hosts_are_hosts(self, shape_run) -> None:
        shape, _meta, kg = shape_run
        roles = _roles(kg)
        for name in shape.expect_hosts:
            assert roles.get(name) == "host", f"{name!r} -> {roles.get(name)!r} in {roles}"

    def test_the_expected_guests_are_guests(self, shape_run) -> None:
        shape, _meta, kg = shape_run
        roles = _roles(kg)
        for name in shape.expect_guests:
            assert roles.get(name) == "guest", f"{name!r} -> {roles.get(name)!r} in {roles}"

    def test_nobody_forbidden_holds_a_speaker_role(self, shape_run) -> None:
        # A name the roster never heard — an absent co-host, an anonymous voice — must not be
        # published as though it spoke.
        shape, _meta, kg = shape_run
        roles = _roles(kg)
        for name in shape.forbid_speakers:
            assert roles.get(name) not in ("host", "guest"), f"{name!r} -> {roles.get(name)!r}"

    def test_no_voice_label_is_published_as_a_person(self, shape_run) -> None:
        _shape, _meta, kg = shape_run
        for name, role in _roles(kg).items():
            assert not (
                role in ("host", "guest") and name.lower().startswith("speaker")
            ), f"{name!r} published as {role}"

    def test_the_show_itself_is_never_a_speaker(self, shape_run) -> None:
        _shape, meta, kg = shape_run
        feed_title = (meta.get("feed") or {}).get("title") or ""
        for name, role in _roles(kg).items():
            if role in ("host", "guest"):
                assert not names_the_show(name, feed_title), f"{name!r} names the show"

    def test_the_coherence_guards_are_clean(self, shape_run) -> None:
        # The cross-artifact rules, applied to a pipeline-produced episode rather than a fixture.
        shape, meta, kg = shape_run
        violations = check_episode(meta, kg, {"nodes": [], "edges": []}, label=shape.name)
        assert not violations, "\n  ".join(violations)

    def test_the_roster_reaches_content_speakers(self, shape_run) -> None:
        shape, meta, _kg = shape_run
        named = {str(s.get("name")) for s in ((meta.get("content") or {}).get("speakers") or [])}
        for expected in shape.expect_hosts + shape.expect_guests:
            assert expected in named, f"{expected!r} missing from {sorted(named)}"
