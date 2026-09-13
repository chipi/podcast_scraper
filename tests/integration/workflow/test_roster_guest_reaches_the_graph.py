"""The diarization roster's guest must reach the knowledge graph (#2062).

MEASURED ON PRODUCTION (330-episode feed-stratified sample, 2026-09-13):

  * the roster names a guest in 176 of 263 episodes (66.9%)
  * ``content.speakers`` carries that guest in 97.7% of those — the roster -> metadata leg works
  * ``kg.json`` carries a Person with ``role="guest"`` in only 4.5% of them

  => 93.2% of roster-named guests are lost between ``content.speakers`` and ``kg.json``.
     Corpus-wide that is 0.6% of Person nodes tagged ``guest`` versus 89.5% ``mentioned``,
     which is why the UI labels essentially every human on an episode a contributor.

THE CAUSE. ``generate_episode_metadata`` receives ``detected_hosts`` / ``detected_guests`` as
PARAMETERS — the pre-diarization hint, which for a network feed is empty because the guest is
only ever named by the transcript self-intro. Diarization then resolves the real roster into the
local ``speakers`` (``_build_speakers_from_diarized_segments``, which reads the authoritative
per-voice ``speaker_role`` off the segments sidecar — present on 39 of 40 prod episodes sampled).
The ``kg_build_artifact`` call passes the stale PARAMETERS and never looks at ``speakers``.

So the graph is told "there are no guests on this episode" by a value computed before the
pipeline knew who was speaking.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

from podcast_scraper.workflow import metadata_generation as metadata

pytestmark = [pytest.mark.integration]

_tests_dir = Path(__file__).parent.parent.parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))
_spec = importlib.util.spec_from_file_location("parent_conftest", _tests_dir / "conftest.py")
assert _spec is not None and _spec.loader is not None
_pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pc)

HOST = "Patrick O'Shaughnessy"
GUEST = "Brian Chesky"
TRANSCRIPT_REL = "transcripts/0001 - Episode_Title.txt"

TRANSCRIPT = (
    f"{HOST}: Welcome back to the show, today we talk about design.\n"
    f"{GUEST}: Thanks for having me, the design story starts in 2008.\n"
    f"{HOST}: Take me back there.\n"
    f"{GUEST}: We were three people with a mattress and no customers.\n"
)


def _seg(label: str, raw: str, role: str, text: str) -> Dict[str, Any]:
    """One diarized segment carrying the roster's authoritative per-voice role."""
    return {
        "start": 0.0,
        "end": 1.0,
        "text": text,
        "speaker": raw,
        "speaker_label": label,
        "speaker_role": role,
    }


SEGMENTS = [
    _seg(HOST, "SPEAKER_00", "host", "Welcome back to the show, today we talk about design."),
    _seg(GUEST, "SPEAKER_01", "guest", "Thanks for having me, the design story starts in 2008."),
    _seg(HOST, "SPEAKER_00", "host", "Take me back there."),
    _seg(GUEST, "SPEAKER_01", "guest", "We were three people with a mattress and no customers."),
]


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    tdir = tmp_path / "out"
    (tdir / "transcripts").mkdir(parents=True)
    (tdir / TRANSCRIPT_REL).write_text(TRANSCRIPT, encoding="utf-8")
    base = str(tdir / TRANSCRIPT_REL)[: -len(".txt")]
    Path(base + ".segments.json").write_text(json.dumps(SEGMENTS), encoding="utf-8")
    return tdir


def _run(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    """Drive metadata generation and capture what the KG builder was actually told."""
    captured: Dict[str, Any] = {}

    import podcast_scraper.kg as kg_mod

    def _spy(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {
            "schema_version": "2.1",
            "episode_id": args[0] if args else "ep",
            "nodes": [],
            "edges": [],
        }

    monkeypatch.setattr(kg_mod, "build_artifact", _spy)

    cfg = _pc.create_test_config(
        output_dir=str(workspace),
        generate_metadata=True,
        metadata_format="json",
        generate_kg=True,
        kg_extraction_source="metadata_only",
    )
    metadata.generate_episode_metadata(
        feed=_pc.create_test_feed(),
        episode=_pc.create_test_episode(),
        feed_url=_pc.TEST_FEED_URL,
        cfg=cfg,
        output_dir=str(workspace),
        run_suffix=None,
        transcript_file_path=TRANSCRIPT_REL,
        transcript_source="whisper_transcription",
        whisper_model="base",
        # The prod condition: the pre-diarization hint knows nothing. The roster knows everything.
        detected_hosts=[],
        detected_guests=[],
    )
    return captured


class TestTheRostersGuestReachesTheKnowledgeGraph:
    def test_the_kg_is_told_about_the_guest(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _run(workspace, monkeypatch)
        assert captured, "kg build_artifact was never called — the test never reached the seam"
        guests = list(captured.get("detected_guests") or [])
        assert GUEST in guests, (
            f"the KG was told detected_guests={guests!r} while the diarization roster had "
            f"{GUEST!r} as role=guest on the segments — this is the 93.2% prod loss"
        )

    def test_the_kg_is_told_about_the_host(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _run(workspace, monkeypatch)
        hosts = list(captured.get("detected_hosts") or [])
        assert HOST in hosts, f"the KG was told detected_hosts={hosts!r}; the roster named {HOST!r}"

    def test_the_guest_is_not_demoted_to_host(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `_build_speakers_from_diarized_segments` falls back to "host" for a roleless voice,
        # so a guest that loses its role does not merely vanish — it is published as a host.
        captured = _run(workspace, monkeypatch)
        assert GUEST not in list(captured.get("detected_hosts") or [])

    def test_the_roster_still_lands_in_content_speakers(self, workspace: Path) -> None:
        # The control: the roster -> metadata leg works on prod (97.7%) and must keep working.
        cfg = _pc.create_test_config(
            output_dir=str(workspace),
            generate_metadata=True,
            metadata_format="json",
        )
        path = metadata.generate_episode_metadata(
            feed=_pc.create_test_feed(),
            episode=_pc.create_test_episode(),
            feed_url=_pc.TEST_FEED_URL,
            cfg=cfg,
            output_dir=str(workspace),
            run_suffix=None,
            transcript_file_path=TRANSCRIPT_REL,
            transcript_source="whisper_transcription",
            whisper_model="base",
            detected_hosts=[],
            detected_guests=[],
        )
        assert path and os.path.exists(path)
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        speakers = (doc.get("content") or {}).get("speakers") or []
        by_role = {(s.get("role"), s.get("name")) for s in speakers}
        assert ("guest", GUEST) in by_role
        assert ("host", HOST) in by_role


class TestTheWholeChainOnDisk:
    """Diarizer output -> real pipeline -> persisted kg.json -> what the API serves.

    ``build_artifact`` is NOT stubbed here: the real KG builder runs (``metadata_only``, so no LLM)
    and the artifact is read back off disk, then through the same view function the episode-entities
    route uses. A role that is right in memory and wrong on the wire is still wrong on the panel.

    The one hop this cannot cover locally is diarization itself — ``pyannote`` has no x86_64 macOS
    wheel, so ``tests/e2e/test_diarization_e2e.py`` skips here. This test therefore starts from the
    diarizer's OUTPUT (the segments sidecar), which is exactly where both #2062 defects live.
    """

    def test_the_guest_is_persisted_and_served_as_a_guest(self, workspace: Path) -> None:
        from podcast_scraper.server.app_kg_view import entities_from_kg

        cfg = _pc.create_test_config(
            output_dir=str(workspace),
            generate_metadata=True,
            metadata_format="json",
            generate_kg=True,
            kg_extraction_source="metadata_only",
        )
        metadata.generate_episode_metadata(
            feed=_pc.create_test_feed(),
            episode=_pc.create_test_episode(),
            feed_url=_pc.TEST_FEED_URL,
            cfg=cfg,
            output_dir=str(workspace),
            run_suffix=None,
            transcript_file_path=TRANSCRIPT_REL,
            transcript_source="whisper_transcription",
            whisper_model="base",
            detected_hosts=[],
            detected_guests=[],
        )

        kg_files = list(workspace.rglob("*.kg.json"))
        assert kg_files, "the pipeline persisted no KG artifact"
        artifact = json.loads(kg_files[0].read_text(encoding="utf-8"))

        on_disk = {
            (n.get("properties") or {}).get("name"): (n.get("properties") or {}).get("role")
            for n in artifact.get("nodes") or []
            if n.get("type") == "Person"
        }
        assert on_disk.get(GUEST) == "guest", f"kg.json on disk says {on_disk!r}"
        assert on_disk.get(HOST) == "host", f"kg.json on disk says {on_disk!r}"

        persons, _orgs, _topics = entities_from_kg(artifact)
        served = {p.name: p.role for p in persons}
        assert served.get(GUEST) == "guest", (
            f"the API would serve the guest as role={served.get(GUEST)!r} — the operator-reported "
            "symptom is the guest always rendering as a contributor"
        )
        assert served.get(HOST) == "host"


class TestTheGraphAgreesWithWhoActuallySpoke:
    """The invariant that would have caught the phantom hosts, and did not exist (#2062).

    Every test in this area checked ONE code path in isolation — does the roster parse a role, does
    the merge prefer the roster, does the view read the field. Not one asserted that the finished
    artifact made sense as a whole, so a graph could claim a host who was never in the episode and
    the entire suite stayed green.

    On production, 39.5% of ``host`` nodes and 40% of ``guest`` nodes named someone who never spoke
    in that episode: a co-host who sat the episode out, the show's own name as a person, and ASR
    variants of a real speaker that put one human in the graph twice.

    HOST AND GUEST ARE SPEAKING ROLES. If a person did not speak, the graph may still know them —
    as ``mentioned``, which is what they are — but it must not call them a host or a guest.
    """

    def _run(self, workspace: Path, hosts, guests) -> Dict[str, Any]:
        cfg = _pc.create_test_config(
            output_dir=str(workspace),
            generate_metadata=True,
            metadata_format="json",
            generate_kg=True,
            kg_extraction_source="metadata_only",
        )
        metadata.generate_episode_metadata(
            feed=_pc.create_test_feed(),
            episode=_pc.create_test_episode(),
            feed_url=_pc.TEST_FEED_URL,
            cfg=cfg,
            output_dir=str(workspace),
            run_suffix=None,
            transcript_file_path=TRANSCRIPT_REL,
            transcript_source="whisper_transcription",
            whisper_model="base",
            detected_hosts=hosts,
            detected_guests=guests,
        )
        kg_files = list(workspace.rglob("*.kg.json"))
        meta_files = [p for p in workspace.rglob("*.metadata.json")]
        assert kg_files and meta_files
        return {
            "kg": json.loads(kg_files[0].read_text(encoding="utf-8")),
            "meta": json.loads(meta_files[0].read_text(encoding="utf-8")),
        }

    @staticmethod
    def _speakers_and_roles(out: Dict[str, Any]):
        spoke = {
            str(s.get("name") or "")
            for s in ((out["meta"].get("content") or {}).get("speakers") or [])
        }
        graph = {
            str((n.get("properties") or {}).get("name") or ""): str(
                (n.get("properties") or {}).get("role") or ""
            )
            for n in (out["kg"].get("nodes") or [])
            if n.get("type") == "Person"
        }
        return spoke, graph

    def test_every_graph_host_or_guest_actually_spoke(self, workspace: Path) -> None:
        # THE INVARIANT. The feed hint names a co-host who is not in this episode at all; the
        # roster heard only the real two. The absentee must not be published as a host.
        out = self._run(workspace, ["Some Absent Cohost"], [])
        spoke, graph = self._speakers_and_roles(out)
        speakers_in_graph = {n for n, r in graph.items() if r in ("host", "guest")}
        phantom = speakers_in_graph - spoke
        assert not phantom, (
            f"the graph calls {sorted(phantom)} a host/guest of this episode, but the roster says "
            f"only {sorted(spoke)} spoke"
        )

    def test_the_absent_cohost_is_not_in_the_graph_as_a_speaker(self, workspace: Path) -> None:
        out = self._run(workspace, ["Some Absent Cohost"], [])
        _spoke, graph = self._speakers_and_roles(out)
        assert graph.get("Some Absent Cohost") not in ("host", "guest")

    def test_the_show_itself_is_never_a_host(self, workspace: Path) -> None:
        # "The China-Global South Project" shipped as a host Person node on real episodes.
        out = self._run(workspace, ["The China-Global South Project"], [])
        _spoke, graph = self._speakers_and_roles(out)
        assert graph.get("The China-Global South Project") not in ("host", "guest")

    def test_the_real_speakers_are_still_there(self, workspace: Path) -> None:
        # An invariant satisfied by publishing nobody would be worthless.
        out = self._run(workspace, ["Some Absent Cohost"], [])
        _spoke, graph = self._speakers_and_roles(out)
        assert graph.get(HOST) == "host"
        assert graph.get(GUEST) == "guest"
