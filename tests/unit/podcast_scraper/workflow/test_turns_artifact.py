"""Writing ``turns.json``: provenance, the two coordinate spaces, and what it refuses to write.

WHAT THESE TESTS ARE ACTUALLY FOR. ``build_turns`` is already covered as a pure function. What is
new here is disk: the artifact claims ``text[char_start:char_end]`` is a turn's speech in a named
file, and a consumer will believe that claim. So every span assertion below reads the text back
OFF DISK rather than using the in-memory string, because the in-memory one is the thing that could
be right while the file is wrong.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper import config
from podcast_scraper.providers.ml.diarization.formatting import (
    format_diarized_screenplay_with_offsets,
)
from podcast_scraper.providers.ml.diarization.turns import TurnInvariantError
from podcast_scraper.workflow import turns_artifact
from podcast_scraper.workflow.episode_processor import _produce_transcript_sidecars
from podcast_scraper.workflow.turns_artifact import (
    turns_manifest_metrics,
    turns_path,
    write_turns_artifact,
)

pytestmark = pytest.mark.unit

REL = "transcripts/01 - ep.txt"


#: A pre-roll ad cluster dense enough for the detector to cut -- the same shape
#: ``test_adfree_transcript`` uses. The long body matters: with a short episode the density pass
#: removes nothing (measured: ``chars_removed: 0``), the ad-free text equals the raw one, and a
#: test claiming to compare two coordinate spaces would be comparing one against itself.
_PREROLL = (
    "Ramp understands no one wants to chase receipts. Ramp saves companies 5 percent. "
    "Check out ramp dot com slash invest. They all use WorkOS for SSO and SCIM and RBAC. "
    "Visit WorkOS dot com to get started. Learn more at rogo dot ai slash Felix. "
)

#: The one short interjection, so the backchannel flag has something real to find.
BACKCHANNEL_TEXT = "Right."


def _segments() -> List[Dict[str, Any]]:
    """A cuttable pre-roll, an alternating two-speaker body, and one backchannel."""
    body = (
        "Hello and welcome everyone I am the host. Today we discuss the bioscience boom. "
        "Our guest has spent twenty years in healthcare investing. Let us dive in. "
    ) * 12
    # The ad gets its OWN speaker on purpose, so it forms its own turn and the ad-free variant
    # DROPS a turn and renumbers -- the exact case RFC-123 §Key Decisions 5 is about. With the ad
    # merged into Maya's opening turn (my first fixture) both variants had 26 turns and the test
    # proved nothing about renumbering.
    segs: List[Dict[str, Any]] = [
        {"start": 0.0, "end": 1.0, "text": _PREROLL, "speaker_label": "Announcer"}
    ]
    clock = 1.0
    # Runs of TWO sentences per speaker, so the interjection below can sit BETWEEN two segments
    # of the same speaker. With a strictly alternating body it cannot: the short turn coalesces
    # into the next same-label segment and stops being short -- which is the formatter behaving
    # correctly and my first fixture measuring nothing (backchannels: 0).
    for i, sentence in enumerate(body.split(". ")):
        sentence = sentence.strip()
        if not sentence:
            continue
        speaker = "Maya" if (i // 2) % 2 == 0 else "Liam"
        segs.append(
            {
                "start": clock,
                "end": clock + 1.0,
                "text": sentence if sentence.endswith(".") else sentence + ".",
                "speaker_label": speaker,
            }
        )
        clock += 1.0
        if i == 8:
            # Sub-1.5s, one word, from the OTHER speaker, with the same speaker either side:
            # a backchannel by RFC-123 §2.2's definition.
            segs.append(
                {
                    "start": clock,
                    "end": clock + 0.8,
                    "text": BACKCHANNEL_TEXT,
                    "speaker_label": "Liam" if speaker == "Maya" else "Maya",
                }
            )
            clock += 0.8
    return segs


def _lay_down_episode(tmp_path: Path) -> tuple[str, List[Dict[str, Any]]]:
    """Write the transcript + segments sidecar exactly as the pipeline does, then return them."""
    segs = _segments()
    text, _ = format_diarized_screenplay_with_offsets(segs)
    (tmp_path / "transcripts").mkdir(exist_ok=True)
    (tmp_path / REL).write_text(text, encoding="utf-8")
    seg_file = tmp_path / "transcripts" / "01 - ep.segments.json"
    with seg_file.open("w", encoding="utf-8") as fh:
        json.dump(segs, fh, indent=0, allow_nan=False)
    return text, segs


def _read_written(tmp_path: Path, outcome: turns_artifact.TurnsOutcome) -> Dict[str, Any]:
    """The written document — asserting it WAS written, which is half of what is under test."""
    assert outcome.relpath is not None, f"no artifact written: {outcome.unavailable_reason}"
    return dict(json.loads((tmp_path / outcome.relpath).read_text(encoding="utf-8")))


class TestTheArtifactOnDisk:
    def test_every_turn_span_indexes_the_text_on_disk(self, tmp_path: Path) -> None:
        """The claim the artifact makes, checked against the bytes a consumer would read."""
        text, segs = _lay_down_episode(tmp_path)
        outcome = write_turns_artifact(text, segs, REL, str(tmp_path), language="en")

        assert outcome.relpath == "transcripts/01 - ep.turns.json"
        on_disk_text = (tmp_path / REL).read_text(encoding="utf-8")
        doc = _read_written(tmp_path, outcome)

        assert len(doc["turns"]) == outcome.count > 2
        # Alternating speakers: no two consecutive turns share a label, which is what makes them
        # screenplay lines rather than an arbitrary grouping.
        labels = [t["speaker_label"] for t in doc["turns"]]
        assert all(a != b for a, b in zip(labels, labels[1:])), labels
        for turn in doc["turns"]:
            span = on_disk_text[turn["char_start"] : turn["char_end"]]
            assert span, turn["turn_id"]
            # The span is pure speech: the `Label: ` prefix is excluded by construction.
            assert not span.startswith(turn["speaker_label"] + ":")
            for sent in turn["sentences"]:
                assert on_disk_text[sent["char_start"] : sent["char_end"]].strip()

    def test_the_envelope_carries_provenance_a_consumer_can_recompute(self, tmp_path: Path) -> None:
        """``segments_sha256`` is the hash of the sidecar AS WRITTEN — the only thing a consumer
        can hash for itself to find out whether its turns are stale."""
        import hashlib

        text, segs = _lay_down_episode(tmp_path)
        outcome = write_turns_artifact(text, segs, REL, str(tmp_path), language="en")
        doc = _read_written(tmp_path, outcome)

        assert doc["version"] == turns_artifact.TURNS_SCHEMA_VERSION
        assert doc["episode_slug"] == "01 - ep"
        assert doc["language"] == "en"
        assert doc["source"]["transcript_ref"] == REL
        assert doc["source"]["segments_ref"] == "transcripts/01 - ep.segments.json"

        expected = hashlib.sha256(
            (tmp_path / "transcripts" / "01 - ep.segments.json").read_bytes()
        ).hexdigest()
        assert doc["source"]["segments_sha256"] == expected

    def test_a_missing_segments_sidecar_gives_a_null_hash_not_a_wrong_one(
        self, tmp_path: Path
    ) -> None:
        """Honest absence. A hash of the in-memory list would be a value no consumer can
        reproduce, which is worse than saying nothing."""
        segs = _segments()
        text, _ = format_diarized_screenplay_with_offsets(segs)
        (tmp_path / "transcripts").mkdir()
        (tmp_path / REL).write_text(text, encoding="utf-8")

        outcome = write_turns_artifact(text, segs, REL, str(tmp_path), language="en")
        doc = _read_written(tmp_path, outcome)
        assert doc["source"]["segments_sha256"] is None

    def test_the_backchannel_survives_into_the_file(self, tmp_path: Path) -> None:
        text, segs = _lay_down_episode(tmp_path)
        outcome = write_turns_artifact(text, segs, REL, str(tmp_path), language="en")
        doc = _read_written(tmp_path, outcome)

        flagged = [t for t in doc["turns"] if t["backchannel"]]
        assert [text[t["char_start"] : t["char_end"]] for t in flagged] == [BACKCHANNEL_TEXT]
        assert outcome.backchannels == 1
        assert outcome.count == len(doc["turns"])


class TestTheTwoCoordinateSpaces:
    def test_each_variant_gets_its_own_file_and_its_own_ids(self, tmp_path: Path) -> None:
        """RFC-123 §Key Decisions 5. The ad-free variant drops whole turns and renumbers from
        t0000, so a join by turn_id across variants would point at the wrong turn — which is only
        prevented by them being separate artifacts, each anchored to its own text.
        """
        text, segs = _lay_down_episode(tmp_path)
        cfg = config.Config(rss="https://e.com/f.xml", save_adfree_transcript=True)
        _produce_transcript_sidecars(cfg, text, segs, REL, str(tmp_path))

        raw = json.loads((tmp_path / "transcripts" / "01 - ep.turns.json").read_text())
        adfree = json.loads((tmp_path / "transcripts" / "01 - ep.adfree.turns.json").read_text())

        assert raw["source"]["transcript_ref"] == REL
        assert adfree["source"]["transcript_ref"] == "transcripts/01 - ep.adfree.txt"
        # The ad was its own turn, so the ad-free variant has one fewer…
        assert len(adfree["turns"]) == len(raw["turns"]) - 1
        # …and both still start at t0000 — which is precisely why a cross-variant join by
        # turn_id points at the WRONG turn, and why these are separate artifacts.
        assert raw["turns"][0]["turn_id"] == adfree["turns"][0]["turn_id"] == "t0000"
        assert raw["turns"][0]["speaker_label"] == "Announcer"
        assert adfree["turns"][0]["speaker_label"] == "Maya"

        # Each file's spans index ITS OWN text.
        adfree_text = (tmp_path / "transcripts" / "01 - ep.adfree.txt").read_text(encoding="utf-8")
        for turn in adfree["turns"]:
            span = adfree_text[turn["char_start"] : turn["char_end"]]
            assert span and not span.startswith(turn["speaker_label"] + ":")
        assert "ramp dot com" not in adfree_text

    def test_reading_the_raw_spans_against_the_adfree_text_would_be_wrong(
        self, tmp_path: Path
    ) -> None:
        """The failure the separation prevents, demonstrated rather than asserted about.

        If a consumer took the RAW artifact's spans and applied them to the AD-FREE text, at least
        one turn's text would silently change identity. This test exists so that a future
        'simplification' to one artifact per episode fails loudly here.
        """
        text, segs = _lay_down_episode(tmp_path)
        cfg = config.Config(rss="https://e.com/f.xml", save_adfree_transcript=True)
        _produce_transcript_sidecars(cfg, text, segs, REL, str(tmp_path))

        raw = json.loads((tmp_path / "transcripts" / "01 - ep.turns.json").read_text())
        adfree_text = (tmp_path / "transcripts" / "01 - ep.adfree.txt").read_text(encoding="utf-8")

        mismatched = [
            t
            for t in raw["turns"]
            if adfree_text[t["char_start"] : t["char_end"]] != text[t["char_start"] : t["char_end"]]
        ]
        assert mismatched, "the two spaces must actually differ, or this guard proves nothing"


class TestWhatItRefusesToWrite:
    def test_a_plain_transcript_gets_no_artifact_and_says_why(self, tmp_path: Path) -> None:
        """A provider transcript has no speaker lines, so 'consecutive same-speaker segments' is
        one turn covering the episode — a fact-shaped lie. RFC-123 §2: `turns: unavailable`."""
        segs = [
            {"start": 0.0, "end": 5.0, "text": "Welcome back to the show."},
            {"start": 5.0, "end": 9.0, "text": "Glad to be here."},
        ]
        plain_text = "Welcome back to the show. Glad to be here."
        (tmp_path / "transcripts").mkdir()
        (tmp_path / REL).write_text(plain_text, encoding="utf-8")

        outcome = write_turns_artifact(plain_text, segs, REL, str(tmp_path), language="en")
        assert outcome.relpath is None
        assert outcome.unavailable_reason == "not_a_diarized_screenplay"
        assert outcome.invariant_failures == 0
        assert not (tmp_path / turns_path(REL, str(tmp_path))).exists()

    def test_no_segments_is_unavailable_not_an_empty_artifact(self, tmp_path: Path) -> None:
        (tmp_path / "transcripts").mkdir()
        (tmp_path / REL).write_text("x", encoding="utf-8")
        outcome = write_turns_artifact("x", None, REL, str(tmp_path))
        assert outcome.relpath is None and outcome.unavailable_reason == "no_segments"

    def test_an_invariant_failure_writes_nothing_records_one_and_loses_no_episode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The deliberate v1 deviation from RFC-123 §Monitoring, pinned so the decision is
        visible rather than incidental: the failure is COUNTED, no partial file is left behind,
        and the caller is not handed an exception. When S1.4 gives turns a real consumer this
        test is the one that has to change.
        """
        text, segs = _lay_down_episode(tmp_path)

        def _boom(*_a: Any, **_k: Any) -> Any:
            raise TurnInvariantError("t0002 disagrees with the rendered text")

        monkeypatch.setattr(turns_artifact, "build_turns", _boom)
        outcome = write_turns_artifact(text, segs, REL, str(tmp_path), language="en")

        assert outcome.invariant_failures == 1
        assert outcome.relpath is None
        assert outcome.unavailable_reason == "invariant_failure"
        assert not (tmp_path / "transcripts" / "01 - ep.turns.json").exists()


class TestTheManifestBlock:
    def test_the_turns_block_reports_both_variants(self, tmp_path: Path) -> None:
        text, segs = _lay_down_episode(tmp_path)
        cfg = config.Config(rss="https://e.com/f.xml", save_adfree_transcript=True)
        _produce_transcript_sidecars(cfg, text, segs, REL, str(tmp_path))

        manifest = json.loads((tmp_path / "transcripts" / "01 - ep.manifest.json").read_text())
        block = manifest["stages"]["turns"]

        assert block["ran"] is True
        assert block["method_version"] == "turns-1"
        # Pure Python, milliseconds: measured and free, which is 0.0 — not None ("unmeasured").
        assert block["cost_usd"] == 0.0
        metrics = block["metrics"]
        assert metrics["count"] > 2
        assert metrics["backchannels"] == 1
        assert metrics["invariant_failures"] == 0
        assert metrics["median_turn_s"] > 0
        assert metrics["adfree"]["count"] < metrics["count"]

    def test_the_block_does_not_move_the_pipeline_composition_version(self, tmp_path: Path) -> None:
        """Adding a stage name to CANONICAL_STAGE_ORDER would rewrite every episode's composition
        hash and invalidate existing reprocess queries — for a sidecar nothing reads yet."""
        from podcast_scraper.workflow import processing_manifest as pm

        core = ["asr", "diarization", "naming", "summary", "gi", "kg"]
        assert pm.pipeline_composition_version(core) == pm.pipeline_composition_version(
            core + ["turns"]
        )

    def test_metrics_carry_the_reason_when_a_variant_is_unavailable(self) -> None:
        raw = turns_artifact.TurnsOutcome(unavailable_reason="not_a_diarized_screenplay")
        metrics = turns_manifest_metrics(raw)
        assert metrics["count"] == 0
        assert metrics["unavailable_reason"] == "not_a_diarized_screenplay"
        assert "adfree" not in metrics
