"""Unit tests for the ad-free processing-base producer (#974)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.providers.ml.diarization.formatting import (
    format_diarized_screenplay_with_offsets,
)
from podcast_scraper.workflow.adfree_transcript import (
    adfree_transcript_relpath,
    build_adfree_artifacts,
    produce_adfree_transcript,
)

pytestmark = pytest.mark.unit

# A pre-roll ad cluster (≥3 distinct patterns in a tight span) the detector will cut.
_PREROLL = (
    "Ramp understands no one wants to chase receipts. Ramp saves companies 5 percent. "
    "Check out ramp dot com slash invest. They all use WorkOS for SSO and SCIM and RBAC. "
    "Visit WorkOS dot com to get started. Learn more at rogo dot ai slash Felix. "
)


def _content_segments():
    body = (
        "Hello and welcome everyone I am the host. Today we discuss the bioscience boom. "
        "Our guest has spent twenty years in healthcare investing. Let us dive in. "
    ) * 12
    segs = [{"start": 0.0, "end": 1.0, "text": _PREROLL, "speaker_label": "Patrick"}]
    t = 1.0
    for i, sentence in enumerate(body.split(". ")):
        sentence = sentence.strip()
        if not sentence:
            continue
        segs.append(
            {
                "start": t,
                "end": t + 1.0,
                "text": sentence,
                "speaker_label": "Patrick" if i % 2 == 0 else "Brian",
            }
        )
        t += 1.0
    return segs


def test_adfree_relpath():
    assert adfree_transcript_relpath("transcripts/01 - ep.txt") == "transcripts/01 - ep.adfree.txt"


def test_build_exact_offsets_and_ad_removed():
    segs = _content_segments()
    text, _ = format_diarized_screenplay_with_offsets(segs)
    arts = build_adfree_artifacts(text, segs)
    assert arts is not None
    assert arts.chars_removed > 0
    assert "Ramp understands" not in arts.text
    # Every ad-free segment maps EXACTLY into the ad-free text (no guard needed).
    for s in arts.segments:
        assert arts.text[s["char_start"] : s["char_end"]] == s["text"]
    # ad-map carries the excised ranges (raw space) for the future player.
    assert arts.ad_map["chars_removed"] == arts.chars_removed
    assert arts.ad_map["excised_ranges"]
    # speaker labels preserved on survivors
    assert any(s["speaker_label"] == "Brian" for s in arts.segments)


def test_build_returns_none_without_segments():
    assert build_adfree_artifacts("some text", None) is None
    assert build_adfree_artifacts("", [{"text": "x"}]) is None


def test_produce_writes_three_sidecars(tmp_path: Path):
    segs = _content_segments()
    text, _ = format_diarized_screenplay_with_offsets(segs)
    rel = "transcripts/01 - ep.txt"
    (tmp_path / "transcripts").mkdir()
    (tmp_path / rel).write_text(text, encoding="utf-8")

    adfree_rel = produce_adfree_transcript(text, segs, rel, str(tmp_path))
    assert adfree_rel == "transcripts/01 - ep.adfree.txt"
    base = tmp_path / "transcripts" / "01 - ep"
    assert (base.with_suffix(".adfree.txt")).exists()
    assert (tmp_path / "transcripts" / "01 - ep.adfree.segments.json").exists()
    admap = json.loads((tmp_path / "transcripts" / "01 - ep.adfree.admap.json").read_text())
    assert admap["chars_removed"] > 0

    # The saved ad-free text + segments round-trip the slice invariant.
    adfree_text = (tmp_path / "transcripts" / "01 - ep.adfree.txt").read_text(encoding="utf-8")
    adfree_segs = json.loads(
        (tmp_path / "transcripts" / "01 - ep.adfree.segments.json").read_text()
    )
    for s in adfree_segs:
        assert adfree_text[s["char_start"] : s["char_end"]] == s["text"]


def test_non_screenplay_text_falls_back_to_find(tmp_path: Path):
    # Plain whisper-style segments (no speaker labels); transcript is their concatenation.
    body = (
        "Hello and welcome everyone I am the host. Today we discuss the bioscience boom. "
        "Our guest has spent twenty years in healthcare investing. Let us dive in. "
    ) * 12
    plain_segments = [
        {"start": float(i), "end": float(i) + 1.0, "text": s + ". "}
        for i, s in enumerate(body.split(". "))
        if s.strip()
    ]
    plain_text = "".join(str(s["text"]) for s in plain_segments)
    arts = build_adfree_artifacts(plain_text, plain_segments)
    assert arts is not None
    for s in arts.segments:
        assert arts.text[s["char_start"] : s["char_end"]] == s["text"]
