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


def test_maybe_produce_adfree_gate(tmp_path: Path):
    """The save-path wrapper honours the config flag and the no-segments guard."""
    from podcast_scraper import config
    from podcast_scraper.workflow.episode_processor import _maybe_produce_adfree

    segs = _content_segments()
    text, _ = format_diarized_screenplay_with_offsets(segs)
    rel = "transcripts/01 - ep.txt"
    (tmp_path / "transcripts").mkdir()
    (tmp_path / rel).write_text(text, encoding="utf-8")

    # Disabled → no ad-free sidecars written.
    cfg_off = config.Config(rss="https://e.com/f.xml", save_adfree_transcript=False)
    _maybe_produce_adfree(cfg_off, text, segs, rel, str(tmp_path))
    assert not (tmp_path / "transcripts" / "01 - ep.adfree.txt").exists()

    # Enabled → sidecars written.
    cfg_on = config.Config(rss="https://e.com/f.xml", save_adfree_transcript=True)
    _maybe_produce_adfree(cfg_on, text, segs, rel, str(tmp_path))
    assert (tmp_path / "transcripts" / "01 - ep.adfree.txt").exists()

    # No-segments guard → no crash, no file.
    rel2 = "transcripts/02 - ep.txt"
    (tmp_path / rel2).write_text(text, encoding="utf-8")
    _maybe_produce_adfree(cfg_on, text, None, rel2, str(tmp_path))
    assert not (tmp_path / "transcripts" / "02 - ep.adfree.txt").exists()


def _crosspromo_segments():
    """Opening host-read cross-promo (#1188): two readers who never recur, then the
    real two-host conversation. Ad readers have NONE of the density-ad markers."""
    segs = [
        {
            "start": 0.0,
            "end": 5.0,
            "speaker_label": "Paul",
            "text": "I'm Paul Tenorio. I cover soccer for The Athletic.",
        },
        {
            "start": 5.0,
            "end": 10.0,
            "speaker_label": "Amy",
            "text": "And I'm Amy Lawrence. I cover football for The Athletic.",
        },
        {
            "start": 10.0,
            "end": 16.0,
            "speaker_label": "Paul",
            "text": "The Athletic's coverage has everything you need for the tournament.",
        },
        {
            "start": 16.0,
            "end": 21.0,
            "speaker_label": "Amy",
            "text": "We've got more than 70 obsessive reporters on the ground.",
        },
        {
            "start": 21.0,
            "end": 26.0,
            "speaker_label": "Paul",
            "text": "Download The Athletic app and get free access to all the coverage.",
        },
    ]
    t = 26.0
    body = (
        "Hello and welcome everyone. Today we discuss the bioscience boom. "
        "Our guest spent twenty years in healthcare. Let us dive in. "
    ) * 12
    for i, sentence in enumerate(s for s in body.split(". ") if s.strip()):
        segs.append(
            {
                "start": t,
                "end": t + 15.0,
                "speaker_label": "Kevin" if i % 2 == 0 else "Casey",
                "text": sentence.strip() + ".",
            }
        )
        t += 15.0
    return segs


def test_opening_crosspromo_dropped_from_roster():
    """The #1188 harm: ad readers pollute the GI speaker roster. The ad-free base
    (what GI/enrich/search read) must drop their segments entirely."""
    segs = _crosspromo_segments()
    text, _ = format_diarized_screenplay_with_offsets(segs)
    arts = build_adfree_artifacts(text, segs)
    assert arts is not None

    seg_speakers = {s["speaker_label"] for s in arts.segments}
    seg_text = " ".join(s["text"] for s in arts.segments)
    # Roster: the non-recurring ad readers are gone.
    assert "Paul" not in seg_speakers
    assert "Amy" not in seg_speakers
    assert "The Athletic" not in arts.text
    assert "The Athletic" not in seg_text
    # The real hosts and content survive, offsets still exact.
    assert {"Kevin", "Casey"} <= seg_speakers
    assert "bioscience boom" in arts.text
    for s in arts.segments:
        assert arts.text[s["char_start"] : s["char_end"]] == s["text"]


def test_opening_crosspromo_dropped_in_plain_branch():
    """The plain / non-screenplay branch removes the opening cross-promo too, for
    consistency with the diarized branch (#1188). Text is the segment concatenation
    (not a ``Name:`` screenplay), but segments still carry speaker ids."""
    ad = [
        {
            "start": 0.0,
            "end": 5.0,
            "speaker": "SPEAKER_90",
            "text": "I'm Dana Lee and I cover the markets desk. ",
        },
        {
            "start": 5.0,
            "end": 10.0,
            "speaker": "SPEAKER_91",
            "text": "We report on every trading day. ",
        },
        {
            "start": 10.0,
            "end": 15.0,
            "speaker": "SPEAKER_90",
            "text": "Download our app to follow along. ",
        },
    ]
    body = [
        {
            "start": 15.0 + i * 15.0,
            "end": 28.0 + i * 15.0,
            "speaker": "SPEAKER_00",
            "text": f"Real host content number {i} today. ",
        }
        for i in range(12)
    ]
    segments = ad + body
    text = "".join(str(s["text"]) for s in segments)

    # Sanity: this is the plain branch (text is NOT the diarized screenplay).
    rebuilt, _ = format_diarized_screenplay_with_offsets(segments)
    assert rebuilt != text

    arts = build_adfree_artifacts(text, segments)
    assert arts is not None
    assert "Download our app" not in arts.text
    assert "Dana Lee" not in arts.text
    assert "Real host content number 0" in arts.text
    for s in arts.segments:
        assert arts.text[s["char_start"] : s["char_end"]] == s["text"]


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
