"""Integration: the #974 ad-free transcript chain, end to end.

Wires the real pieces together on a diarized screenplay with an ad pre-roll:
producer → resolver → GI grounded build → enrich-edges. Asserts the two faults the
two-artifact model fixes:

- Fault A: a quote's ``char_start`` indexes the *saved* ad-free text exactly (so the
  viewer / search / enrich-edges that read it align).
- Fault B: the quote gets a ``speaker_id`` mapped through the screenplay ``Name:``
  markers — which the legacy cumulative-length guard would have dropped.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.gi.grounding import GroundedQuote
from podcast_scraper.gi.pipeline import build_artifact
from podcast_scraper.gi.speakers import add_spoken_by_edges
from podcast_scraper.providers.ml.diarization.formatting import (
    format_diarized_screenplay_with_offsets,
)
from podcast_scraper.workflow.adfree_transcript import (
    load_processing_transcript,
    produce_adfree_transcript,
)

pytestmark = pytest.mark.integration

_PREROLL = (
    "Ramp understands no one wants to chase receipts. Ramp saves companies 5 percent. "
    "Check out ramp dot com slash invest. They all use WorkOS for SSO and SCIM and RBAC. "
    "Visit WorkOS dot com to get started. Learn more at rogo dot ai slash Felix. "
)


def _segments():
    segs = [{"start": 0.0, "end": 3.0, "text": _PREROLL, "speaker_label": "Patrick"}]
    body = [
        ("Patrick", "Welcome to the show, today we explore how founders build durable companies."),
        ("Brian", "Thanks for having me, the key insight is that culture compounds over decades."),
        ("Patrick", "That is a fascinating point about the long term compounding of culture."),
        ("Brian", "We learned that hospitality is really about belonging and not transactions."),
    ] * 8  # keep the screenplay > MIN_TRANSCRIPT_CHARS so ad detection fires
    t = 3.0
    for spk, txt in body:
        segs.append({"start": t, "end": t + 2.0, "text": txt, "speaker_label": spk})
        t += 2.0
    return segs


def _grounding_cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.generate_gi = True
    cfg.gi_require_grounding = True
    cfg.gi_qa_model = "roberta-squad2"
    cfg.gi_nli_model = "nli-deberta-base"
    cfg.extractive_qa_device = None
    cfg.nli_device = None
    cfg.gi_fail_on_missing_grounding = False
    return cfg


def test_adfree_chain_end_to_end(tmp_path: Path) -> None:
    segs = _segments()
    text, _ = format_diarized_screenplay_with_offsets(segs)
    rel = "transcripts/0001 - ep.txt"
    (tmp_path / "transcripts").mkdir()
    (tmp_path / rel).write_text(text, encoding="utf-8")

    # 1. Producer writes the ad-free base (ads removed, exact offsets) next to raw .txt.
    adfree_rel = produce_adfree_transcript(text, segs, rel, str(tmp_path))
    assert adfree_rel == "transcripts/0001 - ep.adfree.txt"
    assert (tmp_path / rel).read_text() == text  # raw untouched

    # 2. Resolver loads the ad-free base; ads gone, segments carry char offsets.
    loaded = load_processing_transcript(str(tmp_path), rel)
    assert loaded.is_adfree
    assert "Ramp understands" not in loaded.text
    assert loaded.segments and all("char_start" in s for s in loaded.segments)

    # Pick a real quote inside a Brian turn so we can assert the speaker maps to Brian.
    brian_seg = next(s for s in loaded.segments if s["speaker_label"] == "Brian")
    quote_text = loaded.text[brian_seg["char_start"] : brian_seg["char_end"]]
    grounded = [
        GroundedQuote(
            char_start=brian_seg["char_start"],
            char_end=brian_seg["char_end"],
            text=quote_text,
            qa_score=0.8,
            nli_score=0.7,
        )
    ]

    # 3. GI grounded build on the ad-free base.
    with (
        patch(
            "podcast_scraper.gi.deps.create_gil_evidence_providers",
            return_value=(MagicMock(), MagicMock()),
        ),
        patch(
            "podcast_scraper.gi.grounding.find_grounded_quotes_via_providers",
            return_value=grounded,
        ),
    ):
        artifact = build_artifact(
            "episode:itb-0001",
            loaded.text,
            cfg=_grounding_cfg(),
            transcript_segments=loaded.segments,
            transcript_ref=adfree_rel,
        )

    quotes = [n for n in artifact["nodes"] if n["type"] == "Quote"]
    assert quotes
    for q in quotes:
        cs = q["properties"]["char_start"]
        ce = q["properties"]["char_end"]
        # Fault A: char_start indexes the SAVED ad-free text exactly.
        assert loaded.text[cs:ce] == q["properties"]["text"]
    # Fault B: speaker_id mapped through the Brian: marker (cumulative guard would drop it).
    sids = [q["properties"].get("speaker_id") for q in quotes]
    assert any(sid and "brian" in sid.lower() for sid in sids), sids
    # In-pipeline attribution already emits SPOKEN_BY (Quote -> Person) from the segments.
    assert any(e["type"] == "SPOKEN_BY" for e in artifact["edges"])


def test_enrich_edges_derives_spoken_by_from_adfree_offsets(tmp_path: Path) -> None:
    """Fault A for the corpus-wide path: a quote with char_start but no speaker_id (e.g.
    Gemini GI) gets SPOKEN_BY when enrich-edges reads the SAME ad-free text the offset
    was computed against. Pre-#974 this failed: char_start indexed the unsaved ad-free
    space while enrich-edges read the raw (ad-ful) transcript → offsets off by the ads."""
    segs = _segments()
    text, _ = format_diarized_screenplay_with_offsets(segs)
    rel = "transcripts/0001 - ep.txt"
    (tmp_path / "transcripts").mkdir()
    (tmp_path / rel).write_text(text, encoding="utf-8")
    produce_adfree_transcript(text, segs, rel, str(tmp_path))
    loaded = load_processing_transcript(str(tmp_path), rel)
    assert loaded.segments

    brian_seg = next(s for s in loaded.segments if s["speaker_label"] == "Brian")
    artifact = {
        "nodes": [
            {
                "id": "quote:1",
                "type": "Quote",
                "properties": {
                    "text": loaded.text[brian_seg["char_start"] : brian_seg["char_end"]],
                    "char_start": brian_seg["char_start"],
                    "char_end": brian_seg["char_end"],
                    "speaker_id": None,
                },
            }
        ],
        "edges": [],
    }
    n_spoken = add_spoken_by_edges(artifact, loaded.text, hosts=["Patrick"], guests=["Brian"])
    assert n_spoken == 1
    spoken = [e for e in artifact["edges"] if e["type"] == "SPOKEN_BY"]
    assert spoken and any("brian" in str(e.get("to", "")).lower() for e in spoken)
