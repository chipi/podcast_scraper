"""Unit tests for the ad-free processing-base producer (#974)."""

from __future__ import annotations

import pytest

from podcast_scraper.workflow.transcript_resolution import adfree_transcript_relpath

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
