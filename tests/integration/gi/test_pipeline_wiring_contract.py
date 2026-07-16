"""End-to-end: does the chain actually CONNECT, and does the net catch it when it doesn't?

The unit fixtures test each stage against traps someone invented. The invariants test each artifact
against structural impossibility. Neither proves the stages are *wired to each other* — and that is
precisely how the corpus broke. Every stage passed its own tests while the pipeline as a whole
published Elon Musk's name on a doctor's words.

This walks one adversarial episode all the way through:

    description (names an impostor AND a real guest)
        -> speaker detection            (proposes both)
        -> corroboration                (must kill the impostor)
        -> diarized transcript turns    (who actually holds the mic)
        -> GI artifact                  (insight + grounded quote + speaker)
        -> invariants                   (must be silent)

and then does it again with the corroboration gate removed, asserting the net CATCHES it. A
guardrail nobody has watched fail is not a guardrail. That negative control is the point of the
file: it is the difference between "we added checks" and "we know the checks work".
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.gi.grounding import GroundedQuote
from podcast_scraper.gi.invariants import check_artifact_invariants
from podcast_scraper.gi.pipeline import build_artifact
from podcast_scraper.providers.ml.diarization.formatting import (
    format_diarized_screenplay_with_offsets,
)
from podcast_scraper.speaker_detectors.corroboration import corroborate_guests

pytestmark = pytest.mark.integration

# The real Hard Fork episode, reduced. Musk is the man SUING OpenAI. Rodman is the guest.
TITLE = "OpenAI's Big Reset + A.I. in the Doctor's Office"
DESCRIPTION = (
    "This week, OpenAI announced a loosened partnership with Microsoft and an aggressive new "
    "strategy to secure computing power. We unpack whether the company can scale while balancing "
    "a trial against Elon Musk and investor concerns over missed financial targets. Then, the "
    "A.I. researcher Dr. Adam Rodman, of Harvard Medical School, returns to discuss how doctors "
    "are using chatbots."
)
KNOWN_HOSTS = {"Kevin Roose", "Casey Newton"}

# What the LLM speaker detector actually returned, live, on this metadata.
LLM_PROPOSED = ["Casey Newton", "Kevin Roose", "Elon Musk", "Sam Altman", "Dr. Adam Rodman"]

GUEST_LINE = "Doctors are already using chatbots for differential diagnosis and it is working."


def _segments() -> List[Dict[str, Any]]:
    body = [
        ("Kevin Roose", "This week OpenAI announced a loosened partnership with Microsoft."),
        ("Casey Newton", "And Elon Musk is suing them, which complicates the whole picture."),
        ("Dr. Adam Rodman", GUEST_LINE),
        ("Kevin Roose", "That is a fascinating point about how clinicians actually work today."),
    ] * 6  # keep the screenplay long enough for the real chain to behave normally
    segs, t = [], 0.0
    for spk, txt in body:
        segs.append({"start": t, "end": t + 2.0, "text": txt, "speaker_label": spk})
        t += 2.0
    return segs


def _grounding_cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.generate_gi = True
    cfg.gi_require_grounding = True
    cfg.gi_fail_on_missing_grounding = False
    cfg.extractive_qa_device = None
    cfg.nli_device = None
    return cfg


def _artifact_for(roster: List[str]) -> Dict[str, Any]:
    """Build a GI artifact where the grounded quote is the GUEST's line.

    ``roster`` is what speaker detection handed the diarizer. In production the roster names are
    assigned to voice clusters positionally, so a bogus name does not sit harmlessly in a list —
    it *replaces* a real speaker's label on their own turns. That substitution is modelled here.
    """
    segs = _segments()
    if "Elon Musk" in roster:
        # The poisoning, exactly as it happened: the impostor takes the guest's voice cluster.
        for s in segs:
            if s["speaker_label"] == "Dr. Adam Rodman":
                s["speaker_label"] = "Elon Musk"

    text, offset_segs = format_diarized_screenplay_with_offsets(segs)
    guest_seg = next(s for s in offset_segs if s["text"] == GUEST_LINE)
    grounded = [
        GroundedQuote(
            char_start=guest_seg["char_start"],
            char_end=guest_seg["char_end"],
            text=text[guest_seg["char_start"] : guest_seg["char_end"]],
            qa_score=0.8,
            nli_score=0.7,
        )
    ]

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
        return build_artifact(
            "episode:hardfork-0001",
            text,
            cfg=_grounding_cfg(),
            transcript_segments=offset_segs,
            transcript_ref="transcripts/0001 - ep.txt",
        )


def _speakers_in(artifact: Dict[str, Any]) -> set[str]:
    out = set()
    for n in artifact.get("nodes") or []:
        if n["type"] == "Insight":
            spk = (n.get("properties") or {}).get("speaker")
            if spk:
                out.add(str(spk))
        if n["type"] == "Person":
            out.add(str((n.get("properties") or {}).get("name") or ""))
    return {s for s in out if s}


def test_the_gate_kills_the_impostor_the_llm_proposed() -> None:
    """Stage 1->2. The detector proposes Musk and Altman; only the corroborated guest survives."""
    guests = corroborate_guests(LLM_PROPOSED, TITLE, DESCRIPTION, known_hosts=KNOWN_HOSTS)
    assert guests == ["Dr. Adam Rodman"], guests


def test_the_connected_chain_attributes_the_guest_and_is_silent() -> None:
    """The whole chain, wired. The doctor's words carry the doctor's name, and nothing complains."""
    roster = corroborate_guests(LLM_PROPOSED, TITLE, DESCRIPTION, known_hosts=KNOWN_HOSTS)
    artifact = _artifact_for(roster)

    speakers = _speakers_in(artifact)
    assert "Dr. Adam Rodman" in speakers
    assert "Elon Musk" not in speakers
    assert "Sam Altman" not in speakers

    assert check_artifact_invariants(artifact) == []


def test_the_stage_actually_RUNS_the_invariants(caplog: pytest.LogCaptureFixture) -> None:
    """Is the checker CALLED, or does it merely exist?

    Found by re-breaking the fix: deleting ``log_artifact_invariants`` from ``build_artifact`` left
    every invariant test green, because they all called the checker directly. The check was never
    reached by the pipeline it was written to protect — the same "works, wired to nothing" failure
    as the bugs it is there to catch.

    So: hand ``build_artifact`` a quote whose offset points at somebody else's turn, and assert the
    STAGE ITSELF complains. If the wiring is cut, no ERROR is logged and this fails.
    """
    segs = _segments()
    text, offset_segs = format_diarized_screenplay_with_offsets(segs)
    guest_seg = next(s for s in offset_segs if s["text"] == GUEST_LINE)

    # The quote is the doctor's line, but char_start points at the top of the transcript — Kevin's
    # turn. This is the offset bug, and every speaker derived from it is a coin flip.
    poisoned = [
        GroundedQuote(
            char_start=0,
            char_end=len(GUEST_LINE),
            text=text[guest_seg["char_start"] : guest_seg["char_end"]],
            qa_score=0.8,
            nli_score=0.7,
        )
    ]

    with (
        patch(
            "podcast_scraper.gi.deps.create_gil_evidence_providers",
            return_value=(MagicMock(), MagicMock()),
        ),
        patch(
            "podcast_scraper.gi.grounding.find_grounded_quotes_via_providers",
            return_value=poisoned,
        ),
        caplog.at_level("ERROR"),
    ):
        build_artifact(
            "episode:hardfork-0001",
            text,
            cfg=_grounding_cfg(),
            transcript_segments=offset_segs,
            transcript_ref="transcripts/0001 - ep.txt",
        )

    assert "GI INVARIANT VIOLATED" in caplog.text, (
        "build_artifact emitted a structurally impossible artifact WITHOUT complaining — the "
        "invariant check is not wired into the stage"
    )


def test_the_net_catches_the_chain_when_the_gate_is_removed() -> None:
    """THE NEGATIVE CONTROL — the guardrail is watched failing.

    Bypass corroboration (i.e. ship the code as it was yesterday) and the LLM's raw roster reaches
    the diarizer. Musk takes the doctor's voice cluster and the doctor's insight is published under
    his name. The artifact is internally consistent — Musk *does* hold that turn in the poisoned
    transcript — which is exactly why nothing caught it for two days.

    What catches it is the transcript-independent truth: Musk was never introduced as a speaker.
    Re-checking the roster against the description is what turns a silent corpus bug into a failure.
    """
    artifact = _artifact_for(LLM_PROPOSED)  # no gate

    speakers = _speakers_in(artifact)
    assert "Elon Musk" in speakers, "precondition: the ungated chain really does misattribute"
    assert "Dr. Adam Rodman" not in speakers

    uncorroborated = [
        s
        for s in speakers
        if s not in KNOWN_HOSTS and not corroborate_guests([s], TITLE, DESCRIPTION)
    ]
    assert "Elon Musk" in uncorroborated, (
        "the corroboration gate must reject a speaker the episode never introduces — this is the "
        "check that stands between the pipeline and another misattributed corpus"
    )
