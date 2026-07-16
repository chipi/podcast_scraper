"""Is the corroboration gate actually CALLED by the pipeline?

Written because it wasn't provable. ``corroborate_guests`` had thorough tests and the wiring
contract called it directly — so the gate could be deleted from ``processing.py`` outright and the
whole suite stayed green. A guardrail that is tested but not *reached* is decoration.

That is the same disease as every bug in this arc: a component that works, wired to nothing. So this
test does not call the gate. It drives ``_detect_speakers_for_episode`` — the real chokepoint where
a name becomes a diarized voice — with a detector that returns the LLM's actual hallucinated roster,
and asserts the impostors do not come out the other end.
"""

from __future__ import annotations

from typing import Any, List, Set, Tuple
from unittest.mock import MagicMock, patch

from podcast_scraper.workflow.stages.processing import _detect_speakers_for_episode

TITLE = "OpenAI's Big Reset + A.I. in the Doctor's Office"
DESCRIPTION = (
    "This week, OpenAI announced a loosened partnership with Microsoft. We unpack whether the "
    "company can scale while balancing a trial against Elon Musk and investor concerns. Then, the "
    "A.I. researcher Dr. Adam Rodman, of Harvard Medical School, returns to discuss how doctors "
    "are using chatbots."
)

# qwen3.5:35b's verbatim live output on this episode's real metadata.
LLM_ROSTER = ["Casey Newton", "Kevin Roose", "Elon Musk", "Sam Altman", "Dr. Adam Rodman"]


class _HallucinatingDetector:
    """A speaker detector that names people the episode merely discusses — and reports success."""

    def detect_speakers(
        self,
        episode_title: str,
        episode_description: str | None,
        known_hosts: Set[str],
    ) -> Tuple[List[str], Set[str], bool, bool]:
        return LLM_ROSTER, {"Kevin Roose", "Casey Newton"}, True, False


def _episode() -> Any:
    ep = MagicMock()
    ep.title = TITLE
    ep.idx = 0
    return ep


def _cfg() -> Any:
    cfg = MagicMock()
    cfg.auto_speakers = True
    cfg.dry_run = False
    cfg.known_hosts = ["Kevin Roose", "Casey Newton"]
    cfg.cache_detected_hosts = False
    cfg.screenplay_speaker_names = []
    return cfg


def _run() -> Any:
    host_result = MagicMock()
    host_result.cached_hosts = set()
    with (
        patch(
            "podcast_scraper.workflow.stages.processing._get_speaker_detector",
            return_value=_HallucinatingDetector(),
        ),
        patch(
            "podcast_scraper.workflow.stages.processing.extract_episode_description",
            return_value=DESCRIPTION,
        ),
    ):
        return _detect_speakers_for_episode(
            episode=_episode(),
            cfg=_cfg(),
            host_detection_result=host_result,
            pipeline_metrics=MagicMock(),
        )


def test_the_pipeline_rejects_the_speakers_the_llm_invented() -> None:
    """THE PRODUCTION BUG, at the chokepoint that produced it.

    Musk is being sued in this episode; Altman is discussed. Neither speaks. If either survives to
    here, their name is assigned to a diarized voice cluster and a real person's words are published
    under it.

    Read `.guests` EXPLICITLY. When this function started returning a `DetectedSpeakers` tuple, a
    bare ``"Elon Musk" not in result`` began comparing the string against the tuple's two LISTS —
    never equal, always true — and this guard, the most important one in the file, passed while
    testing nothing at all.
    """
    detected = _run()
    assert detected is not None
    assert "Elon Musk" not in detected.guests, f"the impostor reached the diarizer: {detected}"
    assert "Sam Altman" not in detected.guests, f"the impostor reached the diarizer: {detected}"


def test_the_invented_speakers_are_still_REMEMBERED_as_stated() -> None:
    """ADR-110: corroboration's rejects are kept, not discarded.

    `stated` is every name the metadata put forward, including the ones this gate threw away. The
    roster needs them — otherwise a guest we could not place is filed as a person NOBODY could have
    named, and our own failure is laundered into "not our fault".
    """
    detected = _run()
    assert "Elon Musk" in detected.stated
    assert "Dr. Adam Rodman" in detected.stated


def test_the_pipeline_still_returns_the_real_guest() -> None:
    """A pipeline that drops everyone would pass the test above and be worthless."""
    detected = _run()
    assert any(
        "Rodman" in g for g in detected.guests
    ), f"the real guest was lost: {detected.guests}"


def test_hosts_do_not_leak_into_the_guest_list() -> None:
    detected = _run()
    assert "Kevin Roose" not in detected.guests
    assert "Casey Newton" not in detected.guests
