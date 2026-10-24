"""A host only the episode's own text states is in the pool, but not a name the LLM may bind (v51).

Step 3 of the 2026-10-24 DGX block (paired relabel, 28 episodes): the one served effect of the
episode host statements (v43) that came through the LLM was a wrong name; their gains (a host
seated, a spelling) came through the pool, the guard and the roster. So a name the pool gains
ONLY from this episode's statement or ``<podcast:person role="host">`` stays in ``known_hosts``
and leaves the resolver's candidate list (synthetic shows).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest

from podcast_scraper.workflow.stages.processing import hosts_for_episode
from podcast_scraper.workflow.types import HostDetectionResult

pytestmark = pytest.mark.unit


def _episode(description: str) -> SimpleNamespace:
    item = ET.Element("item")
    ET.SubElement(item, "description").text = description
    return SimpleNamespace(item=item, title="Rivers of the old world")


def test_a_host_only_the_statement_names_is_pooled_and_unbindable() -> None:
    ep = _episode("A conversation hosted by Tobias Holt.")
    result = HostDetectionResult(
        cached_hosts={"Ada Wren"}, heuristics=None, feed_title="River Notes"
    )
    assert hosts_for_episode(result, ep) == {"Ada Wren", "Tobias Holt"}
    assert ep.unbindable_hosts == ["Tobias Holt"]


def test_a_feed_host_the_statement_repeats_stays_bindable() -> None:
    ep = _episode("This episode is hosted by Ada Wren.")
    result = HostDetectionResult(
        cached_hosts={"Ada Wren"}, heuristics=None, feed_title="River Notes"
    )
    assert hosts_for_episode(result, ep) == {"Ada Wren"}
    assert ep.unbindable_hosts == []


def test_the_pipeline_leaves_unbindable_hosts_out_of_the_candidates() -> None:
    from podcast_scraper.providers.ml.diarization.pipeline import resolution_candidates

    got = resolution_candidates(
        metadata_named=["Mira Lane"],
        guests=[],
        known_hosts=["Ada Wren", "Tobias Holt"],
        unbindable_hosts=["Tobias Holt"],
    )
    assert got == ["Mira Lane", "Ada Wren"]


def test_a_name_detection_also_states_stays_a_candidate() -> None:
    """Metadata detection naming the person is a source of its own; only the statement is not."""
    from podcast_scraper.providers.ml.diarization.pipeline import resolution_candidates

    got = resolution_candidates(
        metadata_named=["Tobias Holt"],
        guests=[],
        known_hosts=["Ada Wren", "Tobias Holt"],
        unbindable_hosts=["Tobias Holt"],
    )
    assert got == ["Tobias Holt", "Ada Wren"]
