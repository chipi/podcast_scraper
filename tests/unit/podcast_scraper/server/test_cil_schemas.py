"""Unit tests for RFC-072 CIL response schemas (codecov patch coverage)."""

from __future__ import annotations

import pytest

from podcast_scraper.server.schemas import (
    CilArcEpisodeBlock,
    CilIdListResponse,
    CilPersonProfileInsightRow,
    CilPersonProfileQuoteRow,
    CilPersonProfileResponse,
    CilPositionArcResponse,
    CilTopicTimelineResponse,
)

pytestmark = [pytest.mark.unit]


def test_cil_position_arc_response_round_trip() -> None:
    m = CilPositionArcResponse(
        path="/data/corpus",
        person_id="person:a",
        topic_id="topic:b",
        episodes=[
            CilArcEpisodeBlock(
                episode_id="ep:1",
                publish_date="2024-01-01",
                insights=[{"id": "i1", "type": "Insight"}],
            )
        ],
    )
    d = m.model_dump()
    assert CilPositionArcResponse.model_validate(d).episodes[0].episode_id == "ep:1"


def test_cil_person_profile_response_round_trip() -> None:
    m = CilPersonProfileResponse(
        path="/c",
        person_id="person:p",
        topics={
            "topic:t": [
                CilPersonProfileInsightRow(
                    episode_id="ep:1",
                    insight={"id": "i"},
                    insight_type="claim",
                    position_hint=0.2,
                )
            ]
        },
        quotes=[CilPersonProfileQuoteRow(episode_id="ep:1", quote={"id": "q"})],
    )
    assert CilPersonProfileResponse.model_validate(m.model_dump()).person_id == "person:p"


def test_cil_topic_timeline_and_id_list() -> None:
    tl = CilTopicTimelineResponse(
        path="/c",
        topic_id="topic:x",
        episodes=[CilArcEpisodeBlock(episode_id="e1")],
    )
    assert CilTopicTimelineResponse.model_validate(tl.model_dump()).topic_id == "topic:x"
    ids = CilIdListResponse(path="/c", anchor_id="topic:x", ids=["person:a"])
    assert CilIdListResponse.model_validate(ids.model_dump()).ids == ["person:a"]
