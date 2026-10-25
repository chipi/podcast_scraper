"""The closed candidate list holds each person once (synthetic names)."""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.diarization.pipeline import one_entry_per_person

pytestmark = [pytest.mark.unit]


def test_a_title_variant_of_a_listed_person_is_dropped() -> None:
    assert one_entry_per_person(["Dr. Ada Wren", "Tobias Holt", "Ada Wren"]) == [
        "Dr. Ada Wren",
        "Tobias Holt",
    ]


def test_a_spelling_with_stray_capitals_gives_way_to_the_clean_one() -> None:
    assert one_entry_per_person(["Ada WrenNE", "Ada Wrenne"]) == ["Ada Wrenne"]


def test_a_legitimate_inner_capital_is_kept() -> None:
    assert one_entry_per_person(["June McGarry", "June Mcgarry"]) == ["June McGarry"]


@pytest.mark.parametrize(
    "names",
    [
        ["Sam Lee", "Sam Lee Jr"],  # a father and a son
        ["Ada", "Ada Wren"],  # a one-word name is not merged with a full name
        ["Ada Wren", "Tom Wren"],  # relatives
    ],
)
def test_different_people_stay_apart(names) -> None:
    assert one_entry_per_person(names) == names


def test_the_order_is_kept() -> None:
    assert one_entry_per_person(["C Dee", "A Bee", "B Cee"]) == ["C Dee", "A Bee", "B Cee"]


def test_the_stage_gives_the_resolver_each_person_once(monkeypatch, tmp_path) -> None:
    from typing import Any, Dict, List, Optional

    from podcast_scraper.config import Config
    from podcast_scraper.providers.ml.diarization import pipeline as P
    from podcast_scraper.providers.ml.diarization.base import (
        DiarizationResult,
        DiarizationSegment,
    )

    seen: List[List[str]] = []

    def fake_llm(*a: Any, stated_names=None, report: Optional[Dict[str, Any]] = None, **k: Any):
        seen.append(list(stated_names or []))
        return {}, {}

    monkeypatch.setattr(P, "_resolve_voices_via_llm", fake_llm)
    P._copresence_cache.clear()
    P.apply_diarization_to_result(
        {"text": "x", "segments": [{"start": 0, "end": 60, "text": "Welcome, Ada."}]},
        "",
        Config(output_dir=str(tmp_path)),
        ["Dr. Ada Wren", "Ada Wren"],
        metadata_named=["Dr. Ada Wren", "Ada Wren"],
        precomputed_diarization=DiarizationResult(
            segments=[DiarizationSegment(0, 60, "SPEAKER_00")], num_speakers=1
        ),
        episode_title="Ep 1",
    )
    assert seen and all(sum(1 for n in names if "Ada" in n) == 1 for names in seen), seen
