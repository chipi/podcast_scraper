"""Candidate widening (``widen_candidates_from_metadata``): the names an episode's own text
presents as people on it.

Synthetic names only (never-commit-real-episodes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper.providers.ml.diarization import candidate_widening as cw, pipeline as P
from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "text, names",
    [
        ("A conversation with Maria Lindqvist about soil.", ["Maria Lindqvist"]),
        ("This week we are joined by Dr. Tobias Wren and Ann Lee.", ["Tobias Wren"]),
        ("Guest: Ana Ruiz", []),  # no cue word in front of the name ("Guest:" has a colon)
        ("Our guest Ana Ruiz explains.", ["Ana Ruiz"]),
        ("Ana Ruiz joins the show to talk water.", ["Ana Ruiz"]),
        ("Featuring Kim J. Park.", ["Kim J. Park"]),
        ("With Ana Ruiz", ["Ana Ruiz"]),
        ("Episode 12 — with Ana Ruiz", ["Ana Ruiz"]),
        ("Soil health, with Maria Lindqvist.", ["Maria Lindqvist"]),
        ("Toby sits down with Ana Ruiz.", ["Ana Ruiz"]),
        ("Toby catches up with Ana Ruiz.", ["Ana Ruiz"]),
        ("Ana Ruiz joins us to talk water.", ["Ana Ruiz"]),
        ("Ana Ruiz is back!", ["Ana Ruiz"]),
        ("Featuring Ana Ruiz and Bo Lee", ["Ana Ruiz"]),
        ("Featuring Ana Ruiz, author of Water.", ["Ana Ruiz"]),
    ],
)
def test_cue_names(text: str, names: List[str]) -> None:
    assert cw.cue_names(text) == names


@pytest.mark.parametrize(
    "text",
    [
        # #876: an episode ABOUT a person must not put him on the list
        "We discuss the lawsuit Elon Musk filed against OpenAI.",
        "A look back at Winston Churchill.",
        # a bare "with" after a noun is text about a person (review m1)
        "His feud with Elon Musk continues.",
        "The deal with Ana Ruiz fell through.",
        # second review M1: "joins / returns / featuring" followed by text about the person,
        # and a sentence opener captured into the name
        "Why Elon Musk joins the AI race",
        "Taylor Swift returns to Nashville: a retrospective",
        "An evening featuring Taylor Swift songs",
        "How Ana Ruiz built a water company",
        # organisations and shows after a cue word
        "An interview with The Economist editors.",
        "In conversation with Our World in Data.",
    ],
)
def test_text_about_people_or_organisations_adds_nothing(text: str) -> None:
    assert cw.widen([], None, text, language="en") == []


def test_names_already_on_the_list_are_not_added_again() -> None:
    desc = "A conversation with Maria Lindqvist and featuring Tobias Wren."
    assert cw.widen(["Maria Lindqvist", "Toby Wren"], "Ep 1", desc, language="en") == []
    assert cw.widen(["Maria Lindqvist"], None, desc, language="en") == ["Tobias Wren"]


def _stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, on: bool) -> Dict[str, Any]:
    from podcast_scraper.config import Config

    seen: List[Any] = []

    def fake_llm(cfg: Any, stated_names: Any = None, **k: Any) -> Any:
        seen.append(list(stated_names or []))
        return {}, {}

    monkeypatch.setattr(P, "_resolve_voices_via_llm", fake_llm)
    P._copresence_cache.clear()
    out: Dict[str, Any] = P.apply_diarization_to_result(
        {"text": "x", "segments": [{"start": 0, "end": 60, "text": "Hello and welcome."}]},
        "",
        Config(
            output_dir=str(tmp_path),
            speaker_resolution_llm=False,
            widen_candidates_from_metadata=on,
        ),
        [],
        precomputed_diarization=DiarizationResult(
            segments=[DiarizationSegment(0, 60, "SPEAKER_00")], num_speakers=1
        ),
        episode_title="Ep 1",
        episode_description="A conversation with Maria Lindqvist about soil.",
    )
    out["_stated"] = seen
    return out


def test_the_stage_widens_only_when_on_and_records_what_it_added(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    off = _stage(monkeypatch, tmp_path, False)
    on = _stage(monkeypatch, tmp_path, True)
    assert "Maria Lindqvist" not in off["_stated"][0]
    assert "Maria Lindqvist" in on["_stated"][0]
    assert "candidate_widening" not in off["speaker_diagnostics"]["decision_trace"]["inputs"]
    assert on["speaker_diagnostics"]["decision_trace"]["inputs"]["candidate_widening"] == [
        "Maria Lindqvist"
    ]


def test_off_by_default() -> None:
    from podcast_scraper import config

    assert config.Config.model_fields["widen_candidates_from_metadata"].default is False
