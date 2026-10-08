"""Shape check for ``e2e/fixtures/production-shaped/search-v3/mocks.json``.

Every response in the fixture must validate against the shipped Pydantic
response model (``CorpusSearchApiResponse`` for GET /api/search scenarios,
``SearchCompareResponse`` for the compare scenario). This keeps the fixture
honest: if the server response shape shifts, the fixture breaks CI and
the Tier-2 specs are steered back into agreement.

Run: ``.venv/bin/pytest tests/integration/fixtures/test_search_v3_mocks_shape.py -q``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("fastapi")  # pydantic ships with fastapi in this repo

from podcast_scraper.server.schemas import (
    CorpusSearchApiResponse,
    SearchCompareResponse,
)

pytestmark = [pytest.mark.integration]

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = (
    REPO_ROOT
    / "web"
    / "gi-kg-viewer"
    / "e2e"
    / "fixtures"
    / "production-shaped"
    / "search-v3"
    / "mocks.json"
)


@pytest.fixture(scope="module")
def fixture_doc() -> dict[str, Any]:
    text = FIXTURE.read_text(encoding="utf-8")
    return cast(dict[str, Any], json.loads(text))


def test_fixture_file_exists_and_is_valid_json(fixture_doc: dict) -> None:
    assert isinstance(fixture_doc, dict)
    assert fixture_doc.get("schema_version") == "2", (
        "Fixture must declare schema_version=2 (post-S4/S5/S8 shape). "
        "Bump this + regenerate mocks.json when a scenario's shape changes."
    )
    scenarios = fixture_doc.get("scenarios")
    assert isinstance(scenarios, dict) and scenarios, "scenarios dict must be non-empty"


SEARCH_SCENARIOS = [
    "compound-lift",
    "enriched-answer",
    "temporal-intent",
]


@pytest.mark.parametrize("scenario_key", SEARCH_SCENARIOS)
def test_get_search_scenario_validates_against_response_model(
    fixture_doc: dict, scenario_key: str
) -> None:
    """Each GET /api/search scenario's response must round-trip through
    ``CorpusSearchApiResponse`` — that's what the FastAPI route actually
    returns. If Pydantic rejects a field, the fixture is out of sync with
    the server code."""
    scenarios = fixture_doc["scenarios"]
    scenario = scenarios[scenario_key]
    response = scenario["response"]
    CorpusSearchApiResponse.model_validate(response)


def test_enriched_answer_scenario_uses_per_hit_query_enrichments(fixture_doc: dict) -> None:
    """S5 shipped shape: enrichment lives on each hit's
    ``metadata.query_enrichments.related_topics[]``. There is no
    top-level ``enriched`` block (the shipped chain doesn't synthesize
    an answer yet — that's a later chunk of RFC-088)."""
    scenario = fixture_doc["scenarios"]["enriched-answer"]
    response = scenario["response"]
    assert "enriched" not in response, (
        "S5 shipped shape has no top-level 'enriched' block — remove it or "
        "downgrade the scenario to schema_version 1"
    )
    results = response.get("results") or []
    assert results, "enriched-answer scenario must include at least one hit"
    decorated = 0
    for hit in results:
        md = hit.get("metadata") or {}
        related = (md.get("query_enrichments") or {}).get("related_topics")
        if related:
            decorated += 1
            for entry in related:
                assert isinstance(entry.get("topic_id"), str)
                assert isinstance(entry.get("similarity"), (int, float))
    assert decorated >= 1, "at least one hit must carry metadata.query_enrichments.related_topics"


def test_operator_compare_scenario_validates_against_response_model(fixture_doc: dict) -> None:
    """The compare scenario response must round-trip through
    ``SearchCompareResponse`` — that's what ``POST /api/search/compare``
    actually returns."""
    scenario = fixture_doc["scenarios"]["operator-compare"]
    response = scenario["response"]
    SearchCompareResponse.model_validate(response)


def test_operator_compare_scenario_judge_summary_matches_grounded_rule(
    fixture_doc: dict,
) -> None:
    """RFC-107 §S8 acceptance: judge_summary is null when either pack
    reports grounded=false. Enforce the rule at the fixture level so a
    future edit can't break the invariant silently."""
    scenario = fixture_doc["scenarios"]["operator-compare"]
    response = scenario["response"]
    pack_a_grounded = bool(response["pack_a"].get("grounded"))
    pack_b_grounded = bool(response["pack_b"].get("grounded"))
    judge = response.get("judge_summary")
    if not (pack_a_grounded and pack_b_grounded):
        assert judge is None, "judge_summary must be null when either pack reports grounded=false"
    else:
        assert (
            isinstance(judge, str) and judge
        ), "judge_summary must be a non-empty string when both packs are grounded"
