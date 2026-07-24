"""Endpoint-level tests for ``POST /api/search/compare`` (Search v3 §S8).

Unit-scope: mounts the FastAPI app via ``create_app`` on a tmp corpus and
monkeypatches ``compare_subjects`` — so we assert the endpoint's
request-body → orchestrator-kwargs contract (including
``insight_types``) without touching LanceDB. Orchestrator internals are
covered by ``test_compare.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from podcast_scraper.search.compare import (
    BriefingPack,
    CompareOutcome,
    SubjectRef,
)
from podcast_scraper.server.app import create_app


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "metadata").mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture()
def client(corpus: Path) -> TestClient:
    return TestClient(create_app(corpus, static_dir=False))


def _grounded_pack(subject: SubjectRef) -> BriefingPack:
    return BriefingPack(
        subject=subject,
        query="q",
        query_type="semantic",
        rendered="[CRITICAL] x",
        token_count=3,
        max_tokens=2000,
        top_insight_id=f"insight:{subject.id}:top",
        top_insight_text=f"top insight for {subject.id}",
        coverage_summary={"episode_count": 2, "show_ids": [], "date_range": None},
        confidence_p50=0.8,
        result_count=2,
        grounded=True,
    )


@pytest.fixture()
def compare_calls(monkeypatch: pytest.MonkeyPatch) -> List[Dict[str, Any]]:
    """Install a fake ``compare_subjects`` that records each call's kwargs
    and returns a two-grounded-packs outcome so the response shape is
    exercised. Every test that needs to assert what the endpoint forwards
    to the orchestrator asks for this fixture."""
    calls: List[Dict[str, Any]] = []

    def _fake(
        root: Path,
        subject_a: SubjectRef,
        subject_b: SubjectRef,
        **kwargs: Any,
    ) -> CompareOutcome:
        calls.append(
            {
                "root": root,
                "subject_a": subject_a,
                "subject_b": subject_b,
                **kwargs,
            }
        )
        return CompareOutcome(
            pack_a=_grounded_pack(subject_a),
            pack_b=_grounded_pack(subject_b),
            judge_summary="deterministic summary",
        )

    monkeypatch.setattr(
        "podcast_scraper.server.routes.search.compare_subjects",
        _fake,
    )
    return calls


class TestSearchCompareEndpointContract:
    def test_happy_path_returns_two_packs_and_judge_summary(
        self, client: TestClient, compare_calls: List[Dict[str, Any]]
    ) -> None:
        resp = client.post(
            "/api/search/compare",
            json={
                "subject_a": {"kind": "person", "id": "person:alice", "label": "Alice"},
                "subject_b": {"kind": "person", "id": "person:bob", "label": "Bob"},
                "q": "compute",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["pack_a"]["subject"]["id"] == "person:alice"
        assert body["pack_b"]["subject"]["id"] == "person:bob"
        assert body["pack_a"]["grounded"] is True
        assert body["pack_b"]["grounded"] is True
        assert body["judge_summary"] == "deterministic summary"
        assert body["error"] is None
        # And the orchestrator was called exactly once, with the request's q.
        assert len(compare_calls) == 1
        assert compare_calls[0]["q"] == "compute"

    def test_bad_corpus_path_rejected_by_path_resolver(self, tmp_path: Path) -> None:
        """An explicit ``path`` that fails ``resolve_corpus_path_param``
        gets a 400 from the path resolver — the endpoint's downstream
        ``no_corpus_path`` branch is defensive-only and unreachable via
        HTTP (``create_app`` always seeds ``app.state.output_dir``)."""
        client = TestClient(create_app(tmp_path, static_dir=False))
        resp = client.post(
            "/api/search/compare",
            json={
                "subject_a": {"kind": "person", "id": "A"},
                "subject_b": {"kind": "person", "id": "B"},
                "path": "/definitely/does/not/exist/nowhere",
            },
        )
        assert resp.status_code == 400


class TestSearchCompareEndpointInsightTypes:
    def test_insight_types_propagates_to_orchestrator(
        self, client: TestClient, compare_calls: List[Dict[str, Any]]
    ) -> None:
        resp = client.post(
            "/api/search/compare",
            json={
                "subject_a": {"kind": "person", "id": "A"},
                "subject_b": {"kind": "person", "id": "B"},
                "insight_types": ["claim", "recommendation"],
            },
        )
        assert resp.status_code == 200
        assert compare_calls[0]["insight_types"] == ["claim", "recommendation"]

    def test_insight_types_omitted_defaults_to_none(
        self, client: TestClient, compare_calls: List[Dict[str, Any]]
    ) -> None:
        resp = client.post(
            "/api/search/compare",
            json={
                "subject_a": {"kind": "person", "id": "A"},
                "subject_b": {"kind": "person", "id": "B"},
            },
        )
        assert resp.status_code == 200
        assert compare_calls[0]["insight_types"] is None

    def test_insight_types_explicit_null_defaults_to_none(
        self, client: TestClient, compare_calls: List[Dict[str, Any]]
    ) -> None:
        resp = client.post(
            "/api/search/compare",
            json={
                "subject_a": {"kind": "person", "id": "A"},
                "subject_b": {"kind": "person", "id": "B"},
                "insight_types": None,
            },
        )
        assert resp.status_code == 200
        assert compare_calls[0]["insight_types"] is None

    def test_insight_types_empty_list_reaches_orchestrator_as_empty(
        self, client: TestClient, compare_calls: List[Dict[str, Any]]
    ) -> None:
        """The orchestrator treats ``[]`` as a no-op; the endpoint's
        contract is faithful pass-through, so we assert the empty list
        arrives unmodified rather than getting coerced to ``None``.
        Behavioural equivalence is tested at the orchestrator layer."""
        resp = client.post(
            "/api/search/compare",
            json={
                "subject_a": {"kind": "person", "id": "A"},
                "subject_b": {"kind": "person", "id": "B"},
                "insight_types": [],
            },
        )
        assert resp.status_code == 200
        assert compare_calls[0]["insight_types"] == []

    def test_top_k_and_max_tokens_propagate(
        self, client: TestClient, compare_calls: List[Dict[str, Any]]
    ) -> None:
        resp = client.post(
            "/api/search/compare",
            json={
                "subject_a": {"kind": "person", "id": "A"},
                "subject_b": {"kind": "person", "id": "B"},
                "top_k": 25,
                "max_tokens": 3500,
            },
        )
        assert resp.status_code == 200
        assert compare_calls[0]["top_k"] == 25
        assert compare_calls[0]["max_tokens"] == 3500


class TestSearchCompareEndpointValidation:
    def test_missing_subject_a_rejected_by_pydantic(self, client: TestClient) -> None:
        resp = client.post(
            "/api/search/compare",
            json={"subject_b": {"kind": "person", "id": "B"}},
        )
        assert resp.status_code == 422

    def test_top_k_over_upper_bound_rejected(self, client: TestClient) -> None:
        resp = client.post(
            "/api/search/compare",
            json={
                "subject_a": {"kind": "person", "id": "A"},
                "subject_b": {"kind": "person", "id": "B"},
                "top_k": 101,
            },
        )
        assert resp.status_code == 422

    def test_max_tokens_over_upper_bound_rejected(self, client: TestClient) -> None:
        resp = client.post(
            "/api/search/compare",
            json={
                "subject_a": {"kind": "person", "id": "A"},
                "subject_b": {"kind": "person", "id": "B"},
                "max_tokens": 9001,
            },
        )
        assert resp.status_code == 422
