"""Integration tests for the RFC-088 v2 enrichment-config routes."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = pytest.mark.integration


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    (tmp_path / "metadata").mkdir(exist_ok=True)
    app = create_app(output_dir=tmp_path, enable_jobs_api=True)
    return TestClient(app)


def test_provider_types_route_groups_by_protocol(client: TestClient) -> None:
    r = client.get("/api/enrichment/provider-types")
    assert r.status_code == 200, r.text
    body = r.json()
    by_proto = body["by_protocol"]
    assert "EmbeddingProvider" in by_proto
    assert "NliScorer" in by_proto
    emb_names = {t["name"] for t in by_proto["EmbeddingProvider"]}
    assert {"fake_for_test", "sentence_transformer_local"} <= emb_names
    nli_names = {t["name"] for t in by_proto["NliScorer"]}
    assert {"fixed_scripted", "deberta_local"} <= nli_names
    # Every type carries the UI-relevant fields.
    for t in by_proto["EmbeddingProvider"] + by_proto["NliScorer"]:
        assert "description" in t
        assert "params_schema" in t


def test_admission_route_reports_gate_status(client: TestClient) -> None:
    r = client.get("/api/enrichment/config/admission")
    assert r.status_code == 200, r.text
    rows = {row["id"]: row for row in r.json()["enrichers"]}
    # All nine known enrichers reported.
    assert len(rows) == 9
    # topic_consensus declares a gate and is NOT promoted (no eval yet → on_missing=reject,
    # ADR-108) — the reason is surfaced for the UI, not a silent absence.
    tc = rows["topic_consensus"]
    assert tc["has_gate"] is True
    assert tc["promoted"] is False
    assert "reject" in tc["reason"] or "gated" in tc["reason"]
    # stance_timeline is likewise gated dark until its eval clears.
    assert rows["stance_timeline"]["has_gate"] is True
    assert rows["stance_timeline"]["promoted"] is False
    # Deterministic enrichers declare no gate → promoted.
    gr = rows["grounding_rate"]
    assert gr["has_gate"] is False
    assert gr["promoted"] is True


def test_schema_route_composes_per_enricher_blocks(client: TestClient) -> None:
    r = client.get("/api/enrichment/config/schema")
    assert r.status_code == 200, r.text
    schema = r.json()
    enrichers = schema["properties"]["enrichers"]["properties"]
    # The known enrichers surface in the schema.
    expected = {
        "topic_cooccurrence_corpus",
        "temporal_velocity",
        "grounding_rate",
        "guest_coappearance",
        "insight_density",
        "topic_similarity",
        "topic_consensus",
        "stance_timeline",
    }
    assert expected <= set(enrichers.keys())
    # temporal_velocity knobs surface in its block.
    tv = enrichers["temporal_velocity"]["properties"]
    assert "alpha" in tv
    assert "window_months" in tv
    # ML enrichers carry a provider oneOf list.
    ts = enrichers["topic_similarity"]["properties"]
    assert "provider" in ts
    assert "oneOf" in ts["provider"]
    types_offered = {alt["properties"]["type"]["const"] for alt in ts["provider"]["oneOf"]}
    assert "fake_for_test" in types_offered
    assert "sentence_transformer_local" in types_offered


def test_get_config_returns_empty_when_no_operator_yaml(client: TestClient, tmp_path: Path) -> None:
    r = client.get(f"/api/enrichment/config?path={tmp_path}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["operator_block"] == {}
    assert body["resolved_block"] == {}  # no profile, no operator → empty


def test_put_config_persists_block_to_operator_yaml(client: TestClient, tmp_path: Path) -> None:
    block = {
        "enabled": True,
        "enrichers": {
            "temporal_velocity": {"alpha": 0.7, "window_months": 6},
            "topic_similarity": {
                "provider": {"type": "fake_for_test", "dim": 16},
            },
        },
    }
    r = client.put(
        f"/api/enrichment/config?path={tmp_path}",
        json={"enrichment_block": block},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["operator_block"] == block
    assert body["resolved_block"]["enrichers"]["temporal_velocity"]["alpha"] == 0.7
    # On-disk file mirrors what we sent.
    on_disk = yaml.safe_load((tmp_path / "viewer_operator.yaml").read_text())
    assert on_disk["enrichment"] == block


def test_put_config_validates_block_and_rejects_garbage(client: TestClient, tmp_path: Path) -> None:
    # `enabled` must be boolean per the schema; "yes please" must fail.
    bad = {"enabled": "yes please"}
    r = client.put(
        f"/api/enrichment/config?path={tmp_path}",
        json={"enrichment_block": bad},
    )
    assert r.status_code == 400, r.text
    assert "invalid enrichment block" in r.json()["detail"]


def test_put_config_rejects_unknown_provider_type(client: TestClient, tmp_path: Path) -> None:
    """Composed-schema validation: ``provider.type`` must match a
    registered provider type. Unknown names get rejected at PUT time,
    not silently at runtime."""
    block = {
        "enrichers": {
            "topic_similarity": {"provider": {"type": "not_a_real_provider"}},
        },
    }
    r = client.put(
        f"/api/enrichment/config?path={tmp_path}",
        json={"enrichment_block": block},
    )
    assert r.status_code == 400, r.text
    assert "not valid" in r.json()["detail"] or "is not valid" in r.json()["detail"]


def test_put_config_rejects_missing_required_provider_param(
    client: TestClient, tmp_path: Path
) -> None:
    """``sentence_transformer_local`` requires ``model``. Omitting it must
    fail validation at PUT time."""
    block = {
        "enrichers": {
            "topic_similarity": {
                "provider": {"type": "sentence_transformer_local"},  # missing model
            },
        },
    }
    r = client.put(
        f"/api/enrichment/config?path={tmp_path}",
        json={"enrichment_block": block},
    )
    assert r.status_code == 400, r.text


def test_put_config_rejects_typoed_knob(client: TestClient, tmp_path: Path) -> None:
    """Typo'd knob names (``alphaa`` instead of ``alpha``) must fail
    validation rather than silently being ignored at runtime."""
    block = {
        "enrichers": {
            "temporal_velocity": {"alphaa": 0.7},
        },
    }
    r = client.put(
        f"/api/enrichment/config?path={tmp_path}",
        json={"enrichment_block": block},
    )
    assert r.status_code == 400, r.text


def test_put_config_rejects_provider_on_deterministic_enricher(
    client: TestClient, tmp_path: Path
) -> None:
    """``temporal_velocity`` has no ``provider_requirement`` on its
    manifest. Adding a ``provider:`` block to it must fail validation —
    the operator is confused about what kind of enricher this is."""
    block = {
        "enrichers": {
            "temporal_velocity": {"provider": {"type": "fake_for_test"}},
        },
    }
    r = client.put(
        f"/api/enrichment/config?path={tmp_path}",
        json={"enrichment_block": block},
    )
    assert r.status_code == 400, r.text


def test_put_config_preserves_unrelated_yaml_keys(client: TestClient, tmp_path: Path) -> None:
    # Seed viewer_operator.yaml with an unrelated top-level key.
    (tmp_path / "viewer_operator.yaml").write_text(
        "profile: cloud_thin\nsome_other_key: hello\n",
        encoding="utf-8",
    )
    r = client.put(
        f"/api/enrichment/config?path={tmp_path}",
        json={"enrichment_block": {"enabled": True}},
    )
    assert r.status_code == 200, r.text
    on_disk = yaml.safe_load((tmp_path / "viewer_operator.yaml").read_text())
    assert on_disk["profile"] == "cloud_thin"
    assert on_disk["some_other_key"] == "hello"
    assert on_disk["enrichment"] == {"enabled": True}
