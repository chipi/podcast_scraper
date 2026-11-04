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


def test_put_config_accepts_and_persists_an_operational_note(
    client: TestClient, tmp_path: Path
) -> None:
    """Operational rationale rides as DATA so it survives a UI save.

    The whole file is rewritten with ``yaml.safe_dump`` on every PUT, which cannot emit
    comments — a ``#`` explaining an override is erased by the first save from the
    enricher tab, and is invisible in that UI anyway (#2086). A ``note`` field survives
    every edit path and can be rendered next to the value it explains.
    """
    why = (
        "Raised 1800->3600 on 2026-09-16: a cold 200-entity chunk needs ~36min at the "
        "measured ~5.5 entities/min, and a timeout banks nothing."
    )
    block = {
        "enrichers": {
            "temporal_velocity": {"alpha": 0.7, "note": why},
        },
    }
    r = client.put(
        f"/api/enrichment/config?path={tmp_path}",
        json={"enrichment_block": block},
    )
    assert r.status_code == 200, r.text
    on_disk = yaml.safe_load((tmp_path / "viewer_operator.yaml").read_text())
    assert on_disk["enrichment"]["enrichers"]["temporal_velocity"]["note"] == why
    # And it comes back out of the resolved view, so the UI can render it.
    assert r.json()["resolved_block"]["enrichers"]["temporal_velocity"]["note"] == why


def test_a_note_survives_a_second_save_that_does_not_mention_it(
    client: TestClient, tmp_path: Path
) -> None:
    """The failure mode this replaces: a comment lost on the NEXT save, not the first.

    Re-PUTting a block that still carries the note must keep it. (A block that omits it
    is a deliberate deletion — the enrichers list is a complete declaration.)
    """
    why = "Overridden because the upstream is rate-limited."
    first = {"enrichers": {"temporal_velocity": {"note": why}}}
    assert (
        client.put(
            f"/api/enrichment/config?path={tmp_path}", json={"enrichment_block": first}
        ).status_code
        == 200
    )
    second = {"enrichers": {"temporal_velocity": {"alpha": 0.5, "note": why}}}
    r = client.put(f"/api/enrichment/config?path={tmp_path}", json={"enrichment_block": second})
    assert r.status_code == 200, r.text
    on_disk = yaml.safe_load((tmp_path / "viewer_operator.yaml").read_text())
    assert on_disk["enrichment"]["enrichers"]["temporal_velocity"]["note"] == why
    assert on_disk["enrichment"]["enrichers"]["temporal_velocity"]["alpha"] == 0.5


def test_note_is_declared_on_every_enricher_in_the_composed_schema(
    client: TestClient,
) -> None:
    """``note`` must be available on every enricher, not just the one under test —
    it is injected once in ``_per_enricher_schema``, so a regression there would
    silently 400 notes on some enrichers and not others."""
    blocks = client.get("/api/enrichment/config/schema").json()["properties"]["enrichers"][
        "properties"
    ]
    assert blocks, "composed schema exposed no enricher blocks"
    missing = [eid for eid, b in blocks.items() if "note" not in b.get("properties", {})]
    assert not missing, f"enrichers missing the `note` knob: {missing}"


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
