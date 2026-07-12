"""Guard rail: the prod-state fixture stays consistent with the prod marker (#1176, CI net C).

The pinned corpus under ``tests/fixtures/upgrade/corpus_at_last_prod_release/``
represents the on-disk shape of the production corpus AFTER the most recent
successful prod deploy. Its identity is tied to
``config/last_deployed_prod_version.json`` — that file names the code version
and the exact set of migrations that ran on the deployed corpus.

These unit tests keep the two artefacts in lock-step. They do NOT test the
framework's transform behaviour — that's the integration test's job. They test
the fixture is a faithful pin so the integration test is actually running
against prod-shape rather than something someone edited mid-session.

Fast (<10 ms). No LanceDB, no subprocess.

Maintenance rule: after every prod deploy, update BOTH
``config/last_deployed_prod_version.json`` AND this fixture (see the README in
the fixture directory).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "upgrade" / "corpus_at_last_prod_release"
PROD_MARKER = REPO_ROOT / "config" / "last_deployed_prod_version.json"


def _load(path: Path) -> dict:
    doc: dict = json.loads(path.read_text(encoding="utf-8"))
    return doc


# ---------- marker + fixture identity match ----------


def test_prod_marker_declares_code_version_sha_and_deploy_date() -> None:
    """The marker file must carry enough to identify the exact deployed build."""
    marker = _load(PROD_MARKER)
    for key in ("code_version", "sha", "deployed_at", "applied_migrations"):
        assert key in marker, f"prod marker missing required key: {key}"
    assert isinstance(marker["applied_migrations"], list)


def test_fixture_manifest_matches_prod_marker_code_version() -> None:
    """The fixture's produced_by.code_version MUST equal the prod marker."""
    marker = _load(PROD_MARKER)
    manifest = _load(FIXTURE / "corpus_manifest.json")
    fixture_ver = manifest["produced_by"]["code_version"]
    marker_ver = marker["code_version"]
    assert fixture_ver == marker_ver, (
        f"fixture manifest code_version={fixture_ver!r} != "
        f"prod marker code_version={marker_ver!r}. "
        "The fixture and marker must be updated together (see fixture README)."
    )


def test_fixture_ledger_matches_prod_marker_applied_migrations() -> None:
    """The fixture's upgrade_ledger.json 'applied' list MUST enumerate exactly
    the ids in the prod marker. Same set, no extras, no misses.
    """
    marker = _load(PROD_MARKER)
    ledger = _load(FIXTURE / "upgrade_ledger.json")

    fixture_ids = {entry["id"] for entry in ledger.get("applied", [])}
    marker_ids = set(marker["applied_migrations"])

    missing = marker_ids - fixture_ids
    extra = fixture_ids - marker_ids
    assert not missing, (
        f"fixture ledger missing prod-applied ids: {sorted(missing)}. "
        "Add them to the fixture's upgrade_ledger.json."
    )
    assert not extra, (
        f"fixture ledger has ids not in the prod marker: {sorted(extra)}. "
        "Either the marker is stale (bump it) or the fixture is (drop them)."
    )


# ---------- fixture is at post-migration prod shape ----------


def test_fixture_gi_is_at_current_prod_schema() -> None:
    """The .gi.json in the fixture must reflect ALL migrations the prod marker
    reports as applied. Concretely: if m0003 is in the marker, the fixture must
    be at schema 3.0 with typed edges and no legacy vocab.
    """
    marker = _load(PROD_MARKER)
    applied = set(marker["applied_migrations"])

    gi = _load(FIXTURE / "metadata" / "ep1.gi.json")

    if "0003_gi_v3_typed_mentions" in applied:
        assert gi.get("schema_version") == "3.0", (
            f".gi.json schema_version={gi.get('schema_version')!r}; "
            "m0003 is in the prod marker → fixture must be at 3.0."
        )
        edge_types = {e.get("type") for e in gi.get("edges") or []}
        assert "MENTIONS" not in edge_types, (
            f"legacy untyped MENTIONS still in the fixture .gi.json: {edge_types}. "
            "m0003 is in the prod marker → these should be MENTIONS_PERSON / "
            "MENTIONS_ORG."
        )

        insight_types = [
            (n.get("properties") or {}).get("insight_type")
            for n in gi.get("nodes") or []
            if n.get("type") == "Insight"
        ]
        legacy_vocab = {"fact", "opinion"}
        assert not any(t in legacy_vocab for t in insight_types), (
            f"fixture insight_type still uses legacy vocab {legacy_vocab}: "
            f"{insight_types}. m0003 in the prod marker → vocab should be "
            "claim/observation."
        )


def test_fixture_has_feeds_spec_yaml() -> None:
    """Sanity check — required by any pack step and by corpus-root validation."""
    assert (FIXTURE / "feeds.spec.yaml").is_file()


def test_fixture_ledger_version_matches_last_applied_to_version() -> None:
    """The ledger's ``version`` field must equal the ``to_version`` of the last
    entry — that's the post-deploy corpus code version.
    """
    ledger = _load(FIXTURE / "upgrade_ledger.json")
    applied = ledger.get("applied", [])
    if not applied:
        pytest.skip("no applied entries — nothing to check")
    last_to = applied[-1]["to_version"]
    assert (
        ledger.get("version") == last_to
    ), f"ledger.version={ledger.get('version')!r} != last applied.to_version={last_to!r}"
