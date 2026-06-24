"""#1076 / ADR-102 — coverage for the lossy reverse migration script.

``scripts/dev/revert_gi_v3_to_v2.py`` is the rollback path used when a v3
sweep contaminates a corpus that was meant to stay v2. It is lossy by
design (per script docstring) — these tests pin the lossy contract so a
future refactor can't silently expand or shrink what it reverts.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPT = _REPO_ROOT / "scripts" / "dev" / "revert_gi_v3_to_v2.py"


def _v3_artifact() -> dict:
    return {
        "schema_version": "3.0",
        "model_version": "stub",
        "prompt_version": "v1",
        "episode_id": "ep1",
        "nodes": [
            {
                "id": "insight:1",
                "type": "Insight",
                "properties": {
                    "text": "Alice told the crowd.",
                    "episode_id": "ep1",
                    "grounded": True,
                    "insight_type": "claim",
                    "position_hint": 0.42,
                },
            }
        ],
        "edges": [
            {"type": "MENTIONS_PERSON", "from": "insight:1", "to": "person:alice"},
            {"type": "MENTIONS_ORG", "from": "insight:1", "to": "org:acme"},
            {"type": "ABOUT", "from": "insight:1", "to": "topic:t1"},
        ],
        "_retro_audit": [
            {
                "marker": "#1076-ner-2026-06-24",
                "applied_at": "2026-06-24T11:00:00+00:00",
                "use_ner": True,
                "edges_added": {"mentions": 1, "has_episode": 0, "spoken_by": 0},
            }
        ],
    }


def _run(corpus: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--corpus", str(corpus), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _write_artifact(corpus: Path, name: str, data: dict) -> Path:
    p = corpus / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


class TestRevertGIv3ToV2:
    def test_reverts_schema_version_and_strips_audit(self, tmp_path: Path) -> None:
        gi_path = _write_artifact(tmp_path, "ep1.gi.json", _v3_artifact())
        # Also drop a sweep summary at the corpus root — should be removed.
        (tmp_path / "_retro_audit_1076-ner-2026-06-24.json").write_text("{}")

        result = _run(tmp_path)
        assert result.returncode == 0
        reverted = json.loads(gi_path.read_text())
        assert reverted["schema_version"] == "2.0"
        assert "_retro_audit" not in reverted
        assert not (tmp_path / "_retro_audit_1076-ner-2026-06-24.json").exists()

    def test_reverts_typed_mentions_to_untyped(self, tmp_path: Path) -> None:
        gi_path = _write_artifact(tmp_path, "ep1.gi.json", _v3_artifact())
        _run(tmp_path)
        reverted = json.loads(gi_path.read_text())
        edge_types = {e["type"] for e in reverted["edges"]}
        # Typed MENTIONS_PERSON / MENTIONS_ORG fold back to untyped MENTIONS.
        assert "MENTIONS_PERSON" not in edge_types
        assert "MENTIONS_ORG" not in edge_types
        # Both edges are now MENTIONS; count preserved (lossy: type info gone).
        assert sum(1 for e in reverted["edges"] if e["type"] == "MENTIONS") == 2
        # Non-MENTIONS edges (ABOUT) preserved unchanged.
        assert "ABOUT" in edge_types

    def test_documented_lossy_fields_stay_normalized(self, tmp_path: Path) -> None:
        """``insight_type`` and ``position_hint`` are documented lossy
        casualties — the script does NOT strip them, accepting v2 will
        carry forward-migration artifacts. Locks the lossy contract."""
        gi_path = _write_artifact(tmp_path, "ep1.gi.json", _v3_artifact())
        _run(tmp_path)
        reverted = json.loads(gi_path.read_text())
        props = reverted["nodes"][0]["properties"]
        # v3 normalization stays — v2 validator is permissive about extras.
        assert props["insight_type"] == "claim"
        assert props["position_hint"] == 0.42

    def test_idempotent_on_already_v2_artifact(self, tmp_path: Path) -> None:
        """Running revert on a clean v2 artifact is a no-op (zero file
        modifications). Lets callers re-run the script safely after a
        partial sweep without fear of further contamination."""
        v2 = _v3_artifact()
        v2["schema_version"] = "2.0"
        v2["edges"] = [{"type": "MENTIONS", "from": "insight:1", "to": "person:alice"}]
        v2.pop("_retro_audit")
        gi_path = _write_artifact(tmp_path, "ep1.gi.json", v2)
        before = gi_path.read_text()
        result = _run(tmp_path)
        assert result.returncode == 0
        after = gi_path.read_text()
        # File content unchanged — idempotent.
        assert before == after

    def test_dry_run_does_not_write(self, tmp_path: Path) -> None:
        gi_path = _write_artifact(tmp_path, "ep1.gi.json", _v3_artifact())
        before = gi_path.read_text()
        result = _run(tmp_path, "--dry-run")
        assert result.returncode == 0
        after = gi_path.read_text()
        assert before == after  # dry-run is truly read-only
        assert "would revert" in result.stdout

    def test_missing_corpus_returns_error_code(self, tmp_path: Path) -> None:
        result = _run(tmp_path / "does-not-exist")
        assert result.returncode == 2
