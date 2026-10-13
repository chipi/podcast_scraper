"""#2345: the one-time re-key — dry run, apply with receipts, undo."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SPEC = importlib.util.spec_from_file_location(
    "rekey_user_references",
    Path(__file__).resolve().parents[4] / "scripts" / "ops" / "rekey_user_references.py",
)
assert _SPEC and _SPEC.loader
_mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _mod
_SPEC.loader.exec_module(_mod)


def _artifact(path: Path, clusters: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"data": {"clusters": clusters}}), encoding="utf-8")
    return path


def _c(cid: str, *topics: str) -> dict:
    return {"graph_compound_parent_id": cid, "members": [{"topic_id": t} for t in topics]}


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    from podcast_scraper.server import app_user_state, reference_integrity

    corpus, appdata = tmp_path / "corpus", tmp_path / "app"
    _artifact(
        corpus / "enrichments" / "topic_theme_clusters.json",
        [_c("thc:managing-risk", "a", "b", "c", "d")],
    )
    prior = _artifact(
        tmp_path / "snap" / "topic_theme_clusters.json",
        [_c("thc:risk", "a", "b", "c"), _c("thc:pair", "x", "y")],
    )
    app_user_state.add_interest(appdata, "u1", "thc:risk")
    app_user_state.add_interest(appdata, "u1", "thc:pair")

    # No KG, catalog or entity map in this corpus: the stub judges storylines only, with the
    # successors the command derived from the saved artifact.
    def resolver(root: Path, extra=None) -> reference_integrity.Resolver:
        return reference_integrity.Resolver(
            storylines={"thc:managing-risk"},
            successors=dict((extra or {}).get("successors") or {}),
        )

    monkeypatch.setattr(reference_integrity, "build_resolver", resolver)
    return corpus, appdata, prior


def test_dry_run_apply_and_undo(world, capsys: pytest.CaptureFixture[str]) -> None:
    from podcast_scraper.server import app_user_state, reference_integrity

    corpus, appdata, prior = world
    args = ["--appdata", str(appdata), "--corpus", str(corpus), "--prior-storylines", str(prior)]

    assert _mod.main(args) == 0
    out = capsys.readouterr().out
    assert "prior renames known: 1" in out and "would re-key 1" in out
    assert app_user_state.get_interests(appdata, "u1") == ["thc:risk", "thc:pair"]

    assert _mod.main([*args, "--apply"]) == 0
    assert app_user_state.get_interests(appdata, "u1") == ["thc:managing-risk", "thc:pair"]
    assert reference_integrity.gone_references(appdata, "u1") == [
        {"ref": "thc:pair", "kind": "storyline"}
    ]
    assert (appdata / _mod.RECEIPTS_FILE).read_text("utf-8").count("\n") == 1

    assert _mod.main(["--appdata", str(appdata), "--undo"]) == 0
    assert app_user_state.get_interests(appdata, "u1") == ["thc:risk", "thc:pair"]
