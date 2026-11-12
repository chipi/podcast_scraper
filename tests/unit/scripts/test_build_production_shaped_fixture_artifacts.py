"""The production-shaped fixture's artifacts: KG in the fixture, GI only into a separate directory.

The viewer's Tier-2 fixture holds no GI artifacts (GI is a downstream build's, E04 step 8b);
``--gi-output`` writes them where that build keeps its fixture.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ARTIFACTS = (
    REPO_ROOT / "web" / "gi-kg-viewer" / "e2e" / "fixtures" / "production-shaped" / "artifacts"
)


def _load_build_module() -> Any:
    script = REPO_ROOT / "scripts" / "build_production_shaped_fixture.py"
    spec = importlib.util.spec_from_file_location("build_prod_shaped_artifacts", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _corpus(tmp_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    corpus = tmp_path / "corpus"
    meta = corpus / "feeds" / "f" / "metadata"
    meta.mkdir(parents=True)
    (meta / "e1.gi.json").write_text('{"kind": "gi"}')
    (meta / "e1.kg.json").write_text('{"kind": "kg"}')
    (meta / "e2.kg.json").write_text('{"kind": "kg2"}')
    episodes = [
        {
            "episode_id": "ep-1",
            "gi_relative_path": "feeds/f/metadata/e1.gi.json",
            "kg_relative_path": "feeds/f/metadata/e1.kg.json",
        },
        {
            "episode_id": "ep-2",
            "gi_relative_path": "feeds/f/metadata/missing.gi.json",
            "kg_relative_path": "feeds/f/metadata/e2.kg.json",
        },
    ]
    return corpus, episodes


def test_without_gi_output_only_kg_is_copied(tmp_path: Path) -> None:
    mod = _load_build_module()
    corpus, episodes = _corpus(tmp_path)
    out = tmp_path / "fixture"
    assert mod.copy_episode_artifacts(episodes, corpus, out, None) == 2
    assert sorted(p.name for p in (out / "artifacts").iterdir()) == ["ep-1.kg.json", "ep-2.kg.json"]


def test_gi_output_receives_gi_and_the_fixture_still_none(tmp_path: Path) -> None:
    mod = _load_build_module()
    corpus, episodes = _corpus(tmp_path)
    out = tmp_path / "fixture"
    gi_out = tmp_path / "downstream" / "artifacts"
    assert mod.copy_episode_artifacts(episodes, corpus, out, gi_out) == 3
    assert [p.name for p in gi_out.iterdir()] == ["ep-1.gi.json"]
    assert (gi_out / "ep-1.gi.json").read_text() == '{"kind": "gi"}'
    assert not list((out / "artifacts").glob("*.gi.json"))


def test_checked_in_fixture_has_no_gi_artifacts() -> None:
    assert FIXTURE_ARTIFACTS.is_dir()
    assert sorted(p.name for p in FIXTURE_ARTIFACTS.glob("*.gi.json")) == []
    assert len(list(FIXTURE_ARTIFACTS.glob("*.kg.json"))) == 9
