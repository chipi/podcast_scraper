"""The KG audit judges the copy the app SERVES, never "the newest file" (#2199).

The 2026-09-29 repair counted by newest kg.json mtime; a migration had given every kg.json the same
mtime, so for two episodes the count landed on a good OLDER run while the app served a fabricated
``topic_labels`` graph from the newer run. The shape is reproduced here, mtimes tied and all.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "audit" / "served_kg_provenance.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("served_kg_provenance", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _episode(corpus: Path, run: str, episode_id: str, provenance: str | None) -> None:
    meta_dir = corpus / "feeds" / "rss_feeds.npr.org_7ce5b183" / run / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    name = f"0001 - {episode_id}"
    (meta_dir / f"{name}.metadata.json").write_text(
        json.dumps({"episode": {"episode_id": episode_id, "title": name}}), encoding="utf-8"
    )
    if provenance is not None:
        kg = meta_dir / f"{name}.kg.json"
        kg.write_text(json.dumps({"extraction": {"model_version": provenance}}), encoding="utf-8")
        os.utime(kg, (1790000000, 1790000000))  # every kg.json rewritten in the same minute


def test_the_served_newer_run_is_judged_not_the_good_older_copy(tmp_path: Path) -> None:
    """THE #2199 shape: Planet Money 85e1ca1e — good Aug-11 run, fabricated Aug-19 run served."""
    _episode(tmp_path, "run_20260811-205400", "85e1ca1e", "provider:podcast-flash-0731")
    _episode(tmp_path, "run_20260819-043606", "85e1ca1e", "topic_labels")
    result = _module().scan(tmp_path)
    assert result["served"] == 1
    assert [(b["episode_id"], b["provenance"]) for b in result["bad"]] == [
        ("85e1ca1e", "topic_labels")
    ]


def test_a_clean_corpus_exits_zero_and_a_bad_one_exits_one(tmp_path: Path) -> None:
    mod = _module()
    _episode(tmp_path, "run_20260811-205400", "good", "provider:NVFP4/Qwen3")
    assert mod.main(["--corpus-dir", str(tmp_path)]) == 0
    _episode(tmp_path, "run_20260811-205400", "empty", "provider:extraction_failed")
    assert mod.main(["--corpus-dir", str(tmp_path)]) == 1


def test_a_served_run_without_a_kg_is_bad(tmp_path: Path) -> None:
    _episode(tmp_path, "run_20260811-205400", "nokg", None)
    assert [b["provenance"] for b in _module().scan(tmp_path)["bad"]] == ["(missing kg.json)"]


def test_the_worklist_holds_one_id_per_line(tmp_path: Path) -> None:
    _episode(tmp_path, "run_20260811-205400", "a", "topic_labels")
    _episode(tmp_path, "run_20260811-205400", "b", "no_extractor")
    out = tmp_path / "ids.txt"
    _module().main(["--corpus-dir", str(tmp_path), "--worklist", str(out)])
    assert out.read_text(encoding="utf-8") == "a\nb\n"
