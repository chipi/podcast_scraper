"""scripts/measure/m0043_census.py and m0043_list.py: which census voices 0043 lists, and as what.

Order (record section 3): class A, class C, a stated host (kept unless an operator-named ad id:
class E), an excluded id (kept), else class B. Every entry names its class and first segment.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

pytestmark = pytest.mark.unit

_DIR = Path(__file__).resolve().parents[4] / "scripts" / "measure"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _DIR / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


census = _load("m0043_census")
lst = _load("m0043_list")


def _episode(
    root: Path, eid: str, label: str, *, hosts: Optional[List[str]] = None, role: str = "guest"
) -> Dict[str, Any]:
    run = root / "feeds" / eid / "run_1"
    (run / "metadata").mkdir(parents=True)
    (run / "transcripts").mkdir()
    meta_rel = f"feeds/{eid}/run_1/metadata/ep.metadata.json"
    content = {
        "transcript_file_path": "transcripts/ep.txt",
        "speakers": [{"name": label, "role": role, "voices": ["SPEAKER_01"]}],
    }
    (root / meta_rel).write_text(
        json.dumps({"episode": {"episode_id": eid}, "content": content}), encoding="utf-8"
    )
    rows = [
        {"speaker": "SPEAKER_00", "speaker_label": "Host Person", "speaker_role": "host"},
        {"speaker": "SPEAKER_01", "speaker_label": label, "speaker_role": role},
    ]
    (run / "transcripts" / "ep.segments.json").write_text(json.dumps(rows), encoding="utf-8")
    (run / "transcripts" / "ep.speakers.diagnostics.json").write_text(
        json.dumps({"tried": {"known_hosts": hosts or []}}), encoding="utf-8"
    )
    return {"meta": meta_rel, "episode_id": eid, "voice": "SPEAKER_01", "name": label}


def test_stored_names_skip_raw_labels(tmp_path: Path) -> None:
    c = _episode(tmp_path, "e1", "Ann Reader")
    meta = tmp_path / c["meta"]
    doc = json.loads(meta.read_text())
    assert census.stored_names(meta, doc["content"]) == {
        "SPEAKER_00": "Host Person",
        "SPEAKER_01": "Ann Reader",
    }


def test_only_unnamed_commercial_voices_are_census_rows() -> None:
    by_voice = {
        "S0": SimpleNamespace(named=False, voice_type="commercial"),
        "S1": SimpleNamespace(named=True, voice_type="commercial"),
        "S2": SimpleNamespace(named=False, voice_type="unknown"),
    }
    stored = {"S0": "A", "S1": "B", "S2": "C", "S3": "D"}
    assert census.commercial_unnamed(by_voice, stored) == ["S0"]


def _build(tmp_path: Path, rows: List[Dict[str, Any]], monkeypatch, a=(), c=()) -> Any:
    hits_a = [(r["meta"], r["voice"], r["name"]) for r in rows if r["episode_id"] in a]
    hits_c = [(r["meta"], r["voice"], r["name"]) for r in rows if r["episode_id"] in c]
    monkeypatch.setattr(lst.m16, "_load_signatures", lambda root: ("sigs", {}))
    monkeypatch.setattr(lst.m16, "find_hits", lambda root, sigs: (hits_c, [], hits_a))
    fake = SimpleNamespace(classify=lambda root, hit: {"verdict": "rule_a"})
    monkeypatch.setattr(lst, "_candidates", lambda: fake)
    return lst.build(tmp_path, rows)


def test_the_classes_in_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        _episode(tmp_path, "a1", "Ann Reader"),
        _episode(tmp_path, "c1", "Cy Promo"),
        _episode(tmp_path, "b1", "Bo Shoutout"),
        _episode(tmp_path, "h1", "Hal Host", hosts=["Hal Host"]),
        _episode(tmp_path, "h2", "Hil Seat", role="host"),
    ]
    entries, skipped = _build(tmp_path, rows, monkeypatch, a={"a1"}, c={"c1"})
    cls = {k[0]: e[4].split(":")[0] for k, e in entries.items()}
    assert cls == {"a1": "class A", "c1": "class C", "b1": "class B"}
    assert set(skipped) == {("h1", "SPEAKER_01"), ("h2", "SPEAKER_01")}
    assert all(e[2] is None and e[3] is None for e in entries.values())
    assert entries[("b1", "SPEAKER_01")][:2] == ("Bo Shoutout", "guest")
    assert all(e[4].endswith("segment 1") for e in entries.values())


def test_a_host_name_on_an_ad_is_listed_only_when_the_operator_named_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [_episode(tmp_path, "h1", "Hal Host", hosts=["Hal Host"])]
    monkeypatch.setattr(lst, "HOST_NAME_ON_AN_AD", {("h1", "SPEAKER_01"): "a German-language ad"})
    entries, skipped = _build(tmp_path, rows, monkeypatch)
    assert entries[("h1", "SPEAKER_01")][4].startswith("class E: the host's name on a German")
    assert skipped == {}


def test_an_excluded_id_is_kept_with_its_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [_episode(tmp_path, "x1", "Xu Own")]
    monkeypatch.setattr(lst, "EXCLUDED", {("x1", "SPEAKER_01"): "the show's own host"})
    entries, skipped = _build(tmp_path, rows, monkeypatch)
    assert entries == {} and skipped == {("x1", "SPEAKER_01"): "the show's own host"}


def test_every_listed_entry_passes_the_migrations_own_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [_episode(tmp_path, "a1", "Ann Reader"), _episode(tmp_path, "b1", "Bo Shoutout")]
    entries, _ = _build(tmp_path, rows, monkeypatch, a={"a1"})
    path = tmp_path / "list.json"
    lst.m34.write_list(path, entries, lst.m43.MIGRATION_ID)
    assert lst.m43.load_list(path) == entries
