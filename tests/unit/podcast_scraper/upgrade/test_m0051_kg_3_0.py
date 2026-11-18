"""m0051: every served KG file rewritten from KG 2.x to KG 3.0 (R3b's KG half, record
r3b_kg_3_0.md). Paths: a served file rewritten and valid; a superseded run untouched; a KG 3.0 file
left as it is; a file that would not validate refused; PUBLISHED_BY from the registry; and the
standing checks (dry run, verify, second apply, undo byte-identical, an older migration's undo
refused on a rewritten file, registration).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from podcast_scraper.publishers import Publisher, Registry
from podcast_scraper.upgrade.file_rewrite import (
    append_receipts,
    undo_from_receipts,
    write_with_backup,
)
from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations import m0051_kg_3_0 as m51
from podcast_scraper.upgrade.registry import get_migrations

pytestmark = [pytest.mark.unit]


def _w(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _kg21(eid: str, *, title: str = "Ep") -> Dict[str, Any]:
    return {
        "schema_version": "2.1",
        "episode_id": eid,
        "extraction": {
            "model_version": "provider:openai:gpt-x",
            "extracted_at": "2026-11-01T00:00:00Z",
            "transcript_ref": f"../run_x/transcripts/{eid}.adfree.txt",
        },
        "nodes": [
            {
                "id": f"episode:{eid}",
                "type": "Episode",
                "properties": {"title": title, "podcast_id": "f1", "feed_id": "f1"},
            },
            {"id": "podcast:f1", "type": "Podcast", "properties": {"title": "Show"}},
            {
                "id": "person:ann",
                "type": "Person",
                "properties": {"name": "Ann", "label": "Ann", "role": "guest"},
            },
        ],
        "edges": [
            {"type": "HAS_EPISODE", "from": "podcast:f1", "to": f"episode:{eid}"},
            {"type": "GUESTS_ON", "from": "person:ann", "to": "podcast:f1"},
        ],
    }


def _episode(root: Path, feed: str, run: str, eid: str, **kw: Any) -> Path:
    base = root / "feeds" / feed / run / "metadata"
    _w(base / "ep.metadata.json", {"episode": {"episode_id": eid, "language": "en"}})
    kg = base / "ep.kg.json"
    _w(kg, _kg21(eid, **kw))
    return kg


def _shas(root: Path) -> Dict[str, str]:
    return {
        str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(root.rglob("*.json"))
        if ".podcast_scraper" not in f.parts
    }


def _ctx(root: Path, dry_run: bool = False) -> MigrationContext:
    return MigrationContext(corpus_root=root, dry_run=dry_run)


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(m51, "load_registry", lambda: Registry())
    _episode(tmp_path, "a", "run_20260201-000000", "ep-a")
    _episode(tmp_path, "b", "run_20260201-000000", "ep-b")
    return tmp_path


def _load(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def test_each_served_file_becomes_kg_3_0(corpus: Path) -> None:
    result = m51.KgThreeMigration().apply(_ctx(corpus))
    kg = _load(corpus / "feeds/a/run_20260201-000000/metadata/ep.kg.json")
    assert kg["schema_version"] == "3.0" and "episode_id" not in kg
    assert {(e["type"], e.get("properties", {}).get("role")) for e in kg["edges"]} == {
        ("HAS_ITEM", None),
        ("CONTRIBUTES_TO", "guest"),
    }
    assert "files_written=2" in result.message and "published_by=0" in result.message


def test_published_by_from_the_registry(corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pub = Publisher(id="org:show-co", name="Show Co", sources=("podcast:f1",))
    monkeypatch.setattr(m51, "load_registry", lambda: Registry((pub,)))
    result = m51.KgThreeMigration().apply(_ctx(corpus))
    assert "published_by=2" in result.message
    kg = _load(corpus / "feeds/a/run_20260201-000000/metadata/ep.kg.json")
    assert {"type": "PUBLISHED_BY", "from": "podcast:f1", "to": "org:show-co"} in kg["edges"]


def test_a_file_that_would_not_validate_is_refused(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m51, "load_registry", lambda: Registry())
    kg = _episode(tmp_path, "a", "run_20260201-000000", "ep-a", title="")
    before = kg.read_bytes()
    result = m51.KgThreeMigration().apply(_ctx(tmp_path))
    assert kg.read_bytes() == before and "refused_invalid=1" in result.message


def test_a_superseded_run_and_a_3_0_file_are_left_as_they_are(corpus: Path) -> None:
    old = _episode(corpus, "a", "run_20260101-000000", "ep-a")
    m51.KgThreeMigration().apply(_ctx(corpus))
    assert _load(old)["schema_version"] == "2.1", "not served: untouched"
    after = _shas(corpus)
    second = m51.KgThreeMigration().apply(_ctx(corpus))
    assert "0 file(s)" in second.message and _shas(corpus) == after


def test_dry_run_verify_second_apply_and_undo(corpus: Path) -> None:
    mig = m51.KgThreeMigration()
    before = _shas(corpus)
    mig.apply(_ctx(corpus, dry_run=True))
    assert _shas(corpus) == before
    assert len(list(m51.dry_run_report(corpus))) == 2
    ok, _ = mig.verify(_ctx(corpus))
    assert not ok
    mig.apply(_ctx(corpus))
    ok, msg = mig.verify(_ctx(corpus))
    assert ok, msg
    assert m51.undo(corpus) == (2, [])
    assert _shas(corpus) == before


def test_an_older_migrations_undo_is_refused_on_a_file_this_one_rewrote(corpus: Path) -> None:
    kg = corpus / "feeds/a/run_20260201-000000/metadata/ep.kg.json"
    older = _load(kg)
    older["nodes"][2]["properties"]["name"] = "Ann B"
    receipt = write_with_backup(corpus, "0050x", kg, older)
    append_receipts(corpus, "older.jsonl", {"migration": "older"}, [receipt])
    m51.KgThreeMigration().apply(_ctx(corpus))
    restored, refused = undo_from_receipts(corpus, "older.jsonl", "0050x", "older")
    assert restored == 0 and refused


def test_registered() -> None:
    assert "0051_kg_3_0" in [m.id for m in get_migrations()]
