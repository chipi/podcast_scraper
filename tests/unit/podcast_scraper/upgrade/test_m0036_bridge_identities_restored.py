"""m0036: every published GI / KG person gets the bridge identity the builder makes, nothing else.

Built on m0025's corpus fixture (one episode with GI and KG); each test writes its own bridge.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pytest

from podcast_scraper.builders.bridge_builder import _collect_identity, build_bridge
from podcast_scraper.speaker_detectors.hosts import is_publishable_speaker_name
from podcast_scraper.upgrade.file_rewrite import backup_dir
from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations import m0036_bridge_identities_restored as m36
from podcast_scraper.upgrade.registry import get_migrations

pytestmark = [pytest.mark.unit]

_spec = importlib.util.spec_from_file_location(
    "m0025_fixture", Path(__file__).with_name("test_m0025_one_person_one_entry.py")
)
assert _spec and _spec.loader
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)


def _episode(tmp_path: Path) -> Dict[str, Any]:
    turns = []
    for i in range(6):
        turns.append(("SPEAKER_00", "Gerald Orban", "host", f"Question {i}."))
        turns.append(("SPEAKER_01", "Stan McCormack", "guest", f"Answer {i}."))
    p: Dict[str, Any] = fx._corpus(
        tmp_path,
        speakers=[
            fx._speaker("Gerald Orban", "host", ["SPEAKER_00"], "known_hosts", "host"),
            fx._speaker("Stan McCormack", "guest", ["SPEAKER_01"], "roster", "guest"),
        ],
        turns=turns,
        known_hosts=["Gerald Orban"],
    )
    p["bridge"] = p["meta"].with_name(p["meta"].name.replace(".metadata.json", ".bridge.json"))
    return p


def _persons(path: Path) -> Dict[str, dict]:
    return {n["id"]: n for n in fx._r(path)["nodes"] if n["type"] == "Person"}


def _bridge(p: Dict[str, Any], identities: list) -> None:
    p["bridge"].write_text(
        json.dumps({"schema_version": "1.0", "episode_id": "ep", "identities": identities}),
        encoding="utf-8",
    )


def _ctx(root: Path) -> MigrationContext:
    return MigrationContext(corpus_root=root)


def _shas(root: Path) -> Dict[str, str]:
    return {
        str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in root.rglob("*")
        if f.is_file()
        and ".podcast_scraper" not in f.parts
        and f.name != m36.RECEIPTS_FILE
        and not f.name.startswith("upgrade_ledger")
    }


def test_a_person_with_no_bridge_identity_gets_the_builders_identity_and_nothing_else_changes(
    tmp_path: Path,
) -> None:
    p = _episode(tmp_path)
    gi, kg = _persons(p["gi"]), _persons(p["kg"])
    stan = next(pid for pid, n in gi.items() if n["properties"]["name"] == "Stan McCormack")
    gerald = next(pid for pid, n in gi.items() if n["properties"]["name"] == "Gerald Orban")
    kept = {
        "id": gerald,
        "type": "person",
        "display_name": "Gerald Orban",
        "aliases": [],
        "sources": {"gi": True, "kg": True},
    }
    _bridge(p, [kept])
    before = _shas(tmp_path)
    mig = m36.BridgeIdentitiesRestoredMigration()
    assert not mig.verify(_ctx(tmp_path))[0]
    res = mig.apply(_ctx(tmp_path))
    assert res.details["identities_added"] == 1 and len(res.details["episodes"]) == 1
    expected: Dict[str, Dict[str, Any]] = {}
    _collect_identity(expected, gi[stan], source="gi")
    if stan in kg:
        _collect_identity(expected, kg[stan], source="kg")
    idents = {i["id"]: i for i in fx._r(p["bridge"])["identities"]}
    assert idents[stan] == expected[stan]  # the builder's identity, field for field
    assert idents[gerald] == kept  # an identity already there is untouched
    after = _shas(tmp_path)
    assert {k for k in after if after[k] != before.get(k)} == {
        str(p["bridge"].relative_to(tmp_path))
    }
    assert mig.verify(_ctx(tmp_path))[0]
    # Idempotent: a second apply writes nothing.
    again = mig.apply(_ctx(tmp_path))
    assert again.details["identities_added"] == 0 and _shas(tmp_path) == after
    restored, refused = m36.undo(tmp_path)
    assert restored == 1 and not refused and _shas(tmp_path) == before


def test_an_alias_carrying_the_persons_id_is_no_gap(tmp_path: Path) -> None:
    p = _episode(tmp_path)
    ids = list(_persons(p["gi"]))
    _bridge(
        p,
        [
            {
                "id": f"person:unresolved-{i}",
                "type": "person",
                "display_name": "x",
                "aliases": [pid],
                "sources": {"gi": True},
            }
            for i, pid in enumerate(ids)
        ],
    )
    assert (
        m36.BridgeIdentitiesRestoredMigration().apply(_ctx(tmp_path)).details["identities_added"]
        == 0
    )


def test_an_unpublishable_name_is_left_out(tmp_path: Path) -> None:
    p = _episode(tmp_path)
    kg = fx._r(p["kg"])
    kg["nodes"].append(
        {"id": "person:but-sun", "type": "Person", "properties": {"name": "But Sun"}}
    )
    p["kg"].write_text(json.dumps(kg), encoding="utf-8")
    _bridge(
        p,
        [
            {
                "id": pid,
                "type": "person",
                "display_name": n["properties"]["name"],
                "aliases": [],
                "sources": {"gi": True, "kg": True},
            }
            for pid, n in _persons(p["gi"]).items()
        ],
    )
    res = m36.BridgeIdentitiesRestoredMigration().apply(_ctx(tmp_path))
    assert res.details["identities_added"] == 0 and res.details["left_out"] == 1
    assert "person:but-sun" not in {i["id"] for i in fx._r(p["bridge"])["identities"]}
    assert m36.BridgeIdentitiesRestoredMigration().verify(_ctx(tmp_path))[0]


def test_no_fuzzy_merge_is_written(tmp_path: Path) -> None:
    p = _episode(tmp_path)
    _bridge(p, [])
    m36.BridgeIdentitiesRestoredMigration().apply(_ctx(tmp_path))
    assert "fuzzy_merges" not in fx._r(p["bridge"])


def test_a_dry_run_writes_nothing_and_its_rows_carry_counts_and_input_hashes(
    tmp_path: Path,
) -> None:
    p = _episode(tmp_path)
    _bridge(p, [])
    before = _shas(tmp_path)
    res = m36.BridgeIdentitiesRestoredMigration().apply(
        MigrationContext(corpus_root=tmp_path, dry_run=True)
    )
    assert res.details["identities_added"] == 2 and _shas(tmp_path) == before
    assert not (tmp_path / m36.RECEIPTS_FILE).exists()
    rows = list(m36.dry_run_report(tmp_path))
    assert rows and rows[0]["counts"] == {"identities_added": 2}
    assert any(k.endswith(".bridge.json") for k in rows[0]["inputs"])


def test_registered_after_0035_with_its_own_backups_and_receipts() -> None:
    ids = [m.id for m in get_migrations()]
    assert ids.index(m36.MIGRATION_ID) == ids.index("0035_voice_identity_names") + 1
    assert m36.BACKUP_TAG == "0036" and m36.RECEIPTS_FILE == "bridge_identities_restored.jsonl"


# ---------------------------------------------------------------------------------------------
# Rule tests (acceptance criterion 14 and addendum B). Real-shaped fixture bridges come from
# ``build_bridge``; episodes are built by ``_ep`` so each test states its own GI / KG / language.
# ---------------------------------------------------------------------------------------------

OLD = datetime(2025, 3, 4, 5, 6, 7, tzinfo=timezone.utc)
ABSENT: Any = object()
MIG = m36.BridgeIdentitiesRestoredMigration


def _n(nid: str, name: Any = None, *, label: Any = None, aliases: Any = None) -> Dict[str, Any]:
    stripped = nid
    for pre in ("kg:", "k:", "g:"):
        if stripped.startswith(pre):
            stripped = stripped[len(pre) :]
            break
    kind = {
        "person": "Person",
        "org": "Organization",
        "topic": "Topic",
        "object": "Object",
    }.get(stripped.split(":", 1)[0], "Other")
    props: Dict[str, Any] = {}
    if name is not None:
        props["name"] = name
    if label is not None:
        props["label"] = label
    if aliases is not None:
        props["aliases"] = aliases
    return {"id": nid, "type": kind, "properties": props}


def _g(*nodes: Dict[str, Any]) -> Dict[str, Any]:
    return {"nodes": list(nodes), "edges": []}


def _put(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _ep(
    root: Path,
    stem: str = "e1",
    *,
    gi: Any = ABSENT,
    kg: Any = ABSENT,
    bridge: Any = ABSENT,
    feed_language: Any = None,
    episode_language: Any = None,
    run: str = "run_20260101-000000_a",
    feed: str = "f",
    meta: Any = ABSENT,
) -> Dict[str, Path]:
    d = root / "feeds" / feed / run / "metadata"
    p = {k: d / f"{stem}.{k}.json" for k in ("gi", "kg", "bridge")}
    p["meta"] = d / f"{stem}.metadata.json"
    p["txt"] = root / "feeds" / feed / run / "transcripts" / f"{stem}.txt"
    if meta is ABSENT:
        meta = {"feed": {"title": "S"}, "episode": {"episode_id": stem}}
        if feed_language:
            meta["feed"]["language"] = feed_language
        if episode_language:
            meta["episode"]["language"] = episode_language
    _put(p["meta"], meta)
    _put(p["txt"], "A transcript.\n")
    for key, val in (("gi", gi), ("kg", kg), ("bridge", bridge)):
        if val is not ABSENT:
            _put(p[key], val)
    return p


def _built(stem: str, gi: Any, kg: Any) -> Dict[str, Any]:
    return dict(build_bridge(stem, gi, kg, emitted_at=OLD, fuzzy_reconcile=False))


def _ids(path: Path) -> list:
    return [i["id"] for i in json.loads(path.read_text(encoding="utf-8"))["identities"]]


def _ident(path: Path, pid: str) -> Dict[str, Any]:
    return next(
        i for i in json.loads(path.read_text(encoding="utf-8"))["identities"] if i["id"] == pid
    )


def _expected_identity(pid: str, gi: Any, kg: Any) -> Dict[str, Any]:
    """What the pipeline's own builder makes for ``pid``: the GI node, then the KG node."""
    built: Dict[str, Dict[str, Any]] = {}
    for doc, src in ((gi, "gi"), (kg, "kg")):
        for node in (doc or {}).get("nodes") or []:
            tail = node["id"].split(":", 1)[-1] if node["id"][:2] in ("g:", "k:") else node["id"]
            tail = tail[3:] if tail.startswith("kg:") else tail
            if tail == pid:
                _collect_identity(built, node, source=src)
    return built[pid]


def _backups(root: Path) -> list:
    d = backup_dir(root, m36.BACKUP_TAG)
    return sorted(str(f.relative_to(d)) for f in d.rglob("*") if f.is_file()) if d.exists() else []


def _receipts(root: Path) -> bytes:
    f = root / m36.RECEIPTS_FILE
    return f.read_bytes() if f.exists() else b""


def _apply(root: Path, dry_run: bool = False):
    return MIG().apply(MigrationContext(corpus_root=root, dry_run=dry_run))


def _verify(root: Path):
    return MIG().verify(MigrationContext(corpus_root=root))


class _StubEmbedder:
    """Every name is identical to every other: a fuzzy pass would merge anything it may merge."""

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, names: Any, normalize_embeddings: bool = True) -> Any:
        import numpy as np

        self.calls += 1
        return np.ones((len(names), 2)) / (2**0.5)


@pytest.fixture
def stub_embedder(monkeypatch: pytest.MonkeyPatch) -> _StubEmbedder:
    from podcast_scraper.providers.ml import embedding_loader

    stub = _StubEmbedder()
    monkeypatch.setattr(embedding_loader, "get_embedding_model", lambda *a, **k: stub)
    return stub


# --- publishable filter ------------------------------------------------------------------------


def test_unpublishable_names_in_gi_and_kg_are_not_added_and_the_bridge_is_byte_identical(
    tmp_path: Path,
) -> None:
    base_gi = _g(_n("person:gerald-orban", "Gerald Orban"))
    bridge = _built("e1", base_gi, base_gi)
    # "But Sun": a sentence-opener; "Host Mike": a role word; "Ethan He": refused only by the
    # person-shape check (require_person_shape=True), which a cleanup flag would skip.
    gi = _g(*base_gi["nodes"], _n("person:but-sun", "But Sun"), _n("person:ethan-he", "Ethan He"))
    kg = _g(*base_gi["nodes"], _n("person:host-mike", "Host Mike"))
    p = _ep(tmp_path, gi=gi, kg=kg, bridge=bridge)
    before = p["bridge"].read_bytes()
    for name in ("But Sun", "Ethan He", "Host Mike"):
        assert not is_publishable_speaker_name(name, language="en")
    res = _apply(tmp_path)
    assert res.details["identities_added"] == 0 and res.details["left_out"] == 3
    assert res.details["episodes"] == []
    assert p["bridge"].read_bytes() == before
    assert not _backups(tmp_path) and _receipts(tmp_path) == b""
    assert _verify(tmp_path)[0]


def test_a_publishable_gap_beside_unpublishable_ones_is_the_only_one_added(tmp_path: Path) -> None:
    gi = _g(_n("person:but-sun", "But Sun"), _n("person:anna-jones", "Anna Jones"))
    p = _ep(tmp_path, gi=gi, kg=_g(), bridge=_built("e1", _g(), _g()))
    res = _apply(tmp_path)
    assert res.details["identities_added"] == 1 and res.details["left_out"] == 1
    assert _ids(p["bridge"]) == ["person:anna-jones"]


@pytest.mark.parametrize("where", ["feed", "episode"])
def test_the_episodes_language_decides_a_language_specific_refusal(
    tmp_path: Path, where: str
) -> None:
    # "Anfitrión Miguel" ("Host Miguel") is refused by the Spanish vocabulary only.
    assert is_publishable_speaker_name("Anfitrión Miguel", language="en")
    assert not is_publishable_speaker_name("Anfitrión Miguel", language="es")
    gi = _g(_n("person:anfitrion-miguel", "Anfitrión Miguel"))
    empty = _built("x", _g(), _g())
    lang = {"feed_language": "es"} if where == "feed" else {"episode_language": "es"}
    es = _ep(tmp_path, "es", gi=gi, kg=_g(), bridge=empty, **lang)
    en = _ep(tmp_path, "en", gi=gi, kg=_g(), bridge=empty)
    res = _apply(tmp_path)
    assert res.details["identities_added"] == 1 and res.details["left_out"] == 1
    assert _ids(es["bridge"]) == [] and _ids(en["bridge"]) == ["person:anfitrion-miguel"]


def test_a_feed_language_wins_over_an_episode_language(tmp_path: Path) -> None:
    gi = _g(_n("person:anfitrion-miguel", "Anfitrión Miguel"))
    p = _ep(
        tmp_path,
        gi=gi,
        kg=_g(),
        bridge=_built("x", _g(), _g()),
        feed_language="es",
        episode_language="en",
    )
    assert _apply(tmp_path).details["left_out"] == 1 and _ids(p["bridge"]) == []


# --- gate per layer ----------------------------------------------------------------------------


def test_gi_name_publishable_and_kg_name_not_the_gi_node_decides_and_the_person_is_added(
    tmp_path: Path,
) -> None:
    gi = _g(_n("person:anna-jones", "Anna Jones"))
    kg = _g(_n("person:anna-jones", "But Sun"))
    p = _ep(tmp_path, gi=gi, kg=kg, bridge=_built("e1", _g(), _g()))
    res = _apply(tmp_path)
    assert res.details["identities_added"] == 1 and res.details["left_out"] == 0
    assert _ident(p["bridge"], "person:anna-jones") == _expected_identity(
        "person:anna-jones", gi, kg
    )


def test_gi_name_unpublishable_and_kg_name_publishable_the_gi_node_decides_and_it_is_left_out(
    tmp_path: Path,
) -> None:
    gi = _g(_n("person:anna-jones", "But Sun"))
    kg = _g(_n("person:anna-jones", "Anna Jones"))
    p = _ep(tmp_path, gi=gi, kg=kg, bridge=_built("e1", _g(), _g()))
    before = p["bridge"].read_bytes()
    res = _apply(tmp_path)
    assert res.details["identities_added"] == 0 and res.details["left_out"] == 1
    assert p["bridge"].read_bytes() == before


def test_a_kg_only_person_is_gated_by_its_kg_name(tmp_path: Path) -> None:
    kg = _g(_n("person:anna-jones", "Anna Jones"), _n("person:but-sun", "But Sun"))
    p = _ep(tmp_path, gi=_g(), kg=kg, bridge=_built("e1", _g(), _g()))
    res = _apply(tmp_path)
    assert res.details["identities_added"] == 1 and res.details["left_out"] == 1
    assert _ident(p["bridge"], "person:anna-jones")["sources"] == {"gi": False, "kg": True}


# --- aliases -----------------------------------------------------------------------------------


def test_the_added_identity_holds_the_sorted_deduplicated_union_of_both_layers_aliases(
    tmp_path: Path,
) -> None:
    gi = _g(_n("person:anna-jones", "Anna Jones", aliases=["Zed Alias", "Ann Alias"]))
    kg = _g(_n("person:anna-jones", "Anna Jones", aliases=["Ann Alias", "Mid Alias"]))
    p = _ep(tmp_path, gi=gi, kg=kg, bridge=_built("e1", _g(), _g()))
    _apply(tmp_path)
    got = _ident(p["bridge"], "person:anna-jones")
    assert got["aliases"] == ["Ann Alias", "Mid Alias", "Zed Alias"]
    assert got == _expected_identity("person:anna-jones", gi, kg)
    assert got["sources"] == {"gi": True, "kg": True}


def test_a_single_layers_aliases_are_carried_as_the_builder_carries_them(tmp_path: Path) -> None:
    gi = _g(_n("person:anna-jones", "Anna Jones", aliases=["A. Jones"]))
    kg = _g(_n("person:bo-li", "Bo Li", aliases=["B. Li"]))
    p = _ep(tmp_path, gi=gi, kg=kg, bridge=_built("e1", _g(), _g()))
    _apply(tmp_path)
    assert _ident(p["bridge"], "person:anna-jones")["aliases"] == ["A. Jones"]
    assert _ident(p["bridge"], "person:bo-li")["aliases"] == ["B. Li"]


# --- by id or alias ----------------------------------------------------------------------------


def test_a_gap_id_listed_as_an_alias_of_an_existing_identity_is_not_added(tmp_path: Path) -> None:
    gi = _g(_n("person:anna-jones", "Anna Jones"))
    bridge = _built("e1", _g(), _g())
    bridge["identities"] = [
        {
            "id": "person:anna-j",
            "type": "person",
            "display_name": "Anna J",
            "aliases": ["person:anna-jones"],
            "sources": {"gi": True, "kg": False},
        }
    ]
    p = _ep(tmp_path, gi=gi, kg=_g(), bridge=bridge)
    before = p["bridge"].read_bytes()
    res = _apply(tmp_path)
    assert res.details["identities_added"] == 0 and res.details["left_out"] == 0
    assert p["bridge"].read_bytes() == before and _verify(tmp_path)[0]


def test_a_gap_id_equal_to_an_existing_identity_id_is_not_added(tmp_path: Path) -> None:
    gi = _g(_n("person:anna-jones", "Anna Jones"))
    p = _ep(tmp_path, gi=gi, kg=_g(), bridge=_built("e1", gi, _g()))
    before = p["bridge"].read_bytes()
    res = _apply(tmp_path)
    assert res.details["identities_added"] == 0 and p["bridge"].read_bytes() == before


def test_an_alias_that_is_not_a_node_id_does_not_hide_a_gap(tmp_path: Path) -> None:
    gi = _g(_n("person:anna-jones", "Anna Jones"))
    bridge = _built("e1", _g(), _g())
    bridge["identities"] = [
        {
            "id": "person:other",
            "type": "person",
            "display_name": "Other",
            "aliases": ["Anna Jones"],
            "sources": {"gi": True, "kg": False},
        }
    ]
    p = _ep(tmp_path, gi=gi, kg=_g(), bridge=bridge)
    assert _apply(tmp_path).details["identities_added"] == 1
    # D4: the existing list was in id order, so the gap is inserted in id order.
    assert _ids(p["bridge"]) == ["person:anna-jones", "person:other"]


# --- no fuzzy merge ----------------------------------------------------------------------------


def test_two_near_named_persons_stay_separate_when_appended(
    tmp_path: Path, stub_embedder: _StubEmbedder
) -> None:
    gi = _g(_n("person:john-smith", "John Smith"))
    kg = _g(_n("person:jon-smith", "Jon Smith"))
    p = _ep(tmp_path, gi=gi, kg=kg, bridge=_built("e1", gi, _g()))
    assert _apply(tmp_path).details["identities_added"] == 1
    doc = json.loads(p["bridge"].read_text(encoding="utf-8"))
    assert [i["id"] for i in doc["identities"]] == ["person:john-smith", "person:jon-smith"]
    assert "fuzzy_merges" not in doc and stub_embedder.calls == 0
    assert doc["identities"][0]["sources"] == {"gi": True, "kg": False}


def test_two_near_named_persons_stay_separate_when_a_bridge_is_rebuilt(
    tmp_path: Path, stub_embedder: _StubEmbedder
) -> None:
    gi = _g(_n("person:john-smith", "John Smith"), _n("topic:ai", "AI"))
    kg = _g(_n("person:jon-smith", "Jon Smith"), _n("topic:ml", "ML"))
    p = _ep(tmp_path, gi=gi, kg=kg, bridge={})
    _apply(tmp_path)
    doc = json.loads(p["bridge"].read_text(encoding="utf-8"))
    assert [i["id"] for i in doc["identities"]] == [
        "person:john-smith",
        "person:jon-smith",
        "topic:ai",
        "topic:ml",
    ]
    assert "fuzzy_merges" not in doc and stub_embedder.calls == 0


# --- nothing else written ----------------------------------------------------------------------


def test_only_the_new_identities_are_added_every_other_key_and_file_is_byte_identical(
    tmp_path: Path,
) -> None:
    gi = _g(
        _n("person:zed-last", "Zed Last"),
        _n("person:aaa-first", "Aaa First"),
        _n("person:mid-one", "Mid One"),
        _n("org:red-table", "Red Table"),
    )
    kg = _g(_n("person:mid-one", "Mid One"))
    bridge = _built("e1", gi, kg)
    kept = [i for i in bridge["identities"] if i["id"] == "person:mid-one"]
    kept.append(
        {
            "id": "org:red-table",
            "type": "org",
            "display_name": "Red Table",
            "aliases": [],
            "sources": {"gi": True, "kg": False},
        }
    )
    kept.insert(0, {**kept[0], "id": "person:zzz-old", "aliases": ["person:old"]})
    bridge["identities"] = kept
    bridge["fuzzy_merges"] = [{"gi_id": "person:x", "kg_id": "person:y", "similarity": 0.9}]
    bridge["emitted_at"] = "2025-03-04T05:06:07Z"
    bridge["note"] = "keep me"
    p = _ep(tmp_path, gi=gi, kg=kg, bridge=bridge)
    clean = _ep(
        tmp_path, "e2", gi=_g(_n("person:bo-li", "Bo Li")), bridge=_built("e2", _g(), _g()), kg=_g()
    )
    clean_doc = json.loads(clean["bridge"].read_text(encoding="utf-8"))
    clean_doc["identities"] = [
        _expected_identity("person:bo-li", _g(_n("person:bo-li", "Bo Li")), _g())
    ]
    _put(clean["bridge"], clean_doc)
    other_before = {k: v.read_bytes() for k, v in {**p, **clean}.items() if k != "bridge"}
    clean_before, clean_stat = clean["bridge"].read_bytes(), clean["bridge"].stat().st_mtime_ns
    shas = _shas(tmp_path)

    res = _apply(tmp_path)

    doc = json.loads(p["bridge"].read_text(encoding="utf-8"))
    assert res.details["identities_added"] == 2
    assert doc["identities"][: len(kept)] == kept  # existing identities keep their place
    assert [i["id"] for i in doc["identities"][len(kept) :]] == [
        "person:aaa-first",
        "person:zed-last",
    ]
    assert list(doc) == list(bridge)  # the same keys in the same order
    for key in bridge:
        if key != "identities":
            assert doc[key] == bridge[key], key  # emitted_at, schema_version, episode_id, ...
    assert {k: v.read_bytes() for k, v in {**p, **clean}.items() if k != "bridge"} == other_before
    # An episode with no gap is not rewritten: same bytes, same inode time, no backup, no receipt.
    assert clean["bridge"].read_bytes() == clean_before
    assert clean["bridge"].stat().st_mtime_ns == clean_stat
    assert _backups(tmp_path) == [str(p["bridge"].relative_to(tmp_path))]
    assert str(clean["bridge"].relative_to(tmp_path)).encode() not in _receipts(tmp_path)
    after = _shas(tmp_path)
    assert {k for k in after if after[k] != shas.get(k)} == {str(p["bridge"].relative_to(tmp_path))}


def test_an_episode_whose_only_gap_is_unpublishable_is_not_rewritten(tmp_path: Path) -> None:
    gi = _g(_n("person:but-sun", "But Sun"))
    p = _ep(tmp_path, gi=gi, kg=_g(), bridge=_built("e1", _g(), _g()))
    stat = p["bridge"].stat().st_mtime_ns
    res = _apply(tmp_path)
    assert res.details["left_out"] == 1 and res.details["episodes"] == []
    assert p["bridge"].stat().st_mtime_ns == stat
    assert not _backups(tmp_path) and _receipts(tmp_path) == b""


# --- non-person ids ----------------------------------------------------------------------------


def test_org_topic_and_object_nodes_with_no_identity_are_left_for_the_pipeline(
    tmp_path: Path,
) -> None:
    gi = _g(_n("org:red-table", "Red Table"), _n("topic:big-data", "Big Data"))
    kg = _g(_n("object:open-door", "Open Door"), _n("person:anna-jones", "Anna Jones"))
    bridge = _built("e1", _g(), _g())
    p = _ep(tmp_path, gi=gi, kg=kg, bridge=bridge)
    res = _apply(tmp_path)
    assert res.details["identities_added"] == 1 and res.details["left_out"] == 0
    assert _ids(p["bridge"]) == ["person:anna-jones"]


def test_a_corpus_whose_only_gaps_are_not_persons_has_no_gap(tmp_path: Path) -> None:
    gi = _g(_n("org:red-table", "Red Table"), _n("topic:big-data", "Big Data"))
    p = _ep(tmp_path, gi=gi, kg=_g(), bridge=_built("e1", _g(), _g()))
    before = p["bridge"].read_bytes()
    res = _apply(tmp_path)
    assert res.details["identities_added"] == 0 and res.details["left_out"] == 0
    assert p["bridge"].read_bytes() == before and _verify(tmp_path)[0]


# --- layer prefixes ----------------------------------------------------------------------------


def test_layer_prefixes_are_stripped_from_the_added_identity_id(tmp_path: Path) -> None:
    gi = _g(_n("g:person:gina-gee", "Gina Gee"))
    kg = _g(_n("k:person:kai-kay", "Kai Kay"), _n("kg:person:kim-kee", "Kim Kee"))
    p = _ep(tmp_path, gi=gi, kg=kg, bridge=_built("e1", _g(), _g()))
    res = _apply(tmp_path)
    assert res.details["identities_added"] == 3
    assert _ids(p["bridge"]) == ["person:gina-gee", "person:kai-kay", "person:kim-kee"]
    assert _ident(p["bridge"], "person:gina-gee") == _expected_identity("person:gina-gee", gi, kg)
    assert _ident(p["bridge"], "person:kim-kee") == _expected_identity("person:kim-kee", gi, kg)


def test_a_prefixed_node_whose_stripped_id_is_carried_is_no_gap(tmp_path: Path) -> None:
    gi = _g(_n("g:person:gina-gee", "Gina Gee"))
    kg = _g(_n("kg:person:gina-gee", "Gina Gee"))
    p = _ep(tmp_path, gi=gi, kg=kg, bridge=_built("e1", gi, kg))
    before = p["bridge"].read_bytes()
    assert _apply(tmp_path).details["identities_added"] == 0
    assert p["bridge"].read_bytes() == before


def test_one_person_in_both_layers_under_different_prefixes_is_one_identity(tmp_path: Path) -> None:
    gi = _g(_n("g:person:gina-gee", "Gina Gee"))
    kg = _g(_n("k:person:gina-gee", "Gina Gee"))
    p = _ep(tmp_path, gi=gi, kg=kg, bridge=_built("e1", _g(), _g()))
    assert _apply(tmp_path).details["identities_added"] == 1
    assert _ident(p["bridge"], "person:gina-gee")["sources"] == {"gi": True, "kg": True}


# --- display name ------------------------------------------------------------------------------


def test_a_blank_name_falls_back_to_the_label(tmp_path: Path) -> None:
    gi = _g(_n("person:lia-lou", "   ", label="Lia Lou"), _n("person:max-moe", "", label="Max Moe"))
    p = _ep(tmp_path, gi=gi, kg=_g(), bridge=_built("e1", _g(), _g()))
    assert _apply(tmp_path).details["identities_added"] == 2
    assert _ident(p["bridge"], "person:lia-lou")["display_name"] == "Lia Lou"
    assert _ident(p["bridge"], "person:lia-lou") == _expected_identity("person:lia-lou", gi, _g())


def test_a_person_with_neither_name_nor_label_is_left_out_and_counted(tmp_path: Path) -> None:
    gi = _g(
        _n("person:no-name"),
        _n("person:blanks", " ", label="  "),
        _n("person:anna-jones", "Anna Jones"),
    )
    p = _ep(tmp_path, gi=gi, kg=_g(), bridge=_built("e1", _g(), _g()))
    res = _apply(tmp_path)
    assert res.details["identities_added"] == 1 and res.details["left_out"] == 2
    assert _ids(p["bridge"]) == ["person:anna-jones"]


# --- missing and malformed files ---------------------------------------------------------------


def test_an_episode_with_only_one_layer_is_still_repaired_from_it(tmp_path: Path) -> None:
    no_gi = _ep(
        tmp_path, "a", kg=_g(_n("person:anna-jones", "Anna Jones")), bridge=_built("a", _g(), _g())
    )
    no_kg = _ep(tmp_path, "b", gi=_g(_n("person:bo-li", "Bo Li")), bridge=_built("b", _g(), _g()))
    neither = _ep(tmp_path, "c", bridge=_built("c", _g(), _g()))
    before = neither["bridge"].read_bytes()
    res = _apply(tmp_path)
    assert res.details["identities_added"] == 2 and res.details["skipped"] == {}
    assert _ids(no_gi["bridge"]) == ["person:anna-jones"] and _ids(no_kg["bridge"]) == [
        "person:bo-li"
    ]
    assert neither["bridge"].read_bytes() == before


def test_unreadable_metadata_or_bridge_or_one_without_identities_is_skipped_and_counted(
    tmp_path: Path,
) -> None:
    gi = _g(_n("person:anna-jones", "Anna Jones"))
    paths = [
        _ep(tmp_path, "bad-meta", gi=gi, kg=_g(), bridge=_built("x", _g(), _g()), meta="{not json"),
        _ep(tmp_path, "no-bridge", gi=gi, kg=_g()),
        _ep(tmp_path, "bad-bridge", gi=gi, kg=_g(), bridge="{not json"),
        _ep(tmp_path, "list-bridge", gi=gi, kg=_g(), bridge=[1, 2]),
        _ep(
            tmp_path, "no-list", gi=gi, kg=_g(), bridge={"schema_version": "1.0", "episode_id": "x"}
        ),
        _ep(tmp_path, "str-list", gi=gi, kg=_g(), bridge={"episode_id": "x", "identities": "nope"}),
    ]
    before = {str(f): f.read_bytes() for p in paths for f in p.values() if f.exists()}
    res = _apply(tmp_path)
    assert res.details["identities_added"] == 0 and res.details["episodes"] == []
    assert res.details["skipped"] == {
        "metadata unreadable": 1,
        "no bridge or bridge unreadable": 3,
        "bridge has no identities list": 2,
    }
    assert "skipped[metadata unreadable]=1" in res.message
    assert "skipped[no bridge or bridge unreadable]=3" in res.message
    assert "skipped[bridge has no identities list]=2" in res.message
    assert {str(f): f.read_bytes() for p in paths for f in p.values() if f.exists()} == before
    assert _receipts(tmp_path) == b"" and not _backups(tmp_path)
    assert _verify(tmp_path)[0]


def test_a_malformed_gi_or_kg_is_read_as_missing(tmp_path: Path) -> None:
    p = _ep(
        tmp_path,
        gi="{not json",
        kg=["not", "a", "dict"],
        bridge=_built("e1", _g(), _g()),
    )
    before = p["bridge"].read_bytes()
    assert _apply(tmp_path).details["identities_added"] == 0
    assert p["bridge"].read_bytes() == before


# --- superseded runs ---------------------------------------------------------------------------


def test_a_superseded_run_is_not_read_or_written_and_does_not_fail_verify(tmp_path: Path) -> None:
    gi = _g(_n("person:anna-jones", "Anna Jones"))
    old = _ep(
        tmp_path, "e1", run="run_20260101-000000_a", gi=gi, kg=_g(), bridge=_built("e1", _g(), _g())
    )
    new = _ep(
        tmp_path, "e1", run="run_20260301-000000_a", gi=gi, kg=_g(), bridge=_built("e1", _g(), _g())
    )
    old_before = old["bridge"].read_bytes()
    res = _apply(tmp_path)
    assert res.details["identities_added"] == 1
    assert res.details["episodes"] == [str(new["meta"].relative_to(tmp_path))]
    assert old["bridge"].read_bytes() == old_before and _ids(new["bridge"]) == ["person:anna-jones"]
    assert _verify(tmp_path)[0]  # the older run still has its gap and is not what verify reads
    assert [r["meta"] for r in m36.dry_run_report(tmp_path)] == []


def test_a_gap_only_in_a_superseded_run_is_no_gap(tmp_path: Path) -> None:
    gi = _g(_n("person:anna-jones", "Anna Jones"))
    _ep(
        tmp_path, "e1", run="run_20260101-000000_a", gi=gi, kg=_g(), bridge=_built("e1", _g(), _g())
    )
    _ep(tmp_path, "e1", run="run_20260301-000000_a", gi=gi, kg=_g(), bridge=_built("e1", gi, _g()))
    assert _verify(tmp_path)[0]
    assert _apply(tmp_path).details["identities_added"] == 0 and not _backups(tmp_path)


# --- the EMPTY bridge is rebuilt ---------------------------------------------------------------


def _rich() -> tuple:
    gi = _g(
        _n("person:anna-jones", "Anna Jones", aliases=["A. Jones"]),
        _n("person:but-sun", "But Sun"),
        _n("topic:big-data", "Big Data"),
        _n("org:tech-summit", "Tech Summit"),
        _n("object:open-door", "Open Door"),
    )
    kg = _g(
        _n("k:person:anna-jones", "Anna Jones", aliases=["Anna J"]),
        _n("person:host-mike", "Host Mike"),
        _n("person:bo-li", "Bo Li"),
        _n("kg:topic:open-ai", "Open AI"),
        _n("org:red-table", "Red Table"),
    )
    return gi, kg


def _expected_rebuild(eid: str, gi: Any, kg: Any, lang: Any = None) -> Dict[str, Any]:
    exp = dict(build_bridge(eid, gi, kg, emitted_at=m36.REBUILT_EMITTED_AT, fuzzy_reconcile=False))
    exp["identities"] = [
        i
        for i in exp["identities"]
        if i["type"] != "person" or is_publishable_speaker_name(i["display_name"], language=lang)
    ]
    return exp


def test_an_empty_bridge_becomes_the_builders_bridge_minus_unpublishable_persons(
    tmp_path: Path,
) -> None:
    gi, kg = _rich()
    p = _ep(tmp_path, gi=gi, kg=kg, bridge={})
    shas = _shas(tmp_path)
    res = _apply(tmp_path)
    expected = _expected_rebuild("e1", gi, kg)
    got = json.loads(p["bridge"].read_text(encoding="utf-8"))
    assert json.dumps(got, ensure_ascii=False) == json.dumps(expected, ensure_ascii=False)
    assert got["episode_id"] == "e1" and got["schema_version"] == "1.0"
    assert (
        got["emitted_at"]
        == "2026-10-26T00:00:00Z"
        == m36.REBUILT_EMITTED_AT.strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    assert "fuzzy_merges" not in got
    types = sorted(i["type"] for i in got["identities"])
    assert types == ["object", "org", "org", "person", "person", "topic", "topic"]
    assert "person:but-sun" not in _ids(p["bridge"]) and "person:host-mike" not in _ids(p["bridge"])
    assert "org:tech-summit" in _ids(p["bridge"])  # the filter is for persons only
    d = res.details
    assert d["bridges_rebuilt"] == 1 and d["rebuilt_identities"] == len(expected["identities"])
    assert d["identities_added"] == 2 and d["left_out"] == 2 and len(d["episodes"]) == 1
    after = _shas(tmp_path)
    assert {k for k in after if after[k] != shas.get(k)} == {str(p["bridge"].relative_to(tmp_path))}


def test_a_rebuilt_bridge_applies_the_episodes_language(tmp_path: Path) -> None:
    gi = _g(_n("person:anfitrion-miguel", "Anfitrión Miguel"), _n("org:red-table", "Red Table"))
    es = _ep(tmp_path, "es", gi=gi, kg=_g(), bridge={}, feed_language="es")
    en = _ep(tmp_path, "en", gi=gi, kg=_g(), bridge={})
    _apply(tmp_path)
    assert _ids(es["bridge"]) == ["org:red-table"]
    assert _ids(en["bridge"]) == ["org:red-table", "person:anfitrion-miguel"]
    assert json.loads(es["bridge"].read_text(encoding="utf-8")) == _expected_rebuild(
        "es", gi, _g(), "es"
    )


def test_an_unpublishable_person_in_one_layer_only_is_left_out_of_the_rebuild(
    tmp_path: Path,
) -> None:
    gi = _g(_n("person:anna-jones", "Anna Jones"))
    kg = _g(_n("person:but-sun", "But Sun"), _n("topic:big-data", "Big Data"))
    p = _ep(tmp_path, gi=gi, kg=kg, bridge={})
    _apply(tmp_path)
    assert _ids(p["bridge"]) == ["person:anna-jones", "topic:big-data"]


def test_only_a_damaged_bridge_is_rebuilt_wholesale(tmp_path: Path) -> None:
    gi, kg = _rich()
    # An `identities: []` bridge is a bridge that is there: persons are appended, nothing else, and
    # its own emitted_at and keys stay.
    empty_list = {
        "schema_version": "1.0",
        "episode_id": "keep",
        "emitted_at": "2025-03-04T05:06:07Z",
        "identities": [],
    }
    a = _ep(tmp_path, "a", gi=gi, kg=kg, bridge=empty_list)
    # A non-empty bridge is appended to, never replaced.
    one = {**empty_list, "identities": [_expected_identity("person:anna-jones", gi, kg)]}
    b = _ep(tmp_path, "b", gi=gi, kg=kg, bridge=one)
    # D3: a bridge without episode_id is damaged and rebuilt like `{}`.
    other = {"schema_version": "1.0"}
    c = _ep(tmp_path, "c", gi=gi, kg=kg, bridge=other)
    res = _apply(tmp_path)
    for key in "ab":
        doc = json.loads({"a": a, "b": b}[key]["bridge"].read_text(encoding="utf-8"))
        assert doc["emitted_at"] == "2025-03-04T05:06:07Z" and doc["episode_id"] == "keep"
        assert all(i["type"] == "person" for i in doc["identities"])
    assert _ids(a["bridge"]) == ["person:anna-jones", "person:bo-li"]
    assert _ids(b["bridge"]) == ["person:anna-jones", "person:bo-li"]
    rebuilt = json.loads(c["bridge"].read_text(encoding="utf-8"))
    assert "episode_id" in rebuilt and rebuilt["emitted_at"] != "2025-03-04T05:06:07Z"
    assert res.details["bridges_rebuilt"] == 1 and res.details["skipped"] == {}


def test_an_empty_bridge_with_no_gi_and_no_kg_is_skipped_with_a_counted_reason(
    tmp_path: Path,
) -> None:
    p = _ep(tmp_path, bridge={})
    before = p["bridge"].read_bytes()
    res = _apply(tmp_path)
    assert p["bridge"].read_bytes() == before, "rebuilt from nothing: an `identities: []` bridge"
    assert res.details["bridges_rebuilt"] == 0 and sum(res.details["skipped"].values()) == 1


def test_an_empty_bridge_with_an_unreadable_gi_is_skipped_with_a_counted_reason(
    tmp_path: Path,
) -> None:
    kg = _g(_n("person:bo-li", "Bo Li"), _n("topic:big-data", "Big Data"))
    p = _ep(tmp_path, gi="{not json", kg=kg, bridge={})
    before = p["bridge"].read_bytes()
    res = _apply(tmp_path)
    assert p["bridge"].read_bytes() == before, "rebuilt from the KG alone, GI silently ignored"
    assert res.details["bridges_rebuilt"] == 0 and sum(res.details["skipped"].values()) == 1


# --- idempotence, undo, verify -----------------------------------------------------------------


def _mixed(root: Path) -> Dict[str, Dict[str, Path]]:
    gi, kg = _rich()
    return {
        "rebuild": _ep(root, "r", gi=gi, kg=kg, bridge={}),
        "append": _ep(
            root,
            "a",
            gi=_g(_n("person:anna-jones", "Anna Jones"), _n("person:bo-li", "Bo Li")),
            kg=_g(),
            bridge=_built("a", _g(_n("person:anna-jones", "Anna Jones")), _g()),
        ),
        "clean": _ep(root, "c", gi=_g(), kg=_g(), bridge=_built("c", _g(), _g())),
        "junk": _ep(
            root,
            "j",
            gi=_g(_n("person:but-sun", "But Sun")),
            kg=_g(),
            bridge=_built("j", _g(), _g()),
        ),
    }


def test_a_second_apply_writes_nothing_and_leaves_the_census_unchanged(tmp_path: Path) -> None:
    eps = _mixed(tmp_path)
    first = _apply(tmp_path)
    assert len(first.details["episodes"]) == 2
    after, receipts, backups = _shas(tmp_path), _receipts(tmp_path), _backups(tmp_path)
    stats = {k: v["bridge"].stat().st_mtime_ns for k, v in eps.items()}
    again = _apply(tmp_path)
    assert again.details["episodes"] == [] and again.details["identities_added"] == 0
    assert again.details["bridges_rebuilt"] == 0 and again.details["rebuilt_identities"] == 0
    assert _shas(tmp_path) == after and _receipts(tmp_path) == receipts
    assert _backups(tmp_path) == backups
    assert {k: v["bridge"].stat().st_mtime_ns for k, v in eps.items()} == stats
    assert _verify(tmp_path)[0]


def test_undo_apply_undo_is_byte_identical_and_the_rebuild_is_deterministic(tmp_path: Path) -> None:
    eps = _mixed(tmp_path)
    base = _shas(tmp_path)
    assert eps["rebuild"]["bridge"].read_text(encoding="utf-8") == json.dumps({}, indent=2) + "\n"
    _apply(tmp_path)
    first = _shas(tmp_path)
    assert first != base
    assert m36.undo(tmp_path) == (2, [])
    assert _shas(tmp_path) == base
    assert not _verify(tmp_path)[0]
    time.sleep(1.1)  # a clock-stamped rebuild would differ by now
    _apply(tmp_path)
    assert _shas(tmp_path) == first
    assert m36.undo(tmp_path) == (2, [])
    assert _shas(tmp_path) == base


def test_an_undo_refuses_a_bridge_changed_since_the_migration_wrote_it(tmp_path: Path) -> None:
    eps = _mixed(tmp_path)
    _apply(tmp_path)
    doc = json.loads(eps["append"]["bridge"].read_text(encoding="utf-8"))
    doc["note"] = "edited later"
    _put(eps["append"]["bridge"], doc)
    restored, refused = m36.undo(tmp_path)
    assert restored == 1 and len(refused) == 1 and "changed since" in refused[0]


def test_verify_fails_on_a_gap_and_on_an_empty_bridge_and_passes_after(tmp_path: Path) -> None:
    eps = _mixed(tmp_path)
    ok, msg = _verify(tmp_path)
    assert not ok and "3 published GI / KG person(s) still lack a bridge identity" in msg
    assert "1 empty bridge(s)" in msg
    _apply(tmp_path)
    assert _verify(tmp_path) == (True, "every published GI / KG person has a bridge identity")
    # one rebuilt file emptied again: FAIL on that one episode (and only on it)
    _put(eps["rebuild"]["bridge"], {})
    ok, msg = _verify(tmp_path)
    assert not ok and "1 empty bridge(s)" in msg
    _apply(tmp_path)
    assert _verify(tmp_path)[0]
    # one identity removed from one bridge: a gap of 1
    doc = json.loads(eps["append"]["bridge"].read_text(encoding="utf-8"))
    doc["identities"] = [i for i in doc["identities"] if i["id"] != "person:bo-li"]
    _put(eps["append"]["bridge"], doc)
    ok, msg = _verify(tmp_path)
    assert not ok and msg.startswith("1 published GI / KG person(s) still lack")


def test_verify_passes_on_an_empty_corpus_and_when_only_unpublishable_gaps_remain(
    tmp_path: Path,
) -> None:
    assert _verify(tmp_path)[0]
    _ep(tmp_path, gi=_g(_n("person:but-sun", "But Sun")), kg=_g(), bridge=_built("e1", _g(), _g()))
    assert _verify(tmp_path)[0]


def test_verify_and_the_plan_read_the_same_publishable_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ep(
        tmp_path,
        gi=_g(_n("person:anna-jones", "Anna Jones")),
        kg=_g(),
        bridge=_built("e1", _g(), _g()),
    )
    assert not _verify(tmp_path)[0]
    monkeypatch.setattr(m36, "is_publishable_speaker_name", lambda *a, **k: False)
    assert _verify(tmp_path)[0]
    assert "0 person identit(ies) to restore in 0 episode(s)" in MIG().plan(_ctx(tmp_path))


def test_the_plan_counts_identities_episodes_and_rebuilt_bridges(tmp_path: Path) -> None:
    _mixed(tmp_path)
    assert MIG().plan(_ctx(tmp_path)) == (
        f"{m36.MIGRATION_ID}: 3 person identit(ies) to restore in 2 episode(s), "
        "1 empty bridge(s) rebuilt"
    )


def test_only_a_person_typed_node_with_a_person_id_is_a_person(tmp_path: Path) -> None:
    # Two guards keep non-persons out: the node type and the id prefix. Real graphs agree on both,
    # so each is pinned alone with a node that breaks only that one.
    org_id = _n("org:red-table", "Red Table")
    org_id["type"] = "Person"
    topic_id = _n("topic:big-data", "Big Data")
    topic_id["type"] = "Person"
    object_id = _n("object:open-door", "Open Door")
    object_id["type"] = "Person"
    typed_wrong = _n("person:anna-jones", "Anna Jones")
    typed_wrong["type"] = "Topic"
    p = _ep(
        tmp_path,
        gi=_g(org_id, topic_id),
        kg=_g(object_id, typed_wrong),
        bridge=_built("e1", _g(), _g()),
    )
    before = p["bridge"].read_bytes()
    res = _apply(tmp_path)
    assert res.details["identities_added"] == 0 and res.details["left_out"] == 0
    assert p["bridge"].read_bytes() == before and _verify(tmp_path)[0]


def test_an_empty_bridge_with_no_person_to_add_is_still_a_gap_and_a_rebuild(
    tmp_path: Path,
) -> None:
    gi = _g(_n("topic:big-data", "Big Data"), _n("org:red-table", "Red Table"))
    p = _ep(tmp_path, gi=gi, kg=_g(), bridge={})
    ok, msg = _verify(tmp_path)
    assert not ok and msg.startswith("0 published GI / KG person(s)") and "1 empty bridge(s)" in msg
    assert MIG().plan(_ctx(tmp_path)) == (
        f"{m36.MIGRATION_ID}: 0 person identit(ies) to restore in 1 episode(s), "
        "1 empty bridge(s) rebuilt"
    )
    assert [r["counts"] for r in m36.dry_run_report(tmp_path)] == [
        {"identities_added": 0, "rebuilt_identities": 2}
    ]
    res = _apply(tmp_path)
    assert res.details["bridges_rebuilt"] == 1 and res.details["identities_added"] == 0
    assert _ids(p["bridge"]) == ["org:red-table", "topic:big-data"]
    assert _verify(tmp_path)[0]


@pytest.mark.parametrize("kg", ["{not json", None])
def test_an_empty_bridge_with_an_unreadable_or_missing_kg_is_skipped_with_a_counted_reason(
    tmp_path: Path, kg: Any
) -> None:
    gi = _g(_n("person:bo-li", "Bo Li"), _n("topic:big-data", "Big Data"))
    p = _ep(tmp_path, gi=gi, kg=kg, bridge={})
    before = p["bridge"].read_bytes()
    res = _apply(tmp_path)
    assert p["bridge"].read_bytes() == before, "rebuilt from the GI alone, KG silently ignored"
    assert res.details["bridges_rebuilt"] == 0
    assert res.details["skipped"] == {"empty bridge without a readable GI and KG": 1}


# --- acceptance-D: D3 damaged bridges (no episode_id) and D4 id order --------------------------


def _read(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


@pytest.mark.parametrize(
    "damaged",
    [{"identities": []}, {"schema_version": "1.0", "identities": []}],
    ids=["identities-only", "identities-and-schema-version"],
)
def test_a_bridge_with_an_empty_identities_list_and_no_episode_id_is_rebuilt_whole(
    tmp_path: Path, damaged: Dict[str, Any]
) -> None:
    gi, kg = _rich()
    p = _ep(tmp_path, gi=gi, kg=kg, bridge=damaged)
    res = _apply(tmp_path)
    got = _read(p["bridge"])
    expected = _expected_rebuild("e1", gi, kg)
    assert json.dumps(got, ensure_ascii=False) == json.dumps(expected, ensure_ascii=False)
    assert got["episode_id"] == "e1" and got["schema_version"] == "1.0"
    assert "fuzzy_merges" not in got
    assert {"org:tech-summit", "topic:big-data", "object:open-door"} <= set(_ids(p["bridge"]))
    assert "person:but-sun" not in _ids(p["bridge"])
    assert res.details["bridges_rebuilt"] == 1 and res.details["skipped"] == {}
    assert _verify(tmp_path)[0]


@pytest.mark.parametrize(
    "bridge",
    [
        {"schema_version": "1.0", "episode_id": "e1", "identities": []},
        {
            "schema_version": "1.0",
            "episode_id": "e1",
            "note": "x",
            "emitted_at": "t",
            "identities": [],
        },
        {"episode_id": "someone-else", "identities": []},
        {"episode_id": "e1", "identities": [{"id": "topic:big-data", "type": "topic"}]},
    ],
    ids=["empty-list", "extra-keys", "episode-id-differs-from-metadata", "entries-no-schema"],
)
def test_a_bridge_with_an_episode_id_is_never_rebuilt_and_a_gapless_one_is_byte_equal(
    tmp_path: Path, bridge: Dict[str, Any]
) -> None:
    # The rule is "no episode_id": `identities: []` WITH an episode_id is a bridge that is there.
    gi = _g(_n("topic:big-data", "Big Data"), _n("org:red-table", "Red Table"))
    p = _ep(tmp_path, gi=gi, kg=_g(), bridge=bridge)
    before = p["bridge"].read_bytes()
    res = _apply(tmp_path)
    assert res.details["bridges_rebuilt"] == 0 and res.details["identities_added"] == 0
    assert res.details["episodes"] == [] and res.details["skipped"] == {}
    assert p["bridge"].read_bytes() == before
    assert _verify(tmp_path)[0]


def test_a_complete_bridge_missing_persons_is_appended_to_not_rebuilt(tmp_path: Path) -> None:
    gi = _g(_n("person:anna-jones", "Anna Jones"), _n("topic:big-data", "Big Data"))
    bridge = {
        "schema_version": "1.0",
        "episode_id": "keep",
        "emitted_at": "2025-03-04T05:06:07Z",
        "identities": [],
    }
    p = _ep(tmp_path, gi=gi, kg=_g(), bridge=bridge)
    res = _apply(tmp_path)
    assert res.details["bridges_rebuilt"] == 0 and res.details["identities_added"] == 1
    got = _read(p["bridge"])
    assert got["episode_id"] == "keep" and got["emitted_at"] == "2025-03-04T05:06:07Z"
    assert _ids(p["bridge"]) == ["person:anna-jones"]  # the topic is not added: not a rebuild


def test_an_empty_string_episode_id_counts_as_present_and_the_bridge_is_not_rebuilt(
    tmp_path: Path,
) -> None:
    # Chosen semantics, pinned: the damage test is `"episode_id" not in bridge`, so a bridge that
    # carries the key with "" is a bridge that is there and is not rebuilt. (Rebuilding it would
    # also write nothing better: the metadata's id is what the builder would use either way.)
    gi = _g(_n("topic:big-data", "Big Data"))
    p = _ep(tmp_path, gi=gi, kg=_g(), bridge={"episode_id": "", "identities": []})
    before = p["bridge"].read_bytes()
    res = _apply(tmp_path)
    assert res.details["bridges_rebuilt"] == 0 and res.details["episodes"] == []
    assert p["bridge"].read_bytes() == before and _verify(tmp_path)[0]


@pytest.mark.parametrize(
    "gi, kg",
    [
        (ABSENT, _g(_n("person:bo-li", "Bo Li"))),
        (_g(_n("person:bo-li", "Bo Li")), ABSENT),
        ("{not json", _g(_n("person:bo-li", "Bo Li"))),
        (_g(_n("person:bo-li", "Bo Li")), "{not json"),
    ],
    ids=["no-gi", "no-kg", "unreadable-gi", "unreadable-kg"],
)
def test_a_damaged_bridge_without_a_readable_gi_or_kg_is_skipped_with_a_counted_reason(
    tmp_path: Path, gi: Any, kg: Any
) -> None:
    p = _ep(tmp_path, gi=gi, kg=kg, bridge={"identities": []})
    before = p["bridge"].read_bytes()
    res = _apply(tmp_path)
    reason = m36.SKIP_REASONS[3]
    assert res.details["bridges_rebuilt"] == 0 and res.details["episodes"] == []
    assert res.details["skipped"] == {reason: 1}
    assert f"skipped[{reason}]=1" in res.message
    assert p["bridge"].read_bytes() == before
    assert _receipts(tmp_path) == b"" and not _backups(tmp_path)


@pytest.mark.parametrize(
    "damaged", [{}, {"identities": []}], ids=["empty-object", "identities-only"]
)
def test_verify_fails_on_a_damaged_bridge_and_names_the_count_and_passes_after_apply(
    tmp_path: Path, damaged: Dict[str, Any]
) -> None:
    gi, kg = _rich()
    _ep(tmp_path, "a", gi=gi, kg=kg, bridge=damaged)
    _ep(tmp_path, "b", gi=gi, kg=kg, bridge=dict(damaged))
    ok, msg = _verify(tmp_path)
    assert not ok and "2 empty bridge(s)" in msg
    _apply(tmp_path)
    assert _verify(tmp_path) == (True, "every published GI / KG person has a bridge identity")


# --- acceptance-D: D4 id order ----------------------------------------------------------------

_PEOPLE = {
    "anna": _n("person:anna-jones", "Anna Jones"),
    "carl": _n("person:carl-davis", "Carl Davis"),
    "eve": _n("person:eve-moore", "Eve Moore"),
    "lena": _n("person:lena-ross", "Lena Ross"),
    "zack": _n("person:zack-young", "Zack Young"),
}


def _existing_bridge(*keys: str, reverse: bool = False) -> Dict[str, Any]:
    bridge = _built("e1", _g(*[_PEOPLE[k] for k in keys]), _g())
    if reverse:
        bridge["identities"] = list(reversed(bridge["identities"]))
    return bridge


def _insert(tmp_path: Path, existing: Dict[str, Any], *new_keys: str) -> Dict[str, Any]:
    """Apply with GI holding the existing persons plus ``new_keys``; return the result checks."""
    held = [i["id"] for i in existing["identities"]]
    nodes = [n for n in _PEOPLE.values() if n["id"] in held] + [_PEOPLE[k] for k in new_keys]
    p = _ep(tmp_path, gi=_g(*nodes), kg=_g(), bridge=existing)
    res = _apply(tmp_path)
    written = _read(p["bridge"])["identities"]
    ids = [i["id"] for i in written]
    assert len(ids) == len(set(ids)), "duplicate ids"
    assert res.details["identities_added"] == len(new_keys) and res.details["bridges_rebuilt"] == 0
    # the existing identities, in the written list, equal the original list: none moved or changed
    assert [i for i in written if i["id"] in held] == existing["identities"]
    return {"ids": ids, "held": held}


@pytest.mark.parametrize(
    "new, expected",
    [
        ("anna", ["anna", "carl", "lena"]),
        ("eve", ["carl", "eve", "lena"]),
        ("zack", ["carl", "lena", "zack"]),
    ],
    ids=["before", "middle", "after"],
)
def test_one_inserted_person_lands_in_id_order_in_a_sorted_list(
    tmp_path: Path, new: str, expected: list
) -> None:
    out = _insert(tmp_path, _existing_bridge("carl", "lena"), new)
    assert out["ids"] == [_PEOPLE[k]["id"] for k in expected]
    assert out["ids"] == sorted(out["ids"])


def test_two_inserted_persons_sort_around_an_existing_one(tmp_path: Path) -> None:
    out = _insert(tmp_path, _existing_bridge("carl"), "zack", "anna")
    assert out["ids"] == [_PEOPLE[k]["id"] for k in ("anna", "carl", "zack")]


def test_persons_inserted_into_an_empty_list_come_out_in_id_order(tmp_path: Path) -> None:
    out = _insert(tmp_path, _existing_bridge(), "zack", "anna", "eve")
    assert out["ids"] == [_PEOPLE[k]["id"] for k in ("anna", "eve", "zack")]


def test_a_person_already_carried_is_not_inserted_twice(tmp_path: Path) -> None:
    existing = _existing_bridge("carl", "lena")
    p = _ep(
        tmp_path,
        gi=_g(_PEOPLE["carl"], _PEOPLE["lena"], _PEOPLE["eve"]),
        kg=_g(_PEOPLE["carl"], _PEOPLE["eve"]),
        bridge=existing,
    )
    res = _apply(tmp_path)
    assert res.details["identities_added"] == 1
    assert _ids(p["bridge"]) == ["person:carl-davis", "person:eve-moore", "person:lena-ross"]


@pytest.mark.parametrize(
    "new, expected",
    [
        (("anna",), ["lena", "carl", "anna"]),
        (("eve",), ["lena", "carl", "eve"]),
        (("anna", "zack"), ["lena", "carl", "anna", "zack"]),
    ],
    ids=["one-sorting-first", "one-sorting-between", "two-at-first-and-last-position"],
)
def test_an_unsorted_list_keeps_its_order_and_the_new_persons_are_appended(
    tmp_path: Path, new: tuple, expected: list
) -> None:
    existing = _existing_bridge("carl", "lena", reverse=True)
    assert [i["id"] for i in existing["identities"]] == ["person:lena-ross", "person:carl-davis"]
    out = _insert(tmp_path, existing, *new)
    assert out["ids"] == [_PEOPLE[k]["id"] for k in expected]
    assert out["ids"] != sorted(out["ids"])
