"""m0041: a joined person's name in the episode files takes the group's one spelling.

Synthetic episodes built with m0025's fixture (the same name-map rewrite); the ledger view is
passed in. Invented names only.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from podcast_scraper.identity.display import PersonDisplay
from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations import m0041_joined_names_respelled as m39
from podcast_scraper.upgrade.registry import get_migrations

pytestmark = [pytest.mark.unit]

_spec = importlib.util.spec_from_file_location(
    "m0025_fixture", Path(__file__).with_name("test_m0025_one_person_one_entry.py")
)
assert _spec and _spec.loader
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)

TURNS = [
    ("SPEAKER_00", "Korin Vaal", "host", "Welcome to the show, everyone."),
    ("SPEAKER_01", "Odile Brand", "guest", "Thanks for having me, I met Korin Vaal last year."),
    ("SPEAKER_00", "Korin Vaal", "host", "Tell me about the book."),
]
VIEW = m39.LedgerView(
    PersonDisplay({"person:korin-vaal": "Korin Vael", "person:korin-vael": "Korin Vael"}),
    {"person:korin-vaal": "person:korin-vael"},
    {"person:korin-vaal": 3, "person:korin-vael": 5},
)


def _episode(tmp_path: Path, **kw):
    return fx._corpus(
        tmp_path,
        speakers=kw.pop(
            "speakers",
            [
                fx._speaker("Korin Vaal", "host", ["SPEAKER_00"], "known_hosts", "host"),
                fx._speaker("Odile Brand", "guest", ["SPEAKER_01"], "llm", "guest"),
            ],
        ),
        turns=kw.pop("turns", TURNS),
        known_hosts=["Korin Vaal"],
        **kw,
    )


def _mig(view=VIEW):
    mig = m39.JoinedNamesRespelledMigration()
    mig._view_override = view
    return mig


def _apply(root: Path, view=VIEW, dry_run: bool = False):
    return _mig(view).apply(MigrationContext(corpus_root=root, dry_run=dry_run))


def _verify(root: Path, view=VIEW):
    return _mig(view).verify(MigrationContext(corpus_root=root))


def _spoken(text: str):
    """Each line's words after its ``name:`` label: what was said, which must never change."""
    return [line.split(": ", 1)[-1] for line in text.splitlines()]


def test_registered_after_0039() -> None:
    ids = [m.id for m in get_migrations()]
    assert ids[ids.index(m39.MIGRATION_ID) - 1] == "0040_gi_quote_offsets_realigned"


def test_a_joined_speaker_takes_the_group_spelling_on_every_surface_and_keeps_its_id(
    tmp_path: Path,
) -> None:
    p = _episode(
        tmp_path,
        speakers=[  # guest first, own ids: 0025's renumbering would reorder and rename them
            fx._speaker("Odile Brand", "guest", ["SPEAKER_01"], "llm", "speaker_b"),
            fx._speaker("Korin Vaal", "host", ["SPEAKER_00"], "known_hosts", "speaker_a"),
        ],
    )
    meta = fx._r(p["meta"])
    meta["content"]["detected_hosts"] = ["Korin Vaal"]
    fx._w(p["meta"], meta)
    text_before = p["txt"].read_text(encoding="utf-8")
    seg_before = fx._r(p["seg"])
    assert not _verify(tmp_path)[0]
    _apply(tmp_path, dry_run=True)
    assert p["txt"].read_text(encoding="utf-8") == text_before  # a dry run writes nothing
    res = _apply(tmp_path)
    assert res.details["refused"] == []
    text = p["txt"].read_text(encoding="utf-8")
    assert text.startswith("Korin Vael: Welcome to the show") and "\nKorin Vaal:" not in text
    assert _spoken(text) == _spoken(text_before)  # every spoken word, the name inside it too
    segs = fx._r(p["seg"])
    assert [r["speaker_label"] for r in segs] == ["Korin Vael", "Odile Brand", "Korin Vael"]
    assert [(r["text"], r["start"], r["end"]) for r in segs] == [
        (r["text"], r["start"], r["end"]) for r in seg_before
    ]
    content = fx._r(p["meta"])["content"]
    assert [s["name"] for s in content["speakers"]] == ["Odile Brand", "Korin Vael"]
    assert [s["id"] for s in content["speakers"]] == ["speaker_b", "speaker_a"]  # kept as is
    assert content["detected_hosts"] == ["Korin Vaal"]  # the publisher hint is evidence
    for layer in ("kg", "gi"):
        people = {
            n["id"]: n["properties"]["name"]
            for n in fx._r(p[layer])["nodes"]
            if n["type"] == "Person"
        }
        assert people == {"person:korin-vaal": "Korin Vael", "person:odile-brand": "Odile Brand"}
    quotes = [n for n in fx._r(p["gi"])["nodes"] if n["type"] == "Quote"]
    assert {q["properties"]["speaker_id"] for q in quotes} == {
        "person:korin-vaal",
        "person:odile-brand",
    }
    fx._quotes_hold(p)
    diag = fx._r(p["diag"])
    assert {v["resolved_name"] for v in diag["voices"]} == {"Korin Vael", "Odile Brand"}
    assert diag["tried"]["known_hosts"] == ["Korin Vaal"]
    turns = json.loads(p["turns"].read_text(encoding="utf-8"))
    assert {t["speaker_label"] for t in turns["turns"]} == {"Korin Vael", "Odile Brand"}
    assert _verify(tmp_path)[0]
    assert _apply(tmp_path).details["episodes"] == []  # a second run writes nothing
    restored, refused = m39.undo(tmp_path)
    assert restored and not refused
    assert p["txt"].read_text(encoding="utf-8") == text_before


def test_a_mentioned_person_is_respelled_and_the_transcript_is_not_touched(tmp_path: Path) -> None:
    turns = [
        ("SPEAKER_00", "Odile Brand", "host", "Welcome to the show."),
        ("SPEAKER_01", "Mara Tull", "guest", "Korin Vaal told me that."),
    ]
    p = _episode(
        tmp_path,
        speakers=[
            fx._speaker("Odile Brand", "host", ["SPEAKER_00"], "known_hosts", "host"),
            fx._speaker("Mara Tull", "guest", ["SPEAKER_01"], "llm", "guest"),
        ],
        turns=turns,
    )
    kg = fx._r(p["kg"])
    kg["nodes"].append(
        {
            "id": "person:korin-vaal",
            "type": "Person",
            "properties": {"name": "Korin Vaal", "label": "Korin Vaal", "role": "mentioned"},
        }
    )
    fx._w(p["kg"], kg)
    bridge = p["meta"].with_name("ep.bridge.json")
    fx._w(bridge, {"identities": [{"id": "person:korin-vaal", "display_name": "Korin Vaal"}]})
    # A transcript that is no longer the exact render: its turns do not rebuild. A mention-only
    # respelling must not touch them, nor refuse the episode for it.
    p["txt"].write_text(p["txt"].read_text(encoding="utf-8") + "\n[music]\n", encoding="utf-8")
    files_before = {k: v.read_bytes() for k, v in p.items() if k in ("txt", "seg", "turns")}
    res = _apply(tmp_path)
    assert res.details["refused"] == []
    node = next(n for n in fx._r(p["kg"])["nodes"] if n["id"] == "person:korin-vaal")
    assert (node["properties"]["name"], node["properties"]["label"]) == ("Korin Vael",) * 2
    assert fx._r(bridge)["identities"][0]["display_name"] == "Korin Vael"
    assert {k: p[k].read_bytes() for k in files_before} == files_before


def test_a_name_with_an_unjoined_node_is_refused_and_left(tmp_path: Path) -> None:
    p = _episode(tmp_path)
    gi = fx._r(p["gi"])
    gi["nodes"].append(
        {"id": "person:korin-vaal-2", "type": "Person", "properties": {"name": "Korin Vaal"}}
    )
    fx._w(p["gi"], gi)
    res = _apply(tmp_path)
    assert res.details["episodes"] == []
    assert fx._r(p["seg"])[0]["speaker_label"] == "Korin Vaal"
    renames, folds, refused = m39.episode_renames(p["meta"], VIEW)
    assert (
        renames == {}
        and folds == {}
        and refused == {"Korin Vaal": "a node of the name is not joined"}
    )


def test_no_ledger_writes_nothing_and_verify_says_so(tmp_path: Path) -> None:
    p = _episode(tmp_path)
    before = p["seg"].read_bytes()
    mig = m39.JoinedNamesRespelledMigration()
    assert mig.apply(MigrationContext(corpus_root=tmp_path)).details["episodes"] == []
    ok, why = mig.verify(MigrationContext(corpus_root=tmp_path))
    assert not ok and "no valid identity ledger" in why
    assert p["seg"].read_bytes() == before


ITEM8_TURNS = [
    ("SPEAKER_00", "Korin Vaal", "host", "Welcome to the show."),
    ("SPEAKER_01", "Odile Brand", "guest", "Thanks."),
]


def _item8(tmp_path: Path):
    """One person listed twice: a voice under a joined spelling, an unplaced entry under another,
    with its own Person node."""
    p = _episode(
        tmp_path,
        speakers=[
            fx._speaker("Korin Vaal", "host", ["SPEAKER_00"], "known_hosts", "host"),
            fx._speaker("Odile Brand", "guest", ["SPEAKER_01"], "llm", "guest"),
            fx._speaker("Korin Vahl", "host", [], "known_hosts", "unplaced_1"),
        ],
        turns=ITEM8_TURNS,
    )
    kg = fx._r(p["kg"])
    kg["nodes"].append(
        {"id": "person:korin-vahl", "type": "Person", "properties": {"name": "Korin Vahl"}}
    )
    kg["edges"] = [{"type": "MENTIONS", "from": "episode:ep", "to": "person:korin-vahl"}]
    fx._w(p["kg"], kg)
    return p


def test_one_person_listed_twice_folds_into_the_joined_id(tmp_path: Path) -> None:
    p = _item8(tmp_path)
    res = _apply(tmp_path)
    assert res.details["refused"] == []
    names = [s["name"] for s in fx._r(p["meta"])["content"]["speakers"]]
    assert names == ["Korin Vael", "Odile Brand"]
    kg = fx._r(p["kg"])
    ids = {n["id"] for n in kg["nodes"] if n["type"] == "Person"}
    assert ids == {"person:korin-vaal", "person:odile-brand"}
    assert kg["edges"] == [{"type": "MENTIONS", "from": "episode:ep", "to": "person:korin-vaal"}]


def test_a_fold_is_refused_when_the_variant_has_episodes_elsewhere(tmp_path: Path) -> None:
    p = _item8(tmp_path)
    view = m39.LedgerView(VIEW.display, VIEW.id_map, {**VIEW.episodes_of, "person:korin-vahl": 4})
    _renames, folds, refused = m39.episode_renames(p["meta"], view)
    assert folds == {} and refused["Korin Vahl"] == "the variant id has episodes elsewhere"
    _apply(tmp_path, view=view)
    ids = {n["id"] for n in fx._r(p["kg"])["nodes"] if n["type"] == "Person"}
    assert "person:korin-vahl" in ids  # left; the voice's own spelling still respelled
    assert fx._r(p["seg"])[0]["speaker_label"] == "Korin Vael"


def test_a_fold_needs_two_full_names(tmp_path: Path, monkeypatch) -> None:
    """0025's predicate does not match a bare first name today; if it ever did, no fold."""
    p = _item8(tmp_path)
    meta = fx._r(p["meta"])
    meta["content"]["speakers"][2]["name"] = "Korin"
    fx._w(p["meta"], meta)
    kg = fx._r(p["kg"])
    kg["nodes"][-1]["properties"]["name"] = "Korin"
    fx._w(p["kg"], kg)
    monkeypatch.setattr(m39, "same_person", lambda a, b: a.split()[0] == b.split()[0])
    _renames, folds, refused = m39.episode_renames(p["meta"], VIEW)
    assert folds == {} and refused["Korin"] == "a fold needs two full names"


def test_0025_leaves_a_voice_the_ledger_spells(tmp_path: Path, monkeypatch) -> None:
    """The ledger owns a joined person's spelling across episodes: 0025 does not rename the voice
    toward this episode's stated spelling, or the two migrations would undo each other."""
    from podcast_scraper.upgrade.migrations import m0025_one_person_one_entry as m25

    p = fx._corpus(
        tmp_path,
        speakers=[
            fx._speaker("Odile Brand", "host", ["SPEAKER_00"], "known_hosts", "host"),
            fx._speaker("Korin Vael", "guest", ["SPEAKER_01"], "self_intro", "guest"),
        ],
        turns=[
            ("SPEAKER_00", "Odile Brand", "host", "Welcome, everyone."),
            ("SPEAKER_01", "Korin Vael", "guest", "Hi, I'm Korin Vael."),
        ],
        known_hosts=["Odile Brand"],
        participants=["Korin K. Vael"],
    )
    ep = m25._Episode(p["meta"], tmp_path)
    assert ep.decide() and ep.renames == {"Korin Vael": "Korin K. Vael"}  # no ledger: as before
    monkeypatch.setattr(
        "podcast_scraper.identity.display.person_display",
        lambda _root: PersonDisplay({"person:korin-vael": "Korin Vael"}),
    )
    ep = m25._Episode(p["meta"], tmp_path)
    ep.decide()
    assert "Korin Vael" not in ep.renames


def test_a_twin_of_the_new_spelling_folds_into_the_respelled_voice(tmp_path: Path) -> None:
    """The publisher's entry matches the group spelling, not the voice's old one: the fold still
    finds the voice's id, filed under the old spelling."""
    p = fx._corpus(
        tmp_path,
        speakers=[
            fx._speaker("Odin Lowe", "host", ["SPEAKER_00"], "self_intro", "host"),
            fx._speaker("Mara Tull", "guest", ["SPEAKER_01"], "llm", "guest"),
            fx._speaker("Odin Vo", "host", [], "feed_statement", "unplaced_1"),
        ],
        turns=[
            ("SPEAKER_00", "Odin Lowe", "host", "Welcome to the show."),
            ("SPEAKER_01", "Mara Tull", "guest", "Thanks."),
        ],
        known_hosts=["Odin Vo"],
    )
    view = m39.LedgerView(
        PersonDisplay({"person:odin-lowe": "Odin Vow", "person:odin-vow": "Odin Vow"}),
        {"person:odin-lowe": "person:odin-vow"},
        {"person:odin-lowe": 1, "person:odin-vow": 1},
    )
    renames, _folds, refused = m39.episode_renames(p["meta"], view)
    assert renames == {"Odin Lowe": "Odin Vow", "Odin Vo": "Odin Vow"} and refused == {}
    _apply(tmp_path, view=view)
    assert [s["name"] for s in fx._r(p["meta"])["content"]["speakers"]] == ["Odin Vow", "Mara Tull"]


NO_JOINS = m39.LedgerView(PersonDisplay({}), {}, {})


def _listed_twice(tmp_path: Path):
    """Item 8's shape: a voice under one spelling, the feed's stated host unplaced under another."""
    return fx._corpus(
        tmp_path,
        speakers=[
            fx._speaker("Korin Vahl", "host", ["SPEAKER_00"], "self_intro", "host"),
            fx._speaker("Odile Brand", "guest", ["SPEAKER_01"], "llm", "guest"),
            fx._speaker("Korin Vael", "host", [], "feed_statement", "unplaced_1"),
        ],
        turns=[
            ("SPEAKER_00", "Korin Vahl", "host", "Welcome to the show."),
            ("SPEAKER_01", "Odile Brand", "guest", "Thanks for having me."),
        ],
        known_hosts=["Korin Vael"],
    )


def test_0025s_pending_decision_is_carried_whole(tmp_path: Path) -> None:
    from podcast_scraper.upgrade.migrations import m0025_one_person_one_entry as m25

    p = _listed_twice(tmp_path)
    e25 = m25._Episode(p["meta"], tmp_path)
    assert e25.decide() and e25.renames == {"Korin Vahl": "Korin Vael"}  # what 0025 still wants
    res = _apply(tmp_path, view=NO_JOINS)
    assert res.details["refused"] == []
    names = [s["name"] for s in fx._r(p["meta"])["content"]["speakers"]]
    assert names == ["Korin Vael", "Odile Brand"]  # the twin merged by 0025's own record step
    assert fx._r(p["seg"])[0]["speaker_label"] == "Korin Vael"
    e25 = m25._Episode(p["meta"], tmp_path)
    e25.decide()
    assert not (e25.renames or e25.dropped)  # 0025 now finds nothing: its verify passes
    assert _apply(tmp_path, view=NO_JOINS).details["episodes"] == []


def test_a_target_only_ner_found_is_not_carried(tmp_path: Path) -> None:
    p = _listed_twice(tmp_path)
    diag = fx._r(p["diag"])
    diag["tried"] = {"known_hosts": [], "metadata_named": [], "detected_guests": ["Korin Vael"]}
    fx._w(p["diag"], diag)
    ep = m39._Episode(p["meta"], tmp_path, view=NO_JOINS)
    ep.decide()
    assert ep.carried == {}


def test_a_twin_0025_drops_without_a_rename_is_dropped(tmp_path: Path) -> None:
    from podcast_scraper.upgrade.migrations import m0025_one_person_one_entry as m25

    p = fx._corpus(
        tmp_path,
        speakers=[
            fx._speaker("Korin Vael", "host", ["SPEAKER_00"], "known_hosts", "host"),
            fx._speaker("Odile Brand", "guest", ["SPEAKER_01"], "llm", "guest"),
            fx._speaker("Korin K. Vael", "host", [], "feed_statement", "unplaced_1"),
        ],
        turns=[
            ("SPEAKER_00", "Korin Vael", "host", "Welcome to the show."),
            ("SPEAKER_01", "Odile Brand", "guest", "Thanks for having me."),
        ],
        known_hosts=["Korin Vael"],
    )
    e25 = m25._Episode(p["meta"], tmp_path)
    e25.decide()
    assert e25.dropped == ["Korin K. Vael"] and not e25.renames
    _apply(tmp_path, view=NO_JOINS)
    names = [s["name"] for s in fx._r(p["meta"])["content"]["speakers"]]
    assert names == ["Korin Vael", "Odile Brand"]


def test_a_listed_name_is_read_as_itself_even_when_another_listed_name_spells_to_it() -> None:
    """A list entry that renames a person from one of their spellings to another: both are in one
    ledger group. The stored old name must stay the old name (the before state)."""
    from podcast_scraper.upgrade.migrations import m0029_naming_arc_voices_repaired as m29

    spellings = {"Korin Vael": "Korin Vaal", "Korin Vaal": "Korin Vaal", "Korin Vahl": "Korin Vaal"}
    assert m29.as_listed("Korin Vaal", ("Korin Vael", "Korin Vaal"), spellings) == "Korin Vaal"
    assert m29.as_listed("Korin Vaal", ("Korin Vael", "Odile Brand"), spellings) == "Korin Vael"
    assert m29.as_listed("Mara Tull", ("Korin Vael", None), spellings) == "Mara Tull"


def test_0025_leaves_a_voice_its_cross_episode_identity_names(tmp_path: Path, monkeypatch) -> None:
    """The episode's metadata spells the guest one way; the voice carries its cross-episode voice
    group's name, confirmed by 0035's review (the person's other episodes' spelling). 0025 does
    not undo it."""
    from podcast_scraper.upgrade.migrations import m0025_one_person_one_entry as m25

    p = fx._corpus(
        tmp_path,
        speakers=[
            fx._speaker("Odile Brand", "host", ["SPEAKER_00"], "known_hosts", "host"),
            fx._speaker("Korin Vael", "guest", ["SPEAKER_01"], "llm_resolution", "guest"),
        ],
        turns=[
            ("SPEAKER_00", "Odile Brand", "host", "Welcome, everyone."),
            ("SPEAKER_01", "Korin Vael", "guest", "Thanks for having me."),
        ],
        known_hosts=["Odile Brand"],
        participants=["Korin K. Vael"],
    )
    ep = m25._Episode(p["meta"], tmp_path)
    assert ep.decide() and ep.renames == {"Korin Vael": "Korin K. Vael"}
    # An unreviewed fingerprint name is not deferred to: a wrong match stays open to correction.
    monkeypatch.setattr(m25, "_reviewed_identity_names", lambda: frozenset())
    ep = m25._Episode(p["meta"], tmp_path)
    ep.decide()
    assert ep.renames == {"Korin Vael": "Korin K. Vael"}
    monkeypatch.setattr(
        m25, "_reviewed_identity_names", lambda: frozenset({("ep", "SPEAKER_01", "Korin Vael")})
    )
    ep = m25._Episode(p["meta"], tmp_path)
    ep.decide()
    assert "Korin Vael" not in ep.renames


def test_an_episode_scoped_placeholder_is_never_respelled(tmp_path: Path) -> None:
    """A bare name scoped to its episode stays bare in the files: a group's full spelling written
    over it makes it a full name the ledger's join test then matches against other people. Read
    time still shows the group spelling."""
    p = _episode(tmp_path)
    kg = fx._r(p["kg"])
    kg["nodes"].append(
        {"id": "person:unresolved-vael-ep", "type": "Person", "properties": {"name": "Vael"}}
    )
    fx._w(p["kg"], kg)
    view = m39.LedgerView(
        PersonDisplay({**VIEW.display._spelling_of, "person:unresolved-vael-ep": "Korin Vael"}),
        {**VIEW.id_map, "person:unresolved-vael-ep": "person:korin-vael"},
        VIEW.episodes_of,
    )
    renames, _folds, refused = m39.episode_renames(p["meta"], view)
    assert "Vael" not in renames and refused["Vael"] == "an episode-scoped id"
    assert renames["Korin Vaal"] == "Korin Vael"


@pytest.mark.parametrize("pid", ["person:speaker-ep-1", "person:host"])
def test_an_unresolved_speaker_placeholder_is_never_respelled(tmp_path: Path, pid: str) -> None:
    p = _episode(tmp_path)
    kg = fx._r(p["kg"])
    kg["nodes"].append({"id": pid, "type": "Person", "properties": {"name": "Korin Vaal"}})
    fx._w(p["kg"], kg)
    view = m39.LedgerView(
        PersonDisplay({**VIEW.display._spelling_of, pid: "Korin Vael"}),
        {**VIEW.id_map, pid: "person:korin-vael"},
        VIEW.episodes_of,
    )
    renames, _folds, refused = m39.episode_renames(p["meta"], view)
    assert renames == {} and refused == {"Korin Vaal": "an episode-scoped id"}
