"""The loss ranking (INGESTION-ARC 2.2, 2.3): every class and sub-class on real-shaped temp corpora.

Built with the S7 tests' corpus and gold builders; every name is invented.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from podcast_scraper.quality import gold_insights as gi, loss_ranking as lr

pytestmark = [pytest.mark.unit]

_spec = importlib.util.spec_from_file_location(
    "s7_tests", Path(__file__).with_name("test_gold_insights.py")
)
assert _spec and _spec.loader
t7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t7)

REPO = Path(__file__).resolve().parents[4]
HOST, GUEST, THIRD = "Arlo Venn", "Katharine Quist", "Cyril Doane"


def _three(root: Path, insights, run: str = "run_1", speakers=None, talk=None) -> str:
    turns = [
        ("SPEAKER_00", HOST, "Welcome, Katharine."),
        ("SPEAKER_01", GUEST, "Thanks, Arlo."),
        ("SPEAKER_02", THIRD, "And hello from me."),
    ]
    meta = t7._episode(root, run=run, turns=turns, insights=insights)
    base = root / meta[: -len("metadata/ep.metadata.json")]
    if speakers is not None:
        m = json.loads((root / meta).read_text())
        m["content"]["speakers"] = speakers
        t7._w(root / meta, m)
    voices = [
        {"voice": v, "talk_time_s": s}
        for v, s in (talk or {"SPEAKER_00": 40, "SPEAKER_01": 40, "SPEAKER_02": 20}).items()
    ]
    t7._w(base / "transcripts" / "ep.speakers.diagnostics.json", {"voices": voices})
    return str(meta)


LABELS3 = [
    t7._lab("SPEAKER_00", HOST, "host"),
    t7._lab("SPEAKER_01", GUEST),
    t7._lab("SPEAKER_02", THIRD),
]


def _gold_rows(tmp_path: Path, insights, labels=LABELS3) -> List[Dict[str, Any]]:
    root, gold = tmp_path / "c", tmp_path / "gold"
    meta = _three(root, insights)
    t7._gold(gold, "g", "c1", meta, labels)
    report = gi.run(root, gold, ["g"], "snap")
    rows: List[Dict[str, Any]] = lr.gold_wrong_names(report, gold, root)["rows"]
    return rows


def _by(rows) -> Dict[str, Any]:
    return {r["insight"]: (r["cls"], r["sub"]) for r in rows}


def test_binding_sub_classes_follow_what_the_owner_publishes(tmp_path: Path) -> None:
    rows = _gold_rows(
        tmp_path,
        [
            ("i:g_as_host", [1], {"speaker": HOST}),  # guest's insight under the host's name
            ("i:h_as_guest", [0], {"speaker": GUEST}),  # and the host's under the guest's: swapped
        ],
    )
    assert _by(rows) == {
        "i:g_as_host": ("binding", "swapped"),
        "i:h_as_guest": ("binding", "swapped"),
    }


def test_a_name_on_two_voices_is_duplicated(tmp_path: Path) -> None:
    rows = _gold_rows(
        tmp_path,
        [("i:g_as_host", [1], {"speaker": HOST}), ("i:host_own", [0], {"speaker": HOST})],
    )
    assert _by(rows) == {"i:g_as_host": ("binding", "duplicated")}


@pytest.mark.parametrize(
    ("owner_insight", "sub"),
    [
        ([("i:host_third", [0], {"speaker": THIRD})], "rotated/third_name"),
        (
            [("i:host_raw", [0], {"speaker": "SPEAKER_00", "surfaceable": False})],
            "rotated/owner_unnamed",
        ),
        ([], "rotated/owner_no_insight"),
    ],
)
def test_rotated_sub_classes(tmp_path: Path, owner_insight, sub) -> None:
    """The owner (the host's voice) carries a third name, a raw label (unnamed), or no insight."""
    rows = _gold_rows(tmp_path, [("i:g_as_host", [1], {"speaker": HOST})] + owner_insight)
    assert _by(rows)["i:g_as_host"] == ("binding", sub)


def test_two_owners_are_order_independent(tmp_path: Path) -> None:
    """Two labelled voices share the stored name; the sub-class does not depend on their order."""
    labels = LABELS3[:2] + [t7._lab("SPEAKER_02", HOST)]
    rows = _gold_rows(
        tmp_path,
        [("i:g_as_host", [1], {"speaker": HOST}), ("i:third_own", [2], {"speaker": GUEST})],
        labels=labels,
    )
    [r] = [r for r in rows if r["insight"] == "i:g_as_host"]
    assert r["owners"] == ["SPEAKER_00", "SPEAKER_02"]
    rows2 = _gold_rows(
        tmp_path / "rev",
        [("i:g_as_host", [1], {"speaker": HOST}), ("i:third_own", [2], {"speaker": GUEST})],
        labels=[labels[2], labels[1], labels[0]],
    )
    assert _by(rows2)["i:g_as_host"] == _by(rows)["i:g_as_host"] == ("binding", "swapped")


@pytest.mark.parametrize(
    ("label", "stored", "spelling"),
    [
        ("Mateo Marsh", "Matteo", True),  # a first-name variant the matcher accepts
        ("Ivo Marsh", "Ivo Marsh", True),
        ("Dr. Ivo Marsh", "Ivo Marsh", True),  # a title dropped
        ("Ivo Marsh", "Ivo Marchetti", False),  # a surname the matcher rejects
        ("Ivo Marsh", "Hey Ivo", False),  # a greeting fused to the first name
        ("Ivo Marsh", "Ivo Marsh Junior Three", False),  # more tokens than the label
        ("Ivo Marsh", "Ola", False),
    ],
)
def test_spelling_boundary(label: str, stored: str, spelling: bool) -> None:
    assert lr.is_spelling(label, stored) is spelling


def test_spelling_and_other_person_and_their_flags(tmp_path: Path) -> None:
    rows = _gold_rows(
        tmp_path,
        [
            ("i:variant", [1], {"speaker": "Katherine"}),  # a variant the matcher accepts
            ("i:greeting", [1], {"speaker": "Hey Katharine"}),
            ("i:stranger", [1], {"speaker": "Ola Fenn"}),
        ],
    )
    by = {r["insight"]: r for r in rows}
    assert by["i:variant"]["cls"] == "spelling"
    assert by["i:greeting"]["cls"] == "other_person" and by["i:greeting"]["shares_label_token"]
    assert by["i:stranger"]["cls"] == "other_person" and not by["i:stranger"]["shares_label_token"]


def test_wrong_names_outside_s7_wrong_are_counted_beside_it(tmp_path: Path) -> None:
    root, gold = tmp_path / "c", tmp_path / "gold"
    meta = _three(root, [("i:nameless", [2], {"speaker": THIRD})])
    t7._gold(gold, "g", "c1", meta, LABELS3[:2] + [t7._lab("SPEAKER_02", None)])
    out = lr.gold_wrong_names(gi.run(root, gold, ["g"], "snap"), gold, root)
    assert out["rows"] == [] and out["outside_s7_wrong"] == {"name_without_support": 1}


def test_the_classes_must_be_exactly_s7s_wrong_ids(tmp_path: Path) -> None:
    """Conservation: a wrong row counted twice (or lost) is refused, not reported."""
    root, gold = tmp_path / "c", tmp_path / "gold"
    meta = _three(root, [("i:g_as_host", [1], {"speaker": HOST})])
    t7._gold(gold, "g", "c1", meta, LABELS3)
    report = gi.run(root, gold, ["g"], "snap")
    [wrong] = [r for r in report["rows"] if r["bucket"] == "S7_wrong"]
    with pytest.raises(AssertionError, match="exactly S7"):
        lr.gold_wrong_names({**report, "rows": report["rows"] + [wrong]}, gold, root)


def _speaker(
    name: str, role: str, voices: List[str], placed: Optional[bool] = True
) -> Dict[str, Any]:
    e: Dict[str, Any] = {"name": name, "role": role, "voices": voices}
    if placed is not None:
        e["placed"] = placed
    return e


def _hidden(props: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"surfaceable": False, "speaker_voice_type": "unknown", **(props or {})}


def _corpus_rows(root: Path) -> Dict[str, Any]:
    out = lr.corpus_hidden(root)
    return {r["insight"]: r for r in out["rows"]}


def test_corpus_classes_by_the_record(tmp_path: Path) -> None:
    unplaced = [_speaker(HOST, "host", ["SPEAKER_00"]), _speaker(GUEST, "guest", [], placed=False)]
    _three(
        tmp_path,
        [("i:prin", [1], _hidden()), ("i:minor", [2], _hidden())],
        run="run_a",
        speakers=unplaced,
    )
    rows = _corpus_rows(tmp_path)
    assert rows["i:prin"]["cls"] == "unplaced_principal"
    assert rows["i:minor"]["cls"] == "unplaced_minor"
    # Two unnamed principal voices? No: SPEAKER_00 is placed, SPEAKER_01 is the one unnamed
    # principal voice (40 of 100), and the insight is on it.
    assert rows["i:prin"]["one_to_one"] is True and rows["i:minor"]["one_to_one"] is False


def test_a_record_without_placed_keys_cannot_say_who_is_unplaced(tmp_path: Path) -> None:
    speakers = [
        _speaker(HOST, "host", ["SPEAKER_00"], placed=None),
        _speaker(GUEST, "guest", [], None),
    ]
    _three(tmp_path, [("i:p", [1], _hidden())], speakers=speakers)
    assert _corpus_rows(tmp_path)["i:p"]["cls"] == "placed_not_recorded_principal"


def test_nobody_named_and_none_unplaced(tmp_path: Path) -> None:
    _three(tmp_path / "a", [("i:n", [1], _hidden())], speakers=[])
    assert _corpus_rows(tmp_path / "a")["i:n"]["cls"] == "nobody_named"
    _three(
        tmp_path / "b", [("i:p", [1], _hidden())], speakers=[_speaker(HOST, "host", ["SPEAKER_00"])]
    )
    assert _corpus_rows(tmp_path / "b")["i:p"]["cls"] == "none_unplaced_principal"


def test_share_exactly_at_the_line_is_principal(tmp_path: Path) -> None:
    speakers = [_speaker(HOST, "host", ["SPEAKER_00"]), _speaker(GUEST, "guest", [], placed=False)]
    _three(
        tmp_path,
        [("i:p", [1], _hidden())],
        speakers=speakers,
        talk={"SPEAKER_00": 40, "SPEAKER_01": 30, "SPEAKER_02": 30},
    )
    row = _corpus_rows(tmp_path)["i:p"]
    assert row["principal"] is True and row["cls"] == "unplaced_principal"


def test_one_to_one_needs_the_insight_on_the_unnamed_voice(tmp_path: Path) -> None:
    speakers = [_speaker(HOST, "host", ["SPEAKER_00"]), _speaker(GUEST, "guest", [], placed=False)]
    _three(tmp_path, [("i:on_host", [0], _hidden())], speakers=speakers)
    row = _corpus_rows(tmp_path)["i:on_host"]
    assert row["cls"] == "unplaced_principal" and row["one_to_one"] is False


def test_gaps_are_counted_apart(tmp_path: Path) -> None:
    speakers = [_speaker(HOST, "host", ["SPEAKER_00"]), _speaker(GUEST, "guest", [], placed=False)]
    meta = _three(tmp_path, [("i:noq", [], _hidden())], speakers=speakers)
    assert _corpus_rows(tmp_path)["i:noq"]["cls"] == "gap/voice_not_found"
    (
        tmp_path
        / meta[: -len("metadata/ep.metadata.json")]
        / "transcripts"
        / "ep.speakers.diagnostics.json"
    ).unlink()
    assert _corpus_rows(tmp_path)["i:noq"]["cls"] == "gap/no_diagnostics"


def test_the_seat_check_scores_the_one_to_one_seat_against_the_label(tmp_path: Path) -> None:
    root, gold = tmp_path / "c", tmp_path / "gold"
    speakers = [_speaker(HOST, "host", ["SPEAKER_00"]), _speaker(GUEST, "guest", [], placed=False)]
    meta = _three(root, [("i:p", [1], _hidden())], speakers=speakers)
    t7._gold(gold, "g", "c1", meta, LABELS3)
    t7._gold(gold, "g", "c2", "feeds/f/gone/metadata/ep.metadata.json", LABELS3)  # not served
    served = {meta}
    out = lr.seat_check(root, gold, {"g": "labels_v1"}, served)
    assert [(r["case"], r["verdict"]) for r in out["rows"]] == [("c1", "right")]
    t7._gold(gold, "g", "c1", meta, [LABELS3[0], t7._lab("SPEAKER_01", "Ola Fenn"), LABELS3[2]])
    assert (
        lr.seat_check(root, gold, {"g": "labels_v1"}, served)["rows"][0]["verdict"]
        == "another_person"
    )


def test_it_writes_nothing_under_the_corpus_or_gold(tmp_path: Path) -> None:
    root, gold = tmp_path / "c", tmp_path / "gold"
    speakers = [_speaker(HOST, "host", ["SPEAKER_00"]), _speaker(GUEST, "guest", [], placed=False)]
    meta = _three(
        root, [("i:p", [1], _hidden()), ("i:w", [1], {"speaker": HOST})], speakers=speakers
    )
    t7._gold(gold, "g", "c1", meta, LABELS3)

    def tree() -> Dict[str, str]:
        out = {}
        for base in (root, gold):
            for dirpath, _dirs, files in os.walk(base):
                for f in files:
                    p = Path(dirpath) / f
                    out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        return out

    before = tree()
    lr.gold_wrong_names(gi.run(root, gold, ["g"], "snap"), gold, root)
    lr.corpus_hidden(root)
    lr.seat_check(root, gold, {"g": "labels_v1"}, {meta})
    assert tree() == before


def test_the_cli_refuses_an_out_inside_the_repo(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "lr_cli", REPO / "scripts/measure/loss_ranking.py"
    )
    assert spec and spec.loader
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    code = cli.main(
        [
            "--corpus",
            str(tmp_path),
            "--gold",
            str(tmp_path),
            "--snapshot",
            "s",
            "--out",
            str(REPO / "x"),
        ]
    )
    assert code == 2 and not (REPO / "x").exists()


def test_a_name_that_matches_another_voice_only_fuzzily_is_not_binding(tmp_path: Path) -> None:
    """Binding is fold-exact; what ``same_person`` would add is a sensitivity count only."""
    rows = _gold_rows(tmp_path, [("i:near", [1], {"speaker": "Arlo Vennn"})])
    [r] = rows
    assert r["cls"] != "binding" and r["same_person_owner_sensitivity"] is True


def test_an_owner_publishing_both_names_is_swapped_not_duplicated(tmp_path: Path) -> None:
    rows = _gold_rows(
        tmp_path,
        [
            ("i:g_as_host", [1], {"speaker": HOST}),
            ("i:host_as_guest", [0], {"speaker": GUEST}),
            ("i:host_own", [0], {"speaker": HOST}),
        ],
    )
    assert _by(rows)["i:g_as_host"] == ("binding", "swapped")


def test_a_repeated_token_cannot_pair_twice() -> None:
    assert lr.is_spelling("Mateo Marsh", "Mateo Mateo") is False


def test_binding_takes_precedence_over_spelling(tmp_path: Path) -> None:
    """The guest is labelled "Mateo Marsh", the host "Matteo": "Matteo" on the guest's voice is
    the host's name (binding), though token by token it is also a spelling of the guest's."""
    labels = [t7._lab("SPEAKER_00", "Matteo", "host"), t7._lab("SPEAKER_01", "Mateo Marsh")]
    assert lr.is_spelling("Mateo Marsh", "Matteo")
    rows = _gold_rows(tmp_path, [("i:g", [1], {"speaker": "Matteo"})], labels=labels + [LABELS3[2]])
    assert _by(rows)["i:g"][0] == "binding"


def test_corpus_conservation_is_asserted(tmp_path: Path, monkeypatch) -> None:
    speakers = [_speaker(HOST, "host", ["SPEAKER_00"]), _speaker(GUEST, "guest", [], placed=False)]
    _three(tmp_path, [("i:p", [1], _hidden())], speakers=speakers)
    real = lr.cs.score_corpus

    def off_by_one(root: Path) -> Dict[str, Any]:
        out = real(root)
        out["total"] = {**out["total"], "S1_hidden_unknown": out["total"]["S1_hidden_unknown"] + 1}
        return out

    monkeypatch.setattr(lr.cs, "score_corpus", off_by_one)
    with pytest.raises(AssertionError, match="the scoreboard says"):
        lr.corpus_hidden(tmp_path)


def test_the_seat_check_skips_a_case_on_disk_that_is_not_served(tmp_path: Path) -> None:
    root, gold = tmp_path / "c", tmp_path / "gold"
    speakers = [_speaker(HOST, "host", ["SPEAKER_00"]), _speaker(GUEST, "guest", [], placed=False)]
    meta = _three(root, [("i:p", [1], _hidden())], speakers=speakers)
    t7._gold(gold, "g", "c1", meta, LABELS3)
    assert lr.seat_check(root, gold, {"g": "labels_v1"}, served=set())["rows"] == []
