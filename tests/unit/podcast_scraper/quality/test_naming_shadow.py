"""The naming shadow report (C9): side namers, talk floor and rescues read from served sidecars.

Built on m0025's corpus fixture; every name is invented.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from podcast_scraper.quality import naming_shadow as ns

pytestmark = [pytest.mark.unit]

_spec = importlib.util.spec_from_file_location(
    "m0025_fixture",
    Path(__file__).resolve().parents[1] / "upgrade" / "test_m0025_one_person_one_entry.py",
)
assert _spec and _spec.loader
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)


def _episode(tmp_path: Path, extra: Dict[str, Any]) -> Dict[str, Any]:
    turns = [
        ("SPEAKER_00", "Arlo Venn", "host", "Welcome to the show."),
        ("SPEAKER_01", "Katharine Quist", "guest", "Thanks for having me."),
    ]
    p: Dict[str, Any] = fx._corpus(
        tmp_path,
        speakers=[
            fx._speaker("Arlo Venn", "host", ["SPEAKER_00"], "known_hosts", "host"),
            fx._speaker("Katharine Quist", "guest", ["SPEAKER_01"], "roster", "guest"),
        ],
        turns=turns,
        known_hosts=["Arlo Venn"],
    )
    diag = fx._r(p["diag"])
    diag.update(extra)
    p["diag"].write_text(json.dumps(diag), encoding="utf-8")
    return p


def _voice(ladder, result, outcome, key="model_result", model=None):
    e = {"ladder": {"name": ladder}, key: {"name": result, "role": "guest"}, "outcome": outcome}
    if model is not None:
        e["model"] = model
    if key == "fingerprint_result":
        # As the rung writes it: the voice's best group, similarity and talk.
        e["match"] = {"group": result or ladder, "similarity": 0.9, "talk_s": 300.0}
    return e


def test_an_episode_without_the_keys_counts_as_without_never_as_agreement(tmp_path: Path) -> None:
    _episode(tmp_path, {})
    rep = ns.report(tmp_path)
    assert rep["episodes"] == 1
    for n in ns.NAMERS:
        assert rep["namers"][n]["episodes"] == 0 and rep["namers"][n]["without"] == 1
        assert rep["namers"][n]["outcomes"] == {o: 0 for o in (*ns.OUTCOMES, ns.NOT_JUDGED)}
    assert rep["floor"]["removed"] == 0 and rep["rescues"]["fires"] == 0


def test_each_namers_outcomes_are_counted_and_every_difference_listed(tmp_path: Path) -> None:
    _episode(
        tmp_path,
        {
            "transcript_naming": {
                "ok": True,
                "voices": {
                    "SPEAKER_00": _voice("Arlo Venn", "Arlo Venn", "agree"),
                    "SPEAKER_01": _voice("Katharine Quist", "Kate Quist", "changes"),
                },
            },
            "voice_identity": {
                "voices": {
                    "SPEAKER_00": _voice("Arlo Venn", "Arlo Venn", "agree", "fingerprint_result"),
                    "SPEAKER_01": _voice(None, "Katharine Quist", "adds", "fingerprint_result"),
                }
            },
        },
    )
    rep = ns.report(tmp_path)
    tn, vi = rep["namers"]["transcript_naming"], rep["namers"]["voice_identity"]
    assert tn["answered"] == 1 and tn["outcomes"] == {
        "not_judged": 0,
        "agree": 1,
        "adds": 0,
        "removes": 0,
        "changes": 1,
    }
    assert [(r["voice"], r["ladder"], r["namer"]) for r in tn["differences"]] == [
        ("SPEAKER_01", "Katharine Quist", "Kate Quist")
    ]
    assert vi["outcomes"]["adds"] == 1 and vi["differences"][0]["namer"] == "Katharine Quist"
    text = ns.render(rep)
    assert "ladder 'Katharine Quist' -> transcript_naming 'Kate Quist'" in text


def test_a_failed_llm_call_is_an_episode_not_an_answer(tmp_path: Path) -> None:
    _episode(tmp_path, {"transcript_naming": {"ok": False, "voices": {"SPEAKER_00": {}}}})
    tn = ns.report(tmp_path)["namers"]["transcript_naming"]
    assert tn["episodes"] == 1 and tn["answered"] == 0 and sum(tn["outcomes"].values()) == 0


def test_the_floor_table_carries_the_llm_role_as_the_proxy(tmp_path: Path) -> None:
    _episode(
        tmp_path,
        {
            "decision_trace": {
                "voices": {
                    "SPEAKER_02": [
                        {"rung": "llm", "decision": "named", "name": "Dan Atkins"},
                        {
                            "rung": "talk_floor",
                            "decision": "removed",
                            "name": "Dan Atkins",
                            "talk_share": 0.012,
                        },
                    ]
                }
            },
            "transcript_naming": {
                "ok": True,
                "voices": {"SPEAKER_02": {"model": {"name": "Dan Atkins", "role": "ad"}}},
            },
        },
    )
    f = ns.report(tmp_path)["floor"]
    assert f["removed"] == 1 and f["by_llm_role"] == {"ad": 1}
    assert f["rows"][0]["name"] == "Dan Atkins" and f["rows"][0]["talk_share"] == 0.012


def test_a_floor_removal_without_an_llm_pass_has_no_role(tmp_path: Path) -> None:
    _episode(
        tmp_path,
        {
            "decision_trace": {
                "voices": {
                    "SPEAKER_02": [{"rung": "talk_floor", "decision": "removed", "name": "X Y"}]
                }
            }
        },
    )
    assert ns.report(tmp_path)["floor"]["by_llm_role"] == {ns.NO_LLM_ANSWER: 1}


def test_rescue_fires_are_listed_with_the_llm_name_beside_them(tmp_path: Path) -> None:
    p = _episode(tmp_path, {})
    diag = fx._r(p["diag"])
    for v in diag["voices"]:
        if v["voice"] == "SPEAKER_01":
            v.update(source="e1d_rescue", named=True, resolved_name="Katharine Quist")
    diag["transcript_naming"] = {
        "ok": True,
        "voices": {"SPEAKER_01": {"model": {"name": "Katharine Quist", "role": "guest"}}},
    }
    p["diag"].write_text(json.dumps(diag), encoding="utf-8")
    s = ns.report(tmp_path)["rescues"]
    assert s["fires"] == 1 and s["by_rule"] == {"e1d_rescue": 1} and s["llm_agrees"] == 1
    assert s["rows"][0]["name"] == "Katharine Quist"


def test_the_script_prints_and_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _episode(tmp_path, {})
    spec = importlib.util.spec_from_file_location(
        "naming_shadow_report",
        Path(__file__).resolve().parents[4] / "scripts" / "measure" / "naming_shadow_report.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "r.json"
    assert mod.main(["--corpus", str(tmp_path), "--json", str(out)]) == 0
    assert "served episodes with a speaker sidecar: 1" in capsys.readouterr().out
    assert json.loads(out.read_text())["episodes"] == 1


def test_an_abstained_voice_rung_is_not_an_answer_and_its_agree_rows_are_not_counted(
    tmp_path: Path,
) -> None:
    # The rung writes "agree" for every voice when it abstains: those rows only repeat the ladder.
    _episode(
        tmp_path,
        {
            "voice_identity": {
                "report": {"abstained": "no voted group in the feed's other episodes"},
                "voices": {
                    "SPEAKER_00": _voice("Arlo Venn", "Arlo Venn", "agree", "fingerprint_result")
                },
            }
        },
    )
    vi = ns.report(tmp_path)["namers"]["voice_identity"]
    assert vi["episodes"] == 1 and vi["answered"] == 0
    assert vi["not_answered"] == {"no voted group in the feed's other episodes": 1}
    assert sum(vi["outcomes"].values()) == 0
    assert "not answered:" in ns.render(ns.report(tmp_path))


def test_a_voice_the_llm_pass_was_never_asked_about_is_not_judged_not_removed(
    tmp_path: Path,
) -> None:
    _episode(
        tmp_path,
        {
            "transcript_naming": {
                "ok": True,
                "voices_asked": ["SPEAKER_00"],
                "voices": {
                    "SPEAKER_00": _voice("Arlo Venn", "Arlo Venn", "agree"),
                    "SPEAKER_01": _voice("Katharine Quist", None, "removes"),
                },
            }
        },
    )
    tn = ns.report(tmp_path)["namers"]["transcript_naming"]
    assert tn["outcomes"]["removes"] == 0 and tn["outcomes"]["not_judged"] == 1
    assert tn["differences"] == []


def test_voices_the_fingerprint_rung_cannot_judge_are_not_judged_not_agree(tmp_path: Path) -> None:
    # Second review m1: no fingerprint, or unnamed and under 10 s: decide() names neither.
    match = {"group": "Arlo Venn", "similarity": 0.2, "talk_s": 300.0}
    short = {"group": "Arlo Venn", "similarity": 0.2, "talk_s": 6.0}

    def entry(ladder, m):
        e = _voice(ladder, ladder, "agree", "fingerprint_result")
        e["match"] = m
        return e

    _episode(
        tmp_path,
        {
            "voice_identity": {
                "report": {"history_voices": 4, "groups": 1},
                "voices": {
                    "SPEAKER_00": entry("Arlo Venn", match),
                    "SPEAKER_01": entry(None, None),
                    "SPEAKER_02": entry(None, short),
                    "SPEAKER_03": entry("Kim Lee", short),
                },
            }
        },
    )
    vi = ns.report(tmp_path)["namers"]["voice_identity"]
    # SPEAKER_00 judged; SPEAKER_03 is named, so its removal was checked; 01 and 02 were not.
    assert vi["outcomes"]["agree"] == 2 and vi["outcomes"]["not_judged"] == 2


_gspec = importlib.util.spec_from_file_location(
    "gold_fixture", Path(__file__).resolve().parent / "test_gold_insights.py"
)
assert _gspec and _gspec.loader
gfx = importlib.util.module_from_spec(_gspec)
_gspec.loader.exec_module(gfx)


def test_insights_are_counted_per_voice_by_their_first_quote(tmp_path: Path) -> None:
    meta = gfx._episode(
        tmp_path,
        insights=[
            ("i:a", [1], {"speaker": "Stan McCormack"}),
            ("i:b", [1, 0], {"speaker": "Stan McCormack"}),
            ("i:c", [0], {"speaker": "Gerald Orban"}),
            ("i:none", [], {}),
        ],
    )
    counts = ns.insights_by_voice(tmp_path / meta)
    assert counts == {"SPEAKER_01": 2, "SPEAKER_00": 1}


def test_an_episode_without_gi_has_no_insight_count(tmp_path: Path) -> None:
    meta = gfx._episode(tmp_path, insights=[])
    (tmp_path / meta.replace(".metadata.json", ".gi.json")).unlink()
    assert ns.insights_by_voice(tmp_path / meta) is None


def test_listed_rows_carry_the_title_and_the_voices_insights(tmp_path: Path) -> None:
    _episode(
        tmp_path,
        {
            "transcript_naming": {
                "ok": True,
                "voices": {"SPEAKER_01": _voice("Katharine Quist", "Kate Quist", "changes")},
            }
        },
    )
    (r,) = ns.report(tmp_path)["namers"]["transcript_naming"]["differences"]
    assert "title" in r and "feed" in r and "insights" in r
    assert " | " in ns.render(ns.report(tmp_path))
