"""pipeline-check (#2287): the recorder sees every keyed choice; the comparator fails when it must.

A check that has only ever passed proves nothing, so most of these build the failure the tool
exists to catch and assert it is reported — a lookup by a spelling the map does not hold, two
variants that drift apart, a stage that no longer matches the base ref.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "validate"))

from pipeline_check import compare, recorder, report  # noqa: E402

pytestmark = pytest.mark.unit


# --- recorder ------------------------------------------------------------------------------------


@pytest.fixture
def fake_package(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """A module with a language-keyed map, a non-keyed dict, and a function taking `language`."""
    mod = types.ModuleType("pc_fake_naming")
    mod.__file__ = "/tmp/pc_fake_naming.py"
    mod.WORDS_BY_LANGUAGE = {"en": ["host"], "es": ["presentador"]}
    mod.UNRELATED = {"alpha": 1, "beta": 2}

    def pick(language: Optional[str] = None) -> List[str]:
        return list(mod.WORDS_BY_LANGUAGE.get(language or "en", []))

    pick.__module__ = "pc_fake_naming"
    mod.pick = pick
    user = types.ModuleType("pc_fake_naming_user")
    user.pick = pick  # a `from pc_fake_naming import pick` copy
    monkeypatch.setitem(sys.modules, "pc_fake_naming", mod)
    monkeypatch.setitem(sys.modules, "pc_fake_naming_user", user)
    monkeypatch.setattr(recorder, "RECORDER", recorder.Recorder())
    return mod


def test_a_language_keyed_map_is_discovered_and_others_are_not(fake_package) -> None:
    rec = recorder.install(["pc_fake_naming"])
    assert rec.maps == ["pc_fake_naming.WORDS_BY_LANGUAGE"]
    assert isinstance(fake_package.WORDS_BY_LANGUAGE, recorder.RecordingMap)
    assert not isinstance(fake_package.UNRELATED, recorder.RecordingMap)


def test_every_lookup_is_logged_with_its_key_and_whether_it_hit(fake_package) -> None:
    rec = recorder.install(["pc_fake_naming"])
    fake_package.WORDS_BY_LANGUAGE.get("en")
    fake_package.WORDS_BY_LANGUAGE.get("en_gb")
    maps = [(e.key, e.hit) for e in rec.events if e.kind == "map"]
    assert maps == [("en", True), ("en_gb", False)]


def test_a_dimension_function_is_wrapped_everywhere_it_was_imported(fake_package) -> None:
    rec = recorder.install(["pc_fake_naming"])
    sys.modules["pc_fake_naming_user"].pick(language="es")
    calls = [e for e in rec.events if e.kind == "call"]
    assert [(c.site, c.key) for c in calls] == [("pc_fake_naming.pick", {"language": "es"})]
    assert calls[0].result == ["presentador"]


def test_iteration_is_not_a_decision(fake_package) -> None:
    """Reading every row (building one regex over all languages) is not a choice between them."""
    rec = recorder.install(["pc_fake_naming"])
    list(fake_package.WORDS_BY_LANGUAGE.items())
    assert rec.events == []


# --- comparator ----------------------------------------------------------------------------------


def _worker(
    results: Dict[str, Dict[str, Any]], events: Sequence[Dict[str, Any]] = ()
) -> Dict[str, Any]:
    episodes = sorted(next(iter(results.values())))
    return {
        "episodes": episodes,
        "variants": list(results),
        "results": results,
        "events": list(events),
        "normalised": {"en": "en", "en-US": "en", "English": "en", "es": "es"},
        "decision_points": {"maps": ["m.WORDS_BY_LANGUAGE"], "functions": ["m.pick"]},
        "import_time_copies": [],
        "skipped_modules": {},
    }


def _ev(kind: str, key: Any, hit: Optional[bool] = True, variant: str = "en") -> Dict[str, Any]:
    return {
        "kind": kind,
        "site": "m.WORDS_BY_LANGUAGE" if kind == "map" else "m.pick",
        "key": key,
        "hit": hit,
        "result": None,
        "caller": "m.py:1",
        "variant": variant,
        "stage": "hosts",
    }


_REC = {"hosts": {"hosts": ["Jane Doe"]}, "language": {"language": "en", "raw": "en"}}


def test_a_lookup_by_a_spelling_the_map_does_not_hold_is_a_hole() -> None:
    cand = _worker({"en": {"e1": _REC}}, [_ev("map", "en"), _ev("map", "en_gb", hit=False)])
    res = compare.evaluate("c", {"decisions_resolve_to": "en"}, cand, None)
    assert [f.kind for f in res.findings] == ["hole"]
    assert "en_gb" in res.findings[0].detail


def test_a_language_argument_that_normalises_to_the_target_is_fine() -> None:
    cand = _worker(
        {"en": {"e1": _REC}},
        [_ev("call", {"language": "English"}), _ev("call", {"language": None})],
    )
    assert compare.evaluate("c", {"decisions_resolve_to": "en"}, cand, None).passed


def test_a_language_argument_for_another_language_is_a_hole() -> None:
    cand = _worker({"en": {"e1": _REC}}, [_ev("call", {"language": "es"})])
    assert not compare.evaluate("c", {"decisions_resolve_to": "en"}, cand, None).passed


def test_variants_that_must_be_identical_but_differ_fail() -> None:
    other = {"hosts": {"hosts": []}, "language": {"language": "en", "raw": "en-US"}}
    cand = _worker({"en": {"e1": _REC}, "en-US": {"e1": other}})
    res = compare.evaluate(
        "c", {"variants_identical": True, "variants_identical_ignore": ["language.raw"]}, cand, None
    )
    assert [(f.kind, f.detail.split(":")[0]) for f in res.findings] == [("variant", "hosts")]


def test_the_ignored_field_alone_does_not_make_variants_differ() -> None:
    other = {"hosts": {"hosts": ["Jane Doe"]}, "language": {"language": "en", "raw": "en-US"}}
    cand = _worker({"en": {"e1": _REC}, "en-US": {"e1": other}})
    check = {"variants_identical": True, "variants_identical_ignore": ["language.raw"]}
    assert compare.evaluate("c", check, cand, None).passed


def test_a_stage_that_no_longer_matches_the_base_fails() -> None:
    cand = _worker({"en": {"e1": _REC}})
    base = _worker({"en": {"e1": {"hosts": {"hosts": ["Someone Else"]}}}})
    res = compare.evaluate("c", {"base_identical_stages": ["hosts"]}, cand, base)
    assert [f.kind for f in res.findings] == ["base"]
    assert res.stage_status["Host detection"] == "1 of 1 differ from base"


def test_a_pinned_value_that_does_not_hold_fails() -> None:
    cand = _worker({"en": {"e1": _REC}})
    res = compare.evaluate("c", {"expect_values": {"language.language": "es"}}, cand, None)
    assert [f.kind for f in res.findings] == ["value"]


def test_a_report_only_check_never_fails_and_reports_the_differences() -> None:
    other = {"hosts": {"hosts": []}, "language": {"language": "pt"}}
    cand = _worker({"pt-BR": {"e1": _REC}, "pt-PT": {"e1": other}})
    res = compare.evaluate("c", {"report_only": True, "variants_identical": True}, cand, None)
    assert res.passed
    assert res.report_only == {"pt-BR vs pt-PT": ["e1:hosts"]}


def test_coverage_names_the_decision_points_never_exercised() -> None:
    cand = _worker({"en": {"e1": _REC}}, [_ev("map", "en")])
    res = compare.evaluate("c", {}, cand, None)
    assert res.coverage["not_exercised"] == ["m.pick"]


def test_the_report_leads_with_the_verdict() -> None:
    cand = _worker({"en": {"e1": _REC}}, [_ev("map", "en_gb", hit=False)])
    res = compare.evaluate("c", {"decisions_resolve_to": "en"}, cand, None)
    page = report.render([res], base_label="main", candidate_label="HEAD")
    assert page.splitlines()[2].startswith("**Verdict: FAIL**")
    assert "en_gb" in page
