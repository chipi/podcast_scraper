"""``speaker_naming_passes``: each naming LLM call asked k times, a name kept only when all agree.

Covers the ladder's resolution call (``_resolve_voices_k_passes``) and transcript naming
(``transcript_naming.agree``). Synthetic fixtures only.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest

from podcast_scraper.providers.ml.diarization import pipeline as P, transcript_naming as tn

pytestmark = pytest.mark.unit


def _runs(
    monkeypatch: pytest.MonkeyPatch, answers: List[Tuple[Dict[str, str], Dict[str, str]]]
) -> List[int]:
    calls: List[int] = []
    it: Iterator[Tuple[Dict[str, str], Dict[str, str]]] = iter(answers)

    def fake(cfg: Any, report: Optional[Dict[str, Any]] = None, **kw: Any) -> Any:
        calls.append(1)
        if report is not None:
            report["verdicts"] = [{"n": len(calls)}]
        return next(it)

    monkeypatch.setattr(P, "_resolve_voices_via_llm", fake)
    return calls


def test_one_pass_is_one_call_exactly_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _runs(monkeypatch, [({"SPEAKER_00": "Ann Lee"}, {"SPEAKER_00": "host"})])
    report: Dict[str, Any] = {}
    out = P._resolve_voices_k_passes(SimpleNamespace(speaker_naming_passes=1), report=report)
    assert out == ({"SPEAKER_00": "Ann Lee"}, {"SPEAKER_00": "host"})
    assert len(calls) == 1 and "passes" not in report


def test_more_passes_keep_only_the_names_and_roles_every_pass_gave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _runs(
        monkeypatch,
        [
            (
                {"SPEAKER_00": "Ann Lee", "SPEAKER_01": "Bo Diaz"},
                {"SPEAKER_00": "host", "SPEAKER_01": "guest"},
            ),
            (
                {"SPEAKER_00": "ann lee", "SPEAKER_01": "Cy Ross"},
                {"SPEAKER_00": "host", "SPEAKER_01": "guest"},
            ),
            (
                {"SPEAKER_00": "Ann Lee", "SPEAKER_01": "Bo Diaz"},
                {"SPEAKER_00": "guest", "SPEAKER_01": "guest"},
            ),
        ],
    )
    report: Dict[str, Any] = {}
    names, roles = P._resolve_voices_k_passes(
        SimpleNamespace(speaker_naming_passes=3), report=report
    )
    assert len(calls) == 3
    assert names == {"SPEAKER_00": "Ann Lee"}  # SPEAKER_01's name differed in pass 2
    assert roles == {"SPEAKER_01": "guest"}  # SPEAKER_00's role differed in pass 3
    assert report["verdicts"] == [{"n": 1}] and len(report["passes"]) == 2
    assert report["passes_agreed"]["dropped"] == ["SPEAKER_01"]


def test_a_bad_setting_means_one_pass() -> None:
    assert P.naming_passes(SimpleNamespace()) == 1
    assert P.naming_passes(SimpleNamespace(speaker_naming_passes=0)) == 1
    assert P.naming_passes(SimpleNamespace(speaker_naming_passes="x")) == 1


def _ask(
    answer: Optional[Dict[str, Tuple[Optional[str], str]]], error: str = ""
) -> tn.NamingAnswer:
    full = (
        {
            v: {"name": n, "role": r, "confidence": "high", "evidence": ""}
            for v, (n, r) in answer.items()
        }
        if answer is not None
        else None
    )
    report = {"error": error} if error else {"raw": "x"}
    return tn.NamingAnswer("standalone", sorted(answer or {}), full, report)


def test_agree_keeps_one_ask_as_it_is() -> None:
    a = _ask({"SPEAKER_00": ("Ann Lee", "host")})
    assert tn.agree([a]) is a


def test_agree_nulls_a_voice_the_passes_disagree_on() -> None:
    out = tn.agree(
        [
            _ask({"SPEAKER_00": ("Ann Lee", "host"), "SPEAKER_01": ("Bo Diaz", "guest")}),
            _ask({"SPEAKER_00": ("ann lee", "host"), "SPEAKER_01": ("Cy Ross", "guest")}),
        ]
    )
    assert out.ok
    assert (out.answer or {})["SPEAKER_00"]["name"] == "Ann Lee"
    assert (out.answer or {})["SPEAKER_01"]["name"] is None
    assert (out.answer or {})["SPEAKER_01"]["role"] == "guest"
    assert out.report["passes_agreed"] == {"asked": 2, "answered": 2, "dropped": ["SPEAKER_01"]}


def test_agree_leaves_out_a_failed_pass_and_reports_it() -> None:
    out = tn.agree(
        [_ask(None, "RuntimeError: gateway down"), _ask({"SPEAKER_00": ("Ann Lee", "host")})]
    )
    assert out.ok and (out.answer or {})["SPEAKER_00"]["name"] == "Ann Lee"
    assert out.report["passes_agreed"]["answered"] == 1
    none = tn.agree([_ask(None, "e1"), _ask(None, "e2")])
    assert not none.ok and none.report["error"] == "e1"


def test_one_pass_by_default() -> None:
    from podcast_scraper import config

    assert config.Config.model_fields["speaker_naming_passes"].default == 1
