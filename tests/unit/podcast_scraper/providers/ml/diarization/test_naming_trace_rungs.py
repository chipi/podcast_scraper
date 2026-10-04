"""Every rung of the naming ladder leaves the trace step that explains it (#2276).

Most scenarios are REUSED, not invented: each row loads an existing roster test, runs it with a
trace injected (so the test's own assertion on the outcome still runs), and checks the trace
credits that outcome to the right rung. A rung no existing test reaches gets a synthetic scenario
here. All fixtures are synthetic (never-commit-real-episodes).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

from podcast_scraper.providers.ml.diarization import pipeline as P, roster as roster_mod
from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.naming_trace import NamingTrace

pytestmark = pytest.mark.unit

_TESTS = Path(__file__).resolve().parents[3]  # tests/unit/podcast_scraper


def _load(rel: str) -> ModuleType:
    path = _TESTS / rel
    spec = importlib.util.spec_from_file_location(f"_reused_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _traced(rel: str, test: str, *args: Any, cls: Optional[str] = None) -> Dict[str, Any]:
    """Run an existing roster test with a trace injected; return the one trace it produced."""
    mod = _load(rel)
    traces: List[NamingTrace] = []
    real = roster_mod.resolve_speaker_roster

    def traced(*a: Any, **kw: Any) -> Any:
        traces.append(NamingTrace())
        return real(*a, **{**kw, "trace": traces[-1]})

    mod.resolve_speaker_roster = traced  # type: ignore[attr-defined]
    fn: Callable[..., None] = getattr(getattr(mod, cls)(), test) if cls else getattr(mod, test)
    fn(*args)
    assert len(traces) == 1, f"{test} built {len(traces)} rosters"
    return traces[0].to_dict()


def _has(steps: List[Dict[str, Any]], want: Dict[str, Any]) -> bool:
    return any(all(s.get(k) == v for k, v in want.items()) for s in steps)


_DZ = "providers/ml/diarization/"

# (existing test file, test, args, where, key, expected step)
REUSED: List[Tuple[str, str, Tuple[Any, ...], str, str, Dict[str, Any]]] = [
    (
        _DZ + "test_seat_logic_v4.py",
        "test_cohost_formula_is_name_bearing_not_positional",
        (),
        "voices",
        "SPEAKER_00",
        {"rung": "self_intro", "decision": "named", "name": "Tobias Wren"},
    ),
    (
        _DZ + "test_seat_logic_v4.py",
        "test_cohost_formula_is_name_bearing_not_positional",
        (),
        "voices",
        "SPEAKER_01",
        {"rung": "cohost_formula", "decision": "named", "name": "Greta Holm"},
    ),
    (
        _DZ + "test_publisher_labels_outrank_inference.py",
        "test_a_bare_cluster_label_never_names_anybody",
        ("SPEAKER_01",),
        "voices",
        "SPEAKER_00",
        {"rung": "publisher_label", "decision": "stated", "name": "Maya"},
    ),
    (
        _DZ + "test_publisher_labels_outrank_inference.py",
        "test_a_bare_cluster_label_never_names_anybody",
        ("SPEAKER_01",),
        "voices",
        "SPEAKER_00",
        {"rung": "recover_stated_names", "decision": "changed", "name": "Maya Koster"},
    ),
    (
        _DZ + "test_presenter_evidence.py",
        "test_a_pool_entry_that_is_the_show_name_is_not_a_person_and_the_presenter_is_seated",
        (),
        "names",
        "Africa Tech Summit",
        {"rung": "host_pool", "decision": "never_names", "reason": "names_the_show"},
    ),
    (
        _DZ + "test_presenter_evidence.py",
        "test_host_introducing_the_guest_keeps_the_seat_despite_a_bled_guest_phrase",
        (),
        "voices",
        "HOST",
        {
            "rung": "recover_stated_names",
            "decision": "changed",
            "name": "Dov Pell",
            "previous": {"name": "Dov"},
        },
    ),
    (
        _DZ + "test_presenter_evidence.py",
        "test_an_evidence_host_takes_the_episodes_stated_spelling_of_the_name",
        (),
        "voices",
        "H1",
        {
            "rung": "stated_spelling_snap",
            "decision": "changed",
            "name": "Imani Moise",
            "previous": {"name": "Imani Moiz"},
        },
    ),
    (
        _DZ + "test_presenter_evidence.py",
        "test_an_evidence_host_takes_the_episodes_stated_spelling_of_the_name",
        (),
        "voices",
        "H2",
        {
            "rung": "one_name_per_person",
            "decision": "changed",
            "role": "host",
            "previous": {"role": "guest"},
        },
    ),
    (
        _DZ + "test_roster_ad_voices.py",
        "test_an_ad_narrator_is_never_named_even_though_it_says_its_own_name",
        (),
        "voices",
        "SPEAKER_AD1",
        {"rung": "ad_voice_placeholder", "decision": "set", "named": False},
    ),
    (
        _DZ + "test_roster_ad_voices.py",
        "test_the_final_gate_demotes_an_opener_laden_name_reaching_the_roster",
        (),
        "voices",
        "SPEAKER_GUEST",
        {"rung": "publish_gate", "decision": "refused", "name": "But Sun"},
    ),
    (
        _DZ + "test_host_not_read_as_guest.py",
        "test_host_posing_a_hypothetical_is_not_named_after_it",
        (),
        "voices",
        "SPEAKER_01",
        {"rung": "intro_reader", "decision": "named", "name": "Maria Lindqvist"},
    ),
    (
        _DZ + "test_host_not_read_as_guest.py",
        "test_host_posing_a_hypothetical_is_not_named_after_it",
        (),
        "names",
        "Maria Lindqvist",
        {"rung": "host_introduction_harvest", "decision": "added"},
    ),
    (
        _DZ + "test_host_not_read_as_guest.py",
        "test_host_posing_a_hypothetical_is_not_named_after_it",
        (),
        "names",
        "Maria Lindqvist",
        {"rung": "guest_pool", "decision": "excluded", "reason": "already_named_on_a_voice"},
    ),
    (
        _DZ + "test_two_voice_host_introduced_guest.py",
        "test_the_guest_the_host_introduces_is_named",
        ("With me today is Maria Lindqvist, a historian of medieval trade.",),
        "names",
        "Maria Lindqvist",
        {"rung": "two_voice_interview", "decision": "added"},
    ),
    (
        "diarization/test_midroll_ad_is_a_recording.py",
        "test_the_roster_types_the_midroll_ad_as_COMMERCIAL",
        (),
        "voices",
        "SPEAKER_09",
        {
            "rung": "voice_types",
            "decision": "typed",
            "voice_type": "commercial",
            "reason": "edge_ad",
        },
    ),
]


@pytest.mark.parametrize(
    "rel,test,args,where,key,step", REUSED, ids=[f"{r[4]}:{r[5]['rung']}" for r in REUSED]
)
def test_an_existing_scenario_is_explained_by_its_rung(rel, test, args, where, key, step) -> None:
    d = _traced(rel, test, *args)
    steps = d[where].get(key, [])
    assert _has(steps, step), (step, steps)


def test_a_guest_host_the_episode_names_is_recorded_as_joining_the_pool() -> None:
    d = _traced(
        _DZ + "test_presenter_evidence.py",
        "test_a_guest_host_the_episode_names_is_seated_over_the_feed_hosts",
    )
    assert {"rung": "guest_host_pool", "added": ["Max Read"]} in d["episode"]


def test_one_person_two_spellings_records_the_loser() -> None:
    d = _traced(
        _DZ + "test_one_person_one_name.py",
        "test_two_spellings_of_one_guest_publish_as_one_name",
        cls="TestTheResolvedRosterCarriesOneName",
    )
    assert _has(
        d["voices"]["SPEAKER_02"],
        {
            "rung": "one_name_per_person",
            "decision": "changed",
            "name": "Elad Gilman",
            "previous": {"name": "Elad Gilmann"},
        },
    )


# --- rungs no existing roster test reaches -------------------------------------------------


def _resolve(turns: List[Tuple[str, str, float]], **kw: Any) -> Tuple[Any, Dict[str, Any]]:
    segs: List[DiarizationSegment] = []
    chunks: Dict[str, List[str]] = {}
    ordered: List[Tuple[str, str]] = []
    t = 30.0
    for spk, text, dur in turns:
        segs.append(DiarizationSegment(start=t, end=t + dur, speaker=spk))
        t += dur
        chunks.setdefault(spk, []).append(" " + text)
        ordered.append((spk, " " + text))
    vt = {v: " ".join(c) for v, c in chunks.items()}
    trace = NamingTrace()
    r = roster_mod.resolve_speaker_roster(
        DiarizationResult(segments=segs, num_speakers=len(vt)),
        " ".join(x for _, x in ordered),
        voice_texts=vt,
        ordered_turns=ordered,
        trace=trace,
        **kw,
    )
    return r, trace.to_dict()


def test_a_talkative_host_is_named_after_the_guest_is_placed() -> None:
    r, d = _resolve(
        [
            ("SPEAKER_00", "Welcome back to the show. Today we are talking about trails.", 30.0),
            ("SPEAKER_01", "Thanks for having me, it is great to be here.", 10.0),
            ("SPEAKER_00", "So tell me, how did you start building trails? " * 6, 300.0),
            ("SPEAKER_01", "I started when I was young and kept going.", 60.0),
            ("SPEAKER_00", "That is fascinating. Let me ask you about drainage. " * 6, 300.0),
        ],
        known_hosts=["Maya Koster"],
        detected_guests=["Liam Hart"],
    )
    assert r.by_voice["SPEAKER_00"].name == "Maya Koster"
    host = d["voices"]["SPEAKER_00"]
    # Seated for performing the host's role; the forced pool name held back by the ownership
    # guard; named only once the stated guest was placed on the other voice.
    assert _has(host, {"rung": "host_seat_step", "step": "2_performs_host_role"})
    assert _has(host, {"rung": "host_naming", "decision": "unnamed"})
    gates = next(e for e in d["episode"] if e["rung"] == "host_naming_forced_gates")
    assert gates["one_name_one_seat"] is True and gates["seat_owns_the_talk"] is True
    assert _has(
        host,
        {
            "rung": "talkative_host",
            "decision": "changed",
            "name": "Maya Koster",
            "previous": {"name": "SPEAKER_00", "named": False, "source": "raw"},
        },
    )


def test_a_self_introduced_show_mononym_is_removed() -> None:
    r, d = _resolve(
        [
            ("SPEAKER_00", "Hello, I'm Trivium, and welcome to the show.", 20.0),
            ("SPEAKER_01", "Thanks so much for having me on today.", 120.0),
            ("SPEAKER_00", "Let us talk about the economy.", 120.0),
        ],
        known_hosts=["Andrew Polk"],
        feed_title="The Trivium China Podcast",
    )
    assert not r.by_voice["SPEAKER_00"].named
    steps = d["voices"]["SPEAKER_00"]
    assert _has(steps, {"rung": "self_intro", "decision": "named", "name": "Trivium"})
    assert _has(steps, {"rung": "show_mononym_filter", "decision": "removed", "name": "Trivium"})


def test_an_unstated_introduced_spelling_is_refused_on_the_voice_and_the_name() -> None:
    r, d = _resolve(
        [
            ("SPEAKER_00", "Welcome to the podcast. I'm Dwarkesh Patel.", 10.0),
            ("SPEAKER_00", "Today I'm chatting with Drance Anderson, who runs the blue one.", 10.0),
            ("SPEAKER_01", "Thanks for having me. Maths is a joy to explain.", 200.0),
            ("SPEAKER_00", "Why animations?", 10.0),
            ("SPEAKER_01", "Because pictures carry the intuition.", 200.0),
        ],
        known_hosts=["Dwarkesh Patel"],
        metadata_named=["Grant Sanderson"],
    )
    assert not r.by_voice["SPEAKER_01"].named
    assert _has(d["voices"]["SPEAKER_01"], {"rung": "intro_reader", "decision": "refused_spelling"})
    assert _has(
        d["names"]["Drance Anderson"],
        {
            "rung": "host_introduction_harvest",
            "decision": "refused",
            "reason": "resembles_an_unbound_stated_name",
        },
    )


def test_an_llm_name_that_is_the_show_is_skipped() -> None:
    r, d = _resolve(
        [
            ("SPEAKER_00", "Hello and welcome. I'm Tobias Wren.", 20.0),
            ("SPEAKER_01", "Thanks for having me.", 200.0),
            ("SPEAKER_00", "Tell me more.", 20.0),
        ],
        known_hosts=["Tobias Wren"],
        feed_title="Baltic Ledgers",
        llm_voice_names={"SPEAKER_01": "Baltic Ledgers"},
    )
    assert not r.by_voice["SPEAKER_01"].named
    assert _has(
        d["voices"]["SPEAKER_01"],
        {
            "rung": "llm_merge",
            "decision": "skipped",
            "proposed": "Baltic Ledgers",
            "reason": "names_the_show",
        },
    )


def test_an_llm_name_on_an_ad_voice_is_skipped() -> None:
    ads = _load(_DZ + "test_roster_ad_voices.py")
    trace = NamingTrace()
    r = roster_mod.resolve_speaker_roster(
        ads._hardfork_shaped(),
        "I'm Paul Tenorio. I cover soccer for The Athletic. And I'm Amy Lawrence.",
        detected_guests=["Dr. Adam Rodman"],
        known_hosts=ads.HOSTS,
        voice_texts=ads._voice_texts(),
        llm_voice_names={"SPEAKER_AD1": "Paul Tenorio"},
        trace=trace,
    )
    assert not r.by_voice["SPEAKER_AD1"].named
    assert _has(
        trace.to_dict()["voices"]["SPEAKER_AD1"],
        {
            "rung": "llm_merge",
            "decision": "skipped",
            "proposed": "Paul Tenorio",
            "reason": "ad_voice",
        },
    )


# --- pipeline: what only the pipeline knows ---------------------------------------------------


def _pipeline_diagnostics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, restore: bool) -> Any:
    from podcast_scraper.config import Config

    monkeypatch.setattr(
        P,
        "_resolve_voices_via_llm",
        lambda *a, **k: ({"SPEAKER_01": "Maria Lindqvist"}, {"SPEAKER_01": "guest"}),
    )
    if restore:
        real = P._reconcile_non_regression

        def restoring(baseline: Any, final: Any) -> Any:
            fixed, _ = real(baseline, final)
            return fixed, ["SPEAKER_00"]

        monkeypatch.setattr(P, "_reconcile_non_regression", restoring)
    P._copresence_cache.clear()
    diar = DiarizationResult(
        segments=[
            DiarizationSegment(0, 30, "SPEAKER_00"),
            DiarizationSegment(30, 90, "SPEAKER_01"),
        ],
        num_speakers=2,
    )
    result = {
        "text": "Welcome. I'm Tobias Wren. Thanks for having me.",
        "segments": [
            {"start": 0, "end": 30, "text": "Welcome to the show. I'm Tobias Wren."},
            {"start": 30, "end": 90, "text": "Thanks for having me, it is a pleasure."},
        ],
    }
    cfg = Config(output_dir=str(tmp_path), speaker_resolution_llm=False)
    out = P.apply_diarization_to_result(
        result,
        "",
        cfg,
        ["Maria Lindqvist"],
        precomputed_diarization=diar,
        feed_hosts=["Tobias Wren"],
    )
    return out["speaker_diagnostics"]


def test_the_pipeline_records_the_rules_only_baseline_with_the_llm_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    diag = _pipeline_diagnostics(monkeypatch, tmp_path, restore=False)
    trace = diag["decision_trace"]
    assert trace["version"] == 1 and trace["degraded"] is False
    assert trace["inputs"]["llm_voice_names"] == {"SPEAKER_01": "Maria Lindqvist"}
    baseline = trace["inputs"]["baseline_without_llm"]
    assert set(baseline) == {"SPEAKER_00", "SPEAKER_01"}
    assert baseline["SPEAKER_00"]["name"] == "Tobias Wren"
    # Only the shipped pass is traced: the baseline pass leaves no second set of steps.
    assert [s["rung"] for s in trace["episode"]].count("host_seats") == 1


def test_the_pipeline_records_a_name_the_non_regression_contract_restored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    diag = _pipeline_diagnostics(monkeypatch, tmp_path, restore=True)
    steps = diag["decision_trace"]["voices"]["SPEAKER_00"]
    assert _has(steps, {"rung": "non_regression", "decision": "restored", "name": "Tobias Wren"})


def test_without_llm_answers_there_is_no_baseline_but_still_a_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(P, "_resolve_voices_via_llm", lambda *a, **k: ({}, {}))
    from podcast_scraper.config import Config

    P._copresence_cache.clear()
    diar = DiarizationResult(segments=[DiarizationSegment(0, 30, "SPEAKER_00")], num_speakers=1)
    result = {"text": "Hi.", "segments": [{"start": 0, "end": 30, "text": "Hi, I'm Tobias Wren."}]}
    out = P.apply_diarization_to_result(
        result,
        "",
        Config(output_dir=str(tmp_path), speaker_resolution_llm=False),
        [],
        precomputed_diarization=diar,
    )
    trace = out["speaker_diagnostics"]["decision_trace"]
    assert "baseline_without_llm" not in trace["inputs"]
    assert trace["voices"]


# --- inside the helpers: which step, which variant, which gate (#2276 phase 1, part 2) ---------


def _ep(d: Dict[str, Any], rung: str) -> Dict[str, Any]:
    return next(e for e in d["episode"] if e["rung"] == rung)


def test_each_host_seat_records_the_step_that_took_it() -> None:
    d = _traced(_DZ + "test_seat_logic_v4.py", "test_cohost_formula_is_name_bearing_not_positional")
    for v in ("SPEAKER_00", "SPEAKER_01"):
        assert _has(d["voices"][v], {"rung": "host_seat_step", "step": "1_named_as_a_stated_host"})
    d = _traced(
        _DZ + "test_roster_ad_voices.py",
        "test_the_final_gate_demotes_an_opener_laden_name_reaching_the_roster",
    )
    assert _has(
        d["voices"]["SPEAKER_HOST"], {"rung": "host_seat_step", "step": "2_performs_host_role"}
    )
    assert "stated_non_host" in _ep(d, "host_seat_guards")


def test_step_4_records_its_arithmetic() -> None:
    d = _traced(
        _DZ + "test_presenter_evidence.py",
        "test_a_guest_host_the_episode_names_is_seated_over_the_feed_hosts",
    )
    step4 = _ep(d, "host_seat_step_4")
    assert step4["empty_seats"] == 2 and step4["fillable"] == 2 and step4["candidates"] == []
    assert step4["guest_present"] is True
    assert _ep(d, "host_naming_forced_gates")["guest_hosted_episode"] is True


def test_a_forced_host_name_records_every_gate_and_veto() -> None:
    d = _traced(
        _DZ + "test_host_not_read_as_guest.py",
        "test_host_posing_a_hypothetical_is_not_named_after_it",
    )
    gates = _ep(d, "host_naming_forced_gates")
    assert gates["spare_names"] == ["Tobias Wren"] and gates["one_name_one_seat"] is True
    host = d["voices"]["SPEAKER_00"]
    assert _has(
        host,
        {
            "rung": "host_naming",
            "decision": "forced_name_vetoes",
            "performs_guest_act": False,
            "greeted_by_that_name": False,
            "forced": True,
        },
    )
    assert _has(
        host, {"rung": "host_naming", "decision": "forced_pool_name", "name": "Tobias Wren"}
    )


def test_a_forced_guest_name_records_which_variant_fired() -> None:
    d = _traced(
        _DZ + "test_two_voice_host_introduced_guest.py",
        "test_the_guest_the_host_introduces_is_named",
        "With me today is Maria Lindqvist, a historian of medieval trade.",
    )
    forced = _ep(d, "guest_naming_forced")
    assert forced["forced_by"] == "one_name_one_voice" and forced["forced_voice"] == "SPEAKER_01"
    assert _has(
        d["voices"]["SPEAKER_01"],
        {"rung": "guest_naming", "decision": "forced_name", "forced_by": "one_name_one_voice"},
    )


def test_an_unnamed_leftover_records_its_role_evidence_and_its_type() -> None:
    d = _traced(_DZ + "test_seat_logic_v4.py", "test_cohost_formula_is_name_bearing_not_positional")
    steps = d["voices"]["SPEAKER_02"]
    assert _has(
        steps, {"rung": "guest_naming", "decision": "unnamed", "role_evidence": "none_left"}
    )
    assert _has(
        steps, {"rung": "voice_types", "decision": "typed", "reason": "no_source_names_them"}
    )


def test_one_name_per_person_records_the_spelling_and_role_it_kept_and_why() -> None:
    d = _traced(
        _DZ + "test_presenter_evidence.py",
        "test_an_evidence_host_takes_the_episodes_stated_spelling_of_the_name",
    )
    unified = _ep(d, "one_name_per_person")
    assert unified["decision"] == "unified" and unified["kept_name"] == "Imani Moise"
    assert unified["kept_because"] == "stated" and unified["role"] == "host"
    assert unified["role_reason"] == "a_known_host_or_a_voice_with_host_evidence"
    d = _traced(
        _DZ + "test_one_person_one_name.py",
        "test_two_spellings_of_one_guest_publish_as_one_name",
        cls="TestTheResolvedRosterCarriesOneName",
    )
    unified = _ep(d, "one_name_per_person")
    assert unified["kept_because"] == "fullest_then_most_talk"
    assert unified["role_reason"] == "voices_agree"


def test_the_intro_reader_records_what_it_heard_and_names_the_spelling_it_refused() -> None:
    _, d = _resolve(
        [
            ("SPEAKER_00", "Welcome to the podcast. I'm Dwarkesh Patel.", 10.0),
            ("SPEAKER_00", "Today I'm chatting with Drance Anderson, who runs the blue one.", 10.0),
            ("SPEAKER_01", "Thanks for having me. Maths is a joy to explain.", 200.0),
            ("SPEAKER_00", "Why animations?", 10.0),
            ("SPEAKER_01", "Because pictures carry the intuition.", 200.0),
        ],
        known_hosts=["Dwarkesh Patel"],
        metadata_named=["Grant Sanderson"],
    )
    steps = d["voices"]["SPEAKER_01"]
    assert _has(steps, {"rung": "intro_reader", "decision": "heard", "heard": "Drance Anderson"})
    assert _has(
        steps,
        {
            "rung": "intro_reader",
            "decision": "refused_spelling",
            "name": "Drance Anderson",
            "resembles": "Grant Sanderson",
        },
    )
