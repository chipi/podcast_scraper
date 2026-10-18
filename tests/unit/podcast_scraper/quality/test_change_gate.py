"""The corpus change gate: content code waits for its record; a migration lands only complete."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from podcast_scraper.quality import change_gate as cg
from podcast_scraper.upgrade.registry import get_migrations

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parents[4]
MIG = "src/podcast_scraper/upgrade/migrations/m0099_x.py"
DONE = {s: f"done for step {s}" for s in cg.LANDING_STEPS}
DONE["4"] = "Dated: 2026-10-18\n\nS1n 1786 -> 7."
DONE["8"] = "Verdict: GO\n\nthe reviewer's reasons; outcome {sha}"


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir()
    shutil.copy(REPO / cg.CONFIG_PATH, tmp_path / cg.CONFIG_PATH)
    (tmp_path / "docs/wip/changes").mkdir(parents=True)
    return tmp_path


def _record(root: Path, rid: str, cls: str, sections: dict, extra: str = "", status: str = "open"):
    body = "".join(f"## {k}. step\n\n{v}\n\n" for k, v in sorted(sections.items()))
    text = (
        f"---\nid: {rid}\nclass: {cls}   # full | light\nstatus: {status}\nissue: 1\ntouches:\n"
        f"  - {MIG}\n---\n\n# {rid}\n\n{body}{extra}"
    )
    (root / "docs/wip/changes" / f"{rid}.md").write_text(text, encoding="utf-8")


GATE = {
    "scoreboard_before": {"S1": 2},
    "scoreboard_after": {"S1": 1},
    "plan_sha256": "p",
    "episode_deltas": {},
}


def _gate(root: Path, rid: str, computed: str, outcome: bool = True, verdict: str = "GO") -> str:
    """The gate report and (for a full change) the shipped outcome; returns the outcome sha."""
    path = root / "docs/wip/changes" / f"{rid}.gate.json"
    path.write_text(json.dumps({"computed_class": computed, **GATE}), encoding="utf-8")
    if not outcome:
        return ""
    od = root / cg.OUTCOMES_DIR
    od.mkdir(parents=True, exist_ok=True)
    raw = json.dumps({"verdict": verdict, "code_dirty": False, **GATE}).encode()
    (od / f"{rid}.json").write_bytes(raw)
    import hashlib

    return hashlib.sha256(raw).hexdigest()


def test_the_config_guards_migrations_naming_gi_and_the_ad_cut() -> None:
    config = cg.load_config(REPO)
    for rel in (
        MIG,
        "src/podcast_scraper/gi/pipeline.py",
        "src/podcast_scraper/workflow/adfree_transcript.py",
        "src/podcast_scraper/providers/ml/diarization/roster.py",
    ):
        assert cg.is_content_code(rel, config), rel
    for rel in ("src/podcast_scraper/server/app_cache_warm.py", "tests/unit/x.py", "docs/a.md"):
        assert not cg.is_content_code(rel, config), rel


def test_content_code_without_a_record_is_refused_and_other_code_is_not(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    assert cg.check_edit(root, ["src/podcast_scraper/server/app_cache_warm.py"]) == []
    [why] = cg.check_edit(root, [MIG])
    assert "no open change record" in why and "TEMPLATE.md" in why


def test_a_record_still_on_todo_names_every_missing_step(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _record(root, "0099_x", "full", {s: cg.TODO for s in cg.PRE_BUILD_STEPS})
    problems = cg.check_edit(root, [MIG])
    assert [p.split(": ")[2] for p in problems] == [f"step {s}" for s in cg.PRE_BUILD_STEPS]


def test_steps_one_to_five_done_lets_the_code_be_written(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _record(root, "0099_x", "full", {s: DONE[s] for s in cg.PRE_BUILD_STEPS})
    assert cg.check_edit(root, [MIG]) == []


def test_a_measured_change_is_revised_only_after_a_newer_prediction(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    sections: dict = {s: DONE[s] for s in cg.PRE_BUILD_STEPS}
    _record(root, "0099_x", "full", sections)
    gate = root / "docs/wip/changes/0099_x.gate.json"
    gate.write_text(json.dumps({"computed_class": "full", "measured_at": "2026-10-18T08:00:00"}))
    why = cg.check_edit(root, [MIG])
    assert len(why) == 2 and "new prediction" in why[0] and "new acceptance" in why[1]
    sections["4"] = DONE["4"] + "\n\nDated: 2026-10-18T07:59\nS1n 7 -> 36."
    _record(root, "0099_x", "full", sections)
    assert len(cg.check_edit(root, [MIG])) == 2  # dated before the measurement: still refused
    sections["4"] = DONE["4"] + "\n\nDated: 2026-10-18T08:30\nS1n 7 -> 36."
    _record(root, "0099_x", "full", sections)
    [only] = cg.check_edit(root, [MIG])
    assert "new acceptance" in only  # the prediction alone is not enough
    sections["5"] = "Dated: 2026-10-18T08:40\nCriteria by the reviewer."
    _record(root, "0099_x", "full", sections)
    assert cg.check_edit(root, [MIG]) == []


def test_a_measured_change_guards_its_paths_against_another_records_edit(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _record(root, "0099_x", "full", {s: DONE[s] for s in cg.PRE_BUILD_STEPS})
    gate = root / "docs/wip/changes/0099_x.gate.json"
    gate.write_text(json.dumps({"computed_class": "full", "measured_at": "2026-10-18T08:00:00"}))
    _record(root, "hook_y", "full", {s: DONE[s] for s in cg.PRE_BUILD_STEPS})  # ready, unmeasured
    problems = cg.check_edit(root, [MIG])
    assert problems and all("0099_x.md" in p for p in problems)


def test_an_unmeasured_record_does_not_block_another_ready_one(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _record(root, "0099_x", "full", {s: DONE[s] for s in cg.PRE_BUILD_STEPS})
    _record(root, "0098_held", "full", {s: cg.TODO for s in cg.PRE_BUILD_STEPS})
    assert cg.check_edit(root, [MIG]) == []


def test_a_landed_record_no_longer_opens_the_path(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _record(root, "0099_x", "full", DONE, status="landed")
    assert "no open change record" in cg.check_edit(root, [MIG])[0]


@pytest.mark.parametrize(
    "prediction, why",
    [
        ("S1n will fall a lot.", "no 'Dated"),
        ("Dated: 2026-10-18\n\nS1n will fall a lot.", "no integers"),
    ],
)
def test_a_prediction_needs_a_date_and_integers(tmp_path: Path, prediction: str, why: str) -> None:
    root = _repo(tmp_path)
    _record(root, "0099_x", "full", {**{s: DONE[s] for s in cg.PRE_BUILD_STEPS}, "4": prediction})
    [problem] = cg.check_edit(root, [MIG])
    assert why in problem


def test_a_missed_step_counts_only_once_the_operator_accepted_it(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    sections = {
        **{s: DONE[s] for s in cg.PRE_BUILD_STEPS},
        "4": "MISSED: measured before predicting",
    }
    _record(root, "0099_x", "full", sections)
    assert "Operator accepted" in cg.check_edit(root, [MIG])[0]
    _record(root, "0099_x", "full", sections, extra="Operator accepted: 2026-10-18 step 4 missed\n")
    assert cg.check_edit(root, [MIG]) == []


def test_light_needs_its_reason_and_lands_only_when_the_gate_measured_light(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _record(root, "0099_x", "light", {"0": cg.TODO})
    assert "Why light" in cg.check_edit(root, [MIG])[0]
    _record(root, "0099_x", "light", {"0": "writes only artwork copies"})
    assert cg.check_edit(root, [MIG]) == []
    assert "no gate report" in cg.check_landing(root, ["0099_x"])[0]
    _gate(root, "0099_x", "full")
    problems = cg.check_landing(root, ["0099_x"])
    assert "record says light, the gate measured 'full'" in problems[0]
    assert len(problems) > 1  # measured full, so every full-method step is required too
    _gate(root, "0099_x", "light", outcome=False)
    assert cg.check_landing(root, ["0099_x"]) == []


def test_landing_needs_steps_one_to_eight_and_ignores_migrations_before_the_regime(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    assert cg.check_landing(root, ["0028_old"]) == []
    assert "no change record" in cg.check_landing(root, ["0099_x"])[0]
    _record(root, "0099_x", "full", {s: DONE[s] for s in cg.PRE_BUILD_STEPS})
    sha = _gate(root, "0099_x", "full")
    assert [p.split(": ")[1] for p in cg.check_landing(root, ["0099_x"])] == [
        "step 6",
        "step 7",
        "step 8",
        "step 8",  # not done, and no 'Verdict: GO' line
    ]
    _record(root, "0099_x", "full", {**DONE, "8": DONE["8"].format(sha=sha)})
    assert cg.check_landing(root, ["0099_x"]) == []


def test_landing_needs_the_shipped_outcome_bound_to_the_record(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _gate(root, "0099_x", "full", outcome=False)
    _record(root, "0099_x", "full", {**DONE, "8": DONE["8"].format(sha="x")})
    assert "no shipped outcome" in cg.check_landing(root, ["0099_x"])[0]
    sha = _gate(root, "0099_x", "full")
    assert "does not name the outcome's sha256" in cg.check_landing(root, ["0099_x"])[0]
    _record(root, "0099_x", "full", {**DONE, "8": DONE["8"].format(sha=sha)})
    assert cg.check_landing(root, ["0099_x"]) == []
    out = root / cg.OUTCOMES_DIR / "0099_x.json"
    doc = json.loads(out.read_text())
    doc["scoreboard_after"] = {"S1": 0}  # one integer changed after the verdict
    out.write_text(json.dumps(doc))
    problems = cg.check_landing(root, ["0099_x"])
    assert any("sha256" in p for p in problems) and any("scoreboard_after" in p for p in problems)
    sha = _gate(root, "0099_x", "full", verdict="GO WITH CONDITIONS")
    _record(root, "0099_x", "full", {**DONE, "8": DONE["8"].format(sha=sha)})
    assert "is not the record's" in cg.check_landing(root, ["0099_x"])[0]


def test_landing_needs_the_last_verdict_to_be_go(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    for text, ok, v in (
        ("Verdict: NO-GO\nH1", False, "NO-GO"),
        ("Verdict: NO-GO\n\nVerdict: GO WITH CONDITIONS\nC1", True, "GO WITH CONDITIONS"),
        ("Verdict: GO\n\nVerdict: NO-GO\nrevised", False, "NO-GO"),
        ("the reviewer liked it", False, "GO"),
    ):
        sha = _gate(root, "0099_x", "full", verdict=v)
        _record(root, "0099_x", "full", {**DONE, "8": text + f" outcome {sha}"})
        assert (cg.check_landing(root, ["0099_x"]) == []) is ok, text


def test_a_written_file_is_classed_by_what_changed_in_it() -> None:
    config = cg.load_config(REPO)
    meta_a = json.dumps({"content": {"speakers": [{"name": "A"}]}, "artwork": {"medium": None}})
    meta_art = json.dumps(
        {"content": {"speakers": [{"name": "A"}]}, "artwork": {"medium": "m.jpg"}}
    )
    meta_named = json.dumps({"content": {"speakers": [{"name": "B"}]}, "artwork": {"medium": None}})
    m = "feeds/f/run/metadata/e.metadata.json"
    assert cg.classify_written([(m, meta_a.encode(), meta_art.encode())], config)[0] == "light"
    cls, why = cg.classify_written([(m, meta_a.encode(), meta_named.encode())], config)
    assert cls == "full" and "content.speakers" in why[0]
    assert (
        cg.classify_written([("feeds/f/run/metadata/e.gi.json", b"{}", b"{}")], config)[0] == "full"
    )
    assert cg.classify_written([("feeds/f/run/transcripts/e.txt", b"a", b"b")], config)[0] == "full"
    assert cg.classify_written([("feeds/f/run/artwork/e.jpg", b"a", b"b")], config)[0] == "light"


def test_any_gating_move_is_reported() -> None:
    config = cg.load_config(REPO)
    assert cg.gating_moves({"S1n_named_but_hidden": 5}, {"S1n_named_but_hidden": 5}, config) == {}
    moves = cg.gating_moves({"S1n_named_but_hidden": 5, "quotes": 1}, {"quotes": 9}, config)
    assert moves == {"S1n_named_but_hidden": -5}


def _hook():
    spec = importlib.util.spec_from_file_location(
        "gate_hook", REPO / "scripts/hooks/corpus_change_gate_hook.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "command, written",
    [
        (f"grep -n x {MIG} 2>&1 | head", False),
        (f"sed -n 1,20p {MIG}", False),
        (f"black --check {MIG}", False),
        (f"sed -i '' 's/a/b/' {MIG}", True),
        (f"cat > {MIG} <<'EOF'", True),
        (f"cp /tmp/draft.py {MIG}", True),
        (f"black {MIG}", True),
    ],
)
def test_the_hook_sees_shell_writes_and_not_reads(
    tmp_path: Path, command: str, written: bool
) -> None:
    root = _repo(tmp_path)
    hook = _hook()
    targets = hook._targets(
        {"tool_name": "Bash", "cwd": str(root), "tool_input": {"command": command}}
    )
    assert bool(targets) is written
    if written:
        assert targets == [(root.resolve(), MIG)]


def test_the_hook_judges_a_file_by_its_own_checkout(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    hook = _hook()
    call = {"tool_name": "Edit", "cwd": "/", "tool_input": {"file_path": str(root / MIG)}}
    assert hook._targets(call) == [(root.resolve(), MIG)]
    outside = {"tool_name": "Write", "cwd": "/", "tool_input": {"file_path": "/etc/hosts"}}
    assert hook._targets(outside) == []


def test_every_registered_migration_under_the_regime_is_allowed_to_land() -> None:
    """The CI half of the gate: a migration from 0031 on lands only with a complete record."""
    assert cg.check_landing(REPO, [m.id for m in get_migrations()]) == []
