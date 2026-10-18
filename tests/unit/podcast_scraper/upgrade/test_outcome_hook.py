"""The migration outcome hook: a regime migration applies only as it was measured on its clone."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper.upgrade import outcome_hook as oh
from podcast_scraper.upgrade.migration import MigrationContext, MigrationResult

pytestmark = [pytest.mark.unit]

# A fake regime migration living in THIS module, so the hook's module-level lookups
# (``dry_run_report``) find it.
PLAN: List[Dict[str, Any]] = []
WRITE_ON_APPLY: List[Path] = []


def dry_run_report(root: Path):
    return [dict(r) for r in PLAN]


class Fake:
    id = "0099_fake"
    to_version = "9.9.9"
    _receipts_file = "fake.jsonl"
    _backup_tag = "0099"

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        for p in WRITE_ON_APPLY:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x")
        return MigrationResult(
            self.id, applied=True, dry_run=False, message="", details={"episodes": []}
        )


ROW = {
    "meta": "feeds/f/run_1/metadata/ep.metadata.json",
    "counts": {"voices_shown": 1},
    "inputs": {"a": "1"},
}


def _outcome(**over: Any) -> Dict[str, Any]:
    doc = {
        "migration": "0099_fake",
        "snapshot": "snap",
        "applied_before": ["0001"],
        "code_digest": "d",
        "code_dirty": False,
        "verdicts_sha256": None,
        "scoreboard_before": {},
        "scoreboard_after": {},
        "episode_deltas": {ROW["meta"]: {"S1_hidden_unknown": -1}},
        "plan": [ROW],
        "computed_class": "full",
        "verdict": "GO",
        "conditions": [],
    }
    doc.update(over)
    return doc


@pytest.fixture(autouse=True)
def _reset(monkeypatch, tmp_path: Path):
    PLAN[:] = [dict(ROW)]
    WRITE_ON_APPLY[:] = []
    out = tmp_path / "outcome.json"
    out.write_text(json.dumps(_outcome()))
    monkeypatch.setattr(oh, "outcome_path", lambda mid: out)
    monkeypatch.setattr(oh, "migration_digest", lambda m: "d")
    yield out


def test_a_good_outcome_loads_with_its_sha(_reset: Path) -> None:
    doc, sha = oh.load_outcome("0099_fake")
    assert doc["verdict"] == "GO" and sha == hashlib.sha256(_reset.read_bytes()).hexdigest()


@pytest.mark.parametrize("field", oh.REQUIRED)
def test_an_outcome_missing_any_field_is_refused_naming_it(_reset: Path, field: str) -> None:
    doc = _outcome()
    doc.pop(field)
    _reset.write_text(json.dumps(doc))
    with pytest.raises(oh.HookRefused, match=f"lacks {field}"):
        oh.load_outcome("0099_fake")


@pytest.mark.parametrize(
    "over, why",
    [
        ({"verdict": "NO-GO"}, "not GO"),
        ({"verdict": ""}, "not GO"),
        ({"verdict": "MAYBE"}, "not GO"),
        ({"code_dirty": True}, "uncommitted"),
        ({"migration": "0098_other"}, "outcome is for"),
    ],
)
def test_a_bad_outcome_is_refused(_reset: Path, over: Dict[str, Any], why: str) -> None:
    _reset.write_text(json.dumps(_outcome(**over)))
    with pytest.raises(oh.HookRefused, match=why):
        oh.load_outcome("0099_fake")


def test_a_missing_or_unreadable_outcome_is_refused(_reset: Path) -> None:
    _reset.write_text("{")
    with pytest.raises(oh.HookRefused, match="unreadable"):
        oh.load_outcome("0099_fake")
    _reset.unlink()
    with pytest.raises(oh.HookRefused, match="no shipped outcome"):
        oh.load_outcome("0099_fake")


def test_the_code_digest_covers_the_imported_modules_and_the_verdicts(tmp_path: Path) -> None:
    name = "podcast_scraper.upgrade.migrations.m0031_named_voices_shown"
    closure = oh.module_closure(name)
    for dep in (
        "podcast_scraper.upgrade.migrations.m0025_one_person_one_entry",
        "podcast_scraper.upgrade.migrations.m0029_naming_arc_voices_repaired",
        "podcast_scraper.upgrade.reviewer_verdicts",
        "podcast_scraper.gi.pipeline",
    ):
        assert dep in closure, dep
    extra = tmp_path / "v.tsv"
    extra.write_text("a")
    first = oh.code_digest(name, [extra])
    assert first == oh.code_digest(name, [extra])
    extra.write_text("b")
    assert oh.code_digest(name, [extra]) != first


def test_static_checks_refuse_a_wrong_run(_reset: Path, monkeypatch) -> None:
    with pytest.raises(oh.HookRefused, match="run 0030_x first"):
        oh.check_static(Fake(), ["0030_x", "0099_fake"], {"0001"})
    with pytest.raises(oh.HookRefused, match="applied migrations differ"):
        oh.check_static(Fake(), ["0099_fake"], {"0001", "0030_x"})
    monkeypatch.setattr(oh, "migration_digest", lambda m: "other")
    with pytest.raises(oh.HookRefused, match="not the code that was measured"):
        oh.check_static(Fake(), ["0099_fake"], {"0001"})


def test_plan_differences_are_every_kind() -> None:
    a = [ROW, {**ROW, "meta": "x"}]
    b = [{**ROW, "inputs": {"a": "2"}}, {**ROW, "meta": "y"}]
    kinds = {(d["meta"], d["kind"]) for d in oh.plan_diff(a, b)}
    assert kinds == {(ROW["meta"], "differs"), ("x", "clone_only"), ("y", "prod_only")}
    assert oh.plan_diff(a, a) == []


def test_a_plan_difference_stops_the_run_and_writes_only_the_diff(
    _reset: Path, tmp_path: Path
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    PLAN[0]["inputs"] = {"a": "2"}
    with pytest.raises(oh.HookStopped, match="differs from the clone's in 1"):
        oh.prepare(Fake(), root, ["0099_fake"], {"0001"})
    written = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]
    assert written == [".podcast_scraper/upgrade-outcomes/0099_fake.plan-diff.json"]


def _accept(root: Path, sha: str, causes: bool = True, name: str = "accept", **over: Any) -> Path:
    diff = json.loads(
        (root / ".podcast_scraper/upgrade-outcomes/0099_fake.plan-diff.json").read_text()
    )
    doc = {
        "migration": "0099_fake",
        "outcome_sha256": sha,
        "operator_go": "go, 2026-10-18",
        "entries": [
            {**e, "cause": "reprocessed on prod" if causes else ""} for e in diff["entries"]
        ],
    }
    doc.update(over)
    path = root / f"{name}.json"
    path.write_text(json.dumps(doc))
    return path


def test_an_accepted_plan_difference_applies_only_as_written(_reset: Path, tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    PLAN[0]["inputs"] = {"a": "2"}
    with pytest.raises(oh.HookStopped):
        oh.prepare(Fake(), root, ["0099_fake"], {"0001"})
    _doc, sha = oh.load_outcome("0099_fake")
    prep = oh.prepare(Fake(), root, ["0099_fake"], {"0001"}, _accept(root, sha))
    assert len(prep.diff) == 1 and prep.accept_sha
    for bad, why in (
        (_accept(root, sha, causes=False, name="a1"), "no cause"),
        (_accept(root, "0" * 64, name="a2"), "another migration or outcome"),
        (_accept(root, sha, name="a3", operator_go=""), "no operator go"),
        (_accept(root, sha, name="a4", entries=[]), "not the accepted one"),
    ):
        with pytest.raises(oh.HookStopped, match=why):
            oh.prepare(Fake(), root, ["0099_fake"], {"0001"}, bad)
    good = _accept(root, sha)
    PLAN.append({**ROW, "meta": "z"})  # prod gained one more difference since the file
    with pytest.raises(oh.HookStopped, match="not the accepted one"):
        oh.prepare(Fake(), root, ["0099_fake"], {"0001"}, good)


def test_the_after_check_is_exact_per_episode_and_on_differing_ones_no_rise() -> None:
    metrics = ["S1_hidden_unknown", "S1n_named_but_hidden"]
    clone = {"e1": {"S1_hidden_unknown": -2}}
    assert oh.compare({"e1": {"S1_hidden_unknown": -2}}, clone, set(), metrics) == []
    assert oh.compare({"e1": {"S1_hidden_unknown": -1}}, clone, set(), metrics)
    assert oh.compare({"e1": {"S1_hidden_unknown": +2}}, clone, set(), metrics)  # sign flipped
    assert oh.compare(
        {"e1": {"S1_hidden_unknown": -2}, "e9": {"S1n_named_but_hidden": 1}}, clone, set(), metrics
    )
    assert oh.compare({"e1": {"S1_hidden_unknown": -5}}, clone, {"e1"}, metrics) == []  # a fall
    assert oh.compare({"e1": {"S1n_named_but_hidden": 1}}, clone, {"e1"}, metrics)  # a rise


def test_measure_fails_loudly_and_keeps_the_apply(
    _reset: Path, tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    prep = oh.prepare(Fake(), root, ["0099_fake"], {"0001"})
    scores = iter(
        [{ROW["meta"]: {"S1_hidden_unknown": 3}}, {ROW["meta"]: {"S1_hidden_unknown": 3}}]
    )
    monkeypatch.setattr(oh, "score_by_episode", lambda r: next(scores))
    applied = []
    with pytest.raises(oh.HookOutcomeFailed, match="not the measured one") as exc:
        oh.measure(prep, root, lambda: applied.append(1), None)
    assert applied == [1] and "to undo" in str(exc.value)
    receipt = json.loads((root / ".podcast_scraper/upgrade-outcomes/0099_fake.json").read_text())
    assert receipt["verdict"] == "FAIL" and receipt["failures"]


def test_measure_passes_when_prod_moves_as_the_clone(
    _reset: Path, tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    prep = oh.prepare(Fake(), root, ["0099_fake"], {"0001"})
    scores = iter(
        [{ROW["meta"]: {"S1_hidden_unknown": 3}}, {ROW["meta"]: {"S1_hidden_unknown": 2}}]
    )
    monkeypatch.setattr(oh, "score_by_episode", lambda r: next(scores))
    oh.measure(prep, root, lambda: None, "/snap")
    receipt = json.loads((root / ".podcast_scraper/upgrade-outcomes/0099_fake.json").read_text())
    assert receipt["verdict"] == "PASS" and receipt["snapshot"] == "/snap"


def test_an_after_score_failure_is_not_measured_not_passed(
    _reset: Path, tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    prep = oh.prepare(Fake(), root, ["0099_fake"], {"0001"})
    calls = []

    def score(r):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("disk")
        return {}

    monkeypatch.setattr(oh, "score_by_episode", score)
    with pytest.raises(oh.HookOutcomeFailed, match="after-scoreboard failed"):
        oh.measure(prep, root, lambda: None, None)
    receipt = json.loads((root / ".podcast_scraper/upgrade-outcomes/0099_fake.json").read_text())
    assert receipt["verdict"] == "NOT MEASURED"


def test_an_empty_plan_must_write_nothing(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    oh.apply_empty(Fake(), root, lambda: Fake().apply(MigrationContext(corpus_root=root)))
    WRITE_ON_APPLY.append(root / "fake.jsonl")
    with pytest.raises(oh.HookOutcomeFailed, match="plan was empty but the apply wrote"):
        oh.apply_empty(Fake(), root, lambda: Fake().apply(MigrationContext(corpus_root=root)))


def test_a_plan_without_counts_or_inputs_or_report_is_refused() -> None:
    PLAN[0].pop("inputs")
    with pytest.raises(oh.HookRefused, match="lack 'counts' or 'inputs'"):
        oh.episode_plan(Fake(), Path("/"))

    class NoReport(Fake):
        pass

    NoReport.__module__ = "podcast_scraper.upgrade.migration"
    with pytest.raises(oh.HookRefused, match="no dry_run_report"):
        oh.episode_plan(NoReport(), Path("/"))


def test_the_runner_decides_empty_again_at_the_apply(_reset: Path, tmp_path: Path) -> None:
    """Criteria R1: a plan that was empty before the run but is not at the apply is refused."""
    from podcast_scraper.upgrade.runner import UpgradeRunner
    from podcast_scraper.upgrade.state import FilesystemStateStore

    root = tmp_path / "corpus"
    root.mkdir()
    runner = UpgradeRunner(FilesystemStateStore(root), [Fake()])
    PLAN[:] = [dict(ROW)]  # non-empty at the apply, never prepared
    with pytest.raises(oh.HookRefused, match="its plan is not empty here"):
        runner.run(MigrationContext(corpus_root=root), hold_lock=False)
    PLAN[:] = []
    runner.run(MigrationContext(corpus_root=root), hold_lock=False)
    assert "0099_fake" in FilesystemStateStore(root).applied_migration_ids()


def test_the_cli_returns_each_hook_exit_code(monkeypatch, tmp_path: Path) -> None:
    from podcast_scraper.upgrade import cli_handlers as ch

    class Status:
        pending = [Fake()]

    class Runner:
        def status(self):
            return Status()

    args = argparse.Namespace(dry_run=False, yes=True, json=False)
    for exc, code in (
        (oh.HookRefused("r"), 3),
        (oh.HookStopped("s"), 4),
        (oh.HookOutcomeFailed("f"), 5),
    ):
        monkeypatch.setattr(ch, "_snapshot_then_run", lambda *a, e=exc: (_ for _ in ()).throw(e))
        assert (
            ch._cmd_run(
                Runner(),
                MigrationContext(corpus_root=tmp_path),
                args,
                __import__("logging").getLogger(),
            )
            == code
        )


def test_there_is_no_flag_that_skips_the_hook() -> None:
    from podcast_scraper.upgrade.cli_handlers import parse_upgrade_argv

    for flag in ("--no-outcome", "--skip-hook", "--force", "--no-hook"):
        with pytest.raises(SystemExit):
            parse_upgrade_argv(["run", flag])


class Earlier(Fake):
    id = "0010_earlier"
    to_version = "9.9.8"


def _regime_args(**over: Any) -> argparse.Namespace:
    args = argparse.Namespace(to_version=None, dry_run=False, accept_plan_diff=None)
    for k, v in over.items():
        setattr(args, k, v)
    return args


def test_a_regime_migration_with_others_pending_is_refused_before_anything(
    _reset: Path, tmp_path: Path
) -> None:
    """Criterion B3: a regime migration runs alone, so its plan is the state its clone measured."""
    from podcast_scraper.upgrade import cli_handlers as ch
    from podcast_scraper.upgrade.runner import UpgradeRunner
    from podcast_scraper.upgrade.state import FilesystemStateStore

    root = tmp_path / "corpus"
    root.mkdir()
    runner = UpgradeRunner(FilesystemStateStore(root), [Earlier(), Fake()])
    with pytest.raises(oh.HookRefused, match="runs alone"):
        ch._prepare_regime(runner, MigrationContext(corpus_root=root), _regime_args())
    assert runner.prepared == {}
    assert sorted(p.name for p in root.iterdir()) == []

    # --to the earlier one's version leaves the regime migration out of the run.
    ch._prepare_regime(runner, MigrationContext(corpus_root=root), _regime_args(to_version="9.9.8"))
    assert runner.prepared == {}


def test_a_lone_regime_migration_is_prepared(_reset: Path, tmp_path: Path) -> None:
    from podcast_scraper.upgrade import cli_handlers as ch
    from podcast_scraper.upgrade.runner import UpgradeRunner
    from podcast_scraper.upgrade.state import FilesystemStateStore

    root = tmp_path / "corpus"
    root.mkdir()
    store = FilesystemStateStore(root)
    runner = UpgradeRunner(store, [Fake()])
    applied = store.applied_migration_ids()
    _reset.write_text(json.dumps(_outcome(applied_before=sorted(applied))))
    ch._prepare_regime(runner, MigrationContext(corpus_root=root), _regime_args())
    assert list(runner.prepared) == ["0099_fake"]


def test_measure_fails_when_s1c_is_left_after_even_if_deltas_match(
    _reset: Path, tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    prep = oh.prepare(Fake(), root, ["0099_fake"], {"0001"})
    before = {ROW["meta"]: {"S1_hidden_unknown": 3, "S1c_inconsistent": 2}}
    after = {ROW["meta"]: {"S1_hidden_unknown": 2, "S1c_inconsistent": 2}}
    scores = iter([before, after])
    monkeypatch.setattr(oh, "score_by_episode", lambda r: next(scores))
    with pytest.raises(oh.HookOutcomeFailed, match="S1c is 2 after"):
        oh.measure(prep, root, lambda: None, None)


def test_an_accepted_run_records_the_accept_file_in_its_receipt(
    _reset: Path, tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    PLAN[0]["inputs"] = {"a": "2"}
    with pytest.raises(oh.HookStopped):
        oh.prepare(Fake(), root, ["0099_fake"], {"0001"})
    _doc, sha = oh.load_outcome("0099_fake")
    accept = _accept(root, sha)
    prep = oh.prepare(Fake(), root, ["0099_fake"], {"0001"}, accept)
    scores = iter(
        [{ROW["meta"]: {"S1_hidden_unknown": 3}}, {ROW["meta"]: {"S1_hidden_unknown": 2}}]
    )
    monkeypatch.setattr(oh, "score_by_episode", lambda r: next(scores))
    oh.measure(prep, root, lambda: None, None)
    receipt = json.loads((root / ".podcast_scraper/upgrade-outcomes/0099_fake.json").read_text())
    assert receipt["accept_sha256"] == hashlib.sha256(accept.read_bytes()).hexdigest()
    assert receipt["accept"]["operator_go"] == "go, 2026-10-18"
    assert receipt["plan_diff"] == prep.diff


def test_the_built_in_gating_metrics_equal_the_gate_config(monkeypatch, tmp_path: Path) -> None:
    """Prod runs the installed package, where the config file may be absent."""
    from_config = oh.gating_metrics()
    monkeypatch.setattr(oh, "PACKAGE_ROOT", tmp_path / "a" / "b")
    assert oh.gating_metrics() == from_config


@pytest.mark.parametrize("side", ["prod_only", "clone_only"])
def test_a_row_on_one_side_only_stops_the_run(_reset: Path, tmp_path: Path, side: str) -> None:
    """0031 C-b: the served selection is not hashed; a different served population on prod shows
    as a plan row on one side only, and that stops the run until it is accepted with a cause."""
    root = tmp_path / "corpus"
    root.mkdir()
    if side == "prod_only":
        PLAN.append({**ROW, "meta": "feeds/f/run_2/metadata/new.metadata.json"})
    else:
        PLAN[:] = []
        PLAN.append({**ROW, "meta": "feeds/f/run_2/metadata/other.metadata.json"})
    with pytest.raises(oh.HookStopped, match="differs from the clone's"):
        oh.prepare(Fake(), root, ["0099_fake"], {"0001"})
    diff = json.loads(
        (root / ".podcast_scraper/upgrade-outcomes/0099_fake.plan-diff.json").read_text()
    )
    assert side in {e["kind"] for e in diff["entries"]}
