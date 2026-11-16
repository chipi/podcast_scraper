"""The migration outcome hook: a regime migration applies only as it was measured on its clone."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast, Dict, List

import pytest

from podcast_scraper.upgrade import outcome_hook as oh
from podcast_scraper.upgrade.migration import Migration, MigrationContext, MigrationResult

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


EP = "feeds/f/run_1/metadata/ep.metadata.json"
OTHER = "feeds/f/run_1/metadata/other.metadata.json"
NEW = "feeds/f/run_2/metadata/new.metadata.json"


def _measured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clone_deltas: Dict[str, Dict[str, int]],
    diff: List[Dict[str, Any]],
    before: Dict[str, Dict[str, int]],
    after: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    """``measure()`` on a prepared run with this outcome delta and plan diff; the receipt."""
    root = tmp_path / "corpus"
    root.mkdir(exist_ok=True)
    prep = oh.Prepared(Fake(), _outcome(episode_deltas=clone_deltas), "sha", diff)
    scores = iter([before, after])
    monkeypatch.setattr(oh, "score_by_episode", lambda r: next(scores))
    try:
        oh.measure(prep, root, lambda: None, None)
    except oh.HookOutcomeFailed:
        pass
    receipt: Dict[str, Any] = json.loads(
        (root / ".podcast_scraper/upgrade-outcomes/0099_fake.json").read_text()
    )
    return receipt


def _row(meta: str, kind: str = "differs", fields: Any = ("inputs",)) -> Dict[str, Any]:
    return {"meta": meta, "kind": kind, "fields": list(fields) if fields is not None else None}


def test_an_inputs_only_row_passes_the_rise_the_clone_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """0043 on prod, 2026-11-05: every row accepted for a ledger-only input difference, and the
    measured S2_hidden +1 failed though prod's delta was the clone's. Replayed: it passes now."""
    r = _measured(
        tmp_path,
        monkeypatch,
        {EP: {"S2_hidden": 1}},
        [_row(EP)],
        {EP: {"S2_hidden": 3}},
        {EP: {"S2_hidden": 4}},
    )
    assert r["verdict"] == "PASS" and r["failures"] == []
    assert r["inputs_only_matched"] == [EP]


def test_a_non_gating_metric_is_never_compared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = _measured(
        tmp_path, monkeypatch, {}, [_row(EP)], {EP: {"not_gating": 0}}, {EP: {"not_gating": 9}}
    )
    assert r["verdict"] == "PASS"


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
    runner = UpgradeRunner(FilesystemStateStore(root), [cast(Migration, Fake())])
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
                cast(Any, Runner()),
                MigrationContext(corpus_root=tmp_path),
                args,
                __import__("logging").getLogger(),
            )
            == code
        )


def test_an_outcome_failure_prints_the_ledger_refresh_it_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """0043 on prod, 2026-11-05: exit 5 left the closing ledger refresh unrun and unsaid."""
    from podcast_scraper.upgrade import cli_handlers as ch

    class Status:
        pending = [Fake()]

    class Runner:
        def status(self):
            return Status()

    args = argparse.Namespace(dry_run=False, yes=True, json=False)
    for exc, said in ((oh.HookOutcomeFailed("f"), True), (oh.HookStopped("s"), False)):
        monkeypatch.setattr(ch, "_snapshot_then_run", lambda *a, e=exc: (_ for _ in ()).throw(e))
        ch._cmd_run(
            cast(Any, Runner()),
            MigrationContext(corpus_root=tmp_path),
            args,
            __import__("logging").getLogger(),
        )
        err = capsys.readouterr().err
        assert (f"identity-ledger --output-dir {tmp_path} --dry-run" in err) is said


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
    runner = UpgradeRunner(
        FilesystemStateStore(root), [cast(Migration, Earlier()), cast(Migration, Fake())]
    )
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
    runner = UpgradeRunner(store, [cast(Migration, Fake())])
    applied = store.applied_migration_ids()
    _reset.write_text(json.dumps(_outcome(applied_before=sorted(applied))))
    ch._prepare_regime(runner, MigrationContext(corpus_root=root), _regime_args())
    assert list(runner.prepared) == ["0099_fake"]


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


def test_measure_refuses_a_scored_corpus_without_a_scoreboard(tmp_path: Path) -> None:
    """Nothing is applied when the corpus holds scored files no installed package can score:
    every gating number would read as unchanged."""
    from types import SimpleNamespace

    from podcast_scraper.extensions import use_extensions

    (tmp_path / "e.gi.json").write_text("{}", encoding="utf-8")
    applied = []
    prepared = SimpleNamespace(migration=SimpleNamespace(id="0099_x"))
    with use_extensions([]), pytest.raises(oh.HookRefused, match="no corpus scoreboard"):
        oh.measure(prepared, tmp_path, lambda: applied.append(1), None)  # type: ignore[arg-type]
    assert applied == []


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


M31 = "podcast_scraper.upgrade.migrations.m0031_named_voices_shown"


def test_the_regime_starts_at_0031() -> None:
    assert oh.in_regime("0031_named_voices_shown") and oh.in_regime("0099_fake")
    assert not oh.in_regime("0030_adfree_cut_resynced") and not oh.in_regime("0001_x")


def test_an_accept_file_with_an_extra_entry_is_not_the_accepted_one(
    _reset: Path, tmp_path: Path
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    PLAN[0]["inputs"] = {"a": "2"}
    with pytest.raises(oh.HookStopped):
        oh.prepare(Fake(), root, ["0099_fake"], {"0001"})
    _doc, sha = oh.load_outcome("0099_fake")
    good = json.loads(_accept(root, sha).read_text())
    extra = {"meta": "feeds/x/run_y/metadata/z.metadata.json", "kind": "prod_only", "cause": "c"}
    superset = _accept(root, sha, name="sup", entries=good["entries"] + [extra])
    with pytest.raises(oh.HookStopped, match="not the accepted one"):
        oh.prepare(Fake(), root, ["0099_fake"], {"0001"}, superset)


@pytest.mark.parametrize("body", [None, b"not json", b"[]"])
def test_an_unreadable_accept_file_stops_with_its_name(_reset: Path, tmp_path: Path, body) -> None:
    """Hook condition C4: missing, malformed or not an object: exit 4, nothing applied."""
    root = tmp_path / "corpus"
    root.mkdir()
    PLAN[0]["inputs"] = {"a": "2"}
    path = tmp_path / "accept.json"
    if body is not None:
        path.write_bytes(body)
    with pytest.raises(oh.HookStopped, match="accept file"):
        oh.prepare(Fake(), root, ["0099_fake"], {"0001"}, path)


def test_a_dry_run_writes_no_plan_diff_file(_reset: Path, tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    PLAN[0]["inputs"] = {"a": "2"}
    with pytest.raises(oh.HookStopped):
        oh.prepare(Fake(), root, ["0099_fake"], {"0001"}, write=False)
    assert list(root.rglob("*")) == []


def test_a_before_scoreboard_failure_of_any_kind_refuses(
    _reset: Path, tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    prep = oh.prepare(Fake(), root, ["0099_fake"], {"0001"})

    def boom(r):
        raise RuntimeError("bad corpus")

    monkeypatch.setattr(oh, "score_by_episode", boom)
    applied: List[int] = []
    with pytest.raises(oh.HookRefused, match="before-scoreboard failed"):
        oh.measure(prep, root, lambda: applied.append(1), None)
    assert applied == []


def test_the_runner_guards_an_empty_plan_that_writes(_reset: Path, tmp_path: Path) -> None:
    """The empty-plan path goes through apply_empty's write guard, not a bare apply."""
    from podcast_scraper.upgrade.runner import UpgradeRunner
    from podcast_scraper.upgrade.state import FilesystemStateStore

    root = tmp_path / "corpus"
    root.mkdir()
    PLAN[:] = []
    WRITE_ON_APPLY[:] = [root / Fake._receipts_file]
    runner = UpgradeRunner(FilesystemStateStore(root), [cast(Migration, Fake())])
    with pytest.raises(oh.HookOutcomeFailed, match="plan was empty but the apply wrote"):
        runner.run(MigrationContext(corpus_root=root), hold_lock=False)


def test_the_outcome_receipt_and_its_new_dirs_go_to_the_corpus_owner(tmp_path, monkeypatch):
    # Prod runs `upgrade run` as root: 0034's and 0035's outcome receipts were left root-owned.
    seen = []
    monkeypatch.setattr(oh, "match_corpus_owner", lambda root, paths: seen.append(list(paths)))
    out = tmp_path / oh.OUT_DIR / "0099_x.json"
    out.parent.mkdir(parents=True)
    out.write_text("{}")
    oh._own(tmp_path, out)
    assert seen == [[out, tmp_path / oh.OUT_DIR, tmp_path / ".podcast_scraper"]]
