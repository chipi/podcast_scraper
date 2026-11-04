"""``upgrade rerun``: an applied regime migration again, for named episodes only, measured."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast, Dict, List

import pytest

from podcast_scraper.upgrade import outcome_hook as oh
from podcast_scraper.upgrade.cli_handlers import parse_upgrade_argv
from podcast_scraper.upgrade.corpus_selection import episode_scope, select_served_artifacts
from podcast_scraper.upgrade.file_rewrite import append_receipts, write_with_backup
from podcast_scraper.upgrade.migration import Migration, MigrationContext, MigrationResult
from podcast_scraper.upgrade.runner import UpgradeRunner
from podcast_scraper.upgrade.state import FilesystemStateStore

pytestmark = [pytest.mark.unit]

EPS = ["e1", "e2", "e3", "e4"]
LEAK_TO: List[str] = []


def _meta(ep: str, run: str = "run_20260101-000000_a") -> str:
    return f"feeds/f/{run}/metadata/{ep}.metadata.json"


def _todo(root: Path) -> List[Path]:
    return [
        m
        for m in select_served_artifacts(root, ".metadata.json")[0]
        if json.loads(m.read_text()).get("todo")
    ]


def dry_run_report(root: Path) -> List[Dict[str, Any]]:
    """Read-only: one row per served episode still marked ``todo``."""
    return [
        {
            "meta": m.relative_to(root).as_posix(),
            "counts": {"fixed": 1},
            "inputs": {m.relative_to(root).as_posix(): hashlib.sha256(m.read_bytes()).hexdigest()},
        }
        for m in _todo(root)
    ]


class Fixer:
    """A regime migration that clears ``todo`` on every served episode it selects."""

    id = "0099_fixer"
    to_version = "9.9.9"
    _receipts_file = "fixer.jsonl"
    _backup_tag = "0099"

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        root = ctx.corpus_root
        targets = _todo(root) + [root / _meta(e) for e in LEAK_TO]
        receipts = []
        for m in targets:
            doc = json.loads(m.read_text())
            doc["todo"] = False
            receipts.append(write_with_backup(root, self._backup_tag, m, doc))
        append_receipts(root, self._receipts_file, {"migration": self.id}, receipts)
        rels = [m.relative_to(root).as_posix() for m in targets]
        return MigrationResult(
            self.id, applied=True, dry_run=False, message="", details={"episodes": rels}
        )


def _corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    for ep in EPS:
        p = root / _meta(ep)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"feed": {"feed_id": "f"}, "episode": {"episode_id": ep}, "todo": True})
        )
    return root


def _manifest(root: Path) -> Dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != ".podcast_scraper.lock"  # the corpus lock's own file
    }


@pytest.fixture
def applied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A corpus where 0099 was applied to e1 and e2 (its outcome planned e1..e4 then), and e3, e4
    came back from a reprocess marked ``todo`` again with the same bytes the outcome measured."""
    LEAK_TO[:] = []
    root = _corpus(tmp_path)
    plan = dry_run_report(root)
    out = tmp_path / "outcome.json"
    out.write_text(
        json.dumps(
            {
                "migration": "0099_fixer",
                "applied_before": [],
                "code_digest": "d",
                "episode_deltas": {},
                "plan": plan,
                "verdict": "GO",
            }
        )
    )
    monkeypatch.setattr(
        oh, "load_outcome", lambda mid, path=None: (json.loads(out.read_text()), "sha")
    )
    monkeypatch.setattr(oh, "migration_digest", lambda m: "d")
    monkeypatch.setattr(
        oh,
        "score_by_episode",
        lambda r: {
            m.relative_to(r).as_posix(): {
                "h": int(hashlib.sha256(m.read_bytes()).hexdigest()[:8], 16)
            }
            for m in select_served_artifacts(Path(r), ".metadata.json")[0]
        },
    )
    with episode_scope(root, [_meta("e1"), _meta("e2")]):
        Fixer().apply(MigrationContext(corpus_root=root))
    FilesystemStateStore(root).record_applied("0099_fixer", to_version="9.9.9", at="t")
    return root


def _runner(root: Path) -> UpgradeRunner:
    return UpgradeRunner(FilesystemStateStore(root), [cast(Migration, Fixer())])


def _rerun(root: Path, names: List[str], dry: bool = False, **kw: Any) -> MigrationResult:
    ctx = MigrationContext(corpus_root=root, dry_run=dry)
    return _runner(root).rerun("0099_fixer", names, ctx, stamp="S", **kw)


def test_a_scope_selects_only_its_episodes_and_no_scope_selects_all(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    assert len(select_served_artifacts(root, ".metadata.json")[0]) == 4
    with episode_scope(root, [_meta("e2")]):
        assert [p.name for p in select_served_artifacts(root, ".metadata.json")[0]] == [
            "e2.metadata.json"
        ]
    assert len(select_served_artifacts(root, ".metadata.json")[0]) == 4
    with pytest.raises(ValueError, match="empty"):
        with episode_scope(root, []):
            pass


@pytest.mark.parametrize(
    "names, why",
    [
        ([], "names no episode"),
        (["", "  "], "names no episode"),
        ([_meta("e3"), _meta("e3")], "more than once"),
        (["feeds/f/run_x/metadata/nope.metadata.json"], "not a served episode"),
    ],
)
def test_a_bad_episodes_file_is_refused_before_any_write(applied: Path, names, why) -> None:
    before = _manifest(applied)
    with pytest.raises(oh.HookRefused, match=why):
        _rerun(applied, names)
    assert _manifest(applied) == before


def test_a_superseded_copy_is_not_a_served_episode(applied: Path) -> None:
    newer = applied / _meta("e3", "run_20260201-000000_b")
    newer.parent.mkdir(parents=True)
    newer.write_text(json.dumps({"feed": {"feed_id": "f"}, "episode": {"episode_id": "e3"}}))
    with pytest.raises(oh.HookRefused, match="not a served episode"):
        _rerun(applied, [_meta("e3")], dry=True)


def test_a_dry_rerun_plans_only_named_planned_episodes_and_writes_nothing(applied: Path) -> None:
    before = _manifest(applied)
    res = _rerun(applied, [_meta("e1"), _meta("e3"), _meta("e4")], dry=True)
    assert res.details["episodes"] == [_meta("e3"), _meta("e4")]
    assert res.details["plan_diff"] == [] and not res.applied
    assert _manifest(applied) == before


def test_a_rerun_writes_its_episodes_and_keeps_the_apply_whole(applied: Path) -> None:
    before = _manifest(applied)
    ledger_files = {
        k: v
        for k, v in before.items()
        if k.startswith(".podcast_scraper/") and "upgrade-backups" not in k
    }
    _rerun(applied, [_meta("e3")])
    after = _manifest(applied)
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    assert _meta("e3") in changed and _meta("e4") not in changed
    # the apply's backups, receipts and ledger: byte-identical
    for k, v in before.items():
        if k == _meta("e3"):
            continue
        assert after[k] == v, k
    assert {k: after[k] for k in ledger_files} == ledger_files
    new = sorted(set(after) - set(before))
    assert "fixer.rerun-S.jsonl" in new
    assert any(k.startswith(".podcast_scraper/upgrade-backups/0099.rerun-S/") for k in new)
    receipt = json.loads(
        (applied / ".podcast_scraper/upgrade-outcomes/0099_fixer.rerun-S.json").read_text()
    )
    assert receipt["verdict"] == "PASS" and receipt["rerun_scope"] == [_meta("e3")]
    assert receipt["snapshot"] == "none"
    assert "0099_fixer" in FilesystemStateStore(applied).applied_migration_ids()


def test_a_changed_episode_stops_at_the_rerun_plan_diff(applied: Path) -> None:
    p = applied / _meta("e3")
    p.write_text(p.read_text().replace('"todo": true', '"todo": true, "new": 1'))
    with pytest.raises(oh.HookStopped, match="differs"):
        _rerun(applied, [_meta("e3")])
    diff_dir = applied / ".podcast_scraper/upgrade-outcomes"
    assert sorted(x.name for x in diff_dir.iterdir()) == ["0099_fixer.rerun-S.plan-diff.json"]
    assert json.loads(p.read_text())["todo"] is True


def test_an_apply_that_writes_outside_the_scope_fails_loudly(applied: Path) -> None:
    LEAK_TO[:] = ["e4"]
    with pytest.raises(oh.HookOutcomeFailed, match="outside the scope") as exc:
        _rerun(applied, [_meta("e3")])
    assert "fixer.rerun-S.jsonl" in str(exc.value) and "0099.rerun-S" in str(exc.value)


def test_an_empty_scoped_plan_writes_nothing(applied: Path) -> None:
    before = _manifest(applied)
    res = _rerun(applied, [_meta("e1")])
    assert not res.applied and res.details["episodes"] == []
    assert _manifest(applied) == before


def test_an_unapplied_or_unmeasured_or_unredirectable_migration_is_refused(
    applied: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(oh, "migration_digest", lambda m: "other")
    with pytest.raises(oh.HookRefused, match="not the code that was measured"):
        _rerun(applied, [_meta("e3")])
    monkeypatch.setattr(oh, "migration_digest", lambda m: "d")

    class Plain(Fixer):
        _backup_tag = ""

    with pytest.raises(oh.HookRefused, match="kept apart"):
        UpgradeRunner(FilesystemStateStore(applied), [cast(Migration, Plain())]).rerun(
            "0099_fixer", [_meta("e3")], MigrationContext(corpus_root=applied), stamp="S"
        )

    class Old(Fixer):
        id = "0029_old"

    with pytest.raises(oh.HookRefused, match="below the regime"):
        UpgradeRunner(FilesystemStateStore(applied), [cast(Migration, Old())]).rerun(
            "0029_old", [_meta("e3")], MigrationContext(corpus_root=applied), stamp="S"
        )
    fresh = applied.parent / "fresh"
    fresh.mkdir()
    for ep in ("e3",):
        p = fresh / _meta(ep)
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"todo": True}))
    with pytest.raises(oh.HookRefused, match="not applied here"):
        _rerun(fresh, [_meta("e3")])


def test_the_cli_parses_rerun() -> None:
    args = parse_upgrade_argv(
        [
            "rerun",
            "--migration",
            "0043_x",
            "--episodes",
            "eps.txt",
            "--dry-run",
            "--corpus-dir",
            "/c",
        ]
    )
    assert args.upgrade_subcommand == "rerun" and args.migration == "0043_x"
    assert args.episodes == Path("eps.txt") and args.dry_run and args.accept_plan_diff is None


def test_a_migration_whose_plan_ignores_the_scope_is_refused(
    applied: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    def globbing(root: Path) -> List[Dict[str, Any]]:
        return [
            {"meta": m.relative_to(root).as_posix(), "counts": {"fixed": 1}, "inputs": {}}
            for m in sorted(root.rglob("*.metadata.json"))
            if json.loads(m.read_text()).get("todo")
        ]

    monkeypatch.setattr(sys.modules[__name__], "dry_run_report", globbing)
    before = _manifest(applied)
    with pytest.raises(oh.HookRefused, match="outside the scope"):
        _rerun(applied, [_meta("e3")])
    assert _manifest(applied) == before


def test_a_written_path_is_owned_after_normalisation() -> None:
    meta = "feeds/f/run_a/metadata/0006 - Greece vs. Persia_ x.metadata.json"
    inside = "feeds/f/run_a/../run_a/transcripts/0006 - Greece vs. Persia_ x.adfree.segments.json"
    other_run = "feeds/f/run_a/../run_b/transcripts/0006 - Greece vs. Persia_ x.txt"
    other_ep = "feeds/f/run_a/metadata/0007 - y.kg.json"
    assert oh.files_outside([meta], [inside]) == []
    assert oh.files_outside([meta], [other_run, other_ep]) == sorted([other_run, other_ep])


def test_a_held_lock_refuses_a_real_rerun_and_not_a_dry_one(
    applied: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import contextlib

    import podcast_scraper.utils.corpus_lock as lock

    @contextlib.contextmanager
    def held(root: Path):
        raise RuntimeError("another job holds the corpus lock")
        yield

    monkeypatch.setattr(lock, "corpus_parent_lock", held)
    before = _manifest(applied)
    assert _rerun(applied, [_meta("e3")], dry=True).details["episodes"] == [_meta("e3")]
    with pytest.raises(RuntimeError, match="holds the corpus lock"):
        _rerun(applied, [_meta("e3")])
    assert _manifest(applied) == before
