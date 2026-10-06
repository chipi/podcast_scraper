"""m0021: fetch each show's declared language and backfill it (#2173 / slice S0.1b).

The migration fetches, so every test here stubs ``_fetch_language``. What is actually under test
is the CONVERGENCE contract — which states keep the migration pending and which do not — because
that is what decides whether an automatic post-deploy upgrade ever finishes the job.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast, Dict, Optional, Tuple

import pytest

from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations import m0021_backfill_feed_language as m0021
from podcast_scraper.upgrade.migrations.m0021_backfill_feed_language import (
    BackfillFeedLanguageMigration,
)

pytestmark = pytest.mark.unit


def _episode(root: Path, feed_id: str, episode_id: str, *, url: Optional[str] = None) -> Path:
    """One served episode's metadata, in the layout ``select_served_artifacts`` walks."""
    meta_dir = root / "feeds" / feed_id / "run_20260101-000000" / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    path = meta_dir / f"{episode_id}.metadata.json"
    path.write_text(
        json.dumps(
            {
                "feed": {
                    "feed_id": feed_id,
                    "title": feed_id,
                    "url": f"https://example.com/{feed_id}.xml" if url is None else url,
                    # What every pre-#2172 artifact carries: the RUN CONFIG, written back out.
                    "language": "en-us",
                },
                "episode": {"episode_id": episode_id, "title": episode_id},
                "content": {"transcript_file_path": f"transcripts/{episode_id}.txt"},
                "schema_version": "1.0",
            }
        ),
        encoding="utf-8",
    )
    return path


def _stub_fetch(monkeypatch: pytest.MonkeyPatch, answers: Dict[str, Any]) -> None:
    """``answers`` maps a feed's URL to ``language_raw``, or to an Exception-ish error string."""

    def fake(url: str, timeout: float) -> Tuple[Optional[str], str]:
        value = answers.get(url, "__unset__")
        if value == "__unset__":
            return None, "RuntimeError: no stub for this url"
        if isinstance(value, str) and value.startswith("ERR:"):
            return None, value[4:]
        return value, ""

    monkeypatch.setattr(m0021, "_fetch_language", fake)


def _load(path: Path) -> Dict[str, Any]:
    return cast(Dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _ctx(root: Path, **kw: Any) -> MigrationContext:
    return MigrationContext(corpus_root=root, **kw)


class TestItCompletesByItself:
    """The property the first draft got wrong: a migration must be able to finish.

    ``runner.py:128`` records a migration only ``if result.applied``, so anything that cannot
    return True stays pending and re-runs on every upgrade for ever.
    """

    def test_a_clean_run_is_recorded(self, tmp_path: Path, monkeypatch) -> None:
        _episode(tmp_path, "p01", "p01_e01")
        _stub_fetch(monkeypatch, {"https://example.com/p01.xml": "es-ES"})

        res = BackfillFeedLanguageMigration().apply(_ctx(tmp_path))

        assert res.applied is True
        assert res.details["complete"] is True

    def test_a_corpus_already_correct_is_COMPLETE_not_pending(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Zero changes is a finished migration.

        The first draft used ``applied=bool(updated)``, so a corpus already carrying the right
        languages returned False and would have stayed pending for ever.
        """
        _episode(tmp_path, "p01", "p01_e01")
        _stub_fetch(monkeypatch, {"https://example.com/p01.xml": "es-ES"})
        mig = BackfillFeedLanguageMigration()
        mig.apply(_ctx(tmp_path))

        second = mig.apply(_ctx(tmp_path))
        assert second.details["updated"] == 0
        assert second.details["already_correct"] == 1
        assert second.applied is True, "nothing to do is still complete"

    def test_a_show_with_no_url_does_not_block_completion(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """It can never be fetched, and the registered CI migration fixture is exactly this
        shape — so treating it as retryable would mean never completing in CI."""
        _episode(tmp_path, "p01", "p01_e01")
        _episode(tmp_path, "p02", "p02_e01", url="")
        _stub_fetch(monkeypatch, {"https://example.com/p01.xml": "es-ES"})

        res = BackfillFeedLanguageMigration().apply(_ctx(tmp_path))

        assert res.details["shows_without_url"] == ["p02"]
        assert res.applied is True, "an unfetchable show is a final answer, not a retry"

    def test_a_feed_declaring_no_language_does_not_block_completion(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _episode(tmp_path, "p01", "p01_e01")
        _stub_fetch(monkeypatch, {"https://example.com/p01.xml": None})

        res = BackfillFeedLanguageMigration().apply(_ctx(tmp_path))

        assert res.details["shows_declaring_no_language"] == ["p01"]
        assert res.applied is True


class TestItConvergesAcrossRuns:
    def test_a_fetch_failure_keeps_it_pending(self, tmp_path: Path, monkeypatch) -> None:
        _episode(tmp_path, "p01", "p01_e01")
        _stub_fetch(monkeypatch, {"https://example.com/p01.xml": "ERR:ConnectTimeout: boom"})

        res = BackfillFeedLanguageMigration().apply(_ctx(tmp_path))

        assert res.applied is False
        assert "ConnectTimeout" in res.details["fetch_failures"]["p01"]
        assert "INCOMPLETE" in res.message

    def test_a_partial_run_writes_what_it_resolved_then_finishes_next_time(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """14-of-20 is the realistic failure. The resolved shows must land, and the rest must be
        picked up on the next upgrade rather than lost."""
        ok = _episode(tmp_path, "p01", "p01_e01")
        flaky = _episode(tmp_path, "p02", "p02_e01")
        mig = BackfillFeedLanguageMigration()

        _stub_fetch(
            monkeypatch,
            {
                "https://example.com/p01.xml": "es-ES",
                "https://example.com/p02.xml": "ERR:ReadTimeout: slow",
            },
        )
        first = mig.apply(_ctx(tmp_path))
        assert first.applied is False
        assert _load(ok)["episode"]["language"] == "es", "the resolved show still landed"
        assert "language" not in _load(flaky)["episode"]

        # Next upgrade: the publisher is back.
        _stub_fetch(
            monkeypatch,
            {
                "https://example.com/p01.xml": "es-ES",
                "https://example.com/p02.xml": "de-DE",
            },
        )
        second = mig.apply(_ctx(tmp_path))
        assert second.applied is True
        assert _load(flaky)["episode"]["language"] == "de"
        assert second.details["per_feed"]["p01"]["already"] == 1, "p01 was not rewritten"


class TestWhatItWrites:
    def test_all_five_fields(self, tmp_path: Path, monkeypatch) -> None:
        ep = _episode(tmp_path, "p01", "p01_e01")
        _stub_fetch(monkeypatch, {"https://example.com/p01.xml": "pt_BR"})

        BackfillFeedLanguageMigration().apply(_ctx(tmp_path))

        doc = _load(ep)
        assert doc["feed"]["language"] == "pt", "normalized, replacing the run-config value"
        assert doc["feed"]["language_raw"] == "pt_BR", "the publisher's original, verbatim"
        assert doc["feed"]["language_source"] == "rss"
        assert doc["episode"]["language"] == "pt"
        assert doc["episode"]["language_source"] == "rss"

    def test_one_fetch_covers_every_episode_under_the_show(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        eps = [_episode(tmp_path, "p01", f"p01_e0{i}") for i in range(1, 5)]
        calls: list[str] = []

        def counting(url: str, timeout: float) -> Tuple[Optional[str], str]:
            calls.append(url)
            return "it-IT", ""

        monkeypatch.setattr(m0021, "_fetch_language", counting)
        res = BackfillFeedLanguageMigration().apply(_ctx(tmp_path))

        assert len(calls) == 1, f"one fetch per SHOW, not per episode: {calls}"
        assert res.details["per_feed"]["p01"]["updated"] == 4
        for ep in eps:
            assert _load(ep)["episode"]["language"] == "it"

    def test_an_unmapped_show_is_left_exactly_as_it_was(self, tmp_path: Path, monkeypatch) -> None:
        untouched = _episode(tmp_path, "p02", "p02_e01", url="")
        before = _load(untouched)
        _episode(tmp_path, "p01", "p01_e01")
        _stub_fetch(monkeypatch, {"https://example.com/p01.xml": "es-ES"})

        BackfillFeedLanguageMigration().apply(_ctx(tmp_path))

        assert _load(untouched) == before, "nothing is ever guessed"


class TestDryRun:
    def test_it_reports_the_plan_and_writes_nothing(self, tmp_path: Path, monkeypatch) -> None:
        ep = _episode(tmp_path, "p01", "p01_e01")
        before = _load(ep)
        _stub_fetch(monkeypatch, {"https://example.com/p01.xml": "es-ES"})

        res = BackfillFeedLanguageMigration().apply(_ctx(tmp_path, dry_run=True))

        assert res.dry_run is True
        assert res.applied is False, "a dry run is never recorded"
        assert res.details["updated"] == 1, "the plan is still reported"
        assert _load(ep) == before


class TestVerify:
    def test_it_passes_once_applied(self, tmp_path: Path, monkeypatch) -> None:
        _episode(tmp_path, "p01", "p01_e01")
        _stub_fetch(monkeypatch, {"https://example.com/p01.xml": "es-ES"})
        mig = BackfillFeedLanguageMigration()
        mig.apply(_ctx(tmp_path))

        ok, msg = mig.verify(_ctx(tmp_path))
        assert ok, msg

    def test_it_can_fail(self, tmp_path: Path) -> None:
        """A verify that cannot fail is not a verify. A half-written artifact — language_raw
        present, the episode pair missing — is what a crash mid-write would leave."""
        ep = _episode(tmp_path, "p01", "p01_e01")
        doc = _load(ep)
        doc["feed"]["language_raw"] = "es-ES"
        ep.write_text(json.dumps(doc), encoding="utf-8")

        ok, msg = BackfillFeedLanguageMigration().verify(_ctx(tmp_path))
        assert not ok and "partial backfill" in msg

    def test_it_does_not_re_fetch(self, tmp_path: Path, monkeypatch) -> None:
        """Verification reads the corpus only.

        Going back to the network would fail on a publisher outage long after the migration was
        correctly applied — reporting someone else's downtime as our defect.
        """
        _episode(tmp_path, "p01", "p01_e01")
        _stub_fetch(monkeypatch, {"https://example.com/p01.xml": "es-ES"})
        mig = BackfillFeedLanguageMigration()
        mig.apply(_ctx(tmp_path))

        def explode(url: str, timeout: float) -> Tuple[Optional[str], str]:
            raise AssertionError("verify must not fetch")

        monkeypatch.setattr(m0021, "_fetch_language", explode)
        ok, _msg = mig.verify(_ctx(tmp_path))
        assert ok


class TestRegistration:
    def test_it_is_registered_and_ordered_after_0014(self) -> None:
        """An unregistered migration never runs — the quietest possible failure.

        NO POSITIONAL ASSERTION, and that is the third version of this test. It first asserted
        this migration was LAST and went red when something followed it. It was then changed to
        assert adjacency to 0014 — "repeating that here would just hand the same trap to 0016",
        said the docstring — and on 2026-10-03 `main` landed m0015-m0018 between them and it went
        red again, for a change that says nothing about this migration. Twice is enough: position
        is not this migration's property, it is a consequence of its id and the registry's sort.

        So what is asserted is the pair of facts that ARE its own: it is registered, and the
        registry is ordered by id. Where it lands then follows, and a sibling arriving anywhere in
        the sequence cannot make this red.
        """
        from podcast_scraper.upgrade.registry import get_migrations

        ids = [m.id for m in get_migrations()]
        assert "0021_backfill_feed_language" in ids
        assert ids == sorted(ids), "registry order is lexicographic by id"
        assert len(ids) == len(set(ids)), (
            "two migrations share an id — the ledger records the id as applied, so a duplicate "
            "makes a corpus's migration history ambiguous about which one ran. This is the "
            "collision that forced this migration from 0011 to 0015 to 0019 to 0021."
        )


class TestItNeverClobbersAnOperatorOverride:
    """An episode whose language came from `language_override` is OFF LIMITS to this migration.

    Found in the whole-branch review, and it only became reachable once S0.2 shipped the override:
    the migration was written when `rss` and `profile_default` were the only two sources there
    were, so `_plan_for_episode` wants `language_source == "rss"` unconditionally.

    Two distinct failures came out of that, both against the SAME corpus shape — a feed whose
    publisher declares the wrong tag and an operator who corrected it, which is the entire reason
    the override exists:

    * `apply` would overwrite `override` back to `rss` and the corrected language back to the
      publisher's wrong one. The migration would undo the correction, on its own, at deploy time.
    * `verify` would count every such episode as "a partial backfill" — a red verify describing
      a corpus that is exactly right.

    The precedence in `resolve_episode_language` already says the override outranks the feed tag.
    A backfill that reverses that precedence is not a backfill.
    """

    def _overridden(self, root: Path, feed_id: str, episode_id: str) -> Path:
        """An episode as the pipeline writes it under a per-feed override.

        `language_raw` is the OVERRIDE's value, not the publisher's — that is what
        `resolve_episode_language` returns as the raw when the override wins, and it is why
        `verify`'s "did this show get backfilled" test (`language_raw` is set) sees these
        episodes at all.
        """
        path = _episode(root, feed_id, episode_id)
        payload = _load(path)
        payload["feed"].update(
            {"language": "pt", "language_raw": "pt", "language_source": "override"}
        )
        payload["episode"].update({"language": "pt", "language_source": "override"})
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def test_apply_leaves_the_override_alone(self, tmp_path: Path, monkeypatch) -> None:
        path = self._overridden(tmp_path, "f1", "e1")
        _stub_fetch(monkeypatch, {"https://example.com/f1.xml": "es-ES"})

        BackfillFeedLanguageMigration().apply(_ctx(tmp_path))

        payload = _load(path)
        assert payload["feed"]["language"] == "pt", "the operator's correction must survive"
        assert payload["feed"]["language_source"] == "override"
        assert payload["episode"]["language"] == "pt"
        assert payload["episode"]["language_source"] == "override"

    def test_apply_counts_it_as_skipped_not_as_already_correct(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Reported distinctly, because "I did not touch these" and "these were already right"
        are different facts about the corpus and an operator reads the difference."""
        self._overridden(tmp_path, "f1", "e1")
        _episode(tmp_path, "f1", "e2")
        _stub_fetch(monkeypatch, {"https://example.com/f1.xml": "es-ES"})

        result = BackfillFeedLanguageMigration().apply(_ctx(tmp_path))

        assert result.details["overridden"] == 1
        assert result.details["updated"] == 1
        assert result.details["already_correct"] == 0
        assert "1 override" in result.message

    def test_apply_still_completes(self, tmp_path: Path, monkeypatch) -> None:
        """An override is a final answer, so it must not hold the migration pending for ever."""
        self._overridden(tmp_path, "f1", "e1")
        _stub_fetch(monkeypatch, {"https://example.com/f1.xml": "es-ES"})

        result = BackfillFeedLanguageMigration().apply(_ctx(tmp_path))

        assert result.applied is True
        assert result.details["complete"] is True

    def test_verify_does_not_call_it_a_partial_backfill(self, tmp_path: Path, monkeypatch) -> None:
        self._overridden(tmp_path, "f1", "e1")
        _stub_fetch(monkeypatch, {"https://example.com/f1.xml": "es-ES"})
        mig = BackfillFeedLanguageMigration()
        mig.apply(_ctx(tmp_path))

        ok, msg = mig.verify(_ctx(tmp_path))
        assert ok, msg

    def test_verify_is_clean_on_an_override_that_was_NEVER_applied(self, tmp_path: Path) -> None:
        """The same corpus with no migration run at all — the state a fresh deploy inherits."""
        self._overridden(tmp_path, "f1", "e1")
        ok, msg = BackfillFeedLanguageMigration().verify(_ctx(tmp_path))
        assert ok, msg

    def test_a_non_overridden_episode_under_the_SAME_show_is_still_backfilled(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The skip is per EPISODE, not per show: the override is recorded on the artifact, and
        skipping the whole show would strand every episode processed before it was set."""
        overridden = self._overridden(tmp_path, "f1", "e1")
        plain = _episode(tmp_path, "f1", "e2")
        _stub_fetch(monkeypatch, {"https://example.com/f1.xml": "es-ES"})

        BackfillFeedLanguageMigration().apply(_ctx(tmp_path))

        assert _load(overridden)["feed"]["language"] == "pt"
        assert _load(plain)["feed"]["language"] == "es"
        assert _load(plain)["feed"]["language_source"] == "rss"


class TestEveryWriteIsBackedUpAndUndoable:
    """The gap this closes, found by review on 2026-10-03.

    Every write in this migration went through a bare ``path.write_text``. It replaces FIVE fields
    on every served episode in the corpus with a value fetched over the network from a third party
    — and publishers declare the wrong ``<language>`` routinely, which is the entire reason
    ``language_override`` exists. So the one case the migration is most likely to get wrong was
    also the one with no way back: the previous value was gone, and the only record that it had
    ever been different was the new value itself.

    ``m0012`` and ``m0014`` already used ``file_rewrite``'s backup + receipt + ``undo``. This one
    did not, and nothing failed — a migration with no undo looks exactly like one with an undo
    nobody has called.

    NOT A HYPOTHETICAL RECOVERY ROUTE. ``verify`` reads the corpus and never re-fetches, so once a
    wrong tag is written, verify AGREES with it. There is no second opinion in the system; the
    backup is the second opinion.
    """

    @staticmethod
    def _backups(root: Path) -> list:
        return sorted(
            p.name
            for p in (root / ".podcast_scraper" / "upgrade-backups" / m0021.BACKUP_TAG).rglob(
                "*.json"
            )
        )

    def test_the_original_bytes_are_kept_and_a_receipt_names_the_file(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        path = _episode(tmp_path, "p10", "p10_e01")
        before = path.read_bytes()
        _stub_fetch(monkeypatch, {"https://example.com/p10.xml": "es-ES"})

        res = BackfillFeedLanguageMigration().apply(_ctx(tmp_path))

        assert res.details["updated"] == 1
        assert res.details["files_written"] == 1, "the write was not receipted"
        assert self._backups(tmp_path) == ["p10_e01.metadata.json"]
        backup = tmp_path / ".podcast_scraper" / "upgrade-backups" / m0021.BACKUP_TAG
        kept = next(backup.rglob("p10_e01.metadata.json"))
        assert kept.read_bytes() == before, "the backup is not the pre-write content"
        receipts = (tmp_path / m0021.RECEIPTS_FILE).read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in receipts]
        header = [r for r in rows if r.get("kind") == "header"]
        assert header and header[0]["feed_id"] == "p10" and header[0]["language"] == "es"
        assert [r["relpath"] for r in rows if "relpath" in r] == [str(path.relative_to(tmp_path))]

    def test_undo_puts_the_corpus_back_byte_for_byte(self, tmp_path: Path, monkeypatch) -> None:
        """The whole point. Asserted on BYTES, not on the five fields: a restore that rewrote the
        file with the right language and a different key order or encoding would make every
        migrated artifact differ from its own regenerated form."""
        paths = [
            _episode(tmp_path, "p10", "p10_e01"),
            _episode(tmp_path, "p10", "p10_e02"),
            _episode(tmp_path, "p11", "p11_e01"),
        ]
        before = {p: p.read_bytes() for p in paths}
        _stub_fetch(
            monkeypatch,
            {"https://example.com/p10.xml": "es-ES", "https://example.com/p11.xml": "it-IT"},
        )
        BackfillFeedLanguageMigration().apply(_ctx(tmp_path))
        assert all(p.read_bytes() != before[p] for p in paths), "setup: nothing was rewritten"

        restored, refused = m0021.undo(tmp_path)

        assert (restored, refused) == (3, [])
        assert {p: p.read_bytes() for p in paths} == before

    def test_undo_REFUSES_a_file_changed_since_the_migration_wrote_it(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Because what is being undone is this migration's write, and nothing here can know what
        a later change meant. Refusing is reported; overwriting would be silent data loss by the
        tool whose job is to prevent it."""
        path = _episode(tmp_path, "p10", "p10_e01")
        _stub_fetch(monkeypatch, {"https://example.com/p10.xml": "es-ES"})
        BackfillFeedLanguageMigration().apply(_ctx(tmp_path))

        payload = _load(path)
        payload["episode"]["title"] = "edited by someone else afterwards"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        restored, refused = m0021.undo(tmp_path)

        assert restored == 0
        assert len(refused) == 1 and "changed since" in refused[0]
        assert _load(path)["episode"]["title"] == "edited by someone else afterwards"

    def test_a_dry_run_leaves_no_backup_and_no_receipt(self, tmp_path: Path, monkeypatch) -> None:
        _episode(tmp_path, "p10", "p10_e01")
        _stub_fetch(monkeypatch, {"https://example.com/p10.xml": "es-ES"})

        res = BackfillFeedLanguageMigration().apply(_ctx(tmp_path, dry_run=True))

        assert res.details["updated"] == 1, "the plan is still reported"
        assert res.details["files_written"] == 0
        assert not (tmp_path / m0021.RECEIPTS_FILE).exists()
        assert not (tmp_path / ".podcast_scraper" / "upgrade-backups").exists()

    def test_receipts_are_appended_PER_SHOW_so_a_crash_loses_one_show_at_most(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """One append at the end of the whole migration would mean a crash on show 12 of 20 left
        eleven shows rewritten with no receipt naming them — and ``undo`` restores only what a
        receipt points at. One header per show is what makes the loss bounded."""
        for feed, ep in (("p10", "p10_e01"), ("p11", "p11_e01"), ("p12", "p12_e01")):
            _episode(tmp_path, feed, ep)
        _stub_fetch(
            monkeypatch,
            {
                "https://example.com/p10.xml": "es-ES",
                "https://example.com/p11.xml": "it-IT",
                "https://example.com/p12.xml": "pt-BR",
            },
        )
        BackfillFeedLanguageMigration().apply(_ctx(tmp_path))

        rows = [
            json.loads(line)
            for line in (tmp_path / m0021.RECEIPTS_FILE).read_text(encoding="utf-8").splitlines()
        ]
        headers = [r for r in rows if r.get("kind") == "header"]
        assert [h["feed_id"] for h in headers] == ["p10", "p11", "p12"]
        assert [h["language"] for h in headers] == ["es", "it", "pt"]

    def test_an_episode_left_to_the_OVERRIDE_is_never_backed_up_either(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """It was never written, so a backup of it would be a receipt for a change that did not
        happen — and ``undo`` would then "restore" a file the migration never touched."""
        path = _episode(tmp_path, "p10", "p10_e01")
        payload = _load(path)
        payload["feed"]["language_source"] = "override"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _stub_fetch(monkeypatch, {"https://example.com/p10.xml": "es-ES"})

        res = BackfillFeedLanguageMigration().apply(_ctx(tmp_path))

        assert res.details["overridden"] == 1
        assert res.details["files_written"] == 0
        assert not (tmp_path / m0021.RECEIPTS_FILE).exists()


class TestOverridesJsonOutranksThePublisher:
    """A feed-level language in ``overrides.json`` (#2283) wins over the fetched ``<language>``.

    Before this, m0021 read only the ``language_source`` already on each artifact, so an override
    set through the API — the remedy for a publisher's WRONG tag — did not stop the migration from
    stamping that wrong tag onto every existing episode (English-path audit, 2026-10-05).
    """

    def _override(self, root: Path, url: str, language: str) -> None:
        from podcast_scraper import overrides as ov

        ov.set_feed_fields(root, url, ov.FeedFields(language=language))

    def test_the_override_is_written_and_the_publisher_is_never_fetched(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        path = _episode(tmp_path, "p01", "p01_e01")
        self._override(tmp_path, "https://example.com/p01.xml", "en")

        def explode(url: str, timeout: float) -> Tuple[Optional[str], str]:
            raise AssertionError(f"fetched {url} although overrides.json sets its language")

        monkeypatch.setattr(m0021, "_fetch_language", explode)

        res = BackfillFeedLanguageMigration().apply(_ctx(tmp_path))

        payload = _load(path)
        assert payload["feed"]["language"] == "en"
        assert payload["feed"]["language_source"] == "override"
        assert payload["episode"]["language"] == "en"
        assert payload["episode"]["language_source"] == "override"
        assert res.applied is True
        assert res.details["per_feed"]["p01"]["source"] == "override"

    def test_a_wrong_publisher_tag_never_reaches_an_overridden_show(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The case the gap was about: the RSS says Spanish, the operator says English."""
        overridden = _episode(tmp_path, "p01", "p01_e01")
        untouched = _episode(tmp_path, "p02", "p02_e01")
        self._override(tmp_path, "https://example.com/p01.xml", "en")
        _stub_fetch(
            monkeypatch,
            {"https://example.com/p01.xml": "es-ES", "https://example.com/p02.xml": "es-ES"},
        )

        BackfillFeedLanguageMigration().apply(_ctx(tmp_path))

        assert _load(overridden)["episode"]["language"] == "en"
        assert _load(untouched)["episode"]["language"] == "es", "other shows still backfill"

    def test_an_override_write_is_undoable_like_any_other(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        path = _episode(tmp_path, "p01", "p01_e01")
        before = path.read_text(encoding="utf-8")
        self._override(tmp_path, "https://example.com/p01.xml", "en")
        monkeypatch.setattr(m0021, "_fetch_language", lambda url, timeout: (None, "unused"))

        BackfillFeedLanguageMigration().apply(_ctx(tmp_path))
        restored, refused = m0021.undo(tmp_path)

        assert (restored, refused) == (1, [])
        assert path.read_text(encoding="utf-8") == before

    def test_a_dry_run_reports_the_override_and_writes_nothing(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        path = _episode(tmp_path, "p01", "p01_e01")
        before = path.read_text(encoding="utf-8")
        self._override(tmp_path, "https://example.com/p01.xml", "en")

        res = BackfillFeedLanguageMigration().apply(_ctx(tmp_path, dry_run=True))

        assert res.details["updated"] == 1
        assert "1 show(s) set from overrides.json" in res.message
        assert path.read_text(encoding="utf-8") == before

    def test_a_broken_overrides_file_stops_the_migration(self, tmp_path: Path, monkeypatch) -> None:
        """Fail hard: applying the publisher's tag because the corrections were unreadable is the
        very failure this class exists to prevent."""
        _episode(tmp_path, "p01", "p01_e01")
        (tmp_path / "overrides.json").write_text("{not json", encoding="utf-8")
        _stub_fetch(monkeypatch, {"https://example.com/p01.xml": "es-ES"})

        with pytest.raises(ValueError):
            BackfillFeedLanguageMigration().apply(_ctx(tmp_path))


class TestTheFetchItself:
    """Not stubbed: the real ``_fetch_language`` against a host that refuses httpx's default UA."""

    def test_it_sends_a_user_agent_a_picky_host_accepts(self, monkeypatch) -> None:
        import httpx

        seen = []

        def buzzsprout_like(request: httpx.Request) -> httpx.Response:
            ua = request.headers.get("user-agent", "")
            seen.append(ua)
            if ua.startswith("python-httpx"):
                return httpx.Response(403)
            return httpx.Response(
                200,
                content=b"<rss><channel><language>en-gb</language></channel></rss>",
            )

        real_client = httpx.Client

        def client_with_mock_transport(*args, **kwargs):
            return real_client(*args, transport=httpx.MockTransport(buzzsprout_like), **kwargs)

        monkeypatch.setattr(httpx, "Client", client_with_mock_transport)

        raw, err = m0021._fetch_language("https://rss.example.com/1.rss", 5.0)

        assert (raw, err) == ("en-gb", "")
        assert seen and not seen[0].startswith("python-httpx")
