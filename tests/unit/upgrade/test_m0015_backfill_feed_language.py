"""m0015: fetch each show's declared language and backfill it (#2173 / slice S0.1b).

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
from podcast_scraper.upgrade.migrations import m0015_backfill_feed_language as m0015
from podcast_scraper.upgrade.migrations.m0015_backfill_feed_language import (
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

    monkeypatch.setattr(m0015, "_fetch_language", fake)


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

        monkeypatch.setattr(m0015, "_fetch_language", counting)
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

        monkeypatch.setattr(m0015, "_fetch_language", explode)
        ok, _msg = mig.verify(_ctx(tmp_path))
        assert ok


class TestRegistration:
    def test_it_is_registered_and_ordered_after_0014(self) -> None:
        """An unregistered migration never runs — the quietest possible failure.

        Position asserted RELATIVE to 0014, not as "last". 0014's test asserted last and went
        red the moment this migration landed, for a change that said nothing about it; repeating
        that here would just hand the same trap to 0016.
        """
        from podcast_scraper.upgrade.registry import get_migrations

        ids = [m.id for m in get_migrations()]
        assert "0015_backfill_feed_language" in ids
        assert ids == sorted(ids), "registry order is lexicographic by id"
        assert (
            ids.index("0015_backfill_feed_language")
            == ids.index("0014_eponymous_hosts_restored") + 1
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
