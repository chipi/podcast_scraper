"""gi-repair's selection-by-identity path (2026-08-31, F3).

``--episode-ids`` + ``--force-healthy`` exist so the SAME episode can be re-derived under two
configurations and diffed — the only way to measure what a prompt / model / rater change does.
The default work-list selects by DAMAGE (legacy placeholders), which cannot express that,
because a healthy artifact is exactly what you want to re-derive when comparing.

Each test here pins a failure that shipped in the first cut of that feature, every one of
which was a silent success: zero matches exited 0 with VERDICT PASS, dry-run promised work the
real run refused, and an unresolvable corpus returned "nothing to do" instead of saying it
could not tell.
"""

from __future__ import annotations

import json

import pytest

from podcast_scraper.gi import repair as repair_mod
from podcast_scraper.gi.corpus import (
    find_gi_artifacts_for_episode_ids,
    LEGACY_PLACEHOLDER_INSIGHT_TEXT,
)


def _insight(text: str) -> dict:
    return {"type": "Insight", "id": "i1", "properties": {"text": text}}


def _write_pair(corpus, feed, run, episode_id, *, placeholder: bool, idx: int = 1):
    """Write a metadata + gi.json pair under feeds/<feed>/<run>/metadata/."""
    meta_dir = corpus / "feeds" / feed / run / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{idx:04d} - Ep_{run}"
    (meta_dir / f"{stem}.metadata.json").write_text(
        json.dumps({"episode": {"episode_id": episode_id, "guid": episode_id}}), encoding="utf-8"
    )
    gi = meta_dir / f"{stem}.gi.json"
    text = LEGACY_PLACEHOLDER_INSIGHT_TEXT if placeholder else "A real, healthy insight."
    gi.write_text(
        json.dumps({"episode_id": episode_id, "nodes": [_insight(text)], "edges": []}),
        encoding="utf-8",
    )
    return gi


class TestSelectionByIdentity:
    def test_selects_the_requested_episode(self, tmp_path):
        want = _write_pair(tmp_path, "f1", "run_20260814-055303", "ep-1", placeholder=False)
        _write_pair(tmp_path, "f1", "run_20260814-055303", "ep-2", placeholder=False, idx=2)

        got = find_gi_artifacts_for_episode_ids(tmp_path, ["ep-1"])
        assert [p for p, _ in got] == [want]

    def test_unknown_id_yields_nothing(self, tmp_path):
        _write_pair(tmp_path, "f1", "run_20260814-055303", "ep-1", placeholder=False)
        assert find_gi_artifacts_for_episode_ids(tmp_path, ["nope"]) == []

    def test_empty_id_list_is_not_a_corpus_wide_sweep(self, tmp_path):
        """A blank selection must select NOTHING, never everything."""
        _write_pair(tmp_path, "f1", "run_20260814-055303", "ep-1", placeholder=False)
        assert find_gi_artifacts_for_episode_ids(tmp_path, []) == []
        assert find_gi_artifacts_for_episode_ids(tmp_path, ["", "  "]) == []

    def test_unresolvable_membership_warns_and_still_finds_work(self, tmp_path, caplog):
        """Scoping must never turn "I cannot tell" into "there is nothing to do".

        A gi.json with no metadata sibling makes corpus membership resolve to zero. Returning
        [] there would report PASS having re-derived nothing — the silent-success shape.
        """
        gi_dir = tmp_path / "feeds" / "f1" / "run_20260814-055303" / "metadata"
        gi_dir.mkdir(parents=True)
        gi = gi_dir / "0001 - Orphan.gi.json"
        gi.write_text(
            json.dumps({"episode_id": "ep-orphan", "nodes": [], "edges": []}), encoding="utf-8"
        )

        caplog.set_level("WARNING")
        got = find_gi_artifacts_for_episode_ids(tmp_path, ["ep-orphan"])
        assert [p for p, _ in got] == [gi]
        assert any("membership resolved 0" in r.getMessage() for r in caplog.records)


class TestZeroMatchIsAFailure:
    def test_requested_id_not_found_makes_the_report_fail(self, tmp_path):
        """THE REGRESSION: this used to be report.ok is True, VERDICT PASS, exit 0."""
        _write_pair(tmp_path, "f1", "run_20260814-055303", "ep-1", placeholder=False)

        report = repair_mod.repair_placeholder_artifacts(
            tmp_path, None, dry_run=True, episode_ids=["ep-MISSING"]
        )
        assert report.requested_not_found == ["ep-MISSING"]
        assert report.ok is False, "asking for an episode and getting none must not be a PASS"
        assert "NOT FOUND" in report.format()

    def test_placeholder_sweep_finding_nothing_is_still_a_pass(self, tmp_path):
        """Distinct case: a sweep that finds no damage is a legitimate success."""
        _write_pair(tmp_path, "f1", "run_20260814-055303", "ep-1", placeholder=False)
        report = repair_mod.repair_placeholder_artifacts(tmp_path, None, dry_run=True)
        assert report.requested_not_found == []
        assert report.ok is True


class TestDryRunMatchesTheRealRun:
    def test_dry_run_refuses_healthy_without_force(self, tmp_path):
        """Preview must not promise work the real run declines.

        Previously: --episode-ids --dry-run printed "1 would be repaired" while the same
        command without --dry-run exited 1 refusing to touch a healthy artifact.
        """
        _write_pair(tmp_path, "f1", "run_20260814-055303", "ep-1", placeholder=False)
        report = repair_mod.repair_placeholder_artifacts(
            tmp_path, None, dry_run=True, episode_ids=["ep-1"], force_healthy=False
        )
        assert report.skipped_dry_run == []

    def test_dry_run_lists_healthy_when_forced(self, tmp_path):
        gi = _write_pair(tmp_path, "f1", "run_20260814-055303", "ep-1", placeholder=False)
        report = repair_mod.repair_placeholder_artifacts(
            tmp_path, None, dry_run=True, episode_ids=["ep-1"], force_healthy=True
        )
        assert report.skipped_dry_run == [str(gi)]

    def test_dry_run_lists_a_placeholder_without_force(self, tmp_path):
        gi = _write_pair(tmp_path, "f1", "run_20260814-055303", "ep-1", placeholder=True)
        report = repair_mod.repair_placeholder_artifacts(
            tmp_path, None, dry_run=True, episode_ids=["ep-1"], force_healthy=False
        )
        assert report.skipped_dry_run == [str(gi)]


class TestForceHealthyGuard:
    """The sweep's safety property: never rewrite work that succeeded, unless asked."""

    def test_healthy_artifact_is_refused_by_default(self, tmp_path):
        gi = _write_pair(tmp_path, "f1", "run_20260814-055303", "ep-1", placeholder=False)
        result = repair_mod.repair_episode(gi, None, force_healthy=False)
        assert result.ok is False
        assert "refusing to rewrite a healthy artifact" in (result.error or "")

    def test_force_healthy_gets_past_the_refusal_and_says_so(self, tmp_path, caplog):
        gi = _write_pair(tmp_path, "f1", "run_20260814-055303", "ep-1", placeholder=False)
        caplog.set_level("WARNING")
        # Fails later (no transcript in this fixture) — what matters is the refusal is gone
        # and the overwrite was announced.
        result = repair_mod.repair_episode(gi, None, force_healthy=True)
        assert "refusing to rewrite a healthy artifact" not in (result.error or "")
        assert any("re-deriving HEALTHY artifact" in r.getMessage() for r in caplog.records)


class TestModelVersionLineage:
    def test_uses_the_real_resolver_not_a_fallback(self):
        """``_model_version`` imported a module that does not exist and swallowed the error.

        It therefore ALWAYS returned ``cfg.summary_model or "unknown"`` — the SUMMARY model,
        where the resolver names the INSIGHT model. Every repaired artifact carried fabricated
        provenance, on the one field that distinguishes two derivations.
        """
        from podcast_scraper import config
        from podcast_scraper.gi.provenance import resolve_gil_artifact_model_version

        cfg = config.Config.model_validate(
            {
                "rss_url": "https://example.com/f.rss",
                "summary_provider": "litellm",
                "transcribe_missing": False,
            }
        )
        assert repair_mod._model_version(cfg) == str(
            resolve_gil_artifact_model_version(cfg, None) or "unknown"
        )

    def test_a_broken_resolver_import_is_loud(self, monkeypatch):
        """No try/except: provenance silently degrading is what caused the original bug."""
        import podcast_scraper.gi.provenance as prov

        def _boom(*_a, **_k):
            raise RuntimeError("resolver exploded")

        monkeypatch.setattr(prov, "resolve_gil_artifact_model_version", _boom)
        with pytest.raises(RuntimeError):
            repair_mod._model_version(object())
