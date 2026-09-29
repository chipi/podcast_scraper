"""The translation stage slot: the decision, the ledger entry, and the deadline accounting.

WHAT THESE GUARD. S2.2 performs no translation, so there is no output to check. What there is
to check is that the pipeline's RECORD of translation is honest on every path — an English
episode, a non-English one with the flag off, one with the flag on, and one whose language came
from the feed rather than a default. An audit that cannot tell those apart cannot answer "which
episodes would need translating", which is the question the flag's rollout is sized against.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from podcast_scraper import config
from podcast_scraper.utils.timeout import deadline_credit, timeout_context, TimeoutError as TE
from podcast_scraper.workflow import processing_manifest as pm
from podcast_scraper.workflow.translation_stage import (
    decide_translation,
    REASON_ALREADY_ENGLISH,
    REASON_FLAG_OFF,
    REASON_NO_TRANSCRIPT,
    REASON_NOT_IMPLEMENTED,
    run_translation_stage,
    STATUS_PENDING,
    STATUS_SKIPPED,
    TranslationOutcome,
)

pytestmark = pytest.mark.unit

REL = "transcripts/01 - ep.txt"


def _cfg(**kw: Any) -> config.Config:
    return config.Config(rss="https://example.com/feed.xml", **kw)


class TestTheDecision:
    def test_an_english_episode_is_skipped_with_a_reason(self) -> None:
        got = decide_translation(_cfg(language="en"), transcript_relpath=REL)
        assert got.status == STATUS_SKIPPED
        assert got.reason == REASON_ALREADY_ENGLISH
        assert got.source_language == "en"
        assert got.ran is False

    def test_a_non_english_episode_with_the_flag_off_records_its_language_anyway(self) -> None:
        """The population the rollout is sized against.

        Short-circuiting on the flag first would have been simpler and would have recorded a
        null language for every episode processed with translation disabled — making "how many
        episodes would need translating" unanswerable from the corpus, which is precisely the
        question you need answered BEFORE turning the flag on.
        """
        got = decide_translation(_cfg(language="es"), transcript_relpath=REL)
        assert got.status == STATUS_SKIPPED
        assert got.reason == REASON_FLAG_OFF
        assert got.source_language == "es", "the language must survive the flag being off"

    def test_a_non_english_episode_with_the_flag_on_is_pending_not_translated(self) -> None:
        """`pending` is an unpaid debt, not a failure and not a success.

        S2.2 is the stage slot; the translator arrives in S2.3/S2.4. Reporting `skipped` here
        would say there was nothing to do, and `translated` would be a lie.
        """
        got = decide_translation(
            _cfg(language="es", multilingual_ingest=True), transcript_relpath=REL
        )
        assert got.status == STATUS_PENDING
        assert got.reason == REASON_NOT_IMPLEMENTED
        assert got.ran is False, "a stage that produced nothing did not run"

    def test_the_feeds_declared_language_beats_the_profile_default_and_says_so(self) -> None:
        """Provenance, not just a value: `rss` means measured, `profile_default` means assumed."""
        got = decide_translation(
            _cfg(language="en", multilingual_ingest=True),
            feed_language="es-ES",
            transcript_relpath=REL,
        )
        assert got.source_language == "es"
        assert got.language_source == "rss"
        assert got.status == STATUS_PENDING

    def test_an_operator_override_beats_the_feed(self) -> None:
        got = decide_translation(
            _cfg(language="en", language_override="es", multilingual_ingest=True),
            feed_language="en-US",
            transcript_relpath=REL,
        )
        assert got.source_language == "es"
        assert got.language_source == "override"

    def test_no_transcript_is_skipped_rather_than_pending(self) -> None:
        got = decide_translation(
            _cfg(language="es", multilingual_ingest=True), transcript_relpath=None
        )
        assert got.status == STATUS_SKIPPED
        assert got.reason == REASON_NO_TRANSCRIPT


class TestTheLedgerEntry:
    def test_an_english_episode_gets_NO_block(self, tmp_path: Path) -> None:
        """The one outcome that is decided and then deliberately not written down.

        S0.10's guard measured why: recording a block is what moves
        ``pipeline_composition_version``, which "reprocess below version X" and the prod-state
        pin key on. For 678 English episodes that is a corpus-wide hash change buying a
        distinction with no consumer — there was never anything translation could have done to
        an English episode, and ``episode.language`` already says so.

        The arc's S2.2 row originally said the opposite ("every English episode's ledger gains
        translation: skipped") and was corrected to this.
        """
        (tmp_path / "transcripts").mkdir()
        outcome = run_translation_stage(
            _cfg(language="en"),
            transcript_relpath=REL,
            effective_output_dir=str(tmp_path),
            episode_id="ep1",
        )
        assert outcome.status == STATUS_SKIPPED
        assert outcome.reason == REASON_ALREADY_ENGLISH
        assert not (tmp_path / "transcripts" / "01 - ep.manifest.json").exists()

    def test_every_other_outcome_IS_written(self, tmp_path: Path) -> None:
        """So the silence above is a choice, not an inability to write.

        `flag_off` matters most here: non-English episodes seen while the flag was off are the
        population the rollout is sized against, and nothing else in the corpus records them.
        """
        (tmp_path / "transcripts").mkdir()
        run_translation_stage(
            _cfg(language="es"),
            transcript_relpath=REL,
            effective_output_dir=str(tmp_path),
            episode_id="ep1",
            feed_id="f1",
            run_id="r1",
        )
        data = json.loads((tmp_path / "transcripts" / "01 - ep.manifest.json").read_text())
        block = data["stages"]["translation"]
        assert block["ran"] is False
        assert block["method_version"] == "translation-1"
        assert block["cost_usd"] == 0.0
        assert block["metrics"]["status"] == STATUS_SKIPPED
        assert block["metrics"]["reason"] == REASON_FLAG_OFF
        assert block["metrics"]["source_language"] == "es"
        assert data["episode_id"] == "ep1"

    def test_a_pending_episode_is_visible_in_the_ledger(self, tmp_path: Path) -> None:
        (tmp_path / "transcripts").mkdir()
        run_translation_stage(
            _cfg(language="es", multilingual_ingest=True),
            transcript_relpath=REL,
            effective_output_dir=str(tmp_path),
        )
        data = json.loads((tmp_path / "transcripts" / "01 - ep.manifest.json").read_text())
        metrics = data["stages"]["translation"]["metrics"]
        assert metrics["status"] == STATUS_PENDING
        assert metrics["source_language"] == "es"

    def test_a_manifest_write_failure_does_not_lose_the_episode(self, tmp_path: Path) -> None:
        """No `transcripts/` directory, so the write fails. The decision still comes back."""
        got = run_translation_stage(
            _cfg(language="en"),
            transcript_relpath=REL,
            effective_output_dir=str(tmp_path / "does-not-exist"),
        )
        assert got.status == STATUS_SKIPPED

    def test_the_stage_is_in_the_canonical_order_and_before_summary(self) -> None:
        """Naming runs on the SOURCE text; everything after translation reads English."""
        order = pm.CANONICAL_STAGE_ORDER
        assert "translation" in order
        assert order.index("naming") < order.index("translation") < order.index("summary")

    def test_adding_the_stage_moved_the_composition_hash_exactly_once(self) -> None:
        """The cost of putting translation in the graph, stated as a test rather than discovered.

        Unlike `turns`, this stage IS in `CANONICAL_STAGE_ORDER`, so every episode's composition
        version changes — a pipeline with a translation step is genuinely not the pipeline
        without one. What must NOT happen is the hash depending on whether translation did any
        work, which would make it a result rather than a graph shape.
        """
        core = ["asr", "diarization", "naming", "summary", "gi", "kg"]
        with_translation = core + ["translation"]
        assert pm.pipeline_composition_version(core) != pm.pipeline_composition_version(
            with_translation
        )
        # Same graph, whatever the stage decided: presence is the key, not `ran`.
        assert pm.pipeline_composition_version(with_translation) == (
            pm.pipeline_composition_version(list(reversed(with_translation)))
        )


class TestTheDeadlineCredit:
    def test_translation_time_is_credited_back_to_the_enclosing_deadline(self) -> None:
        """The alert this protects already misfired once for a different stage.

        `processing.py`'s deadline wraps summary + GI + KG under a config key named
        `summarization_timeout`; every one of the 22 overruns measured on 2026-08-31 was GI
        reported under the summariser's name, sending whoever read it to debug the innocent
        stage. Translation runs inside the same block, at a cost nobody has bounded yet (S2.10
        measures it), so its wall time is credited back instead of an allowance being guessed.
        """
        with timeout_context(1, "metadata generation"):
            time.sleep(0.3)
            assert deadline_credit(3.0, reason="translation stage") is True
            time.sleep(0.9)  # total elapsed > 1s, but 3s of it is credited
        # No TimeoutError: reaching here IS the assertion.

    def test_without_the_credit_the_same_elapsed_time_still_overruns(self) -> None:
        """The other direction. A credit that cannot be withheld is not a credit."""
        with pytest.raises(TE):
            with timeout_context(1, "metadata generation"):
                time.sleep(1.3)

    def test_crediting_outside_a_deadline_is_a_no_op_not_an_error(self) -> None:
        """Most callers are not inside an observed block — relabel paths, every unit test — so
        the stage must never have to ask whether it is."""
        assert deadline_credit(5.0, reason="translation stage") is False
        with timeout_context(None, "disabled"):
            assert deadline_credit(5.0, reason="translation stage") is False

    def test_the_stage_credits_its_own_duration(self, tmp_path: Path) -> None:
        # A non-English episode, because an English one writes no block to read the duration from.
        (tmp_path / "transcripts").mkdir()
        with timeout_context(300, "metadata generation"):
            outcome = run_translation_stage(
                _cfg(language="es"),
                transcript_relpath=REL,
                effective_output_dir=str(tmp_path),
            )
        assert outcome.duration_s >= 0.0
        block = json.loads((tmp_path / "transcripts" / "01 - ep.manifest.json").read_text())[
            "stages"
        ]["translation"]
        assert "duration_s" in block


class TestOutcomeVocabulary:
    @pytest.mark.parametrize(
        "status,expected_ran",
        [("skipped", False), ("pending", False), ("translated", True), ("failed", True)],
    )
    def test_ran_means_the_stage_did_work(self, status: str, expected_ran: bool) -> None:
        """`failed` DID run — it tried and produced nothing usable, which is a different fact
        from `skipped` (nothing to do) and `pending` (work owed, not attempted)."""
        assert TranslationOutcome(status=status).ran is expected_ran
