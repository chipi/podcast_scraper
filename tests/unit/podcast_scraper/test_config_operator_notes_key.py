"""Top-level ``notes:`` in ``viewer_operator.yaml`` — the comment substitute (#2086).

``PUT /api/enrichment/config`` rewrites the whole file with ``yaml.safe_dump``,
which drops comments at parse and cannot emit them. So the first save from the
operator UI deletes every ``#`` line in the file, taking the operational
rationale with it ("we raised this timeout because X").

The answer is to carry that rationale as DATA rather than as comments:
``note`` inside each enricher block for per-enricher rationale, and a top-level
``notes`` mapping for rationale about top-level settings.

For a top-level key that is written but never read, three things have to hold at
once, and each has its own test below:

1. the operator API must preserve it across a save (it does — ``safe_dump`` of a
   dict the handler never prunes),
2. ``Config`` must ACCEPT it — it is ``extra="forbid"``, so without the
   allow-list a notes block round-trips happily through the API and then fails
   the next pipeline run, and
3. the unknown-key gate must still bite on typos, or the allow-list has bought
   compatibility by disabling the guard.
"""

from __future__ import annotations

import yaml

from podcast_scraper import config as config_module
from podcast_scraper.config import Config

NOTES_BLOCK = {
    "audio_storage_backend": (
        "Pinned to s3 on 2026-09-12: the local backend filled the VPS disk during "
        "the 200-episode backfill. Do not let a profile override this."
    ),
}


class TestTopLevelNotesKey:
    def test_notes_is_declared_operator_only(self):
        """Declared once; both the CLI gate and Config read the same frozenset."""
        assert "notes" in config_module.OPERATOR_ONLY_TOP_LEVEL_KEYS

    def test_config_accepts_a_top_level_notes_block(self):
        """The load that would otherwise break on the next pipeline run.

        Mutation check: drop ``"notes"`` from the allow-list and this raises
        ``ValidationError: notes — Extra inputs are not permitted``.
        """
        cfg = Config.model_validate({"notes": NOTES_BLOCK})

        assert cfg is not None

    def test_notes_is_stripped_not_stored(self):
        """It is documentation, not configuration — nothing may read it back.

        Guards against someone later making behaviour depend on a notes value,
        which would turn prose into a load-bearing setting.
        """
        cfg = Config.model_validate({"notes": NOTES_BLOCK})

        assert not hasattr(cfg, "notes")
        assert "notes" not in cfg.model_dump()

    def test_notes_accepts_prose_or_nested_mapping(self):
        """Operators write whatever shape reads well; no schema is imposed."""
        for shape in (
            "single line of rationale",
            ["one reason", "another reason"],
            {"section": {"key": "why"}},
        ):
            assert Config.model_validate({"notes": shape}) is not None

    def test_unknown_top_level_key_is_still_rejected(self):
        """The allow-list must not have blunted the typo guard.

        ``notez`` is one keystroke from ``notes``; if the gate stopped biting,
        a typo'd real setting would be silently ignored instead of refused.
        """
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="notez"):
            Config.model_validate({"notez": NOTES_BLOCK})


class TestNotesSurvivesTheOperatorSave:
    """The reason the key exists: a UI save must not eat the rationale."""

    def test_safe_dump_round_trip_keeps_notes_but_would_have_eaten_a_comment(self):
        original = yaml.safe_load(
            "# operator rationale as a COMMENT — destroyed by the first UI save\n"
            "notes:\n"
            '  audio_storage_backend: "pinned to s3, see #2086"\n'
            "enrichment:\n"
            "  enabled: true\n"
        )
        assert original["notes"]["audio_storage_backend"].endswith("#2086")

        # What PUT /api/enrichment/config does: replace the block, dump the rest.
        original["enrichment"] = {"enabled": False}
        saved = yaml.safe_dump(original, sort_keys=False, default_flow_style=False)

        # The hand-written comment is gone; the notes VALUE survived intact,
        # including its ``#`` (safe_dump re-quotes it to keep it a value).
        assert "destroyed by the first UI save" not in saved
        assert yaml.safe_load(saved)["notes"] == original["notes"]
        assert yaml.safe_load(saved)["notes"]["audio_storage_backend"].endswith("#2086")

    def test_unquoted_hash_in_a_note_truncates_the_value(self):
        """The trap an operator will hit first: notes cite issue numbers.

        In YAML an unquoted `` #`` starts a comment mid-value, so
        ``see #2086`` silently becomes ``see``. Nothing errors — the rationale
        just loses its tail. Once safe_dump re-emits the truncated string the
        loss is permanent, so this is worth knowing before writing notes by hand
        rather than through the UI (which quotes for you).
        """
        truncated = yaml.safe_load("notes:\n  k: see #2086\n")
        quoted = yaml.safe_load('notes:\n  k: "see #2086"\n')

        assert truncated["notes"]["k"] == "see"
        assert quoted["notes"]["k"] == "see #2086"

    def test_the_saved_file_still_loads_as_a_config(self):
        """Closes the loop: preserved by the API AND accepted by the pipeline."""
        saved = yaml.safe_dump({"notes": NOTES_BLOCK}, sort_keys=False)

        assert Config.model_validate(yaml.safe_load(saved)) is not None
