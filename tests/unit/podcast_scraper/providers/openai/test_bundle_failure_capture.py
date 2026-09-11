"""Bundled-quote failure capture (#1893) — the reply is the evidence, so keep it.

The WARNING logged on a bundled-quote parse failure records THAT it failed and whether the
shape was budget-like, but not the reply itself. Without the reply there is no way to tell a
decoding loop from genuine over-generation, and those need opposite fixes (a decoding penalty
vs an output budget computed against the served context).

Reconstructing a failing prompt offline does not work: `gi_require_grounding` drops the insights
that lost their quotes, so the artifact's survivors are exactly the wrong sample, and four
attempts to rebuild one produced healthy replies. Capturing at the moment of failure is the only
way to get the real thing.

These tests pin the two properties that make the capture worth having: it discriminates a loop
from a healthy reply, and it can never break the pipeline it is observing.
"""

from __future__ import annotations

import glob
import json
import os
import types

import pytest

from podcast_scraper.providers.openai.openai_provider import (
    _BUNDLE_CAPTURE_MAX_CHARS,
    _capture_bundle_failure,
    _repetition_signal,
)

pytestmark = pytest.mark.unit


HEALTHY = '{"0": ["a quote about solar costs"], "1": ["a different quote entirely"]}'
LOOPING = '{"0": ["' + "the panel efficiency improved again and again over the decade " * 60


class TestRepetitionSignal:
    """The loop tell, precomputed so it survives the size cap."""

    def test_a_healthy_reply_does_not_look_like_a_loop(self) -> None:
        count, _ = _repetition_signal(HEALTHY)
        assert count == 0

    def test_a_looping_reply_is_obvious_and_names_the_repeated_run(self) -> None:
        count, gram = _repetition_signal(LOOPING)
        assert count > 20, "a decoding loop must be unmistakable, not a judgement call"
        assert "panel efficiency improved" in gram

    def test_short_input_is_not_forced_into_a_verdict(self) -> None:
        # Too few words to say anything: report nothing rather than a spurious signal.
        assert _repetition_signal("too short to judge") == (0, "")
        assert _repetition_signal("") == (0, "")


class TestCaptureWritesTheEvidence:
    def test_capture_records_the_fields_that_discriminate_the_two_causes(self, tmp_path) -> None:
        cfg = types.SimpleNamespace(output_dir=str(tmp_path))
        _capture_bundle_failure(
            cfg=cfg,
            content=LOOPING,
            error="invalid JSON: Unterminated string starting at: line 4 column 5",
            finish_reason="length",
            out_tok=5120,
            max_out=5120,
            insight_texts=["insight one", "insight two"],
            model="NVFP4/Qwen3-30B",
            prompt_chars=61000,
        )
        files = glob.glob(str(tmp_path / ".podcast_scraper/quote-bundle-failures/*.json"))
        assert len(files) == 1
        doc = json.loads(open(files[0], encoding="utf-8").read())
        # finish_reason + the repetition count together are the whole diagnosis.
        assert doc["finish_reason"] == "length"
        assert doc["output_tokens"] == doc["max_output_tokens"] == 5120
        assert doc["repeated_12gram_count"] > 20
        assert doc["insight_count"] == 2
        assert doc["prompt_chars"] == 61000
        assert doc["reply"].startswith('{"0": [')

    def test_an_enormous_reply_is_capped_but_flagged(self, tmp_path) -> None:
        cfg = types.SimpleNamespace(output_dir=str(tmp_path))
        huge = "z " * 200_000
        _capture_bundle_failure(
            cfg=cfg,
            content=huge,
            error="e",
            finish_reason="length",
            out_tok=9,
            max_out=9,
            insight_texts=[],
            model=None,
        )
        doc = json.loads(
            open(
                glob.glob(str(tmp_path / ".podcast_scraper/quote-bundle-failures/*.json"))[0],
                encoding="utf-8",
            ).read()
        )
        assert doc["reply_chars"] == len(huge), "the TRUE size must survive the cap"
        assert len(doc["reply"]) == _BUNDLE_CAPTURE_MAX_CHARS
        assert doc["reply_truncated_for_capture"] is True


class TestCaptureCanNeverBreakThePipeline:
    """A diagnostic that can take down ingestion is worse than no diagnostic."""

    def test_no_output_dir_is_a_silent_no_op(self, tmp_path) -> None:
        _capture_bundle_failure(
            cfg=types.SimpleNamespace(output_dir=None),
            content="x",
            error="e",
            finish_reason=None,
            out_tok=None,
            max_out=None,
            insight_texts=[],
            model=None,
        )
        assert not glob.glob(str(tmp_path / "**/*.json"), recursive=True)

    def test_an_unwritable_destination_is_swallowed(self) -> None:
        # No assertion beyond "does not raise" — that IS the contract.
        _capture_bundle_failure(
            cfg=types.SimpleNamespace(output_dir="/proc/definitely/not/writable"),
            content="x",
            error="e",
            finish_reason=None,
            out_tok=None,
            max_out=None,
            insight_texts=[],
            model=None,
        )

    def test_a_config_without_output_dir_at_all_is_swallowed(self) -> None:
        _capture_bundle_failure(
            cfg=object(),
            content="x",
            error="e",
            finish_reason=None,
            out_tok=None,
            max_out=None,
            insight_texts=[],
            model=None,
        )

    def test_insight_texts_are_truncated_so_one_capture_cannot_balloon(self, tmp_path) -> None:
        cfg = types.SimpleNamespace(output_dir=str(tmp_path))
        _capture_bundle_failure(
            cfg=cfg,
            content=HEALTHY,
            error="e",
            finish_reason="stop",
            out_tok=10,
            max_out=100,
            insight_texts=["x" * 5000, "y" * 5000],
            model=None,
        )
        doc = json.loads(
            open(
                glob.glob(str(tmp_path / ".podcast_scraper/quote-bundle-failures/*.json"))[0],
                encoding="utf-8",
            ).read()
        )
        assert all(len(t) <= 400 for t in doc["insight_texts"])
        assert (
            os.path.getsize(
                glob.glob(str(tmp_path / ".podcast_scraper/quote-bundle-failures/*.json"))[0]
            )
            < 20_000
        )
