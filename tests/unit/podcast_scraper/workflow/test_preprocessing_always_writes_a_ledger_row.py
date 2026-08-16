"""Audio preprocessing must never exit silently — every path writes exactly one ledger row.

The stage ledger (#1647) exists so an operator can ask "what actually happened to this episode"
and get an answer for every stage. A stage that reports only when it succeeds — or only when it
fails — cannot distinguish "ran fine" from "never ran", which is precisely the ambiguity that let
#1646 hide in production for weeks.

Before this, ``_preprocess_audio_if_needed`` had three paths that returned with no row at all:
preprocessing switched off, the media file missing, and ffmpeg not installed. All three produce a
manifest with no ``audio_preprocessing`` block, which reads as "nothing to report" rather than
"skipped, and here is why". The third is the dangerous one: on a host with no ffmpeg, EVERY
episode transcribes from unnormalised full-size audio — costing more and scoring worse — and the
only trace was one WARNING at factory level, invisible per episode.

The last test in this file is the drift guard: it fails when someone adds a new exit path without
a ledger row, so this class of bug cannot come back quietly.
"""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional

import pytest

from podcast_scraper import config
from podcast_scraper.workflow import episode_processor
from podcast_scraper.workflow.episode_processor import _preprocess_audio_if_needed

SOURCE_PATH = episode_processor.__file__.replace(".pyc", ".py")
FUNCTION_NAME = "_preprocess_audio_if_needed"


class _RecordingMetrics:
    """Captures ledger rows; tolerates the rest of the metrics surface the function pokes at."""

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def record_stage_outcome(
        self,
        stage: str,
        job_idx: int,
        outcome: str,
        *,
        reason: Optional[str] = None,
        detail: Optional[Dict[str, Any]] = None,
        duration_seconds: Optional[float] = None,
    ) -> None:
        self.rows.append(
            {
                "stage": stage,
                "job_idx": job_idx,
                "outcome": outcome,
                "reason": reason,
                "detail": detail,
                "duration_seconds": duration_seconds,
            }
        )

    def __getattr__(self, name: str):  # record_preprocessing_attempt, _time, _cache_hit_flag, ...
        if name.startswith("record_"):
            return lambda *a, **k: None
        raise AttributeError(name)

    @property
    def preprocessing_rows(self) -> List[Dict[str, Any]]:
        return [r for r in self.rows if r["stage"] == "audio_preprocessing"]


class _Job:
    idx = 7
    episode_title = "An Episode"
    episode_guid = "guid-1"


@pytest.fixture
def metrics() -> _RecordingMetrics:
    return _RecordingMetrics()


@pytest.fixture
def media_file(tmp_path) -> str:
    path = tmp_path / "episode.mp3"
    path.write_bytes(b"\x00" * 2048)
    return str(path)


def _cfg(**kw) -> config.Config:
    return config.Config(rss_url="https://example.com/feed.xml", **kw)


def _only_row(metrics: _RecordingMetrics) -> Dict[str, Any]:
    rows = metrics.preprocessing_rows
    assert len(rows) == 1, f"expected exactly one audio_preprocessing row, got {rows}"
    return rows[0]


def test_preprocessing_disabled_is_reported_as_skipped(metrics, media_file):
    """Config says off. That is a legitimate outcome — but it must be STATED, not implied."""
    out = _preprocess_audio_if_needed(
        _Job(), _cfg(preprocessing_enabled=False), media_file, metrics
    )

    row = _only_row(metrics)
    assert row["outcome"] == "skipped"
    assert row["reason"] == "preprocessing_disabled"
    assert out == media_file, "the original audio must still be handed to transcription"


def test_a_missing_media_file_is_reported_and_distinguishable_from_disabled(metrics, tmp_path):
    """Preprocessing was ASKED for and could not run — a different fact than "switched off"."""
    missing = str(tmp_path / "never-downloaded.mp3")
    out = _preprocess_audio_if_needed(_Job(), _cfg(preprocessing_enabled=True), missing, metrics)

    row = _only_row(metrics)
    assert row["outcome"] == "skipped"
    assert row["reason"] == "media_file_missing"
    assert row["reason"] != "preprocessing_disabled", "the two causes must not collapse into one"
    assert out == missing


def test_an_empty_media_path_is_reported_too(metrics):
    """Falsy path, not merely a nonexistent one — same guard, and it must not crash on None."""
    _preprocess_audio_if_needed(_Job(), _cfg(preprocessing_enabled=True), "", metrics)

    assert _only_row(metrics)["reason"] == "media_file_missing"


def test_missing_ffmpeg_is_degraded_not_silent(metrics, media_file, monkeypatch):
    """THE expensive one: every episode on this host transcribes from full-size raw audio."""
    monkeypatch.setattr(
        "podcast_scraper.preprocessing.audio.ffmpeg_processor._check_ffmpeg_available",
        lambda: False,
    )

    out = _preprocess_audio_if_needed(_Job(), _cfg(preprocessing_enabled=True), media_file, metrics)

    row = _only_row(metrics)
    assert row["outcome"] == "degraded", (
        "ffmpeg missing is not a clean skip — the audio really did go to the provider "
        "unnormalised and oversized"
    )
    assert row["reason"] == "ffmpeg_unavailable"
    assert (row["detail"] or {}).get("fallback") == "original_audio"
    assert out == media_file


def test_the_row_is_attributed_to_the_right_episode(metrics, media_file):
    """A ledger row on the wrong episode is worse than none — it accuses a healthy episode."""
    job = _Job()
    job.idx = 42
    _preprocess_audio_if_needed(job, _cfg(preprocessing_enabled=False), media_file, metrics)

    assert _only_row(metrics)["job_idx"] == 42


def test_no_metrics_object_still_does_not_crash(media_file):
    """Ledger writes are best-effort: observability must never be what kills an episode."""
    out = _preprocess_audio_if_needed(_Job(), _cfg(preprocessing_enabled=False), media_file, None)

    assert out == media_file


def test_a_metrics_object_that_explodes_does_not_kill_the_episode(media_file):
    """Same rule, harsher: a broken metrics backend must not take transcription down with it."""

    class _Exploding:
        def record_stage_outcome(self, *a, **k):
            raise RuntimeError("metrics backend down")

        def __getattr__(self, name):
            if name.startswith("record_"):
                return lambda *a, **k: None
            raise AttributeError(name)

    out = _preprocess_audio_if_needed(
        _Job(), _cfg(preprocessing_enabled=False), media_file, _Exploding()
    )

    assert out == media_file


# --------------------------------------------------------------------------------------------
# Drift guard
# --------------------------------------------------------------------------------------------


def _function_node(name: str) -> ast.FunctionDef:
    tree = ast.parse(open(SOURCE_PATH, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {SOURCE_PATH}")


#: Both halves of the decision. The guard clauses live in the helper, the real work in the main
#: function; an exit added to EITHER can skip the ledger, so the guard counts both.
GUARDED_FUNCTIONS = (FUNCTION_NAME, "_preprocessing_cannot_run")

#: Exit paths this file drives behaviourally, above. Bump ONLY together with a new test.
COVERED_EXIT_COUNT = 6


def test_no_new_exit_path_escapes_this_files_coverage():
    """Fails when someone adds an early return without adding a ledger row and a test for it.

    This is deliberately a structural count rather than a clever dataflow analysis. The failure
    mode being guarded is a human adding ``return media_for_transcription`` to a new branch and
    not thinking about the ledger — a plain count catches that, and the failure message says
    exactly what to do. It cannot be satisfied by editing the constant alone without lying.
    """
    returns = [
        n
        for name in GUARDED_FUNCTIONS
        for n in ast.walk(_function_node(name))
        if isinstance(n, ast.Return)
    ]

    assert len(returns) == COVERED_EXIT_COUNT, (
        f"{' + '.join(GUARDED_FUNCTIONS)} now have {len(returns)} return statements between "
        f"them; this file covers {COVERED_EXIT_COUNT}. A new exit path must (a) write exactly "
        f"one audio_preprocessing ledger row before returning and (b) get a test here — then "
        f"bump COVERED_EXIT_COUNT. Silent exits are what #1647 exists to prevent."
    )


def test_every_ledger_row_uses_a_known_outcome_and_a_stable_reason_slug():
    """Outcomes are a closed set and reasons are machine-readable slugs — not prose (#1647)."""
    allowed_outcomes = {"ran", "skipped", "failed", "degraded"}

    seen_outcomes, seen_reasons = set(), set()
    for name in GUARDED_FUNCTIONS:
        for call in ast.walk(_function_node(name)):
            if not isinstance(call, ast.Call):
                continue
            if getattr(call.func, "id", "") != "_record_preprocessing_outcome":
                continue
            positional = [a for a in call.args if isinstance(a, ast.Constant)]
            if positional:
                seen_outcomes.add(positional[0].value)
            for kw in call.keywords:
                if kw.arg == "reason" and isinstance(kw.value, ast.Constant):
                    seen_reasons.add(kw.value.value)

    assert seen_outcomes, "no ledger writes found — did the function get renamed?"
    assert (
        seen_outcomes <= allowed_outcomes
    ), f"unknown outcome(s): {seen_outcomes - allowed_outcomes}"
    for reason in seen_reasons:
        assert reason == reason.lower().strip(), f"reason slug not normalised: {reason!r}"
        assert " " not in reason, f"reason must be a slug, not prose: {reason!r}"
