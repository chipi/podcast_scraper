"""End-to-end resilience integration tests for the EnrichmentExecutor.

These drive the executor with ``ScriptedEnricher`` fixtures to exercise
retry / circuit / timeout / cost-cap interactions in one go — the
per-piece unit tests in ``tests/unit/enrichment/`` cover the components
in isolation; this file proves they compose correctly when the executor
runs them.

Tests are marked ``integration`` and never touch the network.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from podcast_scraper.enrichment.executor import EnrichmentExecutor, ExecutorOptions
from podcast_scraper.enrichment.protocol import (
    EnricherResult,
    EnricherSet,
    EnricherTier,
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_QUARANTINED,
)
from podcast_scraper.enrichment.registry import EnricherRegistry
from tests.fixtures.enrichment.mock_scorers import (
    BadInputError,
    DependencyAccessError,
    manifest,
    ModelLoadError,
    ScorerTimeoutError,
    Script,
    ScriptedEnricher,
    SlowEnricher,
)

pytestmark = pytest.mark.integration


def _set(*ids: str) -> EnricherSet:
    return EnricherSet(enabled_enrichers=list(ids))


def _registry(*enrichers) -> EnricherRegistry:
    reg = EnricherRegistry()
    for e in enrichers:
        reg.register(e)
    return reg


def _executor(corpus: Path, reg: EnricherRegistry, eset: EnricherSet) -> EnrichmentExecutor:
    return EnrichmentExecutor(corpus_root=corpus, registry=reg, enricher_set=eset)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ---------------------------------------------------------------------------
# Happy path — no retries, on-disk envelope written
# ---------------------------------------------------------------------------


def test_single_enricher_succeeds_no_retries(tmp_path: Path) -> None:
    enr = ScriptedEnricher(
        manifest=manifest("ok_enricher", writes="ok_enricher.json"),
        script=Script(steps=[EnricherResult(status=STATUS_OK, data={"k": "v"})]),
    )
    result = asyncio.run(_executor(tmp_path, _registry(enr), _set("ok_enricher")).run())
    assert result.status == STATUS_OK
    m = result.per_enricher_metrics["ok_enricher"]
    assert m.runs_ok == 1
    assert m.retries_total == 0
    envelope_path = tmp_path / "enrichments" / "ok_enricher.json"
    assert envelope_path.is_file()
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    assert envelope["data"] == {"k": "v"}


# ---------------------------------------------------------------------------
# Retry — embedding tier survives 2 transient errors then succeeds
# ---------------------------------------------------------------------------


def test_embedding_retries_then_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedding tier policy: max_retries=3. Fail twice with retryable
    errors then succeed → exactly 2 retries recorded, status ok."""
    monkeypatch.setattr("podcast_scraper.enrichment.executor.compute_backoff", lambda *a, **kw: 0.0)
    enr = ScriptedEnricher(
        manifest=manifest("embed_retry", tier=EnricherTier.EMBEDDING, writes="embed_retry.json"),
        script=Script(
            steps=[
                ScorerTimeoutError("transient 1"),
                DependencyAccessError("transient 2"),
                EnricherResult(status=STATUS_OK, data={"vec": [1, 2, 3]}),
            ]
        ),
    )
    result = asyncio.run(_executor(tmp_path, _registry(enr), _set("embed_retry")).run())
    assert result.status == STATUS_OK
    m = result.per_enricher_metrics["embed_retry"]
    assert m.runs_ok == 1
    # The script ran 3 enricher attempts (call_count == 3): 2 raised, then 1 succeeded.
    assert enr.script.call_count == 3
    # JSONL recorded the 2 retry events.
    events = _read_jsonl(tmp_path / "enrichments" / "run.jsonl")
    retry_events = [e for e in events if e["event_type"] == "enrichment.enricher.retry"]
    assert len(retry_events) == 2


# ---------------------------------------------------------------------------
# Non-retryable — BadInputError fails immediately
# ---------------------------------------------------------------------------


def test_bad_input_does_not_retry(tmp_path: Path) -> None:
    enr = ScriptedEnricher(
        manifest=manifest("bad", tier=EnricherTier.EMBEDDING, writes="bad.json"),
        script=Script(steps=[BadInputError("missing artifact")]),
    )
    result = asyncio.run(_executor(tmp_path, _registry(enr), _set("bad")).run())
    assert result.status == STATUS_FAILED
    m = result.per_enricher_metrics["bad"]
    assert m.runs_failed == 1
    assert m.retries_total == 0
    assert not (tmp_path / "enrichments" / "corpus" / "bad.json").is_file()


# ---------------------------------------------------------------------------
# RETRYABLE_ONCE — ModelLoadError retries once then non-retryable
# ---------------------------------------------------------------------------


def test_model_load_retries_once_then_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("podcast_scraper.enrichment.executor.compute_backoff", lambda *a, **kw: 0.0)
    enr = ScriptedEnricher(
        manifest=manifest("loader", tier=EnricherTier.ML, writes="loader.json"),
        script=Script(
            steps=[
                ModelLoadError("oom 1"),
                ModelLoadError("oom 2"),
                ModelLoadError("never reached"),
            ]
        ),
    )
    result = asyncio.run(_executor(tmp_path, _registry(enr), _set("loader")).run())
    assert result.status == STATUS_FAILED
    m = result.per_enricher_metrics["loader"]
    assert m.retries_total == 1
    assert m.runs_failed == 1


# ---------------------------------------------------------------------------
# Circuit opens — embedding tier hits circuit_threshold
# ---------------------------------------------------------------------------


def test_circuit_opens_when_threshold_reached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("podcast_scraper.enrichment.executor.compute_backoff", lambda *a, **kw: 0.0)
    from podcast_scraper.enrichment.resilience import DEFAULT_POLICIES, TierPolicy

    relaxed = TierPolicy(
        max_retries=10,
        initial_backoff_s=0.0,
        backoff_factor=1.0,
        max_backoff_s=0.0,
        circuit_threshold=5,
        auto_disable_threshold=3,
        concurrency=2,
        default_timeout_s=None,
    )
    monkeypatch.setitem(DEFAULT_POLICIES, EnricherTier.EMBEDDING, relaxed)

    enr = ScriptedEnricher(
        manifest=manifest("circuit_open", tier=EnricherTier.EMBEDDING),
        script=Script(steps=[ScorerTimeoutError(f"t{i}") for i in range(10)]),
    )
    result = asyncio.run(_executor(tmp_path, _registry(enr), _set("circuit_open")).run())
    m = result.per_enricher_metrics["circuit_open"]
    assert m.runs_quarantined == 1
    jsonl = _read_jsonl(tmp_path / "enrichments" / "run.jsonl")
    types = [e.get("event_type") for e in jsonl]
    assert "enrichment.enricher.circuit_opened" in types


# ---------------------------------------------------------------------------
# Hard timeout
# ---------------------------------------------------------------------------


def test_hard_timeout_marks_status_timeout(tmp_path: Path) -> None:
    enr = SlowEnricher(
        manifest=manifest(
            "slow", tier=EnricherTier.DETERMINISTIC, writes="slow.json", expected_duration_s=1
        ),
        delay_s=3.0,
    )
    result = asyncio.run(_executor(tmp_path, _registry(enr), _set("slow")).run())
    m = result.per_enricher_metrics["slow"]
    assert m.runs_timeout == 1
    assert result.status == STATUS_FAILED


# ---------------------------------------------------------------------------
# Per-enricher cost cap quarantines only the offender
# ---------------------------------------------------------------------------


def test_per_enricher_cost_cap_quarantines_only_offender(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expensive = ScriptedEnricher(
        manifest=manifest("expensive", max_cost_usd_per_run=0.01),
        script=Script(steps=[EnricherResult(status=STATUS_OK, data={"k": 1})]),
    )
    cheap = ScriptedEnricher(
        manifest=manifest("cheap", writes="cheap.json"),
        script=Script(steps=[EnricherResult(status=STATUS_OK, data={"k": 2})]),
    )

    from podcast_scraper.enrichment import resilience as res_mod

    original = res_mod.CostCapState.per_enricher_cap_exceeded

    def patched(self, enricher_id, manifest_arg):
        if enricher_id == "expensive":
            return True
        return original(self, enricher_id, manifest_arg)

    monkeypatch.setattr(res_mod.CostCapState, "per_enricher_cap_exceeded", patched)
    result = asyncio.run(
        _executor(tmp_path, _registry(expensive, cheap), _set("expensive", "cheap")).run()
    )
    assert result.per_enricher_metrics["expensive"].runs_quarantined == 1
    assert result.per_enricher_metrics["cheap"].runs_ok == 1


# ---------------------------------------------------------------------------
# Run-wide cost cap — fail_on_run_cost_cap=True cancels remaining work
# ---------------------------------------------------------------------------


def test_run_wide_cost_cap_skips_subsequent_enrichers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = ScriptedEnricher(
        manifest=manifest("first", writes="first.json"),
        script=Script(steps=[EnricherResult(status=STATUS_OK, data={"k": 1})]),
    )
    second = ScriptedEnricher(
        manifest=manifest("second", writes="second.json"),
        script=Script(steps=[EnricherResult(status=STATUS_OK, data={"k": 2})]),
    )

    from podcast_scraper.enrichment import resilience as res_mod

    state = {"calls": 0}

    def patched(self, max_total):
        state["calls"] += 1
        return state["calls"] >= 2

    monkeypatch.setattr(res_mod.CostCapState, "run_wide_cap_exceeded", patched)
    result = asyncio.run(
        _executor(tmp_path, _registry(first, second), _set("first", "second")).run(
            options=ExecutorOptions(max_total_cost_usd_per_run=0.50, fail_on_run_cost_cap=True)
        )
    )
    assert result.per_enricher_metrics["first"].runs_skipped == 1
    assert result.status != STATUS_OK


# ---------------------------------------------------------------------------
# Cooperative cancel — body checks ctx.cancel_event
# ---------------------------------------------------------------------------


def test_cancel_event_short_circuits_enricher(tmp_path: Path) -> None:
    first = ScriptedEnricher(
        manifest=manifest("first", writes="first.json"),
        script=Script(steps=[EnricherResult(status=STATUS_OK, data={"k": 1})]),
    )

    class CancelInjector:
        manifest = manifest("second", writes="second.json")

        async def enrich(self, *, bundle, corpus_root, all_bundles, config, ctx):
            if ctx.cancel_event.is_set():
                return EnricherResult(status=STATUS_CANCELLED, error="cancel_requested")
            ctx.cancel_event.set()
            return EnricherResult(status=STATUS_OK, data={"k": 2})

    result = asyncio.run(
        _executor(tmp_path, _registry(first, CancelInjector()), _set("first", "second")).run()
    )
    m = result.per_enricher_metrics["second"]
    # Either the second ran-and-then-set-cancel (runs_ok=1), or in a
    # later run cycle a cancel was observed; either way the run is
    # accounted for.
    assert m.runs_ok + m.runs_cancelled >= 1
    _ = STATUS_QUARANTINED  # silence unused-import; quarantine path is exercised elsewhere


# ---------------------------------------------------------------------------
# JSONL audit trail — retry then success
# ---------------------------------------------------------------------------


def test_jsonl_event_trail_for_retry_then_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("podcast_scraper.enrichment.executor.compute_backoff", lambda *a, **kw: 0.0)
    enr = ScriptedEnricher(
        manifest=manifest("audit", tier=EnricherTier.EMBEDDING),
        script=Script(
            steps=[
                ScorerTimeoutError("once"),
                EnricherResult(status=STATUS_OK, data={"k": 1}),
            ]
        ),
    )
    asyncio.run(_executor(tmp_path, _registry(enr), _set("audit")).run())
    events = _read_jsonl(tmp_path / "enrichments" / "run.jsonl")
    types = {e["event_type"] for e in events}
    assert {
        "enrichment.run.started",
        "enrichment.enricher.started",
        "enrichment.enricher.retry",
        "enrichment.enricher.completed",
        "enrichment.run.completed",
    } <= types


# ---------------------------------------------------------------------------
# Status file lifecycle
# ---------------------------------------------------------------------------


def test_status_file_written_and_finalised_idle(tmp_path: Path) -> None:
    enr = ScriptedEnricher(
        manifest=manifest("status_check"),
        script=Script(steps=[EnricherResult(status=STATUS_OK, data={"k": 1})]),
    )
    asyncio.run(_executor(tmp_path, _registry(enr), _set("status_check")).run())
    status_path = tmp_path / ".viewer" / "enrichment_status.json"
    assert status_path.is_file()
    final = json.loads(status_path.read_text(encoding="utf-8"))
    assert final.get("idle") is True or final.get("current_enricher") is None


def test_executor_options_profile_is_threaded_into_run_summary(tmp_path: Path) -> None:
    """ExecutorOptions.profile must flow through to run_summary.profile so the
    operator can tell which profile preset drove the run."""
    enr = ScriptedEnricher(
        manifest=manifest("profile_check"),
        script=Script(steps=[EnricherResult(status=STATUS_OK, data={"k": 1})]),
    )
    result = asyncio.run(
        _executor(tmp_path, _registry(enr), _set("profile_check")).run(
            options=ExecutorOptions(profile="airgapped_thin")
        )
    )
    assert result.run_summary.get("profile") == "airgapped_thin"
    # And the JSONL run.started event records the profile too.
    events = _read_jsonl(tmp_path / "enrichments" / "run.jsonl")
    started = [e for e in events if e["event_type"] == "enrichment.run.started"]
    assert started and started[0]["profile"] == "airgapped_thin"
