"""Unit tests for the autoresearch sweep driver.

Covers per-model prompt wiring (candidates must run on their own tuned
templates, not on a shared prompt — the W27 bug), plus the leaderboard
renderer\'s v2-only shape after the post-2026-W27 refactor to
generate-plus-two-vLLM-judges.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

sweep = importlib.import_module("scripts.baselines.autoresearch_sweep")


def _seed_prompt_dir(root: Path, model_safe: str) -> Path:
    d = root / "src/podcast_scraper/prompts/ollama" / model_safe / "summarization"
    d.mkdir(parents=True, exist_ok=True)
    (d / "system_v1.j2").write_text("system stub", encoding="utf-8")
    (d / "long_v1.j2").write_text("long stub", encoding="utf-8")
    return d


def test_resolve_per_model_prompts_present(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "REPO_ROOT", tmp_path)
    _seed_prompt_dir(tmp_path, "llama3.1_8b")
    resolved = sweep._resolve_per_model_prompts("llama3.1:8b")
    assert resolved == (
        "ollama/llama3.1_8b/summarization/system_v1",
        "ollama/llama3.1_8b/summarization/long_v1",
    )


def test_resolve_per_model_prompts_keeps_dots(monkeypatch, tmp_path):
    """``qwen3.5:9b`` must resolve to ``qwen3.5_9b`` — dots preserved, only
    the ``:`` replaced. Earlier sweep code stripped dots too (for filenames),
    which would silently miss the existing per-model dir on disk."""
    monkeypatch.setattr(sweep, "REPO_ROOT", tmp_path)
    _seed_prompt_dir(tmp_path, "qwen3.5_9b")
    resolved = sweep._resolve_per_model_prompts("qwen3.5:9b")
    assert resolved is not None
    assert "qwen3.5_9b" in resolved[0]


def test_resolve_per_model_prompts_missing_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "REPO_ROOT", tmp_path)
    # No prompt dir seeded.
    assert sweep._resolve_per_model_prompts("brand_new_model:1b") is None


def test_resolve_per_model_prompts_partial_missing_returns_none(monkeypatch, tmp_path):
    """If only ``system_v1.j2`` exists (no ``long_v1.j2``), treat as missing —
    don\'t silently mix a tuned system prompt with a missing user prompt."""
    monkeypatch.setattr(sweep, "REPO_ROOT", tmp_path)
    d = tmp_path / "src/podcast_scraper/prompts/ollama/halfway_3b/summarization"
    d.mkdir(parents=True)
    (d / "system_v1.j2").write_text("only system", encoding="utf-8")
    assert sweep._resolve_per_model_prompts("halfway:3b") is None


def test_materialize_candidate_config_writes_tuned_prompts(monkeypatch, tmp_path):
    """Materialized config must override BOTH backend.model AND prompts.{system,user}
    with the per-model tuned templates — not just backend.model (the W27 bug)."""
    monkeypatch.setattr(sweep, "REPO_ROOT", tmp_path)
    _seed_prompt_dir(tmp_path, "hermes3_8b")

    base_config = tmp_path / "base.yaml"
    base_config.write_text(
        yaml.safe_dump(
            {
                "id": "baseline",
                "backend": {"type": "ollama", "model": "PLACEHOLDER"},
                "prompts": {"system": "shared/bullets_v1", "user": "shared/bullets_user_v1"},
                "data": {"dataset_id": "curated_5feeds_smoke_v2"},
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "tmp"
    out_dir.mkdir()
    cfg_path, source = sweep._materialize_candidate_config(base_config, "hermes3:8b", out_dir)
    assert source == "tuned"
    written = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert written["backend"]["model"] == "hermes3:8b"
    assert written["prompts"]["system"] == "ollama/hermes3_8b/summarization/system_v1"
    assert written["prompts"]["user"] == "ollama/hermes3_8b/summarization/long_v1"


def test_materialize_candidate_config_returns_none_when_prompts_missing(monkeypatch, tmp_path):
    """Missing prompts → (None, \'missing_prompts\'); the caller records a
    fail-fast row rather than producing a degraded score."""
    monkeypatch.setattr(sweep, "REPO_ROOT", tmp_path)
    base_config = tmp_path / "base.yaml"
    base_config.write_text(
        yaml.safe_dump(
            {
                "id": "baseline",
                "backend": {"type": "ollama", "model": "PLACEHOLDER"},
                "prompts": {"system": "shared/x", "user": "shared/y"},
                "data": {"dataset_id": "anything"},
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "tmp"
    out_dir.mkdir()
    cfg_path, source = sweep._materialize_candidate_config(base_config, "ghost:1b", out_dir)
    assert cfg_path is None
    assert source == "missing_prompts"


def _v2_ledger_with(cohort: list[dict], *, extra_phases: list[dict] | None = None) -> dict:
    """Build a v2 ledger with the standard implicit generate phase + one
    judge_qwen judge phase, and any extra_phases the test wants."""
    phases: list[dict] = [
        {"name": "generate", "mode": "inference_only", "judge_a": {}, "judge_b": {}},
        {
            "name": "judge_qwen",
            "mode": "pairwise",
            "judge_a": {"provider": "vllm", "model": "judge-a"},
            "judge_b": {},
        },
    ]
    if extra_phases:
        phases.extend(extra_phases)
    return {
        "schema_version": 2,
        "week_id": "2026-W27",
        "silver": "silver_sonnet46_smoke_v2",
        "dataset": "curated_5feeds_smoke_v2",
        "judges": {"phases": phases},
        "cohort": cohort,
    }


def test_print_leaderboard_renders_v2_markdown_table(capsys):
    """v2 shape: one block per JUDGE phase (generate is skipped — no scores).
    Sort by that phase\'s ``scores_by_phase[phase]["scores"]["final"]``,
    descending."""
    ledger = _v2_ledger_with(
        [
            {
                "model": "phi4:14b",
                "family": "Microsoft",
                "status": "ok",
                "prompts_source": "tuned",
                "scores_by_phase": {
                    "judge_qwen": {
                        "scores": {
                            "final": 0.7654,
                            "rougeL_f1": 0.451,
                            "judge_a_mean": 0.97,
                            "judge_b_mean": 0.0,
                            "judges_delta": 0.0,
                        },
                        "latency_ms": {"p95": 24946.0},
                    }
                },
                "same_family_judge_by_phase": {"judge_qwen": False},
            },
            {
                "model": "hermes3:8b",
                "family": "Nous",
                "status": "missing_prompts",
                "prompts_source": "missing_prompts",
            },
        ]
    )
    sweep._print_leaderboard(ledger)
    out = capsys.readouterr().out
    assert "## Autoresearch sweep — 2026-W27" in out
    # generate phase MUST be skipped — nothing was judged there.
    assert "### Phase `generate`" not in out
    # judge_qwen phase block IS present.
    assert "### Phase `judge_qwen`" in out
    assert "| candidate | family | prompts |" in out
    # phi4 row — highest final score sorts first.
    assert "`phi4:14b`" in out
    assert "0.7654" in out
    # missing_prompts row falls through to the failure path.
    assert "⚠ missing_prompts" in out
    # Order matters: phi4 (ok) appears before hermes3 (missing).
    assert out.index("phi4:14b") < out.index("hermes3:8b")


def test_print_leaderboard_sorts_descending_by_final_per_phase(capsys):
    """Two ok candidates with different final scores → higher one appears first
    within each judge phase\'s block."""
    ledger = _v2_ledger_with(
        [
            {
                "model": "lower:7b",
                "family": "X",
                "status": "ok",
                "prompts_source": "tuned",
                "scores_by_phase": {"judge_qwen": {"scores": {"final": 0.50}, "latency_ms": {}}},
                "same_family_judge_by_phase": {"judge_qwen": False},
            },
            {
                "model": "higher:8b",
                "family": "Y",
                "status": "ok",
                "prompts_source": "tuned",
                "scores_by_phase": {"judge_qwen": {"scores": {"final": 0.80}, "latency_ms": {}}},
                "same_family_judge_by_phase": {"judge_qwen": False},
            },
        ]
    )
    sweep._print_leaderboard(ledger)
    out = capsys.readouterr().out
    assert out.index("higher:8b") < out.index("lower:7b")


def test_print_leaderboard_renders_multiple_judge_phases(capsys):
    """Two judge phases (judge_qwen + judge_llama) — each gets its own block
    with independent sort. Same-family flag comes from
    ``same_family_judge_by_phase[phase]``."""
    llama_phase = {
        "name": "judge_llama",
        "mode": "pairwise",
        "judge_a": {"provider": "vllm", "model": "judge-b"},
        "judge_b": {},
    }
    ledger = _v2_ledger_with(
        [
            {
                "model": "llama3.1:8b",
                "family": "Meta",
                "status": "ok",
                "prompts_source": "tuned",
                "scores_by_phase": {
                    "judge_qwen": {"scores": {"final": 0.16}, "latency_ms": {}},
                    "judge_llama": {"scores": {"final": 0.28}, "latency_ms": {}},
                },
                "same_family_judge_by_phase": {
                    "judge_qwen": False,
                    # Meta candidate + Meta judge — the flag SHOULD render.
                    "judge_llama": True,
                },
            }
        ],
        extra_phases=[llama_phase],
    )
    sweep._print_leaderboard(ledger)
    out = capsys.readouterr().out
    # Both judge phase blocks present, generate skipped.
    assert "### Phase `generate`" not in out
    assert "### Phase `judge_qwen`" in out
    assert "### Phase `judge_llama`" in out
    # Same-family flag renders in the judge_llama block for the Meta candidate.
    assert "⚠ same-family" in out


def test_phase_name_strips_judge_config_prefix() -> None:
    """The phase name in the v2 ledger is derived from the judge_config
    filename. Legacy ``judge_config_<name>.yaml`` files strip the prefix;
    new ``judge_<name>.yaml`` files pass through as-is."""
    assert sweep._phase_name_from_judge_config(Path("judge_qwen.yaml")) == "judge_qwen"
    assert sweep._phase_name_from_judge_config(Path("judge_llama.yaml")) == "judge_llama"
    # Legacy prefix (kept for backward-compat with older configs).
    assert sweep._phase_name_from_judge_config(Path("judge_config_foo.yaml")) == "foo"


def test_phase_name_falls_back_to_stem() -> None:
    assert sweep._phase_name_from_judge_config(Path("foo.yaml")) == "foo"
    assert (
        sweep._phase_name_from_judge_config(Path("/abs/path/custom_judges.yaml")) == "custom_judges"
    )


def test_build_ledger_includes_implicit_generate_phase() -> None:
    """``_build_ledger`` prepends a ``generate`` phase entry (mode=inference_only)
    before the operator\'s judge phases. Drift check + leaderboard renderer
    both key on this."""
    judge_cfgs = [
        {"mode": "pairwise", "judge_a": {"provider": "vllm", "model": "judge-a"}},
    ]
    ledger = sweep._build_ledger(
        week_id="2026-W27",
        reference_id="silver_x",
        dataset_id="curated_5feeds_smoke_v2",
        judge_phase_names=["judge_qwen"],
        judge_cfgs=judge_cfgs,
        cohort_rows=[],
    )
    phases = ledger["judges"]["phases"]
    assert phases[0]["name"] == "generate"
    assert phases[0]["mode"] == "inference_only"
    assert phases[1]["name"] == "judge_qwen"
    assert phases[1]["mode"] == "pairwise"
    assert ledger["schema_version"] == 2


def test_parse_prep_cmd_splits_env_from_argv() -> None:
    """``FOO=bar scripts/x.sh arg1`` → env={FOO: bar}, argv=[scripts/x.sh, arg1].
    Multiple leading assignments accumulate; anything after the first
    non-assignment token is argv."""
    env, argv = sweep._parse_prep_cmd(
        "GPU_MODE_START_TIMEOUT=600 A=b scripts/ops/dgx_gpu_mode.sh judging a"
    )
    assert env == {"GPU_MODE_START_TIMEOUT": "600", "A": "b"}
    assert argv == ["scripts/ops/dgx_gpu_mode.sh", "judging", "a"]


def test_parse_prep_cmd_no_env_still_ok() -> None:
    env, argv = sweep._parse_prep_cmd("scripts/x.sh a b")
    assert env == {}
    assert argv == ["scripts/x.sh", "a", "b"]


def test_parse_prep_cmd_empty_command_raises() -> None:
    with pytest.raises(ValueError):
        sweep._parse_prep_cmd("FOO=bar BAZ=qux")


class _FakeCompletedProcess:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


def _state_ok(cfg_path=None, missing=None):
    return {
        "candidate": None,
        "cfg_path": cfg_path,
        "prompts_source": "tuned",
        "missing_row": missing,
        "phase_outputs": {},
        "wall_clock_by_phase": {},
        "failed_phase": None,
        "failed_status": None,
    }


def test_run_phase_prep_returns_true_on_zero_exit(monkeypatch) -> None:
    """Successful prep_cmd returns True and does not touch any candidate state."""
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **kw: _FakeCompletedProcess(0))
    cands = [{"model": "m1", "family": "F"}]
    state = {"m1": _state_ok()}
    ok = sweep._run_phase_prep("scripts/x.sh judging a", 0, "judge_qwen", cands, state)
    assert ok is True
    assert state["m1"]["failed_phase"] is None
    assert state["m1"]["failed_status"] is None


def test_run_phase_prep_marks_all_healthy_candidates_failed_on_nonzero_exit(monkeypatch) -> None:
    """When prep_cmd fails, every not-yet-failed / not-missing candidate must
    be marked ``failed_phase = <phase>`` and ``failed_status = prep_cmd_failed (exit N)``."""
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **kw: _FakeCompletedProcess(2))
    cands = [
        {"model": "m1", "family": "F"},
        {"model": "m2", "family": "G"},
    ]
    state = {"m1": _state_ok(), "m2": _state_ok()}
    ok = sweep._run_phase_prep("scripts/x.sh judging a", 1, "judge_llama", cands, state)
    assert ok is False
    for m in ("m1", "m2"):
        assert state[m]["failed_phase"] == "judge_llama"
        assert state[m]["failed_status"] == "prep_cmd_failed (exit 2)"


def test_run_phase_prep_does_not_overwrite_prior_failures(monkeypatch) -> None:
    """Critical invariant: a candidate that already has failed_phase (from an
    earlier phase) or missing_row (missing_prompts) must NOT get its failure
    reason clobbered by a later prep_cmd fail."""
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **kw: _FakeCompletedProcess(1))
    cands = [
        {"model": "m1", "family": "F"},
        {"model": "m2", "family": "G"},
        {"model": "m3", "family": "H"},
    ]
    state = {
        "m1": _state_ok(),
        # m2 already failed earlier — must survive untouched
        "m2": {
            **_state_ok(),
            "failed_phase": "judge_qwen",
            "failed_status": "failed",
        },
        # m3 is missing_prompts — must survive untouched
        "m3": _state_ok(missing={"model": "m3", "status": "missing_prompts"}),
    }
    ok = sweep._run_phase_prep("scripts/x.sh judging b", 2, "judge_llama", cands, state)
    assert ok is False
    assert state["m1"]["failed_phase"] == "judge_llama"
    # m2 keeps its earlier failure
    assert state["m2"]["failed_phase"] == "judge_qwen"
    assert state["m2"]["failed_status"] == "failed"
    # m3 keeps missing_row semantics — no failed_phase assigned
    assert state["m3"]["failed_phase"] is None


def test_model_safe_strips_latest_tag() -> None:
    """``:latest`` is Ollama's implicit default tag — the prompt dir convention
    omits it. _model_safe strips ``:latest`` before mapping ``:`` and ``/`` to
    ``_`` so ``mistral-small3.2:latest`` → ``mistral-small3.2`` (matches
    the existing prompt dir on disk)."""
    assert sweep._model_safe("mistral-small3.2:latest") == "mistral-small3.2"
    assert sweep._model_safe("qwen3.6:latest") == "qwen3.6"
    # Non-latest tags are preserved (with ``:`` → ``_`` mapping).
    assert sweep._model_safe("qwen3.5:9b") == "qwen3.5_9b"
    assert sweep._model_safe("llama3.1:8b") == "llama3.1_8b"


def test_print_leaderboard_renders_cross_phase_contested_section(capsys) -> None:
    """With 2+ judge phases and a contested candidate (Δ > 0.30), the
    leaderboard renders a dedicated ``Cross-phase contestation`` section
    listing that candidate."""
    llama_phase = {
        "name": "judge_llama",
        "mode": "pairwise",
        "judge_a": {"provider": "vllm", "model": "judge-b"},
        "judge_b": {},
    }
    ledger = _v2_ledger_with(
        [
            {
                "model": "contested:1b",
                "family": "X",
                "status": "ok",
                "prompts_source": "tuned",
                "scores_by_phase": {
                    "judge_qwen": {
                        "scores": {"final": 0.10, "judge_a_mean": 0.00},
                        "latency_ms": {},
                    },
                    "judge_llama": {
                        "scores": {"final": 0.40, "judge_a_mean": 0.58},
                        "latency_ms": {},
                    },
                },
                "same_family_judge_by_phase": {
                    "judge_qwen": False,
                    "judge_llama": False,
                },
                "cross_phase_jA": {"judge_qwen": 0.0, "judge_llama": 0.58},
                "cross_phase_delta": 0.58,
                "cross_phase_contested": True,
            },
            {
                "model": "agreed:1b",
                "family": "Y",
                "status": "ok",
                "prompts_source": "tuned",
                "scores_by_phase": {
                    "judge_qwen": {
                        "scores": {"final": 0.20, "judge_a_mean": 0.10},
                        "latency_ms": {},
                    },
                    "judge_llama": {
                        "scores": {"final": 0.22, "judge_a_mean": 0.10},
                        "latency_ms": {},
                    },
                },
                "same_family_judge_by_phase": {
                    "judge_qwen": False,
                    "judge_llama": False,
                },
                "cross_phase_jA": {"judge_qwen": 0.10, "judge_llama": 0.10},
                "cross_phase_delta": 0.0,
                "cross_phase_contested": False,
            },
        ],
        extra_phases=[llama_phase],
    )
    sweep._print_leaderboard(ledger)
    out = capsys.readouterr().out
    # Section rendered
    assert "Cross-phase contestation" in out
    # Contested candidate listed
    assert "`contested:1b`" in out
    # Δ column shown
    assert "0.580" in out
    # Agreed candidate NOT in the contested section (still appears in phase
    # tables above, but the contested section only lists contested rows).
    tail = out[out.index("Cross-phase contestation") :]
    assert "`agreed:1b`" not in tail


def test_aggregate_rows_emits_cross_phase_fields_for_ok_rows() -> None:
    """_aggregate_rows must decorate each ok candidate with cross_phase_jA,
    cross_phase_delta, and cross_phase_contested. Missing_prompts / failed
    candidates don't get these keys."""
    cands = [
        {"model": "ok:1b", "family": "F"},
        {"model": "miss:1b", "family": "G"},
    ]
    state = {
        "ok:1b": {
            **_state_ok(),
            "phase_outputs": {
                "judge_qwen": {
                    "scores": {"final": 0.1, "judge_a_mean": 0.0},
                    "latency_ms": {},
                },
                "judge_llama": {
                    "scores": {"final": 0.4, "judge_a_mean": 0.58},
                    "latency_ms": {},
                },
            },
            "wall_clock_by_phase": {"generate": 5.0, "judge_qwen": 1.0, "judge_llama": 1.0},
        },
        "miss:1b": {
            **_state_ok(missing={"model": "miss:1b", "family": "G", "status": "missing_prompts"}),
        },
    }
    judge_cfgs = [
        {"mode": "pairwise", "judge_a": {"provider": "vllm", "model": "judge-a"}},
        {"mode": "pairwise", "judge_a": {"provider": "vllm", "model": "judge-b"}},
    ]
    rows = sweep._aggregate_rows(
        order_of_candidates=cands,
        per_model_state=state,
        judge_phase_names=["judge_qwen", "judge_llama"],
        judge_cfgs=judge_cfgs,
        judge_families=[],
    )
    ok_row = next(r for r in rows if r["model"] == "ok:1b")
    assert ok_row["cross_phase_jA"] == {"judge_qwen": 0.0, "judge_llama": 0.58}
    assert round(ok_row["cross_phase_delta"], 4) == 0.58
    assert ok_row["cross_phase_contested"] is True

    miss_row = next(r for r in rows if r["model"] == "miss:1b")
    assert "cross_phase_delta" not in miss_row


def test_ollama_reasoning_auto_detect() -> None:
    """OllamaChatJudge auto-applies ``reasoning_effort='low'`` + ``max_tokens=1024``
    for known reasoning-tuned models (gpt-oss, qwen3.6). Non-reasoning models
    (llama3.1:8b, gemma) receive neither knob."""
    from podcast_scraper.evaluation.judges.ollama_chat import OllamaChatJudge

    # Capture the outgoing payload without hitting the network.
    class _Captured:
        def __init__(self) -> None:
            self.payload: dict = {}

        def post(self, url, json):  # noqa: A002
            self.payload = json

            class _R:
                def json(self_inner):
                    return {"choices": [{"message": {"content": "5"}}]}

            return _R()

    for model, want_effort in (
        ("gpt-oss:120b", "low"),
        ("qwen3.6:latest", "low"),
        ("llama3.1:8b", None),
    ):
        cap = _Captured()
        j = OllamaChatJudge(model=model, api_base="http://x/v1", client=cap)
        j.raw(user_content="hi")
        got_effort = cap.payload.get("reasoning_effort")
        assert got_effort == want_effort, f"{model} → got={got_effort} want={want_effort}"
        if want_effort is not None:
            assert cap.payload.get("max_tokens") == 1024


def test_check_drift_missing_thresholds_returns_no_thresholds_report(tmp_path, capsys) -> None:
    """When --thresholds points at a missing file, the drift script exits 0
    and writes a ``no_thresholds`` report instead of raising FileNotFoundError.
    Guards the workflow against the intentional drift_thresholds.yaml deletion
    (post-2026-W27 methodology reset)."""
    import json as _json
    import subprocess
    import sys as _sys

    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts/baselines/check_autoresearch_drift.py"
    out_path = tmp_path / "report.json"
    missing_thresholds = tmp_path / "does_not_exist.yaml"
    proc = subprocess.run(
        [
            _sys.executable,
            str(script),
            "--thresholds",
            str(missing_thresholds),
            "--output",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    assert out_path.is_file()
    report = _json.loads(out_path.read_text())
    assert report["status"] == "no_thresholds"
    assert report["breaches"] == []
    assert any("SKIPPED" in msg for msg in report["informational"])
