#!/usr/bin/env python3
"""Weekly autoresearch sweep driver — runs the cohort + writes the ledger.

Design (post-2026-W27 refactor): the sweep is a clean two-stage pipeline.

  Stage 1 — GENERATE (Ollama, no judging).
    Each candidate is an Ollama model. We run
    ``run_experiment.py --force`` per candidate (inference + ROUGE metrics,
    no LLM judges) to
    materialize ``predictions.jsonl`` — inference only, no scoring. This
    is candidate work, not judging work. See autoresearch/JUDGING.md
    for why we do NOT judge on Ollama here: the available Ollama models
    on the DGX (gemma3:27b, mistral-small:24b) are same-tier as the
    candidates being tested, so any judging done by them is peer
    review, not authoritative judging.

  Stage 2 — JUDGE (vLLM, one at a time via GPU swap).
    Each ``--judge-configs`` entry names a real judge (Qwen3-30B-A3B,
    Llama-3.3-70B, etc.) strictly bigger/stronger than any candidate.
    For each judge in turn: run ``prep_cmd`` (swaps the vLLM into
    place), then re-judge every candidate's predictions (via score.py
    ``--rejudge`` — reuses stage-1 output, no re-inference).

The single-GPU DGX can only host one vLLM judge at a time — Qwen and
Llama-70B don't fit together. That's what the serial phases + prep_cmd
GPU swaps are for. See autoresearch/JUDGING.md for the multi-judge
rationale and per-phase-vendor bias analysis.

Each candidate's per-model tuned paragraph templates
(``src/podcast_scraper/prompts/ollama/<model>/summarization/{system,long}_v1.j2``)
are wired into the materialized config — the sweep measures every
candidate on the prompt we\'d ship it with, not on someone else\'s
shared prompt. A candidate that lacks tuned prompts fails fast
(``status: missing_prompts``) rather than silently degrading on a
foreign prompt.

The ledger written to
``data/autoresearch_baselines/autoresearch-YYYY-WNN.json`` records:
  - ``schema_version: 2``
  - ``judges.phases`` — first entry is the implicit ``generate`` phase
    (``mode: inference_only``, no judges), followed by one entry per
    judge phase (``mode: pairwise``, judge_a from the yaml).
  - ``cohort[i].wall_clock_by_phase`` — includes generate + every judge
    phase, so operators see where wall-clock goes.
  - ``cohort[i].scores_by_phase`` — ONLY judge-phase entries (generate
    has no scores). Drift check reads a designated primary phase name
    (see ``check_autoresearch_drift.py::_PRIMARY_PHASE``).

CLI:
  --cohort PATH            Cohort YAML (default: data/autoresearch_baselines/cohort.yaml)
  --base-config PATH       Base experiment YAML (default: Ollama smoke paragraph sweep v1)
  --judge-configs LIST     Comma-separated judge yaml paths (real judges only —
                           the generate phase is implicit, no config needed)
  --reference ID           Silver reference id (default: silver_sonnet46_smoke_v2)
  --output PATH            Weekly ledger output (default: autoresearch-<week>.json)
  --limit N                Run only the first N candidates (overrides cohort.default_limit)
  --print-leaderboard      Print the markdown leaderboard to stdout after the sweep
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Name of the implicit generation phase in the ledger. It records
# wall_clock_by_phase[GENERATE_PHASE_NAME] but has no scores_by_phase
# entry (nothing was judged there).
GENERATE_PHASE_NAME = "generate"


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _model_safe(model: str) -> str:
    """Filesystem/config-safe encoding of an Ollama model tag.

    Ollama treats ``foo`` and ``foo:latest`` as the same model — ``:latest``
    is the default tag when a caller omits one. Our per-model prompt dirs
    (``src/podcast_scraper/prompts/ollama/<safe>/summarization/``) use the
    tag-less form as the canonical convention. Strip ``:latest`` FIRST,
    then map ``:`` → ``_`` and ``/`` → ``_`` for the rest of the tag chars.
    Dots survive (``qwen3.5:9b`` → ``qwen3.5_9b``).
    """
    if model.endswith(":latest"):
        model = model[: -len(":latest")]
    return model.replace(":", "_").replace("/", "_")


def _resolve_per_model_prompts(candidate_model: str) -> tuple[str, str] | None:
    """Return (system_prompt_id, user_prompt_id) for the candidate\'s tuned
    paragraph templates if both exist on disk, else None.

    Convention: ``src/podcast_scraper/prompts/ollama/<model_safe>/summarization/
    {system_v1,long_v1}.j2`` — same dir naming the production summarization
    factory uses. ``model_safe`` replaces ``:`` with ``_`` but preserves dots
    so it matches the existing dir layout (e.g. ``qwen3.5:9b`` → ``qwen3.5_9b``).

    See [[project_autoresearch]] for the harness contract.
    """
    safe = _model_safe(candidate_model)
    prompt_dir = REPO_ROOT / "src/podcast_scraper/prompts/ollama" / safe / "summarization"
    system_file = prompt_dir / "system_v1.j2"
    user_file = prompt_dir / "long_v1.j2"
    if system_file.is_file() and user_file.is_file():
        return (
            f"ollama/{safe}/summarization/system_v1",
            f"ollama/{safe}/summarization/long_v1",
        )
    return None


_LEADERBOARD_ROW_HEADER = (
    "| candidate | family | prompts | final | rougeL | jA | jB | Δjudges " "| p95 (ms) | flag |"
)
_LEADERBOARD_ROW_SEP = "|---|---|---|---:|---:|---:|---:|---:|---:|---|"


def _format_row(
    r: dict[str, Any],
    sc: dict[str, Any],
    lat: dict[str, Any],
    *,
    same_family: bool,
) -> str:
    prompts = r.get("prompts_source", "?")
    flag = "⚠ same-family" if same_family else ""
    return (
        f"| `{r.get('model')}` | {r.get('family', '?')} | {prompts} "
        f"| {sc.get('final', 0):.4f} | {sc.get('rougeL_f1', 0):.4f} "
        f"| {sc.get('judge_a_mean') or 0:.3f} | {sc.get('judge_b_mean') or 0:.3f} "
        f"| {sc.get('judges_delta') or 0:.3f} | {lat.get('p95') or 0:.0f} "
        f"| {flag} |"
    )


def _format_failure_row(r: dict[str, Any]) -> str:
    prompts = r.get("prompts_source", "?")
    return (
        f"| `{r.get('model')}` | {r.get('family', '?')} | {prompts} "
        f"| — | — | — | — | — | — | ⚠ {r.get('status')} |"
    )


def _print_leaderboard(ledger: dict[str, Any]) -> None:
    """Print the markdown leaderboard to stdout.

    v2 ledgers only (post-refactor). Phases with ``mode: inference_only``
    (the implicit ``generate`` phase) are skipped — nothing to score.
    Every judge phase gets its own leaderboard block; rows are sorted by
    that phase\'s ``scores_by_phase[phase]["scores"]["final"]``.
    Candidates with ``status != ok`` show the same failure row in every
    block.
    """
    out: list[str] = []
    out.append(f"## Autoresearch sweep — {ledger.get('week_id')}")
    out.append("")
    out.append(f"**Silver:** `{ledger.get('silver')}`")
    out.append(f"**Dataset:** `{ledger.get('dataset')}`")
    out.append("")

    cohort = ledger.get("cohort") or []
    phases = (ledger.get("judges") or {}).get("phases") or []

    def _pretty(judge: dict[str, Any]) -> str:
        model = judge.get("model")
        provider = judge.get("provider")
        if not model and not provider:
            return "—"
        return f"`{model}` (provider=`{provider}`)"

    for phase in phases:
        pname = phase.get("name", "?")
        mode = phase.get("mode", "?")
        # Generate-only phases have no scores to render.
        if mode == "inference_only":
            continue
        ja = phase.get("judge_a") or {}
        jb = phase.get("judge_b") or {}
        out.append(f"### Phase `{pname}` — mode=`{mode}`")
        out.append("")
        out.append(f"**Judge A:** {_pretty(ja)}")
        out.append(f"**Judge B:** {_pretty(jb)}")
        out.append("")
        out.append(_LEADERBOARD_ROW_HEADER)
        out.append(_LEADERBOARD_ROW_SEP)

        def _sort_key(r: dict[str, Any], _p: str = pname) -> float:
            phase_block = (r.get("scores_by_phase") or {}).get(_p) or {}
            return -((phase_block.get("scores") or {}).get("final") or -1)

        for r in sorted(cohort, key=_sort_key):
            if r.get("status") != "ok":
                out.append(_format_failure_row(r))
                continue
            phase_block = (r.get("scores_by_phase") or {}).get(pname) or {}
            same_family = bool((r.get("same_family_judge_by_phase") or {}).get(pname))
            out.append(
                _format_row(
                    r,
                    phase_block.get("scores") or {},
                    phase_block.get("latency_ms") or {},
                    same_family=same_family,
                )
            )
        out.append("")

    # Cross-phase contested candidates — surface disagreement between judge
    # phases. Only meaningful when we have 2+ judge phases (pairwise mode
    # jA_mean per phase). Threshold matches the within-phase pairwise
    # contestation threshold (0.30 absolute delta).
    judge_phase_names_local = [p.get("name") for p in phases if p.get("mode") != "inference_only"]
    if len(judge_phase_names_local) >= 2:
        contested = sorted(
            (r for r in cohort if r.get("status") == "ok" and r.get("cross_phase_contested")),
            key=lambda r: -(r.get("cross_phase_delta") or 0),
        )
        if contested:
            out.append("### Cross-phase contestation (judge disagreement > 0.30)")
            out.append("")
            out.append(
                "Candidates where the pairwise `judge_a_mean` differs by more "
                "than 0.30 across judge phases. High Δ = judges materially "
                "disagree on quality; worth manual inspection or a cloud "
                "sanity check."
            )
            out.append("")
            hdr = (
                "| candidate | "
                + " | ".join(f"jA `{n}`" for n in judge_phase_names_local)
                + " | Δ |"
            )
            sep = "|---|" + "".join("---:|" for _ in judge_phase_names_local) + "---:|"
            out.append(hdr)
            out.append(sep)
            for r in contested:
                cells = [f"`{r.get('model')}`"]
                jA = r.get("cross_phase_jA") or {}
                for n in judge_phase_names_local:
                    v = jA.get(n)
                    cells.append(f"{v:.3f}" if isinstance(v, (int, float)) else "—")
                cells.append(f"**{r.get('cross_phase_delta', 0):.3f}**")
                out.append("| " + " | ".join(cells) + " |")
            out.append("")

    print("\n".join(out))


def _materialize_candidate_config(
    base_config_path: Path, candidate_model: str, out_dir: Path
) -> tuple[Path, str] | tuple[None, str]:
    """Read base config, override backend.model + prompts + id, write to out_dir.

    Returns ``(config_path, "tuned")`` when the candidate\'s per-model paragraph
    prompts exist (the normal sweep case — see _resolve_per_model_prompts).
    Returns ``(None, "missing_prompts")`` when they don\'t, so the caller can
    record a clean failure row instead of silently scoring the candidate on
    someone else\'s prompt (the W27 problem).
    """
    resolved = _resolve_per_model_prompts(candidate_model)
    if resolved is None:
        return None, "missing_prompts"
    system_id, user_id = resolved

    base = load_yaml(base_config_path)
    base["backend"]["model"] = candidate_model
    base["prompts"] = {"system": system_id, "user": user_id}
    safe = _model_safe(candidate_model).replace(".", "_")
    base["id"] = f"autoresearch_prompt_ollama_{safe}_smoke_paragraph_sweep"
    out_path = out_dir / f"config_{safe}.yaml"
    out_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return out_path, "tuned"


_ENV_VAR_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _parse_prep_cmd(prep_cmd: str) -> tuple[dict[str, str], list[str]]:
    """Split ``prep_cmd`` into (env-overrides, argv) — no shell.

    yaml prep_cmd values look like ``KEY=VAL scripts/ops/foo.sh arg1 arg2``.
    We split on shell tokens (shlex), promote leading ``KEY=VAL`` tokens to
    the child\'s environment, and keep the rest as argv. This lets us call
    ``subprocess.run(argv, env=...)`` with ``shell=False`` — no arbitrary
    command execution surface (bandit B602), operator-authored yaml is
    still fully expressive.

    Not intended to be a shell reimplementation: unquoted globs, pipes,
    backticks, and ``$VAR`` substitution are NOT supported. Every real
    prep_cmd is a single command with args (see judge_qwen.yaml,
    judge_llama.yaml).
    """
    tokens = shlex.split(prep_cmd)
    env: dict[str, str] = {}
    i = 0
    while i < len(tokens) and _ENV_VAR_ASSIGN.match(tokens[i]):
        key, _, value = tokens[i].partition("=")
        env[key] = value
        i += 1
    argv = tokens[i:]
    if not argv:
        raise ValueError(f"prep_cmd has no command after env assignments: {prep_cmd!r}")
    return env, argv


def _run_phase_prep(
    prep_cmd: str,
    phase_idx: int,
    phase_name: str,
    order_of_candidates: list[dict[str, Any]],
    per_model_state: dict[str, dict[str, Any]],
) -> bool:
    """Run a phase\'s ``prep_cmd`` (GPU-swap hook).

    Returns True on success. On failure, marks every not-yet-failed
    candidate as failed at this phase (so later phases short-circuit via
    the existing ``failed_phase`` check) and returns False — the caller
    should ``continue`` to the next phase or fall through to aggregation.
    """
    logger.info("Phase %d prep: %s", phase_idx + 1, prep_cmd)
    prep_env, prep_argv = _parse_prep_cmd(prep_cmd)
    prep_proc = subprocess.run(
        prep_argv,
        env={**os.environ, **prep_env},
        check=False,
        cwd=str(REPO_ROOT),
    )
    if prep_proc.returncode == 0:
        return True
    logger.error(
        "Phase %d prep_cmd failed (exit %d): %s — "
        "marking remaining candidates failed at phase %s",
        phase_idx + 1,
        prep_proc.returncode,
        prep_cmd,
        phase_name,
    )
    for cand in order_of_candidates:
        st = per_model_state[cand["model"]]
        if st["missing_row"] is None and st["failed_phase"] is None:
            st["failed_phase"] = phase_name
            st["failed_status"] = f"prep_cmd_failed (exit {prep_proc.returncode})"
    return False


def _phase_name_from_judge_config(judge_config_path: Path) -> str:
    """Derive a short phase name from the judge_config filename.

    ``judge_qwen.yaml``      → ``judge_qwen``
    ``judge_llama.yaml``     → ``judge_llama``
    ``judge_config_foo.yaml`` → ``foo`` (legacy naming)
    ``foo.yaml``             → ``foo``

    Used to key per-phase results in the v2 ledger so operators can compare
    "the same candidate under N different judges" side by side.
    """
    stem = judge_config_path.stem
    if stem.startswith("judge_config_"):
        stem = stem[len("judge_config_") :]
    return stem or "phase"


def _run_experiment_for_generate(
    *,
    cfg_path: Path,
    reference_id: str,
) -> tuple[int, float]:
    """Invoke ``run_experiment.py --force`` for one candidate.

    Runs inference + ROUGE metrics, no LLM judges (judges run in later phases).

    Writes predictions.jsonl to the standard results dir for the run id
    encoded in the config. Judge phases later pick that up via ``score.py
    --rejudge``.

    Returns (exit_code, wall_clock_seconds). Only Ollama is supported here
    — the sweep\'s candidate cohort is all Ollama models. Cloud-API-key
    injection (which score.py handles for its callers) is intentionally
    out of scope: candidate inference here is Ollama-only.
    """
    started = time.time()
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/eval/experiment/run_experiment.py"),
        str(cfg_path),
        "--reference",
        reference_id,
        "--log-level",
        "INFO",
        "--force",
    ]
    # run_experiment.py imports scripts.eval.data.materialize_baseline —
    # needs the repo root on PYTHONPATH.
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
        if env.get("PYTHONPATH")
        else str(REPO_ROOT)
    )
    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=False)
    return proc.returncode, time.time() - started


def _run_score_for_candidate(
    *,
    cfg_path: Path,
    judge_config_path: Path,
    reference_id: str,
    output_json: Path,
    rejudge: bool,
) -> tuple[int, float]:
    """Invoke ``make autoresearch-score`` once for one (candidate, judge_config).

    Returns (exit_code, wall_clock_seconds). When ``rejudge`` is True, sets
    ``REJUDGE=1`` so the Makefile target passes ``--rejudge`` to score.py —
    the run skips the generation subprocess and only re-runs the judges
    against existing predictions.jsonl / metrics.json.

    In the current sweep design ALL judge-phase invocations are rejudge
    (generation happened in the earlier ``generate`` phase); this
    parameter is kept for symmetry with the score.py contract.
    """
    started = time.time()
    make_args = [
        "make",
        "autoresearch-score",
        f"CONFIG={cfg_path}",
        f"JUDGE_CONFIG={judge_config_path}",
        f"REFERENCE={reference_id}",
        f"OUTPUT_JSON={output_json}",
    ]
    if rejudge:
        make_args.append("REJUDGE=1")
    proc = subprocess.run(make_args, cwd=str(REPO_ROOT), check=False)
    return proc.returncode, time.time() - started


def _materialize_or_missing(
    *, candidate: dict[str, Any], base_config_path: Path, tmp_dir: Path
) -> tuple[Path | None, str, dict[str, Any] | None]:
    """Materialize per-candidate config or produce a ``missing_prompts`` row.

    Returns (cfg_path, prompts_source, missing_row) where missing_row is
    populated only when the candidate lacks its tuned templates; callers
    should short-circuit on that.
    """
    model = candidate["model"]
    family = candidate["family"]
    cfg_path, prompts_source = _materialize_candidate_config(base_config_path, model, tmp_dir)
    if cfg_path is None:
        logger.error(
            "Candidate %s: no tuned prompts under "
            "src/podcast_scraper/prompts/ollama/%s/summarization/; "
            "skipping (status=missing_prompts)",
            model,
            _model_safe(model),
        )
        return (
            None,
            prompts_source,
            {
                "model": model,
                "family": family,
                "status": "missing_prompts",
                "prompts_source": prompts_source,
                "wall_clock_s": 0.0,
            },
        )
    return cfg_path, prompts_source, None


def _run_candidate_single_phase(
    *,
    candidate: dict[str, Any],
    cfg_path: Path,
    prompts_source: str,
    judge_config_path: Path,
    reference_id: str,
    tmp_dir: Path,
    rejudge: bool,
) -> tuple[dict[str, Any] | None, str, float]:
    """Run ONE candidate through ONE judge phase and return the phase output.

    Returns (breakdown_or_None, status, elapsed_seconds). status is one of
    ``ok`` / ``failed`` / ``no_output``. The caller aggregates across phases
    (phase-outer iteration) so vLLM model swaps happen once per phase, not
    once per candidate.
    """
    model = candidate["model"]
    safe_model = _model_safe(model)
    phase_name = _phase_name_from_judge_config(judge_config_path)
    output_json = tmp_dir / f"output_{safe_model}_{phase_name}.json"

    exit_code, elapsed = _run_score_for_candidate(
        cfg_path=cfg_path,
        judge_config_path=judge_config_path,
        reference_id=reference_id,
        output_json=output_json,
        rejudge=rejudge,
    )
    if exit_code != 0:
        logger.error("Candidate %s phase %s failed (exit %d)", model, phase_name, exit_code)
        return None, "failed", elapsed
    if not output_json.is_file():
        logger.error(
            "Candidate %s phase %s exited 0 but no output JSON at %s",
            model,
            phase_name,
            output_json,
        )
        return None, "no_output", elapsed
    return (
        json.loads(output_json.read_text(encoding="utf-8")),
        "ok",
        elapsed,
    )


def _run_generate_phase(
    *,
    order_of_candidates: list[dict[str, Any]],
    per_model_state: dict[str, dict[str, Any]],
    reference_id: str,
) -> None:
    """Stage 1 — run inference-only for every not-yet-failed candidate.

    Ollama-only. Writes predictions.jsonl to each candidate\'s standard
    results dir (encoded in the config\'s ``id`` field). Judge phases
    consume that via ``score.py --rejudge``. Mutates ``per_model_state``
    with wall_clock + failed_phase/status on generate failure.
    """
    logger.info("=" * 70)
    logger.info(
        "Stage 1 / %s — candidate inference on Ollama, no judging",
        GENERATE_PHASE_NAME,
    )
    logger.info("=" * 70)
    for candidate in order_of_candidates:
        model = candidate["model"]
        state = per_model_state[model]
        if state["missing_row"] is not None:
            continue
        logger.info("Candidate: %s (%s)", model, candidate["family"])
        exit_code, elapsed = _run_experiment_for_generate(
            cfg_path=state["cfg_path"],
            reference_id=reference_id,
        )
        state["wall_clock_by_phase"][GENERATE_PHASE_NAME] = round(elapsed, 1)
        if exit_code != 0:
            logger.error("Candidate %s generate failed (exit %d)", model, exit_code)
            state["failed_phase"] = GENERATE_PHASE_NAME
            state["failed_status"] = f"generate_failed (exit {exit_code})"


def _run_judge_phase(
    *,
    phase_idx: int,
    phase_name: str,
    judge_config_path: Path,
    judge_cfg: dict[str, Any],
    order_of_candidates: list[dict[str, Any]],
    per_model_state: dict[str, dict[str, Any]],
    reference_id: str,
    tmp_dir: Path,
    total_judge_phases: int,
) -> None:
    """Stage 2 — one judge phase across every candidate that has predictions.

    Runs the phase\'s prep_cmd first (GPU swap). On failure, marks all
    remaining candidates failed at this phase and returns early. Then for
    each not-yet-failed candidate, re-judges its stage-1 predictions with
    THIS phase\'s judge and records the score in ``phase_outputs``.
    """
    logger.info("=" * 70)
    logger.info(
        "Stage 2 %d/%d — judge phase %s",
        phase_idx + 1,
        total_judge_phases,
        phase_name,
    )
    logger.info("=" * 70)

    prep_cmd = (judge_cfg.get("prep_cmd") or "").strip()
    if prep_cmd and not _run_phase_prep(
        prep_cmd,
        phase_idx,
        phase_name,
        order_of_candidates,
        per_model_state,
    ):
        return

    for candidate in order_of_candidates:
        model = candidate["model"]
        state = per_model_state[model]
        if state["missing_row"] is not None:
            continue
        if state["failed_phase"] is not None:
            # Candidate already failed at generate or an earlier judge phase.
            continue
        logger.info("Candidate: %s (%s)", model, candidate["family"])
        phase_output, phase_status, elapsed = _run_candidate_single_phase(
            candidate=candidate,
            cfg_path=state["cfg_path"],
            prompts_source=state["prompts_source"],
            judge_config_path=judge_config_path,
            reference_id=reference_id,
            tmp_dir=tmp_dir,
            rejudge=True,
        )
        state["wall_clock_by_phase"][phase_name] = round(elapsed, 1)
        if phase_output is None:
            state["failed_phase"] = phase_name
            state["failed_status"] = phase_status
            continue
        state["phase_outputs"][phase_name] = phase_output


def _aggregate_rows(
    *,
    order_of_candidates: list[dict[str, Any]],
    per_model_state: dict[str, dict[str, Any]],
    judge_phase_names: list[str],
    judge_cfgs: list[dict[str, Any]],
    judge_families: list[str],
) -> list[dict[str, Any]]:
    """Reduce per-candidate state into cohort rows for the ledger.

    Each candidate ends up as one dict:
      - missing_prompts → the pre-built missing_row
      - failed at any phase → failed row with ``failed_phase`` +
        ``failed_status`` + ``wall_clock_by_phase``
      - ok → v2 row with ``scores_by_phase`` (judge phases only) +
        ``same_family_judge_by_phase`` per judge
    """
    rows: list[dict[str, Any]] = []
    for candidate in order_of_candidates:
        model = candidate["model"]
        family = candidate["family"]
        state = per_model_state[model]
        if state["missing_row"] is not None:
            rows.append(state["missing_row"])
            continue
        total_wall = round(sum(state["wall_clock_by_phase"].values()), 1)
        if state["failed_phase"] is not None:
            rows.append(
                {
                    "model": model,
                    "family": family,
                    "status": state["failed_status"],
                    "failed_phase": state["failed_phase"],
                    "prompts_source": state["prompts_source"],
                    "wall_clock_s": total_wall,
                    "wall_clock_by_phase": state["wall_clock_by_phase"],
                }
            )
            continue
        # Cross-phase contestation: for each ok candidate, compute how much
        # judge pairwise-means disagree across the N judge phases. Judge_a
        # 's pairwise mean lives at scores_by_phase[phase]["scores"]["judge_a_mean"]
        # in pairwise mode. Extract per-phase, then delta = max-min. Flag
        # if delta > 0.30 (aligns with the within-phase contestation
        # threshold — see autoresearch/JUDGING.md).
        jA_by_phase: dict[str, float] = {}
        for pname in judge_phase_names:
            block = state["phase_outputs"].get(pname) or {}
            scores = block.get("scores") or {}
            jA = scores.get("judge_a_mean")
            if isinstance(jA, (int, float)):
                jA_by_phase[pname] = float(jA)
        if len(jA_by_phase) >= 2:
            cross_delta = max(jA_by_phase.values()) - min(jA_by_phase.values())
        else:
            cross_delta = 0.0
        cross_contested = cross_delta > 0.30

        rows.append(
            {
                "model": model,
                "family": family,
                "status": "ok",
                "prompts_source": state["prompts_source"],
                "wall_clock_s": total_wall,
                "wall_clock_by_phase": state["wall_clock_by_phase"],
                "scores_by_phase": state["phase_outputs"],
                "same_family_judge_by_phase": {
                    judge_phase_names[i]: family
                    in ((judge_cfgs[i].get("judge_families") or []) or judge_families)
                    for i in range(len(judge_phase_names))
                },
                "cross_phase_jA": jA_by_phase,
                "cross_phase_delta": round(cross_delta, 4),
                "cross_phase_contested": cross_contested,
            }
        )
    return rows


def _build_ledger(
    *,
    week_id: str,
    reference_id: str,
    dataset_id: str | None,
    judge_phase_names: list[str],
    judge_cfgs: list[dict[str, Any]],
    cohort_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the v2 ledger dict — implicit generate phase entry first,
    then one entry per judge phase."""
    phases: list[dict[str, Any]] = [
        {
            "name": GENERATE_PHASE_NAME,
            "mode": "inference_only",
            "judge_a": {},
            "judge_b": {},
        }
    ]
    for i, name in enumerate(judge_phase_names):
        phases.append(
            {
                "name": name,
                "mode": judge_cfgs[i].get("mode", "scalar"),
                "judge_a": judge_cfgs[i].get("judge_a") or {},
                "judge_b": judge_cfgs[i].get("judge_b") or {},
            }
        )
    return {
        "schema_version": 2,
        "week_id": week_id,
        "captured_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "judges": {"phases": phases},
        "silver": reference_id,
        "dataset": dataset_id,
        "cohort": cohort_rows,
    }


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Weekly autoresearch sweep driver.")
    parser.add_argument(
        "--cohort",
        type=Path,
        default=REPO_ROOT / "data/autoresearch_baselines/cohort.yaml",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=(
            REPO_ROOT
            / "data/eval/configs/summarization"
            / "autoresearch_prompt_ollama_smoke_paragraph_sweep_v1.yaml"
        ),
    )
    parser.add_argument(
        "--judge-configs",
        type=str,
        required=True,
        help=(
            "Comma-separated paths to judge_config yamls. The generate phase is "
            "implicit (no config needed); every entry here is a REAL judge. "
            "Example: judge_qwen.yaml,judge_llama.yaml."
        ),
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="silver_sonnet46_smoke_v2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: data/autoresearch_baselines/autoresearch-YYYY-WNN.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only first N candidates (overrides cohort.default_limit)",
    )
    parser.add_argument(
        "--print-leaderboard",
        action="store_true",
        help=(
            "After the sweep, print the markdown leaderboard table to stdout "
            "(same shape the GHA workflow prints to $GITHUB_STEP_SUMMARY)."
        ),
    )
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    args = _build_argparser().parse_args()

    cohort_doc = load_yaml(args.cohort)
    candidates = cohort_doc.get("candidates") or []
    if not candidates:
        logger.error("No candidates in %s", args.cohort)
        return 1

    limit = args.limit
    if limit is None:
        limit = cohort_doc.get("default_limit")
    if limit is not None and limit > 0:
        candidates = candidates[: int(limit)]
        logger.info("Running first %d candidate(s) (limit=%d)", len(candidates), limit)
    else:
        logger.info("Running full cohort (%d candidate(s))", len(candidates))

    judge_families = cohort_doc.get("judge_families") or []
    week_id = dt.datetime.utcnow().strftime("%G-W%V")
    out_path = (
        args.output or REPO_ROOT / "data/autoresearch_baselines" / f"autoresearch-{week_id}.json"
    )

    # Resolve judge config paths (real judges only — generate phase is implicit).
    judge_config_paths = [
        (Path(p) if Path(p).is_absolute() else REPO_ROOT / p)
        for p in [x.strip() for x in args.judge_configs.split(",") if x.strip()]
    ]
    if not judge_config_paths:
        logger.error("--judge-configs value produced no paths")
        return 1
    for jcp in judge_config_paths:
        if not jcp.is_file():
            logger.error("judge_config not found: %s", jcp)
            return 1

    judge_cfgs = [load_yaml(p) for p in judge_config_paths]
    judge_phase_names = [_phase_name_from_judge_config(p) for p in judge_config_paths]
    logger.info(
        "Sweep plan: %s → %s",
        GENERATE_PHASE_NAME,
        " → ".join(judge_phase_names),
    )
    base_cfg = load_yaml(args.base_config)

    per_model_state: dict[str, dict[str, Any]] = {}
    order_of_candidates: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="autoresearch_sweep_") as tmp:
        tmp_dir = Path(tmp)
        # Materialize each candidate\'s config once — reused across phases.
        for candidate in candidates:
            model = candidate["model"]
            cfg_path, prompts_source, missing_row = _materialize_or_missing(
                candidate=candidate, base_config_path=args.base_config, tmp_dir=tmp_dir
            )
            per_model_state[model] = {
                "candidate": candidate,
                "cfg_path": cfg_path,
                "prompts_source": prompts_source,
                "missing_row": missing_row,
                "phase_outputs": {},
                "wall_clock_by_phase": {},
                "failed_phase": None,
                "failed_status": None,
            }
            order_of_candidates.append(candidate)

        _run_generate_phase(
            order_of_candidates=order_of_candidates,
            per_model_state=per_model_state,
            reference_id=args.reference,
        )

        for phase_idx, (judge_config_path, judge_cfg, phase_name) in enumerate(
            zip(judge_config_paths, judge_cfgs, judge_phase_names)
        ):
            _run_judge_phase(
                phase_idx=phase_idx,
                phase_name=phase_name,
                judge_config_path=judge_config_path,
                judge_cfg=judge_cfg,
                order_of_candidates=order_of_candidates,
                per_model_state=per_model_state,
                reference_id=args.reference,
                tmp_dir=tmp_dir,
                total_judge_phases=len(judge_config_paths),
            )

        sweep_results = _aggregate_rows(
            order_of_candidates=order_of_candidates,
            per_model_state=per_model_state,
            judge_phase_names=judge_phase_names,
            judge_cfgs=judge_cfgs,
            judge_families=judge_families,
        )

    ledger = _build_ledger(
        week_id=week_id,
        reference_id=args.reference,
        dataset_id=(base_cfg.get("data") or {}).get("dataset_id"),
        judge_phase_names=judge_phase_names,
        judge_cfgs=judge_cfgs,
        cohort_rows=sweep_results,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    logger.info("Wrote weekly ledger: %s", out_path)

    ok_count = sum(1 for r in sweep_results if r.get("status") == "ok")
    logger.info("Cohort summary: %d/%d candidates OK", ok_count, len(sweep_results))

    if args.print_leaderboard:
        print()  # blank line between log lines and the markdown table
        _print_leaderboard(ledger)

    # Exit 0 even on per-candidate failures — the ledger captures them as
    # ``status != ok`` rows and the drift check / issue management handles
    # surfacing them. Sweep exits non-zero only on driver-level errors.
    return (
        0
        if os.environ.get("AUTORESEARCH_SWEEP_STRICT") != "1"
        else (0 if ok_count == len(sweep_results) else 2)
    )


if __name__ == "__main__":
    sys.exit(main())
