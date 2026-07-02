#!/usr/bin/env python3
"""Weekly autoresearch sweep driver — runs the cohort + writes the ledger.

Reads ``data/autoresearch_baselines/cohort.yaml`` for the candidate list,
runs ``make autoresearch-score`` per candidate (overriding ``backend.model``
in a temp config), collects the per-candidate breakdown JSONs, and emits
one weekly ledger ``data/autoresearch_baselines/autoresearch-YYYY-WNN.json``.

The ledger is what ``check_autoresearch_drift.py`` consumes to detect
week-over-week regressions per candidate.

Each candidate's per-model tuned paragraph templates
(``src/podcast_scraper/prompts/ollama/<model>/summarization/{system,long}_v1.j2``)
are wired into the materialized config — the sweep measures every candidate
on the prompt we'd actually ship it with, not on someone else's shared prompt.
A candidate that lacks tuned prompts fails fast (``status: missing_prompts``)
rather than silently degrading on a foreign prompt.

CLI:
  --cohort PATH            Cohort YAML (default: data/autoresearch_baselines/cohort.yaml)
  --base-config PATH       Base experiment YAML (default: Ollama smoke paragraph sweep v1)
  --judge-config PATH      Judge config (default: judge_config_ollama.yaml)
  --reference ID           Silver reference id (default: silver_sonnet46_smoke_v2)
  --output PATH            Weekly ledger output (default: autoresearch-<week>.json)
  --limit N                Run only the first N candidates (overrides cohort.default_limit)
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


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve_per_model_prompts(candidate_model: str) -> tuple[str, str] | None:
    """Return (system_prompt_id, user_prompt_id) for the candidate's tuned
    paragraph templates if both exist on disk, else None.

    Convention: ``src/podcast_scraper/prompts/ollama/<model_safe>/summarization/
    {system_v1,long_v1}.j2`` — same dir naming the production summarization
    factory uses. ``model_safe`` replaces ``:`` with ``_`` but preserves dots
    so it matches the existing dir layout (e.g. ``qwen3.5:9b`` → ``qwen3.5_9b``).

    See [[project_autoresearch]] for the harness contract.
    """
    safe = candidate_model.replace(":", "_").replace("/", "_")
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

    v1 ledgers (single-phase): one Leaderboard block with a single header
    identifying Judge A/B, one row per candidate.

    v2 ledgers (multi-phase): one Leaderboard block per phase in
    ``ledger["judges"]["phases"]``. Rows are sorted by that phase's
    ``scores_by_phase[phase]["scores"]["final"]``; a candidate with
    ``status != ok`` shows the same failure row in every phase.
    """
    out: list[str] = []
    out.append(f"## Autoresearch sweep — {ledger.get('week_id')}")
    out.append("")
    out.append(f"**Silver:** `{ledger.get('silver')}`")
    out.append(f"**Dataset:** `{ledger.get('dataset')}`")
    out.append("")

    cohort = ledger.get("cohort") or []
    judges = ledger.get("judges") or {}
    phases = judges.get("phases")

    def _pretty(judge: dict[str, Any]) -> str:
        model = judge.get("model")
        provider = judge.get("provider")
        if not model and not provider:
            return "—"
        return f"`{model}` (provider=`{provider}`)"

    if not phases:
        # v1 shape — flat "scores" at top of each cohort row.
        ja = judges.get("judge_a") or {}
        jb = judges.get("judge_b") or {}
        out.append(f"**Judge A:** {_pretty(ja)}")
        out.append(f"**Judge B:** {_pretty(jb)}")
        out.append("")
        out.append("### Leaderboard")
        out.append("")
        out.append(_LEADERBOARD_ROW_HEADER)
        out.append(_LEADERBOARD_ROW_SEP)
        for r in sorted(cohort, key=lambda x: -((x.get("scores") or {}).get("final") or -1)):
            if r.get("status") != "ok":
                out.append(_format_failure_row(r))
                continue
            out.append(
                _format_row(
                    r,
                    r.get("scores") or {},
                    r.get("latency_ms") or {},
                    same_family=bool(r.get("same_family_judge")),
                )
            )
        print("\n".join(out))
        return

    # v2 shape — one block per phase, scores keyed by phase name.
    for phase in phases:
        pname = phase.get("name", "?")
        mode = phase.get("mode", "?")
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

    print("\n".join(out))


def _materialize_candidate_config(
    base_config_path: Path, candidate_model: str, out_dir: Path
) -> tuple[Path, str] | tuple[None, str]:
    """Read base config, override backend.model + prompts + id, write to out_dir.

    Returns ``(config_path, "tuned")`` when the candidate's per-model paragraph
    prompts exist (the normal sweep case — see _resolve_per_model_prompts).
    Returns ``(None, "missing_prompts")`` when they don't, so the caller can
    record a clean failure row instead of silently scoring the candidate on
    someone else's prompt (the W27 problem).
    """
    resolved = _resolve_per_model_prompts(candidate_model)
    if resolved is None:
        return None, "missing_prompts"
    system_id, user_id = resolved

    base = load_yaml(base_config_path)
    base["backend"]["model"] = candidate_model
    base["prompts"] = {"system": system_id, "user": user_id}
    safe = candidate_model.replace(":", "_").replace("/", "_").replace(".", "_")
    base["id"] = f"autoresearch_prompt_ollama_{safe}_smoke_paragraph_sweep"
    out_path = out_dir / f"config_{safe}.yaml"
    out_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return out_path, "tuned"


_ENV_VAR_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _parse_prep_cmd(prep_cmd: str) -> tuple[dict[str, str], list[str]]:
    """Split ``prep_cmd`` into (env-overrides, argv) — no shell.

    yaml prep_cmd values look like ``KEY=VAL scripts/ops/foo.sh arg1 arg2``.
    We split on shell tokens (shlex), promote leading ``KEY=VAL`` tokens to
    the child's environment, and keep the rest as argv. This lets us call
    ``subprocess.run(argv, env=...)`` with ``shell=False`` — no arbitrary
    command execution surface (bandit B602), operator-authored yaml is
    still fully expressive.

    Not intended to be a shell reimplementation: unquoted globs, pipes,
    backticks, and ``$VAR`` substitution are NOT supported. Every real
    prep_cmd is a single command with args (see judge_config_vllm_*.yaml).
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


def _phase_name_from_judge_config(judge_config_path: Path) -> str:
    """Derive a short phase name from the judge_config filename.

    ``judge_config_ollama.yaml`` → ``ollama``
    ``judge_config_vllm.yaml``   → ``vllm``
    ``foo.yaml``                 → ``foo``

    Used to key per-phase results in the v2 ledger so operators can compare
    "the same candidate under 3 different judge panels" side by side.
    """
    stem = judge_config_path.stem
    if stem.startswith("judge_config_"):
        stem = stem[len("judge_config_") :]
    return stem or "phase"


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
            model.replace(":", "_").replace("/", "_"),
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
    safe_model = model.replace(":", "_").replace("/", "_")
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


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

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
        "--judge-config",
        type=Path,
        default=REPO_ROOT
        / "autoresearch/initial_prompt_tuning/prompt_tuning/eval/judge_config_ollama.yaml",
        help=(
            "Path to a single judge_config yaml. Mutually exclusive with "
            "``--judge-configs``. Backward compat for the single-phase sweep."
        ),
    )
    parser.add_argument(
        "--judge-configs",
        type=str,
        default=None,
        help=(
            "Comma-separated paths to multiple judge_config yamls for the "
            "multi-phase sweep. Phase 1 generates + judges; phases 2..N reuse "
            "the same predictions and only re-run judges (via --rejudge). "
            "The v2 ledger records ``scores_by_phase`` per candidate — "
            "operator gets the 3-column comparison (Ollama vs vLLM-A vs "
            "vLLM-B) in one file."
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
            "(same shape the GHA workflow prints to $GITHUB_STEP_SUMMARY). "
            "Designed for local iteration — see ``make autoresearch-sweep-local``."
        ),
    )
    parser.add_argument(
        "--pause-between-phases",
        action="store_true",
        help=(
            "In multi-phase mode, pause before phase 2..N and wait for the "
            "operator to press Enter. Prints the phase's expected judge "
            "config so the operator can swap vLLM to the right model first. "
            "Ignored when running non-interactively (no TTY on stdin). GH "
            "workflow leaves this off and drives docker swaps in yaml steps."
        ),
    )
    parser.add_argument(
        "--rejudge-existing",
        action="store_true",
        help=(
            "Force ``--rejudge`` for every candidate — skip generation, "
            "reuse existing predictions.jsonl / metrics.json, run only "
            "the judges from the current judge_config. Enables the GH "
            "workflow to split multi-judge phases across yaml steps with "
            "docker interstitials between them: one step runs phase 1 "
            "normally, docker swap step brings up vLLM, next step runs "
            "phase 2 with --rejudge-existing. Requires prior generation."
        ),
    )
    args = parser.parse_args()

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

    # Resolve N judge_config paths — multi-phase (``--judge-configs``) or
    # single-phase (``--judge-config``, backward compat).
    if args.judge_configs:
        judge_config_paths = [
            (Path(p) if Path(p).is_absolute() else REPO_ROOT / p)
            for p in [x.strip() for x in args.judge_configs.split(",") if x.strip()]
        ]
        if not judge_config_paths:
            logger.error("--judge-configs value produced no paths")
            return 1
    else:
        judge_config_paths = [args.judge_config]

    for jcp in judge_config_paths:
        if not jcp.is_file():
            logger.error("judge_config not found: %s", jcp)
            return 1

    judge_cfgs = [load_yaml(p) for p in judge_config_paths]
    phase_names = [_phase_name_from_judge_config(p) for p in judge_config_paths]
    is_multi_phase = len(judge_config_paths) > 1
    logger.info(
        "Judge phases (%d): %s",
        len(judge_config_paths),
        " → ".join(phase_names),
    )
    base_cfg = load_yaml(args.base_config)

    # Phase-outer iteration: each phase brings up its judges once, then runs
    # all candidates against them. This matches the vLLM reality on DGX where
    # the container serves ONE model at a time — booting vLLM + loading a
    # 30B/70B model takes real time (~30-90s), so we swap once per phase, not
    # once per candidate. Predictions from phase 1 stay cached in
    # ``data/eval/runs/<run_id>/`` and phase 2..N use ``--rejudge``.
    per_model_state: dict[str, dict[str, Any]] = {}
    order_of_candidates: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="autoresearch_sweep_") as tmp:
        tmp_dir = Path(tmp)
        # Materialize each candidate's config once — reused across phases.
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

        # For each phase, iterate all candidates.
        for phase_idx, judge_config_path in enumerate(judge_config_paths):
            phase_name = phase_names[phase_idx]
            is_first_phase = phase_idx == 0
            logger.info("=" * 70)
            logger.info(
                "Phase %d/%d: %s (%s)",
                phase_idx + 1,
                len(judge_config_paths),
                phase_name,
                "generate + judge" if is_first_phase else "rejudge existing predictions",
            )
            logger.info("=" * 70)

            # Per-phase prep hook. Each judge_config yaml may declare a
            # ``prep_cmd`` — typically the SSH gpu-mode-swap invocation
            # that brings the target vLLM up and flushes competing GPU
            # owners. Run BEFORE the phase's judging starts. Blocks until
            # the swap script exits successfully (or fails the phase).
            prep_cmd = (judge_cfgs[phase_idx].get("prep_cmd") or "").strip()
            if prep_cmd:
                logger.info("Phase %d prep: %s", phase_idx + 1, prep_cmd)
                prep_env, prep_argv = _parse_prep_cmd(prep_cmd)
                prep_proc = subprocess.run(
                    prep_argv,
                    env={**os.environ, **prep_env},
                    check=False,
                    cwd=str(REPO_ROOT),
                )
                if prep_proc.returncode != 0:
                    logger.error(
                        "Phase %d prep_cmd failed (exit %d): %s",
                        phase_idx + 1,
                        prep_proc.returncode,
                        prep_cmd,
                    )
                    return 2

            # Optional interactive pause AFTER the prep hook (belt-and-braces
            # for the operator-driven local flow). GHA workflow leaves the
            # flag off and relies on the automated prep_cmd alone.
            if not is_first_phase and args.pause_between_phases and sys.stdin.isatty():
                ja = judge_cfgs[phase_idx].get("judge_a") or {}
                jb = judge_cfgs[phase_idx].get("judge_b") or {}
                logger.info(
                    "PAUSE: verify the models below are reachable, "
                    "then press Enter to start phase %d.",
                    phase_idx + 1,
                )
                logger.info(
                    "  judge_a: provider=%s model=%s",
                    ja.get("provider", "?"),
                    ja.get("model", "?"),
                )
                logger.info(
                    "  judge_b: provider=%s model=%s",
                    jb.get("provider", "?"),
                    jb.get("model", "?"),
                )
                try:
                    input(f"[Press Enter to start phase {phase_idx + 1} ({phase_name})] ")
                except EOFError:
                    # stdin closed mid-run (e.g. shell exited); proceed
                    # anyway rather than deadlock.
                    logger.warning("stdin closed; skipping pause")

            for candidate in order_of_candidates:
                model = candidate["model"]
                state = per_model_state[model]
                if state["missing_row"] is not None:
                    continue
                if state["failed_phase"] is not None:
                    # This candidate already died in an earlier phase — skip.
                    continue
                logger.info("Candidate: %s (%s)", model, candidate["family"])

                # ``--rejudge-existing`` forces --rejudge on every phase
                # (including phase 1), because in the per-step GHA design
                # phase 1's generation already ran in a prior workflow step
                # and this invocation only re-judges.
                phase_output, phase_status, elapsed = _run_candidate_single_phase(
                    candidate=candidate,
                    cfg_path=state["cfg_path"],
                    prompts_source=state["prompts_source"],
                    judge_config_path=judge_config_path,
                    reference_id=args.reference,
                    tmp_dir=tmp_dir,
                    rejudge=args.rejudge_existing or (not is_first_phase),
                )
                state["wall_clock_by_phase"][phase_name] = round(elapsed, 1)
                if phase_output is None:
                    state["failed_phase"] = phase_name
                    state["failed_status"] = phase_status
                    continue
                state["phase_outputs"][phase_name] = phase_output

        # Aggregate per-candidate rows in the original cohort order.
        sweep_results: list[dict[str, Any]] = []
        for candidate in order_of_candidates:
            model = candidate["model"]
            family = candidate["family"]
            state = per_model_state[model]
            if state["missing_row"] is not None:
                sweep_results.append(state["missing_row"])
                continue
            total_wall = round(sum(state["wall_clock_by_phase"].values()), 1)
            if state["failed_phase"] is not None:
                sweep_results.append(
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
            if not is_multi_phase:
                # v1 shape — flatten single phase's breakdown at top level.
                breakdown = dict(state["phase_outputs"][phase_names[0]])
                breakdown["status"] = "ok"
                breakdown["wall_clock_s"] = total_wall
                breakdown["family"] = family
                breakdown["prompts_source"] = state["prompts_source"]
                breakdown["same_family_judge"] = family in judge_families
                sweep_results.append(breakdown)
                continue
            # v2 shape — scores_by_phase + per-phase same-family flag.
            sweep_results.append(
                {
                    "model": model,
                    "family": family,
                    "status": "ok",
                    "prompts_source": state["prompts_source"],
                    "wall_clock_s": total_wall,
                    "wall_clock_by_phase": state["wall_clock_by_phase"],
                    "scores_by_phase": state["phase_outputs"],
                    "same_family_judge_by_phase": {
                        phase_names[i]: family
                        in ((judge_cfgs[i].get("judge_families") or []) or judge_families)
                        for i in range(len(judge_config_paths))
                    },
                }
            )

    # v1 for single-phase (backward compat with drift check + committed
    # weekly ledgers). v2 for multi-phase — introduces judge_phases[] +
    # scores_by_phase per candidate.
    if is_multi_phase:
        judges_block: dict[str, Any] = {
            "phases": [
                {
                    "name": phase_names[i],
                    "judge_a": judge_cfgs[i].get("judge_a") or {},
                    "judge_b": judge_cfgs[i].get("judge_b") or {},
                    "mode": judge_cfgs[i].get("mode", "scalar"),
                }
                for i in range(len(judge_config_paths))
            ]
        }
        schema_version = 2
    else:
        judges_block = {
            "judge_a": judge_cfgs[0].get("judge_a") or {},
            "judge_b": judge_cfgs[0].get("judge_b") or {},
        }
        schema_version = 1

    ledger = {
        "schema_version": schema_version,
        "week_id": week_id,
        "captured_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "judges": judges_block,
        "silver": args.reference,
        "dataset": (base_cfg.get("data") or {}).get("dataset_id"),
        "cohort": sweep_results,
    }

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
    # surfacing them. Sweep exits non-zero only on driver-level errors
    # (no cohort, missing configs, etc.).
    return (
        0
        if os.environ.get("AUTORESEARCH_SWEEP_STRICT") != "1"
        else (0 if ok_count == len(sweep_results) else 2)
    )


if __name__ == "__main__":
    sys.exit(main())
