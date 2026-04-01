#!/usr/bin/env python3
"""Silver reference selection: pairwise LLM judge comparing two candidate summaries.

For each episode in the dataset, presents both summaries to two judges (OpenAI + Anthropic)
and asks which is better. Aggregates episode-level preferences into an overall winner.

Usage:
    python scripts/eval/pairwise_judge.py \\
        --candidate-a data/eval/runs/silver_candidate_openai_gpt54_smoke_v1 \\
        --candidate-b data/eval/runs/silver_candidate_anthropic_sonnet46_smoke_v1 \\
        --transcripts data/eval/materialized/curated_5feeds_smoke_v1 \\
        [--output results/pairwise_gpt54_vs_sonnet46.json]

    Or via make:
        make silver-pairwise \\
            CANDIDATE_A=silver_candidate_openai_gpt54_smoke_v1 \\
            CANDIDATE_B=silver_candidate_anthropic_sonnet46_smoke_v1

Environment (loaded from .env then .env.autoresearch):
    AUTORESEARCH_JUDGE_OPENAI_API_KEY   — OpenAI judge key
    AUTORESEARCH_JUDGE_ANTHROPIC_API_KEY — Anthropic judge key
    AUTORESEARCH_ALLOW_PRODUCTION_KEYS=1 — fallback to OPENAI_API_KEY / ANTHROPIC_API_KEY
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from podcast_scraper.evaluation.autoresearch_track_a import (  # noqa: E402
    load_local_dotenv_files,
    resolve_judge_anthropic_key,
    resolve_judge_openai_key,
    summary_text_from_prediction,
)

logger = logging.getLogger(__name__)

RUBRIC = """
You are comparing two candidate summaries of the same podcast episode transcript.
Evaluate which summary better satisfies all three dimensions:

1. **Topic coverage** — Main themes appear; nothing central is missing.
2. **Factual accuracy** — No contradictions or invented facts vs. the transcript.
3. **Conciseness** — Reasonable length; no fluff, repetition, or padding.

A higher-quality summary covers more of the episode accurately and concisely.
"""


def _pairwise_user_message(*, transcript: str, summary_a: str, summary_b: str) -> str:
    max_chars = 28_000
    t = transcript if len(transcript) <= max_chars else transcript[:max_chars]
    return (
        "Compare these two summaries of the podcast episode transcript below.\n\n"
        f"### Rubric\n{RUBRIC}\n\n"
        f"### Transcript (may be truncated)\n{t}\n\n"
        f"### Summary A\n{summary_a}\n\n"
        f"### Summary B\n{summary_b}\n\n"
        "Reply with a single JSON object only, no markdown:\n"
        '{"winner": "A" | "B" | "tie", "confidence": <float 0.0-1.0>, '
        '"reason": "<one short sentence>"}\n'
        "winner A = Summary A is clearly better; B = Summary B; tie = too close to call."
    )


def _call_openai_pairwise(*, api_key: str, model: str, user_content: str) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[{"role": "user", "content": user_content}],
    )
    content = (resp.choices[0].message.content or "").strip()
    if content.startswith("```"):
        import re

        content = re.sub(r"^```[a-zA-Z0-9]*\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    return json.loads(content)


def _call_anthropic_pairwise(*, api_key: str, model: str, user_content: str) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=512,
        temperature=0.0,
        messages=[{"role": "user", "content": user_content}],
    )
    content = "".join(block.text for block in msg.content if hasattr(block, "text")).strip()
    if content.startswith("```"):
        import re

        content = re.sub(r"^```[a-zA-Z0-9]*\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    return json.loads(content)


def _load_predictions(run_dir: Path) -> dict[str, str]:
    """Load predictions.jsonl from a run dir, return {episode_id: summary_text}."""
    preds_path = run_dir / "predictions.jsonl"
    if not preds_path.is_file():
        raise FileNotFoundError(f"predictions.jsonl not found in {run_dir}")
    out: dict[str, str] = {}
    with preds_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pred = json.loads(line)
            eid = pred.get("episode_id", "")
            text = summary_text_from_prediction(pred)
            if eid and text:
                out[eid] = text
    return out


def _load_transcripts(transcripts_dir: Path, episode_ids: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for eid in episode_ids:
        p = transcripts_dir / f"{eid}.txt"
        if not p.is_file():
            raise FileNotFoundError(f"Transcript missing: {p}")
        out[eid] = p.read_text(encoding="utf-8")
    return out


def run_pairwise(
    *,
    candidate_a_dir: Path,
    candidate_b_dir: Path,
    transcripts_dir: Path,
    openai_key: str,
    anthropic_key: str,
    judge_openai_model: str = "gpt-4o-mini",
    judge_anthropic_model: str = "claude-haiku-4-5",
) -> dict:
    """Run pairwise comparison, return full results dict."""
    preds_a = _load_predictions(candidate_a_dir)
    preds_b = _load_predictions(candidate_b_dir)

    episode_ids = sorted(set(preds_a) & set(preds_b))
    if not episode_ids:
        raise ValueError(
            f"No common episode_ids between {candidate_a_dir.name} and {candidate_b_dir.name}"
        )
    logger.info("Comparing %d episodes: %s", len(episode_ids), episode_ids)

    transcripts = _load_transcripts(transcripts_dir, episode_ids)

    episodes: list[dict] = []
    votes_a = 0
    votes_b = 0
    ties = 0

    for eid in episode_ids:
        logger.info("Judging episode %s …", eid)
        transcript = transcripts[eid]
        summary_a = preds_a[eid]
        summary_b = preds_b[eid]
        user_msg = _pairwise_user_message(
            transcript=transcript, summary_a=summary_a, summary_b=summary_b
        )

        result_openai = _call_openai_pairwise(
            api_key=openai_key, model=judge_openai_model, user_content=user_msg
        )
        result_anthropic = _call_anthropic_pairwise(
            api_key=anthropic_key, model=judge_anthropic_model, user_content=user_msg
        )

        winner_openai = result_openai.get("winner", "tie")
        winner_anthropic = result_anthropic.get("winner", "tie")
        contested = winner_openai != winner_anthropic

        # Resolve: if both agree, use their answer; if contested, mark as tie
        if not contested:
            episode_winner = winner_openai
        else:
            episode_winner = "tie"

        if episode_winner == "A":
            votes_a += 1
        elif episode_winner == "B":
            votes_b += 1
        else:
            ties += 1

        episode_result = {
            "episode_id": eid,
            "winner": episode_winner,
            "contested": contested,
            "judge_openai": result_openai,
            "judge_anthropic": result_anthropic,
            "summary_a_length": len(summary_a.split()),
            "summary_b_length": len(summary_b.split()),
        }
        episodes.append(episode_result)
        logger.info(
            "  Episode %s: openai→%s anthropic→%s resolved→%s contested=%s",
            eid,
            winner_openai,
            winner_anthropic,
            episode_winner,
            contested,
        )

    total = len(episode_ids)
    # Overall winner: most votes; tie if equal
    if votes_a > votes_b:
        overall_winner = "A"
    elif votes_b > votes_a:
        overall_winner = "B"
    else:
        overall_winner = "tie"

    results = {
        "candidate_a": candidate_a_dir.name,
        "candidate_b": candidate_b_dir.name,
        "judge_openai_model": judge_openai_model,
        "judge_anthropic_model": judge_anthropic_model,
        "total_episodes": total,
        "votes_a": votes_a,
        "votes_b": votes_b,
        "ties": ties,
        "overall_winner": overall_winner,
        "episodes": episodes,
    }
    return results


def main() -> int:
    load_local_dotenv_files(REPO_ROOT)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-a",
        type=Path,
        required=True,
        help="Path to run dir for candidate A (must contain predictions.jsonl)",
    )
    parser.add_argument(
        "--candidate-b",
        type=Path,
        required=True,
        help="Path to run dir for candidate B",
    )
    parser.add_argument(
        "--transcripts",
        type=Path,
        required=True,
        help="Path to materialized transcripts dir (e.g. curated_5feeds_smoke_v1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON results to this path (default: print to stdout only)",
    )
    parser.add_argument(
        "--judge-openai-model",
        default="gpt-4o-mini",
        help="OpenAI judge model (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--judge-anthropic-model",
        default="claude-haiku-4-5",
        help="Anthropic judge model (default: claude-haiku-4-5)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    candidate_a = (
        args.candidate_a if args.candidate_a.is_absolute() else REPO_ROOT / args.candidate_a
    )
    candidate_b = (
        args.candidate_b if args.candidate_b.is_absolute() else REPO_ROOT / args.candidate_b
    )
    transcripts = (
        args.transcripts if args.transcripts.is_absolute() else REPO_ROOT / args.transcripts
    )

    for p, label in [
        (candidate_a, "candidate-a"),
        (candidate_b, "candidate-b"),
        (transcripts, "transcripts"),
    ]:
        if not p.exists():
            logger.error("Path not found for --%s: %s", label, p)
            return 1

    try:
        openai_key = resolve_judge_openai_key()
        anthropic_key = resolve_judge_anthropic_key()
    except Exception as e:
        logger.error("%s", e)
        return 1

    try:
        results = run_pairwise(
            candidate_a_dir=candidate_a,
            candidate_b_dir=candidate_b,
            transcripts_dir=transcripts,
            openai_key=openai_key,
            anthropic_key=anthropic_key,
            judge_openai_model=args.judge_openai_model,
            judge_anthropic_model=args.judge_anthropic_model,
        )
    except Exception as e:
        logger.error("Pairwise comparison failed: %s", e, exc_info=True)
        return 1

    output_json = json.dumps(results, indent=2)

    if args.output:
        out_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_json, encoding="utf-8")
        logger.info("Results written to %s", out_path)

    # Always print summary to stderr and winner to stdout
    w = results["overall_winner"]
    a_name = results["candidate_a"]
    b_name = results["candidate_b"]
    winner_name = a_name if w == "A" else (b_name if w == "B" else "TIE")
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"PAIRWISE RESULT: {a_name} vs {b_name}", file=sys.stderr)
    print(
        f"  Votes A: {results['votes_a']}  Votes B: {results['votes_b']}  Ties: {results['ties']}",
        file=sys.stderr,
    )
    print(f"  Overall winner: {winner_name}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)
    print(winner_name)  # machine-readable winner on stdout

    return 0


if __name__ == "__main__":
    sys.exit(main())
