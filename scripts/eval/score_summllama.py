"""Score SummLlama predictions with full v2 framework (ROUGE + judges + final scalar)."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "src")

from podcast_scraper.evaluation.autoresearch_track_a import (
    combine_track_a_scalar,
    load_judge_config,
    load_local_dotenv_files,
    mean_judge_scores,
    resolve_judge_anthropic_key,
    resolve_judge_openai_key,
    rouge_weight_from_env,
)
from podcast_scraper.evaluation.scorer import score_run

load_local_dotenv_files(Path.cwd())
os.environ.setdefault("AUTORESEARCH_ALLOW_PRODUCTION_KEYS", "1")

run_id = sys.argv[1]
dataset_id = sys.argv[2]
ref_id = sys.argv[3]

run_dir = Path(f"data/eval/runs/{run_id}")
ref_dir = Path(f"data/eval/references/silver/{ref_id}")

# ROUGE
metrics = score_run(
    predictions_path=run_dir / "predictions.jsonl",
    dataset_id=dataset_id,
    run_id=run_id,
    reference_paths={ref_id: ref_dir},
    task="summarization",
)
rouge_l = metrics["vs_reference"][ref_id]["rougeL_f1"]
print(f"ROUGE-L: {rouge_l*100:.2f}%")

# Judges
predictions = [
    json.loads(line)
    for line in (run_dir / "predictions.jsonl").read_text().splitlines()
    if line.strip()
]
rubric = Path("autoresearch/bundled_prompt_tuning/eval/rubric.md").read_text()
judge_cfg = load_judge_config(Path("autoresearch/bundled_prompt_tuning/eval/judge_config.yaml"))

judge_mean, any_contested, outcomes = mean_judge_scores(
    predictions=predictions,
    rubric=rubric,
    judge_cfg=judge_cfg,
    dataset_id=dataset_id,
    eval_root=Path("data/eval"),
    openai_key=resolve_judge_openai_key(),
    anthropic_key=resolve_judge_anthropic_key(),
)
print(f"judge_mean: {judge_mean:.4f}, contested: {any_contested}")

rouge_w = rouge_weight_from_env()
final = combine_track_a_scalar(
    rouge_l_f1=rouge_l,
    judge_mean=judge_mean,
    contested=any_contested,
    rouge_weight=rouge_w,
)
print(f"FINAL ({rouge_w:.2f} rouge + {1-rouge_w:.2f} judge): {final:.4f}")
