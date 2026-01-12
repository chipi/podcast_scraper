# Quality Evaluation Guide

This guide details the methodology, tools, and processes for evaluating the **quality** of the podcast scraper's outputs—specifically transcript cleaning and summarization.

Unlike functional testing (which ensures the code runs without crashing), quality evaluation measures how **good** the results are using both human-in-the-loop and automated metrics.

---

## 1. Evaluation Philosophy

We evaluate quality at three distinct levels:

| Level | Focus | How |
| :--- | :--- | :--- |
| **Unit-Level** | Plumbing and components | Regex checks, length validation, and edge case unit tests. |
| **Functional-Level** | End-to-end execution | Ensuring the pipeline produces non-empty, well-formatted results. |
| **Quality-Level** | Accuracy and synthesis | Measuring faithfulness, coverage, and ROUGE scores against a baseline. |

---

## 2. The Golden Dataset

Quality is measured against a "Golden Dataset"—a collection of episodes with human-verified ground truth. This dataset is stored in `data/eval/`.

### Dataset Structure

* `transcript.raw.txt`: Original Whisper output.
* `transcript.cleaned.txt`: Manually corrected/verified clean transcript.
* `summary.gold.long.txt`: Detailed human-written reference summary.
* `summary.gold.short.txt`: Concise human-written reference summary.
* `metadata.json`: Verified hosts and guest names.

See [ADR-026](../adr/ADR-026-explicit-golden-dataset-versioning.md) for the versioning strategy of this dataset.

---

## 3. Automated Evaluation Tools

We provide specialized scripts to automate quality checks.

### A. Summarization Evaluation (`eval_summaries.py`)

This script benchmarks summarization quality by comparing model output against the golden reference.

**Key Metrics:**

* **ROUGE-1/2/L**: Measures overlap with the human reference summary.
* **Compression Ratio**: Ratio of transcript length to summary length (Ideal: 5x–15x).
* **Repetition Score**: Detects n-gram loops and circular logic.
* **Keyword Coverage**: Verifies that key topics from the transcript appear in the summary.

**Usage:**

```bash
python scripts/eval/eval_summaries.py --map-model bart-large --reduce-model long-fast
```

### B. Cleaning Evaluation (`eval_cleaning.py`)

This script measures how effectively the pipeline removes sponsors, ads, and outro boilerplate.

**Key Metrics:**

* **Removal Statistics**: Character and word reduction percentages.
* **Pattern Detection**: Counts sponsor phrases (e.g., "sponsored by") before and after cleaning.
* **Brand Detection**: Verifies removal of known podcast sponsors (e.g., "Justworks", "Figma").

**Usage:**

```bash
python scripts/eval/eval_cleaning.py
```

---

## 4. Manual Evaluation (Human-in-the-loop)

Automated metrics like ROUGE can be misleading (e.g., a summary can have high overlap but be unreadable). We use a manual rubric to score summaries on a 1–5 scale.

### Rubric Categories

1. **Coverage**: Does it capture all major segments and insights?
2. **Faithfulness**: Does it avoid hallucinations and misattributions?
3. **Clarity**: Is the structure logical and the language natural?
4. **Conciseness**: Is it free of filler while maintaining detail?

Use the **[Manual Eval Checklist](https://github.com/chipi/podcast_scraper/blob/main/data/eval/MANUAL_EVAL_CHECKLIST.md)** to track these scores in a CSV for regression analysis.

---

## 5. Regression Testing Strategy

Whenever you change a model, tweak a prompt template, or update cleaning regexes:

1. **Run the Baseline**: Run the evaluation scripts on the current state and save the JSON results.
2. **Apply Changes**: Implement your logic or model changes.
3. **Run Comparison**: Run the evaluation scripts again.
4. **Analyze Deltas**:
    * **ROUGE-L** drop by more than 0.03? (Potential regression)
    * **Generation Time** increase significantly?
    * **Repetition** flags increase?

For complex changes, use the **[AI Quality & Experimentation Platform](../prd/PRD-007-ai-quality-experiment-platform.md)** (PRD-007) to manage these benchmarks.

---

## 6. Scaling for Long Content

When evaluating on very long transcripts (30k+ tokens), the **MAP-REDUCE** strategy is critical. Evaluations on these episodes should pay special attention to "Lost in the Middle" syndrome—where the model remembers the beginning and end but misses the core middle sections.

For more on hardware optimization during evaluation, see the **[ML Provider Reference](ML_PROVIDER_REFERENCE.md#hardware-acceleration-scaling)**.
