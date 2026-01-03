# Testing & Evaluation of Your Summarization Pipeline

You're running a local, multi-stage summarization pipeline on an M4 Pro.
To make this useful long term, you need a way to test, compare, and avoid regressions when you:

- change models,
- tweak chunking,
- add cleaning steps,
- or swap BART ↔ local LLM.

This section covers:

1. What to test (levels)
2. How to build a small "golden" evaluation set
3. Manual evaluation checklist
4. Automatic metrics (ROUGE, etc.)
5. Reference-free checks (no ground-truth required)
6. Regression testing when you change the system

---

## 1. What to Test (Levels)

Think of 3 levels:

### 1. Unit-level (plumbing)

Test the pieces:

- **Chunker:**
  - produces overlapping chunks
  - no empty chunks
  - reconstructing them roughly recovers original length
- **Cleaner:**
  - removes timestamps correctly
  - strips speaker tags
- **Model wrapper:**
  - returns non-empty string
  - doesn't crash on long text
  - respects max_length / min_length

These tests ensure the system runs reliably.

### 2. Functional-level (does it summarize?)

End-to-end test:

- Given a transcript file, you get:
  - A non-empty summary
  - Reasonable length
  - No obvious garbage ("[CLS]", repeated patterns, etc.)
- Ensure runtime is acceptable for typical episode length.

### 3. Quality-level (is the summary good?)

This is about:

- **Faithfulness** (not hallucinating)
- **Coverage** (captures key points)
- **Readability** (clear, concise, structured)

You'll do this with manual review + automatic metrics.

---

## 2. Build a "Golden Set" of Episodes

Pick a small but representative set:

- ~5–10 episodes with:
  - Different lengths
  - Different topics
  - Different guest styles (monologue, interview, panel)

For each:

1. **Prepare:**
   - Cleaned transcript (input)
   - A human-written "reference summary":

```text
     - Could be:
       - Show notes
       - Detailed episode description
       - Manually written by you (best, but time-consuming)
```

2. **Store in a simple structure, e.g.:**

   ```text
   data/eval/
       ep01/
           transcript.txt
           reference_summary.txt
       ep02/
           transcript.txt
           reference_summary.txt
       ...
   ```

   This becomes your evaluation dataset.

---

## 3. Manual Evaluation Checklist

For each episode, compare model summary vs transcript (and reference, if you have one).
Use a small rubric, e.g. score 1–5 for each dimension:

### A. Coverage (1–5)

- Does it mention all major topics / segments?
- Are any important sections completely missing?

### B. Faithfulness (1–5)

- Does it misrepresent or hallucinate details?
- Does it attribute wrong statements to guest/host?

### C. Clarity & Structure (1–5)

- Is it easy to read?
- Does it have a clear structure (paragraphs, bullets)?
- Is it free of weird artifacts (cut sentences, random tokens)?

### D. Conciseness (1–5)

- Too verbose?
- Too short / telegraphic?
- Good compression (~5–15× shorter than transcript)?

You can keep a simple CSV:

````csv
    episode, model, coverage, faithfulness, clarity, conciseness, notes
    ep01, bart_map_reduce, 4, 4, 3, 4, "Missed one subtopic, otherwise good"
    ep02, bart_map_reduce, 3, 3, 4, 4, "Some minor hallucinations"
    ...
    ```

This is very useful when you later compare against a new model or settings.

---

## 4. Automatic Metrics (if you have reference summaries)

If you have human/reference summaries, you can compute:

- ROUGE-1 / ROUGE-2 / ROUGE-L (recall-oriented, classic for summarization)
- Optionally advanced ones like BERTScore / BLEURT (but ROUGE is enough to start)

### Example: Simple ROUGE with rouge-score library

    ```bash
    pip install rouge-score
    ```

    ```python
```python

    from pathlib import Path
    from rouge_score import rouge_scorer

```python

    def load_text(path):
        return Path(path).read_text(encoding="utf-8")

```python

    def rouge_for_pair(pred_path, ref_path):
        pred = load_text(pred_path)
        ref = load_text(ref_path)

```json
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True,
        )
        scores = scorer.score(ref, pred)

```text

        # scores is a dict with precision/recall/fmeasure

```
        return scores

```

    # example

```
        "data/eval/ep01/reference_summary.txt"
    )
    print(scores["rougeL"].fmeasure)
    ```

Use this to:

- Compare different chunk sizes
- Compare different models (BART vs PEGASUS vs local LLM)
- Detect regressions (scores suddenly drop after a code change)

**Important:** ROUGE is not perfect, but good for relative comparisons (A vs B).

---

## 1. Reference-Free Checks (even if you have no reference)

Often you won't have reference summaries for everything.
You can still do cheap, automatic sanity checks:

### A. Length & Compression Ratio

Let:

- `len_transcript_words`
- `len_summary_words`

Compute compression ratio:

- `r = len_transcript_words / max(len_summary_words, 1)`

Check:

- If `r < 2`, summary is basically too long (not much summarization).
- If `r > 50`, summary is extremely short (likely missing detail).
- You can alert or log when summaries are too short/long.

### B. Repetition

- Check for repeated n-grams (like "the company the company the company…").
- Simple heuristic:
  - Split summary into words
  - Count how many times each 3-gram appears
  - If any 3-gram appears > X times, flag as "repetitive".

### C. Keyword Coverage

- Extract top-N important terms from the transcript:
  - e.g., via TF-IDF or simple frequency (excluding stopwords)
- Check how many of these appear in the summary.
- Low coverage → summary might be missing core topics.

Even these naive checks help you catch really bad outputs automatically.

---

## 2. Regression Testing Strategy

Once you have:

- A golden set (transcript + reference summaries)
- A script that runs summarization and computes ROUGE + basic checks

You can create a simple regression test workflow:

1. Run a command like:

    ```bash
    python scripts/eval_summaries.py \
        --model bart-large \
        --output results/bart_run_01.json
    ```

```text

    Or use a config file:

```
        --config config.yaml \
        --output results/config_run.json
    ```

1. `scripts/eval_summaries.py` does:

   - Loops over `data/eval/*` episode directories
   - Loads `transcript.txt` and `reference_summary.txt` (if available)
   - Cleans transcript using `summarizer.clean_transcript()` (same as pipeline)
   - Generates summary using `summarizer.summarize_long_text()` with configured model
   - Computes ROUGE scores (if reference summary exists)
   - Performs reference-free checks (compression ratio, repetition, keyword coverage)
   - Outputs comprehensive JSON report with per-episode and aggregate metrics

1. Keep baseline numbers in version control, e.g.:

    ```json
```text

    {
    "model": "bart_map_reduce",
    "mean_rougeL_f": 0.29,
    "episodes": 8,
    "avg_runtime_seconds": 41.2
    }
    ```
```
1. When you change model, chunking strategy, or cleaning, re-run eval and compare:

   - Did mean ROUGE go up or down?
   - Did runtime get worse?
   - Did any heuristic check fail?

You can even make a tiny test that fails CI if quality drops below a threshold:

- e.g. `mean_rougeL_f < baseline − 0.03`

---

## 7. Specific Suggestions for Your Setup

Given you:

- Run on an M4 Pro 48GB
- Use local models only
- Handle 30–40k token podcasts

Here's a practical plan:

1. **Create golden set**: Add 5–10 episodes to `data/eval/` with:
   - `transcript.txt` (cleaned transcript)
   - `reference_summary.txt` (human-written reference)
   - `metadata.json` (optional episode metadata)

2. **Manual evaluation**: Use `data/eval/MANUAL_EVAL_CHECKLIST.md` to score each episode
   - Coverage, Faithfulness, Clarity, Conciseness (1-5 each)
   - Track scores in CSV format for comparison over time

3. **Automated evaluation**: Use `scripts/eval_summaries.py`:
   - Generates summaries using your configured models
   - Computes ROUGE scores (if references exist)
   - Performs reference-free checks (compression, repetition, keyword coverage)
   - Outputs JSON report for regression testing

4. **Baseline**: Run evaluation with default BART-large (MAP) + LED (REDUCE):

   ```bash
   python scripts/eval_summaries.py \
       --model bart-large \
       --reduce-model long-fast \
       --output results/baseline_bart_led.json
````

1. **Compare models**: When you change models or settings, re-run on same episodes:

   ```bash
   python scripts/eval_summaries.py \
       --config config_pegasus.yaml \
       --output results/pegasus_run.json
   ```

   Compare ROUGE scores, generation times, and check pass rates.

2. **Regression testing**: Store baseline JSON files in version control and compare:
   - Did mean ROUGE-L F-measure drop below baseline - 0.03?
   - Did generation time increase significantly?
   - Did any reference-free checks fail?

---

## 8. TL;DR

To test and evaluate your summarization:

- Build a small evaluation set with transcripts + reference summaries.
- Have a manual rubric (coverage, faithfulness, clarity, conciseness).
- Use automatic metrics (ROUGE) to compare different models/settings.
- Add reference-free checks (compression ratio, repetition, keyword coverage).
- Store baseline results and use them for regression testing as you iterate.

This gives you a solid feedback loop so you can confidently improve your pipeline instead of guessing.
