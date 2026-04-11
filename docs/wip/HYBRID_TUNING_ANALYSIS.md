# Hybrid pipeline tuning: prod vs hybrid analysis and iteration approach

## 1. What we have in prod vs hybrid

### Prod (baseline_ml_prod_authority_v1)

|Aspect|Setting|
|------|--------|
|Backend|`hf_local` (single HuggingFace pipeline)|
|MAP model|`google/pegasus-cnn_dailymail` (CNN/DM-trained, extractive-style)|
|REDUCE model|`allenai/led-base-16384` (long-context encoder-decoder)|
|MAP params|beams=6, max_new_tokens=200, min=80, no_repeat_ngram=3, rep_pen=1.10|
|REDUCE params|beams=4, max_new_tokens=650, min=220, no_repeat_ngram=3, rep_pen=1.12|
|Chunking|word_chunk_size=900, word_overlap=150|
|Tokenize|map 1024, reduce 4096, truncation=true|
|Dataset|curated_5feeds_benchmark_v1 (10 episodes)|

**Observed (vs silver reference):** avg_tokens ≈ 229, ROUGE-L ≈ 16%, ROUGE-1 ≈ 29%, coverage_ratio ≈ 0.44.

### Hybrid (hybrid_ml_tier1_authority_v1)

|Aspect|Setting|
|------|--------|
|Backend|`hybrid_ml` (MAP → REDUCE via transformers)|
|MAP model|`longt5-base` (LongT5 TGlobal base)|
|REDUCE model|`google/flan-t5-base` (instruction-tuned T5)|
|MAP/REDUCE params|Same numeric params as prod (beams, max/min tokens, ngram, rep_pen)|
|Chunking / tokenize|Same as prod|

**Observed (vs silver reference):** avg_tokens ≈ 522, ROUGE-L ≈ 2.3%, ROUGE-1 ≈ 2.4%, coverage_ratio ≈ 1.01.

### Main difference

- **Output length:** Hybrid outputs ~2.3× more tokens per episode (522 vs 229). Same `max_new_tokens` (650) and `min_new_tokens` (220) are set, so the driver is model behaviour (FLAN-T5 tends to generate longer, more verbose text than LED-base).
- **Lexical overlap:** ROUGE is much lower for hybrid. Silver reference (GPT-4o) is likely closer in style and length to prod (Pegasus + LED) than to FLAN-T5’s more explanatory style.
- **Coverage:** Hybrid’s coverage_ratio > 1 suggests different definition or that the model is “covering” the reference in a different way (e.g. longer summary overlapping in content but not in exact n-grams).

So the gap is not from different config knobs (we copied prod’s params) but from **model choice and output length/style**.

---

## 2. How to tune the hybrid pipeline for better results

### Levers to try (in order of impact / ease)

1. **Cap REDUCE length closer to prod**
   - Lower `reduce_params.max_new_tokens` (e.g. 400–500) and/or raise `min_new_tokens` to avoid very short segments.
   - Goal: bring avg_tokens down toward ~230 so ROUGE isn’t penalized by length mismatch.

2. **REDUCE model**
   - Try a REDUCE model that is more extractive or closer to LED behaviour:
     - Keep `google/flan-t5-base` as baseline.
     - Try e.g. `t5-base` (non-instruction) or another model with similar context length.
   - If you have a larger GPU: try `hybrid_ml_tier2` (e.g. Qwen 7B) and compare.

3. **Length and repetition**
   - Slightly increase `length_penalty` for REDUCE (e.g. 0.8–1.2) to favour shorter summaries.
   - Increase `no_repeat_ngram_size` or `repetition_penalty` if outputs are repetitive.

4. **Chunking**
   - Prod and hybrid both use 900/150. You can try slightly smaller chunks (e.g. 800/120) so MAP sees less context per chunk and REDUCE gets more, shorter segments (might reduce verbosity).

5. **MAP model**
   - LongT5 is already a reasonable choice for long context. Switching MAP to Pegasus in hybrid would mean a different pipeline (not “hybrid” in the current sense); lower priority than fixing REDUCE length and model first.

### Suggested order of experiments

- First: **shorten REDUCE** (max_new_tokens 400–500, maybe min 180) and re-run.
- Second: **different REDUCE model** (e.g. t5-base or another 512/1024 context model) with same shortened length.
- Third: **length_penalty / no_repeat_ngram** tweaks.
- Fourth: **chunking** if length and model changes aren’t enough.

---

## 3. Shorter iterations: smoke tests before full benchmark

### Idea

- Run **small tests** on **5 episodes** (`curated_5feeds_smoke_v1`) with **fast iteration** (one config change → run → compare).
- When a hybrid config **beats or ties prod on smoke** (ROUGE vs silver, no regressions), run the **full benchmark** (10 episodes) once to confirm.

### What already exists

- **curated_5feeds_smoke_v1:** 5 episodes (1 per feed). ~half the runtime of the 10-episode benchmark.
- **baseline_ml_dev_authority** uses smoke for dev; archived **param_sweeps** used smoke for BART/LED tuning.
- **Silver reference:** `silver_gpt4o_benchmark_v1` is on **benchmark** (10 episodes). For smoke we need either:
  - A **silver reference on smoke** (same 5 episodes), or
  - Run **prod and hybrid on smoke** and compare **only intrinsic + relative ROUGE** if we don’t have silver for those 5 episodes.

Checking: silver reference has 10 episodes (benchmark). So for smoke (5 episodes) we can still run hybrid vs prod on the same 5 episodes; ROUGE would be computed only over those 5 if the reference has matching episode_ids. If silver only exists for benchmark, we have two options: (a) create a silver_smoke run (5 episodes) and use it as reference for smoke experiments, or (b) on smoke just compare hybrid vs prod (no silver) for relative gains and latency, then run full benchmark with silver when promising.

Practical approach: **use smoke for relative tuning (hybrid vs prod, same 5 episodes).** If we have silver predictions for the same 5 episodes (subset of silver_gpt4o_benchmark_v1), we can score both with it; otherwise we still get latency and length and can run full benchmark when a config looks good.

### Concrete workflow

1. **Smoke config added:** `data/eval/configs/hybrid_ml_tier1_smoke_v1.yaml` (same as tier1, `dataset_id: curated_5feeds_smoke_v1`).
2. **One-time: prod-on-smoke baseline.** Create a config that is prod with `dataset_id: curated_5feeds_smoke_v1`, run it, then freeze as `baseline_ml_prod_authority_smoke_v1` so smoke experiments can use `BASELINE=baseline_ml_prod_authority_smoke_v1`.
3. **Run hybrid on smoke:**
   `make experiment-run CONFIG=data/eval/configs/hybrid_ml_tier1_smoke_v1.yaml BASELINE=baseline_ml_prod_authority_smoke_v1`
   (Prod baseline is on benchmark; for apples-to-apples smoke comparison we’d run a prod run on smoke too, or use a prod-smoke baseline if one exists. Dev baseline `baseline_ml_dev_authority_smoke_v1` is on smoke but different models. So: either create a **prod-on-smoke** baseline once, or run both prod and hybrid on smoke in the same way and compare runs.)
4. **Iterate:** Edit `hybrid_ml_tier1_smoke_v1.yaml` (e.g. reduce `max_new_tokens` to 450), run again, compare. One lever at a time.
5. **Promotion:** When hybrid on smoke is clearly better or on par with prod, copy winning params into `hybrid_ml_tier1_authority_v1.yaml` (or a new “v2” config with the chosen params) and compare vs `baseline_ml_prod_authority_v1` with silver reference.

---

## 4. Summary

|Question|Answer|
|--------|--------|
|Why is hybrid worse than prod?|Different models (LongT5 + FLAN-T5 vs Pegasus + LED); hybrid outputs ~2.3× longer summaries and has much lower lexical overlap (ROUGE) with the silver reference.|
|What to tune first?|Shorten REDUCE output (max_new_tokens 400–500), then try another REDUCE model (e.g. t5-base), then length_penalty/repetition.|
|How to iterate quickly?|Use **smoke** (5 episodes): use `hybrid_ml_tier1_smoke_v1` config, run on smoke, compare vs prod-on-smoke baseline. Run full benchmark only when a config looks good on smoke.|
