# RFC-041: Podcast ML Benchmarking Framework

## Status

Proposed

## RFC Number

041

## Authors

Podcast Scraper Team

## Date

2026-01-08

## Related RFCs

- RFC-025: Test Metrics and Health Tracking
- RFC-027: Pipeline Metrics Improvements
- RFC-028: ML Model Preloading and Caching

## Motivation

As the podcast processing pipeline evolves (audio preprocessing, ASR, chunking, LLM summarization),
we need a **repeatable, objective benchmarking system** to ensure improvements do not introduce
regressions in quality, latency, stability, or cost.

This RFC proposes a **lightweight, engineering-focused benchmarking framework** designed for:

- Fast iteration
- Automation in CI
- Clear regression signals
- Low operational overhead

The framework avoids academic complexity and focuses on metrics that directly impact product and cost.

---

## Goals

- Detect regressions in ASR quality, latency, and cost
- Measure the impact of audio preprocessing decisions
- Track chunking and summarization stability over time
- Enable safe provider and model swaps
- Provide a frozen, reproducible benchmark dataset

## Non-Goals

- Human-level transcription evaluation at scale
- Academic benchmarking or leaderboard comparisons
- Real-time streaming evaluation (future work)

---

## Benchmark Datasets

### Primary Dataset (Clean Baseline)

**Podcast:** The Indicator from Planet Money

**Rationale:**

- Highly consistent structure
- Short episodes (8–12 min)
- Studio-grade audio
- Minimal ads and sound effects

**Selection Criteria:**

- 20 fixed episodes
- Mix of older and recent episodes
- Single-topic episodes only

This dataset is versioned and frozen (e.g., `indicator_v1`).

### Secondary Dataset (Medium Noise)

**Podcast:** Short Wave

**Rationale:**

- Slightly more conversational
- Moderate sound design
- Still structurally consistent

Used for stress-testing beyond the clean baseline.

---

## Pipeline Stages Under Test

Each stage is benchmarked independently to isolate regressions.

```text
→ Audio Preprocessing
→ ASR
→ Text Post-processing
→ Chunking
→ LLM (Summarization / Embeddings)
```yaml

Each stage emits structured metrics and artifacts.

---

## Metrics

### 1. Audio Preprocessing Metrics

| Metric | Description |
| -------- | ------------- |
| Original duration | Raw audio length |
| Processed duration | Post-trimming duration |
| File size delta | Compression effectiveness |
| Sample rate | Output consistency |
| Channels | Mono vs stereo |

Output: `audio_metrics.json`

---

### 2. ASR Metrics

| Metric | Description |
| -------- | ------------- |
| WER | Compared to reference |
| Tokens per minute | Cost proxy |
| Real-time factor | Performance (speed) |
| ASR latency | Wall-clock time |

**Reference Strategy:**

- Use Whisper-large-v3 as a frozen baseline when human transcripts are unavailable
- **Note:** WER is relative to Whisper-large-v3, not absolute human ground truth
- **Future improvement:** Obtain 5-10 human-verified transcripts from NPR for gold standard validation

Output: `asr_metrics.json`

---

### 3. Chunking Metrics

| Metric | Description |
| -------- | ------------- |
| Avg chunk size | Tokens |
| Chunk variance | Stability |
| Overlap percentage | Redundancy |
| Sentence boundary accuracy | Readability |

Output: `chunk_metrics.json`

---

### 4. LLM Output Metrics

#### Summarization

| Metric | Description |
| -------- | ------------- |
| Summary token count | Cost |
| Summary latency | UX |
| Semantic similarity | Drift detection |

#### Stability Test

Run identical input 3× with **different random seeds** to measure:

- Temperature-induced variance (for LLMs with temperature > 0)
- Non-deterministic behavior (even at temperature=0, some APIs vary)

**Metrics:**

- Embedding cosine similarity (pairwise avg)
- Length variance (tokens)
- BLEU score between outputs

**Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (local, no API costs)

**Threshold:** Cosine similarity < 0.90 triggers instability alert

Output: `llm_metrics.json`

---

## Golden Signals

Each benchmark run produces the following **top-level signals**:

1. ASR WER
2. ASR latency (sec / min audio)
3. Tokens per episode
4. Average chunk size
5. Summary latency
6. Total estimated cost

These are tracked as time series.

---

## Cost Calculation Methodology

Cost includes:

- **ASR cost:** GPU time × compute rate (for Whisper) OR API cost (for OpenAI/Deepgram)
- **LLM cost:** (Input tokens × $X) + (Output tokens × $Y)
- **Storage cost:** Audio files + transcripts + artifacts

**Pricing Assumptions:**

- Whisper local: $0.50/hour GPU compute (A100 equivalent)
- OpenAI Whisper API: $0.006/minute audio
- GPT-4: $30/1M input tokens, $60/1M output tokens
- Claude 3.5 Sonnet: $3/1M input tokens, $15/1M output tokens
- Storage: $0.023/GB/month (S3 Standard)

**Cost Formula:**

```python
total_cost = (
    asr_cost +
    (input_tokens * input_price_per_million / 1_000_000) +
    (output_tokens * output_price_per_million / 1_000_000) +
    (storage_gb * storage_price_per_gb_month * days / 30)
)
```yaml

Output: `cost.json`

---

## Regression Rules

A benchmark run fails if:

- WER increases by >3%
- Token usage increases by >10%
- Latency increases by >15%
- Chunk size variance increases by >20%
- Cost increases without justification

Failures block merges unless explicitly overridden.

---

## Baseline Establishment

### Initial Baseline

1. Run benchmark suite on current `main` branch
2. Manual review of results for sanity
3. Tag as `benchmark-baseline-v1`
4. Store golden artifacts in `benchmarks/baselines/v1/`

### Updating Baselines

Baselines are updated only when:

1. Intentional improvements are merged (e.g., new model upgrade)
2. Manual approval from team
3. Documented in `benchmarks/CHANGELOG.md`

### Baseline Storage

- Store baseline metrics in git (JSON files)
- Store baseline artifacts (transcripts, summaries) in git-lfs or S3
- Never delete old baselines (for historical comparison)

---

## CI Integration Strategy

### When Benchmarks Run

**Option A (Comprehensive):**

- Run on all PRs to `main`
- Run nightly on `main`
- ~15-30 min duration (20 episodes × 10 min avg)

**Option B (Lightweight - Recommended):**

- Run smoke test (3 episodes) on all PRs (~5 min)
- Run full benchmark (20 episodes) nightly + on release branches
- This balances speed vs coverage

**Recommendation:** Start with Option B

### Workflow

```yaml
name: ML Benchmarks

on:
  pull_request:
    paths:

      - 'podcast_scraper/transcription/**'
      - 'podcast_scraper/summarization/**'
      - 'podcast_scraper/audio_processing/**'
      - 'benchmarks/**'
  schedule:

    - cron: '0 3 * * *'  # 3 AM UTC

jobs:
  smoke-test:
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    steps:

      - uses: actions/checkout@v4
      - name: Run smoke test (3 episodes)
        run: make benchmark-smoke

      - name: Upload results
        uses: actions/upload-artifact@v4

```text

        with:
          name: benchmark-smoke
          path: benchmarks/runs/

```
  full-benchmark:
    if: github.event_name == 'schedule'
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:

      - uses: actions/checkout@v4
      - name: Run full benchmark (20 episodes)
        run: make benchmark-full

      - name: Upload results
        uses: actions/upload-artifact@v4

```text

        with:
          name: benchmark-full
          path: benchmarks/runs/

```
      - name: Post summary to dashboard
        run: python scripts/upload_benchmark_metrics.py

```python

### Artifacts

- Upload benchmark reports as GitHub Actions artifacts
- Publish results to metrics dashboard (reuse existing dashboard from RFC-026)
- Post summary to PR comments (for quick feedback)

---

## Directory Structure

```text
benchmarks/
  datasets/
    indicator_v1.json          # 20 episode metadata + download URLs
    shortwave_v1.json          # Secondary dataset
  baselines/
    v1/
      audio_metrics.json
      asr_metrics.json
      chunk_metrics.json
      llm_metrics.json
      cost.json
  runs/
    2026-01-08-pr-123/
      audio_metrics.json
      asr_metrics.json
      chunk_metrics.json
      llm_metrics.json
      cost.json
      summary.md
    2026-01-09-nightly/
      audio_metrics.json
      asr_metrics.json
      chunk_metrics.json
      llm_metrics.json
      cost.json
      summary.md
  reports/
    latest.md                  # Human-readable report
    history.jsonl              # Time series data
  CHANGELOG.md                 # Baseline update log
```yaml

---

## Example Benchmark Report

```text
=== Benchmark Run: 2026-01-08 ===
Dataset: indicator_v1 (20 episodes)
Trigger: PR #123 (feat: add silence trimming)

Golden Signals:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Metric               Current    Baseline   Delta      Status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASR WER              3.2%       3.1%       +0.1%      ⚠️  WARN
ASR latency          0.42s/min  0.40s/min  +5%        ✅  PASS
Tokens/episode       2,450      2,400      +2.1%      ⚠️  WARN
Avg chunk size       512 tok    510 tok    +0.4%      ✅  PASS
Summary latency      1.8s       1.9s       -5.3%      ✅  PASS
Total cost/episode   $0.35      $0.33      +6%        ⚠️  WARN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Regression Check: ⚠️  WARN (3 metrics increased but within thresholds)

Details:
- ASR: Silence trimming reduced audio duration by 8%, slightly increased WER
- Cost: Higher token count due to preserving more context at boundaries
- Recommendation: Acceptable trade-off for reduced processing time

Episodes Processed: 20/20
Total Duration: 187 minutes
Total Runtime: 14.3 minutes (real-time factor: 0.08)
```yaml

---

## Implementation Timeline

### Phase 0: Infrastructure Setup (Week 1)

- Set up `benchmarks/` directory structure
- Create dataset collection script (`scripts/collect_benchmark_dataset.py`)
- Download and version The Indicator episodes (20 episodes)
- Document dataset in `benchmarks/datasets/indicator_v1.json`

**Deliverables:**

- [ ] Directory structure created
- [ ] Dataset collected and committed to git-lfs
- [ ] README.md in `benchmarks/` with usage instructions

### Phase 1: Audio Preprocessing Benchmarks (Week 2)

- Implement audio preprocessing metrics collection
- Baseline establishment for stereo vs mono, silence trimming, sample rate
- Run initial experiments and document findings

**Deliverables:**

- [ ] `scripts/benchmark_audio.py`
- [ ] `benchmarks/baselines/v1/audio_metrics.json`
- [ ] Initial findings report

### Phase 2: ASR Benchmarks (Week 3)

- Implement ASR benchmarks (WER, latency, tokens)
- Establish Whisper-large-v3 as reference
- Compare local vs API Whisper

**Deliverables:**

- [ ] `scripts/benchmark_asr.py`
- [ ] `benchmarks/baselines/v1/asr_metrics.json`
- [ ] WER baseline established

### Phase 3: Chunking + LLM Benchmarks (Week 4)

- Implement chunking metrics
- Implement LLM stability tests
- Cost calculation automation

**Deliverables:**

- [ ] `scripts/benchmark_chunking.py`
- [ ] `scripts/benchmark_llm.py`
- [ ] `benchmarks/baselines/v1/llm_metrics.json`

### Phase 4: CI Integration + Dashboard (Week 5)

- Create GitHub Actions workflow
- Integrate with metrics dashboard
- PR comment automation

**Deliverables:**

- [ ] `.github/workflows/ml-benchmarks.yml`
- [ ] Dashboard integration (reuse RFC-026 dashboard)
- [ ] PR comment bot

**Total Effort:** ~5 weeks (1 person, part-time)

---

## Dependencies

### New Python Packages

```toml
[project.optional-dependencies]
benchmarks = [
    "jiwer>=3.0.0",              # WER calculation
    "scipy>=1.11.0",             # Audio processing metrics
    "sentence-transformers>=2.2.0",  # Semantic similarity (local)
    "nltk>=3.8.0",               # Text processing for BLEU
    "pydub>=0.25.0",             # Audio file handling
]
```

### External Services (Optional)

- OpenAI API (for embeddings, if not using sentence-transformers)
- AWS S3 / GCS (for large artifact storage, ~5-10 GB)

### Infrastructure

- GitHub Actions minutes: +30 min/day (nightly runs) + 5 min/PR (smoke tests)
- Storage: 5-10 GB for dataset + artifacts (recommend git-lfs or S3)
- GPU access (optional): For faster local Whisper benchmarking

---

## Initial Experiments

The first benchmarking phase focuses on:

1. **Stereo vs mono audio**
   - Hypothesis: Mono reduces Whisper inference time by 30-40%
   - Metric: ASR latency, WER, file size

2. **Silence trimming on/off**
   - Hypothesis: Trimming reduces tokens by 5-10%
   - Metric: Tokens/episode, cost, WER

3. **Sample rate: 16 kHz vs 44.1 kHz**
   - Hypothesis: 16 kHz is sufficient for speech, reduces processing time
   - Metric: ASR latency, WER, file size

These experiments are expected to yield measurable gains in:

- ASR speed (target: 20-30% improvement)
- Token usage (target: 5-10% reduction)
- Cost predictability (target: consistent costs within ±5%)

---

## Out of Scope (Explicitly Excluded from v1)

The following are intentionally NOT benchmarked in v1:

1. **Ad detection accuracy** - No ground truth available yet
2. **Speaker name accuracy** - Requires manual annotation
3. **Metadata extraction** - Too subjective for automated testing
4. **End-to-end user workflows** - Covered by E2E tests instead (see RFC-019)
5. **Real podcast variability** - Using controlled datasets intentionally
6. **Cost optimization strategies** - Focus is on measurement, not optimization

These may be added in future versions as the system matures.

---

## Alternatives Considered

### Option A: Use Existing Benchmarking Frameworks

**Tools:** MLflow, Weights & Biases (W&B), DVC

**Pros:**

- Rich ecosystem, established patterns
- Built-in visualization and experiment tracking
- Industry-standard tools

**Cons:**

- Too heavyweight for our needs
- Requires infrastructure (MLflow server, W&B account)
- Steep learning curve for team
- Vendor lock-in risk (W&B)
- Overkill for simple regression detection

**Decision:** Rejected - build lightweight custom framework

### Option B: Manual Testing Only

**Pros:**

- No infrastructure needed
- Flexible, human judgment

**Cons:**

- Not scalable
- No regression prevention
- High risk of missing regressions
- Inconsistent methodology

**Decision:** Rejected - automation is critical

### Option C: Academic Benchmarks (LibriSpeech, CommonVoice)

**Pros:**

- Established baselines
- Large datasets
- Comparable to other systems

**Cons:**

- Not representative of podcast audio
- Long episodes (real podcasts are 30-90 min)
- Different domain (audiobooks vs podcasts)
- No chunking/summarization context

**Decision:** Rejected - use real podcast data for domain-specific insights

---

## Future Work

- Ad detection and removal metrics
- Speaker diarization benchmarking
- Real-time / streaming ASR evaluation
- Human-evaluated gold transcripts (partnership with NPR?)
- Multi-language podcast support
- Podcast-specific metrics (music detection, intro/outro detection)

---

## Conclusion

This framework provides **fast feedback, objective guardrails, and long-term visibility**
into the health of the podcast ML pipeline. It is intentionally simple, automation-friendly,
and designed to evolve as the system grows.

By focusing on engineering-relevant metrics (cost, latency, quality) and avoiding academic
complexity, we can iterate quickly while maintaining confidence in our changes.

**Next Steps:**

1. Review and approve this RFC
2. Kick off Phase 0 (infrastructure setup)
3. Establish baseline metrics within 2 weeks
4. Deploy to CI within 5 weeks

---

## References

- [LibriSpeech ASR Benchmark](https://www.openslr.org/12/)
- [Whisper Model Card](https://github.com/openai/whisper/blob/main/model-card.md)
- [jiwer (WER calculation)](https://github.com/jitsi/jiwer)
- [Sentence Transformers](https://www.sbert.net/)
