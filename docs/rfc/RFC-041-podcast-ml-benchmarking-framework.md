# RFC-041: Podcast ML Benchmarking Framework

## Status

🟢 Phase 0-1 Complete - Dataset materialization, baseline creation, and metrics structure implemented. CI integration pending.

## RFC Number

041

## Authors

Podcast Scraper Team

## Date

2026-01-08

## Updated

2026-01-16

## Related ADRs

- [ADR-025: Codified Comparison Baselines](../adr/ADR-025-codified-comparison-baselines.md)
- [ADR-026: Explicit Golden Dataset Versioning](../adr/ADR-026-explicit-golden-dataset-versioning.md)
- [ADR-030: Multi-Tiered Benchmarking Strategy](../adr/ADR-030-multi-tiered-benchmarking-strategy.md)
- [ADR-031: Heuristic-Based Quality Gates](../adr/ADR-031-heuristic-based-quality-gates.md)

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

## 🚀 Evolution & Improvements (2026-01-10 Update)

### Critical Enhancements for Phase 0 Implementation

Based on lessons learned from RFC-015 and RFC-016, the following improvements are **critical** before Phase 0 implementation.

---

### 1. Align Dataset Definitions with Experiment Runner

**Problem:** RFC-041 defines `indicator_v1.json` datasets. RFC-015 defines `episode globs`. Two systems of truth will cause confusion.

**Solution:** Make dataset JSON the canonical definition.

#### Dataset JSON Format

```json
{
  "dataset_id": "indicator_v1",
  "version": "1.0",
  "description": "NPR Planet Money: The Indicator episodes (explainer style)",
  "created_at": "2026-01-10T14:00:00Z",
  "content_regime": "explainer",
  "episodes": [
    {
      "episode_id": "ep001",
      "title": "Why gas prices are so high",
      "duration_minutes": 8,
      "audio_path": "data/eval/datasets/indicator_v1/audio/ep001.mp3",
      "transcript_path": "data/eval/datasets/indicator_v1/transcripts/ep001.txt",
      "golden_summary_path": "data/eval/golden/indicator_v1/ep001.txt",
      "content_hash": "abc123...",
      "preprocessing_profile": "cleaning_v3"
    }
  ]
}
```

#### Materialization Script

```bash
# scripts/eval/materialize_dataset.py

def materialize_dataset(dataset_json: Path, output_dir: Path):
    """Materialize dataset JSON into episode folders."""

    dataset = json.loads(dataset_json.read_text())

    for episode in dataset["episodes"]:
        ep_dir = output_dir / dataset["dataset_id"] / episode["episode_id"]
        ep_dir.mkdir(parents=True, exist_ok=True)

        # Copy/link files
        shutil.copy(episode["transcript_path"], ep_dir / "transcript.txt")
        if episode.get("golden_summary_path"):
            shutil.copy(episode["golden_summary_path"], ep_dir / "golden.txt")

        # Write metadata
        (ep_dir / "metadata.json").write_text(json.dumps({
            "episode_id": episode["episode_id"],
            "dataset_id": dataset["dataset_id"],
            "content_hash": episode["content_hash"],
            "preprocessing_profile": episode["preprocessing_profile"],
        }, indent=2))
```

**Why:** Single source of truth. Experiment runner reads dataset JSON, not globs. Prevents mismatched comparisons.

---

### 2. Add Summarization Quality Gates

**Problem:** Current regression rules only cover ASR/chunking (WER, latency, cost). Missing summarization-specific failures.

**Solution:** Add quality gates that match known podcast summary issues.

#### Summarization-Specific Regression Rules

```yaml
# benchmarks/regression_rules.yaml

summarization_gates:
  # Boilerplate leak (MUST be zero)
  boilerplate_leak_rate:
    baseline: 0.0
    max_delta: 0.0  # Zero tolerance
    severity: "critical"

  # Repetition score (n-gram duplication)
  repetition_score:
    baseline: 0.15  # 15% duplicate trigrams (from baseline measurement)
    max_delta: 0.05  # Allow up to 20% (15% + 5%)
    severity: "major"

  # Truncation rate (ellipsis, incomplete sentences)
  truncation_rate:
    baseline: 0.02  # 2% of sentences incomplete
    max_delta: 0.03  # Allow up to 5%
    severity: "major"

  # Numbers retained (preserve quantitative data)
  numbers_retained:
    baseline: 0.85  # 85% of numbers from reference
    min_threshold: 0.80  # Never drop below 80%
    severity: "minor"

  # Speaker label leak
  speaker_label_leak_rate:
    baseline: 0.0
    max_delta: 0.0  # Zero tolerance
    severity: "critical"

  # Summary length variance (stability)
  summary_length_variance:
    baseline_mean: 450  # chars
    baseline_std: 50    # chars
    max_std_delta: 20   # Allow std up to 70 chars
    severity: "minor"
```

#### Gate Evaluation

```python
# src/podcast_scraper/benchmarks/gates.py

def evaluate_quality_gates(metrics: Dict[str, float], rules: Dict[str, Any]) -> List[GateViolation]:
    """Evaluate quality gates and return violations."""

    violations = []

    for metric_name, rule in rules.items():
        current_value = metrics.get(metric_name)
        if current_value is None:
            continue

        baseline = rule.get("baseline")
        max_delta = rule.get("max_delta")
        min_threshold = rule.get("min_threshold")
        severity = rule["severity"]

        # Check delta from baseline
        if max_delta is not None and baseline is not None:
            delta = current_value - baseline
            if delta > max_delta:
                violations.append(GateViolation(
                    metric=metric_name,
                    severity=severity,
                    baseline=baseline,
                    current=current_value,
                    delta=delta,
                    threshold=max_delta,
                    message=f"{metric_name} increased by {delta:.3f} (max allowed: {max_delta})"
                ))

        # Check absolute threshold
        if min_threshold is not None:
            if current_value < min_threshold:
                violations.append(GateViolation(
                    metric=metric_name,
                    severity=severity,
                    current=current_value,
                    threshold=min_threshold,
                    message=f"{metric_name} is {current_value:.3f} (min required: {min_threshold})"
                ))

    return violations
```

**Why:** Catches real regressions that ROUGE misses. Zero-tolerance for critical failures (boilerplate, speaker labels).

---

### 3. Add Content Regime Datasets (Feed-Style Buckets)

**Problem:** Current plan uses Indicator (explainer) + Short Wave (science). Missing narrative journalism stress case.

**Solution:** Add The Journal as third dataset to cover narrative regime.

#### Three Content Regimes

```python
CONTENT_REGIMES = {
    "explainer": {
        "datasets": ["indicator_v1"],
        "characteristics": [
            "Short episodes (5-10 min)",
            "Single concept deep-dives",
            "Data/stats heavy",
            "Educational tone",
        ],
        "stress_tests": ["Numbers retention", "Concept clarity"],
    },
    "science": {
        "datasets": ["shortwave_v1"],
        "characteristics": [
            "Medium episodes (15-20 min)",
            "Scientific topics",
            "Interview format",
            "Technical vocabulary",
        ],
        "stress_tests": ["Technical term preservation", "Interview structure"],
    },
    "narrative": {
        "datasets": ["journal_v1"],  # NEW
        "characteristics": [
            "Long episodes (20-30 min)",
            "Story-driven journalism",
            "Multiple speakers/sources",
            "Chronological narrative",
        ],
        "stress_tests": ["Narrative flow", "Multi-speaker attribution", "Chronology preservation"],
    },
}
```

#### Dataset Sizing

```yaml
# Phase 0 (Baseline Establishment)
datasets:
  indicator_v1:
    episodes: 10  # Representative explainer set
    content_regime: "explainer"

  shortwave_v1:
    episodes: 10  # Representative science set
    content_regime: "science"

  journal_v1:  # NEW
    episodes: 5-10  # Smaller but critical stress case
    content_regime: "narrative"
```

**Why:** The Journal is your "real-world stress case" for long-form narrative. Don't skip it.

---

### 4. Baseline Integration (Shared with RFC-015/016)

**Problem:** RFC-041 regression rules reference "baseline" but don't define the artifact structure.

**Solution:** Use shared `baseline_id` concept from RFC-015.

#### Baseline Reference in Benchmarks

```yaml
# benchmarks/benchmark_config.yaml

benchmark_id: "podcast_ml_v1"
baseline_id: "bart_led_baseline_v2"  # Shared with RFC-015
baseline_path: "benchmarks/baselines/bart_led_baseline_v2/"

datasets:
  - indicator_v1
  - shortwave_v1
  - journal_v1

regression_rules:
  baseline: "bart_led_baseline_v2"  # Reference by ID
  asr_gates: {...}
  summarization_gates: {...}
```

**Why:** Single baseline artifact used by both experiment runner (RFC-015) and benchmarking (RFC-041). No duplication.

---

## Implementation Status

### ✅ Phase 0: Dataset Freezing + Baseline Artifacts (Complete)

1. ✅ Dataset JSON format implemented (`data/eval/datasets/` and `benchmarks/datasets/`)
2. ✅ Dataset creation scripts (`scripts/eval/create_dataset_json.py`)
3. ✅ Dataset materialization (`scripts/eval/materialize_dataset.py`) with hash validation
4. ✅ Source data inventory (`scripts/eval/generate_source_index.py`, `scripts/eval/generate_episode_metadata.py`)
5. ✅ Baseline creation (`scripts/eval/materialize_baseline.py`) with comprehensive fingerprinting
6. ✅ Baseline storage structure (`data/eval/baselines/`) with `predictions.jsonl`, `metrics.json`, `fingerprint.json`, `baseline.json`
7. ✅ Metrics structure (`metrics.json`) with intrinsic and vs_reference sections
8. ✅ README governance layer for all artifact types

### ✅ Phase 1: Integration with RFC-015 (Complete)

1. ✅ Experiment runner reads dataset JSONs (`scripts/eval/run_experiment.py` supports `dataset_id`)
2. ✅ Experiment runner references baseline_id and optional reference_ids
3. ✅ Quality gates evaluated automatically (intrinsic metrics: gates, length, performance, cost)
4. ✅ Regression detection via comparison deltas (`comparisons/vs_{baseline_id}.json`)
5. ✅ Reference model implemented (baseline, silver, gold references)
6. ✅ Promotion workflow (`scripts/eval/promote_run.py`, `make run-promote`)

### 🟡 Phase 2: CI Integration (Pending)

1. ⏳ Smoke tests (3 episodes from curated datasets)
2. ⏳ Nightly full benchmarks (all datasets)
3. ⏳ Regression alerts (Slack/email)

**Timeline:** Phase 0-1 complete. Phase 2 depends on RFC-015 Phase 4 (CI integration).

---

## Conclusion

This framework provides **fast feedback, objective guardrails, and long-term visibility**
into the health of the podcast ML pipeline. It is intentionally simple, automation-friendly,
and designed to evolve as the system grows.

By focusing on engineering-relevant metrics (cost, latency, quality) and avoiding academic
complexity, we can iterate quickly while maintaining confidence in our changes.

**Current Status:**

1. ✅ Phase 0-1 complete (dataset materialization, baseline creation, metrics structure, promotion workflow)
2. ✅ Integration with RFC-015 complete (experiment runner uses dataset JSONs, baseline/reference support)
3. ⏳ Phase 2 pending (CI integration - depends on RFC-015 Phase 4)

**Next Steps:**

1. Complete RFC-015 Phase 4 (CI integration)
2. Deploy smoke tests to CI
3. Deploy nightly benchmarks
4. Set up regression alerts

---

## References

- [LibriSpeech ASR Benchmark](https://www.openslr.org/12/)
- [Whisper Model Card](https://github.com/openai/whisper/blob/main/model-card.md)
- [jiwer (WER calculation)](https://github.com/jitsi/jiwer)
- [Sentence Transformers](https://www.sbert.net/)
