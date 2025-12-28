# RFC-027: Pipeline Metrics Improvements

- **Status**: Draft
- **Authors**:
- **Stakeholders**: Maintainers, developers, performance analysts
- **Related Issues**:
  - GitHub Issue #120 - Implementation tracking
- **Related PRDs**:
  - `docs/prd/PRD-001-transcript-pipeline.md` (core pipeline)
- **Related RFCs**:
  - `docs/rfc/RFC-025-test-metrics-and-health-tracking.md` (test metrics - different domain)
  - `docs/rfc/RFC-026-metrics-consumption-and-dashboards.md` (metrics consumption - different domain)
- **Related Documents**:
  - `src/podcast_scraper/metrics.py` - Current metrics implementation
  - `src/podcast_scraper/workflow.py` - Pipeline orchestration
  - `docs/wip/METRICS_REVIEW.md` - Original analysis (to be deleted after RFC creation)

## Abstract

This RFC defines improvements to the **pipeline metrics system** that tracks application performance during podcast scraping operations. The current system provides comprehensive metrics but has issues with duplicate output, missing metrics, inconsistent formatting, and lack of export capabilities.

**Key Principle:** Pipeline metrics should provide clear, actionable insights without cluttering logs, and should be exportable for analysis and monitoring.

## Problem Statement

**Current Issues:**

1. **Duplicate Metric Output**
   - Both `log_metrics()` and `_generate_pipeline_summary()` output metrics
   - Creates redundant information in logs
   - `log_metrics()` is too verbose for normal use (all metrics at INFO level)

2. **Missing Metrics**
   - No separate tracking of RSS fetch time (combined with scraping)
   - No model loading times (Whisper, summarization models)
   - No cache hit/miss rates for speaker detection
   - No parallel processing efficiency metrics
   - No memory usage tracking

3. **Inconsistent Formatting**
   - `log_metrics()` uses "Title Case" (e.g., "Run Duration Seconds")
   - `_generate_pipeline_summary()` uses lowercase with dashes (e.g., "transcripts_saved")
   - Makes comparison and parsing difficult

4. **No Export Capabilities**
   - Metrics only available in logs
   - No JSON/CSV export for programmatic analysis
   - No integration with monitoring systems (Prometheus, etc.)

**Impact:**

- Logs are cluttered with duplicate metric information
- Difficult to analyze performance trends over time
- Missing visibility into important performance bottlenecks (model loading, cache efficiency)
- No way to integrate metrics into monitoring dashboards

## Current Implementation

### Metrics Collected

**Per-Run Metrics:**

- `run_duration_seconds`: Total pipeline execution time
- `episodes_scraped_total`: Total episodes found in RSS feed
- `episodes_skipped_total`: Episodes skipped (e.g., due to `--skip-existing`)
- `errors_total`: Total errors encountered
- `bytes_downloaded_total`: Total bytes downloaded

**Processing Statistics:**

- `transcripts_downloaded`: Direct transcript downloads
- `transcripts_transcribed`: Whisper transcriptions performed
- `episodes_summarized`: Episodes with summaries generated
- `metadata_files_generated`: Metadata files created

**Per-Stage Timing:**

- `time_scraping`: Time spent scraping RSS feed
- `time_parsing`: Time spent parsing RSS feed
- `time_normalizing`: Time spent normalizing episode data
- `time_writing_storage`: Time spent writing files

**Per-Episode Operation Timing:**

- `download_media_times`: List of media download times per episode
- `transcribe_times`: List of transcription times per episode
- `extract_names_times`: List of speaker detection times per episode
- `summarize_times`: List of summary generation times per episode

### Collection Points

1. **Media Download** (`episode_processor.py:267`)
   - Records time when media is downloaded for transcription
   - Only records if `dl_elapsed > 0` and `pipeline_metrics` is available

2. **Transcription** (`episode_processor.py:430`)
   - Records Whisper transcription time
   - Only records if `pipeline_metrics` is available

3. **Speaker Detection** (`workflow.py:580`)
   - Records time spent extracting speaker names
   - Only records if `pipeline_metrics` is available

4. **Summary Generation** (`metadata.py:778`)
   - Records time spent generating summaries
   - Only records if `summary_elapsed > 0` and `pipeline_metrics` is available

### Current Reporting

**Two Separate Outputs:**

1. **`pipeline_metrics.log_metrics()`** (`metrics.py:158`)
   - Logs ALL metrics from `finish()` dictionary
   - Format: "Pipeline finished:" with all metrics listed
   - Called in `workflow.py:537` before summary generation
   - Uses INFO level logging

2. **`_generate_pipeline_summary()`** (`workflow.py:1759`)
   - Generates a selective summary with key metrics
   - Includes: counts, averages, errors, skipped episodes
   - Returns formatted string that is logged by `cli.py:718`
   - Uses different formatting (lowercase with dashes)

## Goals

### Primary Goals

1. **Eliminate Duplicate Output**
   - Single source of truth for metric reporting
   - Clear separation between detailed and summary metrics
   - Configurable verbosity

2. **Add Missing Metrics**
   - RSS fetch time (separate from parsing)
   - Model loading times (Whisper, summarization)
   - Cache hit rates (speaker detection)
   - Parallel processing efficiency
   - Memory usage (peak memory)

3. **Standardize Formatting**
   - Consistent key naming (snake_case)
   - Consistent units (seconds, bytes, counts)
   - Consistent number formatting (decimals, rounding)

4. **Enable Export**
   - JSON export for programmatic analysis
   - CSV export for spreadsheet analysis
   - Optional integration with monitoring systems

### Success Criteria

- ✅ No duplicate metric output in logs
- ✅ Detailed metrics available at DEBUG level
- ✅ Summary metrics at INFO level
- ✅ All recommended metrics tracked
- ✅ Consistent formatting across all outputs
- ✅ JSON/CSV export available
- ✅ Metrics can be integrated into monitoring systems

## Solution: Two-Tier Output with Export

### Approach

**Option B: Two-Tier Output (Recommended)**

- Keep `log_metrics()` for detailed metrics (DEBUG level)
- Keep `_generate_pipeline_summary()` for summary (INFO level)
- Make `log_metrics()` conditional on log level
- Add export capabilities (JSON/CSV)
- Standardize formatting across both outputs

**Benefits:**

- ✅ Provides flexibility without cluttering normal logs
- ✅ Detailed metrics available when needed (DEBUG level)
- ✅ Summary metrics always visible (INFO level)
- ✅ No breaking changes to existing behavior
- ✅ Enables export for analysis

### Implementation Plan

#### Phase 1: Consolidate Output (1-2 days)

**Tasks:**

- [ ] Modify `log_metrics()` to use DEBUG level instead of INFO
- [ ] Update `workflow.py` to conditionally call `log_metrics()` based on log level
- [ ] Standardize formatting in both `log_metrics()` and `_generate_pipeline_summary()`
- [ ] Use consistent key naming (snake_case) in both outputs
- [ ] Ensure consistent units and number formatting

**Deliverables:**

- `src/podcast_scraper/metrics.py` - Updated `log_metrics()` method
- `src/podcast_scraper/workflow.py` - Conditional logging logic
- Consistent formatting across all metric outputs

**Example Changes:**

```python
# metrics.py
def log_metrics(self) -> None:
    """Log detailed metrics at DEBUG level."""
    metrics_dict = self.finish()
    summary_lines = ["Pipeline finished (detailed metrics):"]
    for key, value in metrics_dict.items():
        # Use snake_case consistently
        summary_lines.append(f"  - {key}: {value}")
    summary = "\n".join(summary_lines)
    logger.debug(summary)  # Changed from logger.info()
```

#### Phase 2: Add Missing Metrics (2-3 days)

**Tasks:**

- [ ] Add RSS fetch time tracking (separate from scraping)
- [ ] Add model loading time tracking (Whisper, summarization)
- [ ] Add cache hit/miss tracking for speaker detection
- [ ] Add parallel processing efficiency metrics
- [ ] Add memory usage tracking (peak memory)

**Deliverables:**

- `src/podcast_scraper/metrics.py` - New metric fields and methods
- `src/podcast_scraper/workflow.py` - Metric collection points
- `src/podcast_scraper/episode_processor.py` - Model loading time tracking
- `src/podcast_scraper/speaker_detection.py` - Cache hit/miss tracking

**New Metrics:**

```python
@dataclass
class Metrics:
    # ... existing metrics ...
    
    # New metrics
    time_rss_fetch: float = 0.0  # RSS fetch time (separate from parsing)
    whisper_model_loading_time: float = 0.0  # Whisper model loading time
    summarization_model_loading_time: float = 0.0  # Summarization model loading time
    speaker_detection_cache_hits: int = 0  # Cache hits for speaker detection
    speaker_detection_cache_misses: int = 0  # Cache misses for speaker detection
    parallel_processing_efficiency: float = 0.0  # Efficiency of parallel processing
    peak_memory_mb: float = 0.0  # Peak memory usage in MB
```

#### Phase 3: Add Export Capabilities (2-3 days)

**Tasks:**

- [ ] Add `export_json()` method to `Metrics` class
- [ ] Add `export_csv()` method to `Metrics` class
- [ ] Add CLI flag `--export-metrics` with format option (json/csv)
- [ ] Add optional Prometheus integration (future work)

**Deliverables:**

- `src/podcast_scraper/metrics.py` - Export methods
- `src/podcast_scraper/cli.py` - CLI flag and export logic
- Export functionality for JSON and CSV formats

**Example Export:**

```python
# metrics.py
def export_json(self, filepath: str) -> None:
    """Export metrics to JSON file."""
    metrics_dict = self.finish()
    with open(filepath, 'w') as f:
        json.dump(metrics_dict, f, indent=2)

def export_csv(self, filepath: str) -> None:
    """Export metrics to CSV file."""
    metrics_dict = self.finish()
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
        writer.writeheader()
        writer.writerow(metrics_dict)
```

#### Phase 4: Testing and Documentation (1-2 days)

**Tasks:**

- [ ] Test metrics collection in dry-run mode
- [ ] Test metrics accuracy when operations are skipped
- [ ] Test metrics collection with parallel processing
- [ ] Validate metric calculations (averages, totals)
- [ ] Update documentation with new metrics and export options

**Deliverables:**

- Test coverage for new metrics
- Documentation updates
- Examples of exported metrics

## Design Decisions

### 1. Two-Tier Output vs Single Output

**Decision:** Use two-tier output (DEBUG for detailed, INFO for summary)

**Rationale:**

- Provides flexibility without cluttering normal logs
- Detailed metrics available when needed (DEBUG level)
- Summary metrics always visible (INFO level)
- No breaking changes to existing behavior

**Alternative Considered:** Single comprehensive output (Option A)

**Why Not:** Would require removing one of the existing outputs, potentially breaking existing workflows or scripts that parse logs.

### 2. Export Format

**Decision:** Support both JSON and CSV export

**Rationale:**

- JSON: Machine-readable, easy to parse programmatically
- CSV: Easy to import into spreadsheets for analysis
- Both formats are standard and widely supported

**Future:** Prometheus integration can be added later if needed

### 3. Metric Collection Timing

**Decision:** Keep conditional collection (only record if `pipeline_metrics` is available and `elapsed > 0`)

**Rationale:**

- Prevents false metrics (zero times when operations didn't occur)
- Correct behavior - only record actual operations
- Should be documented but not changed

### 4. Formatting Standardization

**Decision:** Use snake_case consistently across all outputs

**Rationale:**

- Matches Python naming conventions
- Consistent with existing codebase patterns
- Easier to parse programmatically

## Benefits

### Developer Experience

- ✅ Cleaner logs (no duplicate metrics)
- ✅ Detailed metrics available when needed (DEBUG level)
- ✅ Summary metrics always visible (INFO level)
- ✅ Consistent formatting makes logs easier to read

### Analysis Capabilities

- ✅ Export to JSON/CSV for programmatic analysis
- ✅ Additional metrics provide better visibility
- ✅ Can track performance trends over time
- ✅ Can identify bottlenecks (model loading, cache efficiency)

### Monitoring Integration

- ✅ JSON export enables integration with monitoring systems
- ✅ Future Prometheus integration possible
- ✅ Metrics can be tracked in dashboards

## Related Files

- `src/podcast_scraper/metrics.py` - Metrics collection and reporting
- `src/podcast_scraper/workflow.py` - Pipeline orchestration and metric logging
- `src/podcast_scraper/episode_processor.py` - Episode processing and metric collection
- `src/podcast_scraper/metadata.py` - Metadata generation and metric collection
- `src/podcast_scraper/cli.py` - CLI interface and export options

## Notes

- This RFC focuses on **pipeline/application metrics**, not test metrics (see RFC-025 for test metrics)
- Conditional collection behavior is correct and should be documented, not changed
- Export capabilities enable future integration with monitoring systems
- Two-tier output provides flexibility without breaking existing workflows
