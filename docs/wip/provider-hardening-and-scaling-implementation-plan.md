# Provider Hardening & Stabilization - Implementation Plan

**Date:** 2026-02-06
**Status:** Planning Phase
**Objective:** Harden, stabilize, and improve observability of the existing codebase before architectural changes.

**Theme:** Make what we have more robust, observable, and maintainable.

---

## Scope Decision: Stabilization First, Architecture Later

### ✅ IN SCOPE (This Work - v2.6-stabilization)

| Issue | Type | Description |
|-------|------|-------------|
| #399 | Stabilization | Provider hardening (retry, timeouts, capabilities, schema) |
| #401 | Stabilization | DevEx/Ops (degradation policy, secrets logging) |
| #402 | Feature | Transcript caching + JSONL metrics |
| #383 | Performance | Bounded queue for transcription |
| #259 | Safety | Protocol runtime verification |
| #331 | Refactoring | Cleaning externalization |
| #393 | Documentation | Speaker detection observation |
| #392 | Documentation | Preprocessing performance observation |

---

## RFC Cross-Reference Analysis

The following RFCs are relevant to this work and should inform our implementation:

### RFC-029: Provider Refactoring Consolidation (✅ Completed)

**Relevance:** High - Provider architecture foundation

| Our Issue | RFC-029 Section | Alignment |
|-----------|-----------------|-----------|
| #259 Protocol Verification | "Phase 2: Protocol Compliance Testing" | ⚠️ RFC-029 Phase 2 is pending - our #259 completes it |
| #399 Provider Capabilities | Factory Pattern, Protocol-Based Design | ✅ Aligns - extend existing pattern |

**Action:** Our #259 work completes RFC-029 Phase 2.

---

### RFC-027: Pipeline Metrics Improvements (Draft)

**Relevance:** High - Metrics infrastructure

| Our Issue | RFC-027 Feature | Alignment |
|-----------|-----------------|-----------|
| #402.2 JSONL Metrics | JSON/CSV export, two-tier output | ✅ Our JSONL complements RFC-027 |
| #401 Structured Logs | Consistent formatting, snake_case | ✅ Follow RFC-027 conventions |

**Missing metrics mentioned in RFC-027 we should consider:**
- RSS fetch time (separate from scraping)
- Model loading times
- Cache hit/miss rates ← relevant for #402.1 transcript caching

**Action:** Follow RFC-027 formatting conventions; our JSONL metrics provide data for RFC-027's export goals.

---

### RFC-040: Audio Pre-Processing Pipeline (Draft)

**Relevance:** Medium - Related but different from #331

| Concept | RFC-040 | Our #331 |
|---------|---------|----------|
| **When** | BEFORE transcription | AFTER transcription, BEFORE summarization |
| **What** | Audio processing (silence removal, VAD) | Text cleaning (sponsor blocks, speaker labels) |
| **Cache** | Content-hash based | Can follow same pattern |

**Action:** #331 should NOT conflict with RFC-040. Keep transcript cleaning separate from audio preprocessing.

---

### RFC-044: Model Registry for Architecture Limits (Draft)

**Relevance:** Medium - Pattern for capability declaration

| Our Issue | RFC-044 Concept | How to Apply |
|-----------|-----------------|--------------|
| #399 Capability Contract | `ModelCapabilities` dataclass | Follow similar pattern for `ProviderCapabilities` |

**RFC-044 `ModelCapabilities` structure:**
```python
@dataclass(frozen=True)
class ModelCapabilities:
    max_position_embeddings: int
    model_type: str
    supports_long_context: bool
```

**Action:** Use similar frozen dataclass pattern for provider capabilities in #399.

---

### RFC-046: Materialization Architecture (Draft)

**Relevance:** Medium - Informs #331 cleaning architecture

| Concept | RFC-046 | Our #331 |
|---------|---------|----------|
| Two-layer preprocessing | Canonical (shared) + Adapter (provider-specific) | Pattern-based (shared) + Provider-specific |

**RFC-046 insight:** Preprocessing has 80% impact on output quality. Treat it as explicit, not hidden.

**Action:** #331 should follow the two-layer pattern:
1. **Base cleaner:** Standard pattern-based cleaning (shared)
2. **Provider adapters:** ML-specific, LLM-specific variations

---

### RFC-028: ML Model Preloading and Caching (✅ Completed)

**Relevance:** Low - Different type of caching

| RFC-028 | Our #402.1 |
|---------|------------|
| ML model caching (`~/.cache/whisper/`) | Transcript caching (content hash) |

**Action:** Use similar content-hash approach for transcript caching. Different cache location.

---

### RFC-043: Automated Metrics Alerts (Draft)

**Relevance:** Low - Future consumer of our metrics

Our JSONL metrics (#402.2) will provide the data source for RFC-043's PR comments and webhook alerts.

**Action:** Ensure JSONL format is compatible with RFC-043's expectations.

---

## Summary: RFC Alignment Checklist

- [ ] #259: Completes RFC-029 Phase 2 (Protocol Compliance Testing)
- [ ] #399: Follow RFC-044 pattern for capability dataclass
- [ ] #402.2: Follow RFC-027 formatting conventions
- [ ] #331: Follow RFC-046 two-layer pattern (base + adapters)
- [ ] #331: Don't conflict with RFC-040 (audio vs text)

---

## GitHub Issues Covered

---

## ⚠️ CRITICAL: Codebase Review Findings

**Before implementing, the following EXISTING code must be considered to avoid rework:**

### 1. Retry Logic ALREADY EXISTS and is WIDELY USED

**Location:** `src/podcast_scraper/utils/provider_metrics.py`

The codebase already has `retry_with_metrics()` function that:

- ✅ Implements exponential backoff
- ✅ Tracks retries in metrics
- ✅ Logs compact format: `provider_retry: provider=openai attempt=2 sleep=4.0 reason=429`
- ✅ Distinguishes 429/rate limits from other errors
- ✅ Used by ALL providers (OpenAI, Gemini, Anthropic, DeepSeek, Grok, Mistral, Ollama)

**GAP:** Missing jitter (important for avoiding thundering herd)

**Action:** ENHANCE existing `retry_with_metrics()`, don't create new `providers/retry.py`

### 2. Timeouts ALREADY EXIST in Config

**Location:** `src/podcast_scraper/config.py`

Existing timeout configs:

- `timeout: int = 60` - General HTTP timeout
- `transcription_timeout: int = 1800` (30 min)
- `summarization_timeout: int = 600` (10 min)
- `ollama_timeout: int = 120` (for local Ollama)

**GAP:** No separate connect_timeout vs read_timeout (HTTP client level)

**Action:** Add connect/read timeout separation to HTTP clients, not new config fields

### 3. `max_context_tokens` ALREADY EXISTS on Providers

**Location:** Each provider class (OpenAIProvider, GeminiProvider, etc.)

All providers already have:

```python
self.max_context_tokens = 128000  # varies by provider
```

**GAP:** Not formalized as a capability contract interface

**Action:** Formalize existing patterns into `ProviderCapabilities` contract

### 4. Transcript Cleaning ALREADY EXISTS

**Location:** `src/podcast_scraper/preprocessing/core.py`

Existing functions:

- `clean_for_summarization()` - Complete pipeline
- `clean_transcript()` - Basic cleaning
- `remove_sponsor_blocks()` - Sponsor removal
- `remove_outro_blocks()` - Outro removal
- Plus many more specialized cleaners

**GAP:** Not provider-specific (all providers use same cleaning)

**Action:** Externalize into provider-specific processors, refactor existing code

### 5. Metrics System ALREADY EXTENSIVE

**Location:** `src/podcast_scraper/workflow/metrics.py`, `src/podcast_scraper/utils/provider_metrics.py`

Existing capabilities:

- `Metrics` class with 50+ metrics fields
- `EpisodeMetrics` with per-episode tracking
- `EpisodeStatus` with status tracking
- `ProviderCallMetrics` with tokens, costs, retries
- JSON output via `save_to_file()`

**GAP:** No JSONL streaming output (current is single JSON file)

**Action:** Add JSONL emitter that wraps/uses existing metrics, don't duplicate

### 6. TranscriptionResources Uses List (Not Queue)

**Location:** `src/podcast_scraper/workflow/types.py`

```python
class TranscriptionResources(NamedTuple):
    transcription_jobs: List[models.TranscriptionJob]
    transcription_jobs_lock: Optional[Any]  # threading.Lock
```

**Status:** Confirmed issue - needs bounded queue implementation

---

## Risk Mitigations

### Risk 1: NamedTuple Immutability (Bounded Queue #383)

**Problem:** `TranscriptionResources` is a `NamedTuple` - can't change field types.

**Mitigation Plan:**

```bash
# Step 1: Find all usages BEFORE changing
grep -rn "TranscriptionResources" --include="*.py" src/ tests/
```

**Implementation:**

1. Change `TranscriptionResources` from `NamedTuple` to `@dataclass`
2. Keep same field names for compatibility
3. Update type from `List[TranscriptionJob]` to `Queue[TranscriptionJob]`
4. Run full test suite to catch any breakage

**Code Change:**

```python
# Before (NamedTuple - immutable)
class TranscriptionResources(NamedTuple):
    transcription_jobs: List[models.TranscriptionJob]
    ...

# After (dataclass - mutable, can use Queue)
@dataclass
class TranscriptionResources:
    transcription_provider: Any
    temp_dir: Optional[str]
    transcription_jobs: Queue[models.TranscriptionJob]  # Changed!
    transcription_jobs_lock: Optional[Any]  # May become redundant
    saved_counter_lock: Optional[Any]
```

### Risk 2: Thread Safety (Bounded Queue #383)

**Problem:** Adding thread-safe `Queue` while existing code has manual `Lock`.

**Mitigation Plan:**

```bash
# Step 1: Map all lock usages
grep -rn "transcription_jobs_lock" --include="*.py" src/
```

**Implementation (Two-Phase):**

**Phase A (This PR):**
- Add `Queue` but KEEP the lock
- Verify all tests pass
- Lock becomes redundant but harmless

**Phase B (Follow-up PR - optional):**
- Remove redundant lock
- Simplify code

**Why two phases?** Reduces risk. If something breaks, we know it's the Queue, not the lock removal.

### Risk 3: Test Coverage Before Modifications

**Mitigation:** Run coverage BEFORE any changes.

```bash
# Retry/timeout coverage
make test-unit -- --cov=src/podcast_scraper/utils/provider_metrics \
                  --cov=src/podcast_scraper/utils/retry \
                  --cov-report=term-missing

# Transcription queue coverage
make test-unit -- --cov=src/podcast_scraper/workflow/stages/transcription \
                  --cov=src/podcast_scraper/workflow/types \
                  --cov-report=term-missing

# Preprocessing/cleaning coverage
make test-unit -- --cov=src/podcast_scraper/preprocessing \
                  --cov-report=term-missing
```

**Document baseline coverage before starting. Don't reduce coverage.**

### Risk 4: Large Scope Creep

**Mitigation:** Strict phase gates.

- Complete Phase 1 fully before Phase 2
- Run `make ci-fast` after EACH step
- If any phase takes 2x estimated time, STOP and reassess

---

## Revised Plan Summary

Based on codebase review, here's what each issue ACTUALLY requires:

| Issue | Original Scope | Revised Scope |
| ----- | -------------- | ------------- |
| #399.1 | Create new retry wrapper | ENHANCE `retry_with_metrics()` with jitter |
| #399.2 | Add timeouts everywhere | Add connect/read timeout to HTTP clients |
| #399.3 | Create capability contract | FORMALIZE existing `max_context_tokens` pattern |
| #399.4 | Normalized output schema | NEW - create summary schema validation |
| #402.1 | Transcript caching | NEW - cache transcripts by audio hash |
| #402.2 | JSONL metrics | EXTEND existing `Metrics` with JSONL output |
| #383 | Bounded queue | NEW - replace List with Queue |
| #331 | Externalize cleaning | REFACTOR existing `preprocessing/core.py` |
| #259 | Protocol verification | NEW - add `@runtime_checkable` |
| #401 | Structured logs, etc. | EXTEND existing logging + NEW degradation |

---

## Dependency Analysis

### Critical Path Dependencies

```
#259 (Protocol Verification)
  ↓
#399 (Provider Hardening)
  ↓
#401 (DevEx/Ops) ─────────────┐
  ↓                            │
#331 (Cleaning) ───────────────┤
  ↓                            │
#402 (Caching + Metrics) ──────┘
  ↓
#383 (Bounded Queue)

Note: #383 can run in parallel with Phase 2 (independent of other work)
```

### Issue Groupings

**Foundation Layer (Must Complete First):**
- #259: Protocol runtime verification (enables safe refactoring)
- #399: Provider hardening (unified retry, timeouts, capability contract, output schema)

**Infrastructure Layer (Builds on Foundation):**
- #383: Bounded queue (performance, independent)
- #402: Transcript caching + JSONL metrics (high-leverage, enables experimentation)

**DevEx & Ops Layer (Builds on Foundation):**
- #401: Secrets management, graceful degradation

**Refactoring Layer (Can overlap with Infrastructure):**
- #331: Transcript cleaning externalization

**Documentation Layer (Can be done anytime):**
- #393: Speaker detection observation (documentation update)
- #392: Preprocessing performance observation (documentation update)


---

## Phase 1: Foundation - Protocol Verification & Provider Hardening

**Timeline:** ~2-3 weeks
**Risk Level:** Low (foundational, well-defined scope)
**Dependencies:** None

### Step 1.1: Protocol Runtime Verification (#259)

**Objective:** Add runtime protocol verification to catch implementation errors early.

**Tasks:**

1. **Assessment & Decision**
   - [ ] Review git history for protocol-related bugs
   - [ ] Assess current test coverage for protocol compliance
   - [ ] Choose solution (recommend Option B: `@runtime_checkable` with dev-mode checks)
   - [ ] Document decision in issue #259

2. **Implementation**
   - [ ] Add `@runtime_checkable` decorator to protocol definitions:
     - `src/podcast_scraper/transcription/base.py` - `TranscriptionProvider`
     - `src/podcast_scraper/speaker_detectors/base.py` - `SpeakerDetector`
     - `src/podcast_scraper/summarization/base.py` - `SummarizationProvider`
   - [ ] Add optional runtime verification to factories:
     - `src/podcast_scraper/transcription/factory.py`
     - `src/podcast_scraper/speaker_detectors/factory.py`
     - `src/podcast_scraper/summarization/factory.py`
   - [ ] Use `__debug__` mode for zero production overhead
   - [ ] Add verification helper function: `_verify_protocol_compliance(provider, protocol)`

3. **Testing**
   - [ ] Unit tests for verification logic (valid providers pass, invalid fail)
   - [ ] Integration tests verify all existing providers pass
   - [ ] Test that `__debug__=False` disables verification (zero overhead)

4. **Documentation**
   - [ ] Update provider implementation guide with verification behavior
   - [ ] Document in `docs/guides/PROVIDER_IMPLEMENTATION_GUIDE.md`

**Files to Modify:**
- `src/podcast_scraper/transcription/base.py`
- `src/podcast_scraper/speaker_detectors/base.py`
- `src/podcast_scraper/summarization/base.py`
- `src/podcast_scraper/transcription/factory.py`
- `src/podcast_scraper/speaker_detectors/factory.py`
- `src/podcast_scraper/summarization/factory.py`
- `tests/unit/test_protocol_verification.py` (new)
- `docs/guides/PROVIDER_IMPLEMENTATION_GUIDE.md`

**Acceptance Criteria:**
- [ ] All protocols marked with `@runtime_checkable`
- [ ] Factories verify protocol compliance in `__debug__` mode
- [ ] All existing providers pass verification
- [ ] Tests verify invalid providers are caught
- [ ] Zero production overhead when `__debug__=False`

---

### Step 1.2: Provider Hardening - ENHANCE Retry & Backoff (#399.1)

**Objective:** ENHANCE existing `retry_with_metrics()` with jitter and improved error classification.

**⚠️ IMPORTANT:** Retry already exists! Don't create new - enhance existing.

**Existing Code:**

```python
# src/podcast_scraper/utils/provider_metrics.py
def retry_with_metrics(
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    metrics: Optional[ProviderCallMetrics] = None,
) -> T:
```

Already logs: `provider_retry: provider=openai attempt=2 sleep=4.0 reason=429`

**Tasks:**

1. **Add Jitter to Existing Retry**
   - [ ] Add jitter parameter to `retry_with_metrics()`:
     - `jitter: bool = True` - enable by default
     - Jitter formula: `sleep_time = base_delay * (1 + random.uniform(-0.1, 0.1))`
   - [ ] Prevents thundering herd on retries

2. **Improve Error Classification**
   - [ ] Create `src/podcast_scraper/utils/retryable_errors.py`:
     - `is_retryable_error(error: Exception) -> bool`
     - Classify: 429, 5xx, connection errors, timeout errors
     - Classify: Non-retryable: 4xx (except 429), auth errors
   - [ ] Update `retry_with_metrics()` to use improved classification

3. **Add Retry-After Header Support**
   - [ ] Check for `Retry-After` header in rate limit responses
   - [ ] Use provider-specified delay when available
   - [ ] Already partially implemented - verify and enhance

4. **Configuration (OPTIONAL - existing hardcoded values work)**
   - [ ] Consider making retry params configurable via Config
   - [ ] Current hardcoded: max_retries=3, initial_delay=1.0, max_delay=30.0
   - [ ] May not need config if current values are acceptable

5. **Testing**
   - [ ] Unit tests for jitter behavior
   - [ ] Unit tests for error classification
   - [ ] Verify existing provider tests still pass

**Files to Modify:**

- `src/podcast_scraper/utils/provider_metrics.py` (ENHANCE)
- `src/podcast_scraper/utils/retryable_errors.py` (NEW - optional helper)
- `tests/unit/test_provider_metrics.py` (ENHANCE)

**Acceptance Criteria:**

- [ ] Jitter added to retry delays
- [ ] Error classification improved
- [ ] All existing provider tests pass
- [ ] Logging format unchanged (already correct)

---

### Step 1.3: Provider Hardening - ENHANCE Timeouts (#399.2)

**Objective:** Add connect/read timeout separation to HTTP clients.

**⚠️ IMPORTANT:** Operation-level timeouts already exist! Focus on HTTP client level.

**Existing Timeouts in Config:**

```python
# src/podcast_scraper/config.py
timeout: int = 60                      # General HTTP timeout
transcription_timeout: int = 1800      # 30 min for transcription
summarization_timeout: int = 600       # 10 min for summarization
ollama_timeout: int = 120              # Local Ollama timeout
```

**GAP:** HTTP clients (httpx, requests) don't have separate connect vs read timeouts.

**Tasks:**

1. **Audit Current HTTP Client Usage**
   - [ ] Review OpenAI SDK timeout configuration (uses httpx internally)
   - [ ] Review Gemini SDK timeout configuration
   - [ ] Review Anthropic SDK timeout configuration
   - [ ] Review other provider SDKs

2. **Add Connect/Read Timeout to HTTP Clients**
   - [ ] OpenAI: Configure `httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)`
   - [ ] For SDKs that support it, configure connect/read separately
   - [ ] For SDKs that don't, document limitation

3. **Create Timeout Helper (if needed)**
   - [ ] Create `src/podcast_scraper/utils/timeout_config.py`:
     - `get_http_timeout(cfg: Config) -> httpx.Timeout`
     - Default: connect=10s, read=60s, write=10s
   - [ ] Only create if multiple providers need same logic

4. **Add Config Options (if needed)**
   - [ ] Consider: `http_connect_timeout: float = 10.0`
   - [ ] Consider: `http_read_timeout: float = 60.0`
   - [ ] May not need if defaults are reasonable

5. **Testing**
   - [ ] Verify timeout behavior with slow mock servers
   - [ ] Test that connect timeout < read timeout works correctly

**Files to Modify:**

- `src/podcast_scraper/providers/openai/openai_provider.py` (ENHANCE client init)
- `src/podcast_scraper/providers/gemini/gemini_provider.py` (ENHANCE if applicable)
- `src/podcast_scraper/providers/anthropic/anthropic_provider.py` (ENHANCE if applicable)
- `src/podcast_scraper/utils/timeout_config.py` (NEW - optional helper)
- `tests/integration/test_provider_timeouts.py` (NEW)

**Acceptance Criteria:**

- [ ] Connect timeout is shorter than read timeout
- [ ] Long-running operations (transcription) respect their dedicated timeouts
- [ ] Stuck connections fail fast (connect timeout)
- [ ] Slow responses have reasonable time to complete (read timeout)

---

### Step 1.4: Provider Hardening - Capability Contract (#399.3)

**Objective:** Replace provider name checks with capability contract interface.

**Tasks:**

1. **Design Capability Contract**
   - [ ] Define `ProviderCapabilities` dataclass/protocol:
     - `supports_audio: bool`
     - `supports_json_mode: bool`
     - `max_context_tokens: int`
     - `supports_tool_calls: bool`
     - `supports_system_prompt: bool`
     - `supports_streaming: bool`
   - [ ] Review current `if provider == X` conditionals in codebase

2. **Implementation**
   - [ ] Create `src/podcast_scraper/providers/capabilities.py`:
     - `ProviderCapabilities` dataclass
     - `get_provider_capabilities(provider)` function
   - [ ] Add `get_capabilities()` method to provider protocols (optional, can use introspection)
   - [ ] Implement capability detection for all providers:
     - MLProvider capabilities
     - OpenAIProvider capabilities
     - GeminiProvider capabilities
     - AnthropicProvider capabilities
     - Other providers as needed
   - [ ] Replace provider name checks with capability checks:
     - `src/podcast_scraper/workflow/stages/summarization.py`
     - `src/podcast_scraper/workflow/stages/transcription.py`
     - Other files using provider name checks

3. **Pipeline Integration**
   - [ ] Use JSON outputs when `supports_json_mode=True`
   - [ ] Downgrade gracefully when capabilities not available
   - [ ] Warn once at startup about missing capabilities

4. **Testing**
   - [ ] Unit tests for capability detection
   - [ ] Integration tests for capability-based behavior
   - [ ] Verify graceful degradation when capabilities missing

5. **Documentation**
   - [ ] Document capability contract in `docs/ARCHITECTURE.md`
   - [ ] Update provider guides with capability examples

**Files to Modify:**
- `src/podcast_scraper/providers/capabilities.py` (new)
- `src/podcast_scraper/providers/ml/ml_provider.py`
- `src/podcast_scraper/providers/openai/openai_provider.py`
- `src/podcast_scraper/providers/gemini/gemini_provider.py`
- `src/podcast_scraper/workflow/stages/summarization.py`
- `src/podcast_scraper/workflow/stages/transcription.py`
- `tests/unit/test_capabilities.py` (new)
- `docs/ARCHITECTURE.md`

**Acceptance Criteria:**
- [ ] Provider capability contract interface defined and implemented
- [ ] All providers expose their capabilities via the contract
- [ ] Pipeline uses capability contract instead of provider name checks
- [ ] Graceful degradation when capabilities missing
- [ ] Startup warnings for missing capabilities
- [ ] All tests pass

---

### Step 1.5: Provider Hardening - Normalized Output Schema (#399.4)

**Objective:** Define stable schema for summaries with strict parsing and validation.

**Tasks:**

1. **Design Output Schema**
   - [ ] Define normalized summary schema:
     - `title: str`
     - `bullets: List[str]`
     - `key_quotes: List[str]` (optional)
     - `named_entities: List[str]` (optional)
     - `timestamps: List[Dict]` (optional)
   - [ ] Design parsing strategy:
     - Best effort parse for non-JSON providers
     - Validation and repair attempt
     - Degraded status tracking

2. **Implementation**
   - [ ] Create `src/podcast_scraper/models/summary_schema.py`:
     - `SummarySchema` Pydantic model
     - `parse_summary_output(text: str, provider)` function
     - `validate_summary_schema(data: dict)` function
   - [ ] Implement parsing logic:
     - JSON mode parsing (strict)
     - Best effort parsing (regex, heuristics)
     - Repair attempt for malformed JSON
   - [ ] Add degraded status tracking:
     - `status: Literal["valid", "degraded", "invalid"]`
     - Save raw text when parsing fails
   - [ ] Integrate into summarization pipeline:
     - `src/podcast_scraper/workflow/stages/summarization.py`

3. **Testing**
   - [ ] Unit tests for schema parsing (valid JSON, best effort, repair)
   - [ ] Unit tests for validation logic
   - [ ] Integration tests with different providers
   - [ ] Test degraded status handling

4. **Documentation**
   - [ ] Document schema in `docs/api/SUMMARY_SCHEMA.md` (new)
   - [ ] Update summarization guide with schema examples

**Files to Modify:**
- `src/podcast_scraper/models/summary_schema.py` (new)
- `src/podcast_scraper/workflow/stages/summarization.py`
- `tests/unit/test_summary_schema.py` (new)
- `tests/integration/test_summary_parsing.py` (new)
- `docs/api/SUMMARY_SCHEMA.md` (new)

**Acceptance Criteria:**
- [ ] Normalized output schema defined for summaries
- [ ] Strict parsing with validation and repair attempts implemented
- [ ] Degraded status tracking for failed parsing
- [ ] Raw text saved when parsing fails
- [ ] All tests pass

---

## Phase 2: Infrastructure - Bounded Queue & Transcript Caching

**Timeline:** ~2-3 weeks
**Risk Level:** Medium (performance-critical, threading changes)
**Dependencies:** Phase 1 (can start in parallel with Phase 1.4-1.5)

### Step 2.1: Bounded Queue for Download → Transcription (#383)

**Objective:** Replace simple list with bounded queue to prevent unbounded memory growth.

**Tasks:**

1. **Update Type Definitions**
   - [ ] Modify `TranscriptionResources` in `src/podcast_scraper/workflow/types.py`:
     - Change `transcription_jobs: List[models.TranscriptionJob]` to `transcription_jobs: queue.Queue[models.TranscriptionJob]`
     - Review if `transcription_jobs_lock` still needed (Queue is thread-safe)

2. **Update Resource Setup**
   - [ ] Modify `setup_transcription_resources()` in `src/podcast_scraper/workflow/stages/transcription.py`:
     - Create `queue.Queue(maxsize=cfg.transcription_queue_size)` instead of empty list
   - [ ] Add config option `transcription_queue_size` to `Config` (default: 50)

3. **Update Download Stage**
   - [ ] Modify `process_episode_download()` in `src/podcast_scraper/workflow/episode_processor.py`:
     - Replace `transcription_jobs.append(job)` with `transcription_jobs.put(job, block=True, timeout=None)`
     - This provides backpressure when queue is full

4. **Update Transcription Stage**
   - [ ] Modify `process_transcription_jobs_concurrent()` in `src/podcast_scraper/workflow/stages/transcription.py`:
     - Replace polling logic (`_find_next_unprocessed_transcription_job()`) with `queue.get(block=True, timeout=0.1)`
     - Handle `queue.Empty` exception for timeout-based polling
     - Remove `processed_job_indices` tracking (queue handles this automatically)

5. **Configuration**
   - [ ] Add `transcription_queue_size` to `Config` model
   - [ ] Document in `docs/api/CONFIGURATION.md`

6. **Testing**
   - [ ] Unit tests for queue behavior (full queue, empty queue, timeout)
   - [ ] Integration test with many episodes to verify backpressure works
   - [ ] Verify existing E2E tests still pass

**Files to Modify:**
- `src/podcast_scraper/workflow/types.py`
- `src/podcast_scraper/workflow/stages/transcription.py`
- `src/podcast_scraper/workflow/episode_processor.py`
- `src/podcast_scraper/config.py`
- `tests/unit/test_transcription_queue.py` (new)
- `tests/integration/test_bounded_queue.py` (new)
- `docs/api/CONFIGURATION.md`

**Acceptance Criteria:**
- [ ] Bounded queue implemented with configurable size
- [ ] Downloads block when queue is full (backpressure)
- [ ] Transcription uses `queue.get()` instead of polling
- [ ] All existing tests pass
- [ ] Integration test verifies backpressure works

---

### Step 2.2: Transcript Caching by Audio Hash (#402.1)

**Objective:** Cache transcripts by audio hash to enable fast multi-provider experimentation.

**Tasks:**

1. **Design Caching Strategy**
   - [ ] Review existing caching patterns: `src/podcast_scraper/preprocessing/audio/cache.py`
   - [ ] Design transcript cache structure:
     - Cache key: SHA256 hash of audio file (or first 1MB for speed)
     - Cache location: `.cache/transcripts/` or configurable
     - Cache format: Store transcript text + metadata (provider, model, timestamp)

2. **Implementation**
   - [ ] Create `src/podcast_scraper/cache/transcript_cache.py`:
     - `get_cached_transcript(audio_hash: str, cache_dir: str) -> Optional[str]`
     - `save_transcript_to_cache(audio_hash: str, transcript: str, cache_dir: str) -> str`
     - `get_audio_hash(audio_path: str) -> str` (hash first 1MB for speed)
   - [ ] Integrate into transcription pipeline:
     - Check cache before transcription
     - Save to cache after transcription
     - Skip transcription entirely if cache hit
   - [ ] Update `transcribe_media_to_text()` in `src/podcast_scraper/workflow/episode_processor.py`

3. **Configuration**
   - [ ] Add `transcript_cache_dir` to `Config` (default: `.cache/transcripts/`)
   - [ ] Add `transcript_cache_enabled: bool = True` to `Config`
   - [ ] Document in `docs/api/CONFIGURATION.md`

4. **Testing**
   - [ ] Unit tests for cache operations (get, save, hash generation)
   - [ ] Integration tests verify cache hits skip transcription
   - [ ] Test cache invalidation (different audio = different hash)

5. **Documentation**
   - [ ] Document transcript caching in `docs/ARCHITECTURE.md`
   - [ ] Add caching guide for experimentation workflows

**Files to Modify:**
- `src/podcast_scraper/cache/transcript_cache.py` (new)
- `src/podcast_scraper/workflow/episode_processor.py`
- `src/podcast_scraper/config.py`
- `tests/unit/test_transcript_cache.py` (new)
- `tests/integration/test_transcript_caching.py` (new)
- `docs/api/CONFIGURATION.md`
- `docs/ARCHITECTURE.md`

**Acceptance Criteria:**
- [ ] Transcripts cached by audio hash
   - [ ] Cache hits skip transcription entirely
   - [ ] Cache key generation is fast (1MB hash)
   - [ ] Cache directory configurable
   - [ ] All tests pass

---

### Step 2.3: Standardized JSONL Metrics (#402.2, #401.8)

**Objective:** Add JSONL streaming output to existing metrics system.

**⚠️ IMPORTANT:** Extensive metrics already exist! Build on them, don't duplicate.

**Existing Metrics System:**

```python
# src/podcast_scraper/workflow/metrics.py
@dataclass
class Metrics:
    run_duration_seconds: float = 0.0
    episodes_scraped_total: int = 0
    # ... 50+ more fields ...
    episode_statuses: List[EpisodeStatus]
    episode_metrics: List[EpisodeMetrics]

    def save_to_file(self, filepath: str) -> None:
        # Saves single JSON file at end of run
```

```python
# src/podcast_scraper/utils/provider_metrics.py
@dataclass
class ProviderCallMetrics:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    retries: int = 0
    rate_limit_sleep_sec: float = 0.0
    estimated_cost: Optional[float] = None
```

**GAP:** Current output is single JSON file at end - no streaming JSONL during run.

**Tasks:**

1. **Create JSONL Emitter That Uses Existing Metrics**
   - [ ] Create `src/podcast_scraper/workflow/jsonl_emitter.py`:
     - `JSONLEmitter` class
     - Constructor takes `Metrics` instance (the existing one)
     - `emit_run_started(cfg, run_id)` - writes to JSONL file
     - `emit_episode_finished(episode_id)` - pulls from existing `episode_metrics`
     - `emit_run_finished()` - pulls from existing `Metrics.finish()`
   - [ ] JSONL file is ADDITIONAL output, not replacement

2. **Define JSONL Event Schemas (reuse existing data)**
   - [ ] `run_started`: config fingerprint, run_id, timestamp
   - [ ] `episode_finished`: pull from existing `EpisodeMetrics` dataclass
   - [ ] `run_finished`: pull from existing `Metrics.finish()` dict

3. **Integrate into Pipeline**
   - [ ] Create emitter in `run_pipeline()` before processing
   - [ ] Call `emit_episode_finished()` after each episode completes
   - [ ] Call `emit_run_finished()` at end (alongside existing `save_to_file()`)

4. **Configuration**
   - [ ] Add `jsonl_metrics_enabled: bool = False` (opt-in initially)
   - [ ] Add `jsonl_metrics_path: Optional[str] = None` (default: `{output_dir}/run.jsonl`)

5. **Testing**
   - [ ] Unit tests for JSONL emitter
   - [ ] Verify JSONL output matches existing metrics data
   - [ ] Integration test: run pipeline, verify both JSON and JSONL outputs

**Files to Modify:**

- `src/podcast_scraper/workflow/jsonl_emitter.py` (NEW)
- `src/podcast_scraper/workflow/orchestration.py` (INTEGRATE emitter)
- `src/podcast_scraper/config.py` (ADD config options)
- `tests/unit/test_jsonl_emitter.py` (NEW)

**Key Principle:** The JSONL emitter should be a VIEW of existing metrics, not a parallel tracking system. All data flows through the existing `Metrics` and `EpisodeMetrics` classes.

**Acceptance Criteria:**

- [ ] JSONL output contains same data as JSON output (different format)
- [ ] JSONL streams during run (not just at end)
- [ ] Existing JSON output unchanged
- [ ] No duplicate metric tracking code

---

## Phase 3: DevEx & Ops - Secrets, Degradation

**Timeline:** ~1-2 weeks
**Risk Level:** Low-Medium (operational improvements, well-defined)
**Dependencies:** Phase 1, Phase 2 (JSONL metrics already done in Phase 2)

### Step 3.1: Structured Logs - Human-Readable Format (#401.8)

**Objective:** Keep existing human-readable logs while adding structured JSONL (already done in Phase 2.3).

**Tasks:**

1. **Review & Enhance**
   - [ ] Review JSONL metrics implementation (Phase 2.3)
   - [ ] Ensure human-readable logs remain unchanged
   - [ ] Add any missing metrics to JSONL format

2. **Documentation**
   - [ ] Document dual logging approach (human + JSONL)
   - [ ] Add examples for parsing JSONL

**Files to Modify:**
- `docs/api/METRICS.md` (update)

**Acceptance Criteria:**
- [ ] Human-readable logs unchanged
- [ ] JSONL metrics complement human logs
- [ ] Documentation updated

---

### Step 3.2: Secrets & Key Selection Sanity (#401.9)

**Objective:** Log non-sensitive provider metadata for debugging.

**Tasks:**

1. **Implementation**
   - [ ] Add provider metadata logging:
     - `provider_account` / `project` (non-sensitive identifiers)
     - `region` / `endpoint` (for region-specific providers)
     - Never log API keys (already implemented, verify consistency)
   - [ ] Add to structured logs (JSONL metrics)
   - [ ] Add to human-readable logs (debug level)

2. **Validation**
   - [ ] Add validation to detect mismatched keys/projects
   - [ ] Add clearer error messages when authentication fails

3. **Testing**
   - [ ] Unit tests for metadata logging
   - [ ] Verify API keys are never logged

4. **Documentation**
   - [ ] Document provider metadata in logs
   - [ ] Add troubleshooting guide for key issues

**Files to Modify:**
- `src/podcast_scraper/providers/openai/openai_provider.py`
- `src/podcast_scraper/providers/gemini/gemini_provider.py`
- `src/podcast_scraper/metrics/jsonl_logger.py`
- `tests/unit/test_secrets_logging.py` (new)
- `docs/guides/TROUBLESHOOTING.md`

**Acceptance Criteria:**
- [ ] Provider account/project logged (non-sensitive)
- [ ] Region/endpoint logged for region-specific providers
- [ ] API keys never logged
- [ ] Validation detects mismatched keys
- [ ] All tests pass

---

### Step 3.3: Graceful Degradation Rules (#401.10)

**Objective:** Define and encode policy for component failures.

**Tasks:**

1. **Design Degradation Policy**
   - [ ] Define `DegradationPolicy` configuration model:
     - `save_transcript_on_summarization_failure: bool = True`
     - `save_summary_on_entity_extraction_failure: bool = True`
     - `fallback_provider_on_failure: Optional[str] = None`
     - `continue_on_stage_failure: bool = True`
   - [ ] Document default behavior for each failure scenario

2. **Implementation**
   - [ ] Create `src/podcast_scraper/workflow/degradation.py`:
     - `DegradationPolicy` Pydantic model
     - `handle_stage_failure(stage, error, policy)` function
   - [ ] Integrate into pipeline:
     - Summarization failures
     - Entity extraction failures
     - Provider failures (with fallback)
   - [ ] Log degradation decisions clearly

3. **Configuration**
   - [ ] Add `degradation_policy` to `Config` model
   - [ ] Document in `docs/api/CONFIGURATION.md`

4. **Testing**
   - [ ] Unit tests for degradation policy
   - [ ] Integration tests for failure scenarios

5. **Documentation**
   - [ ] Document degradation behavior in `docs/ARCHITECTURE.md`
   - [ ] Add failure handling guide

**Files to Modify:**
- `src/podcast_scraper/workflow/degradation.py` (new)
- `src/podcast_scraper/config.py`
- `src/podcast_scraper/workflow/orchestration.py`
- `tests/unit/test_degradation.py` (new)
- `tests/integration/test_graceful_degradation.py` (new)
- `docs/api/CONFIGURATION.md`
- `docs/ARCHITECTURE.md`

**Acceptance Criteria:**
- [ ] DegradationPolicy configuration model defined
- [ ] Default behavior documented for each failure scenario
- [ ] Degradation decisions logged clearly
- [ ] Configurable fallback strategies
- [ ] All tests pass

---

## Phase 4: Refactoring - Transcript Cleaning Externalization

**Timeline:** ~1-2 weeks
**Risk Level:** Low (refactoring, well-defined scope)
**Dependencies:** Phase 1 (capability contract helps)

### Step 4.1: Externalize Transcript Cleaning (#331)

**Objective:** Decouple cleaning from summarization, make it provider-specific.

**Tasks:**

1. **Create Cleaning Module**
   - [ ] Create `src/podcast_scraper/cleaning/` module structure:
     - `base.py` - `TranscriptCleaningProcessor` protocol
     - `pattern_based.py` - `PatternBasedCleaner` (current implementation)
   - [ ] Extract current cleaning logic from `src/podcast_scraper/preprocessing.py`:
     - `clean_transcript()` → `PatternBasedCleaner.clean()`
     - `remove_sponsor_blocks()` → `PatternBasedCleaner.remove_sponsors()`
     - `remove_outro_blocks()` → `PatternBasedCleaner.remove_outros()`

2. **Provider Integration**
   - [ ] Add `cleaning_processor` property to provider interface
   - [ ] Default all providers to `PatternBasedCleaner`
   - [ ] Update summarization pipeline to use provider's cleaning processor

3. **Testing**
   - [ ] Unit tests for cleaning module
   - [ ] Integration tests for provider-specific cleaning
   - [ ] Verify no regression in cleaning quality

4. **Documentation**
   - [ ] Document cleaning architecture in `docs/ARCHITECTURE.md`
   - [ ] Update provider guides

**Files to Modify:**
- `src/podcast_scraper/cleaning/base.py` (new)
- `src/podcast_scraper/cleaning/pattern_based.py` (new)
- `src/podcast_scraper/preprocessing.py` (refactor)
- `src/podcast_scraper/workflow/stages/summarization.py`
- `tests/unit/test_cleaning.py` (new)
- `tests/integration/test_provider_cleaning.py` (new)
- `docs/ARCHITECTURE.md`

**Acceptance Criteria:**
- [ ] Cleaning logic is provider-specific and extensible
- [ ] Pattern-based cleaning remains default
- [ ] No regression in current cleaning quality
- [ ] All tests pass

---

## Phase 5: Documentation - Observations

**Timeline:** ~1 day
**Risk Level:** None (documentation only)
**Dependencies:** None

### Step 5.1: Document Speaker Detection Observation (#393)

**Tasks:**
- [ ] Review issue #393 (already well-documented)
- [ ] Add note to `docs/guides/TROUBLESHOOTING.md` about organization names in RSS
- [ ] Update speaker detection guide if needed

**Files to Modify:**
- `docs/guides/TROUBLESHOOTING.md`

---

### Step 5.2: Document Preprocessing Performance Observation (#392)

**Tasks:**
- [ ] Review issue #392 (already well-documented)
- [ ] Add note to `docs/guides/PERFORMANCE.md` (or create if needed) about cache miss cost
- [ ] Document optimization opportunities

**Files to Modify:**
- `docs/guides/PERFORMANCE.md` (create or update)

---

## Testing Strategy

### Unit Tests
- Each new module/feature requires comprehensive unit tests
- Mock external dependencies (HTTP, file I/O, providers)
- Test error cases and edge cases

### Integration Tests
- Test provider interactions end-to-end
- Test pipeline stages with real (but mocked) providers
- Test caching, queueing, and metrics collection

### E2E Tests
- Run full pipeline with test podcast feed
- Verify all features work together
- Ensure no regressions

### Test Execution Order
1. Run unit tests first (fast feedback)
2. Run integration tests (medium speed)
3. Run E2E tests before committing (slower, but comprehensive)

---

## Risk Mitigation

### High-Risk Areas
1. **Threading Changes (#383)**: Bounded queue changes threading model
   - Mitigation: Extensive integration testing, careful code review
2. **Provider Hardening (#399)**: Changes affect all providers
   - Mitigation: Test each provider individually, backward compatibility

### Rollback Strategy
- Each phase should be independently committable
- Feature flags for new behaviors (when possible)
- Comprehensive test coverage before merging

---

## Progress Tracking

### Phase Completion Checklist

**Phase 1: Foundation**
- [ ] #259: Protocol runtime verification
- [ ] #399.1: Unified retry & backoff
- [ ] #399.2: Timeouts everywhere
- [ ] #399.3: Provider capability contract
- [ ] #399.4: Normalized output schema

**Phase 2: Infrastructure**
- [ ] #383: Bounded queue
- [ ] #402.1: Transcript caching
- [ ] #402.2: JSONL metrics

**Phase 3: DevEx & Ops**
- [ ] #401.9: Secrets & key selection
- [ ] #401.10: Graceful degradation

**Phase 4: Refactoring**
- [ ] #331: Transcript cleaning externalization

**Phase 5: Documentation**
- [ ] #393: Speaker detection observation
- [ ] #392: Preprocessing performance observation

---

## Estimated Timeline

- **Phase 1**: 2-3 weeks (Foundation - Protocol & Provider Hardening)
- **Phase 2**: 2-3 weeks (Infrastructure - Queue & Caching)
- **Phase 3**: 1-2 weeks (DevEx & Ops - Secrets, Degradation)
- **Phase 4**: 1-2 weeks (Refactoring - Cleaning Externalization)
- **Phase 5**: 1 day (Documentation)

**Total**: ~6-10 weeks

---

## Success Criteria

1. ✅ All 8 in-scope issues addressed (#399, #401, #402, #383, #259, #331, #393, #392)
2. ✅ All tests pass (unit, integration, E2E)
3. ✅ No regressions in existing functionality
4. ✅ Code is shippable (linting, formatting, type checking)
5. ✅ Documentation updated
6. ✅ Single commit at end of all work (as requested)

---

## Notes

- This plan assumes sequential execution within phases, but some steps can be parallelized
- Each phase should be independently testable and committable
- Final commit will include all changes from all phases
- Regular progress updates should be made to this document as work progresses
