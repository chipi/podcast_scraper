# RFC-022: Environment Variable Candidates Analysis

- **Status**: Completed (Historical Reference)
- **Authors**:
- **Date**: 2025-12-22
- **Stakeholders**: Maintainers, developers, DevOps engineers
- **Related RFCs**:
  - `docs/rfc/RFC-013-openai-provider-implementation.md` - OpenAI provider implementation (uses environment variables)
- **Related Documents**:
  - `docs/api/configuration.md` - Complete documentation of implemented environment variables

## Overview

**Note**: This RFC documents the analysis and implementation plan for environment variable support. All recommended high and medium priority environment variables have been implemented. This document is kept as a historical reference for understanding the design decisions.

**Purpose**: Analyze all configuration options to identify good candidates for environment variable support

## Current State

Currently supported environment variables:

- ✅ `OPENAI_API_KEY` - API key (security-sensitive)
- ✅ `LOG_LEVEL` - Runtime logging control

## Analysis Framework

We evaluate candidates based on:

1. **Security**: Contains sensitive data (API keys, secrets)
2. **Deployment Flexibility**: Values differ per environment (dev/staging/prod)
3. **Runtime Control**: Users want to change without modifying config files
4. **Docker/CI/CD**: Commonly set in containerized deployments
5. **User Convenience**: Frequently changed values

## Field-by-Field Analysis

### 🔴 High Priority Candidates

#### 1. **`OUTPUT_DIR`** / `OUTPUT_DIRECTORY`

- **Rationale**:
  - Different per environment (dev vs prod)
  - Docker volumes often need custom paths
  - CI/CD pipelines need flexible output locations
- **Use Case**: `OUTPUT_DIR=/data/transcripts python3 -m podcast_scraper ...`
- **Priority**: HIGH
- **Recommendation**: ✅ **ADD**

#### 2. **`CACHE_DIR`** / `SUMMARY_CACHE_DIR`

- **Rationale**:
  - Model cache location varies by deployment
  - Docker containers need persistent cache volumes
  - Different users/devices have different cache locations
- **Use Case**: `SUMMARY_CACHE_DIR=/cache/models python3 -m podcast_scraper ...`
- **Priority**: HIGH
- **Recommendation**: ✅ **ADD**

#### 3. **`LOG_FILE`**

- **Rationale**:
  - Log file location often environment-specific
  - Docker/containers need log file paths
  - Similar to LOG_LEVEL (already supported)
- **Use Case**: `LOG_FILE=/var/log/podcast_scraper.log python3 -m podcast_scraper ...`
- **Priority**: HIGH
- **Recommendation**: ✅ **ADD**

### 🟡 Medium Priority Candidates

#### 4. **`WORKERS`**

- **Rationale**:
  - Performance tuning per environment
  - Docker containers may have CPU limits
  - CI/CD might want fewer workers
- **Use Case**: `WORKERS=4 python3 -m podcast_scraper ...`
- **Priority**: MEDIUM
- **Recommendation**: ✅ **ADD** (useful for deployment flexibility)

#### 5. **`TRANSCRIPTION_PARALLELISM`**

- **Rationale**:
  - Performance tuning
  - OpenAI API rate limits might require adjustment
- **Use Case**: `TRANSCRIPTION_PARALLELISM=3 python3 -m podcast_scraper ...`
- **Priority**: MEDIUM
- **Recommendation**: ✅ **ADD** (useful for OpenAI provider tuning)

#### 6. **`PROCESSING_PARALLELISM`**

- **Rationale**:
  - Performance tuning per environment
  - Memory constraints might require adjustment
- **Use Case**: `PROCESSING_PARALLELISM=4 python3 -m podcast_scraper ...`
- **Priority**: MEDIUM
- **Recommendation**: ✅ **ADD**

#### 7. **`SUMMARY_BATCH_SIZE`**

- **Rationale**:
  - Performance tuning
  - Memory constraints per environment
- **Use Case**: `SUMMARY_BATCH_SIZE=2 python3 -m podcast_scraper ...`
- **Priority**: MEDIUM
- **Recommendation**: ✅ **ADD**

#### 8. **`SUMMARY_CHUNK_PARALLELISM`**

- **Rationale**:
  - CPU-bound performance tuning
  - Varies by hardware
- **Use Case**: `SUMMARY_CHUNK_PARALLELISM=2 python3 -m podcast_scraper ...`
- **Priority**: MEDIUM
- **Recommendation**: ✅ **ADD**

#### 9. **`TIMEOUT`**

- **Rationale**:
  - Network conditions vary by environment
  - CI/CD might need longer timeouts
- **Use Case**: `TIMEOUT=60 python3 -m podcast_scraper ...`
- **Priority**: MEDIUM
- **Recommendation**: ✅ **ADD**

#### 10. **`SUMMARY_DEVICE`**

- **Rationale**:
  - Hardware-specific (CPU vs CUDA vs MPS)
  - Docker containers might not have GPU access
  - CI/CD runs on CPU
- **Use Case**: `SUMMARY_DEVICE=cpu python3 -m podcast_scraper ...`
- **Priority**: MEDIUM
- **Recommendation**: ✅ **ADD** (useful for deployment flexibility)

### 🟢 Low Priority / Maybe Candidates

#### 11. **`DRY_RUN`**

- **Rationale**:
  - Testing/debugging flag
  - Useful for CI/CD validation
- **Use Case**: `DRY_RUN=true python3 -m podcast_scraper ...`
- **Priority**: LOW
- **Recommendation**: ⚠️ **MAYBE** (useful but not critical)

#### 12. **`SKIP_EXISTING`**

- **Rationale**:
  - Common flag for resuming interrupted runs
  - Useful in scripts
- **Use Case**: `SKIP_EXISTING=true python3 -m podcast_scraper ...`
- **Priority**: LOW
- **Recommendation**: ⚠️ **MAYBE** (convenient but not critical)

#### 13. **`CLEAN_OUTPUT`**

- **Rationale**:
  - Dangerous flag, might want to control via env
  - CI/CD might want clean runs
- **Use Case**: `CLEAN_OUTPUT=true python3 -m podcast_scraper ...`
- **Priority**: LOW
- **Recommendation**: ⚠️ **MAYBE** (safety consideration)

#### 14. **`LANGUAGE`**

- **Rationale**:
  - Might vary per deployment
  - Some users might want to override
- **Use Case**: `LANGUAGE=fr python3 -m podcast_scraper ...`
- **Priority**: LOW
- **Recommendation**: ⚠️ **MAYBE** (rarely changes)

#### 15. **`WHISPER_MODEL`**

- **Rationale**:
  - Model selection might vary by hardware
  - Some deployments might prefer smaller models
- **Use Case**: `WHISPER_MODEL=small python3 -m podcast_scraper ...`
- **Priority**: LOW
- **Recommendation**: ⚠️ **MAYBE** (rarely changes)

### ❌ Not Recommended

#### Fields that should NOT be environment variables

1. **`RSS_URL`** - Required parameter, better as CLI arg
2. **`MAX_EPISODES`** - Workflow-specific, better in config
3. **`PREFER_TYPES`** - List/array, complex to parse from env
4. **`SCREENPLAY_*`** - Feature flags, better in config
5. **`SPEAKER_NAMES`** - List/array, complex to parse
6. **`RUN_ID`** - Workflow-specific
7. **`GENERATE_METADATA`** - Feature flag, better in config
8. **`GENERATE_SUMMARIES`** - Feature flag, better in config
9. **`METADATA_FORMAT`** - Workflow preference, better in config
10. **`METADATA_SUBDIRECTORY`** - Workflow-specific
11. **`SUMMARY_PROVIDER`** - Provider selection, better in config
12. **`TRANSCRIPTION_PROVIDER`** - Provider selection, better in config
13. **`SPEAKER_DETECTOR_TYPE`** - Provider selection, better in config
14. **`SUMMARY_MODEL`** - Model identifier, better in config
15. **`SUMMARY_PROMPT`** - Long text, better in config file
16. **`USER_AGENT`** - Rarely changes, better in config
17. **`DELAY_MS`** - Workflow-specific, better in config
18. **`TRANSCRIBE_MISSING`** - Feature flag, better in config
19. **`AUTO_SPEAKERS`** - Feature flag, better in config
20. **`CACHE_DETECTED_HOSTS`** - Feature flag, better in config
21. **`REUSE_MEDIA`** - Testing flag, better in config
22. **`SAVE_CLEANED_TRANSCRIPT`** - Feature flag, better in config
23. **`NER_MODEL`** - Model identifier, better in config
24. **OpenAI model names** (`OPENAI_TRANSCRIPTION_MODEL`, etc.) - Better in config
25. **`OPENAI_TEMPERATURE`** - Better in config (workflow-specific)
26. **`OPENAI_MAX_TOKENS`** - Better in config (workflow-specific)
27. **Summary chunking parameters** - Better in config (workflow-specific)

## Implementation Status

**All recommended environment variables (Phase 1 and Phase 2) have been implemented.**

### Phase 1: High Priority (Deployment Essentials) - ✅ COMPLETED

1. ✅ `OUTPUT_DIR` - Critical for Docker/CI/CD - **IMPLEMENTED**
2. ✅ `CACHE_DIR` / `SUMMARY_CACHE_DIR` - Critical for Docker/CI/CD - **IMPLEMENTED**
3. ✅ `LOG_FILE` - Common deployment need - **IMPLEMENTED**

### Phase 2: Medium Priority (Performance Tuning) - ✅ COMPLETED

4. ✅ `WORKERS` - Performance tuning - **IMPLEMENTED**
5. ✅ `TRANSCRIPTION_PARALLELISM` - OpenAI provider tuning - **IMPLEMENTED**
6. ✅ `PROCESSING_PARALLELISM` - Performance tuning - **IMPLEMENTED**
7. ✅ `SUMMARY_BATCH_SIZE` - Memory management - **IMPLEMENTED**
8. ✅ `SUMMARY_CHUNK_PARALLELISM` - CPU tuning - **IMPLEMENTED**
9. ✅ `TIMEOUT` - Network flexibility - **IMPLEMENTED**
10. ✅ `SUMMARY_DEVICE` - Hardware flexibility - **IMPLEMENTED**

### Phase 3: Low Priority (Convenience) - ⚠️ NOT IMPLEMENTED (As Recommended)

11. ⚠️ `DRY_RUN` - Testing convenience - **NOT IMPLEMENTED** (marked as "maybe" in original analysis)
12. ⚠️ `SKIP_EXISTING` - Resumption convenience - **NOT IMPLEMENTED** (marked as "maybe" in original analysis)
13. ⚠️ `CLEAN_OUTPUT` - Safety control - **NOT IMPLEMENTED** (marked as "maybe" in original analysis)

**Note**: Phase 3 variables were marked as "maybe" in the original analysis and were intentionally not implemented, as they are better suited for config files or CLI flags.

## Implementation Notes

### Naming Convention

- Use UPPER_SNAKE_CASE for environment variables
- Match field names where possible: `output_dir` → `OUTPUT_DIR`
- For nested concepts: `summary_cache_dir` → `SUMMARY_CACHE_DIR`

### Priority Order

- **Standard**: Config file > Environment variable > Default
- **Exception**: `LOG_LEVEL` (env var takes precedence, as implemented)

### Validation

- All environment variables should go through existing field validators
- Use `@model_validator(mode="before")` pattern (like `LOG_LEVEL`)
- Or use `@field_validator(mode="before")` pattern (like `OPENAI_API_KEY`)

### Documentation

- Update `docs/api/configuration.md` for each new variable
- Update `examples/.env.example` template
- Add examples in usage sections

## Summary

**Implementation Status**: ✅ **COMPLETED**

- **Implemented**: 10 fields (3 high priority, 7 medium priority) - **ALL COMPLETED**
- **Not implemented**: 3 fields (low priority, convenience) - **As recommended** (marked as "maybe")
- **Not recommended**: 27+ fields (better suited for config files) - **As recommended**

The implemented fields focus on:

- **Deployment flexibility** (paths, cache locations) - ✅ All implemented
- **Performance tuning** (parallelism, workers, timeouts) - ✅ All implemented
- **Hardware adaptation** (device selection) - ✅ All implemented

These align with common use cases in Docker, CI/CD, and multi-environment deployments.

**Documentation**: All implemented environment variables are documented in `docs/api/configuration.md`.

**See Also**:

- `docs/api/configuration.md` - Complete documentation of all supported environment variables
- `src/podcast_scraper/config.py` - Implementation in `_preprocess_config_data()` method
