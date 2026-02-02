# Preprocessing Profiles Guide

**Status:** ✅ Complete - Profiles are fully integrated into the experiment pipeline (Issue #369, RFC-045).

This guide explains preprocessing profiles: what they are, why they matter, how to use them in experiments, and how to create new profiles.

---

## What Are Preprocessing Profiles?

**Preprocessing profiles** are versioned, reproducible configurations for cleaning transcript text before it's fed to AI models. They encapsulate all the cleaning steps (removing timestamps, normalizing speakers, stripping headers, etc.) into named, versioned configurations.

### Why Profiles Matter

Transcript cleaning has **as much impact on summary quality as the model itself**. However, without profiles:

- ❌ Cleaning logic is "hidden" inside functions
- ❌ Changes to regex patterns are untracked
- ❌ You can't isolate whether quality improvements came from model changes or preprocessing changes
- ❌ Reproducing results becomes impossible

**With profiles:**

- ✅ Every cleaning step is explicit and versioned
- ✅ Profiles are recorded in experiment fingerprints
- ✅ You can test Model A vs. Model B while keeping preprocessing identical
- ✅ You can test Preprocessing Profile A vs. Profile B while keeping the model identical
- ✅ Results are fully reproducible

### The Core Principle

> **Preprocessing changes are as impactful as model changes, so they must be tracked and versioned just like models.**

---

## Available Profiles

### `cleaning_v1` - Basic Cleaning

**Version:** 1.0
**Description:** Minimal cleaning for basic transcript processing.

**Steps:**

- ✅ Remove timestamps
- ✅ Normalize generic speaker labels (`Host:`, `Guest:` → `Speaker 1:`, `Speaker 2:`)
- ✅ Collapse blank lines
- ❌ Remove fillers (disabled)
- ❌ Remove sponsor blocks (disabled)
- ❌ Remove outro blocks (disabled)

**Use Case:** When you want minimal preprocessing, preserving most original transcript structure.

### `cleaning_v2` - Basic + Sponsor Removal

**Version:** 2.0
**Description:** Basic cleaning plus sponsor block removal.

**Steps:**

- ✅ All `cleaning_v1` steps
- ✅ Remove sponsor blocks

**Use Case:** When you want to remove ads but preserve outros and other structure.

### `cleaning_v3` - Full Cleaning (Default)

**Version:** 3.0
**Description:** Comprehensive cleaning for production use. This is the **default profile** for all experiments.

**Steps:**

- ✅ All `cleaning_v2` steps
- ✅ Remove outro blocks
- ✅ Remove garbage lines (punctuation artifacts, `////`, `====`)
- ✅ Remove credit blocks
- ✅ Remove summarization artifacts

**Use Case:** Standard preprocessing for most experiments. Use this unless you have a specific reason to use a different profile.

### `cleaning_none` - No Cleaning

**Version:** 1.0
**Description:** Pass-through profile that returns text unchanged.

**Steps:**

- ❌ All cleaning disabled

**Use Case:** When you want to test models on completely raw transcripts (rare, mainly for debugging).

---

## Using Profiles in Experiments

### In Experiment YAML Configs

Specify the preprocessing profile in your experiment configuration:

```yaml
id: "baseline_bart_v7_cleaning_v4"
task: "summarization"

backend:
  type: "hf_local"
  map_model: "bart-small"
  reduce_model: "long-fast"

data:
  dataset_id: "curated_5feeds_smoke_v1"

# Preprocessing profile selection
preprocessing_profile: "cleaning_v3"  # or "cleaning_v4", "cleaning_v1", etc.

map_params:
  max_new_tokens: 200
  # ... other params
```

**Default:** If you don't specify `preprocessing_profile`, it defaults to `"cleaning_v3"`.

### How Profiles Are Tracked

The preprocessing profile is automatically recorded in the experiment fingerprint:

```json
{
  "preprocessing": {
    "profile_id": "cleaning_v3",
    "profile_version": "3.0",
    "steps": {
      "remove_timestamps": true,
      "normalize_speakers": true,
      "remove_sponsor_blocks": true,
      "collapse_blank_lines": true,
      "remove_fillers": false,
      "remove_garbage_lines": true,
      "remove_credit_blocks": true,
      "remove_outro_blocks": true,
      "remove_artifacts": true
    }
  }
}
```

This ensures that:

- You can always see which profile was used for any experiment
- Fingerprints change when you switch profiles (enabling proper comparison)
- Results are fully reproducible

---

## When to Use Different Profiles

### Use `cleaning_v3` (Default) When

- ✅ Running standard experiments
- ✅ Comparing models (keeps preprocessing constant)
- ✅ Creating baselines
- ✅ You're not sure which profile to use

### Use `cleaning_v1` or `cleaning_v2` When

- ✅ You want to preserve more original transcript structure
- ✅ You're testing preprocessing impact (comparing v1/v2/v3)
- ✅ You have transcripts that don't need aggressive cleaning

### Use `cleaning_none` When

- ✅ Debugging preprocessing issues
- ✅ Testing models on completely raw input
- ✅ You want to see exactly what the model receives

### Create `cleaning_v4` When

- ✅ You need speaker anonymization (Maya: → A:)
- ✅ You need episode header stripping
- ✅ You're following RFC-045 optimization guide
- ✅ You want to test preprocessing improvements

---

## Creating New Profiles

### Step 1: Implement the Cleaning Function

Create a function that takes raw text and returns cleaned text:

```python
# In src/podcast_scraper/preprocessing/profiles.py

def _cleaning_v4(text: str) -> str:
    """Enhanced cleaning with speaker anonymization and header stripping.

    This profile adds:
    - Speaker anonymization (Maya: → A:)
    - Episode header stripping
    - All cleaning_v3 steps
    """
    # 1. Strip episode header (title, Host:, Guest:)
    cleaned = strip_episode_header(text)

    # 2. Strip credits
    cleaned = strip_credits(cleaned)

    # 3. Strip garbage lines
    cleaned = strip_garbage_lines(cleaned)

    # 4. Anonymize speakers (Maya: → A:)
    cleaned = anonymize_speakers(cleaned)

    # 5. Standard cleaning (from cleaning_v3)
    cleaned = clean_transcript(
        cleaned,
        remove_timestamps=True,
        normalize_speakers=True,
        collapse_blank_lines=True,
        remove_fillers=False,
    )

    # 6. Remove sponsor/outro blocks
    cleaned = remove_sponsor_blocks(cleaned)
    cleaned = remove_outro_blocks(cleaned)

    # 7. Remove BART/LED artifacts
    cleaned = remove_summarization_artifacts(cleaned)

    return cleaned.strip()
```

### Step 2: Register the Profile

Register your function with a unique profile ID:

```python
# In src/podcast_scraper/preprocessing/profiles.py

register_profile("cleaning_v4", _cleaning_v4)
```

### Step 3: Add Profile Metadata (Optional)

If you want the profile to appear in fingerprints with version info, update the metadata functions:

```python
# In scripts/eval/materialize_baseline.py

def get_preprocessing_profile_version(profile_id: str) -> str:
    """Get version string for a preprocessing profile."""
    versions = {
        "cleaning_v1": "1.0",
        "cleaning_v2": "2.0",
        "cleaning_v3": "3.0",
        "cleaning_v4": "4.0",  # Add your new profile
        "cleaning_none": "1.0",
    }
    return versions.get(profile_id, "unknown")
```

### Step 4: Test Your Profile

Test the profile in an experiment:

```yaml
# data/eval/configs/test_cleaning_v4.yaml
id: "test_cleaning_v4"
preprocessing_profile: "cleaning_v4"
# ... rest of config
```

Run the experiment:

```bash
make experiment-run CONFIG=data/eval/configs/test_cleaning_v4.yaml
```

Check the fingerprint to verify the profile was recorded:

```bash
cat data/eval/runs/test_cleaning_v4/fingerprint.json | jq '.preprocessing'
```

---

## Profile Comparison Strategy

When optimizing ML quality, use profiles to isolate preprocessing impact:

| Experiment | Preprocessing | Parameters | Purpose |
| ---------- | ------------- | ---------- | ------- |
| v1 (baseline) | `cleaning_v3` | default | Control |
| v2-v6 | `cleaning_v3` | varied | Parameter tuning |
| **v7** | **`cleaning_v4`** | default | **Preprocessing impact** |
| **v8** | **`cleaning_v4`** | best from v2-v6 | **Combined optimization** |

**Key Insight:** v7 isolates preprocessing impact, v8 combines both improvements.

See [RFC-045: ML Model Optimization Guide](../rfc/RFC-045-ml-model-optimization-guide.md) for detailed optimization strategies.

---

## Technical Details

### How Profiles Are Applied

The preprocessing profile is wired through the experiment pipeline:

```text
experiment.yaml                    # preprocessing_profile: "cleaning_v4"
    ↓
run_experiment.py                  # Reads config, passes to provider
    ↓
MLProvider.summarize()             # Accepts preprocessing_profile param
    ↓
summarize_long_text()              # Passes to preprocessing
    ↓
apply_profile(text, profile_id)    # Actually uses the profile!
```

### Profile Registry

Profiles are stored in a registry:

```python
from podcast_scraper.preprocessing.profiles import (
    apply_profile,
    list_profiles,
    get_profile,
)

# Apply a profile
cleaned = apply_profile(text, "cleaning_v3")

# List available profiles
profiles = list_profiles()  # ["cleaning_v1", "cleaning_v2", "cleaning_v3", ...]

# Get profile function (for inspection)
profile_func = get_profile("cleaning_v3")
```

### Profile Functions

Profile functions must:

- ✅ Take a single `str` argument (raw text)
- ✅ Return a single `str` (cleaned text)
- ✅ Be deterministic (same input → same output)
- ✅ Be idempotent (applying twice should be safe)

---

## Common Questions

### Q: Can I use different profiles for different episodes?

**A:** No. The preprocessing profile is set per-experiment, not per-episode. All episodes in an experiment use the same profile. This ensures consistency and enables fair comparison.

### Q: What if I need custom preprocessing for a specific use case?

**A:** Create a new profile (e.g., `cleaning_custom_v1`) and register it. Then use it in your experiment config.

### Q: How do I know which profile was used for a baseline?

**A:** Check the baseline's fingerprint:

```bash
cat data/eval/baselines/bart_led_baseline_v1/fingerprint.json | jq '.preprocessing.profile_id'
```

### Q: Can I change a profile after creating a baseline?

**A:** Technically yes, but **don't**. Profiles should be immutable once used in baselines. If you need changes, create a new profile version (e.g., `cleaning_v4` instead of modifying `cleaning_v3`).

### Q: Do profiles affect all providers (ML, OpenAI, etc.)?

**A:** Currently, profiles are applied in the ML provider's summarization pipeline. OpenAI provider doesn't use preprocessing profiles (it receives pre-cleaned text if preprocessing happens earlier in the pipeline).

---

## Related Documentation

- **[ADR-029: Registered Preprocessing Profiles](../adr/ADR-029-registered-preprocessing-profiles.md)** - Design decision rationale
- **[RFC-045: ML Model Optimization Guide](../rfc/RFC-045-ml-model-optimization-guide.md)** - How to use profiles for quality optimization
- **[Experiment Guide](EXPERIMENT_GUIDE.md)** - How to run experiments with profiles
- **[API: Configuration](../api/CONFIGURATION.md)** - Technical API reference

---

## Summary

**Preprocessing profiles** are versioned, reproducible configurations for cleaning transcripts. They:

1. **Isolate variables** - Test models vs. preprocessing independently
2. **Enable reproducibility** - Every experiment records which profile was used
3. **Standardize cleaning** - Consistent preprocessing across all episodes
4. **Track changes** - Fingerprints change when profiles change

**Default:** Use `cleaning_v3` unless you have a specific reason to use a different profile.

**Next Steps:**

- Read [RFC-045](../rfc/RFC-045-ml-model-optimization-guide.md) for optimization strategies
- Check [Experiment Guide](EXPERIMENT_GUIDE.md) for how to use profiles in experiments
- See [ADR-029](../adr/ADR-029-registered-preprocessing-profiles.md) for design rationale
