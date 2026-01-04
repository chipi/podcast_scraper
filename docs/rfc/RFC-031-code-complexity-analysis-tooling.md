# RFC-031: Code Complexity Analysis Tooling

- **Status**: Draft
- **Authors**:
- **Stakeholders**: Maintainers, developers, CI/CD pipeline maintainers
- **Related PRDs**: None
- **Related RFCs**:
  - `docs/rfc/RFC-030-python-test-coverage-improvements.md` (coverage improvements)
- **Related Documents**:
  - `docs/guides/DEVELOPMENT_GUIDE.md` - Development workflow
  - `docs/CI_CD.md` - CI/CD pipeline documentation
  - `.github/workflows/python-app.yml` - Main CI workflow

## Abstract

This RFC proposes adding code complexity analysis tooling to augment the existing CI/CD setup.
The goal is to identify overly complex code, improve code documentation, detect dead code, and
maintain code quality standards through automated tooling.

**Proposed Tools:**

1. **radon** - Code complexity metrics (Cyclomatic Complexity, Maintainability Index)
2. **vulture** - Dead code detection
3. **interrogate** - Docstring coverage checking
4. **codespell** - Spell checking for code and documentation

## Current State Analysis

### Existing Static Analysis Tools

The project already has a solid foundation of static analysis:

| Tool | Purpose | Status |
|------|---------|--------|
| **black** | Code formatting | ✅ In CI |
| **isort** | Import sorting | ✅ In CI |
| **flake8** | Linting + basic complexity | ✅ In CI |
| **mypy** | Type checking | ✅ In CI |
| **bandit** | Security scanning | ✅ In CI |
| **pip-audit** | Dependency vulnerability | ✅ In CI |
| **markdownlint** | Markdown linting | ✅ In CI |

### Flake8 Complexity (Current)

The project has McCabe complexity checking via flake8:

```ini
# .flake8
max-complexity = 25
per-file-ignores =
    config.py:C901
    episode_processor.py:C901
    workflow.py:C901
    speaker_detection.py:C901
    whisper_integration.py:C901
```

**Issues:**

- Threshold of 25 is very high (10-15 is typical)
- 5 modules are exempted from complexity checks
- No detailed metrics beyond pass/fail
- No maintainability index or other insights

### Codebase Size

| Module | Lines | Notes |
|--------|-------|-------|
| `workflow.py` | 2,580 | Largest, orchestration |
| `summarizer.py` | 2,401 | ML summarization |
| `speaker_detection.py` | 1,076 | NER extraction |
| Other modules | ~6,500 | Various |
| **Total** | ~12,600 | Main package |

## Problem Statement

### Gaps Identified

1. **No Detailed Complexity Metrics**
   - Flake8's C901 is pass/fail only
   - No visibility into which functions are most complex
   - No maintainability index tracking
   - High threshold (25) masks issues

2. **No Dead Code Detection**
   - Unused functions, variables, and imports may exist
   - No automated detection of orphaned code
   - Manual cleanup is time-consuming and error-prone

3. **No Docstring Coverage Checking**
   - No enforcement of documentation standards
   - Public APIs may lack documentation
   - No visibility into docstring coverage percentage

4. **No Spell Checking**
   - Typos in code comments, docstrings, and docs
   - No automated detection during CI

## Goals

### Primary Goals

1. **Visibility**: Detailed complexity metrics for informed decisions
2. **Code Quality**: Identify and reduce complexity hotspots
3. **Documentation**: Enforce docstring coverage for public APIs
4. **Cleanliness**: Detect and remove dead code
5. **Correctness**: Catch typos in code and documentation

### Success Criteria

- ✅ Complexity metrics visible in CI job summary
- ✅ Dead code detection runs in CI (informational initially)
- ✅ Docstring coverage tracked with threshold
- ✅ Spell checking catches common typos
- ✅ No false positives blocking CI (thresholds tuned)

## Proposed Tools

### 1. radon - Code Complexity Metrics

**Purpose:** Detailed complexity analysis beyond flake8's pass/fail.

**Metrics Provided:**

- **Cyclomatic Complexity (CC)**: Number of linearly independent paths
- **Maintainability Index (MI)**: 0-100 score for maintainability
- **Raw Metrics**: LOC, SLOC, comments, blank lines
- **Halstead Metrics**: Difficulty, effort, bugs predicted

**Installation:**

```bash
pip install radon
```

**Usage:**

```bash
# Cyclomatic complexity (functions with CC >= 10)
radon cc src/podcast_scraper/ -a -s -nc

# Maintainability index (lower is worse)
radon mi src/podcast_scraper/ -s

# Raw metrics
radon raw src/podcast_scraper/ -s
```

**Recommended Thresholds:**

| Grade | CC Range | Interpretation |
|-------|----------|----------------|
| A | 1-5 | Low risk, simple |
| B | 6-10 | Moderate complexity |
| C | 11-20 | Complex, higher risk |
| D | 21-30 | Very complex |
| E | 31-40 | Untestable |
| F | 41+ | Error-prone |

**CI Integration:**

```yaml
- name: Check code complexity
  run: |
    pip install radon
    echo "## Code Complexity" >> $GITHUB_STEP_SUMMARY
    echo '```' >> $GITHUB_STEP_SUMMARY
    radon cc src/podcast_scraper/ -a -s --total-average >> $GITHUB_STEP_SUMMARY
    echo '```' >> $GITHUB_STEP_SUMMARY
```

**Recommendation:** Start with informational output, then add thresholds:

```bash
# Fail if any function has CC > 15
radon cc src/podcast_scraper/ -a -s --max-cc 15
```

### 2. vulture - Dead Code Detection

**Purpose:** Find unused code (functions, variables, classes, imports).

**Installation:**

```bash
pip install vulture
```

**Usage:**

```bash
# Find unused code (60% confidence threshold)
vulture src/podcast_scraper/ --min-confidence 60

# Generate whitelist for false positives
vulture src/podcast_scraper/ --make-whitelist > vulture_whitelist.py
```

**Configuration:**

Create `vulture_whitelist.py` for known false positives:

```python
# vulture_whitelist.py
# These are used dynamically or externally

# Pydantic validators are called by framework
_.model_validator  # unused method
_.field_validator  # unused method

# Click decorators
_.callback  # unused method

# Test fixtures
_.fixture  # unused function
```

**Recommended Approach:**

1. Run initially to identify dead code
2. Create whitelist for false positives
3. Add to CI as informational
4. Gradually enforce as cleanup progresses

**CI Integration:**

```yaml
- name: Check for dead code
  run: |
    pip install vulture
    vulture src/podcast_scraper/ --min-confidence 80 || true
  continue-on-error: true  # Informational only initially
```

### 3. interrogate - Docstring Coverage

**Purpose:** Check for missing docstrings in public APIs.

**Installation:**

```bash
pip install interrogate
```

**Usage:**

```bash
# Check docstring coverage
interrogate src/podcast_scraper/ -v

# With badge generation
interrogate src/podcast_scraper/ -v --generate-badge docs/badges/

# Fail if coverage below threshold
interrogate src/podcast_scraper/ --fail-under 80
```

**Configuration (`pyproject.toml`):**

```toml
[tool.interrogate]
ignore-init-module = true
ignore-init-method = true
ignore-magic = true
ignore-semiprivate = true
ignore-private = true
ignore-property-decorators = true
ignore-module = true
ignore-nested-functions = true
fail-under = 80
exclude = ["tests", "scripts"]
verbose = 1
```

**Recommended Thresholds:**

| Level | Coverage | Action |
|-------|----------|--------|
| 🟢 Good | ≥ 80% | Target |
| 🟡 Warning | 60-80% | Improve gradually |
| 🔴 Fail | < 60% | Needs attention |

**CI Integration:**

```yaml
- name: Check docstring coverage
  run: |
    pip install interrogate
    interrogate src/podcast_scraper/ -v --fail-under 60
```

### 4. codespell - Spell Checking

**Purpose:** Catch typos in code, comments, and documentation.

**Installation:**

```bash
pip install codespell
```

**Usage:**

```bash
# Check for typos
codespell src/ docs/ --skip="*.pyc,*.pyo,*.egg-info,*.git"

# Auto-fix typos
codespell src/ docs/ -w

# With custom dictionary
codespell src/ docs/ -I .codespell-ignore
```

**Configuration:**

Create `.codespell-ignore` for false positives:

```text
# Known technical terms that look like typos
ba  # Used in audio contexts
fo  # Used in some variable names
```

**CI Integration:**

```yaml
- name: Check spelling
  run: |
    pip install codespell
    codespell src/ docs/ --skip="*.pyc,*.json,*.xml,*.lock"
```

## Implementation Plan

### Phase 1: Add Dependencies (15 min)

Add to `pyproject.toml`:

```toml
dev = [
  # ... existing tools ...
  "radon>=6.0.0,<7.0.0",
  "vulture>=2.10,<3.0.0",
  "interrogate>=1.5.0,<2.0.0",
  "codespell>=2.2.0,<3.0.0",
]
```

### Phase 2: Add Makefile Targets (30 min)

```makefile
# Code complexity analysis
complexity:
	radon cc src/podcast_scraper/ -a -s --total-average
	@echo ""
	@echo "Maintainability Index:"
	radon mi src/podcast_scraper/ -s

# Dead code detection
deadcode:
	vulture src/podcast_scraper/ --min-confidence 80

# Docstring coverage
docstrings:
	interrogate src/podcast_scraper/ -v

# Spell checking
spelling:
	codespell src/ docs/ --skip="*.pyc,*.json,*.xml,*.lock,*.mp3"

# All code quality checks
quality: complexity deadcode docstrings spelling
```

### Phase 3: Configure Tools (1 hour)

Add to `pyproject.toml`:

```toml
[tool.interrogate]
ignore-init-module = true
ignore-init-method = true
ignore-magic = true
ignore-semiprivate = true
ignore-private = true
ignore-property-decorators = true
ignore-module = true
ignore-nested-functions = true
fail-under = 60
exclude = ["tests", "scripts"]
verbose = 1

[tool.vulture]
min_confidence = 80
paths = ["src/podcast_scraper"]
exclude = ["tests/", "scripts/"]
```

Create `.codespell-ignore`:

```text
# Project-specific terms that look like typos
# Add words here as needed
```

### Phase 4: CI Integration - Informational (1 hour)

Add to `.github/workflows/python-app.yml` lint job:

```yaml
- name: Code quality analysis
  run: |
    pip install radon vulture interrogate codespell

    echo "## 📊 Code Quality Report" >> $GITHUB_STEP_SUMMARY

    echo "### Complexity Analysis" >> $GITHUB_STEP_SUMMARY
    echo '```' >> $GITHUB_STEP_SUMMARY
    radon cc src/podcast_scraper/ -a -s --total-average >> $GITHUB_STEP_SUMMARY
    echo '```' >> $GITHUB_STEP_SUMMARY

    echo "### Docstring Coverage" >> $GITHUB_STEP_SUMMARY
    echo '```' >> $GITHUB_STEP_SUMMARY
    interrogate src/podcast_scraper/ -v >> $GITHUB_STEP_SUMMARY 2>&1 || true
    echo '```' >> $GITHUB_STEP_SUMMARY
  continue-on-error: true  # Informational only initially
```

### Phase 5: Enable Enforcement (Future)

After baseline is established and issues addressed:

```yaml
- name: Enforce code quality
  run: |
    # Fail if complexity too high
    radon cc src/podcast_scraper/ -a --max-cc 15

    # Fail if docstring coverage too low
    interrogate src/podcast_scraper/ --fail-under 70

    # Fail on typos
    codespell src/ docs/ --skip="*.pyc,*.json,*.xml,*.lock"
```

## Phased Rollout Strategy

### Week 1: Baseline

1. Add tools to dev dependencies
2. Run locally to establish baseline
3. Document current state (complexity hotspots, missing docstrings)

### Week 2: CI Integration (Informational)

1. Add to CI with `continue-on-error: true`
2. Add GitHub Job Summary for visibility
3. Monitor output for false positives

### Week 3: Tune Thresholds

1. Create whitelists for false positives
2. Adjust thresholds based on baseline
3. Address low-hanging fruit (easy fixes)

### Week 4: Enable Enforcement

1. Enable fail-on-threshold for selected tools
2. Start with codespell (most straightforward)
3. Gradually enable others as codebase improves

## Recommendations

### Immediate Actions (Quick Wins)

| Action | Tool | Effort | Impact |
|--------|------|--------|--------|
| Add codespell to CI | codespell | 15 min | Catch typos |
| Add complexity report | radon | 30 min | Visibility |
| Add docstring check | interrogate | 30 min | Documentation |

### Short-term (1-2 Weeks)

| Action | Tool | Effort | Impact |
|--------|------|--------|--------|
| Configure interrogate thresholds | interrogate | 1 hour | Enforce docs |
| Create vulture whitelist | vulture | 1 hour | Dead code visibility |
| Add quality Makefile target | All | 30 min | Local workflow |

### Medium-term (1 Month)

| Action | Tool | Effort | Impact |
|--------|------|--------|--------|
| Address complexity hotspots | radon | 4-8 hours | Code quality |
| Improve docstring coverage | interrogate | 4-8 hours | Documentation |
| Clean up dead code | vulture | 2-4 hours | Cleaner codebase |

## Existing Complexity Exemptions

The current `.flake8` exempts these files from C901 (complexity):

| File | Lines | Why Exempt |
|------|-------|------------|
| `config.py` | ~500 | Pydantic validators |
| `episode_processor.py` | ~600 | Processing logic |
| `workflow.py` | 2,580 | Orchestration |
| `speaker_detection.py` | 1,076 | NER logic |
| `whisper_integration.py` | 328 | Whisper interface |

**Recommendation:** Review these files with radon to identify specific complex functions, then refactor rather than blanket exemptions.

## Benefits

### Developer Experience

- ✅ Clear visibility into code quality metrics
- ✅ Automated detection of common issues
- ✅ Guidance on what to improve
- ✅ Consistent standards across team

### Code Quality

- ✅ Identify complexity hotspots before they grow
- ✅ Enforce documentation standards
- ✅ Remove unused code
- ✅ Catch typos automatically

### Maintainability

- ✅ Easier onboarding (better docs)
- ✅ Lower bug risk (simpler code)
- ✅ Smaller codebase (no dead code)
- ✅ Professional appearance (no typos)

## Risks and Mitigations

### Risk 1: False Positives Block CI

**Mitigation:**

- Start with `continue-on-error: true`
- Create whitelists for known false positives
- Enable enforcement gradually

### Risk 2: Too Many Initial Findings

**Mitigation:**

- Focus on new code first (higher standards)
- Address existing issues incrementally
- Start with high thresholds, lower over time

### Risk 3: Tool Conflicts

**Mitigation:**

- Run tools in sequence, not parallel
- Ensure compatible versions
- Test locally before CI

## Related Files

- `pyproject.toml` - Tool configuration and dependencies
- `.flake8` - Existing complexity configuration
- `Makefile` - Development commands
- `.github/workflows/python-app.yml` - CI workflow

## Notes

- All tools are pure Python with no system dependencies
- Tools can be run locally with `make quality`
- Start informational, enable enforcement after tuning
- Complexity exemptions in `.flake8` should be reviewed
- Large modules (workflow.py, summarizer.py) are prime candidates for refactoring

