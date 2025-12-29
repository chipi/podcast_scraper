# CI/CD Pipeline

## Overview

The Podcast Scraper project uses GitHub Actions for continuous integration and deployment. The CI/CD
pipeline consists of **five main workflows** that automate testing, code quality checks, security
scanning, Docker validation, and documentation deployment.

### Workflows Summary

| Workflow                 | File             | Purpose                                               | Trigger                                                                    |
| ------------------------ | ---------------- | ----------------------------------------------------- | -------------------------------------------------------------------------- |
| **Python Application**   | `python-app.yml` | Main CI pipeline with testing, linting, and builds    | Push/PR to `main` (only when Python/config files change)                   |
| **Documentation Deploy** | `docs.yml`       | Build and deploy MkDocs documentation to GitHub Pages | Push to `main`, PR with doc changes, manual                                |
| **CodeQL Security**      | `codeql.yml`     | Security vulnerability scanning                       | Push/PR to `main` (only when code/workflow files change), scheduled weekly |
| **Docker Build & Test**  | `docker.yml`     | Build and test Docker images                          | Push/PR to `main` (only when Docker/Python files change)                   |
| **Snyk Security Scan**   | `snyk.yml`       | Dependency and Docker image vulnerability scanning    | Push/PR to `main`, scheduled weekly (Mondays), manual                      |

---

## Complete Pipeline Visualization

````mermaid
graph TB
    subgraph "Trigger Events"
        T1[Push to main]
        T2[Pull Request to main]
        T3[Schedule: Weekly Thu 13:17 UTC]
        T4[Manual Dispatch]
    end

    subgraph "Python Application Workflow"
        P1[Lint Job]
        P2[Test Job]
        P3[Docs Build Job]
        P4[Build Package Job]

        P1 -->|Format Check| P1A[Black/isort]
        P1 -->|Linting| P1B[flake8]
        P1 -->|Markdown| P1C[markdownlint]
        P1 -->|Type Check| P1D[mypy]
        P1 -->|Security| P1E[bandit/safety]

        P2 -->|Install ML deps| P2A[pytest + coverage]

        P3 -->|Install docs deps| P3A[mkdocs build]

        P4 -->|Install build tools| P4A[python -m build]
    end

    subgraph "Documentation Workflow"
        D1[Build Docs]
        D2[Deploy to Pages]

        D1 -->|mkdocs build| D1A[Generate Site]
        D1 -->|Upload| D1B[Pages Artifact]
        D1B -->|Only on push| D2
        D2 -->|Deploy| D2A[GitHub Pages]
    end

    subgraph "CodeQL Workflow"
    C1[CodeQL Matrix]
    C1 -->|Python Analysis| C1A[Scan Python Code]
    C1 -->|Actions Analysis| C1B[Scan GitHub Actions]
    end

    subgraph "Docker Workflow"
    DOCK1[Build Docker Image]
    DOCK2[Test Docker Image]
    DOCK3[Hadolint Validation]
    DOCK1 --> DOCK2
    DOCK1 --> DOCK3
    end

    subgraph "Snyk Workflow"
    SNYK1[Dependencies Scan]
    SNYK2[Docker Scan]
    SNYK3[Monitor]
    end

    T1 --> P1 & P2 & P3 & P4
    T2 --> P1 & P2 & P3 & P4

    T1 --> D1
    T2 --> D1
    T4 --> D1

    T1 --> C1
    T2 --> C1
    T3 --> C1

    T1 --> DOCK1
    T2 --> DOCK1

    T1 --> SNYK1 & SNYK2
    T2 --> SNYK1 & SNYK2
    T3 --> SNYK3

    style P1 fill:#e1f5e1
    style P2 fill:#e1f5e1
    style P3 fill:#e1f5e1
    style P4 fill:#e1f5e1
    style D1 fill:#e1e5ff
    style D2 fill:#e1e5ff
    style C1 fill:#ffe1e1
    style DOCK1 fill:#fff4e1
    style SNYK1 fill:#ffe1f5
```text

**File:** `.github/workflows/python-app.yml`
**Triggers:** Push and Pull Requests to `main` branch (only when relevant files change)

**Path Filters:**

- `**.py` - All Python source files
- `tests/**` - Test files
- `pyproject.toml` - Project configuration
- `Makefile` - Build configuration
- `Dockerfile`, `.dockerignore` - Docker files
- `.github/workflows/python-app.yml` - Workflow itself

**Skips when:** Only documentation, markdown, or non-code files change

This is the main CI pipeline that ensures code quality, runs tests, builds documentation, and validates the package.

### Two-Tier Testing Strategy

The workflow uses a **two-tier testing strategy** optimized for speed and coverage:

1. **Pull Requests:** Fast feedback + Full validation (both run in parallel)
2. **Push to Main:** Separate test jobs for maximum parallelization

### Pull Request Execution Flow

On pull requests, jobs run in parallel for fast feedback:

```mermaid
graph LR
    A[PR Opened/Updated] --> B[Lint Job]
    A --> C[test-fast Job<br/>Fast Feedback]
    A --> D[test Job<br/>Full Validation]
    A --> E[Docs Job]
    A --> F[Build Job]

    B --> G[✓ All Complete]
    C --> G
    D --> G
    E --> G
    F --> G

    style B fill:#90EE90
    style C fill:#FFE4B5
    style D fill:#87CEEB
    style E fill:#90EE90
    style F fill:#90EE90
```yaml

- **Fast Feedback (`test-fast`):** Completes in ~6-10 minutes, provides early pass/fail signal
- **Full Validation (`test`):** Completes in ~10-15 minutes, provides unified coverage report
- **Both run simultaneously:** No waiting, maximum speed
- **No redundancy:** Different test sets (fast vs full)

### Push to Main Execution Flow

On push to main branch, separate test jobs run in parallel:

```mermaid
graph LR
    A[Push to Main] --> B[Lint Job]
    A --> C[test-unit Job]
    A --> D[test-integration Job]
    A --> E[test-e2e Job<br/>All E2E Tests]
    A --> F[Docs Job]
    A --> G[Build Job]

    B --> H[✓ All Complete]
    C --> H
    D --> H
    E --> H
    F --> H
    G --> H

    style B fill:#90EE90
    style C fill:#90EE90
    style D fill:#90EE90
    style E fill:#FFE4B5
    style F fill:#90EE90
    style G fill:#90EE90
```yaml

- **Separate jobs:** Maximum parallelization for fastest overall completion
- **All tests run:** Includes slow integration and slow E2E tests
- **Complete validation:** Full test coverage before code is merged

### Job Details

#### 1. Lint Job (Always Runs)

**Purpose:** Perform quick code quality checks without heavy ML dependencies

**When:** Both PRs and push to main

**Duration:** ~1-2 minutes

**Steps:**

1. Checkout code
2. Set up Python 3.11 with pip caching
3. Set up Node.js 20 for markdown linting
4. Install dev dependencies (excluding ML packages)
5. Install markdownlint-cli
6. Run all lint checks:
   - `make format-check` - Black & isort formatting validation
   - `make lint` - flake8 code linting
   - `make lint-markdown` - Markdown file linting
   - `make type` - mypy static type checking
   - `make security` - bandit & pip-audit security scanning

**Why separate from test?** Linting is much faster without ML dependencies, providing quick feedback.

---

#### 2. test-fast Job (PRs Only - Fast Feedback)

**Purpose:** Provide quick feedback on fast tests without waiting for slow tests

**When:** Pull requests only

**Duration:** ~6-10 minutes

**What it runs:** `make test-ci-fast`
- **Unit tests:** All unit tests
- **Fast integration tests:** Excludes slow/ml_models integration tests
- **Fast E2E tests:** Excludes slow/ml_models E2E tests
- **No coverage:** Excluded for faster execution

**Key Features:**
- **Fast feedback:** Completes before full validation job
- **No coverage overhead:** Faster execution
- **Early signal:** Developer gets quick pass/fail status
- **Network guard:** Enabled for E2E tests

**Why separate?** Provides early feedback so developers can start fixing issues immediately if fast tests fail.

---

#### 3. test Job (PRs Only - Full Validation)

**Purpose:** Complete validation with unified coverage report

**When:** Pull requests only

**Duration:** ~10-15 minutes

**What it runs:** `make test-ci`
- **Unit tests:** All unit tests
- **Fast integration tests:** Excludes slow/ml_models integration tests (same as test-fast)
- **Fast E2E tests:** Excludes slow/ml_models E2E tests
- **With coverage:** Unified coverage report for all tests

**Key Features:**
- **Unified coverage:** Single coverage report for all tests
- **Full validation:** Ensures all fast tests pass
- **Network guard:** Enabled for E2E tests
- **Runs in parallel:** Starts at same time as test-fast, no waiting

**Test Coverage on PRs:**
- ✅ All unit tests
- ✅ Fast integration tests (excludes slow/ml_models)
- ✅ Fast E2E tests (excludes slow/ml_models)
- ❌ Slow integration tests (run on main branch only)
- ❌ Slow E2E tests (run on main branch only)

**Why exclude slow tests from PRs?** Faster PR feedback while maintaining full validation on main branch.

---

#### 4. test-unit Job (Main Branch Only)

**Purpose:** Run unit tests separately for maximum parallelization

**When:** Push to main branch only

**Duration:** ~2-5 minutes

**What it runs:** `make test-unit`
- **Unit tests:** All unit tests with coverage
- **Network isolation:** Enforced and verified
- **Import verification:** Ensures modules work without ML dependencies

**Key Features:**
- **No ML dependencies:** Fast execution
- **Coverage included:** Part of overall coverage
- **Network isolation:** Verified with dedicated test

---

#### 5. test-integration Job (Main Branch Only)

**Purpose:** Run all integration tests including slow/ml_models tests

**When:** Push to main branch only

**Duration:** ~5-10 minutes

**What it runs:** `make test-integration`
- **All integration tests:** Includes fast + slow/ml_models tests
- **Re-runs enabled:** 2 retries with 1s delay for flaky tests
- **Parallel execution:** Uses `-n auto` for speed

**Key Features:**
- **Complete coverage:** All integration tests run
- **Re-runs:** Handles flaky tests automatically
- **ML dependencies:** Required for slow tests

---

#### 6. test-e2e Job (Main Branch Only)

**Purpose:** Run all E2E tests including slow/ml_models tests

**When:** Push to main branch only

**Duration:** ~20-30 minutes

**What it runs:** `make test-e2e`
- **All E2E tests:** Includes fast + slow/ml_models tests
- **Re-runs enabled:** 2 retries with 1s delay for flaky tests
- **Network guard:** Blocks external network calls
- **Parallel execution:** Uses `-n auto` for speed

**Key Features:**
- **Complete validation:** All E2E tests including ML model tests
- **Network isolation:** Enforced via pytest-socket
- **Re-runs:** Handles flaky tests automatically
- **ML dependencies:** Required for slow tests

---

#### 7. Docs Build Job (Always Runs)

**Purpose:** Validate documentation builds correctly

**When:** Both PRs and push to main

**Duration:** ~2-3 minutes

**Steps:**

1. Checkout code
2. Set up Python 3.11 with pip caching (docs + pyproject.toml)
3. Install documentation dependencies + ML packages (needed for mkdocstrings API docs)
4. Build documentation: `make docs`
   - Runs `mkdocs build --strict`
   - Generates API documentation from docstrings
   - Validates all internal links

**Note:** This job runs in the main workflow to validate docs on every PR, even if not touching doc files.

---

#### 8. Build Package Job (Always Runs)

**Purpose:** Validate the package can be built for distribution

**When:** Both PRs and push to main

**Duration:** ~1-2 minutes

**Steps:**

1. Checkout code
2. Set up Python 3.11 with pip caching
3. Install build tools (`pip install build`)
4. Build package: `make build`
   - Creates source distribution (`.tar.gz`)
   - Creates wheel distribution (`.whl`)
   - Validates `pyproject.toml` configuration

### Fast Feedback vs Full Validation

On pull requests, two test jobs run simultaneously to provide both speed and completeness:

#### Fast Feedback (`test-fast` Job)

**Purpose:** Quick pass/fail signal without waiting for slow tests

**Timeline:** Completes in ~6-10 minutes

**What it tests:**
- All unit tests
- Fast integration tests (excludes slow/ml_models)
- Fast E2E tests (excludes slow/ml_models)

**Features:**
- No coverage overhead (faster execution)
- Early signal if fast tests fail
- Developer can start fixing immediately

**When to check:** After ~6-10 minutes for early feedback

#### Full Validation (`test` Job)

**Purpose:** Complete validation with unified coverage report

**Timeline:** Completes in ~10-15 minutes

**What it tests:**
- All unit tests
- Fast integration tests (excludes slow/ml_models)
- Fast E2E tests (excludes slow/ml_models)

**Features:**
- Unified coverage report for all tests
- Comprehensive validation
- Runs in parallel with test-fast (no waiting)

**When to check:** After ~10-15 minutes for final validation

#### Why Both?

1. **Fast feedback:** Developer gets early signal if something is broken
2. **Full validation:** Ensures all tests pass with coverage report
3. **No redundancy:** Both run simultaneously, no sequential waiting
4. **Different purposes:** Fast feedback for speed, full validation for completeness

#### What About Slow Tests?

Slow integration and slow E2E tests are excluded from PRs for faster feedback. They run on the main branch only:

- **Slow integration tests:** Run in `test-integration` job on main branch
- **Slow E2E tests:** Run in `test-e2e` job on main branch

This ensures:
- ✅ Fast PR feedback (no waiting for slow tests)
- ✅ Full validation on main branch (all tests run)
- ✅ Balance between speed and coverage

### Re-runs for Flaky Tests

Some test jobs include automatic re-runs to handle flaky tests:

**Configuration:**

- `--reruns 2`: Retry failed tests up to 2 times (3 total attempts)
- `--reruns-delay 1`: Wait 1 second between retries

**Which jobs use re-runs:**
- ✅ `test-integration` (main branch): All integration tests
- ✅ `test-e2e` (main branch): All E2E tests
- ❌ `test-fast` (PRs): No re-runs (fast tests should be stable)
- ❌ `test` (PRs): No re-runs (fast tests should be stable)
- ❌ `test-unit` (main branch): No re-runs (unit tests should be stable)

**How it works:**
1. Test runs and fails
2. Wait 1 second
3. Retry test (attempt 1)
4. If still fails, wait 1 second
5. Retry test (attempt 2)
6. If all attempts fail, mark as FAILED

**Why re-runs?**
- Integration and E2E tests can be flaky due to timing, I/O, or resource contention
- Re-runs reduce false negatives from transient failures
- Improves CI reliability without masking real issues

### Dependency Management

```mermaid
graph TD
    A[pyproject.toml] --> B{Job Type}

    B -->|Lint Job| C[dev dependencies only]
    C --> C1[black]
    C --> C2[flake8]
    C --> C3[mypy]
    C --> C4[bandit]

    B -->|test-fast Job PRs| E2[dev + ml + pytest-socket]
    E2 --> E2A[pytest]
    E2 --> E2B[pytest-socket for network guard]
    E2 --> E2C[ML deps for integration tests]

    B -->|test Job PRs| E4[dev + ml dependencies]
    E4 --> E4A[pytest]
    E4 --> E4B[ML deps for integration tests]
    E4 --> E4C[Coverage tools]

    B -->|test-unit Job Main| C2[dev dependencies only]
    C2 --> C2A[pytest]
    C2 --> C2B[No ML deps - fast]
    C2 --> C2C[Import check script]

    B -->|test-integration Job Main| D[dev + ml dependencies]
    D --> D1[pytest]
    D --> D2[transformers]
    D --> D3[torch]
    D --> D4[whisper]
    D --> D5[spacy]

    B -->|test-e2e Job Main| E3[dev + ml dependencies + pytest-socket]
    E3 --> E3A[pytest]
    E3 --> E3B[pytest-socket for network guard]
    E3 --> E3C[transformers]
    E3 --> E3D[torch]
    E3 --> E3E[whisper]
    E3 --> E3F[spacy]

    B -->|Docs Job| E[docs + ml dependencies]
    E --> E1[mkdocs-material]
    E --> E2D[mkdocstrings]
    E --> E3G[ML packages for API docs]

    B -->|Build Job| F[build tools only]
    F --> F1[python -m build]
```text

**File:** `.github/workflows/docs.yml`
**Triggers:**

- Push to `main` branch (when docs or related files change)
- Pull Requests modifying docs or related files
- Manual dispatch (`workflow_dispatch`)

**Path Filters:**

- `docs/**` - Documentation files
- `mkdocs.yml` - MkDocs configuration
- `**.py` - Python files (needed for API documentation)
- `README.md` - Main readme file
- `.github/workflows/docs.yml` - Workflow itself

### Sequential Pipeline (Build → Deploy)

Unlike the Python app workflow, this has a sequential dependency:

```mermaid
graph LR
    A[Trigger] --> B[Build Job]
    B -->|Generate Site| C[Upload Artifact]
    C -->|Only on push to main| D[Deploy Job]
    D -->|Deploy| E[GitHub Pages]

    style B fill:#87CEEB
    style D fill:#90EE90
```python

**Purpose:** Build MkDocs site from documentation sources

**Runs:** On all triggers (push, PR, manual)

**Steps:**

1. Check out repository
2. Set up Python 3.11 with pip caching
3. Install documentation dependencies
4. Build MkDocs site with strict mode (`mkdocs build --strict`)
5. Upload artifact (only on push to `main`)

**Output:** `.build/site/` directory containing the static website

#### 2. Deploy Job

**Purpose:** Deploy built site to GitHub Pages

**Runs:** Only on push to `main` (conditional)

**Depends on:** Build job must succeed

**Steps:**

1. Deploy to GitHub Pages using the uploaded artifact

**Environment:**

- Name: `github-pages`
- URL: Deployment URL exposed as output

### Concurrency Control

```yaml
concurrency:
  group: "pages"
  cancel-in-progress: true
```text

## CodeQL Security Workflow

**File:** `.github/workflows/codeql.yml`
**Triggers:**

- Push to `main` branch (only when code or workflow files change)
- Pull Requests to `main` branch (only when code or workflow files change)
- **Scheduled:** Every Thursday at 13:17 UTC (weekly security scan)

**Path Filters:**

- `**.py` - All Python source files
- `.github/workflows/**` - GitHub Actions workflow files

**Skips when:** Only documentation or non-code files change

### Matrix Strategy (Parallel Language Analysis)

CodeQL analyzes multiple languages in parallel using a matrix:

```mermaid
graph TB
    A[CodeQL Workflow] --> B{Matrix Strategy}

    B --> C[Python Analysis]
    B --> D[GitHub Actions Analysis]

    C --> C1[Initialize CodeQL]
    C --> C2[Analyze Python Code]
    C --> C3[Upload Results]

    D --> D1[Initialize CodeQL]
    D --> D2[Analyze Actions YAML]
    D --> D3[Upload Results]

    style C fill:#FFE4B5
    style D fill:#FFE4B5
```text

| Language | Build Mode | Purpose |
| -------- | ---------- | ------- |
| **Python** | `none` | Analyze application code for security vulnerabilities |
| **GitHub Actions** | `none` | Analyze workflow YAML files for security issues |

**Build Mode:** `none` means no compilation required (interpreted languages)

### CodeQL Job Details

**Per-language steps:**

1. Checkout repository
2. Initialize CodeQL tools
3. Scan codebase for security vulnerabilities
4. Upload results to GitHub Security tab

**Scan Categories:**

- SQL injection
- Cross-site scripting (XSS)
- Path traversal
- Code injection
- Hardcoded secrets
- Unsafe deserialization
- And more...

### Schedule Details

```yaml
schedule:

  - cron: '17 13 * * 4'
```text

**Runs:** Every Thursday at 13:17 UTC
**Purpose:** Catch newly discovered vulnerabilities in dependencies

---

## Docker Build & Test Workflow

**File:** `.github/workflows/docker.yml`
**Triggers:** Push and Pull Requests to `main` branch (only when Docker or Python files change)

**Path Filters:**

- `Dockerfile`, `.dockerignore` - Docker-related files
- `Dockerfile` - Main Dockerfile
- `pyproject.toml` - Project configuration
- `*.py` - Python source files

**Skips when:** Only documentation, markdown, or unrelated files change

### Workflow Purpose

Validates that Docker images can be built correctly and pass basic smoke tests. Ensures Docker configuration remains functional as code evolves.

### Docker Job Details

#### Docker Build & Test Job

**Purpose:** Build Docker images and run smoke tests

**Duration:** ~5-10 minutes

**Steps:**

1. Checkout code
2. Free disk space (removes unnecessary system packages)
3. Set up Docker Buildx
4. Build Docker image (default configuration)
   - Uses `Dockerfile`
   - Caches layers using GitHub Actions cache
   - Tags as `podcast-scraper:test`
5. Build Docker image (multiple Whisper models)
   - Preloads `tiny.en` and `base.en` Whisper models
   - Tags as `podcast-scraper:multi-model`
6. Test Docker images:
   - Test `--help` command
   - Test `--version` command
   - Test error handling (no args should show error)
7. Validate Dockerfile with hadolint
   - Lints Dockerfile for best practices
   - Catches common Docker anti-patterns

**Docker Image Tests:**

- ✅ Help command works
- ✅ Version command works
- ✅ Error handling works (no args shows error)
- ✅ Multiple model preloading works

**Dockerfile Validation:**

- ✅ Hadolint linting passes
- ✅ Best practices enforced

---

## Snyk Security Scan Workflow

**File:** `.github/workflows/snyk.yml`
**Triggers:**

- Push to `main` branch (only when code/Docker files change)
- Pull Requests to `main` branch (only when code/Docker files change)
- **Scheduled:** Every Monday at 00:00 UTC (weekly security scan)
- Manual dispatch (`workflow_dispatch`)

**Path Filters:**

- `**.py` - Python source files
- `pyproject.toml` - Project configuration
- `Dockerfile`, `.dockerignore` - Docker files
- `Dockerfile` - Main Dockerfile

**Skips when:** Only documentation or unrelated files change

### Snyk Workflow Purpose

Provides comprehensive security scanning for both Python dependencies and Docker images using Snyk's vulnerability database.

### Snyk Job Details

#### 1. Snyk Dependencies Scan

**Purpose:** Scan Python dependencies for known vulnerabilities

**Runs:** On all triggers (push, PR, schedule, manual)

**Steps:**

1. Checkout code
2. Set up Python 3.11 with pip caching
3. Install dependencies (`pip install -e .[dev,ml]`)
4. Run Snyk scan on Python dependencies
   - Scans installed packages from `pyproject.toml`
   - Severity threshold: `high` (only reports high/critical issues)
   - Generates SARIF file for GitHub Code Scanning integration
5. Upload results to GitHub Code Scanning
   - Results appear in Security tab
   - Integrated with GitHub's security dashboard

**Output:** SARIF file uploaded to GitHub Code Scanning

#### 2. Snyk Docker Scan

**Purpose:** Scan Docker image for vulnerabilities

**Runs:** On push/PR (not on schedule)

**Steps:**

1. Checkout code
2. Free disk space
3. Set up Docker Buildx
4. Build Docker image
   - Tags as `podcast-scraper:snyk-scan`
   - Uses GitHub Actions cache
5. Verify Docker image exists
6. Run Snyk scan on Docker image
   - Scans built image for vulnerabilities
   - Severity threshold: `high`
   - Generates SARIF file
7. Upload results to GitHub Code Scanning

**Output:** SARIF file uploaded to GitHub Code Scanning

#### 3. Snyk Monitor

**Purpose:** Monitor dependencies for ongoing security tracking

**Runs:** On PRs and scheduled runs (not on push)

**Steps:**

1. Checkout code
2. Set up Python 3.11 with pip caching
3. Install dependencies
4. Run Snyk monitor
   - Tracks dependencies in Snyk dashboard
   - Enables ongoing monitoring and alerts

**Purpose:** Provides long-term security monitoring and alerts for new vulnerabilities

### Snyk Schedule Details

```yaml
schedule:

  - cron: '0 0 * * 1'
```text

**Runs:** Every Monday at 00:00 UTC
**Purpose:** Weekly security scan to catch newly discovered vulnerabilities

### Integration with GitHub Security

- Results uploaded to GitHub Code Scanning
- Appears in Security tab
- Integrated with Dependabot alerts
- SARIF format for standardized reporting

---

## Parallel Execution Summary

### Workflow Independence

All workflows run independently and in parallel when triggered:

- **Python Application Workflow** (7 parallel jobs: lint, unit test, integration test, fast E2E test, docs, build, slow E2E test (main only))
- **Documentation Deployment Workflow** (sequential: build → deploy)
- **CodeQL Workflow** (matrix: Python + Actions analysis in parallel)
- **Docker Workflow** (single job: build → test → validate)
- **Snyk Workflow** (3 jobs: dependencies scan, Docker scan, monitor - run based on trigger type)

### Parallel Workflow Execution

Each workflow is independent and can run simultaneously:

- Python app workflow doesn't wait for Docker
- Docker workflow doesn't wait for Snyk
- CodeQL runs independently of other workflows
- Documentation workflow runs independently

This maximizes parallelism and reduces total CI time.

### Parallel Execution Details

#### ✅ Completely Parallel

**Within Python Application Workflow - Pull Requests:**

```text
├── Lint Job (1-2 min)
├── test-fast Job (6-10 min) - Fast feedback, no coverage
├── test Job (10-15 min) - Full validation with coverage
├── Docs Job (2-3 min)
└── Build Job (1-2 min)
```text
```text
├── Lint Job (1-2 min)
├── test-unit Job (2-5 min) - No ML deps, fast
├── test-integration Job (5-10 min) - With ML deps, includes re-runs
├── test-e2e Job (20-30 min) - With ML deps, includes re-runs, network guard
├── Docs Job (2-3 min)
└── Build Job (1-2 min)
```text
```text
├── Python Analysis
└── Actions Analysis
```text

- All three workflows (Python app, docs, CodeQL) trigger independently
- They run in parallel when triggered by the same event

#### ❌ Sequential

**Documentation Workflow:**

```text
Build Job → Deploy Job
```text

## Performance Optimizations

### 1. Pip Caching

All workflows use pip caching to speed up dependency installation:

```yaml
- uses: actions/setup-python@v5
  with:

    python-version: "3.11"
    cache: "pip"
    cache-dependency-path: pyproject.toml
```text
```mermaid
graph TD
    A[Dependency Strategy] --> B[Lint: dev only]
    A --> C[Test: dev + ml]
    A --> D[Docs: docs + ml]
    A --> E[Build: build tools only]

    B --> F[Fast: 2-3 min]
    C --> G[Slow: 10-15 min]
    D --> H[Medium: 3-5 min]
    E --> I[Fast: 2-3 min]
```text

Test job proactively frees ~30GB of disk space before installing ML dependencies:

```bash
sudo rm -rf /usr/share/dotnet
sudo rm -rf /usr/local/lib/android
sudo rm -rf /opt/ghc

# ... more cleanup

```text
```bash
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/torch
rm -rf ~/.cache/whisper
```text

## Workflow Triggers Matrix

| Workflow | Push to main | PR to main | Schedule | Manual | Doc Changes | Code Changes |
| -------- | ------------ | ---------- | -------- | ------ | ----------- | ------------ |
| **Python Application** | ✅ (code only) | ✅ (code only) | ❌ | ❌ | ❌ | ✅ |
| **Documentation Deploy** | ✅ (deploy) | ✅ (build only) | ❌ | ✅ | ✅ | ✅ (API docs) |
| **CodeQL Security** | ✅ (code only) | ✅ (code only) | ✅ Weekly | ❌ | ❌ | ✅ |

---

## Path-Based Optimization Strategy

### Strategy Overview

All workflows implement **intelligent path-based filtering** to ensure CI/CD runs only when necessary. This optimization dramatically reduces unnecessary CI runs, saves compute resources, and provides faster feedback.

### Decision Matrix

When you change files, here's what runs:

| Files Changed | Python App | Docs Deploy | CodeQL | Reasoning |
| ------------- | ---------- | ----------- | ------ | --------- |
| **Only `docs/`** | ❌ Skip | ✅ Run | ❌ Skip | Docs changes don't require code validation |
| **Only `.py` files** | ✅ Run | ✅ Run | ✅ Run | Code changes need full validation + API docs rebuild |
| **Only `README.md`** | ❌ Skip | ✅ Run | ❌ Skip | README is included in docs site |
| **`pyproject.toml`** | ✅ Run | ❌ Skip | ❌ Skip | Config changes affect dependencies/build |
| **`Dockerfile`** | ✅ Run | ❌ Skip | ❌ Skip | Docker builds depend on package validation |
| **`.github/workflows/`** | ✅ (if python-app.yml) | ✅ (if docs.yml) | ✅ Run | Workflow changes need validation |
| **Mixed changes** | ✅ Run | ✅ Run | ✅ Run | Any match triggers the workflow |

### Benefits

**Time Savings:**

- Docs-only change: ~18 minutes saved (only 3-5 min for docs vs. 20+ min for everything)
- README-only change: ~18 minutes saved
- Config-only change: ~5 minutes saved (skips docs and CodeQL)

**Resource Savings:**

- ~70% fewer runner minutes for documentation updates
- ~30GB less disk space operations per docs-only change
- Reduced ML dependency installations

**Developer Experience:**

- ✅ Faster feedback loop for documentation updates
- ✅ Clear separation: code changes = full CI, docs changes = docs only
- ✅ No wasted time waiting for unrelated checks

### Examples

#### Example 1: Documentation Update

```bash

# You change only: docs/api/REFERENCE.md

git commit -m "Update API documentation"
```text

- ✅ `docs.yml` runs (3-5 min)
- ❌ `python-app.yml` skipped
- ❌ `codeql.yml` skipped

**Total CI time:** ~3-5 minutes (vs. 20+ minutes before)

#### Example 2: Python Code Change

```bash

# You change: downloader.py

git commit -m "Fix download retry logic"
```text

- ✅ `python-app.yml` runs (lint, test, docs, build)
- ✅ `docs.yml` runs (API docs need rebuild)
- ✅ `codeql.yml` runs (security scan on code)

**Total CI time:** ~15-20 minutes (all workflows needed)

#### Example 3: Mixed Changes

```bash

# You change: docs/index.md AND service.py

git commit -m "Update docs and fix service"
```text

- ✅ All workflows run (code changed = full validation needed)

**Total CI time:** ~15-20 minutes (appropriate for code changes)

### Minimal Docs CI/CD Validation ✅

The system now passes the "minimal docs CI/CD" requirement:

**When changing ONLY documentation files:**

- ✅ Docs build and deploy (required)
- ❌ NO Python linting
- ❌ NO Python testing
- ❌ NO security scanning
- ❌ NO package building

**Status:** ✅ **VALIDATED - Optimization complete**

---

## CI/CD Evolution Highlights

### Key Improvements Over Time

1. **Path-Based Workflow Filtering** ⭐ NEW
   - Intelligent path filtering prevents unnecessary workflow runs
   - Docs-only changes skip Python testing, linting, and security scanning
   - Saves ~18 minutes per docs-only commit
   - ~70% reduction in runner minutes for documentation updates

2. **Parallel Job Execution**
   - Separated lint, unit tests, integration tests, docs, and build into independent parallel jobs
   - Reduced total CI time from ~20 minutes sequential to ~15 minutes parallel (limited by slowest job)
   - Unit tests run fast (2-3 min) without ML dependencies, integration tests run in parallel (10-15 min)

3. **Smart Dependency Management**
   - Lint job runs without ML dependencies for fast feedback (2-3 min)
   - Unit test job runs without ML dependencies for fast feedback (2-3 min)
   - Integration test job includes full ML stack for complete validation
   - Separate dependency groups in `pyproject.toml`: `[dev]`, `[ml]`, `[docs]`

4. **ML Dependency Import Verification** ⭐ NEW
   - Automatic check that unit tests can import modules without ML dependencies
   - Prevents modules from importing ML deps at top level (which would break CI)
   - Script: `scripts/check_unit_test_imports.py`
   - Runs before unit tests in CI, catches issues early
   - Local command: `make test-unit-no-ml` to verify locally

4. **Comprehensive Security Scanning**
   - CodeQL for static analysis (Python + Actions)
   - Snyk for dependency and Docker image scanning
   - Scheduled weekly scans for newly discovered vulnerabilities
   - Bandit & pip-audit in lint job for immediate feedback
   - Multiple layers of security validation

5. **Documentation as Code**
   - Docs build validated on every PR
   - Automatic deployment to GitHub Pages on merge
   - API documentation auto-generated from docstrings

7. **Resource Optimization**
   - Pip caching reduces dependency install time
   - Proactive disk space management
   - Post-test cache cleanup

8. **Developer Experience**
   - Fast lint feedback (~2-3 min)
   - Clear separation of concerns (lint vs test)
   - `make ci` command to run full CI suite locally

---

## Local Development

### Automatic Pre-commit Checks

**Prevent linting failures before they reach CI!**

Install the git pre-commit hook to automatically check your code before every commit:

```bash

# One-time setup

make install-hooks
```python

- ✅ **Black** formatting check
- ✅ **isort** import sorting check
- ✅ **flake8** linting
- ✅ **markdownlint** (if installed)
- ✅ **mypy** type checking

**If any check fails, the commit is blocked** until you fix the issues.

#### Skip Hook (Not Recommended)

```bash

# Skip pre-commit checks for a specific commit

git commit --no-verify -m "your message"
```bash

make format

# Then try committing again

git commit -m "your message"

```text

# Run full CI suite (matches GitHub Actions PR validation)

# - Runs unit + fast integration + fast e2e tests (excludes slow/ml_models)

# - Full validation before commits/PRs

# - Note: No cleanup step (faster), use ci-full for complete validation

make ci

# Fast CI checks (quick feedback during development)

# - Skips cleanup step (faster)

# - Runs unit + fast integration + fast e2e (excludes slow/ml_models, no coverage)

# - Use for quick validation during development

make ci-fast

# Complete CI suite (all tests including slow/ml_models)

# - Cleans cache first (clean-all)

# - Runs all tests: unit + integration + e2e (all slow/fast variants)

# - Use for complete validation before releases

make ci-full

# Individual checks (same as CI)

make format-check  # Black & isort
make lint          # flake8
make lint-markdown # markdownlint
make type          # mypy
make security      # bandit & pip-audit
make test-unit     # pytest with coverage (parallel, unit tests only)
make test-unit-sequential  # pytest sequentially (for debugging)
make test-integration      # All integration tests (parallel, with re-runs)
make test-e2e             # All E2E tests (parallel, with re-runs, network guard)
make docs          # mkdocs build
make build         # package build

```yaml

- **`make ci`**: Full validation before commits/PRs (unit + fast integration + fast e2e tests), matches GitHub Actions PR validation exactly
- **`make ci-fast`**: Quick feedback during development (unit + fast integration + fast e2e, no coverage), faster iteration
- **`make ci-full`**: Complete validation with all tests including slow/ml_models tests (unit + integration + e2e, all variants), use before releases

## Local CI Validation Flow

```mermaid

graph TD
    A[Local Development] --> B{git commit}

    B --> C[Pre-commit Hook]
    C --> C1[format-check]
    C --> C2[lint]
    C --> C3[lint-markdown]
    C --> C4[type]

    C1 & C2 & C3 & C4 --> D{Hook Pass?}
    D -->|No| E[Commit Blocked]
    E --> F[make format to fix]
    F --> A

    D -->|Yes| G[Commit Created]
    G --> H[git push]

    H --> I{make ci}
    I --> J[All CI Checks]
    J --> K{CI Pass?}
    K -->|Yes| L[PR Ready]
    K -->|No| M[Fix Issues]
    M --> A

    style L fill:#90EE90
    style E fill:#FFB6C6
    style M fill:#FFB6C6
    style G fill:#87CEEB

```text

### ✅ Prevention

- **Pre-commit hooks:** Catch issues before they're committed
- **Local CI validation:** `make ci` runs full suite before push
- **Auto-fix formatting:** `make format` fixes issues automatically

### ✅ Speed

- **Parallel execution:** All independent jobs run simultaneously
- **Caching:** Pip cache for faster dependency installation
- **Early feedback:** Fast lint job without ML dependencies

### ✅ Reliability

- **Reproducible:** Same checks run locally via `make ci`
- **Isolated:** Jobs don't depend on each other (except docs deploy)
- **Clean environment:** Each job starts fresh, post-cleanup prevents cache pollution

### ✅ Security

- **Multi-layered:** CodeQL + bandit + safety
- **Continuous:** Weekly scheduled scans
- **Early detection:** Security checks on every PR

### ✅ Documentation

- **Validated:** Docs build checked on every PR
- **Automated:** Deployment on merge to main
- **Complete:** Code + architecture + API reference

### ✅ Developer Experience

- **Fast feedback:** Lint results in 2-3 minutes
- **Local parity:** `make ci` runs same checks as GitHub
- **Quick iteration:** `make ci-fast` for rapid development feedback
- **Clear errors:** Strict mode for docs and type checking

---

## Monitoring & Debugging

### Viewing Workflow Results

1. **GitHub Actions Tab:** [View all runs](https://github.com/chipi/podcast_scraper/actions)
2. **PR Checks:** Status checks appear on pull requests
3. **Branch Protection:** Can require specific jobs to pass before merge

### Common Issues & Solutions

| Issue | Cause | Solution |
| ----- | ----- | -------- |
| Test timeout | Large ML models download | Already handled by disk space management |
| Lint failures | Formatting issues | Run `make format` locally before push |
| Docs build failure | Broken links or invalid syntax | Run `make docs` locally, check `mkdocs build` output |
| CodeQL alerts | Security vulnerabilities | Review in Security tab, address findings |
| Out of disk space | ML model caches | Cleanup is automatic, check disk usage logs |

### Debugging Failed Runs

```bash

# Reproduce lint failures locally

make format-check lint lint-markdown type security

# Reproduce test failures locally

make test

# Reproduce E2E test failures locally

make test-e2e  # All E2E tests
make test-e2e-fast      # Fast E2E tests only (excludes slow/ml_models)
make test-e2e-slow      # Slow E2E tests only (requires ML dependencies)

# Reproduce docs failures locally

make docs

# Run everything (matches full CI)

make ci

```bash

# Run all E2E tests (with network guard)

make test-e2e

# Run fast E2E tests only (excludes slow/ml_models, faster feedback)

make test-e2e-fast

# Run slow E2E tests only (includes slow/ml_models, requires ML dependencies)

make test-e2e-slow

```python
- All RSS and audio must be served from local E2E HTTP server
- Tests fail hard if a real URL is hit

**Test Markers:**

- `e2e`: All E2E tests
- `slow`: Slow tests (Whisper, ML models)
- `ml_models`: Tests requiring ML dependencies

See [Testing Guide](guides/TESTING_GUIDE.md#e2e-test-implementation) for detailed E2E test documentation.

---

## Future Enhancements

### Planned Enhancements

1. **AI Experiment Pipeline CI/CD** (See PRD-007, RFC-015)
   - Layer A: Fast smoke tests on every push/PR
   - Layer B: Full evaluation pipeline (nightly/on-demand)
   - Integration with experiment runner
   - Automated regression detection

2. **Environment Variable Management**
   - `.env` file support via `python-dotenv`
   - Environment-specific configurations
   - Secure API key management

### Potential Improvements

1. **Test Sharding**
   - Split test suite across multiple jobs for faster execution
   - Parallel test execution could reduce test time (if needed, but sequential is simpler)

2. **Artifact Caching**
   - Cache built wheels for dependencies
   - Cache ML models between runs

3. **Container-based Testing**
   - Run tests in Docker for better reproducibility
   - Pre-built images with ML dependencies

4. **Performance Benchmarking**
   - Track execution time trends
   - Automated performance regression detection

5. **Release Automation**
   - Automated version bumping
   - Automated changelog generation
   - PyPI publishing on tag creation

---

## Testing Path Filtering

After merging the path filtering optimization, validate it works correctly:

### Test 1: Documentation-Only Change

```bash

# Edit a docs file

echo "Test update" >> docs/CI_CD.md
git add docs/CI_CD.md
git commit -m "docs: test path filtering"
git push

```bash

# Edit a Python file

echo "# Test comment" >> downloader.py
git add downloader.py
git commit -m "feat: test python path filtering"
git push
```text
```bash

# Edit Dockerfile

echo "# Test comment" >> Dockerfile
git add Dockerfile
git commit -m "chore: test docker path filtering"
git push
```text
```bash

# Edit README

echo "Test update" >> README.md
git add README.md
git commit -m "docs: test readme path filtering"
git push
```text
```text

## Related Documentation

- **[Contributing Guide](https://github.com/chipi/podcast_scraper/blob/main/CONTRIBUTING.md)** - Development workflow and local testing
- **[Testing Strategy](TESTING_STRATEGY.md)** - Test coverage and quality standards
- **[Architecture](ARCHITECTURE.md)** - System design and module boundaries

---

## Quick Reference

### Workflow Files

```text

.github/workflows/
├── python-app.yml    # Main CI (lint, test, docs, build)
├── docs.yml          # Documentation deployment
└── codeql.yml        # Security scanning

```bash

# Individual checks

make format-check lint type security test docs build

# Auto-fix formatting

make format

```yaml

- [Documentation Deployment](https://github.com/chipi/podcast_scraper/actions/workflows/docs.yml)
- [CodeQL Security Scanning](https://github.com/chipi/podcast_scraper/actions/workflows/codeql.yml)

````
