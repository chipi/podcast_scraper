# CI/CD Path Filtering - Validation Test Cases

This document provides comprehensive test cases to validate that the path-based filtering works correctly for all change scenarios.

---

## Test Case 1: Documentation-Only Change ✅

### Scenario
Edit a file in the `docs/` directory only.

### Example Files
- `docs/api/API_REFERENCE.md`
- `docs/ARCHITECTURE.md`
- `docs/CI_CD.md`

### Expected Workflow Behavior

| Workflow | Should Run? | Reason |
|----------|-------------|--------|
| **python-app.yml** | ❌ NO | No Python/config files changed |
| **docs.yml** | ✅ YES | Matches `docs/**` pattern |
| **codeql.yml** | ❌ NO | No Python/workflow files changed |

### Expected CI Time
- **Before optimization:** ~20 minutes (all workflows)
- **After optimization:** ~3-5 minutes (docs only)
- **Savings:** ~18 minutes (73% reduction)

### Validation Command
```bash
# Make a docs change
echo "Test update" >> docs/CI_CD.md
git add docs/CI_CD.md
git commit -m "docs: test path filtering"
git push

# Check GitHub Actions - should see only docs.yml running
```

---

## Test Case 2: Python Code Change ✅

### Scenario
Edit any Python source file.

### Example Files
- `downloader.py`
- `workflow.py`
- `cli.py`
- `episode_processor.py`

### Expected Workflow Behavior

| Workflow | Should Run? | Reason |
|----------|-------------|--------|
| **python-app.yml** | ✅ YES | Matches `**.py` pattern |
| **docs.yml** | ✅ YES | Matches `**.py` pattern (needed for API docs) |
| **codeql.yml** | ✅ YES | Matches `**.py` pattern (security scanning) |

### Expected CI Time
- **All workflows run:** ~15-20 minutes
- **This is correct:** Full validation needed for code changes

### Path Filter Match Details

#### python-app.yml triggers because:
- ✅ `**.py` matches any Python file
- Runs: lint, test, docs, build jobs

#### docs.yml triggers because:
- ✅ `**.py` matches any Python file
- Reason: API documentation needs to be regenerated from Python docstrings

#### codeql.yml triggers because:
- ✅ `**.py` matches any Python file
- Reason: Security scanning needed for code changes

### Validation Command
```bash
# Make a Python change
echo "# Test comment" >> downloader.py
git add downloader.py
git commit -m "feat: test path filtering"
git push

# Check GitHub Actions - should see ALL 3 workflows running
```

---

## Test Case 3: Docker File Change ✅

### Scenario
Edit Docker-related files.

### Example Files
- `docker/Dockerfile`
- `docker/docker-compose.yml` (if added)

### Expected Workflow Behavior

| Workflow | Should Run? | Reason |
|----------|-------------|--------|
| **python-app.yml** | ✅ YES | Matches `docker/**` pattern |
| **docs.yml** | ❌ NO | Docker changes don't affect docs |
| **codeql.yml** | ❌ NO | Docker files don't need Python security scan |

### Expected CI Time
- **python-app.yml runs:** ~15 minutes
- **This is correct:** Need to validate package builds correctly for Docker

### Why python-app.yml Should Run

Docker builds depend on:
1. **Package build** (validated by build job)
2. **Requirements** (tested by test job)
3. **Python code** (validated by lint/test jobs)

If Docker build breaks but we don't catch it, production deployments fail!

### Path Filter Match Details

#### python-app.yml triggers because:
- ✅ `docker/**` matches Docker files
- Runs all jobs to ensure package is valid for Docker build

#### docs.yml skips because:
- ❌ No matching paths (Docker changes don't affect documentation)

#### codeql.yml skips because:
- ❌ No matching paths (Docker files aren't Python code or workflows)

### Validation Command
```bash
# Make a Docker change
echo "# Test comment" >> docker/Dockerfile
git add docker/Dockerfile
git commit -m "chore: test Docker path filtering"
git push

# Check GitHub Actions - should see only python-app.yml running
```

---

## Test Case 4: README Change ✅

### Scenario
Edit the main README file.

### Example Files
- `README.md`

### Expected Workflow Behavior

| Workflow | Should Run? | Reason |
|----------|-------------|--------|
| **python-app.yml** | ❌ NO | README not in Python path filters |
| **docs.yml** | ✅ YES | Matches `README.md` pattern |
| **codeql.yml** | ❌ NO | README not in security scan paths |

### Expected CI Time
- **docs.yml runs:** ~3-5 minutes
- **Savings:** ~18 minutes (73% reduction)

### Validation Command
```bash
# Make a README change
echo "Test update" >> README.md
git add README.md
git commit -m "docs: test README path filtering"
git push

# Check GitHub Actions - should see only docs.yml running
```

---

## Test Case 5: Configuration File Change ✅

### Scenario
Edit project configuration files.

### Example Files
- `pyproject.toml`
- `requirements.txt`
- `Makefile`

### Expected Workflow Behavior

| Workflow | Should Run? | Reason |
|----------|-------------|--------|
| **python-app.yml** | ✅ YES | Config files affect Python builds |
| **docs.yml** | ❌ NO | Config changes don't affect docs |
| **codeql.yml** | ❌ NO | Config files don't need security scan |

### Expected CI Time
- **python-app.yml runs:** ~15 minutes
- **This is correct:** Config changes affect dependencies and builds

### Path Filter Match Details

#### python-app.yml triggers because:
- ✅ `pyproject.toml` - Project metadata and dependencies
- ✅ `requirements.txt` - Python dependencies
- ✅ `Makefile` - Build commands and CI targets

### Validation Command
```bash
# Make a config change
echo "# Test comment" >> pyproject.toml
git add pyproject.toml
git commit -m "chore: test config path filtering"
git push

# Check GitHub Actions - should see only python-app.yml running
```

---

## Test Case 6: Test File Change ✅

### Scenario
Edit test files only.

### Example Files
- `tests/test_downloader.py`
- `tests/test_cli.py`
- `tests/conftest.py`

### Expected Workflow Behavior

| Workflow | Should Run? | Reason |
|----------|-------------|--------|
| **python-app.yml** | ✅ YES | Test files need validation |
| **docs.yml** | ✅ YES | Test files are Python files (API docs) |
| **codeql.yml** | ✅ YES | Test files are Python code (security) |

### Expected CI Time
- **All workflows run:** ~15-20 minutes
- **This is correct:** Test changes need full validation

### Path Filter Match Details

#### All workflows trigger because:
- ✅ `**.py` matches test files
- ✅ `tests/**` explicitly included in python-app.yml

### Validation Command
```bash
# Make a test change
echo "# Test comment" >> tests/test_downloader.py
git add tests/test_downloader.py
git commit -m "test: add validation"
git push

# Check GitHub Actions - should see ALL 3 workflows running
```

---

## Test Case 7: Workflow File Change ✅

### Scenario
Edit GitHub Actions workflow files.

### Example Files
- `.github/workflows/python-app.yml`
- `.github/workflows/docs.yml`
- `.github/workflows/codeql.yml`

### Expected Workflow Behavior

| Workflow | Should Run? | Reason |
|----------|-------------|--------|
| **python-app.yml** | ✅ YES (if self) | Workflow includes itself in paths |
| **docs.yml** | ✅ YES (if self) | Workflow includes itself in paths |
| **codeql.yml** | ✅ YES | Matches `.github/workflows/**` pattern |

### Expected CI Time
- **Varies:** Depends on which workflow file changed

### Path Filter Match Details

Each workflow includes itself:
- `python-app.yml` → `.github/workflows/python-app.yml` in its paths
- `docs.yml` → `.github/workflows/docs.yml` in its paths
- `codeql.yml` → `.github/workflows/**` matches all workflows

### Validation Command
```bash
# Make a workflow change
echo "# Test comment" >> .github/workflows/python-app.yml
git add .github/workflows/python-app.yml
git commit -m "ci: test workflow path filtering"
git push

# Check GitHub Actions - should see python-app.yml and codeql.yml running
```

---

## Test Case 8: Mixed Changes ✅

### Scenario
Edit files across multiple categories in one commit.

### Example
- `docs/ARCHITECTURE.md` (docs)
- `downloader.py` (Python code)

### Expected Workflow Behavior

| Workflow | Should Run? | Reason |
|----------|-------------|--------|
| **python-app.yml** | ✅ YES | Python file changed |
| **docs.yml** | ✅ YES | Both docs and Python changed |
| **codeql.yml** | ✅ YES | Python file changed |

### Expected CI Time
- **All workflows run:** ~15-20 minutes
- **This is correct:** Code changed = full validation needed

### Logic
Path filters use **OR logic** - if ANY path matches, the workflow runs.
- Docs file matches docs.yml
- Python file matches ALL workflows
- Result: All workflows run

### Validation Command
```bash
# Make mixed changes
echo "Test" >> docs/ARCHITECTURE.md
echo "# Test" >> downloader.py
git add docs/ARCHITECTURE.md downloader.py
git commit -m "feat: mixed changes"
git push

# Check GitHub Actions - should see ALL 3 workflows running
```

---

## Summary Matrix

| Change Type | python-app.yml | docs.yml | codeql.yml | CI Time |
|------------|----------------|----------|------------|---------|
| **Docs only** | ❌ Skip | ✅ Run | ❌ Skip | 3-5 min |
| **Python code** | ✅ Run | ✅ Run | ✅ Run | 15-20 min |
| **Docker files** | ✅ Run | ❌ Skip | ❌ Skip | 15 min |
| **README only** | ❌ Skip | ✅ Run | ❌ Skip | 3-5 min |
| **Config files** | ✅ Run | ❌ Skip | ❌ Skip | 15 min |
| **Test files** | ✅ Run | ✅ Run | ✅ Run | 15-20 min |
| **Workflow files** | ✅ Self | ✅ Self | ✅ All | Varies |
| **Mixed changes** | ✅ Run | ✅ Run | ✅ Run | 15-20 min |

---

## Validation Checklist

Use this checklist to validate the optimization is working:

- [ ] **Test 1 - Docs only:** Edit `docs/CI_CD.md` → Only docs.yml runs
- [ ] **Test 2 - Python code:** Edit `downloader.py` → All workflows run
- [ ] **Test 3 - Docker:** Edit `docker/Dockerfile` → Only python-app.yml runs
- [ ] **Test 4 - README:** Edit `README.md` → Only docs.yml runs
- [ ] **Test 5 - Config:** Edit `pyproject.toml` → Only python-app.yml runs
- [ ] **Test 6 - Tests:** Edit `tests/test_cli.py` → All workflows run
- [ ] **Test 7 - Workflows:** Edit `.github/workflows/python-app.yml` → python-app.yml + codeql.yml run
- [ ] **Test 8 - Mixed:** Edit docs + Python → All workflows run

---

## How to Run Validation

### Option 1: Automated Test (Recommended)

Create a test script to validate all scenarios:

```bash
#!/bin/bash
# test_ci_paths.sh

echo "Testing CI/CD path filtering..."

# Test 1: Docs only
git checkout -b test/ci-paths-docs
echo "Test" >> docs/CI_CD.md
git add docs/CI_CD.md
git commit -m "test: docs path filtering"
git push -u origin test/ci-paths-docs
echo "✓ Test 1 pushed - check GitHub Actions"

# Test 2: Python code
git checkout -b test/ci-paths-python
echo "# Test" >> downloader.py
git add downloader.py
git commit -m "test: python path filtering"
git push -u origin test/ci-paths-python
echo "✓ Test 2 pushed - check GitHub Actions"

# Test 3: Docker
git checkout -b test/ci-paths-docker
echo "# Test" >> docker/Dockerfile
git add docker/Dockerfile
git commit -m "test: docker path filtering"
git push -u origin test/ci-paths-docker
echo "✓ Test 3 pushed - check GitHub Actions"

echo "All tests pushed. Check GitHub Actions to validate workflows."
```

### Option 2: Manual Testing

For each test case:
1. Create a new branch
2. Make the specified change
3. Commit and push
4. Check GitHub Actions tab
5. Verify expected workflows run/skip
6. Delete test branch after validation

---

## Expected Results After Optimization

### Time Savings by Change Type

```
Documentation updates: 73% faster (18 min → 3-5 min)
README updates:        73% faster (18 min → 3-5 min)
Config updates:        25-50% faster (depends on jobs)
Python/Test changes:   No change (full validation needed)
```

### Resource Savings

```
Docs-only commits:     ~70% fewer runner minutes
README commits:        ~70% fewer runner minutes
Config commits:        ~30-50% fewer runner minutes
```

### Developer Experience

- ✅ **Fast feedback** for documentation work
- ✅ **Full validation** for code changes
- ✅ **Smart routing** based on what actually changed
- ✅ **No manual workflow selection** needed

---

## Troubleshooting

### Issue: Workflow doesn't run when expected

**Check:**
1. Path patterns use `**` for recursive matching
2. File path matches exactly (case-sensitive)
3. Changes are committed and pushed
4. Branch protection rules allow workflow runs

### Issue: Workflow runs when it shouldn't

**Check:**
1. Mixed changes in commit (use `git show --stat`)
2. Path patterns too broad (e.g., `*` instead of `**.py`)
3. Base branch merge included other changes

### Issue: Validation unclear

**Solution:**
- Use `gh pr checks` to see which workflows ran
- Check workflow run logs for trigger reason
- Compare with path filter patterns in workflow files

---

## References

- [GitHub Actions path filtering docs](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#onpushpull_requestpull_request_targetpathspaths-ignore)
- Project workflows: `.github/workflows/`
- CI/CD documentation: `docs/CI_CD.md`
