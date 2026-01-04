# Cursor AI Best Practices for podcast_scraper

## 🚀 Quick Reference: Model Selection

**Default Rule of Thumb:**

- **Auto** → fast iteration, small/local changes
- **Manual model selection** → planning, debugging, architecture, CI failures

> Auto optimizes for speed & cost, not deep reasoning.

### Model Roles (Quick Guide)

**Fast / Mechanical Tasks** (small, obvious changes):

- Rename functions, generate boilerplate/tests, minor refactors, formatting & docstrings
- **Models:** GPT-5.1 Codex Mini, Sonnet 3.5, Gemini 3 Flash

**Main Coding & Reviews** (daily driver):

- Multi-file changes, writing tests, medium refactors, code reviews
- **Models:** GPT-5.1 Codex Max (default), Sonnet 4.5

**Deep Reasoning / Hard Problems** (correctness matters):

- CI failures you can't reproduce, packaging/Docker/deps, architecture decisions,
  concurrency/async, edge cases, "Why is this happening?"

- **Models:** GPT-5.2, Opus 4.5

**High-Risk / Strict Correctness** (use sparingly):

- Security-sensitive code, release prep, complex refactors
- **Models:** GPT-5.1 Codex Max High

### How to Influence Cursor Auto

1. **Control Scope** (most important):
   - Small selection → fast model
   - Multiple files / logs → stronger model
   - CI logs + diff → Auto upgrades

2. **Use Reasoning Language:**
   - Words that push Auto upward: `analyze`, `root cause`, `tradeoffs`, `why`, `alternatives`
   - Words that keep it cheap: `rewrite`, `generate`, `convert`

3. **Ask for Options:**
   - "Give 2–3 possible fixes, explain tradeoffs, recommend one."

4. **Split Planning and Execution:**
   - Chat (no code selected): "Analyze and propose a plan. No code yet."
   - Inline edit (code selected): "Implement step 2 from the plan."

5. **Use Composer for Big Tasks:**
   - Composer sessions bias Auto toward stronger models and longer reasoning.

### Recommended Workflow (Python Project)

1. **GPT-5.2** → decide approach & edge cases
2. **Codex Max** → implement + tests
3. **Sonnet 4.5** → review diff
4. **Codex Max / High** → apply fixes
5. **Composer** → PR description & checklist

### One-Line Mental Model

> **Auto for hands, manual for brain.**

---

## Current Situation

> **See also:**
>
> - [Development Guide](DEVELOPMENT_GUIDE.md) - Implementation instructions
> - [Testing Guide](TESTING_GUIDE.md) - Test execution commands
> - [CI/CD](../CI_CD.md) - CI/CD pipeline details

This document captures best practices for using Cursor effectively in the **podcast_scraper**
project, which is a Python-heavy codebase with:

- **CI/CD**: GitHub Actions workflows, Docker builds, Snyk security scans
- **Testing**: Unit, integration, workflow E2E, and acceptance tests with parallel execution
- **Documentation**: RFCs, PRDs, MkDocs site, markdown linting
- **Complex workflows**: AI experiment pipeline, prompt management, multi-threaded processing
- **Strict guidelines**: `.ai-coding-guidelines.md` with mandatory commit/PR workflows

**Problem**: Without proper model selection and workflow patterns, Cursor can:

- Choose suboptimal models for complex tasks (CI debugging, architecture decisions)
- Miss project-specific patterns (mandatory commit approval, `make ci` before PR push)
- Generate code that doesn't follow project standards (module boundaries, testing requirements)
- Waste time on mechanical tasks that could be faster

## Mental Model

Cursor is best thought of as:

- **Auto** → speed-first, cost-aware assistant (good for mechanical work)
- **Manual model selection** → correctness, reasoning, and design (required for complex tasks)

> **Key Principle**: Auto optimizes for fast iteration. You must explicitly ask for depth.

## Model Selection Strategy

### Fast / Mechanical Work

**Use when**: Changes are local and obvious.

**Examples for this project:**

- Renaming functions/variables
- Generating boilerplate tests
- Small refactors (single file)
- Formatting or docstrings
- Updating imports
- Fixing simple linting errors

**Typical models:**

- GPT-5.1 Codex Mini
- Sonnet 3.5
- Gemini 3 Flash

**Project-specific use cases:**

- Running `make format` and applying fixes
- Adding type hints to simple functions
- Updating docstrings to match project standards
- Generating test fixtures

---

### Main Coding & Reviews (Daily Driver)

**Use for**: Most feature work in this project.

**Examples for this project:**

- Multi-file changes (workflow, transcription, summarization modules)
- Writing or updating tests (unit, integration, E2E)
- Medium refactors (module reorganization)
- Primary code reviews
- Implementing RFC features
- Adding new provider implementations

**Typical models:**

- GPT-5.1 Codex Max
- Sonnet 4.5

**Project-specific use cases:**

- Implementing new transcription providers
- Adding new summarization backends
- Creating new test categories (acceptance tests)
- Refactoring workflow pipeline
- Implementing RFC-019 E2E test improvements
- Adding metrics collection (RFC-025)

---

### Deep Reasoning / Hard Problems

**Use when**: Understanding and correctness matter.

**Examples for this project:**

- CI failures with unclear cause (GitHub Actions, Docker builds)
- Docker / dependency issues (ML dependencies, version conflicts)
- Architecture decisions (module boundaries, protocol design)
- Concurrency or async bugs (multi-threaded downloads, parallel processing)
- "Why is this happening?" (Whisper progress bar, test failures)
- Complex refactors (pipeline refactoring, protocol extensions)

**Typical models:**

- GPT-5.2
- Opus 4.5

**Project-specific use cases:**

- Debugging `make ci` failures
- Analyzing test timeout issues
- Understanding parallel test execution problems
- Designing new protocol extensions
- Resolving merge conflicts in complex refactors
- Investigating Whisper transcription issues
- Debugging Docker build failures

---

### High-Risk / Strict Correctness

**Use sparingly** for sensitive changes.

**Examples for this project:**

- Security-related code (secrets handling, API keys)
- Release prep (version bumps, changelog)
- Complex refactors (protocol changes, breaking API changes)
- Critical bug fixes (data loss, corruption)

**Typical models:**

- GPT-5.1 Codex Max High

**Project-specific use cases:**

- Modifying `.ai-coding-guidelines.md` (affects all AI workflows)
- Changing git workflow rules
- Security audit fixes
- Release preparation
- Breaking API changes

## Cursor Auto: How It Works (Practically)

**Auto behavior:**

- Chooses model based on **scope, context size, and task type**
- Prefers cheaper/faster models
- Rarely escalates to top reasoning models unless forced

**Auto tends to pick:**

- Small selection → fast model
- Medium changes → Codex Max / Sonnet
- Large context → Codex Max
- Text-only → Gemini Flash

**Auto almost never chooses GPT-5.2 or Opus 4.5 on its own.**

## How to Influence Auto (What Actually Works)

### 1. Control Scope (Most Important)

**For this project:**

- Select one function → fast model (good for simple fixes)
- Select multiple files → stronger model (good for multi-file refactors)
- Paste CI logs + diff → Auto upgrades (essential for debugging `make ci` failures)
- Select entire test file → stronger model (good for test refactoring)

**Example:**

````text
❌ Bad: "Fix the CI failure" (no context)
✅ Good: Select CI log output + paste into chat + "Analyze root cause of this CI failure"
```text

- analyze
- root cause
- tradeoffs
- why
- alternatives
- risks
- architecture
- design decision

**Words that keep it cheap:**

- rewrite
- generate
- convert
- format
- fix linting

**Project-specific examples:**

```text
❌ "Fix the test failure"
✅ "Analyze root cause of this test timeout and propose solutions with tradeoffs"

❌ "Update the workflow"
✅ "Review this workflow change for architecture implications and potential risks"
````

> "Give 2–3 viable approaches for implementing RFC-025 metrics collection, explain tradeoffs, recommend one."

**Project-specific:**

> "Analyze this CI failure and propose 2-3 solutions. Consider: test execution time, parallel vs sequential, and CI/CD impact."

### 4. Split Planning and Execution

**Recommended workflow for this project:**

1. **Chat (no code selected)**
   - "Analyze RFC-023 and propose implementation plan. Do not write code yet."
   - "Review this GitHub issue and create a plan. Consider project guidelines."

2. **Composer (for RFCs/PRDs)**
   - Paste RFC template
   - Add context (feature, related RFCs)
   - Let Composer reason
   - Follow up with "Now implement step X"

3. **Inline chat (code selected)**
   - "Implement step 2 of the plan"
   - "Apply the recommended solution"

**Project-specific use cases:**

- **RFC creation**: Use Composer with RFC template
- **Issue analysis**: Use Chat to analyze and plan
- **Implementation**: Use Inline Chat with selected code
- **Code review**: Use Chat with diff selected

### 5. Use Composer for Big Tasks

**Composer sessions:**

- Bias Auto toward stronger models
- Encourage structured reasoning
- Ideal for RFCs and multi-step features

**Project-specific use cases:**

- Creating new RFCs (RFC-024, RFC-025, RFC-026)
- Implementing complex features (E2E test improvements)
- Multi-file refactors (pipeline refactoring)
- Documentation updates (DEVELOPMENT_GUIDE.md expansion)

## Prompt Files in `.cursor/prompts/`

### Current Structure

The project has `.cursor/rules/ai-guidelines.mdc` which references `.ai-coding-guidelines.md`, and
**prompt files are manual templates** stored in `.cursor/prompts/`.

**Current structure:**

````text
.cursor/
├── rules/
│   ├── ai-guidelines.mdc (auto-loaded by Cursor)
│   └── markdown-style.mdc (auto-loaded by Cursor)
└── prompts/ (manual templates - copy/paste)
    ├── debug-ci.txt          ✅ Created (Issue #95)
    ├── design-rfc.md          ✅ Created (Issue #95)
    ├── code-review.txt       ✅ Created (Issue #95)
    └── implementation-plan.txt ✅ Created (Issue #95)
```text

- Cursor does **not** automatically read prompt files
- They are **manual prompt templates**, not configuration
- You copy & paste them into Chat / Composer / Inline Chat
- Think of them as: **Your personal prompt library**
- All prompt files are tracked in git and available to all contributors

### Available Prompt Templates

The following prompt templates are available in `.cursor/prompts/`:

#### `debug-ci.txt` - CI Failure Debugging

**Purpose:** Step-by-step analysis of CI failures with root cause identification.

**Content:** Analyzes CI failures systematically:
- Summarizes what failed (job, step, error)
- Lists 2-3 plausible root causes, ranked by likelihood
- Explains how to confirm or rule out each cause
- Recommends safest fix with minimal side effects
- Calls out hidden risks and follow-up checks

**Use when:** CI fails, logs are long, cause unclear
**Effect:** Auto almost always upgrades to a reasoning-heavy model (GPT-5.2/Opus 4.5)

**Location:** `.cursor/prompts/debug-ci.txt`

#### `design-rfc.md` - RFC and Feature Design

**Purpose:** Structured approach to designing RFCs, features, and refactors.

**Content:** Guides through design process:
- Restates problem in own words
- Defines explicit goals and non-goals
- Proposes 2-3 viable designs with tradeoffs (complexity, risk, testability)
- Recommends one design with justification
- Outlines concrete implementation plan
- Identifies risks and mitigation strategies
- Proposes test strategy

**Use when:** Starting features, refactors, or PRDs
**Effect:** Pushes Auto to Opus / GPT-5.2-level reasoning

**Location:** `.cursor/prompts/design-rfc.md`

#### `code-review.txt` - Code Review

**Purpose:** Systematic code review focusing on correctness, tests, and maintainability.

**Content:** Reviews code or diffs with focus on:
- Correctness and edge cases
- Missing or weak tests
- API and behavior consistency
- Readability and maintainability

**Output:** Bullet list of findings labeled as MUST / SHOULD / NICE-TO-HAVE

**Use when:** Reviewing your own diff or as a second reviewer
**Effect:** Keeps reviews sharp without overengineering

**Location:** `.cursor/prompts/code-review.txt`

#### `implementation-plan.txt` - Implementation Planning

**Purpose:** Create structured implementation plans before writing code.

**Content:** Checklist-style planning:
- Lists files/modules that will change
- Describes changes per file
- Identifies edge cases and failure modes
- Specifies tests to add or update
- Notes CI, Docker, or packaging implications

**Use when:** Changes span multiple files or systems
**Effect:** Auto upgrades model + prevents premature coding

**Location:** `.cursor/prompts/implementation-plan.txt`

### How to Use Prompt Templates

1. **Open the prompt file** in `.cursor/prompts/`
2. **Copy the entire content**
3. **Paste into Cursor Chat, Composer, or Inline Chat**
4. **Add your specific context** (CI logs, code diff, feature description, etc.)
5. **Send the prompt** - Cursor will use appropriate model based on prompt structure

```python
2. **Codex Max (Composer)**
   - Paste `design-rfc.md` template
   - "Implement Phase 1: Basic Metrics Collection from JUnit XML"

3. **Sonnet 4.5 (Inline Chat)**
   - Select generated code
   - "Review this implementation for correctness and project patterns"

4. **Codex Max (Inline Chat)**
   - "Apply review feedback and ensure tests pass"

5. **Composer**
   - "Generate PR description with checklist based on RFC-025 Phase 1"

### Workflow 2: Debugging CI Failure

**Example: `make ci` failure**

1. **GPT-5.2 (Chat)**
   - Paste CI logs + `git diff`
   - "Analyze root cause of this CI failure. Consider: test execution, parallel vs sequential, project guidelines"

2. **Codex Max (Inline Chat)**
   - Select failing test file
   - "Fix this test based on the analysis. Ensure it follows project patterns."

3. **Sonnet 4.5 (Chat)**
   - "Review the fix. Does it address root cause? Any edge cases?"

4. **Codex Max (Inline Chat)**
   - "Apply review feedback"

5. **Run `make ci` locally** (mandatory before PR push)

### Workflow 3: Creating New Test Category

**Example: Adding acceptance tests (RFC-023)**

1. **GPT-5.2 (Chat)**
   - "Analyze RFC-023 and propose test structure. Consider: existing test categories, fixtures, CI integration"

2. **Codex Max (Composer)**
   - "Implement acceptance test structure following project patterns"

3. **Sonnet 4.5 (Inline Chat)**
   - Select test files
   - "Review test structure and ensure it follows TESTING_STRATEGY.md"

4. **Codex Max (Inline Chat)**
   - "Add missing tests and ensure all pass"

### Workflow 4: Refactoring Module

**Example: Refactoring workflow pipeline**

1. **GPT-5.2 (Chat)**
   - "Analyze current workflow.py and propose refactoring plan. Consider: module boundaries, protocol design, backward compatibility"

2. **Codex Max (Composer)**
   - "Implement refactoring following the plan. Maintain backward compatibility."

3. **Sonnet 4.5 (Chat)**
   - "Review refactoring. Check: module boundaries, protocol compliance, test coverage"

4. **Codex Max (Inline Chat)**
   - "Apply review feedback and update tests"

## Project-Specific Recommendations

### 1. Always Reference Project Guidelines

**Before starting any task:**

- Read `.ai-coding-guidelines.md` (or acknowledge you've read it)
- Reference `docs/guides/DEVELOPMENT_GUIDE.md` for technical patterns
- Check `docs/TESTING_STRATEGY.md` for test requirements
- Review related RFCs/PRDs for feature context

**Example prompt:**

> "Implement RFC-025 Phase 1. Reference .ai-coding-guidelines.md for commit workflow and docs/guides/DEVELOPMENT_GUIDE.md for code patterns."

### 2. Mandatory Workflow Steps

**Before committing:**

1. Show `git status` (mandatory)
2. Show `git diff` (mandatory)
3. Wait for explicit approval (mandatory)
4. Get commit message from user (mandatory)

**Before pushing to PR:**

1. Run `make ci` locally (mandatory)
2. Ensure all checks pass (mandatory)
3. Fix any failures before pushing

**Example prompt for Cursor:**

> "Before committing, show git status and git diff. Wait for my approval. Do not commit automatically."

### 3. Test Execution Strategy

**For this project:**

- **Unit tests**: Sequential (faster for fast tests)
- **Integration tests**: Parallel (3.4x faster)
- **E2E tests**: Parallel (faster for slow tests)
- **CI**: Parallel by default

**When debugging tests:**

- Use `pytest tests/unit/ -n 0` for sequential execution (cleaner output)
- Use `make test-unit` for fast feedback
- Use `make test-integration` for integration-specific issues

**Example prompt:**

> "Debug this test failure. Use `pytest -n 0` for sequential execution if parallel issues suspected."

### 4. Documentation Updates

**When implementing features:**

- Update relevant RFCs (status: Draft → Accepted → Completed)
- Update `docs/guides/DEVELOPMENT_GUIDE.md` if adding new patterns
- Update `docs/TESTING_STRATEGY.md` if adding new test categories
- Update `mkdocs.yml` if adding new docs

**Example prompt:**

> "Implement this feature and update documentation: mark RFC-025 Phase 1 as Completed, add notes to DEVELOPMENT_GUIDE.md"

### 5. CI/CD Integration

**For CI-related tasks:**

- Always test locally with `make ci` before pushing
- Consider path-based optimization (`.github/workflows/python-app.yml`)
- Check Docker builds if Dockerfile changed
- Verify Snyk scans if dependencies changed

**Example prompt:**

> "Fix this CI failure. Test locally with make ci. Consider: path filters, Docker build, test execution time"

### 6. Model Selection for Common Tasks

**RFC/PRD creation:**

- Use **GPT-5.2** or **Opus 4.5** (deep reasoning required)
- Use **Composer** for structured output

**Test implementation:**

- Use **Codex Max** (daily driver)
- Use **Sonnet 4.5** for review

**CI debugging:**

- Use **GPT-5.2** (root cause analysis)
- Use **Codex Max** for fixes

**Simple refactors:**

- Use **Auto** or **Codex Mini** (mechanical work)

**Security-related:**

- Use **Codex Max High** (strict correctness)

## Making This Fast (Recommended)

### Option A: Split View

- **Left pane**: `.cursor/prompts/`
- **Right pane**: Chat / Composer

Copy → paste becomes muscle memory.

### Option B: VS Code / Cursor Snippets (Best UX)

Convert prompts into snippets:

- Type `ci-debug` → Enter (pastes `debug-ci.txt`)
- Type `rfc-design` → Enter (pastes `design-rfc.md`)
- Type `impl-plan` → Enter (pastes `implementation-plan.txt`)
- Type `code-review` → Enter (pastes `code-review.txt`)

This removes copy/paste entirely.

### Option C: Quick Access to Guidelines

**Create snippet for project guidelines:**

- Type `guidelines` → Shows key rules from `.ai-coding-guidelines.md`
- Type `test-strategy` → Shows test requirements
- Type `commit-workflow` → Shows mandatory commit steps

## Key Takeaways

### General Principles

- **Auto for hands, manual for brain**
- Auto is good, but conservative
- Prompt structure matters more than model picker
- Plan first, code second
- Use multiple models intentionally, not randomly

### Project-Specific Principles

- **Always reference `.ai-coding-guidelines.md`** before major tasks
- **Never commit without showing changes and getting approval**
- **Never push to PR without running `make ci` first**
- **Use appropriate test execution mode** (sequential vs parallel)
- **Update documentation** when implementing features
- **Follow RFC/PRD process** for new features

### One-Line Rule

> **Auto for hands, manual for brain. For this project: Always check guidelines first.**

## Next Steps

1. ✅ **Create prompt templates** in `.cursor/prompts/` (Completed - Issue #95):
   - ✅ `debug-ci.txt`
   - ✅ `design-rfc.md`
   - ✅ `code-review.txt`
   - ✅ `implementation-plan.txt`

2. **Set up snippets** in Cursor/VS Code for quick access:
   - Convert prompts into VS Code snippets for instant access
   - Type `ci-debug` → Enter (pastes `debug-ci.txt`)
   - Type `rfc-design` → Enter (pastes `design-rfc.md`)
   - Type `impl-plan` → Enter (pastes `implementation-plan.txt`)
   - Type `code-review` → Enter (pastes `code-review.txt`)

3. **Practice workflows** with common tasks:
   - RFC implementation (use `design-rfc.md`)
   - CI debugging (use `debug-ci.txt`)
   - Code reviews (use `code-review.txt`)
   - Multi-file changes (use `implementation-plan.txt`)

4. **Review and update** this guide based on experience
````
