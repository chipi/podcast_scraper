# Cursor AI Best Practices for podcast_scraper

## 🚀 Quick Reference: Model Selection

**Default Rule of Thumb:**

- **Auto** → fast iteration, small/local changes
- **Manual model selection** → planning, debugging, architecture, CI failures

> Auto optimizes for speed & cost, not deep reasoning.

## 🎯 Quick Reference: @Rules System

**Two separate rule systems:**

| System | How Loaded | When to Use |
| -------- | ----------- | ------------- |
| `.cursorrules` | **Automatic** (always active) | Core rules for every interaction |
| `.cursor/rules/*.mdc` | **Manual** via `@Rules filename` | Specialized context when needed |

**Load @Rules for specific tasks:**

```text
@Rules ai-guidelines        # Detailed commit/push checklists
@Rules markdown-style       # Markdown formatting guide
@Rules testing-strategy     # Testing patterns and pytest markers
@Rules git-worktree         # Worktree workflow commands
@Rules module-boundaries    # Architecture constraints
```

**Example:**

```text

# Starting to write tests

@Rules testing-strategy
Write unit tests for the new provider

# Creating worktree

@Rules git-worktree
Set up worktree for issue #200
```

## Model Roles (Quick Guide)

**Fast / Mechanical Tasks** (small, obvious changes):

- Rename functions, generate boilerplate/tests, minor refactors, formatting & docstrings
- **Models:** GPT-5.1 Codex Mini, Sonnet 3.5, Gemini 3 Flash
- **@Rules:** Usually none needed (.cursorrules handles it)

**Main Coding & Reviews** (daily driver):

- Multi-file changes, writing tests, medium refactors, code reviews
- **Models:** GPT-5.1 Codex Max (default), Sonnet 4.5
- **@Rules:** Load as needed for context (`@Rules testing-strategy`, etc.)

**Deep Reasoning / Hard Problems** (correctness matters):

- CI failures you can't reproduce, packaging/Docker/deps, architecture decisions,
  concurrency/async, edge cases, "Why is this happening?"

- **Models:** GPT-5.2, Opus 4.5
- **@Rules:** Always load relevant context (`@Rules module-boundaries`, etc.)

**High-Risk / Strict Correctness** (use sparingly):

- Security-sensitive code, release prep, complex refactors
- **Models:** GPT-5.1 Codex Max High
- **@Rules:** `@Rules ai-guidelines` for critical workflows

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

6. **Load @Rules for Context:**
   - `@Rules testing-strategy` for test-related tasks
   - `@Rules module-boundaries` for architecture work
   - `@Rules git-worktree` for worktree management

### One-Line Mental Model

> **Auto for hands, manual for brain. Load @Rules for specialized context.**

---

## Current Situation

> **See also:**
>
> - [Development Guide](DEVELOPMENT_GUIDE.md) - Implementation instructions
> - [Testing Guide](TESTING_GUIDE.md) - Test execution commands
> - [CI/CD](../ci/index.md) - CI/CD pipeline details
> - [Git Worktree Guide](GIT_WORKTREE_GUIDE.md) - Worktree workflow
> - [Markdown Linting Guide](MARKDOWN_LINTING_GUIDE.md) - Markdown standards

This document captures best practices for using Cursor effectively in the **podcast_scraper**
project, which is a Python-heavy codebase with:

- **AI Rules System**: `.cursorrules` (always active) + `.cursor/rules/*.mdc` (on-demand via `@Rules`)
- **CI/CD**: GitHub Actions workflows, Docker builds, Snyk security scans
- **Testing**: Unit, integration, E2E tests with two-tier strategy (fast/slow)
- **Documentation**: RFCs, PRDs, MkDocs site, markdown linting
- **Git Workflow**: Worktree-based parallel development, never push to main
- **Strict guidelines**: Mandatory commit approval, `make ci-fast` before commit

**Problem**: Without proper model selection and @Rules usage, Cursor can:

- Choose suboptimal models for complex tasks (CI debugging, architecture decisions)
- Miss project-specific patterns (mandatory commit approval, module boundaries)
- Generate code that doesn't follow project standards (testing requirements, code style)
- Waste time on mechanical tasks that could be faster
- Lack specialized context for complex workflows (testing, worktrees, architecture)

**Solution**: Layered context system:

1. **`.cursorrules`** (287 lines) - Always active, enforces core rules
2. **`.cursor/rules/*.mdc`** (200-400 lines each) - Load with `@Rules` for specialized tasks
3. **Model selection** - Choose appropriate model for task complexity
4. **Prompt templates** - `.cursor/prompts/` for structured prompts

## Mental Model

Cursor is best thought of as:

- **Auto** → speed-first, cost-aware assistant (good for mechanical work)
- **Manual model selection** → correctness, reasoning, and design (required for complex tasks)
- **@Rules system** → contextual knowledge loader (activate specialized guides when needed)

> **Key Principle**: Auto optimizes for fast iteration. You must explicitly ask for depth
> and load specialized context with @Rules.

## Recommended Workflows with @Rules

### Workflow 1: Implementing New Feature (RFC-Based)

**Scenario:** Implementing RFC-025 metrics collection

```text
Step 1: Design phase
Chat: "@Rules ai-guidelines
       Read RFC-025 and analyze implementation approach"

Step 2: Architecture check
Chat: "@Rules module-boundaries
       Review proposed changes for module boundary violations"

Step 3: Implementation
Composer: [Paste design-rfc.md template]
          "Implement Phase 1: Basic Metrics Collection"

Step 4: Testing
Inline Chat: "@Rules testing-strategy
             Write unit and integration tests for metrics"

Step 5: Documentation
Inline Chat: "@Rules markdown-style
             Update DEVELOPMENT_GUIDE.md with metrics usage"
```

### Workflow 2: Debugging CI Failure

**Scenario:** `make ci` failure

```text
Step 1: Root cause analysis
Chat: [Paste CI logs + git diff]
      "@Rules testing-strategy
       Analyze root cause of this CI failure"

Step 2: Fix implementation
Inline Chat: [Select failing test]
            "Fix based on analysis, following project patterns"

Step 3: Verify commit
Terminal: make ci-fast

Step 4: Commit
Chat: "@Rules ai-guidelines
       Show git status and git diff, wait for approval"
```

### Workflow 3: Creating New Worktree

**Scenario:** Start work on issue #200

```text
Step 1: Setup worktree
Chat: "@Rules git-worktree
       Set up worktree for issue #200 (add Dependabot)"

Step 2: Verify setup
Terminal: cd ../podcast_scraper-200-dependabot
          source .venv/bin/activate
          cursor .

Step 3: Daily work
[Work in isolated worktree with clean AI context]

Step 4: Cleanup after merge
Chat: "@Rules git-worktree
       Remove worktree and cleanup references"
```

### Workflow 4: Refactoring Module

**Scenario:** Refactoring workflow pipeline

```text
Step 1: Architecture review
Chat: "@Rules module-boundaries
       Analyze current workflow.py for boundary violations"

Step 2: Design refactor
Chat: [Paste design-rfc.md template]
      "Propose refactoring plan maintaining module boundaries"

Step 3: Implement
Composer: "@Rules module-boundaries
          Implement refactoring following the plan"

Step 4: Test coverage
Inline Chat: "@Rules testing-strategy
             Ensure test coverage for refactored code"
```

### Workflow 5: Writing Documentation

**Scenario:** Update README and guides

```text
Step 1: Content changes
Inline Chat: [Select markdown file]
            "Update with new feature information"

Step 2: Formatting
Chat: "@Rules markdown-style
       Review markdown for style issues"

Step 3: Validation
Terminal: make fix-md
          make lint-markdown
          make docs

Step 4: Commit
Chat: "@Rules ai-guidelines
       Show changes and wait for approval"
```

### Pro Tips for @Rules Usage

**1. Multiple @Rules at once**

```text
You can load multiple rules in one prompt:

@Rules testing-strategy @Rules module-boundaries
Write tests for the new module following architecture constraints
```

**2. Rules persist in conversation**

```text
Once loaded, @Rules stay active in that conversation:

First message:
@Rules testing-strategy
Help me write tests

Later messages:
Add more test cases
[testing-strategy.mdc still loaded]
```

**3. New chat = reset**

```text
Start new chat → Must load @Rules again
Previous @Rules don't carry over
```

**4. Start without @Rules**

```text
For simple tasks, .cursorrules is usually enough:

You: "Fix this typo in config.py"
[.cursorrules handles this fine]

You: "@Rules testing-strategy
     Write comprehensive test suite"
[Now you need the detailed testing guide]
```

### Recommended Workflow (Python Project)

1. **GPT-5.2** → decide approach & edge cases
2. **Codex Max** → implement + tests
3. **Sonnet 4.5** → review diff
4. **Codex Max / High** → apply fixes
5. **Composer** → PR description & checklist

### Summary: One-Line Mental Model

> **Auto for hands, manual for brain.**

---

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

## Understanding Cursor's Rule Systems

### Two Separate Rule Systems

Cursor has **two different systems** for loading context:

#### 1. `.cursorrules` (Root File) - AUTOMATIC ✅

**Location:** `/.cursorrules` (project root)

**How it works:**

- ✅ **Loaded automatically** at session start
- ✅ **Always active** for every AI interaction
- ✅ **You do nothing** - Cursor reads it automatically
- ✅ **Priority:** Highest - always in context

**What it contains (287 lines):**

- Core git workflow rules (never push to main)
- Testing strategy (`make ci-fast` before commit)
- Code quality standards (imports, type hints, docstrings)
- Module boundaries (architecture constraints)
- Documentation requirements

**Think of it as:** Your project's "constitution" - always in effect.

#### 2. `.cursor/rules/*.mdc` Files - MANUAL ON-DEMAND 📋

**Location:** `/.cursor/rules/` folder

**How it works:**

- ❌ **NOT loaded automatically** - Cursor ignores them by default
- ✅ **Loaded manually** when you mention them with `@Rules`
- ✅ **You control when** - Load specific context when needed
- ✅ **Scope:** Current conversation only

**How to load:**

In Cursor Chat, type `@Rules` followed by filename (without `.mdc`):

```text
@Rules ai-guidelines        # Loads detailed commit/push checklists
@Rules markdown-style       # Loads markdown formatting guide
@Rules testing-strategy     # Loads testing guide with examples
@Rules git-worktree         # Loads worktree workflow
@Rules module-boundaries    # Loads architecture constraints
```

**What happens:**

```text
You type: @Rules testing-strategy
         ↓
Cursor loads: .cursor/rules/testing-strategy.mdc
         ↓
AI now has: .cursorrules (always) + testing-strategy.mdc (just added)
```

**Available `.mdc` files:**

- `ai-guidelines.mdc` (289 lines) - Detailed commit/push checklists
- `markdown-style.mdc` (128 lines) - Markdown formatting with examples
- `testing-strategy.mdc` (254 lines) - Testing guide with pytest markers
- `git-worktree.mdc` (309 lines) - Worktree workflow and commands
- `module-boundaries.mdc` (238 lines) - Architecture constraints

**Think of these as:** "Power-ups" you activate for specific tasks.

### When to Use Each

#### Use `.cursorrules` (Always Active)

✅ Rules that apply to **every** interaction
✅ Critical workflows (git, commits, pushes)
✅ Code standards that never change
✅ Architecture rules that must always be enforced

#### Use `@Rules *.mdc` (On-Demand)

✅ **Specific workflows** you're about to perform
✅ **Detailed guides** for complex tasks
✅ **Context that's only sometimes relevant**

**Examples:**

```text

# About to write tests

@Rules testing-strategy
Write unit tests for the new provider

# About to create worktree

@Rules git-worktree
Set up worktree for issue #200

# About to edit documentation

@Rules markdown-style
Update the README with new features

# About to refactor modules

@Rules module-boundaries
Refactor the workflow pipeline
```yaml

## Comparison Table

| Feature | `.cursorrules` | `.cursor/rules/*.mdc` |
| --------- | ---------------- | ---------------------- |
| **Who loads it?** | Cursor (automatic) | You (manual via `@Rules`) |
| **When?** | Every session start | Only when you mention it |
| **Scope** | All interactions | Current conversation only |
| **Control** | Always on | You decide when to use |
| **Size** | 287 lines (optimized) | 200-400 lines each (detailed) |
| **Purpose** | Core project rules | Specialized workflows |

### Real Examples

#### Example 1: Normal Coding (No @Rules)

```text
You: "Add a new function to downloader.py"

AI has access to:
✅ .cursorrules (automatic)
❌ testing-strategy.mdc (not loaded)
❌ git-worktree.mdc (not loaded)

AI uses .cursorrules to:
- Follow module boundaries
- Use correct import order
- Add type hints
```

#### Example 2: Writing Tests (With @Rules)

```text
You: "@Rules testing-strategy
     Help me write tests for the new downloader function"

AI has access to:
✅ .cursorrules (automatic)
✅ testing-strategy.mdc (you just loaded it)

AI uses BOTH to:
- Follow .cursorrules (always)
- Use testing-strategy.mdc for pytest markers, mock patterns, etc.
```

#### Example 3: Creating Worktree (With @Rules)

```text
You: "@Rules git-worktree
     I need to create a new worktree for issue #200"

AI has access to:
✅ .cursorrules (automatic)
✅ git-worktree.mdc (you just loaded it)

AI provides:
- Specific worktree commands
- Branch naming with issue number
- Complete setup workflow
```

### Why Have Both Systems?

**Problem if everything was in `.cursorrules`:**

- File would be 2,000+ lines
- Loads into EVERY conversation
- Wastes context on irrelevant info
- Slower session starts

**Solution with layered approach:**

- Core rules always active (287 lines)
- Specialized rules on-demand (load when needed)
- Efficient context usage

### Current Project Structure

```text
.cursorrules                    # Always loaded (287 lines)
.cursor/
├── rules/                      # Load manually with @Rules
│   ├── ai-guidelines.mdc       # Commit/push checklists (289 lines)
│   ├── markdown-style.mdc      # Markdown guide (128 lines)
│   ├── testing-strategy.mdc    # Testing guide (254 lines)
│   ├── git-worktree.mdc        # Worktree workflow (309 lines)
│   └── module-boundaries.mdc   # Architecture (238 lines)
└── prompts/                    # Copy/paste templates
    ├── debug-ci.txt
    ├── design-rfc.md
    ├── code-review.txt
    └── implementation-plan.txt
```python

## Prompt Files in `.cursor/prompts/`

### Manual Templates (Copy/Paste)

**Prompt files are different from rules files:**

- Rules (`.cursorrules` and `.mdc`) = Context that Cursor loads
- Prompts (`.txt` and `.md`) = Templates you copy/paste

**Current structure:**

```text
.cursor/prompts/
├── debug-ci.txt          ✅ Created (Issue #95)
├── design-rfc.md          ✅ Created (Issue #95)
├── code-review.txt       ✅ Created (Issue #95)
└── implementation-plan.txt ✅ Created (Issue #95)
```

**How to use:**

1. Open the prompt file in `.cursor/prompts/`
2. Copy the entire content
3. Paste into Cursor Chat, Composer, or Inline Chat
4. Add your specific context (CI logs, code diff, etc.)
5. Send the prompt

**These are NOT loaded by Cursor** - you manually copy/paste them.

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

## Subagents and Commands

### What Are Subagents?

**Subagents** are separate AI agents the main Cursor agent can **delegate** to. Each has its own context; long or noisy work (e.g. full CI, acceptance tests) runs there and the main chat gets a **summary**. Rule 9 (no background make in main chat) applies to the **main** agent; a subagent running `make ci` in its own context returns a short result and does not violate that. All subagents use the **project venv only**: run from project root and use Makefile (which uses `.venv/bin/python`); never use global `python`/`pip`/`pytest`/`npx`.

| Aspect | Meaning |
|--------|--------|
| **Own context** | Subagent's logs and output stay in its context; main chat stays clean. |
| **Foreground / background** | Foreground = main agent waits for result. Background = you can keep working while subagent runs. |
| **Built-in** | Explore (codebase search), Bash (shell commands), Browser (MCP). Cursor uses these when appropriate. |
| **Custom** | `.cursor/agents/*.md` (project) or `~/.cursor/agents/` (user) with YAML frontmatter and prompt. |

### Commands vs Subagents

| | **Commands** (`/name`) | **Subagents** |
|---|-------------------------|----------------|
| **What** | You type `/verify` etc.; Cursor injects the command's Markdown as the prompt. **Main** agent does the work in the **same** chat. | Main agent **delegates** to another agent; that agent has its **own** context and returns a summary. |
| **Where** | `.cursor/commands/*.md` or `~/.cursor/commands/` | Built-in (Explore, Bash, Browser) or custom `.cursor/agents/*.md` |
| **Best for** | Repeatable start of a workflow (verify, review, pr, debug-ci, rfc) in one conversation. | Long/noisy runs (full CI, acceptance tests), deep search, parallel work, dedicated verifier. |

- **Use a command** when you want the main agent to do the work in this chat (e.g. `/review`, `/pr`).
- **Use a subagent** when you want context isolation or a short summary (e.g. "run ci and tell me the result," "run acceptance tests for planet money").

### Project Custom Subagents (`.cursor/agents/`)

This project defines three custom subagents. All run from **project root** and use the **project venv only** (Makefile uses `.venv` when present).

| Subagent | File | When to use |
|----------|------|--------------|
| **Verifier** | `verifier.md` | "Verify before commit," "run ci and report," "is the tree green?" Runs format-check, lint, lint-markdown, full **make ci**; optionally docker-test. Returns PASSED/FAILED + short summary. |
| **CI Fix Loop** | `ci-fix-loop.md` | "Run ci and fix until it passes," "make ci green." Runs full **make ci**; on failure fixes and re-runs up to 3 times. Returns final status + what was fixed. |
| **Acceptance** | `acceptance.md` | "Run acceptance tests," "run all acceptance configs." Runs `make test-acceptance CONFIGS="config/acceptance/*.yaml"` (or user pattern). Returns Status, Session ID, Summary; suggests `make analyze-acceptance SESSION_ID=…`. |

Optional later: **Docs check** (lint-markdown + docs), **PR prep** (status/diff + docker-test). Use Cursor's **built-in Explore** for "find all X in codebase."

### Subagents vs Skills

- **Subagents** = separate context and/or parallel work (verify, run CI and fix, run acceptance).
- **Skills** = procedures the main agent follows in the same context (commit-with-approval, push-to-pr, efficient pytest). Use skills for single-shot, repeatable procedures; use subagents when you want isolation or summarization.

## Project-Specific Recommendations

### 1. Always Use @Rules for Specialized Tasks

**Load appropriate context when needed:**

| Task | @Rules to Load | Why |
| ------ | ---------------- | ----- |
| Writing tests | `@Rules testing-strategy` | Pytest markers, mock patterns, coverage |
| Creating worktree | `@Rules git-worktree` | Setup commands, branch naming, cleanup |
| Refactoring modules | `@Rules module-boundaries` | Architecture constraints, SRP |
| Editing markdown | `@Rules markdown-style` | Formatting rules, auto-fix commands |
| Committing changes | `@Rules ai-guidelines` | Detailed commit/push checklists |

**Example:**

```text

❌ Bad: "Write tests for downloader.py"
✅ Good: "@Rules testing-strategy
        Write tests for downloader.py with appropriate pytest markers"

❌ Bad: "Update README.md"
✅ Good: "@Rules markdown-style
        Update README.md following project markdown standards"

```

### 2. Core Rules Are Always Active

**You don't need to mention these** - `.cursorrules` handles them:

✅ Git workflow (never push to main)
✅ Basic code standards (imports, type hints)
✅ Testing requirement (`make ci-fast` before commit)
✅ Module boundaries (high-level)
✅ Documentation requirements

**Example:**

```text

You: "Add a new function to config.py"

AI automatically:
- Follows module boundaries (.cursorrules)
- Adds type hints (.cursorrules)
- Uses correct import order (.cursorrules)
[No @Rules needed for basic standards]

```python

### 3. Mandatory Workflow Steps (Enforced by .cursorrules)

**Before committing:**

1. Run `make ci-fast` (pre-commit hook will also run this)
2. Show `git status` (mandatory)
3. Show `git diff` (mandatory)
4. Wait for explicit approval (mandatory)
5. Get commit message from user (mandatory)

**Before pushing to PR:**

1. Pre-commit hook passed during commit
2. Show `git status` (mandatory)
3. Show `git diff` or summary (mandatory)
4. Conditionally run `make docker-test` (if Docker-related changes)
5. Wait for explicit approval (mandatory)

**Example prompt:**

```text

@Rules ai-guidelines
Commit these changes following the mandatory workflow

```

### 4. Test Execution Strategy (Use @Rules testing-strategy)

**For this project:**

```bash

# Default: Fast tests first

make ci-fast  # ~6-10 min (unit + fast integration + fast e2e)

# If ML code changed: Add slow tests

make test-integration-slow  # Whisper, summarization, speaker detection
make test-e2e-slow         # Full E2E with ML models

# Before final PR: Full validation

make ci  # ~10-15 min (includes coverage)

```text
I changed whisper_integration.py. What tests should I run?

AI response:

1. make ci-fast (always)
2. make test-integration-slow (because ML code changed)
3. Verify specific test: pytest tests/integration/test_whisper_integration.py -v
```

## 5. Documentation Updates (Use @Rules markdown-style)

**When implementing features:**

- Update relevant RFCs (Draft → Accepted → Completed)
- Update guides if adding new patterns
- Update `mkdocs.yml` if adding new docs
- Run `make fix-md` before committing

**Example with @Rules:**

```text

@Rules markdown-style
Update README.md with new provider feature and run markdown validation

```

### 6. Git Worktree Workflow (Use @Rules git-worktree)

**For parallel development:**

```text

@Rules git-worktree
Create worktree for issue #200 with proper branch naming

AI provides:
- make wt-setup command
- Branch naming: feat/200-description
- Isolated venv setup
- Cursor instance management

```

- One branch = One worktree = One Cursor window
- Clean AI context per task
- No branch switching
- Parallel development

### 7. Model Selection for Common Tasks

**With @Rules integration:**

| Task | Model | @Rules |
| ------ | ------- | -------- |
| RFC/PRD creation | GPT-5.2, Opus 4.5 | `@Rules module-boundaries` |
| Test implementation | Codex Max | `@Rules testing-strategy` |
| CI debugging | GPT-5.2 | `@Rules testing-strategy` |
| Module refactoring | GPT-5.2 | `@Rules module-boundaries` |
| Documentation | Codex Max | `@Rules markdown-style` |
| Worktree setup | Auto or Codex Max | `@Rules git-worktree` |
| Commit/push | Auto | `@Rules ai-guidelines` |

### 8. Always Reference Project Guidelines

**Core documentation to mention:**

```text

# For architecture decisions

@Rules module-boundaries
Consider ARCHITECTURE.md for this refactoring

# For testing

@Rules testing-strategy
Follow TESTING_GUIDE.md patterns

# For CI/CD

Check docs/ci/index.md for CI pipeline details

# For markdown

@Rules markdown-style
Follow MARKDOWN_LINTING_GUIDE.md standards

```

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

### @Rules System Principles

- **`.cursorrules` is always active** - Core project rules loaded automatically
- **`.mdc` files are on-demand** - Load with `@Rules filename` when needed
- **Start without @Rules** - .cursorrules handles most cases
- **Add @Rules for specialized tasks** - Testing, worktrees, markdown, architecture
- **Multiple @Rules allowed** - `@Rules testing-strategy @Rules module-boundaries`
- **New chat = reset** - Must reload @Rules in new conversations

### Project-Specific Principles

- **Always reference `.cursorrules`** - Automatically enforces critical rules
- **Use @Rules for specialized context** - Load detailed guides when needed
- **Never commit without showing changes and getting approval** - Enforced by .cursorrules
- **Always run `make ci-fast` before committing** - Pre-commit hook also runs this
- **Use appropriate test execution mode** - Fast tests → slow tests (if ML changed) → full CI
- **Update documentation** when implementing features
- **Follow RFC/PRD process** for new features
- **Use worktrees for parallel development** - `@Rules git-worktree` for setup

### @Rules Quick Reference

| When You're... | Load This | Why |
| ---------------- | ----------- | ----- |
| Writing tests | `@Rules testing-strategy` | Pytest markers, mock patterns |
| Creating worktree | `@Rules git-worktree` | Setup commands, branch naming |
| Refactoring modules | `@Rules module-boundaries` | Architecture constraints |
| Editing markdown | `@Rules markdown-style` | Formatting, auto-fix commands |
| Committing changes | `@Rules ai-guidelines` | Detailed commit checklist |
| Regular coding | (none needed) | .cursorrules handles it |

### One-Line Rule

> **Auto for hands, manual for brain. For this project: .cursorrules always on, @Rules when specialized.**

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
