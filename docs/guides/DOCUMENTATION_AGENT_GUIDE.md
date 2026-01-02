# Documentation Agent Guide

This guide extracts documentation-relevant practices from the Cursor AI Best Practices Guide for use
as a documentation agent.

## Project Documentation Context

The **podcast_scraper** project includes:

- **Documentation**: RFCs, PRDs, MkDocs site, markdown linting
- **Documentation structure**:
  - RFCs in `docs/rfc/`
  - PRDs in `docs/prd/`
  - Guides in `docs/guides/`
  - API docs in `docs/api/`
  - MkDocs configuration in `mkdocs.yml`

## Documentation Updates Workflow

### When Implementing Features

**Always update documentation when implementing features:**

- Update relevant RFCs (status: Draft → Accepted → Completed)
- Update `docs/guides/DEVELOPMENT_GUIDE.md` if adding new patterns
- Update `docs/TESTING_STRATEGY.md` if adding new test categories
- Update `mkdocs.yml` if adding new docs

**Example prompt:**

> "Implement this feature and update documentation: mark RFC-025 Phase 1 as Completed, add notes to DEVELOPMENT_GUIDE.md"

### Documentation Update Checklist

When working on documentation tasks:

1. **Check project guidelines** before starting:
   - Reference `docs/guides/DEVELOPMENT_GUIDE.md` for technical patterns
   - Check `docs/TESTING_STRATEGY.md` for test requirements
   - Review related RFCs/PRDs for feature context

2. **Update relevant files**:
   - RFC status updates
   - Development guide additions
   - Testing strategy updates
   - MkDocs configuration

3. **Follow documentation standards**:
   - Use markdown linting
   - Follow project markdown style guide
   - Ensure proper formatting

## RFC/PRD Creation Workflow

### Model Selection for Documentation

**RFC/PRD creation:**

- Use **GPT-5.2** or **Opus 4.5** (deep reasoning required)
- Use **Composer** for structured output

### Recommended Workflow

1. **Chat (no code selected)**
   - "Analyze RFC-023 and propose implementation plan. Do not write code yet."
   - "Review this GitHub issue and create a plan. Consider project guidelines."

2. **Composer (for RFCs/PRDs)**
   - Paste RFC template
   - Add context (feature, related RFCs)
   - Let Composer reason
   - Follow up with "Now implement step X"

3. **Project-specific use cases:**
   - **RFC creation**: Use Composer with RFC template
   - **Issue analysis**: Use Chat to analyze and plan
   - **Documentation updates**: Use Inline Chat with selected code

### Using the Design RFC Prompt Template

**Location:** `.cursor/prompts/design-rfc.md`

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

**How to use:**

1. Open `.cursor/prompts/design-rfc.md`
2. Copy the entire content
3. Paste into Cursor Chat, Composer, or Inline Chat
4. Add your specific context (feature description, related RFCs, etc.)
5. Send the prompt - Cursor will use appropriate model based on prompt structure

**Example workflow:**

```text
1. **Codex Max (Composer)**
   - Paste `design-rfc.md` template
   - "Implement Phase 1: Basic Metrics Collection from JUnit XML"

2. **Composer**
   - "Generate PR description with checklist based on RFC-025 Phase 1"
```

## Documentation-Specific Tasks

### Fast / Mechanical Documentation Work

**Use when**: Simple documentation updates.

**Examples:**

- Updating docstrings to match project standards
- Formatting documentation
- Fixing markdown linting errors
- Updating import statements in examples

**Typical models:**

- GPT-5.1 Codex Mini
- Sonnet 3.5
- Gemini 3 Flash

### Main Documentation Work

**Use for**: Most documentation tasks.

**Examples:**

- Writing new RFCs or PRDs
- Updating development guides
- Creating API documentation
- Writing test strategy documentation

**Typical models:**

- GPT-5.1 Codex Max
- Sonnet 4.5

### Deep Reasoning Documentation Work

**Use when**: Complex documentation decisions.

**Examples:**

- Architecture documentation decisions
- Designing new documentation structure
- Complex RFC/PRD creation
- Documentation strategy planning

**Typical models:**

- GPT-5.2
- Opus 4.5

## Key Documentation Principles

### General Principles

- **Always reference project guidelines** before starting documentation tasks
- **Update documentation** when implementing features
- **Follow RFC/PRD process** for new features
- **Use appropriate model** for documentation complexity

### Project-Specific Principles

- **Always reference `.ai-coding-guidelines.md`** before major documentation tasks
- **Update documentation** when implementing features
- **Follow RFC/PRD process** for new features
- **Use markdown linting** for consistency
- **Update `mkdocs.yml`** when adding new documentation

## Quick Reference: Documentation Prompts

### Creating New RFC

```text
Use Composer with design-rfc.md template:

1. Paste design-rfc.md template
2. Add feature context
3. Let Composer reason through design
4. Generate RFC following template
```

### Updating Documentation

```text
"Implement this feature and update documentation:

- Mark RFC-025 Phase 1 as Completed
- Add notes to DEVELOPMENT_GUIDE.md
- Update mkdocs.yml if needed"
```

### Documentation Review

```text
"Review this documentation update for:

- Consistency with project style
- Completeness
- Accuracy
- Proper formatting"
```

## Workflow Example: Documentation Update

**Example: Updating documentation after feature implementation**

1. **Codex Max (Inline Chat)**
   - Select relevant documentation files
   - "Update documentation to reflect new feature implementation"

2. **Sonnet 4.5 (Chat)**
   - "Review documentation updates for consistency and completeness"

3. **Codex Max (Inline Chat)**
   - "Apply review feedback and ensure markdown linting passes"

4. **Verify**
   - Run markdown linting
   - Check MkDocs build
   - Verify RFC status updates

## Next Steps for Documentation Agent

1. **Familiarize with documentation structure**:
   - Review `docs/` directory structure
   - Understand RFC/PRD templates
   - Check `mkdocs.yml` configuration

2. **Set up quick access**:
   - Create snippets for common documentation prompts
   - Type `rfc-design` → Enter (pastes `design-rfc.md`)
   - Type `doc-update` → Shows documentation update checklist

3. **Practice workflows**:
   - RFC creation (use `design-rfc.md`)
   - Documentation updates after features
   - Documentation reviews

4. **Follow project standards**:
   - Always check markdown style guide
   - Run markdown linting
   - Verify MkDocs builds correctly
