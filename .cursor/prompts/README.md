# Cursor Prompt Templates

This directory contains reusable prompt templates for Cursor AI. These templates help ensure consistent, high-quality prompts and influence Cursor's Auto model selection.

## Available Templates

### `debug-ci.txt`

**Purpose:** Step-by-step CI failure analysis
**Use when:** CI fails, logs are long, cause unclear
**Effect:** Upgrades Auto to reasoning-heavy models (GPT-5.2/Opus 4.5)

### `design-rfc.md`

**Purpose:** Structured RFC and feature design
**Use when:** Starting features, refactors, or PRDs
**Effect:** Pushes Auto to Opus / GPT-5.2-level reasoning

### `code-review.txt`

**Purpose:** Systematic code review
**Use when:** Reviewing your own diff or as a second reviewer
**Effect:** Keeps reviews sharp without overengineering

### `implementation-plan.txt`

**Purpose:** Create implementation plans before coding
**Use when:** Changes span multiple files or systems
**Effect:** Auto upgrades model + prevents premature coding

## How to Use

1. **Open the prompt file** you need
2. **Copy the entire content**
3. **Paste into Cursor Chat, Composer, or Inline Chat**
4. **Add your specific context** (CI logs, code diff, feature description, etc.)
5. **Send the prompt** - Cursor will automatically select an appropriate model

## Quick Access Tips

### Option 1: Split View

- **Left pane:** `.cursor/prompts/` directory
- **Right pane:** Cursor Chat/Composer
- Copy → paste becomes muscle memory

### Option 2: VS Code / Cursor Snippets (Recommended)

Convert prompts into snippets for instant access:

- Type `ci-debug` → Enter (pastes `debug-ci.txt`)
- Type `rfc-design` → Enter (pastes `design-rfc.md`)
- Type `impl-plan` → Enter (pastes `implementation-plan.txt`)
- Type `code-review` → Enter (pastes `code-review.txt`)

## Related Documentation

- **Full guide:** [`docs/guides/CURSOR_AI_BEST_PRACTICES_GUIDE.md`](../../docs/guides/CURSOR_AI_BEST_PRACTICES_GUIDE.md) - Comprehensive Cursor AI usage guide
- **Model selection:** See Quick Reference section in CURSOR_AI_BEST_PRACTICES.md
- **Project guidelines:** [`.ai-coding-guidelines.md`](../../.ai-coding-guidelines.md) - Critical workflow rules

## Notes

- These are **manual templates** - Cursor does not automatically read them
- They are **tracked in git** - available to all contributors
- **Customize as needed** - adapt prompts to your specific use case
- **Share improvements** - if you enhance a prompt, consider updating the template
