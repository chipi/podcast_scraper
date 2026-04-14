# Markdown Style and Linting Guide

This guide covers markdown style rules, linting practices, tools, and workflows for the podcast scraper
project. It includes style guidelines, automated fixing, table formatting solutions, pre-commit hooks,
and CI/CD integration.

**For Cursor Users:** Pin this file in Cursor's context (Rules/Context features) so it's always
available when writing markdown. This helps Cursor follow these rules automatically.

For general development practices, see [Development Guide](DEVELOPMENT_GUIDE.md).

## Quick Reference: Style Rules

### Structure

- **One H1 per document** - Only the title should be `# Heading`
- **Blank line after headings** - Always add a blank line after `#`, `##`, etc.
- **80–100 char wrap** - Prefer wrapping at 100 characters (matches code style)

### Lists

- **Use `-` for unordered lists** - Consistent marker throughout
- **Blank lines around lists** - Add blank line before and after list blocks
- **Consistent indentation** - Use 2 spaces for nested lists

### Code

- **Code fences always specify language** - Use ` ```python`, ` ```bash`, ` ```yaml`, etc.
- **No bare code blocks** - Never use ` ``` ` without a language
- **Inline code with backticks** - Use `` `code` `` not `'code'`

### Formatting

- **No trailing spaces** - Trim whitespace at end of lines
- **No multiple blank lines** - Maximum one blank line between sections
- **Tables: avoid unless necessary** - Prefer lists when possible
- **Table formatting** - Use spaces around pipes: `| Column |` (compact style)

### Documentation prose (no checkmark emoji)

In all markdown under `docs/` (including PRDs, RFCs, ADRs, guides, API docs, and `docs/wip/`):

- **Do not** use checkmark-style emoji, cross-style emoji, clipboard emoji, or a decorative tick
  character as list bullets, table cells, or status prefixes.
- **Prefer** plain words: Yes / No, Run / Skip, Done, Good / Bad, Allowed / Forbidden, or normal
  bullets without emoji.
- **Fictional CLI samples** in docs: use a neutral marker such as `[ok]` instead of a tick character.

Rationale: the project dropped checkmark styling in published docs; keep new edits consistent.
Agents: see `.cursor/rules/documentation.mdc` (load when editing markdown per project rules).

### Links

- **Descriptive link text** - Use `[Link Text](url)` not `[url](url)`
- **Relative paths for internal links** - Use `rfc/RFC-019.md` not `docs/rfc/RFC-019.md`

### Paragraphs

- **Prefer short paragraphs** - 2–5 lines maximum
- **One idea per paragraph** - Keep paragraphs focused

## Quick Checklist

Before finishing any markdown file, verify:

- [ ] Only one H1 (`#`) in the document
- [ ] Blank line after all headings
- [ ] No trailing spaces
- [ ] Lists have blank lines before and after
- [ ] Code fences have language specifiers
- [ ] Tables use consistent spacing around pipes
- [ ] No multiple consecutive blank lines

## Rules Configuration

**Important:** The project uses `.markdownlint.json` with `"default": true`, which means
**ALL default markdownlint rules are enabled** unless explicitly disabled.

### Explicitly Enabled Rules

These rules are explicitly configured in `.markdownlint.json`:

- **MD013** - Line length (120 chars, with exceptions for code blocks, tables, headings)
- **MD022** - Blanks around headings
- **MD024** - No duplicate headings
- **MD029** - Ordered list item prefix
- **MD031** - Blanks around fenced code blocks
- **MD032** - Blanks around lists
- **MD040** - Fenced code language
- **MD041** - First line heading
- **MD047** - Single trailing newline

### Explicitly Disabled Rules

These rules are explicitly disabled:

- **MD025** - Multiple top-level headings (disabled - allows multiple H1s)
- **MD033** - HTML allowed (disabled - allows HTML in markdown)
- **MD036** - Emphasis used for heading (disabled - allows bold/italic as headings)
- **MD051** - Link fragments (disabled - can produce false positives)

### Default Rules (All Enabled)

Because `"default": true`, **all other default markdownlint rules are also enabled**. This includes rules like:

- **MD001** - Heading levels should only increment by one level at a time
- **MD003** - Heading style
- **MD004** - Unordered list style
- **MD005** - Inconsistent indentation for list items
- **MD007** - Unordered list indentation
- **MD009** - Trailing spaces
- **MD010** - Hard tabs
- **MD012** - Multiple consecutive blank lines
- **MD014** - Dollar signs used before commands without showing output
- **MD018** - No space after hash on atx style heading
- **MD019** - Multiple spaces after hash on atx style heading
- **MD020** - No space inside hashes on closed atx style heading
- **MD021** - Multiple spaces inside hashes on closed atx style heading
- **MD023** - Headings must start at the beginning of the line
- **MD026** - Trailing punctuation in heading
- **MD027** - Multiple spaces after blockquote symbol
- **MD028** - Blank line inside blockquote
- **MD030** - Spaces after list markers
- **MD034** - Bare URL used
- **MD035** - Horizontal rule style
- **MD037** - Spaces inside emphasis markers
- **MD038** - Spaces inside code span elements
- **MD039** - Spaces inside link text
- **MD042** - No empty links
- **MD043** - Required heading structure
- **MD044** - Proper names should have the correct capitalization
- **MD045** - Images should have alternate text (alt text)
- **MD046** - Code block style
- **MD048** - Code fence style
- **MD049** - Emphasis style should be consistent
- **MD050** - Strong style should be consistent
- **MD052** - Reference links should be used
- **MD053** - Link and image reference definitions should be needed

**Note:** The pre-commit hook and CI check **all of these rules**, not just the ones
explicitly listed above. If you see an error from a rule not mentioned in this guide,
it's likely from the default rule set.

## Automated Markdown Fixing

**Recommended (same rules as CI):** `make fix-md` runs `markdownlint --fix` with the same file globs, `--ignore` paths, and `.markdownlint.json` as `make lint-markdown`. That avoids “fix one way, CI fails another” drift.

```bash
make fix-md
make lint-markdown
```

**Optional helper:** `scripts/tools/fix_markdown.py` applies extra heuristics (e.g. some table separator tweaks) that may help beyond markdownlint; it does not use the exact same path filters as `lint-markdown`, so prefer `make fix-md` first for consistency.

```bash
# Fix all markdown files (custom walker / heuristics)

python scripts/tools/fix_markdown.py

# Fix specific files

python scripts/tools/fix_markdown.py TESTING_STRATEGY.md rfc/RFC-020.md

# Dry run (see what would be fixed)

python scripts/tools/fix_markdown.py --dry-run
```

**What markdownlint typically auto-fixes:** depends on rule support in your installed `markdownlint-cli` (for example spacing/blank lines where the rule supports `--fix`).

**Important:** Pre-commit and CI still enforce **all** enabled rules from `.markdownlint.json`. Anything `markdownlint --fix` cannot correct must be fixed manually.

**When to use:**

- Before committing markdown files
- After bulk documentation edits
- When CI fails on markdown linting errors

See `scripts/README.md` for full documentation.

## Format-on-Save (Recommended)

**For format-on-save:**

1. Install Prettier extension in Cursor/VSCode
2. `.vscode/settings.json` is already configured for format-on-save on markdown files
3. `.prettierrc` provides formatting rules (100 char wrap, preserves prose)

**Format on save will handle:**

- Wrapping at 100 characters
- Spacing and list formatting
- Trailing whitespace removal

**markdownlint handles:**

- Heading structure (one H1, blank lines after headings)
- List indentation
- Code fence language specifiers
- Table formatting

## Catching Table Formatting Issues Locally

To catch markdown table formatting issues (MD060) before pushing:

```bash

# Run automated fixer (recommended)

python scripts/tools/fix_markdown.py

# Or run markdown linting locally

make lint-markdown

# Or directly with markdownlint

markdownlint "**/*.md" --ignore node_modules --ignore .venv --ignore .build/site

# Auto-fix issues (when possible)

markdownlint --fix "**/*.md" --ignore node_modules --ignore .venv --ignore .build/site
```

## Table Formatting Rules

1. **Compact style**: Tables need spaces around pipes
   - Bad: `|Column1|Column2|`
   - Good: `| Column1 | Column2 |`

2. **Aligned style**: ALL rows (header, separator, and data) must have pipes at the same column positions
   - Bad: `| Header |` followed by `|-------|` (separator misaligned)
   - Bad: `| Header |` followed by `| Value |` (data row pipes don't align)
   - Good: All pipes align vertically:

     ```markdown
     | Header | Column2 |
     | ------ | ------- |
     | Value  | Data    |
     ```

3. **Compact style**: Consistent spacing throughout
   - Separator row: No spaces around pipes `|----------|`
   - Data rows: Minimal spacing `| Value |` (not `| Value  |` with extra spaces)
   - Bad: `| Header |` followed by `| -------- |` (separator has spaces)
   - Bad: `| Value  |` (extra spaces in data row)
   - Good: `| Header |` followed by `|----------|` and `| Value |`

### Solution: Use Python to Generate Exact Alignment

**Important**: When dealing with complex tables or "aligned" style errors, manually aligning pipes is error-prone and time-consuming. Use Python to generate perfectly aligned rows:

```python

# Step 1: Define header and extract pipe positions

header = "| Provider | Transcription | Speaker Detection | Summarization | Notes |"
h_pipes = [i for i, c in enumerate(header) if c == '|']

# Result: [0, 11, 27, 47, 63, 71]

# Step 2: Calculate column widths (distance between pipes)

col_widths = [h_pipes[i+1] - h_pipes[i] for i in range(len(h_pipes)-1)]

# Result: [11, 16, 20, 16, 8]

# Step 3: Build data rows with exact alignment using f-strings

# Content width = column width - 3 (for "| " and " |")

data1 = f"| {'Local':<9}| {'~2-5x realtime':<14}| {'~10ms/episode':<18}| {'~5-30s/episode':<14}| {'GPU-de':<6}|"
data2 = f"| {'OpenAI':<9}| {'~1x realtime':<14}| {'~500ms/episode':<18}| {'~2-10s/episode':<14}| {'API':<6}|"

# Step 4: Verify alignment programmatically

d1_pipes = [i for i, c in enumerate(data1) if c == '|']
d2_pipes = [i for i, c in enumerate(data2) if c == '|']
assert h_pipes == d1_pipes == d2_pipes, "Pipes must align exactly!"
assert len(header) == len(data1) == len(data2), "All rows must be same length!"

print("Alignment verified!")
```

- **Column width** = distance between pipes (e.g., 11-0=11, 27-11=16)
- **Content width** = column width - 3 (accounts for `|` prefix and `|` suffix with spaces)
- Use Python f-strings with left alignment (`:<width`) to pad content
- **Always verify** alignment programmatically before committing
- This approach saved significant debugging time when manual alignment failed

**Quick workflow:**

1. Get MD060 error → Identify it's "aligned" style
2. Extract pipe positions from header using Python
3. Generate data rows using f-string formatting
4. Verify alignment programmatically
5. Copy exact strings to markdown file
6. Run `make lint-markdown` to confirm fix

## Pre-commit Hook

**Already Integrated!** The project includes a pre-commit hook (`.github/hooks/pre-commit`) that automatically checks markdown files before commits.

**Install the hook:**

```bash

# Install the pre-commit hook

make install-hooks

# Or manually:

cp .github/hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

- Automatically checks staged markdown files before each commit
- Uses same ignore patterns as `make lint-markdown`
- Prevents commits with markdown linting errors
- Auto-fix support (see below)

**Enable auto-fix in pre-commit hook:**

```bash
# Set environment variable to enable auto-fix
export MARKDOWNLINT_FIX=1
git commit -m "your message"
# Auto-fixed files will be re-staged automatically
```

```bash
git commit --no-verify
```

- All markdown files are checked, not just changed files

## Workflow Summary

**Local Development:**

1. Edit markdown files
2. **Fix common issues automatically:** `python scripts/tools/fix_markdown.py`
3. Stage files: `git add docs/...`
4. Commit: `git commit` (pre-commit hook runs automatically)
5. If errors: Fix manually or use `markdownlint --fix`

**Pre-commit Hook:**

- Already installed and active
- Checks only staged files (fast)
- Uses same rules as CI/CD
- Optional auto-fix with `MARKDOWNLINT_FIX=1`

**CI/CD:**

- Checks all markdown files
- Fails build on any errors
- Ensures consistency across all files

## Lessons Learned from Large-Scale Cleanup

**Context:** In December 2024, we fixed ~1,016 markdown linting errors across 91 files, reducing
errors to zero. Here are key learnings:

### 1. Incremental Rule Re-enabling

**Don't re-enable all rules at once.** Start with rules that have the most errors and are easiest to
fix:

1. **First:** Fix auto-fixable rules (MD031, MD032, MD022, MD040, MD047)
2. **Second:** Re-enable rules with few errors (MD024, MD029, MD041)
3. **Third:** Handle edge cases and manual fixes
4. **Last:** Decide on structural rules (MD025, MD036) - may need to stay disabled

**Why:** Re-enabling all rules at once creates overwhelming error counts and makes it hard to track
progress.

### 2. Automated Fixing First

**Always run automated fixes before manual fixes:**

```bash

# Step 1: Run automated fixer (fixes ~80% of issues)

python scripts/tools/fix_markdown.py

# Step 2: Run again to catch issues fixed by first pass

python scripts/tools/fix_markdown.py

# Step 3: Check remaining errors

make lint-markdown

# Step 4: Fix remaining errors manually

```

**Why:** Automated fixes are fast, consistent, and handle the majority of issues. Manual fixes
should only be for edge cases.

## 3. Edge Cases to Watch For

**MD032 (blanks-around-lists):** Lists that come immediately after bold text (not headings) need
blank lines:

```markdown

# Bad

**Section Title:**

- Item 1

# Good

**Section Title:**

- Item 1

```

**MD051 (link-fragments):** Can produce false positives. If headings exist and fragments match
correctly, consider disabling this rule:

```json

{
  "MD051": false
}

```

**MD013 (line-length):** Many long lines are legitimate (URLs, code examples, technical
descriptions). Consider:

- Increasing limit to 120 chars
- Disabling for specific file types (RFCs, PRDs, technical docs)
- Using Prettier for automatic wrapping

## 4. Script Enhancement Strategy

**Enhance the fix script incrementally:**

1. Start with simple fixes (trailing spaces, table formatting)
2. Add more complex fixes (blank lines around lists, headings, fences)
3. Improve language detection for code blocks
4. Handle edge cases as they're discovered

**Why:** A comprehensive fix script saves significant time and ensures consistency.

### 5. Maintenance Workflow

**To prevent errors from accumulating:**

1. **Before committing:** Run `make fix-md` or `python scripts/tools/fix_markdown.py`
2. **Use format-on-save:** Configure Prettier to format markdown on save
3. **Pre-commit hook:** Already integrated - catches errors before commit
4. **Regular cleanup:** Run fix script monthly to catch accumulated issues

### 6. When to Disable Rules

**Disable rules only when:**

- They produce false positives (e.g., MD051 with valid link fragments)
- They conflict with legitimate use cases (e.g., MD025 for docs with multiple H1s)
- They're structural and can't be fixed (e.g., MD036 for emphasis-as-heading)

**Don't disable rules to:**

- Hide technical debt
- Avoid fixing issues
- Make CI pass quickly

### 7. Metrics and Tracking

**Track progress with metrics:**

- Total errors before/after
- Errors by rule type
- Files affected
- Time to fix

**Why:** Metrics help justify effort and demonstrate value.

### 8. Tool Configuration

**Ensure consistency:**

- Use same `.markdownlint.json` in CI and locally
- Configure Prettier to match markdownlint rules where possible
- Document any rule exceptions clearly

**Why:** Prevents confusion and ensures developers see same errors locally and in CI.

## Related Documentation

- [Development Guide](DEVELOPMENT_GUIDE.md) - General development practices
- [CI/CD](../ci/index.md) - Continuous integration pipeline details
- `scripts/tools/fix_markdown.py` - Automated fixer script (see repository root)
- `.markdownlint.json` - Linter configuration (see repository root)
- `.prettierrc` - Prettier formatting rules (see repository root)
- `.vscode/settings.json` - Editor settings for format-on-save (see repository root)
