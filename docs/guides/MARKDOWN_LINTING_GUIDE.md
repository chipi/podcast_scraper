# Markdown Linting Guide

This guide covers markdown linting practices, tools, and workflows for the podcast scraper
project. It includes automated fixing, table formatting solutions, pre-commit hooks, and CI/CD
integration.

For general development practices, see [Development Guide](DEVELOPMENT_GUIDE.md).

## Automated Markdown Fixing

**Recommended:** Use `scripts/fix_markdown.py` to automatically fix common markdown linting issues:

````bash

# Fix all markdown files in the project

python scripts/fix_markdown.py

# Fix specific files

python scripts/fix_markdown.py TESTING_STRATEGY.md rfc/RFC-020.md

# Dry run (see what would be fixed)

python scripts/fix_markdown.py --dry-run
```text

- Table separator formatting (MD060) - adds spaces around pipes
- Trailing spaces (MD009)
- Blank lines around lists (MD032)
- Code block language specifiers (MD040) - when detectable

**When to use:**

- Before committing markdown files
- After bulk documentation edits
- When CI fails on markdown linting errors

See `scripts/README.md` for full documentation.

## Catching Table Formatting Issues Locally

To catch markdown table formatting issues (MD060) before pushing:

```bash

# Run automated fixer (recommended)

python scripts/fix_markdown.py

# Or run markdown linting locally

make lint-markdown

# Or directly with markdownlint

markdownlint "**/*.md" --ignore node_modules --ignore .venv --ignore .build/site

# Auto-fix issues (when possible)

markdownlint --fix "**/*.md" --ignore node_modules --ignore .venv --ignore .build/site
```text

1. **Compact style**: Tables need spaces around pipes
   - ❌ Bad: `|Column1|Column2|`
   - ✅ Good: `| Column1 | Column2 |`

2. **Aligned style**: ALL rows (header, separator, and data) must have pipes at the same column positions
   - ❌ Bad: `| Header |` followed by `|-------|` (separator misaligned)
   - ❌ Bad: `| Header |` followed by `| Value |` (data row pipes don't align)
   - ✅ Good: All pipes align vertically:

     ```markdown
     | Header | Column2 |
     | ------ | ------- |
     | Value  | Data    |
     ```

3. **Compact style**: Consistent spacing throughout
   - Separator row: No spaces around pipes `|----------|`
   - Data rows: Minimal spacing `| Value |` (not `| Value  |` with extra spaces)
   - ❌ Bad: `| Header |` followed by `| -------- |` (separator has spaces)
   - ❌ Bad: `| Value  |` (extra spaces in data row)
   - ✅ Good: `| Header |` followed by `|----------|` and `| Value |`

## Solution: Use Python to Generate Exact Alignment

**⚠️ Important**: When dealing with complex tables or "aligned" style errors, manually aligning pipes is error-prone and time-consuming. Use Python to generate perfectly aligned rows:

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

print("✅ Alignment verified!")
```text

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

**✅ Already Integrated!** The project includes a pre-commit hook (`.github/hooks/pre-commit`) that automatically checks markdown files before commits.

**Install the hook:**

```bash

# Install the pre-commit hook

make install-hooks

# Or manually:

cp .github/hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```text

- ✅ Automatically checks staged markdown files before each commit
- ✅ Uses same ignore patterns as `make lint-markdown`
- ✅ Prevents commits with markdown linting errors
- ✅ Auto-fix support (see below)

**Enable auto-fix in pre-commit hook:**

```bash

# Set environment variable to enable auto-fix

export MARKDOWNLINT_FIX=1
git commit -m "your message"

# Auto-fixed files will be re-staged automatically

```bash

git commit --no-verify

```text
- All markdown files are checked, not just changed files

## Workflow Summary

**Local Development:**

1. Edit markdown files
2. **Fix common issues automatically:** `python scripts/fix_markdown.py`
3. Stage files: `git add docs/...`
4. Commit: `git commit` (pre-commit hook runs automatically)
5. If errors: Fix manually or use `markdownlint --fix`

**Pre-commit Hook:**

- ✅ Already installed and active
- ✅ Checks only staged files (fast)
- ✅ Uses same rules as CI/CD
- ✅ Optional auto-fix with `MARKDOWNLINT_FIX=1`

**CI/CD:**

- ✅ Checks all markdown files
- ✅ Fails build on any errors
- ✅ Ensures consistency across all files

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

python scripts/fix_markdown.py

# Step 2: Run again to catch issues fixed by first pass

python scripts/fix_markdown.py

# Step 3: Check remaining errors

make lint-markdown

# Step 4: Fix remaining errors manually

```text
**Why:** Automated fixes are fast, consistent, and handle the majority of issues. Manual fixes
should only be for edge cases.

### 3. Edge Cases to Watch For

**MD032 (blanks-around-lists):** Lists that come immediately after bold text (not headings) need
blank lines:

```markdown

# Bad

**Section Title:**

- Item 1

# Good

**Section Title:**

- Item 1

```text
**MD051 (link-fragments):** Can produce false positives. If headings exist and fragments match
correctly, consider disabling this rule:

```json

{
  "MD051": false
}

```text
**MD013 (line-length):** Many long lines are legitimate (URLs, code examples, technical
descriptions). Consider:

- Increasing limit to 120 chars
- Disabling for specific file types (RFCs, PRDs, technical docs)
- Using Prettier for automatic wrapping

### 4. Script Enhancement Strategy

**Enhance the fix script incrementally:**

1. Start with simple fixes (trailing spaces, table formatting)
2. Add more complex fixes (blank lines around lists, headings, fences)
3. Improve language detection for code blocks
4. Handle edge cases as they're discovered

**Why:** A comprehensive fix script saves significant time and ensures consistency.

### 5. Maintenance Workflow

**To prevent errors from accumulating:**

1. **Before committing:** Run `make fix-md` or `python scripts/fix_markdown.py`
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
- [CI/CD](../CI_CD.md) - Continuous integration pipeline details
- [Markdown Style Reference](MD_STYLE_REFERENCE.md) - Quick reference style guide
