# Development Notes

> **Maintenance Note**: This document should be kept up-to-date as linting rules, Makefile targets, pre-commit hooks, or CI/CD workflows evolve. When adding new checks, tools, or workflows, update this document accordingly.

## Markdown Linting

### Catching Table Formatting Issues Locally

To catch markdown table formatting issues (MD060) before pushing:

```bash
# Run markdown linting locally
make lint-markdown

# Or directly with markdownlint
markdownlint "**/*.md" --ignore node_modules --ignore .venv --ignore .build/site

# Auto-fix issues (when possible)
markdownlint --fix "**/*.md" --ignore node_modules --ignore .venv --ignore .build/site
```

### Common Table Formatting Issues

**MD060/table-column-style** errors occur when:

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

### Pre-commit Hook

**✅ Already Integrated!** The project includes a pre-commit hook (`.github/hooks/pre-commit`) that automatically checks markdown files before commits.

**Install the hook:**

```bash
# Install the pre-commit hook
make install-hooks

# Or manually:
cp .github/hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

**Features:**

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
```

**Skip pre-commit checks (if needed):**

```bash
git commit --no-verify
```

### CI/CD Integration

The GitHub Actions workflow runs `make lint-markdown` which includes:

- `markdownlint "**/*.md"` with proper ignores
- Fails the build if any errors are found
- All markdown files are checked, not just changed files

### Workflow Summary

**Local Development:**

1. Edit markdown files
2. Stage files: `git add docs/...`
3. Commit: `git commit` (pre-commit hook runs automatically)
4. If errors: Fix manually or use `markdownlint --fix`

**Pre-commit Hook:**

- ✅ Already installed and active
- ✅ Checks only staged files (fast)
- ✅ Uses same rules as CI/CD
- ✅ Optional auto-fix with `MARKDOWNLINT_FIX=1`

**CI/CD:**

- ✅ Checks all markdown files
- ✅ Fails build on any errors
- ✅ Ensures consistency across all files
