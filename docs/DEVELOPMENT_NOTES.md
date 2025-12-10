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

### Solution: Use Python to Generate Exact Alignment

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
```

**Key points:**

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
