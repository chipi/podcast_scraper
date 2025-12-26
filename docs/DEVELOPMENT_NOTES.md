# Development Notes

> **Maintenance Note**: This document should be kept up-to-date as linting rules, Makefile targets, pre-commit hooks, CI/CD workflows, or development setup procedures evolve. When adding new checks, tools, workflows, or environment setup steps, update this document accordingly.

## Environment Setup

### Virtual Environment

**Quick setup:**

```bash
bash scripts/setup_venv.sh
source .venv/bin/activate
```

**Install dependencies:**

```bash
make init  # Installs dev + ML dependencies
```

### Environment Variables

**Supported environment variables:**

The podcast scraper supports configuration via environment variables for flexible deployment. Many settings can be configured via environment variables or `.env` files.

1. **Copy example `.env` file:**

   ```bash
   cp examples/.env.example .env
   ```

2. **Edit `.env` and add your settings:**

   ```bash
   # OpenAI API key (required for OpenAI providers)
   OPENAI_API_KEY=sk-your-actual-key-here

   # Logging
   LOG_LEVEL=DEBUG

   # Paths
   OUTPUT_DIR=/data/transcripts
   LOG_FILE=/var/log/podcast_scraper.log
   CACHE_DIR=/cache/models

   # Performance tuning
   WORKERS=4
   TRANSCRIPTION_PARALLELISM=3
   PROCESSING_PARALLELISM=4
   SUMMARY_BATCH_SIZE=2
   SUMMARY_CHUNK_PARALLELISM=2
   TIMEOUT=60
   SUMMARY_DEVICE=cpu
   ```

3. **The `.env` file is automatically loaded** via `python-dotenv` when `podcast_scraper.config` module is imported.

**Security notes:**

- ✅ `.env` is in `.gitignore` (never committed)
- ✅ `examples/.env.example` is safe to commit (template only)
- ✅ API keys are never logged or exposed
- ✅ Environment variables take precedence over `.env` file

**Priority order:**

- Config file field (highest priority)
- Environment variable
- Default value
- Exception: `LOG_LEVEL` (env var takes precedence)

**See also:**

- `docs/ENVIRONMENT_VARIABLES.md` - Complete environment variable documentation
- `docs/rfc/RFC-013-openai-provider-implementation.md` - API key management details
- `docs/prd/PRD-006-openai-provider-integration.md` - OpenAI provider requirements

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

## AI Coding Guidelines

This project includes comprehensive AI coding guidelines to ensure consistent code quality and workflow when using AI assistants.

### Overview

**Primary reference:** `.ai-coding-guidelines.md` - This is the PRIMARY source of truth for all AI actions in this project.

**Purpose:**

- Provides project-specific context and patterns for AI assistants
- Ensures consistent code quality and workflow
- Prevents common mistakes (auto-committing, skipping CI, etc.)

### Entry Points by AI Tool

Different AI assistants load guidelines from different locations:

| Tool | Entry Point | Auto-Loaded |
|------|-------------|-------------|
| **Cursor** | `.cursor/rules/ai-guidelines.mdc` | ✅ Yes (modern format) |
| **Claude Desktop** | `CLAUDE.md` (root directory) | ✅ Yes |
| **GitHub Copilot** | `.github/copilot-instructions.md` | ✅ Yes |

**All entry points reference `.ai-coding-guidelines.md` as the primary source.**

### Critical Workflow Rules

**NEVER commit without:**

- Showing user what files changed (`git status`)
- Showing user the actual changes (`git diff`)
- Getting explicit user approval
- User deciding commit message

**NEVER push to PR without:**

- Running `make ci` locally first
- Ensuring `make ci` passes completely
- Fixing all failures before pushing

### What's in `.ai-coding-guidelines.md`

**Sections include:**

- **Git Workflow** - Commit approval, PR workflow, branch naming
- **Code Organization** - Module boundaries, when to create new files
- **Testing Requirements** - Mocking patterns, test structure
- **Documentation Standards** - PRDs, RFCs, docstrings
- **Common Patterns** - Configuration, error handling, logging
- **Decision Trees** - When to create modules, PRDs, RFCs
- **When to Ask** - When AI should ask vs. act autonomously

### For Developers

**If you're using an AI assistant:**

- The guidelines are automatically loaded (no setup needed)
- AI assistants will follow project patterns and workflows
- Guidelines ensure consistent code quality

**If you're not using an AI assistant:**

- You don't need to read these files
- They're for AI tools, not human developers
- Human contributors should follow [CONTRIBUTING.md](https://github.com/chipi/podcast_scraper/blob/main/CONTRIBUTING.md)

### Maintenance

**When to update `.ai-coding-guidelines.md`:**

- New patterns or conventions are established
- Workflow changes (e.g., new CI checks)
- Architecture decisions that affect code organization
- New tools or processes are added

**Keep entry points in sync:**

- When updating `.ai-coding-guidelines.md`, ensure entry points (`CLAUDE.md`, `.github/copilot-instructions.md`, `.cursor/rules/ai-guidelines.mdc`) still reference it correctly

**See:** `.ai-coding-guidelines.md` for complete guidelines.
